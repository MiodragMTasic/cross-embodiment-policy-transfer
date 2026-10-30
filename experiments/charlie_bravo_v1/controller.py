"""CPU-only release supervisor. No simulator, credentials, or network imports.

Arm commands hold the last executed joint target during intervention, preserving the
physical pose already reached with the adapter's lift offset. Never recompute
that held pose from an open-gripper adapter target.
"""
from dataclasses import dataclass
from enum import Enum
import json
import math


class State(str, Enum):
    POLICY = 'POLICY'
    HOLD = 'HOLD'
    OPEN = 'OPEN'
    DONE = 'DONE'


@dataclass(frozen=True)
class Config:
    cadence: int = 12             # 15 Hz, provisional development choice
    hold_ticks: int = 9
    open_ticks: int = 90          # 6 s settle window; gripper remains open afterwards
    max_hold_ticks: int = 15
    max_hold_entries: int = 3
    max_calls: int = 30           # bounded; not a promised coverage rate
    max_wall_s: float = 1800
    timeout_s: float = 10
    api_cap: float = 0.05
    reserve_per_call: float = 0.002  # provisional upper bound; must validate prices/tokens

    def __post_init__(self):
        for k in ('cadence', 'hold_ticks', 'open_ticks', 'max_hold_ticks', 'max_calls', 'max_hold_entries'):
            if type(getattr(self, k)) is not int or getattr(self, k) <= 0:
                raise ValueError(k)
        for k in ('max_wall_s', 'timeout_s', 'api_cap', 'reserve_per_call'):
            v = getattr(self, k)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(k)
        if self.hold_ticks > self.max_hold_ticks:
            raise ValueError('hold exceeds limit')


@dataclass(frozen=True)
class Frame:
    tick: int
    exterior: bytes
    wrist: bytes
    source: str = 'actual'


@dataclass(frozen=True)
class Decision:
    verdict: str
    held: bool
    central: bool
    clearance: bool
    views_agree: bool
    motion: str
    reason: str

    @classmethod
    def parse(cls, raw):
        # No markdown extraction, coercion, partial object, or confidence shortcut.
        def unique(pairs):
            d = {}
            for k, v in pairs:
                if k in d:
                    raise ValueError('duplicate key')
                d[k] = v
            return d
        d = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(d, dict) or set(d) != set(cls.__dataclass_fields__):
            raise ValueError('schema keys')
        if d['verdict'] not in ('WAIT', 'READY_TO_RELEASE', 'UNCERTAIN'):
            raise ValueError('verdict')
        if d['motion'] not in ('inward', 'outward', 'still', 'unknown'):
            raise ValueError('motion')
        if any(type(d[k]) is not bool for k in ('held', 'central', 'clearance', 'views_agree')):
            raise ValueError('boolean field')
        if not isinstance(d['reason'], str) or not 1 <= len(d['reason']) <= 240:
            raise ValueError('reason')
        return cls(**d)

    @property
    def ready(self):
        return (self.verdict == 'READY_TO_RELEASE' and self.held and self.central
                and self.clearance and self.views_agree)


@dataclass(frozen=True)
class Command:
    arm: tuple
    grip: float
    state: str
    stop: bool = False
    reason: str = ''


class Supervisor:
    def __init__(self, config=Config(), enabled=True):
        self.config = config
        self.enabled = enabled
        self.state = State.POLICY
        self.calls = 0
        self.reserved = 0.0
        self.last_query = -config.cadence
        self.last_tick = -1
        self.start_wall = None
        self.last_wall = None
        self.hold_tick = None
        self.open_tick = None
        self.pose = None
        self.confirmation_tick = None
        self.used = False
        self.hold_entries = 0
        self.exhausted = False
        self.invalid_reason = None
        self.pending = None
        self.reason = 'policy'

    def end_failed_hold(self):
        self.state = State.POLICY
        self.hold_tick = None
        if self.hold_entries >= self.config.max_hold_entries:
            self.exhausted = True
            self.reason = 'hold_attempts_exhausted; continue policy and count episode'

    def advance(self, tick, now, measured, policy_arm, policy_grip):
        if type(tick) is not int or tick <= self.last_tick:
            raise ValueError('ticks must increase')
        if not math.isfinite(now) or (self.last_wall is not None and now < self.last_wall):
            raise ValueError('monotonic wall clock required')
        for values in (measured, policy_arm):
            if len(values) != 7 or not all(math.isfinite(v) for v in values):
                raise ValueError('seven finite joints required')
        if not math.isfinite(policy_grip) or not 0 <= policy_grip <= 1:
            raise ValueError('grip range')
        self.last_tick, self.last_wall = tick, now
        if self.start_wall is None:
            self.start_wall = now
        if self.pending is not None:
            raise RuntimeError('simulation advanced during model request')
        if now - self.start_wall >= self.config.max_wall_s:
            return Command(tuple(measured), policy_grip, self.state.value, True, 'wall deadline')
        if self.state == State.HOLD and tick - self.hold_tick > self.config.max_hold_ticks:
            self.reason = 'hold expired without confirmation'
            self.end_failed_hold()
        if self.state == State.OPEN and tick - self.open_tick >= self.config.open_ticks:
            self.state = State.DONE
        if self.state == State.DONE:
            return Command(tuple(policy_arm), 0.0, self.state.value, reason='resume arm; gripper permanently open')
        if self.state == State.OPEN:
            return Command(self.pose, 0.0, self.state.value, reason='latched open')
        if self.state == State.HOLD:
            return Command(self.pose, 1.0, self.state.value, reason='hold for fresh confirmation')
        return Command(tuple(policy_arm), policy_grip, self.state.value, reason=self.reason)

    def request(self, frames, policy_grip, now):
        if self.pending is not None or self.used or self.exhausted or self.state in (State.OPEN, State.DONE):
            return None
        if now - self.start_wall >= self.config.max_wall_s:
            return None
        if policy_grip <= .5 and self.state == State.POLICY:
            return None
        if self.state == State.HOLD:
            if self.last_tick - self.hold_tick < self.config.hold_ticks:
                return None
        elif self.last_tick - self.last_query < self.config.cadence:
            return None
        if len(frames) < 2 or frames[-1].tick != self.last_tick:
            return None
        if any(f.source != 'actual' or not f.exterior or not f.wrist for f in frames):
            return None
        ticks = [f.tick for f in frames]
        if ticks != sorted(set(ticks)) or ticks[0] < self.last_tick - self.config.cadence * 2:
            return None
        if self.state == State.HOLD and (len(frames) != 3 or ticks[0] <= self.hold_tick):
            return None
        if self.calls >= self.config.max_calls or self.reserved + self.config.reserve_per_call > self.config.api_cap + 1e-12:
            if self.state == State.HOLD:
                self.state = State.POLICY
            self.invalid_reason = 'drop_budget'
            self.reason = 'model budget exhausted'
            return None
        self.calls += 1
        self.reserved += self.config.reserve_per_call
        self.last_query = self.last_tick
        self.pending = (self.last_tick, now, self.state)
        return {'call': self.calls, 'tick': self.last_tick, 'phase': self.state.value,
                'hold_since_tick': self.hold_tick if self.state == State.HOLD else None,
                'deadline_s': min(self.config.timeout_s, self.config.max_wall_s - (now - self.start_wall)),
                'reserved_cost': self.config.reserve_per_call}

    def receive(self, raw, now, tick, measured):
        if self.pending is None:
            raise RuntimeError('unsolicited response')
        capture_tick, start, phase = self.pending
        self.pending = None
        valid = (math.isfinite(now) and now >= start and now - start < self.config.timeout_s
                 and now - self.start_wall < self.config.max_wall_s and tick == capture_tick)
        decision = None
        try:
            if valid:
                decision = Decision.parse(raw)
        except (ValueError, TypeError):
            pass
        if decision is None or not decision.ready or (phase == State.HOLD and decision.motion != 'still'):
            self.reason = 'no intervention: invalid, late, uncertain or not ready'
            if phase == State.HOLD:
                self.end_failed_hold()
            return
        if not self.enabled:
            self.reason = 'baseline: visual decision recorded, no override'
            return
        if len(measured) != 7 or not all(math.isfinite(v) for v in measured):
            raise ValueError('invalid measured pose')
        if phase == State.POLICY:
            if self.hold_entries >= self.config.max_hold_entries:
                self.exhausted = True
                self.reason = 'hold_attempts_exhausted; continue policy and count episode'
                return
            self.hold_entries += 1
            self.pose = tuple(measured)
            self.hold_tick = tick
            self.state = State.HOLD
        elif phase == State.HOLD and tick > self.hold_tick:
            self.confirmation_tick = tick
            self.open_tick = tick
            self.state = State.OPEN
            self.used = True

    def current_override(self):
        if self.state == State.HOLD:
            return self.pose, 1.0
        if self.state == State.OPEN:
            return self.pose, 0.0
        return None
