"""Conservative sampled command-linked release metric. No policy control."""
from dataclasses import dataclass, field
import math
from stability import StableWindow


@dataclass
class ReleaseGate:
    lift_duration_s: float = .2
    lift_clearance_m: float = .01
    release_timeout_s: float = .5
    open_closure: float = .2
    minimum_opening: float = .05
    stable: StableWindow = field(default_factory=StableWindow)
    lift_since: float | None = None
    armed: bool = False
    phase: str = 'idle'
    previous_command: float | None = None
    command: float | None = None
    opened_at: float | None = None
    detached_at: float | None = None
    opening_closure: float | None = None
    previous_post_time: float | None = None
    events: list = field(default_factory=list)

    def invalidate(self, t, reason):
        if self.armed or self.phase != 'idle':
            self.events.append({'time_s': t, 'event': 'invalidated', 'reason': reason})
        self.phase = 'idle'
        self.armed = False
        self.lift_since = None
        self.opened_at = self.detached_at = self.opening_closure = None
        self.stable.since = None

    def before(self, t, command, sample):
        """Pre-action signal. t is the current, not next, simulation time."""
        self.command = float(command)
        if not sample['finite'] or not math.isfinite(self.command):
            self.invalidate(t, 'nonfinite_pre_action')
            self.previous_command = None
            return
        closed = self.command > .5
        if self.phase != 'idle' and closed:
            self.invalidate(t, 'reclosed')
        if self.detached_at is not None and sample['contact']:
            self.invalidate(t, 'pre_action_recontact_after_detachment')
        if self.phase == 'idle':
            if not sample['contact']:
                self.invalidate(t, 'lost_contact_before_open')
            lifting = (closed and sample['contact'] and not sample['table_contact']
                       and not sample['bowl_contact']
                       and sample['clearance_m'] >= self.lift_clearance_m)
            if lifting:
                if self.lift_since is None:
                    self.lift_since = t
                if t - self.lift_since >= self.lift_duration_s - 1e-9 and not self.armed:
                    self.armed = True
                    self.events.append({'time_s': t, 'event': 'lift_armed'})
            else:
                self.lift_since = None
            edge = self.previous_command is not None and self.previous_command > .5 and not closed
            if (edge and self.armed and sample['contact'] and sample['near_bowl']
                    and sample['closure'] > self.open_closure):
                self.phase = 'opening'
                self.opened_at = t
                self.opening_closure = sample['closure']
                self.events.append({'time_s': t, 'event': 'eligible_open_command'})
            elif edge:
                self.invalidate(t, 'ineligible_open_command')
        self.previous_command = self.command

    def after(self, t, sample, raw_success, linear_speed, angular_speed):
        """Post-action signal, called once per executed control tick."""
        if self.previous_post_time is not None and t <= self.previous_post_time:
            raise ValueError('Non-monotonic simulation time / unexpected episode reset')
        self.previous_post_time = t
        if not sample['finite'] or not all(map(math.isfinite, (linear_speed, angular_speed))):
            self.invalidate(t, 'nonfinite_post_action')
        elif self.phase == 'idle':
            if not sample['contact']:
                self.invalidate(t, 'lost_contact_before_open')
        elif self.command is None or self.command > .5:
            self.invalidate(t, 'closed_after_open')
        elif self.detached_at is not None and sample['contact']:
            self.invalidate(t, 'recontact_after_detachment')
        elif self.phase == 'opening':
            if t - self.opened_at > self.release_timeout_s + 1e-9:
                self.invalidate(t, 'opening_timeout')
            else:
                if not sample['contact'] and self.detached_at is None:
                    if sample['near_bowl']:
                        self.detached_at = t
                        self.events.append({'time_s': t, 'event': 'detached_near_bowl'})
                    else:
                        self.invalidate(t, 'detached_away_from_bowl')
                if (self.phase == 'opening' and self.detached_at is not None
                        and sample['closure'] <= self.open_closure
                        and self.opening_closure - sample['closure'] >= self.minimum_opening):
                    self.phase = 'released'
                    self.events.append({'time_s': t, 'event': 'measured_open_release'})
        elif self.phase == 'released' and sample['closure'] > self.open_closure:
            self.invalidate(t, 'measured_reclosure')
        qualifies = (self.phase == 'released' and sample['finite']
                     and not sample['contact'] and sample['closure'] <= self.open_closure
                     and bool(raw_success))
        return self.stable.update(t, qualifies, linear_speed, angular_speed)

    def status(self):
        return {'phase': self.phase, 'lift_armed': self.armed, 'lift_since_s': self.lift_since,
                'opening_command_s': self.opened_at, 'detachment_s': self.detached_at,
                'stable_since_s': self.stable.since}
