"""Synchronous simulator hook, injected judge only: this file cannot call an API.

Caller supplies actual encoded RGB frames and steps physics ONLY after this hook
returns. Existing robot limits must be applied to the returned command. A stop
result means preserve evidence and terminate the episode, not stop VM billing.
"""
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from controller import Supervisor, Frame, Command, State, Decision


class Runtime:
    def __init__(self, output, judge, enabled=True, config=None, clock=time.monotonic):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.judge = judge
        self.clock = clock
        self.controller = Supervisor(enabled=enabled, **({'config': config} if config else {}))
        self.history = deque(maxlen=3)
        self.last_capture = -100000
        self.consecutive_errors = 0
        self.prompt = Path(__file__).with_name('vision_prompt.txt').read_text()

    def log(self, record):
        with (self.output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
            f.flush()

    def step(self, tick, measured, policy_arm, policy_grip, capture, record_command=True):
        c = self.controller
        command = c.advance(tick, self.clock(), measured, policy_arm, policy_grip)
        if command.stop:
            self.log({'tick': tick, 'command': asdict(command)})
            return command
        # Capture during hold even between normal checks; history must be current.
        if tick - self.last_capture >= (3 if c.state == State.HOLD else 6):
            ext, wrist = capture()
            self.history.append(Frame(tick, ext, wrist))
            self.last_capture = tick
        request = c.request(list(self.history), policy_grip, self.clock())
        if request:
            folder = self.output / f'query_{request["call"]:04d}'
            folder.mkdir()
            manifest = []
            for frame in self.history:
                for view, payload in [('exterior', frame.exterior), ('wrist', frame.wrist)]:
                    name = f'{frame.tick}_{view}.png'
                    (folder / name).write_bytes(payload)
                    manifest.append({'file': name, 'sha256': hashlib.sha256(payload).hexdigest()})
            (folder / 'request.json').write_text(json.dumps({'request': request, 'prompt': self.prompt, 'images': manifest}, indent=2))
            raw = ''
            error = None
            try:
                # Judge must enforce this timeout in its transport; no asynchronous
                # worker is allowed to deliver a late action after this call returns.
                if c.enabled:
                    raw = self.judge(self.prompt, tuple(self.history), request)
                else:
                    raw = ''  # baseline records requests only; no network
                if not isinstance(raw, str):
                    raise TypeError('judge must return JSON text')
                if c.enabled:
                    try: Decision.parse(raw)
                    except (ValueError, TypeError): error = 'SchemaError'
            except Exception as exc:
                # Do not serialize exception messages (may contain tokens/URLs).
                error = type(exc).__name__
            before_state = c.state
            c.receive(raw, self.clock(), tick, policy_arm)  # last executed B target, not sagged measured pose
            if before_state != State.HOLD and c.state == State.HOLD:
                self.history.clear()
                self.last_capture = tick
            self.consecutive_errors = self.consecutive_errors + 1 if error else 0
            if c.enabled and (error == 'PermissionError' or self.consecutive_errors >= 3):
                c.invalid_reason = 'transport_' + (error or 'unknown')
            (folder / 'response.json').write_text(json.dumps({'raw': raw, 'error_type': error, 'state': c.state.value}))
        if c.invalid_reason and c.enabled:
            command = Command(tuple(measured), command.grip, c.state.value, True, c.invalid_reason)
            self.log({'tick':tick, 'invalid':c.invalid_reason})
            return command
        override = c.current_override()
        if override:
            command = Command(override[0], override[1], c.state.value, reason='vision hold/release override')
        elif command.state == 'HOLD':
            command = Command(tuple(policy_arm), policy_grip, c.state.value, reason=c.reason)
        # Deadline includes model wait, not only simulation stepping.
        if self.clock() - c.start_wall >= c.config.max_wall_s:
            command = Command(tuple(measured), command.grip, c.state.value, True, 'wall deadline after query')
        self.log({'tick': tick, 'original_arm': list(policy_arm) if record_command else None, 'original_grip': policy_grip,
                  'executed' if record_command else 'proposed_override': asdict(command),
                  'calls': c.calls, 'reserved_usd': c.reserved, 'hold_attempts_exhausted':c.exhausted})
        return command
