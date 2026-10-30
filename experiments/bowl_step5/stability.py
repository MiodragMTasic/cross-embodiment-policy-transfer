"""Pure elapsed-time criterion, separately tested without Isaac Sim."""
from dataclasses import dataclass


@dataclass
class StableWindow:
    duration_s: float = 1.0
    linear_limit: float = 0.03
    angular_limit: float = 0.3
    since: float | None = None
    previous_time: float | None = None

    def update(self, time_s, raw_success, linear_speed, angular_speed):
        import math
        if self.previous_time is not None and time_s <= self.previous_time:
            raise ValueError("Samples must have strictly increasing simulation times")
        self.previous_time = time_s
        qualifies = (bool(raw_success) and math.isfinite(linear_speed)
                     and math.isfinite(angular_speed)
                     and linear_speed <= self.linear_limit
                     and angular_speed <= self.angular_limit)
        if not qualifies:
            self.since = None
            return False
        if self.since is None:
            self.since = time_s
        return time_s - self.since >= self.duration_s - 1e-9
