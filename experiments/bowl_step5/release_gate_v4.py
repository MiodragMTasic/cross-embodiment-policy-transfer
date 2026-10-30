"""v4 adds physically supported rim placement after verified release; v3 preserved."""
from release_gate_v3 import ReleaseGateV3


class ReleaseGateV4(ReleaseGateV3):
    def supported_region(self, sample):
        sample = dict(sample)
        if self.phase == 'released' and sample['rim_support']:
            sample['near_bowl'] = True
        return sample

    def before(self, t, command, sample):
        return super().before(t, command, self.supported_region(sample))

    def after(self, t, sample, raw_success, linear_speed, angular_speed):
        sample = self.supported_region(sample)
        supported_placement = sample['bowl_support'] and (bool(raw_success) or sample['rim_support'])
        return super().after(t, sample, supported_placement, linear_speed, angular_speed)
