import unittest
from release_gate import ReleaseGate


def sample(**overrides):
    s = dict(finite=True, contact=True, table_contact=False, bowl_contact=False,
             clearance_m=.03, near_bowl=True, closure=.7)
    s.update(overrides)
    return s


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.g = ReleaseGate()
        self.t = 0.

    def tick(self, command, pre=None, post=None, raw=False, linear=0., angular=0.):
        pre = sample() if pre is None else pre
        post = pre if post is None else post
        self.g.before(self.t, command, pre)
        self.t += 1/15
        return self.g.after(self.t, post, raw, linear, angular)

    def lift(self):
        for _ in range(5):
            self.tick(1.)
        self.assertTrue(self.g.armed)

    def release(self):
        self.lift()
        self.assertFalse(self.tick(0., sample(), sample(contact=False, closure=.1), True))
        self.assertEqual(self.g.phase, 'released')

    def test_valid_release_and_one_elapsed_second(self):
        self.release()
        for _ in range(14):
            self.assertFalse(self.tick(0., sample(contact=False, closure=.1), raw=True))
        self.assertTrue(self.tick(0., sample(contact=False, closure=.1), raw=True))

    def test_closed_slip_then_late_open_fails(self):
        self.lift()
        self.tick(1., sample(), sample(contact=False), True)
        self.assertFalse(self.g.armed)
        for _ in range(30):
            self.assertFalse(self.tick(0., sample(contact=False, closure=.1), raw=True))

    def test_already_detached_before_open_fails(self):
        self.lift()
        self.tick(0., sample(contact=False), sample(contact=False, closure=.1), True)
        self.assertEqual(self.g.phase, 'idle')

    def test_no_lift_or_supported_contact_cannot_arm(self):
        for pre in (sample(clearance_m=.005), sample(table_contact=True), sample(bowl_contact=True)):
            self.setUp()
            for _ in range(10): self.tick(1., pre)
            self.assertFalse(self.g.armed)
            self.assertFalse(self.tick(0., pre, sample(contact=False, closure=.1), True))

    def test_no_open_command_cannot_succeed(self):
        self.lift()
        for _ in range(30): self.assertFalse(self.tick(1., raw=True))

    def test_open_away_from_bowl_fails(self):
        self.lift()
        self.tick(0., sample(near_bowl=False), sample(contact=False, closure=.1), True)
        self.assertEqual(self.g.phase, 'idle')

    def test_detachment_away_from_bowl_fails(self):
        self.lift()
        self.tick(0., sample(), sample(contact=False, near_bowl=False, closure=.1), True)
        self.assertEqual(self.g.phase, 'idle')

    def test_reclosure_and_recontact_reset_release(self):
        for command, pre in ((1., sample(contact=False, closure=.1)),
                             (0., sample(contact=True, closure=.1))):
            self.setUp(); self.release()
            self.assertFalse(self.tick(command, pre, raw=True))
            self.assertEqual(self.g.phase, 'idle')

    def test_timeout_requires_actual_open_and_detachment(self):
        self.lift()
        for _ in range(9): self.assertFalse(self.tick(0., raw=True))
        self.assertEqual(self.g.phase, 'idle')

    def test_speed_raw_and_nonfinite_reset_stability(self):
        self.release()
        self.tick(0., sample(contact=False, closure=.1), raw=True, linear=.04)
        self.assertIsNone(self.g.stable.since)
        self.tick(0., sample(contact=False, closure=.1), raw=False)
        self.assertIsNone(self.g.stable.since)
        self.tick(0., sample(contact=False, closure=.1, finite=False), raw=True)
        self.assertEqual(self.g.phase, 'idle')

    def test_regrasp_can_start_fresh_attempt(self):
        self.lift(); self.tick(1., sample(), sample(contact=False))
        self.release()
        self.assertEqual(self.g.phase, 'released')

    def test_time_reset_fails_loudly(self):
        self.tick(1.)
        with self.assertRaises(ValueError):
            self.g.after(0., sample(), False, 0., 0.)


if __name__ == '__main__': unittest.main()
