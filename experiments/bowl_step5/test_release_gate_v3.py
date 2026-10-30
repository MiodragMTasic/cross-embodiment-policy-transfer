import unittest
from test_release_gate import ReleaseTests, sample
from release_gate_v3 import ReleaseGateV3


class ReleaseV3Tests(ReleaseTests):
    def setUp(self):
        self.g = ReleaseGateV3()
        self.t = 0.

    def test_reclosure_and_recontact_reset_release(self):
        # Supersedes v2: revoke a real regrasp, not a detached empty-jaw close.
        for command, pre in ((1., sample(contact=True, closure=.1)),
                             (0., sample(contact=True, closure=.7))):
            self.setUp(); self.release()
            self.assertFalse(self.tick(command, pre, raw=True))
            self.assertEqual(self.g.phase, 'idle')

    def test_open_touch_restarts_whole_stability_window(self):
        self.release()
        for _ in range(10): self.tick(0., sample(contact=False, closure=.1), raw=True)
        self.assertFalse(self.tick(0., sample(contact=False, closure=.1),
                                  sample(contact=True, closure=.1), raw=True))
        self.assertEqual(self.g.phase, 'released')
        self.assertIsNone(self.g.stable.since)
        for _ in range(15):
            self.assertFalse(self.tick(0., sample(contact=False, closure=.1), raw=True))
        self.assertTrue(self.tick(0., sample(contact=False, closure=.1), raw=True))

    def test_pre_only_open_touch_resets_window(self):
        self.release()
        for _ in range(14): self.tick(0., sample(contact=False, closure=.1), raw=True)
        self.assertFalse(self.tick(0., sample(contact=True, closure=.1),
                                  sample(contact=False, closure=.1), raw=True))
        self.assertEqual(self.g.phase, 'released')
        for _ in range(14):
            self.assertFalse(self.tick(0., sample(contact=False, closure=.1), raw=True))
        self.assertTrue(self.tick(0., sample(contact=False, closure=.1), raw=True))

    def test_detached_closed_fingers_do_not_erase_release(self):
        self.release()
        for _ in range(14):
            self.assertFalse(self.tick(1., sample(contact=False, closure=.9), raw=True))
        self.assertTrue(self.tick(1., sample(contact=False, closure=.9), raw=True))

    def test_post_only_regrasp_revokes_release(self):
        for command, closure in ((1., .1), (0., .8)):
            self.setUp(); self.release()
            self.assertFalse(self.tick(command, sample(contact=False, closure=.1),
                                      sample(contact=True, closure=closure), raw=True))
            self.assertEqual(self.g.phase, 'idle')

    def test_geometric_escape_pre_or_post_requires_fresh_release(self):
        for pre, post in ((sample(contact=False, near_bowl=False), sample(contact=False, closure=.1)),
                          (sample(contact=False, closure=.1), sample(contact=False, near_bowl=False))):
            self.setUp(); self.release()
            self.assertFalse(self.tick(0., pre, post, raw=True))
            for _ in range(30):
                self.assertFalse(self.tick(0., sample(contact=False, closure=.1), raw=True))

    def test_open_brush_without_previous_release_never_passes(self):
        for _ in range(30):
            self.assertFalse(self.tick(0., sample(contact=True, closure=.1),
                                      sample(contact=False, closure=.1), raw=True))


if __name__ == '__main__': unittest.main()
