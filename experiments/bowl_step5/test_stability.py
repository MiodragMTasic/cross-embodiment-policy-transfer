import unittest
from stability import StableWindow


class StabilityTests(unittest.TestCase):
    def test_fifteen_samples_are_not_one_second(self):
        w = StableWindow()
        for i in range(15):
            self.assertFalse(w.update((i + 1) / 15, True, .01, .1))
        self.assertTrue(w.update(16 / 15, True, .01, .1))

    def test_brief_success_and_then_drop_resets_clock(self):
        w = StableWindow()
        self.assertFalse(w.update(0., True, 0., 0.))
        self.assertFalse(w.update(.9, False, 0., 0.))
        self.assertFalse(w.update(1., True, 0., 0.))
        self.assertFalse(w.update(1.9, True, 0., 0.))
        self.assertTrue(w.update(2., True, 0., 0.))

    def test_both_velocity_limits_and_nan(self):
        for linear, angular in [(.031, 0.), (0., .301), (float('nan'), 0.)]:
            w = StableWindow()
            w.update(0., True, 0., 0.)
            self.assertFalse(w.update(1., True, linear, angular))
            self.assertIsNone(w.since)

    def test_inclusive_thresholds_and_duplicate_time(self):
        w = StableWindow()
        w.update(0., True, .03, .3)
        self.assertTrue(w.update(1., True, .03, .3))
        with self.assertRaises(ValueError):
            w.update(1., True, 0., 0.)


if __name__ == '__main__':
    unittest.main()
