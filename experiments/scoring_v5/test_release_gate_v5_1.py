import sys
import unittest
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent / 'bowl_step5')]
from release_gate_v5_1 import ReleaseGateV51
from stages import mcnemar_exact, wilson
from test_release_gate_v5 import sample, AIR, BOWL, V5Tests

TABLE = dict(AIR, table_contact=True)


class V51Tests(V5Tests):
    def setUp(self): self.g = ReleaseGateV51(); self.t = 0.

    # v5 cases whose expected outcome V1/V3 deliberately change; their v5.1 versions follow
    def test_drop_onto_table_fails_even_if_pushed_in_later(self): self.skipTest('superseded by V1/V3')
    def test_first_touch_bowl_then_table_is_left_bowl(self): self.skipTest('superseded by V1/V3')
    def test_rim_with_table_contact_is_not_left_bowl(self): self.skipTest('superseded by V1/V3')

    def test_v1_straddle_never_settles_until_table_contact_ends(self):
        self.lift(); self.tick(0., sample(), sample(**BOWL))
        for _ in range(40): self.assertFalse(self.tick(0., sample(**BOWL, table_contact=True, rim_support=True), raw=True))
        self.assertEqual(self.g.phase, 'released')
        self.assertTrue(self.settle(rim_support=True))

    def test_v3_two_table_posts_are_a_blip(self):
        self.lift(); self.tick(0., sample(), sample(**BOWL))
        for _ in range(2): self.tick(0., sample(**TABLE))
        self.assertEqual(self.g.phase, 'released')
        self.assertTrue(self.settle())

    def test_v3_three_table_posts_leave_the_bowl(self):
        self.lift(); self.tick(0., sample(), sample(**BOWL))
        for _ in range(3): self.tick(0., sample(**TABLE))
        self.assertEqual(self.g.events[-1]['reason'], 'left_bowl')
        for _ in range(40): self.assertFalse(self.tick(0., sample(**BOWL), raw=True))

    def test_v3_drop_onto_table(self):
        self.lift(); self.tick(0., sample(), sample(**AIR))
        for _ in range(3): self.tick(0., sample(**TABLE))
        self.assertEqual(self.g.events[-1]['reason'], 'landed_off_bowl')
        for _ in range(40): self.assertFalse(self.tick(0., sample(**BOWL), raw=True))

    def test_v3_pre_samples_do_not_count(self):
        self.lift(); self.tick(0., sample(), sample(**BOWL))
        for _ in range(5): self.tick(0., sample(**TABLE), sample(**BOWL), raw=True)
        self.assertEqual(self.g.phase, 'released')

    def test_v3_counter_resets_on_invalidate(self):
        # N1: left_bowl after 3 table posts, re-arm, new drop with 1 table post then bowl must still land
        self.lift(); self.tick(0., sample(), sample(**BOWL))
        for _ in range(3): self.tick(0., sample(**TABLE))
        self.assertEqual(self.g.events[-1]['reason'], 'left_bowl')
        self.lift(); self.tick(0., sample(), sample(**TABLE))  # the detaching post itself touches only the table
        self.tick(0., sample(**BOWL))
        self.assertTrue(self.g.landed); self.assertEqual(self.g.phase, 'released')
        self.assertTrue(self.settle())

    def test_m5_helpers(self):
        self.assertEqual(mcnemar_exact(5, 0), .0625); self.assertEqual(mcnemar_exact(6, 0), .03125)
        self.assertEqual(mcnemar_exact(0, 0), 1.); self.assertEqual(mcnemar_exact(3, 3), 1.)
        lo, hi = wilson(0, 12); self.assertEqual(lo, 0.); self.assertAlmostEqual(hi, .2425, 3)


if __name__ == '__main__':
    unittest.main()
