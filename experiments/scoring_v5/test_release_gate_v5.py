import sys
import unittest
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent / 'bowl_step5')]
from release_gate_v5 import ReleaseGateV5


def sample(**kw):
    d = dict(finite=True, contact=True, table_contact=False, bowl_contact=False, clearance_m=.03, near_bowl=False,
             closure=.7, bowl_support=False, rim_support=False)
    d.update(kw); return d


AIR = dict(contact=False, closure=.1)
BOWL = dict(contact=False, closure=.1, bowl_contact=True, bowl_support=True)


class V5Tests(unittest.TestCase):
    def setUp(self): self.g = ReleaseGateV5(); self.t = 0.
    def tick(self, cmd, pre=None, post=None, raw=False, speed=0.):
        pre = sample() if pre is None else pre; post = pre if post is None else post
        self.g.before(self.t, cmd, pre); self.t += 1/15
        return self.g.after(self.t, post, raw, speed, 0.)
    def lift(self):
        for _ in range(5): self.tick(1.)
        self.assertTrue(self.g.armed)
    def settle(self, **kw):
        for _ in range(15): self.assertFalse(self.tick(0., sample(**{**BOWL, **kw}), raw=True))  # 16 samples span 1 s
        return self.tick(0., sample(**{**BOWL, **kw}), raw=True)

    def test_drop_from_above_outside_geometry_lands_and_passes(self):
        self.lift()
        self.tick(0., sample(), sample(**AIR))  # open edge with near_bowl False: v4 rejects this
        self.assertEqual(self.g.phase, 'released'); self.assertFalse(self.g.landed)
        for _ in range(3): self.tick(0., sample(**AIR))  # falling
        self.assertTrue(self.settle()); self.assertTrue(self.g.landed)

    def test_set_down_touching_bowl_at_open_passes(self):
        self.lift()
        self.tick(0., sample(bowl_contact=True), sample(**BOWL))
        self.assertTrue(self.settle())

    def test_rim_counts(self):
        self.lift(); self.tick(0., sample(), sample(**AIR))
        self.assertTrue(self.settle(rim_support=True))

    def test_drop_onto_table_fails_even_if_pushed_in_later(self):
        self.lift(); self.tick(0., sample(), sample(**AIR))
        self.tick(0., sample(**AIR, table_contact=True))
        self.assertEqual(self.g.events[-1]['reason'], 'landed_off_bowl')
        for _ in range(40): self.assertFalse(self.tick(0., sample(**BOWL), raw=True))

    def test_first_touch_bowl_then_table_is_left_bowl(self):
        self.lift(); self.tick(0., sample(), sample(**BOWL))
        self.tick(0., sample(**AIR, table_contact=True))
        self.assertEqual(self.g.events[-1]['reason'], 'left_bowl')
        for _ in range(40): self.assertFalse(self.tick(0., sample(**BOWL), raw=True))

    def test_rim_with_table_contact_is_not_left_bowl(self):
        self.lift(); self.tick(0., sample(), sample(**BOWL))
        self.assertTrue(self.settle(table_contact=True, rim_support=True))

    def test_bounce_without_support_only_resets_window(self):
        self.lift(); self.tick(0., sample(), sample(**BOWL))
        for _ in range(5): self.tick(0., sample(**BOWL), raw=True)
        self.tick(0., sample(**AIR)); self.assertEqual(self.g.phase, 'released')
        self.assertTrue(self.settle())

    def test_closed_slip_onto_bowl_never_counts(self):
        self.lift()
        self.tick(1., sample(), sample(**{**BOWL, 'closure': .7}))
        for _ in range(40): self.assertFalse(self.tick(0., sample(**BOWL), raw=True))

    def test_regrasp_before_one_second_fails(self):
        self.lift(); self.tick(0., sample(), sample(**BOWL))
        for _ in range(5): self.tick(0., sample(**BOWL), raw=True)
        self.tick(1., sample(**{**BOWL, 'contact': True}))
        self.assertEqual(self.g.phase, 'idle')

    def test_unlifted_open_is_ineligible(self):
        for _ in range(10): self.tick(1., sample(table_contact=True))
        self.assertFalse(self.g.armed)
        self.tick(0., sample(table_contact=True), sample(**BOWL))
        for _ in range(40): self.assertFalse(self.tick(0., sample(**BOWL), raw=True))


if __name__ == '__main__':
    unittest.main()
