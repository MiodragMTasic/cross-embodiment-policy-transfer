"""Synthetic checks of the EXPLORATORY pose-based first rest (pose_variant.py)."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent / 'bowl_step5')]
import pose_variant as pv
import score_step8
import validate_score_step8 as vss
from release_gate_v5_1 import load, score
from stability import StableWindow
from test_release_gate_v5 import sample

T = [(i + 2) / 15 for i in range(40)]  # B post times: (control + 1)/15
HELD = dict(centroid_height_above_bowl_m=.05)
AIR = dict(HELD, contact=False, closure=.1)
BOWL = dict(AIR, bowl_contact=True, bowl_support=True)


def about_z(theta):
    return [math.cos(theta / 2), 0., 0., math.sin(theta / 2)]


def b_trace(d, angular, quats):
    """Lift for 6 controls, open over the bowl at control 7, then rest in the bowl: v5.1 wins when the reported speeds are still."""
    posts = [sample(**HELD)] * 6 + [sample(**AIR)] + [sample(**BOWL)] * 33
    rows, pre = [], sample(**HELD)
    for i, post in enumerate(posts):
        rows.append({'step': i + 1, 'simulation_s': T[i], 'command': 1. if i < 6 else 0., 'upstream_predicate': i > 6,
                     'linear_m_s': 0., 'angular_rad_s': angular, 'pre': pre, 'post': post})
        pre = post
    where = np.zeros((len(rows), 2, 13))
    where[:, 0, 3:7] = quats
    (d / 'gate_trace.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    np.savez(d / 'tracking.npz', objects_banana_bowl=where)
    return pv.episode(d)


class PoseRateTests(unittest.TestCase):
    def test_sign_of_a_quaternion_does_not_matter(self):
        q = np.array([about_z(.01 * i) for i in range(10)])
        flipped = q * np.where(np.arange(10) % 2, -1., 1.)[:, None]  # q and -q are the same rotation
        np.testing.assert_allclose(pv.pose_rates(q, T[:10]), pv.pose_rates(flipped, T[:10]))
        self.assertAlmostEqual(pv.pose_rates(q, T[:10])[5], .01 * 15)

    def test_unnormalized_quaternions_are_normalized(self):
        q = np.array([about_z(.02 * i) for i in range(5)])
        np.testing.assert_allclose(pv.pose_rates(q * 3.7, T[:5]), pv.pose_rates(q, T[:5]))

    def test_time_step_must_be_one_fifteenth(self):
        q = [about_z(0)] * 4
        pv.pose_rates(q, [0, 1 / 15 + 5e-7, 2 / 15, 3 / 15])
        for t in ([0, 1 / 15 + 2e-6, 2 / 15, 3 / 15], [0, 1 / 15, 3 / 15, 4 / 15]):
            with self.assertRaisesRegex(pv.NotComputable, 'not 1/15'):
                pv.pose_rates(q, t)

    def test_count_mismatch_is_not_computable(self):
        with self.assertRaises(pv.NotComputable):
            pv.pose_rates([about_z(0)] * 3, T[:4])

    def test_first_row_never_qualifies_and_stays_finite(self):
        r = pv.pose_rates([about_z(0)] * 3, T[:3])
        self.assertTrue(math.isfinite(r[0]))  # a non-finite value would trip the gate's nonfinite_post_action, unlike v5.1
        self.assertFalse(StableWindow(duration_s=0.).update(1., True, 0., r[0]))
        self.assertEqual(r[1:], [0., 0.])

    def test_union_never_removes_a_v51_win(self):
        self.assertEqual(pv.combine(100, None), (100, 'v5.1'))
        self.assertEqual(pv.combine(100, 120), (100, 'v5.1'))
        self.assertEqual(pv.combine(100, 90), (90, 'pose'))
        self.assertEqual(pv.combine(None, 50), (50, 'pose'))
        self.assertEqual(pv.combine(100, 100), (100, 'both'))
        self.assertEqual(pv.combine(None, None), (None, None))

    def test_end_to_end_on_synthetic_b_traces(self):
        still, spinning = [about_z(0)] * 40, [about_z(.1 * i) for i in range(40)]  # 1.5 rad/s, over the 0.3 limit
        with tempfile.TemporaryDirectory() as t:
            v51_only = b_trace(Path(t), 0., spinning)
        self.assertEqual((v51_only['v5_1_win_control'], v51_only['pose_win_control']), (23, None))
        self.assertEqual((v51_only['union_win_control'], v51_only['won_by']), (23, 'v5.1'))  # the pose replay cannot take it away
        with tempfile.TemporaryDirectory() as t:
            pose_only = b_trace(Path(t), 1., still)
        self.assertEqual((pose_only['v5_1_win_control'], pose_only['pose_win_control']), (None, 23))
        self.assertEqual((pose_only['union_win_control'], pose_only['won_by']), (23, 'pose'))

    def test_one_after_call_per_row(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            b_trace(d, 0., [about_z(0)] * 40)
            path = d / 'gate_trace.jsonl'
            robot, rows, _, _ = load(path)
            gate = pv.pose_gate(pv.pose_rates(pv.quaternions(robot, path, rows), [r['simulation_s'] for r in rows]))
            score(path, gate_cls=gate)
        self.assertEqual(gate.last_row, len(rows) - 1)

    def test_b_outside_step8_needs_equal_lengths(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            b_trace(d, 0., [about_z(0)] * 40)
            z = np.load(d / 'tracking.npz')['objects_banana_bowl']
            np.savez(d / 'tracking.npz', objects_banana_bowl=z[:-1])
            r = pv.episode(d)
        self.assertEqual((r['scored'], r['unscored_reason']), (False, '40 gate rows, 39 tracking rows'))

    def test_step8_is_an_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / 'step8_B_P0_l4000_s0_v1'  # synthetic name in a tmp dir; nothing real is read
            d.mkdir()
            with self.assertRaisesRegex(SystemExit, 'without --step8'):
                pv.main([str(d)])
            for argv in (['--phase', 'after', t], ['--step8', '--phase', 'mock', t], ['--step8', t]):
                with self.assertRaises(SystemExit):
                    pv.main(argv)
            empty = pv.main(['--step8', '--phase', 'after', t])['episodes']  # an empty step8-named dir: listed, unscored (R4)
        self.assertEqual([(e['case'], e['scored'], e['unscored_reason'][-1]) for e in empty], [(d.name, False, 'no trace found')])

    def test_step8_rides_on_score_step8(self):
        """F2 on score_step8's synthetic after-phase fixtures (validate_score_step8.py, l4000-l4006), run while they exist."""
        real, seen = score_step8.main, {}
        def spy(argv):
            seen['headline'] = real(argv)
            seen['variant'] = pv.step8(argv[argv.index('--out') + 2:], 'after', headline=seen['headline'])
            return seen['headline']
        with mock.patch.object(score_step8, 'main', spy):
            checks = vss.synthetic_after()['checks']
        self.assertEqual([k for k, ok in checks.items() if not ok], [])  # score_step8's own checks still hold
        h, var = {e['case']: e for e in seen['headline']['episodes']}, seen['variant']
        ep = {e['case']: e for e in var['episodes']}
        self.assertEqual(var['headline_check'], [])
        self.assertEqual(var['disagreements'], [])
        self.assertEqual(ep.keys(), h.keys())
        for c, e in h.items():  # the union contains the headline
            self.assertEqual(ep[c]['scored'], e['scored'])
            if e['scored'] and e['v5_1_pass']:
                self.assertTrue(ep[c]['union_pass'])
                self.assertLessEqual(ep[c]['union_win_control'], e['win_control'])
        for c in ('step8_B_P1_l4003_s0_v1', 'step8_B_P1_l4004_s0_v1'):
            self.assertFalse(h[c]['scored'] or ep[c]['scored'])
        self.assertEqual((ep['step8_B_P1_l4002_s0_v1']['scored_trace_rows'], h['step8_B_P1_l4002_s0_v1']['safety_abort']['scored_rows']), (1249, 1249))
        step0 = ep['step8_B_P1_l4006_s0_v1']
        self.assertEqual((step0['union_pass'], step0['not_computable']), (False, 'empty_trace'))
        self.assertEqual(ep['step8_B_P1_l4005_s0_v1']['union_win_control'], 1554)


if __name__ == '__main__':
    unittest.main()
