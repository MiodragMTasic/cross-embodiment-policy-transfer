"""Synthetic checks of the revision 2 lift rule (S7), on B-format traces written to a temporary directory."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent / 'bowl_step5')]
from stages_r2 import stages
from test_release_gate_v5 import sample

UP = .03  # banana root rise, above LIFT_M


def trace(d, posts, rise):
    """One closed-command attempt: posts[i] overrides sample(); rise[i] is the banana root height above control 1."""
    rows, pre = [], sample(contact=False, closure=.1)
    for i, kw in enumerate(posts):
        post = sample(**kw)
        rows.append({'step': i + 1, 'simulation_s': (i + 2) / 15, 'command': 1., 'upstream_predicate': False,
                     'linear_m_s': 0., 'angular_rad_s': 0., 'pre': pre, 'post': post})
        pre = post
    where = np.zeros((len(rows), 2, 13))
    where[:, 0, 2] = rise
    where[:, 1, :2] = .3, 0.  # bowl far away: no over_bowl
    (d / 'gate_trace.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    np.savez(d / 'tracking.npz', objects_banana_bowl=where)
    return stages(d / 'gate_trace.jsonl')


class LiftR2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_table_contact_post_breaks_the_lift_run(self):
        rise = [0, 0, 0] + [UP] * 7
        clean = trace(self.d, [{}] * 10, rise)
        self.assertEqual(clean['best_attempt']['lift'], 6)  # posts 4-6 (rows 3-5) are free
        broken = trace(self.d, [{}] * 5 + [{'table_contact': True}] + [{}] * 4, rise)
        self.assertEqual(broken['best_attempt']['grasp'], 3)
        self.assertEqual(broken['best_attempt']['lift'], 9)  # the run restarts after the table post at row 5

    def test_posts_before_grasp_do_not_count(self):
        # free and high from row 1 (rise is measured from row 0), but contact drops at row 2, so grasp is row 5 (contact on rows 3-5)
        posts = [{}, {}, {'contact': False}] + [{}] * 7
        r = trace(self.d, posts, [0] + [UP] * 9)
        self.assertEqual(r['best_attempt']['grasp'], 6)
        self.assertEqual(r['best_attempt']['lift'], 8)  # rows 5-7: rows 3-4 are free but come before grasp
        self.assertEqual(r['furthest'], 'lift')

    def test_no_lift_without_grasp(self):
        r = trace(self.d, [{}, {}, {'contact': False}, {}, {}], [0] + [UP] * 4)
        self.assertIsNone(r['furthest'])


if __name__ == '__main__':
    unittest.main()
