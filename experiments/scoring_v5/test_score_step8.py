"""score_step8.py: mock reproduction of the validation numbers, a synthetic step 8 after phase (names, R4, N4, B2 aborts),
S5's flag, and abort markers that must stay unscored."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_step8
import validate_score_step8 as v


class ScoreStep8Tests(unittest.TestCase):
    def test_mock_reproduces_validation(self):
        with tempfile.TemporaryDirectory() as t:
            r = v.main(t, '')
        self.assertEqual(r['episodes_checked'], 21)
        self.assertEqual(r['diffs'], [])
        self.assertTrue(r['m5_demo_reproduced'])
        self.assertTrue(r['S3_caveat_printed'])
        self.assertEqual(r['S5_disagreement_flags'], [])
        self.assertEqual(r['per_condition']['A/P0'], '6/15 [0.1982, 0.6425]')

    def test_synthetic_after_phase(self):
        r = v.synthetic_after()
        self.assertEqual([k for k, ok in r['checks'].items() if not ok], [])

    def test_stages_disagreement_is_a_flag_not_a_crash(self):
        real = score_step8.stages_r2.stages
        flip = lambda path: {**real(path), 'win': False}
        with tempfile.TemporaryDirectory() as t, mock.patch.object(score_step8.stages_r2, 'stages', flip):
            r = score_step8.main(['--phase', 'mock', '--out', t, str(v.ROOTS[3])])
        ep = {e['case']: e for e in r['episodes']}
        flags = ep['step5_training_layout2002_v4']['review_flags']
        self.assertTrue(ep['step5_training_layout2002_v4']['v5_1_pass'])  # headline stays score()['pass']
        self.assertTrue(any(f.startswith('stages win False') for f in flags), flags)
        self.assertFalse(any(f.startswith('stages win') for f in ep['step5_training_layout2003_v4']['review_flags']))

    def test_abort_markers_that_must_not_be_scored(self):
        soft, lift = v.B / 'step6_B_paint_live_base_s0_painted_v1', v.LIFT
        with tempfile.TemporaryDirectory() as t:
            runs = Path(t) / 'runs'
            v.write_abort(runs, 'step8_B_P1_l4007_s0_v1', soft, 1250, 'B_joint_beyond_soft_limit', short=1)  # no episodes.tsv row
            v.write_abort(runs, 'step8_B_P1_l4008_s0_v1', lift, 1700, 'ik_action_A_to_B', short=1)  # a surplus row only :298 may leave
            v.write_abort(runs, 'step8_B_P1_l4009_s0_v1', soft, 1250, 'B_joint_beyond_soft_limit', short=1)  # tsv exit status disagrees
            v.write_abort(runs, 'step8_B_P1_l4010_s0_v1', soft, 1250, 'B_joint_beyond_soft_limit', short=1,
                          marker=('safety_abort', 'policy_A_target_outside_limits'))  # marker kind disagrees with the classifier
            sup = runs / 'step8_after_v1-supervisor'
            sup.mkdir()
            (sup / 'episodes.tsv').write_text('\n'.join([v.TSV_HEAD, v.tsv_row('after', 'B', 'P1', '4008', 'step8_B_P1_l4008_s0_v1', '1', 'safety_abort'),
                                                         v.tsv_row('after', 'B', 'P1', '4009', 'step8_B_P1_l4009_s0_v1', '137', 'safety_abort'),
                                                         v.tsv_row('after', 'B', 'P1', '4010', 'step8_B_P1_l4010_s0_v1', '1', 'safety_abort')]) + '\n')
            r = score_step8.main(['--phase', 'after', '--out', str(Path(t) / 'out'), str(runs)])
        why = {e['layout']: e['unscored_reason'] for e in r['episodes']}
        self.assertFalse(any(e['scored'] for e in r['episodes']))
        self.assertEqual(why['l4007'], ['no_episode_row'])
        self.assertTrue(why['l4008'][0].startswith('trace shape: 1700 gate rows, 1699 tracking rows'), why['l4008'])
        self.assertEqual(why['l4009'], ['exit status: abort-classification.json 1, episodes.tsv 137'])
        self.assertTrue(why['l4010'][0].startswith('marker_disagrees_with_classifier: marker safety_abort/policy_A_target_outside_limits'), why['l4010'])


if __name__ == '__main__':
    unittest.main()
