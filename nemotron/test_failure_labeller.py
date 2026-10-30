"""Offline tests for the evidence-summary builder and the reviewer reference, on the real sealed step 8 evidence."""
import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import failure_labeller as fl  # noqa: E402
import tf_client as tf  # noqa: E402
from common import EVIDENCE, REVIEW_LOG  # noqa: E402

A, B = 'step8_A_P0_l4000_s0_v1', 'step8_B_P0_l4004_s0_v1'


def vm_hashes():
    out = {}
    for name in ('vm-sha256.txt', 'vm-sha256-views.txt'):
        for line in (EVIDENCE / name).read_text().splitlines():
            h, p = line.split(maxsplit=1)
            out[p.strip()] = h
    return out


class TestSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a, cls.a_refs = fl.summary(A)
        cls.b, cls.b_refs = fl.summary(B)

    def test_a4000_timeline(self):
        # the reviewer's missed win: open at 193 while held, banana in the bowl and still from 203
        self.assertIn('c193 OPEN cmd; tip contact off', self.a)
        self.assertIn('c203 free banana at rest 1 s', self.a)
        self.assertIn('RoboLab object-in-bowl predicate true', self.a)
        self.assertRegex(self.a, r'c85 tip contact off; touches table')  # the earlier slip

    def test_b4004_abort(self):
        self.assertIn('SAFETY ABORT (ik_action_A_to_B): runner stopped at control 7488', self.b)
        self.assertIn('joint 6 q=-2.0900 at lower bound -2.0900', self.b)
        self.assertIn('Trace: controls 1..7488', self.b)

    def test_blind_arm_has_no_verdicts(self):
        for text in (self.a, self.b):
            self.assertNotRegex(text, r'(?i)v5|v4|review|adjudicat|\bwin\b|pass\b|fail_reason')

    def test_with_v51_arm_adds_locked_fields(self):
        text, _ = fl.summary(A, 'with_v51')
        self.assertIn('Locked v5.1 scorer (offline): first-rest FAIL; fail_reason never_lifted', text)
        self.assertTrue(text.startswith(self.a))

    def test_refs_match_vm_hashes(self):
        vm, prefix = vm_hashes(), 'assets/report/evidence/2026-10-24/step8_before_v1/'
        refs = self.a_refs + self.b_refs
        self.assertEqual(len(refs), 1 + 5)
        for r in refs:
            self.assertTrue(r['file'].startswith(prefix), r)
            self.assertEqual(vm[r['file'][len(prefix):]], r['sha256'], r['file'])

    def test_all_24_build_within_budget(self):
        cases = fl.cases()
        self.assertEqual(len(cases), 24)
        for c in cases:
            text, refs = fl.summary(c)
            self.assertLess(len(text), 4000, c)
            self.assertLessEqual(len([l for l in text.splitlines() if l.startswith('  ')]), fl.MAX_EVENTS + 1, c)
            self.assertTrue(all(r['sha256'] for r in refs))

    def test_grounding(self):
        g = fl.grounding({'evidence_controls': [193, 203, 99999]}, self.a)
        self.assertEqual((g['grounded'], g['ungrounded']), ([193, 203], [99999]))

    def test_debounce(self):
        self.assertEqual(fl._debounce([0, 0, 1, 0, 0, 1, 1, 1, 0]), [(0, 0), (5, 1)])

    def test_messages_carry_schema(self):
        sch = fl.schema(5)
        m = fl.messages(self.a, sch)
        self.assertIn(json.dumps(sch), m[0]['content'])
        self.assertEqual(m[1]['content'], self.a + '\nLabel this episode.')
        self.assertEqual(tf.validate(sch, {'outcome_guess': 'success', 'failure_mode': 'success', 'evidence_controls': [193],
                                           'one_line_explanation': 'x', 'confidence': 0.5}), [])


class TestReviewerReference(unittest.TestCase):
    def test_quotes_verbatim_and_complete(self):
        log = REVIEW_LOG.read_text()
        section = log[log.index('## Step 8 "before" half adjudication'):]
        ref = fl.reviewer_reference()
        self.assertEqual(set(ref), set(fl.cases()))
        for case, r in ref.items():
            self.assertIn(r['quote'], section, case)

    def test_counts_match_adjudicated_headline(self):
        ref = fl.reviewer_reference()  # EXPERIMENT_REPORT §22.14: reviewer-adjudicated A 8/12, B 7/12
        self.assertEqual(sum(r['outcome'] == 'success' for c, r in ref.items() if '_A_' in c), 8)
        self.assertEqual(sum(r['outcome'] == 'success' for c, r in ref.items() if '_B_' in c), 7)
        locked = fl.locked_episodes()
        disagree = [c for c in ref if (ref[c]['outcome'] == 'success') != locked[c]['v5_1_pass']]
        self.assertEqual(disagree, ['step8_A_P0_l4000_s0_v1'])

    def test_modes_are_schema_labels(self):
        for r in fl.reviewer_reference().values():
            self.assertTrue(r['modes'] is None or set(r['modes']) <= set(fl.FAILURE_MODES))


if __name__ == '__main__':
    unittest.main()
