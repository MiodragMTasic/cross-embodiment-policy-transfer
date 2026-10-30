"""Offline tests for analyse_run: the outcome metrics on synthetic cases, the ledger window, the numbers from the saved
run 20261024T182236Z (no key, no call), and the freeze of prompt v2 against PREREG_after_v2.md."""
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_run as ar  # noqa: E402
import failure_labeller as fl  # noqa: E402
from common import PKG, RUNS  # noqa: E402

RUN = RUNS / '20261024T182236Z'
PROMPT_V2 = PKG / 'prompts/failure_labeller_v2.txt'
PREREG = PKG / 'PREREG_after_v2.md'


class TestMetrics(unittest.TestCase):
    def test_confusion_and_balanced_accuracy(self):
        pairs = [('success', 'success')] * 3 + [('success', 'failure')] + [('failure', 'failure')] * 2 + \
                [('failure', 'success'), ('failure', 'split'), ('success', 'uncertain'), ('failure', None)]
        c = ar.confusion(pairs)
        self.assertEqual(c['matrix'], {'success': {'success': 3, 'failure': 1, 'other': 1},
                                       'failure': {'success': 1, 'failure': 2, 'other': 2}})
        self.assertEqual(c['recall'], {'success': [3, 5], 'failure': [2, 5]})
        self.assertEqual((c['correct'], c['n']), (5, 10))
        self.assertAlmostEqual(c['balanced_accuracy'], (3 / 5 + 2 / 5) / 2)

    def test_absent_class_uses_the_class_present(self):
        c = ar.confusion([('success', 'success'), ('success', 'failure')])
        self.assertEqual(c['recall']['failure'], [0, 0])
        self.assertAlmostEqual(c['balanced_accuracy'], 0.5)

    def test_majority_baseline(self):
        b = ar.majority_baseline(['failure'] * 5 + ['success'] * 3)
        self.assertEqual((b['always'], b['correct'], b['balanced_accuracy']), ('failure', 5, 0.5))
        self.assertEqual(ar.majority_baseline(['failure', 'success'])['always'], 'success')  # tie goes to success

    def test_ledger_window(self):
        row = lambda p, t, usd: {'purpose': p, 'utc': t, 'usd': usd}  # noqa: E731
        rows = [row('smoke', '2026-01-01T00:00:00Z', 1.0), row('a', '2026-01-01T00:01:00Z', 2.0),
                row('b', '2026-01-01T00:02:00Z', 3.0), row('a', '2026-01-02T00:00:00Z', 100.0)]  # a later run reusing 'a'
        w = ar.ledger_window(rows, {'a', 'b'}, '2026-01-01T00:00:30Z', '2026-01-01T00:05:00Z')
        self.assertEqual((w['before_run_usd'], w['run_usd'], w['through_run_usd'], w['run_rows']), (1.0, 5.0, 6.0, 2))
        self.assertEqual(w['before_run_purposes'], {'smoke': 1})
        self.assertIsNone(ar.ledger_window(rows, {'zzz'}, '2026', '2027'))


class TestSavedRun(unittest.TestCase):
    """The saved run's numbers, recomputed offline (expected values from the audit, EXPERIMENT_REPORT §22.21)."""

    @classmethod
    def setUpClass(cls):
        cls.out, cls.text = ar.render(RUN)
        cls.o = cls.out['outcome_vs_reviewer']

    def test_blind_majority(self):
        b = self.o['majority_of_repeats']['blind']
        self.assertEqual((b['correct'], b['n']), (15, 24))
        self.assertEqual(b['recall'], {'success': [10, 15], 'failure': [5, 9]})
        self.assertAlmostEqual(b['balanced_accuracy'], (10 / 15 + 5 / 9) / 2)
        self.assertEqual(round(b['balanced_accuracy'], 2), 0.61)

    def test_baseline_equals_blind(self):
        base = self.o['baseline']
        self.assertEqual((base['always'], base['correct'], base['balanced_accuracy']), ('success', 15, 0.5))
        self.assertEqual(base['correct'], self.o['majority_of_repeats']['blind']['correct'])

    def test_with_v51_and_locked_scorer(self):
        self.assertEqual(self.o['majority_of_repeats']['with_v51']['correct'], 19)
        v = self.o['v5_1']
        self.assertEqual((v['correct'], v['recall']), (23, {'success': [14, 15], 'failure': [9, 9]}))
        self.assertEqual(self.o['errors']['v5_1'], {'false_failure': ['A 4000'], 'false_success': [], 'other': []})

    def test_per_call(self):
        self.assertEqual(self.o['per_call']['blind']['correct'], 46)
        self.assertEqual(self.o['per_call']['with_v51']['correct'], 58)

    def test_ledger_before_run_is_the_smoke_test(self):
        w = self.out['tf']['ledger']
        self.assertEqual(w['before_run_purposes'], {'smoke': 1})
        self.assertAlmostEqual(w['before_run_usd'], 6.12e-06)
        self.assertEqual(w['run_rows'], 177)
        self.assertAlmostEqual(w['through_run_usd'], 0.01977318)
        self.assertIn('Ledger total before this run: $0.000006', self.text)

    def test_intent_faithful_checks(self):
        self.assertEqual(self.out['intent_faithful_by_group']['adversarial'],
                         {'instructions': 5, 'fields': 14, 'checks': 42, 'pass': 42})

    def test_report_wording(self):
        self.assertTrue(self.text.splitlines()[2].startswith('**Headline.**'))
        self.assertIn('did not beat a trivial always-"success" baseline on these 24 episodes', self.text)
        self.assertIn('This is a measured negative result', self.text)
        self.assertNotRegex(self.text, r'(?i)\bbilled\b|exact local ledger|schema enforced|\bwe\b|\bour\b')

    def test_tables_match_the_run_as_generated(self):
        """Everything run_all generated at run time comes back byte-identical, except the relabelled ledger line."""
        v1 = (RUN / 'report_v1_as_generated.md').read_text().split('\n## Run statistics')[0]
        new = self.text.split('\n## Outcome agreement')[0]
        new = '\n'.join(x for x in new.splitlines() if not x.startswith('**Headline.**'))
        drop = re.compile(r' Ledger (spent|total) before this run.*$')
        strip = lambda t: [drop.sub('', x) for x in t.splitlines() if x]  # noqa: E731
        self.assertEqual(strip(new), strip(v1))

    def test_notes_appended_once(self):
        self.assertEqual(self.text.count('## Takeaways'), 1)


class TestPrereg(unittest.TestCase):
    def test_prompt_hash_registered(self):
        sha = hashlib.sha256(PROMPT_V2.read_bytes()).hexdigest()
        self.assertIn(f'`{sha}`', PREREG.read_text())

    def test_v1_prompt_unchanged(self):
        recorded = {json.loads(x)['prompt_sha256'] for x in (RUN / 'outputs/annotations.jsonl').read_text().splitlines()
                    if json.loads(x)['kind'] == 'failure_label'}
        self.assertEqual(recorded, {hashlib.sha256(fl.SYSTEM.encode()).hexdigest()})
        self.assertIn(f'`{recorded.pop()}`', PREREG.read_text())

    def test_v2_changes_only_the_declared_lines(self):
        v1, v2 = fl.SYSTEM.splitlines(), PROMPT_V2.read_text().splitlines()
        changed = [x for x in v1 if x not in v2]
        self.assertEqual([x.split(':')[0] for x in changed], ['- ends_on_table', '- ends_on_rim'])
        added = [x for x in v2 if x not in v1]
        self.assertEqual([x.split(':')[0] for x in added], ['First-rest rule', 'Rim rule', '- ends_on_table', '- ends_on_rim'])
        self.assertTrue(PROMPT_V2.read_text().endswith('matching this schema:\n'))


if __name__ == '__main__':
    unittest.main()
