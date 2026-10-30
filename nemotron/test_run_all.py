"""Offline end-to-end: run_all with a fake transport writes annotations with provenance and a report, never the key;
--dry-run needs no key at all."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import run_all  # noqa: E402
import tf_client as tf  # noqa: E402

FAKE_KEY = 'sk-FAKE-e2e-0123456789-never-real'
LABEL = {'outcome_guess': 'success', 'failure_mode': 'success', 'evidence_controls': [193], 'one_line_explanation': 'x',
         'confidence': 0.5}
SPEC = {'object': 'banana', 'target': 'bowl', 'relation': 'in_or_on', 'constraints': [], 'ambiguities': [],
        'success_predicate': {'settle_on_or_in_target': True, 'commanded_release_required': True, 'rim_counts': False},
        'confidence': 0.9}


def fake(payload):
    body = LABEL if payload['response_format']['json_schema']['name'] == 'failure_label' else SPEC
    return {'id': 'x', 'model': payload['model'], 'usage': {'prompt_tokens': 900, 'completion_tokens': 80},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(body), 'refusal': None}}]}


class TestRunAll(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        runs = self.tmp / 'runs'
        self.patches = [mock.patch.object(common, 'RUNS', runs), mock.patch.object(run_all, 'RUNS', runs),
                        mock.patch.object(tf, 'LEDGER_PATH', runs / 'spend.jsonl'),
                        mock.patch.object(run_all, 'MIN_INTERVAL_S', 0),
                        mock.patch.dict(os.environ, {'HOME': str(self.tmp), 'NEBIUS_API_KEY': FAKE_KEY}),
                        mock.patch.object(tf.TFClient, '_default_transport', lambda self: fake)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_main(self, args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = run_all.main(args)
        return code, out.getvalue()

    def test_end_to_end_offline(self):
        code, out = self.run_main(['--limit', '2', '--repeats', '2'])
        self.assertEqual(code, 0, out)
        run = next(p for p in (self.tmp / 'runs').iterdir() if p.is_dir())
        recs = [json.loads(x) for x in (run / 'outputs/annotations.jsonl').read_text().splitlines()]
        self.assertEqual(len(recs), 2 * 2 * 2 + 11 * 2)
        for r in recs:
            self.assertTrue(r['ok'])
            self.assertEqual((r['model'], r['served_model']), (tf.LIGHTNING, tf.LIGHTNING))
            self.assertTrue(r['prompt_version'] and len(r['prompt_sha256']) == 64 and len(r['request_sha256']) == 64)
            self.assertIn('Not a score', r['status'])
        lab = [r for r in recs if r['kind'] == 'failure_label']
        self.assertTrue(all(e['sha256'] and e['file'].startswith('assets/report/evidence/') or 'scoring_v5' in e['file']
                            for r in lab for e in r['input_evidence']))
        report = json.loads((run / 'report.json').read_text())
        self.assertEqual(report['labeller']['metrics']['blind']['cases'], 2)
        self.assertEqual(len(report['intent']['rows']), 11)
        self.assertIn('Intent contract', (run / 'report.md').read_text())
        self.assertIn('Ledger total before this run: $0.000000', (run / 'report.md').read_text())  # not the after total
        self.assertEqual(report['spent_before_usd'], 0)
        self.assertGreater(report['spent_after_usd'], 0)
        self.assertEqual(len(tf.ledger_rows()), len(recs))
        everything = out + ''.join(p.read_text() for p in (self.tmp / 'runs').rglob('*') if p.is_file())
        self.assertNotIn(FAKE_KEY, everything)

    def test_dry_run_needs_no_key_and_calls_nothing(self):
        key = self.tmp / '.config/nebius/token-factory.key'
        key.parent.mkdir(parents=True)
        key.write_text(FAKE_KEY)
        key.chmod(0)
        with mock.patch.dict(os.environ, {'NEBIUS_API_KEY': ''}), \
                mock.patch.object(tf.TFClient, '_default_transport', side_effect=AssertionError('network')):
            code, out = self.run_main(['--dry-run', '--limit', '1', '--repeats', '1'])
        self.assertEqual(code, 0, out)
        self.assertIn('POST https://api.tokenfactory.nebius.com/v1/chat/completions', out)
        self.assertFalse(tf.LEDGER_PATH.exists())
        run = next(p for p in (self.tmp / 'runs').iterdir() if p.is_dir())
        self.assertTrue(json.loads((run / 'report.json').read_text())['dry_run'])

    def test_refuses_when_estimate_exceeds_cap(self):
        common.append_jsonl(tf.LEDGER_PATH, {'usd': tf.SPEND_CAP_USD - 1e-6})
        code, out = self.run_main(['--limit', '1', '--repeats', '1', '--only', 'labeller'])
        self.assertEqual(code, 2)
        self.assertIn('refused', out)
        self.assertEqual(len(tf.ledger_rows()), 1)


if __name__ == '__main__':
    unittest.main()
