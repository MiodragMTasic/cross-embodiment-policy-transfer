"""Offline tests for tf_client: validation, spend cap, ledger, retries, key handling. No network: every client gets a
fake transport, HOME points at an empty temp dir (so the real key file can never be read) and the ledger is a temp file."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import failure_labeller as fl  # noqa: E402
import intent_contract as ic  # noqa: E402
import tf_client as tf  # noqa: E402

FAKE_KEY = 'sk-FAKE-0123456789abcdef-never-real'
GOOD_LABEL = {'outcome_guess': 'failure', 'failure_mode': 'ik_stuck_at_joint_limit', 'evidence_controls': [7488],
              'one_line_explanation': 'IK stagnation with joint 6 at its lower bound.', 'confidence': 0.8}


def response(content, prompt=1000, completion=100, finish='stop'):
    return {'id': 'r1', 'model': tf.LIGHTNING, 'usage': {'prompt_tokens': prompt, 'completion_tokens': completion},
            'choices': [{'finish_reason': finish, 'message': {'role': 'assistant', 'content': content, 'refusal': None}}]}


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        runs = self.tmp / 'runs'
        self.patches = [mock.patch.object(tf, 'LEDGER_PATH', runs / 'spend.jsonl'), mock.patch.object(common, 'RUNS', runs),
                        mock.patch.dict(os.environ, {'HOME': str(self.tmp), 'NEBIUS_API_KEY': FAKE_KEY}),
                        mock.patch.object(tf.time, 'sleep')]
        for p in self.patches:
            p.start()
        self.payload = tf.build_request(tf.LIGHTNING, [{'role': 'user', 'content': 'x' * 300}], fl.schema(), max_tokens=800)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def client(self, replies):
        calls = []

        def transport(payload):
            calls.append(payload)
            r = replies.pop(0)
            if isinstance(r, Exception):
                raise r
            return r
        c = tf.TFClient(transport=transport)
        return c, calls


class TestValidation(unittest.TestCase):
    def test_label_schema(self):
        self.assertEqual(tf.validate(fl.schema(), GOOD_LABEL), [])
        bad = dict(GOOD_LABEL, failure_mode='gremlins', confidence=2, extra=1)
        self.assertGreaterEqual(len(tf.validate(fl.schema(), bad)), 3)
        self.assertTrue(tf.validate(fl.schema(), {k: v for k, v in GOOD_LABEL.items() if k != 'evidence_controls'}))

    def test_hand_validator_agrees(self):
        for sch, good in ((fl.schema(7), GOOD_LABEL), (ic.schema(['banana', 'bowl', 'table']), {
                'object': 'banana', 'target': 'bowl', 'relation': 'in_or_on', 'constraints': [], 'ambiguities': [],
                'success_predicate': {'settle_on_or_in_target': True, 'commanded_release_required': True, 'rim_counts': True},
                'confidence': 0.9})):
            self.assertEqual(tf._hand_validate(sch, good, '<root>'), [])
            self.assertEqual(tf.validate(sch, good), [])
        bad = dict(GOOD_LABEL, evidence_controls=['c12'], confidence=True)
        self.assertEqual(bool(tf._hand_validate(fl.schema(), bad, '<root>')), bool(tf.validate(fl.schema(), bad)))

    def test_shuffled_enums_same_set(self):
        a, b = fl.schema(), fl.schema(123)
        self.assertEqual(set(a['properties']['failure_mode']['enum']), set(b['properties']['failure_mode']['enum']))

    def test_unknown_model_refused(self):
        with self.assertRaises(ValueError):
            tf.build_request('nvidia/nemotron-3_5-lightning', [], {})  # ids are case-sensitive


class TestSpend(Sandbox):
    def test_cap_refuses_before_calling(self):
        common.append_jsonl(tf.LEDGER_PATH, {'usd': tf.SPEND_CAP_USD - 0.0001, 'status': 'ok'})
        c, calls = self.client([response(json.dumps(GOOD_LABEL))])
        with self.assertRaises(tf.SpendCapExceeded):
            c.complete(self.payload, 'test')
        self.assertEqual(calls, [])
        self.assertEqual(len(tf.ledger_rows()), 1)

    def test_cap_is_not_env_overridable(self):
        import re
        src = Path(tf.__file__).read_text()
        self.assertEqual(re.findall(r"environ\.get\('(\w+)'", src), ['NEBIUS_API_KEY'])
        self.assertIn('SPEND_CAP_USD = 2.00\n', src)

    def test_usage_recorded(self):
        c, _ = self.client([response(json.dumps(GOOD_LABEL), prompt=1500, completion=200)])
        c.complete(self.payload, 'test')
        row = tf.ledger_rows()[-1]
        self.assertEqual((row['prompt_tokens'], row['completion_tokens'], row['usage_source']), (1500, 200, 'response'))
        self.assertAlmostEqual(row['usd'], (1500 * 0.06 + 200 * 0.24) / 1e6)

    def test_missing_usage_charged_worst_case(self):
        r = response(json.dumps(GOOD_LABEL))
        del r['usage']
        c, _ = self.client([r])
        c.complete(self.payload, 'test')
        self.assertAlmostEqual(tf.ledger_rows()[-1]['usd'], tf.estimate(self.payload)['worst_usd'])

    def test_estimate_worst_bounds_likely(self):
        e = tf.estimate(self.payload)
        self.assertGreaterEqual(e['worst_usd'], e['likely_usd'])
        self.assertGreater(e['prompt_tokens_est'], 100)


class TestRetries(Sandbox):
    def test_malformed_then_good(self):
        c, calls = self.client([response('{"outcome_guess": "fail'), response(json.dumps(GOOD_LABEL))])
        out = c.chat_json(self.payload, fl.schema(), 'test')
        self.assertTrue(out['ok'])
        self.assertEqual(len(out['attempts']), 2)
        self.assertIn('malformed', out['attempts'][0]['error'])
        self.assertEqual(len(calls[1]['messages']), len(calls[0]['messages']) + 2)

    def test_two_truncations_recorded_as_failure(self):
        c, calls = self.client([response('{"a"', finish='length'), response('{"a"', finish='length')])
        out = c.chat_json(self.payload, fl.schema(), 'test')
        self.assertFalse(out['ok'])
        self.assertEqual(calls[1]['max_tokens'], 2 * calls[0]['max_tokens'])
        self.assertEqual(len(tf.ledger_rows()), 2)

    def test_refusal_and_schema_violation(self):
        refusal = response('')
        refusal['choices'][0]['message']['refusal'] = 'no'
        c, _ = self.client([refusal, response(json.dumps(dict(GOOD_LABEL, failure_mode='gremlins')))])
        out = c.chat_json(self.payload, fl.schema(), 'test')
        self.assertFalse(out['ok'])
        self.assertIn('refusal', out['attempts'][0]['error'])
        self.assertIn('gremlins', out['error'])

    def test_json_schema_400_falls_back_to_json_object(self):
        c, calls = self.client([tf.TransportError(400, 'response_format json_schema unsupported'),
                                response(json.dumps(GOOD_LABEL))])
        out = c.chat_json(self.payload, fl.schema(), 'test')
        self.assertTrue(out['ok'])
        self.assertEqual(calls[1]['response_format'], {'type': 'json_object'})
        self.assertEqual(tf.ledger_rows()[0]['usd'], 0.0)

    def test_extra_check_runs(self):
        c, _ = self.client([response(json.dumps(GOOD_LABEL))] * 2)
        out = c.chat_json(self.payload, fl.schema(), 'test', check=lambda d: ['nope'])
        self.assertFalse(out['ok'])


class TestKey(Sandbox):
    def test_dry_run_reads_no_key(self):
        key = self.tmp / '.config/nebius/token-factory.key'
        key.parent.mkdir(parents=True)
        key.write_text(FAKE_KEY)
        key.chmod(0)  # any read would raise PermissionError
        with mock.patch.dict(os.environ, {'NEBIUS_API_KEY': ''}):
            c = tf.TFClient(dry_run=True)
            self.assertIsNone(c._key)
            self.assertEqual(c.chat_json(self.payload, fl.schema(), 'x')['dry_run'], True)

    def test_key_file_path_used_when_env_missing(self):
        (self.tmp / '.config/nebius').mkdir(parents=True)
        (self.tmp / '.config/nebius/token-factory.key').write_text(FAKE_KEY + '\n')
        with mock.patch.dict(os.environ, {'NEBIUS_API_KEY': ''}):
            self.assertEqual(tf.TFClient(transport=lambda p: None)._key, FAKE_KEY)

    def test_key_never_in_outputs(self):
        c, _ = self.client([])
        err = io.StringIO()

        def leaky(payload):  # a server or library that echoes the credential back, unredacted
            raise tf.TransportError(None, f'auth failed for Bearer {FAKE_KEY}')
        c._transport = leaky
        with contextlib.redirect_stdout(err), contextlib.redirect_stderr(err):
            out = c.chat_json(self.payload, fl.schema(), 'test')
            print(c, c.describe(self.payload), out)
        blob = err.getvalue() + tf.LEDGER_PATH.read_text() + json.dumps(out) + repr(c)
        self.assertNotIn(FAKE_KEY, blob)
        self.assertIn('[REDACTED]', tf.LEDGER_PATH.read_text())

    def test_urllib_transport_redacts_http_errors(self):
        c = tf.TFClient(transport=lambda p: None)
        body = io.BytesIO(f'{{"error": "bad key {FAKE_KEY}"}}'.encode())
        e = urllib.error.HTTPError(tf.BASE_URL, 401, 'Unauthorized', {}, body)
        with mock.patch('urllib.request.urlopen', side_effect=e):
            with self.assertRaises(tf.TransportError) as cm:
                c._urllib_transport(self.payload)
        self.assertNotIn(FAKE_KEY, str(cm.exception))
        self.assertEqual(cm.exception.status, 401)
        e.close()

    def test_openai_transport_redacts(self):
        try:
            import openai
        except ImportError:
            self.skipTest('openai not installed')
        import httpx2 as httpx
        c = tf.TFClient(transport=lambda p: None)
        req = httpx.Request('POST', tf.BASE_URL + 'chat/completions')
        err = openai.AuthenticationError(f'bad key {FAKE_KEY}', response=httpx.Response(401, request=req), body=None)
        with mock.patch.object(openai.resources.chat.Completions, 'create', side_effect=err):
            with self.assertRaises(tf.TransportError) as cm:
                c._openai_transport(self.payload)
        self.assertNotIn(FAKE_KEY, str(cm.exception))



class TestSmoke(Sandbox):
    def smoke(self, content):
        calls = []

        def transport(payload):
            calls.append(payload)
            return response(content, prompt=40, completion=12)
        out = io.StringIO()
        with mock.patch.object(tf.TFClient, '_default_transport', lambda self: transport), contextlib.redirect_stdout(out):
            code = tf.main(['--smoke'])
        return code, calls, out.getvalue()

    def test_smoke_single_call_conforming(self):
        code, calls, out = self.smoke('{"is_fruit": true, "colour_code": 3}')
        self.assertEqual((code, len(calls), len(tf.ledger_rows())), (0, 1, 1))
        self.assertEqual(json.loads(out)['parsed'], {'is_fruit': True, 'colour_code': 3})
        saved = list((common.RUNS / 'smoke').glob('*.json'))
        self.assertEqual(len(saved), 1)
        self.assertNotIn(FAKE_KEY, out + saved[0].read_text())

    def test_smoke_nonconforming_is_not_retried(self):
        code, calls, out = self.smoke('{"answer": "yes"}')
        self.assertEqual((code, len(calls)), (1, 1))
        self.assertTrue(json.loads(out)['schema_errors'])


if __name__ == '__main__':
    unittest.main()
