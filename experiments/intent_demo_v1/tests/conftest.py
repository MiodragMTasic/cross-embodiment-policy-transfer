import json
import os
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
sys.dont_write_bytecode = True

FIXTURES = json.loads((DEMO / 'fixtures/expected_specs.json').read_text())
EXPECTED = {k: v for k, v in FIXTURES.items() if not k.startswith('_')}
EVIDENCE = DEMO.parents[1] / 'assets/report/evidence/2026-10-24/step8_before_v1'
TASK = 'BananaInBowlUniformInitPose10cmTask'
needs_evidence = pytest.mark.skipif(not EVIDENCE.exists(), reason='sealed before-phase evidence not present')


def a_trace(layout):
    return EVIDENCE / 'RoboLab/output' / f'step8_A_P0_l{layout}_s0_v1' / TASK / 'strict-metric.json'


def b_trace(layout):
    return EVIDENCE / 'runs' / f'step8_B_P0_l{layout}_s0_v1' / 'gate_trace.jsonl'


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """No network, no real key, no real ledger: HOME is empty, the key is fake, the ledger and runs/ are temp paths."""
    import common
    import intent_to_spec as its
    import tf_client as tf
    runs = tmp_path / 'runs'
    monkeypatch.setattr(tf, 'LEDGER_PATH', runs / 'spend.jsonl')
    monkeypatch.setattr(common, 'RUNS', runs)
    monkeypatch.setattr(its, 'RUNS', tmp_path / 'demo_runs')
    monkeypatch.setattr(its, 'SEALED_LEDGER', tmp_path / 'sealed_spend.jsonl')
    monkeypatch.setattr(its, 'LADDER_LEDGER', tmp_path / 'ladder_spend.jsonl')
    monkeypatch.setattr(tf.time, 'sleep', lambda s: None)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('NEBIUS_API_KEY', 'sk-FAKE-intent-demo-never-real')

    def no_network(*a, **k):
        raise AssertionError('a test tried to reach the network')
    monkeypatch.setattr(tf.urllib.request, 'urlopen', no_network)
    monkeypatch.setattr(tf.TFClient, '_default_transport', lambda self: no_network)
    return tmp_path


def response(content, prompt=1500, completion=200):
    import intent_to_spec as its
    return {'id': 'r-mock', 'model': its.COMPILE_MODEL, 'usage': {'prompt_tokens': prompt, 'completion_tokens': completion},
            'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': content, 'refusal': None}}]}


def mock_client(replies):
    """A DemoClient whose transport pops canned replies and records every payload."""
    import intent_to_spec as its
    calls = []

    def transport(payload):
        calls.append(payload)
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    return its.DemoClient(transport=transport), calls


def write_literal_like(path, **over):
    spec = {'schema': 'intent-demo-literal-control-v1', 'policy_instruction': 'put the fruit away',
            'decision': {'attempt': True, 'category': 'literal_control'}, **over}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec))
    return path


def env_without(*names):
    return {k: v for k, v in os.environ.items() if k not in names}
