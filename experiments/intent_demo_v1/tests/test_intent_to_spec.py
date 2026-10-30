"""Spec schema, consistency, the local attempt rule, the frozen prompt, and the mocked Token Factory path (no network)."""
import copy
import json

import pytest

import intent_to_spec as its
import tf_client as tf
from conftest import EXPECTED, mock_client, response

FRUIT, LEFT, BOWL = 'put the fruit away', 'put the banana beside the bowl, on its left', 'pick up the bowl and put it on the banana'
OBJECTS = ['banana', 'bowl', 'table']


def test_prompt_is_frozen_and_filled(monkeypatch):
    objects, validated = its.scene()
    assert objects == OBJECTS and validated == 'Pick up the banana and place it in the bowl'
    text = its.system_prompt(objects, validated)
    assert 'Scene objects: banana, bowl, table.' in text and f'"{validated}"' in text and '<<' not in text
    monkeypatch.setattr(its, 'PROMPT_SHA256', '0' * 64)
    with pytest.raises(RuntimeError, match='refusing'):
        its.system_prompt(objects, validated)


@pytest.mark.parametrize('name', list(EXPECTED))
def test_expected_specs_validate(name):
    sch = its.schema(OBJECTS)
    spec = EXPECTED[name]
    assert tf.validate(sch, spec) == []
    assert tf._hand_validate(sch, spec, '<root>') == []
    assert its.consistency(spec) == []


@pytest.mark.parametrize('mutate, field', [
    (lambda s: s.update(relation='under'), 'relation'),
    (lambda s: s.update(extra=1), 'extra'),
    (lambda s: s['success_check'].update(max_gap_m=0.5), 'max_gap_m'),
    (lambda s: s['success_check'].pop('require_rest'), 'require_rest'),
    (lambda s: s.update(object='apple'), 'object'),
    (lambda s: s.update(attempt='maybe'), 'attempt'),
])
def test_schema_rejects(mutate, field):
    spec = copy.deepcopy(EXPECTED[FRUIT])
    mutate(spec)
    errs = tf.validate(its.schema(OBJECTS), spec)
    assert errs and any(field in e or 'enum' in e or 'Additional' in e or 'required' in e or 'greater' in e for e in errs)


def test_consistency_catches_mismatched_check_and_bad_instruction():
    spec = copy.deepcopy(EXPECTED[LEFT])
    spec['success_check']['relation'] = 'in'
    spec['policy_instruction'] = 'line one\nline two'
    errs = its.consistency(spec)
    assert any('success_check.relation' in e for e in errs) and any('one non-empty printable line' in e for e in errs)


@pytest.mark.parametrize('name, category, attempt', [
    (FRUIT, 'in_distribution', True), (LEFT, 'out_of_distribution', True), (BOWL, 'not_attempted', False)])
def test_three_demo_instructions_decide_as_planned(name, category, attempt):
    d = its.decide(EXPECTED[name])
    assert (d['category'], d['attempt'], d['disagreements']) == (category, attempt, [])


@pytest.mark.parametrize('over, why', [
    ({'object': 'not_in_scene'}, 'not in the scene'),
    ({'relation': 'none', 'target': 'none'}, 'no checkable'),
    ({'target': 'table', 'relation': 'on'}, 'outside the tested'),
    ({'object': 'bowl', 'target': 'banana'}, 'moves the bowl')])
def test_local_rule_refuses_everything_else(over, why):
    spec = dict(EXPECTED[FRUIT], **over)
    category, reason = its.assess(spec)
    assert category == 'not_attempted' and why in reason


def test_model_and_rule_must_both_agree_to_attempt():
    overclaim = dict(EXPECTED[BOWL], attempt='in_distribution', feasible_for_this_robot=True)
    d = its.decide(overclaim)
    assert not d['attempt'] and len(d['disagreements']) == 2
    timid = dict(EXPECTED[FRUIT], attempt='not_attempted')
    d = its.decide(timid)
    assert not d['attempt'] and d['category'] == 'not_attempted' and d['local_rule'] == 'in_distribution'


def test_compile_with_mocked_transport(sandbox):
    client, calls = mock_client([response(json.dumps(EXPECTED[LEFT]))])
    rec = its.compile_spec(LEFT, client)
    assert rec['ok'] and rec['decision']['attempt'] and rec['decision']['category'] == 'out_of_distribution'
    assert rec['policy_instruction'] == EXPECTED[LEFT]['policy_instruction']
    assert rec['success_check'] == EXPECTED[LEFT]['success_check']
    assert rec['provenance']['prompt_sha256'] == its.PROMPT_SHA256 and len(rec['provenance']['request_sha256']) == 64
    body = calls[0]
    assert body['model'] == its.COMPILE_MODEL and body['response_format']['type'] == 'json_schema'
    assert body['chat_template_kwargs'] == {'enable_thinking': False} and body['max_tokens'] == its.MAX_TOKENS
    assert json.dumps(LEFT) [1:-1] in body['messages'][1]['content']
    rows = tf.ledger_rows(its.ledger())
    assert len(rows) == 1 and rows[0]['purpose'].startswith(its.PURPOSE) and rows[0]['status'] == 'ok'
    assert rows[0]['usd'] == pytest.approx(its.usd(its.COMPILE_MODEL, 1500, 200))
    assert not tf.LEDGER_PATH.exists()   # the shared nemotron ledger (sealed) is never written
    assert 'sk-FAKE' not in json.dumps(rec)
    call = rec['provenance']['calls'][0]
    assert call['served_model'] == its.COMPILE_MODEL and call['usage'] == {'prompt_tokens': 1500, 'completion_tokens': 200}
    assert 'request_file' not in call   # no out folder: nothing written


def test_inconsistent_reply_gets_one_retry(sandbox):
    bad = copy.deepcopy(EXPECTED[FRUIT])
    bad['success_check']['reference'] = 'table'
    client, calls = mock_client([response(json.dumps(bad)), response(json.dumps(EXPECTED[FRUIT]))])
    rec = its.compile_spec(FRUIT, client)
    assert rec['ok'] and len(calls) == 2 and 'success_check.reference' in rec['provenance']['attempt_errors'][0]
    assert len(rec['provenance']['calls']) == 2


def test_two_bad_replies_give_a_not_attempted_record(sandbox):
    client, _ = mock_client([response('not json'), response('{"object": "banana"}')])
    rec = its.compile_spec(FRUIT, client)
    assert not rec['ok'] and rec['decision'] == {'attempt': False, 'category': 'not_attempted',
                                                 'local_reason': 'no valid spec from the model'}


def test_demo_cap_refuses_before_any_call(sandbox):
    its.ledger().parent.mkdir(parents=True)
    rows = [{'purpose': f'{its.PURPOSE}:x:r1', 'usd': 0.4999}, {'purpose': 'failure_label', 'usd': 1.0}]
    its.ledger().write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert its.demo_spent() == pytest.approx(0.4999)
    client, calls = mock_client([response(json.dumps(EXPECTED[FRUIT]))])
    payload, _ = its.request(FRUIT)
    with pytest.raises(tf.SpendCapExceeded, match='demo cap'):
        client.complete(payload, f'{its.PURPOSE}:t')
    assert calls == []
    with pytest.raises(ValueError):
        client.complete(payload, 'not_the_demo')


def test_global_cap_counts_all_three_ledgers(sandbox):
    its.SEALED_LEDGER.write_text(json.dumps({'purpose': 'other', 'usd': 2.0}) + '\n')
    its.LADDER_LEDGER.write_text(json.dumps({'purpose': 'intent_ladder_v1:x', 'usd': 2.9}) + '\n')
    its.ledger().parent.mkdir(parents=True)
    its.ledger().write_text(json.dumps({'purpose': f'{its.PURPOSE}:x', 'usd': 0.1}) + '\n')
    client, calls = mock_client([response(json.dumps(EXPECTED[FRUIT]))])
    payload, _ = its.request(FRUIT)
    with pytest.raises(tf.SpendCapExceeded, match=r'all ledgers \$5\.000000 .* cap \$5\.00'):
        client.complete(payload, f'{its.PURPOSE}:t')
    assert calls == []


def test_reasoning_is_stripped_before_parsing_and_logged(sandbox):
    raw = response('<think>the user wants the fruit in the bowl</think>\n' + json.dumps(EXPECTED[FRUIT]))
    client, _ = mock_client([raw])
    rec = its.compile_spec(FRUIT, client)
    assert rec['ok'] and rec['policy_instruction'] == EXPECTED[FRUIT]['policy_instruction']
    row = tf.ledger_rows(its.ledger())[0]
    assert row['reasoning_strip']['stripped'] and row['reasoning_strip']['think_tag_chars'] > 0
    assert client.calls[0]['response']['choices'][0]['message']['content'].startswith('<think>')   # raw body kept


def test_request_matches_the_ladder_body_shape(sandbox):
    payload, _ = its.request(FRUIT)
    base = tf.build_request(tf.LIGHTNING, payload['messages'], its.schema(its.scene()[0]), name='intent_spec',
                            max_tokens=its.MAX_TOKENS)
    assert list(payload) == list(base) and {**payload, 'model': tf.LIGHTNING} == base


def test_worst_case_per_call_is_tiny():
    payload, _ = its.request(FRUIT)
    est = tf.estimate(payload)
    assert est['worst_usd'] < 0.001  # 9 calls (3 instructions x 3 repeats, plus retries) stay far below the $0.50 cap


def test_dry_run_reads_no_key(sandbox, monkeypatch, capsys):
    monkeypatch.delenv('NEBIUS_API_KEY')
    assert its.main(['--dry-run', FRUIT]) == 0
    out = capsys.readouterr().out
    assert 'chat/completions' in out and 'never printed' in out and 'demo cap' in out
    with pytest.raises(FileNotFoundError):  # a live client would have to read the key file, which the sandbox lacks
        its.DemoClient()


def test_cli_needs_an_explicit_mode():
    with pytest.raises(SystemExit):
        its.main([FRUIT])


def test_guarded_refuses_outside_runs_and_overwrite(sandbox):
    its.write_json(its.RUNS / 'x' / 'a.json', {})
    with pytest.raises(PermissionError):
        its.write_json(its.RUNS / 'x' / 'a.json', {})
    with pytest.raises(PermissionError):
        its.write_json(its.RUNS / '..' / 'escape.json', {})


def test_live_needs_a_session(sandbox):
    client, calls = mock_client([])
    with pytest.raises(ValueError, match='demo_session.py new'):
        its.run_live(FRUIT, sandbox / 'nowhere', 1, client=client)
    with pytest.raises(SystemExit):
        its.main(['--live', FRUIT])
    assert calls == []


def _session(name, prompt=None):
    s = its.RUNS / name
    (s / 'snapshot').mkdir(parents=True)
    (s / 'declared_plan.json').write_text('{}')
    (s / 'snapshot' / (prompt or its.PROMPT_FILE.name)).write_text('prompt')
    return s


def test_reask_refused_across_sessions_by_folder(sandbox):
    """F3: an earlier session's nemotron/<slug> folder blocks a new compile; the client is never called."""
    (_session('20261025T000000Z') / 'nemotron' / its.slug(FRUIT)).mkdir(parents=True)
    later = _session('20261025T010000Z')
    client, calls = mock_client([])
    with pytest.raises(PermissionError, match='once per prompt version'):
        its.run_live(FRUIT, later, 3, client=client)
    assert calls == [] and not (later / 'nemotron').exists()


def test_reask_refused_across_sessions_by_ledger(sandbox):
    """F3: a demo-ledger row for the request blocks it even if every session folder is gone."""
    its.RUNS.mkdir(parents=True, exist_ok=True)
    its.ledger().write_text(json.dumps({'utc': '2026-10-25T00:00:00Z', 'usd': 0.0001,
                                        'purpose': f'{its.PURPOSE}:{its.slug(FRUIT)}:r1'}) + '\n')
    later = _session('20261025T010000Z')
    client, calls = mock_client([])
    with pytest.raises(PermissionError, match='demo ledger row'):
        its.run_live(FRUIT, later, 3, client=client)
    assert calls == []


def test_asked_before_ignores_other_requests(sandbox):
    its.RUNS.mkdir(parents=True, exist_ok=True)
    its.ledger().write_text(json.dumps({'utc': '2026-10-25T00:00:00Z', 'usd': 0.0001,
                                        'purpose': f'{its.PURPOSE}:put_the_fruit_away_now:r1'}) + '\n')
    (_session('20261025T000000Z') / 'nemotron' / 'something_else').mkdir(parents=True)
    assert its.asked_before(FRUIT) is None


def test_earlier_prompt_version_does_not_block(sandbox):
    """A v1 session's folder and v1 ledger rows do not block the current prompt version; the demo cap still counts v1 spend."""
    (_session('20261025T013018Z', prompt='intent_to_spec_v1.txt') / 'nemotron' / its.slug(FRUIT)).mkdir(parents=True)
    its.ledger().write_text(json.dumps({'utc': '2026-10-25T01:31:00Z', 'usd': 0.0004,
                                        'purpose': f'intent_demo_v1:{its.slug(FRUIT)}:r1'}) + '\n')
    assert its.PROMPT_FILE.name != 'intent_to_spec_v1.txt' and its.asked_before(FRUIT) is None
    assert abs(its.demo_spent() - 0.0004) < 1e-12
