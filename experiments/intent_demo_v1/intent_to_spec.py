"""Intent demo v1: free text -> Nemotron (Token Factory) -> validated task spec for the A (Franka) / P0 robot.

The model call goes through nemotron/tf_client.py's chat_json unchanged; complete() is replaced so that spend goes to
this demo's own ledger, runs/spend.jsonl (nemotron/runs is sealed and only read). Before every call: demo ledger +
worst case <= DEMO_CAP_USD, and the sealed ledger + the model-ladder ledger + the demo ledger + worst case <=
GLOBAL_CAP_USD. COMPILE_MODEL is what the intent model ladder picked by its pre-declared rule
(experiments/intent_model_ladder_v1/RESULT.md): Lightning, as originally declared. Reasoning (<think> blocks) is
stripped before parsing and logged; the raw response body is what is saved. The system prompt is the frozen file
prompts/intent_to_spec_v2.txt (v1 kept beside it); it is refused if its sha256 changed. Scene objects and the tested RoboLab wording come
from nemotron/intent_contract.py, which hash-checks the evidence copies of the RoboLab task files.

Nemotron proposes the spec. assess() then decides, without the model, whether the robot attempts it: banana in/on bowl
is in distribution, banana left_of/right_of/beside the bowl is attempted out of distribution, anything else is not
attempted. The robot only runs a spec when both agree it is attemptable; any disagreement is recorded in the spec file.

Proof: a live run saves, under the session folder runs/<ts>/nemotron/<slug>/, the exact user text, every HTTP request
and response body (served model id and token usage included), one record per repeat and the agreement across repeats.
The robot's spec, runs/<ts>/specs/<slug>.json, is repeat 1, and is written only when all repeats are valid and agree.

  python intent_to_spec.py --dry-run "put the fruit away"                  exact request and cost; no key read, no call
  python intent_to_spec.py --live --session runs/<ts> "put the fruit away"  3 calls (one retry each at most)
  python intent_to_spec.py --spent                                         demo spend, and all ledgers
--live spends money: run it only after Miodrag's yes.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'nemotron'))
import intent_contract as ic  # noqa: E402
import tf_client as tf  # noqa: E402

SCHEMA_ID = 'intent-demo-spec-v1'
PROMPT_VERSION = 'intent-to-spec-v2'
# v2 (25 Oct ~01:45 UTC) replaced v1 (sha 1ac61c22…) after v1's answers on the declared requests: v1 read "fruit" as not in
# the scene and once called left_of the tested task. v1, its session and its answers are kept; see REVIEW_PACKET.md Round 3.
PROMPT_FILE = HERE / 'prompts/intent_to_spec_v2.txt'
PROMPT_SHA256 = 'b184cf5660a45141a2dcec1369048603276bc5393bd25fa1bd77f1168894dd7d'
DEMO_PREFIX = 'intent_demo_v1'     # every ledger row of this demo, any prompt version: the demo cap counts them all
PURPOSE = 'intent_demo_v1_p2'      # rows written with this prompt version
DEMO_CAP_USD = 0.50
GLOBAL_CAP_USD = 5.00
COMPILE_MODEL = tf.LIGHTNING   # kept by the model ladder's pick rule: experiments/intent_model_ladder_v1/RESULT.md
# USD per 1M tokens (input, output), uncached: nemotron/docs_snapshot/models_info.json, as in the model ladder
PRICES = {'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B': (0.06, 0.24), 'nvidia/Nemotron-3_5-Lightning': (0.06, 0.24),
          'nvidia/nemotron-3-super-120b-a12b': (0.30, 0.90), 'nvidia/Nemotron-3-Ultra-550b-a55b': (1.00, 3.00)}
SEALED_LEDGER = PROJECT / 'nemotron/runs/spend.jsonl'
LADDER_LEDGER = PROJECT / 'experiments/intent_model_ladder_v1/runs/spend.jsonl'
RUNS = Path(os.environ.get('INTENT_DEMO_RUNS') or HERE / 'runs')   # the override is for the make_clips.sh test only
THINK = re.compile(r'<think>.*?</think>\s*', re.S)
MAX_TOKENS = 700
RELATIONS = ['in', 'on', 'left_of', 'right_of', 'beside', 'none']
ATTEMPTS = ['in_distribution', 'out_of_distribution', 'not_attempted']
ABSENT, NONE = ic.ABSENT, 'none'


def scene():
    """Scene objects and the tested wording, from the hash-checked RoboLab task copies."""
    objects, insts = ic.robolab_task()
    return objects, insts[0]['text']


def system_prompt(objects, validated):
    text = PROMPT_FILE.read_text()
    got = hashlib.sha256(text.encode()).hexdigest()
    if got != PROMPT_SHA256:
        raise RuntimeError(f'{PROMPT_FILE.name} sha256 {got} != frozen {PROMPT_SHA256}; refusing to use it')
    return text.replace('<<SCENE>>', ', '.join(objects)).replace('<<VALIDATED>>', validated)


def schema(objects):
    names = list(objects) + [ABSENT]
    text = {'type': 'array', 'items': {'type': 'string', 'maxLength': 160}, 'maxItems': 6}
    check = {'type': 'object', 'additionalProperties': False,
             'required': ['relation', 'object', 'reference', 'rim_counts', 'max_gap_m', 'require_released', 'require_rest',
                          'require_commanded_release'],
             'properties': {'relation': {'type': 'string', 'enum': RELATIONS},
                            'object': {'type': 'string', 'enum': names},
                            'reference': {'type': 'string', 'enum': names + [NONE]},
                            'rim_counts': {'type': 'boolean'},
                            'max_gap_m': {'type': 'number', 'minimum': 0, 'maximum': 0.3},
                            'require_released': {'type': 'boolean'}, 'require_rest': {'type': 'boolean'},
                            'require_commanded_release': {'type': 'boolean'}}}
    return {'type': 'object', 'additionalProperties': False,
            'required': ['object', 'target', 'relation', 'policy_instruction', 'feasible_for_this_robot', 'attempt', 'reason',
                         'success_check', 'ambiguities', 'confidence'],
            'properties': {'object': {'type': 'string', 'enum': names},
                           'target': {'type': 'string', 'enum': names + [NONE]},
                           'relation': {'type': 'string', 'enum': RELATIONS},
                           'policy_instruction': {'type': 'string', 'maxLength': 160},
                           'feasible_for_this_robot': {'type': 'boolean'},
                           'attempt': {'type': 'string', 'enum': ATTEMPTS},
                           'reason': {'type': 'string', 'maxLength': 400},
                           'success_check': check, 'ambiguities': text,
                           'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}}}


def consistency(spec):
    """Checks the schema cannot express; passed to chat_json as check=, so a failure triggers its one retry."""
    errs, sc = [], spec.get('success_check', {})
    for field, mirror in (('object', 'object'), ('target', 'reference'), ('relation', 'relation')):
        if sc.get(mirror) != spec.get(field):
            errs.append(f'success_check.{mirror} {sc.get(mirror)!r} != {field} {spec.get(field)!r}')
    inst = spec.get('policy_instruction', '')
    if not inst.strip() or '\n' in inst or not inst.isprintable():
        errs.append('policy_instruction must be one non-empty printable line')
    return errs


def assess(spec):
    """The robot-side decision, made without the model. Returns (category, reason)."""
    o, t, r = spec['object'], spec['target'], spec['relation']
    if o == 'banana' and t == 'bowl' and r in ('in', 'on'):
        return 'in_distribution', 'banana into the bowl: the only task P0 has been tested on in this scene'
    if o == 'banana' and t == 'bowl' and r in ('left_of', 'right_of', 'beside'):
        return 'out_of_distribution', f'banana {r} the bowl: never tried in this scene; attempted to see what P0 does'
    if ABSENT in (o, t):
        return 'not_attempted', 'names an object that is not in the scene'
    if r == 'none' or t == NONE:
        return 'not_attempted', 'no checkable placement goal'
    if o != 'banana':
        return 'not_attempted', f'moves the {o}; P0 has only been tested moving the banana, and the checker has no tested rule for it'
    return 'not_attempted', f'banana {r} the {t}: outside the tested task family and without a validated check'


def decide(spec):
    category, why = assess(spec)
    model_says = spec['attempt']
    attempt = category != 'not_attempted' and model_says != 'not_attempted'
    disagreements = []
    if model_says != category:
        disagreements.append(f'Nemotron attempt={model_says}, local rule={category}')
    if spec['feasible_for_this_robot'] != (category == 'in_distribution'):
        disagreements.append(f"Nemotron feasible_for_this_robot={spec['feasible_for_this_robot']}, local rule={category}")
    return {'attempt': attempt, 'category': category if attempt else 'not_attempted', 'local_rule': category,
            'local_reason': why, 'nemotron_attempt': model_says, 'disagreements': disagreements}


def request(text, objects=None, validated=None):
    if objects is None:
        objects, validated = scene()
    sch = schema(objects)
    messages = [{'role': 'system', 'content': system_prompt(objects, validated) + json.dumps(sch)},
                {'role': 'user', 'content': f'Request: "{text}"'}]
    body = tf.build_request(tf.LIGHTNING, messages, sch, name='intent_spec', max_tokens=MAX_TOKENS)
    return dict(body, model=COMPILE_MODEL), sch   # same body as the ladder tested: only the model field differs


def ledger():
    return RUNS / 'spend.jsonl'


def demo_spent(path=None):
    return sum(r['usd'] for r in tf.ledger_rows(path or ledger()) if str(r.get('purpose', '')).startswith(DEMO_PREFIX))


def all_spent():
    return tf.spent(SEALED_LEDGER) + tf.spent(LADDER_LEDGER) + tf.spent(ledger())


def usd(model, prompt_tokens, completion_tokens):
    p_in, p_out = PRICES[model]
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1e6


def worst_case(payload):
    text = ''.join(m['content'] for m in payload['messages']) + json.dumps(payload.get('response_format', ''))
    prompt = tf.estimate_tokens(text) + 8 * len(payload['messages'])
    return prompt, usd(payload['model'], prompt, payload['max_tokens'])


def strip_reasoning(resp):
    """(response for parsing, log). Removes <think> blocks from the content; never repairs the JSON itself."""
    resp = copy.deepcopy(resp)
    msg = ((resp.get('choices') or [{}])[0]).get('message') or {}
    content = msg.get('content') or ''
    stripped = THINK.sub('', content)
    log = {'stripped': stripped != content, 'think_tag_chars': len(content) - len(stripped),
           'reasoning_field_chars': sum(len(str(msg.get(f) or '')) for f in ('reasoning_content', 'reasoning'))}
    if log['stripped']:
        msg['content'] = stripped.strip()
    return resp, log


class DemoClient(tf.TFClient):
    """TFClient whose complete() checks both caps, calls and records in the demo ledger as one locked section, and
    logs every HTTP body sent and received (the raw proof)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = []

    def complete(self, payload, purpose):
        if not purpose.startswith(PURPOSE):
            raise ValueError(f'purpose must start with {PURPOSE!r}')
        prompt_est, worst = worst_case(payload)
        path = ledger()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix('.lock'), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            used, everything = demo_spent(), all_spent()
            if used + worst > DEMO_CAP_USD:
                raise tf.SpendCapExceeded(f'refused: demo spent ${used:.6f} + worst case ${worst:.6f} > demo cap ${DEMO_CAP_USD:.2f}')
            if everything + worst > GLOBAL_CAP_USD:
                raise tf.SpendCapExceeded(f'refused: all ledgers ${everything:.6f} + worst case ${worst:.6f} > cap ${GLOBAL_CAP_USD:.2f}')
            row = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'model': payload['model'], 'purpose': purpose}
            t0 = time.time()
            try:
                resp = self._transport(payload)
            except Exception as e:
                if not isinstance(e, tf.TransportError):
                    e = tf.TransportError(None, self.redact(f'{type(e).__name__}: {e}'))
                timeout = e.status is None   # a dropped connection may still be billed: charge the worst case
                append_ledger(path, {**row, 'status': 'transport_error', 'http_status': e.status,
                                     'prompt_tokens': prompt_est if timeout else 0,
                                     'completion_tokens': payload['max_tokens'] if timeout else 0,
                                     'usd': worst if timeout else 0.0,
                                     'usage_source': 'worst_case_estimate' if timeout else 'none_http_error',
                                     'error': self.redact(e)[:300]})
                self.calls.append({'request': payload, 'error': self.redact(e)})
                raise e from None
            u = resp.get('usage') or {}
            have = isinstance(u.get('prompt_tokens'), int) and isinstance(u.get('completion_tokens'), int)
            pt, ct = (u['prompt_tokens'], u['completion_tokens']) if have else (prompt_est, payload['max_tokens'])
            reasoning = (u.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
            if reasoning > ct:   # reported outside completion_tokens: bill it too
                ct += reasoning
            parsed, log = strip_reasoning(resp)
            append_ledger(path, {**row, 'status': 'ok', 'served_model': resp.get('model'), 'response_id': resp.get('id'),
                                 'prompt_tokens': pt, 'completion_tokens': ct, 'reasoning_tokens': reasoning,
                                 'usd': usd(payload['model'], pt, ct), 'usage_source': 'response' if have else 'worst_case_estimate',
                                 'latency_s': round(time.time() - t0, 3), 'reasoning_strip': log})
            if log['stripped'] or log['reasoning_field_chars']:
                print(f'  reasoning stripped before parsing: {purpose} {log}', flush=True)
            self.calls.append({'request': payload, 'response': resp, 'reasoning_strip': log})
            return parsed


def append_ledger(path, row):
    """The demo ledger is the one file under runs/ that is appended to rather than written once."""
    if Path(path).resolve() != ledger().resolve():
        raise PermissionError(f'refusing to append to {path}')
    with open(path, 'a') as f:
        f.write(json.dumps(row) + '\n')


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')[:60] or 'empty'


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def guarded(path):
    """Every write of this demo goes under runs/; anything else, or an existing file, is refused."""
    path = Path(path).resolve()
    if not path.is_relative_to(RUNS.resolve()) or path.exists():
        raise PermissionError(f'refusing to write {path}: outside runs/ or already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, obj):
    text = json.dumps(obj, indent=1) + '\n'
    guarded(path).write_text(text)
    return sha_text(text)


def compile_spec(text, client, repeat=1, out=None):
    """One validated spec record (never raises for model output; ok=False carries the error). With out (a folder under
    runs/), every HTTP request and response body of this compile is saved beside the record, with its sha256."""
    payload, sch = request(text)
    first = len(client.calls)
    result = client.chat_json(payload, sch, f'{PURPOSE}:{slug(text)}:r{repeat}', check=consistency)
    calls = []
    for k, c in enumerate(client.calls[first:], 1):
        resp = c.get('response') or {}
        row = {'call': k, 'served_model': resp.get('model'), 'response_id': resp.get('id'), 'usage': resp.get('usage'),
               'request_sha256': sha_text(json.dumps(c['request'], sort_keys=True)), 'error': c.get('error')}
        if out is not None:
            row['request_file'] = f'r{repeat}_call{k}_request.json'
            row['request_file_sha256'] = write_json(Path(out) / row['request_file'], c['request'])
            row['response_file'] = f'r{repeat}_call{k}_response.json'
            row['response_file_sha256'] = write_json(Path(out) / row['response_file'], resp or {'error': c.get('error')})
        calls.append(row)
    record = {'schema': SCHEMA_ID, 'input_text': text, 'input_sha256': sha_text(text), 'repeat': repeat,
              'ok': bool(result.get('ok')), 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'provenance': {'model_requested': payload['model'], 'endpoint': tf.BASE_URL, 'prompt_version': PROMPT_VERSION,
                             'prompt_file': str(PROMPT_FILE.relative_to(HERE)), 'prompt_sha256': PROMPT_SHA256,
                             'request_sha256': sha_text(json.dumps(payload, sort_keys=True)),
                             'scene_source': [ic.ref(p) for p in (ic.TASK_10CM, ic.TASK_BENCH)],
                             'calls': calls, 'attempt_errors': [a.get('error') for a in result.get('attempts', [])]},
              'label': 'DEMO. Nemotron annotation compiled into a robot spec; the robot acts only when decision.attempt is true.'}
    if not result.get('ok'):
        record.update(error=result.get('error'), decision={'attempt': False, 'category': 'not_attempted',
                                                           'local_reason': 'no valid spec from the model'})
        return record
    spec = result['data']
    record.update(nemotron=spec, decision=decide(spec), policy_instruction=spec['policy_instruction'],
                  success_check=spec['success_check'])
    return record


AGREE_KEYS = ['object', 'target', 'relation', 'policy_instruction', 'attempt', 'feasible_for_this_robot', 'success_check']


def agreement(records):
    specs = [r['nemotron'] for r in records if r['ok']]
    same = {k: len({json.dumps(s[k], sort_keys=True) for s in specs}) <= 1 for k in AGREE_KEYS}
    return {'valid': f'{len(specs)}/{len(records)}', 'all_valid': len(specs) == len(records), **same,
            'all_agree': len(specs) == len(records) and all(same.values())}


def asked_before(text):
    """F3: a request goes to Nemotron once per prompt version, across every session. Evidence of an earlier ask is a
    demo-ledger row for its slug under this version's PURPOSE, or a nemotron/<slug> folder in any session under runs/ whose
    snapshot holds this prompt file. Returns where it was found, or None."""
    s = slug(text)
    for r in tf.ledger_rows(ledger()):
        if str(r.get('purpose', '')).startswith(f'{PURPOSE}:{s}:'):
            return f"demo ledger row at {r.get('utc')}"
    for d in sorted(RUNS.glob(f'*/nemotron/{s}')):
        if (d.parents[1] / 'snapshot' / PROMPT_FILE.name).is_file():
            return str(d.relative_to(RUNS))
    return None


def run_live(text, session, repeats, client=None):
    """Live session step: nemotron/<slug>/{input.txt, r{n}.json, request/response bodies}; specs/<slug>.json is r1, written
    only when every repeat is valid and all agree on AGREE_KEYS. Returns (ok, summary)."""
    session = Path(session).resolve()
    if not session.is_relative_to(RUNS.resolve()) or not (session / 'declared_plan.json').is_file():
        raise ValueError(f'{session} is not a session made by demo_session.py new')
    out = session / 'nemotron' / slug(text)
    if out.exists():
        raise PermissionError(f'{out} exists; a request is compiled once per session')
    earlier = asked_before(text)
    if earlier:
        raise PermissionError(f'{text!r} was already sent to Nemotron ({earlier}); a request is compiled once per prompt version')
    client = client or DemoClient()
    guarded(out / 'input.txt').write_text(text)
    records = []
    for n in range(1, repeats + 1):
        rec = compile_spec(text, client, repeat=n, out=out)
        write_json(out / f'r{n}.json', rec)
        records.append(rec)
    agree = agreement(records)
    summary = {'request': text, 'agreement': agree, 'decision': records[0].get('decision'),
               'policy_instruction': records[0].get('policy_instruction'), 'spec_file': None}
    if agree['all_agree']:
        chosen = dict(records[0], chosen='r1 of ' + str(repeats), agreement=agree)
        spec_path = session / 'specs' / f'{slug(text)}.json'
        summary['spec_sha256'] = write_json(spec_path, chosen)
        summary['spec_file'] = str(spec_path.relative_to(session))
    write_json(out / 'agreement.json', summary)
    return agree['all_agree'], summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--dry-run', action='store_true')
    g.add_argument('--live', action='store_true')
    g.add_argument('--spent', action='store_true')
    ap.add_argument('--session', help='runs/<ts> made by demo_session.py new (required with --live)')
    ap.add_argument('--repeats', type=int, default=3, choices=[1, 2, 3])
    ap.add_argument('text', nargs='?')
    a = ap.parse_args(argv)
    if a.spent:
        print(f'demo spent ${demo_spent():.6f} of ${DEMO_CAP_USD:.2f}; all ledgers ${all_spent():.6f} of ${GLOBAL_CAP_USD:.2f}')
        return 0
    if not a.text:
        ap.error('an instruction text is required')
    if a.dry_run:
        payload, _ = request(a.text)
        print(f'POST {tf.BASE_URL}chat/completions\nAuthorization: Bearer <NEBIUS_API_KEY, loaded at runtime, never printed>\n'
              f'Content-Type: application/json\n\n{json.dumps(payload, indent=1)}\n\nworst case ${worst_case(payload)[1]:.6f}')
        print(f'demo cap: spent ${demo_spent():.6f} of ${DEMO_CAP_USD:.2f}; a live run makes {a.repeats} call(s), +1 retry each at most')
        return 0
    if not a.session:
        ap.error('--live needs --session runs/<ts>')
    ok, summary = run_live(a.text, a.session, a.repeats)
    print(json.dumps(summary, indent=1))
    print(f'demo spent ${demo_spent():.6f} of ${DEMO_CAP_USD:.2f}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
