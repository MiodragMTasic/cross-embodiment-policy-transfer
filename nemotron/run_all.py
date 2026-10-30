"""Run #1 (failure labeller, 24 step 8 "before" episodes) and #2 (intent contract, RoboLab + paraphrased instructions).

Writes nemotron/runs/<UTC stamp>/{inputs/, outputs/, report.md, report.json}. Prints the total estimated cost and
refuses to start if the worst case would pass the spend cap. Outputs are annotations: they are compared with the
locked v5.1 scores and the reviewer's adjudication and are never merged into either.

  python run_all.py --dry-run          build every request, write inputs/ and an estimate report; no key read, no call
  python run_all.py                    real run (only after "full run GO")
  python run_all.py --only intent --repeats 1 --dry-run
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import failure_labeller as fl  # noqa: E402
import intent_contract as ic  # noqa: E402
import tf_client as tf  # noqa: E402
from common import RUNS, SEPARATION, append_jsonl, text_sha, write_json, write_text  # noqa: E402

MIN_INTERVAL_S = 1.1  # stay under the documented ~60 requests per minute


def seed(*parts):
    return int(text_sha('|'.join(map(str, parts)))[:8], 16)


def jobs(model, repeats, arms, only, limit):
    """Every request as a dict: id, kind, payload, schema, and what the annotation must record."""
    out = []
    if only in (None, 'labeller'):
        locked = fl.locked_episodes()
        for case in fl.cases()[:limit]:
            for arm in arms:
                text, refs = fl.summary(case, arm, locked)
                for k in range(repeats):
                    order = None if k == 0 else seed(case, arm, k)
                    sch = fl.schema(order)
                    payload = tf.build_request(model, fl.messages(text, sch), sch, name='failure_label', max_tokens=800)
                    out.append({'id': f'label__{case}__{arm}__r{k}', 'kind': 'failure_label', 'case': case, 'arm': arm,
                                'repeat': k, 'enum_order_seed': order, 'payload': payload, 'schema': sch, 'summary': text,
                                'input_evidence': refs + ([fl.ref(fl.LOCKED_SCORE)] if arm == 'with_v51' else []),
                                'prompt_version': fl.PROMPT_VERSION, 'prompt_sha256': text_sha(fl.SYSTEM)})
    if only in (None, 'intent'):
        scene, insts = ic.instructions()
        for inst in insts:
            for k in range(repeats):
                order = None if k == 0 else seed(inst['id'], k)
                sch = ic.schema(scene, order)
                payload = tf.build_request(model, ic.messages(inst['text'], scene, sch), sch, name='intent_contract',
                                           max_tokens=600)
                out.append({'id': f"intent__{inst['id']}__r{k}", 'kind': 'intent_contract', 'instruction': inst,
                            'repeat': k, 'enum_order_seed': order, 'payload': payload, 'schema': sch,
                            'input_evidence': [inst['source']] if inst['source']['sha256'] else [],
                            'prompt_version': ic.PROMPT_VERSION, 'prompt_sha256': text_sha(ic.SYSTEM)})
    return out


def estimate_all(js):
    est = [tf.estimate(j['payload']) for j in js]
    return {'calls': len(js), 'prompt_tokens_est': sum(e['prompt_tokens_est'] for e in est),
            'max_completion_tokens': sum(e['max_completion_tokens'] for e in est),
            'likely_usd': sum(e['likely_usd'] for e in est), 'worst_usd': sum(e['worst_usd'] for e in est),
            'by_kind': {k: {'calls': sum(j['kind'] == k for j in js),
                            'prompt_tokens_est': sum(e['prompt_tokens_est'] for j, e in zip(js, est) if j['kind'] == k),
                            'likely_usd': sum(e['likely_usd'] for j, e in zip(js, est) if j['kind'] == k)}
                        for k in sorted({j['kind'] for j in js})}}


def annotate(j, res):
    """One annotation record; the raw attempts go to outputs/raw/."""
    last = (res.get('attempts') or [{}])[-1]
    resp = last.get('response') or {}
    rec = {'status': SEPARATION, 'id': j['id'], 'kind': j['kind'], 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'model': j['payload']['model'], 'served_model': resp.get('model'), 'prompt_version': j['prompt_version'],
           'prompt_sha256': j['prompt_sha256'], 'request_sha256': text_sha(json.dumps(j['payload'], sort_keys=True)),
           'response_format': last.get('mode'), 'input_evidence': j['input_evidence'], 'repeat': j['repeat'],
           'enum_order_seed': j['enum_order_seed'], 'ok': res.get('ok'), 'error': res.get('error'),
           'attempts': len(res.get('attempts') or []), 'usage': resp.get('usage'), 'output': res.get('data')}
    if j['kind'] == 'failure_label':
        rec.update(case=j['case'], arm=j['arm'], summary_sha256=text_sha(j['summary']))
        if res.get('ok'):
            rec['grounding'] = fl.grounding(res['data'], j['summary'])
    else:
        rec.update(instruction_id=j['instruction']['id'], instruction=j['instruction']['text'])
        if res.get('ok'):
            rec['comparison'] = ic.compare(res['data'], j['instruction'])
    return rec


def _majority(xs):
    if not xs:
        return None, 0
    (v, n), *rest = Counter(xs).most_common()
    return (v if not rest or rest[0][1] < n else 'split'), n


def labeller_report(recs):
    locked, reviewer = fl.locked_episodes(), fl.reviewer_reference()
    rows, metrics = [], {}
    for arm in sorted({r['arm'] for r in recs}):
        agree_rev = agree_v51 = mode_hits = mode_n = cited = grounded = n_cases = 0
        stab = []
        for case in fl.cases():
            mine = [r for r in recs if r['case'] == case and r['arm'] == arm]
            if not mine:
                continue
            n_cases += 1
            ok = [r for r in mine if r['ok']]
            outcome, n_out = _majority([r['output']['outcome_guess'] for r in ok])
            mode, n_mode = _majority([r['output']['failure_mode'] for r in ok])
            rv, v51 = reviewer[case], 'success' if locked[case]['v5_1_pass'] else 'failure'
            a_rev, a_v51 = outcome == rv['outcome'], outcome == v51
            m_ok = None if rv['modes'] is None else mode in rv['modes']
            agree_rev += a_rev
            agree_v51 += a_v51
            if m_ok is not None:
                mode_n += 1
                mode_hits += m_ok
            for r in ok:
                cited += len(r['grounding']['cited'])
                grounded += len(r['grounding']['grounded'])
            stab.append(n_mode / len(ok) if ok else 0)
            rows.append({'arm': arm, 'case': case, 'n_ok': len(ok), 'n_calls': len(mine), 'outcome': outcome, 'mode': mode,
                         'mode_votes': dict(Counter(r['output']['failure_mode'] for r in ok)),
                         'v5_1': v51, 'v5_1_fail_reason': locked[case]['fail_reason'], 'reviewer': rv['outcome'],
                         'reviewer_modes': rv['modes'], 'agree_reviewer': a_rev, 'agree_v5_1': a_v51, 'mode_agrees_reviewer': m_ok,
                         'explanation': ok[0]['output']['one_line_explanation'] if ok else None,
                         'evidence_controls': ok[0]['output']['evidence_controls'] if ok else None})
        metrics[arm] = {'cases': n_cases, 'outcome_agrees_reviewer': agree_rev, 'outcome_agrees_v5_1': agree_v51,
                        'mode_agrees_reviewer': mode_hits, 'mode_cases_with_reviewer_mechanism': mode_n,
                        'mean_mode_stability': round(sum(stab) / len(stab), 3) if stab else None,
                        'citations_grounded': f'{grounded}/{cited}'}
    return rows, metrics


def intent_report(recs):
    rows, by_group, insts = [], {}, {x['id']: x for x in ic.instructions()[1]}
    for iid in dict.fromkeys(r['instruction_id'] for r in recs):
        mine = [r for r in recs if r['instruction_id'] == iid]
        ok = [r for r in mine if r['ok']]
        inst = insts[iid]
        field = {f: sum(r['comparison']['vs_project'][f] for r in ok) for f in ic.FIELDS}
        faithful = {f: sum(r['comparison']['vs_faithful'][f] for r in ok) for f in inst['faithful']}
        flagged = sum(r['comparison']['flagged_ambiguity'] for r in ok)
        rows.append({'instruction_id': iid, 'text': inst['text'], 'group': inst['group'], 'n_ok': len(ok), 'n_calls': len(mine),
                     'agree_project_per_field': field, 'faithful_per_field': faithful, 'ambiguity_flagged': flagged,
                     'ambiguity_expected': inst['ambiguous'], 'first_output': ok[0]['output'] if ok else None})
        g = by_group.setdefault(inst['group'], {'n': 0, **{f: 0 for f in ic.FIELDS}})
        g['n'] += len(ok)
        for f in ic.FIELDS:
            g[f] += field[f]
    return rows, by_group


def markdown(stamp, est, lab, intent, dry, spent_before, headline=None):
    """spent_before is the ledger total when the run started (an estimate from returned token counts, not a bill)."""
    L = [f'# Nemotron on Token Factory: run {stamp}{" (DRY RUN, no calls made)" if dry else ""}', '',
         *([headline, ''] if headline else []),
         f'> {SEPARATION}', '',
         f"Calls {est['calls']}; estimated prompt tokens {est['prompt_tokens_est']:,} (chars/3); completion cap "
         f"{est['max_completion_tokens']:,}; likely ${est['likely_usd']:.4f}, worst ${est['worst_usd']:.4f}. "
         f'Ledger total before this run: ${spent_before:.6f} of the ${tf.SPEND_CAP_USD:.2f} cap (estimated usage '
         'cost from returned token counts and recorded prices).', '']
    for k, v in est['by_kind'].items():
        L.append(f"- {k}: {v['calls']} calls, ~{v['prompt_tokens_est']:,} prompt tokens, likely ${v['likely_usd']:.4f}")
    if lab:
        rows, metrics = lab
        L += ['', '## #1 Failure labeller vs locked v5.1 and the reviewer (compared, not merged)', '']
        for arm, m in metrics.items():
            L.append(f"- **{arm}**: outcome agrees with reviewer {m['outcome_agrees_reviewer']}/{m['cases']}, with v5.1 "
                     f"{m['outcome_agrees_v5_1']}/{m['cases']}; failure mode agrees with the reviewer's stated mechanism "
                     f"{m['mode_agrees_reviewer']}/{m['mode_cases_with_reviewer_mechanism']}; mean label stability "
                     f"{m['mean_mode_stability']}; cited controls present in the summary {m['citations_grounded']}")
        L += ['', '| Arm | Case | Nemotron outcome | Nemotron mode (votes) | v5.1 | v5.1 fail_reason | Reviewer | Agree rev / v5.1 | Explanation |',
              '|---|---|---|---|---|---|---|---|---|']
        for r in rows:
            L.append(f"| {r['arm']} | {r['case'][6:16]} | {r['outcome']} | {r['mode']} {r['mode_votes']} | {r['v5_1']} | "
                     f"{r['v5_1_fail_reason'] or '—'} | {r['reviewer']}{' ' + '/'.join(r['reviewer_modes']) if r['reviewer_modes'] else ''} | "
                     f"{'Y' if r['agree_reviewer'] else 'N'} / {'Y' if r['agree_v5_1'] else 'N'} | {(r['explanation'] or '').replace('|', '/')} |")
    if intent:
        rows, groups = intent
        L += ['', '## #2 Intent contract vs the project success rule (SCORING_V5.md)', '',
              'Counts are repeats agreeing with the project rule per field. For adversarial sentences, disagreement on '
              'the field the sentence changes is the correct reading; that is the "faithful" column.', '',
              '| Instruction | Group | ' + ' | '.join(ic.FIELDS) + ' | Faithful | Ambiguity flagged (expected) |',
              '|---|---|' + '---|' * len(ic.FIELDS) + '---|---|']
        for r in rows:
            L.append(f"| {r['text']} | {r['group']} | " + ' | '.join(f"{r['agree_project_per_field'][f]}/{r['n_ok']}" for f in ic.FIELDS)
                     + f" | {r['faithful_per_field']} | {r['ambiguity_flagged']}/{r['n_ok']} ({'yes' if r['ambiguity_expected'] else 'no'}) |")
        L += ['', '| Group | n | ' + ' | '.join(ic.FIELDS) + ' |', '|---|---|' + '---|' * len(ic.FIELDS)]
        for g, v in groups.items():
            L.append(f"| {g} | {v['n']} | " + ' | '.join(f"{v[f]}/{v['n']}" for f in ic.FIELDS) + ' |')
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--model', default=tf.LIGHTNING, choices=sorted(tf.PRICES))
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--arms', default='blind,with_v51')
    ap.add_argument('--only', choices=['labeller', 'intent'])
    ap.add_argument('--limit', type=int, help='first N episodes only')
    a = ap.parse_args(argv)
    arms = a.arms.split(',')
    assert set(arms) <= {'blind', 'with_v51'}, arms
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + ('-dryrun' if a.dry_run else '')
    out = RUNS / stamp
    spent_before = tf.spent()
    js = jobs(a.model, a.repeats, arms, a.only, a.limit)
    est = estimate_all(js)
    for j in js:
        write_json(out / 'inputs/requests' / f"{j['id']}.json", j['payload'])
        if j['kind'] == 'failure_label' and j['repeat'] == 0:
            write_text(out / 'inputs/summaries' / f"{j['case']}__{j['arm']}.txt", j['summary'])
    print(f"{est['calls']} calls; ~{est['prompt_tokens_est']:,} prompt tokens (chars/3 estimate), completion cap "
          f"{est['max_completion_tokens']:,}; estimated likely ${est['likely_usd']:.4f}, worst case ${est['worst_usd']:.4f}; "
          f'ledger ${tf.spent():.6f} of ${tf.SPEND_CAP_USD:.2f}')
    if a.dry_run:
        client = tf.TFClient(dry_run=True)
        for kind in ('failure_label', 'intent_contract'):
            first = next((j for j in js if j['kind'] == kind), None)
            if first:
                print(f"\n=== exact request for {first['id']} ===\n{client.describe(first['payload'])}")
        write_json(out / 'report.json', {'dry_run': True, 'status': SEPARATION, 'estimate': est})
        write_text(out / 'report.md', markdown(stamp, est, None, None, True, spent_before))
        print(f'\nwrote {out}')
        return 0
    if tf.spent() + est['worst_usd'] > tf.SPEND_CAP_USD:
        print(f"refused: worst case ${est['worst_usd']:.4f} on top of ${tf.spent():.6f} passes the ${tf.SPEND_CAP_USD:.2f} cap")
        return 2
    client, recs, last = tf.TFClient(), [], 0.0
    for n, j in enumerate(js, 1):
        time.sleep(max(0.0, MIN_INTERVAL_S - (time.time() - last)))
        last = time.time()
        try:
            res = client.chat_json(j['payload'], j['schema'], purpose=j['id'])
        except tf.SpendCapExceeded as e:
            print(f'stopping: {e}')
            break
        write_json(out / 'outputs/raw' / f"{j['id']}.json", res)
        rec = annotate(j, res)
        recs.append(rec)
        append_jsonl(out / 'outputs/annotations.jsonl', rec)
        print(f"[{n}/{len(js)}] {j['id']}: {'ok' if res['ok'] else 'FAILED ' + str(res.get('error'))[:120]}", flush=True)
    lab = labeller_report([r for r in recs if r['kind'] == 'failure_label']) if any(r['kind'] == 'failure_label' for r in recs) else None
    intent = intent_report([r for r in recs if r['kind'] == 'intent_contract']) if any(r['kind'] == 'intent_contract' for r in recs) else None
    write_json(out / 'report.json', {'status': SEPARATION, 'estimate': est, 'spent_before_usd': spent_before,
                                     'spent_after_usd': tf.spent(),
                                     'reviewer_source': fl.reviewer_source(), 'locked_score_source': fl.ref(fl.LOCKED_SCORE),
                                     'project_rule_source': ic.project_rule_source(), 'project_rule_quotes': ic.PROJECT_RULE_QUOTES,
                                     'labeller': {'rows': lab[0], 'metrics': lab[1]} if lab else None,
                                     'intent': {'rows': intent[0], 'groups': intent[1]} if intent else None})
    write_text(out / 'report.md', markdown(stamp, est, lab, intent, False, spent_before))
    print(f'\nwrote {out}; ledger now ${tf.spent():.6f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
