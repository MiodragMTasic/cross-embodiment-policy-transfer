"""Run statistics for a finished run_all.py run, computed from its saved outputs only (no key, no call): validity,
outcome agreement with the reviewer (confusion matrix, per-class recall, balanced accuracy, the always-majority-class
baseline and the locked v5.1 scorer on the same footing), repeat consistency, citation checks, intent faithful-reading
results and Token Factory behaviour (latency, errors, retries, response_format).

  python analyse_run.py runs/<stamp>      writes runs/<stamp>/analysis.json and rebuilds runs/<stamp>/report.md

report.md is rebuilt from scratch each time, so re-running is idempotent: headline, then run_all's tables (recomputed
from outputs/annotations.jsonl and report.json's estimate), then these statistics, then the hand-written
runs/<stamp>/notes.md if it exists.
"""
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import failure_labeller as fl  # noqa: E402
import intent_contract as ic  # noqa: E402
import run_all  # noqa: E402
import tf_client as tf  # noqa: E402
from common import write_json, write_text  # noqa: E402

CLASSES = ('success', 'failure')
ARM_NAMES = {'blind': 'Nemotron blind', 'with_v51': 'Nemotron with v5.1 shown', 'v5_1': 'Locked v5.1 alone'}


def confusion(pairs):
    """pairs [(reference, predicted)] -> matrix, per-class recall, accuracy and balanced accuracy (mean recall over the
    classes present in the reference). A prediction outside CLASSES ('uncertain', or 'split' when repeats tie, or None
    when no repeat was valid) goes in its own column and counts as wrong."""
    m = {t: {p: 0 for p in (*CLASSES, 'other')} for t in CLASSES}
    for t, p in pairs:
        m[t][p if p in CLASSES else 'other'] += 1
    recall = {t: [m[t][t], sum(m[t].values())] for t in CLASSES}
    present = [h / n for h, n in recall.values() if n]
    correct = sum(m[t][t] for t in CLASSES)
    return {'n': len(pairs), 'correct': correct, 'accuracy': correct / len(pairs) if pairs else None,
            'recall': recall, 'balanced_accuracy': sum(present) / len(present) if present else None, 'matrix': m}


def majority_baseline(refs):
    """Always answer the reference's most common class; a tie goes to the first of CLASSES."""
    c = Counter(refs)
    always = max(CLASSES, key=lambda k: (c[k], -CLASSES.index(k)))
    return {'always': always, **confusion([(t, always) for t in refs])}


def short(case):
    _, robot, _, layout, *_ = case.split('_')
    return f'{robot} {layout[1:]}'


def outcomes(lab_rows, lab_recs):
    """Outcome agreement with the reviewer: per arm by majority of the repeats (primary) and per call (secondary), the
    locked v5.1 scorer judged the same way, and the always-majority baseline."""
    reviewer, locked = fl.reviewer_reference(), fl.locked_episodes()
    cases = fl.cases()
    v51 = {c: 'success' if locked[c]['v5_1_pass'] else 'failure' for c in cases}
    refs = [reviewer[c]['outcome'] for c in cases]
    out = {'reference': 'independent reviewer adjudication (failure_labeller.reviewer_reference)',
           'n_cases': len(cases), 'majority_of_repeats': {}, 'per_call': {}, 'by_robot': {}, 'errors': {},
           'v5_1': confusion([(reviewer[c]['outcome'], v51[c]) for c in cases]), 'baseline': majority_baseline(refs)}
    judged = {'v5_1': v51}
    for arm in sorted({r['arm'] for r in lab_rows}):
        judged[arm] = {r['case']: r['outcome'] for r in lab_rows if r['arm'] == arm}
        out['majority_of_repeats'][arm] = confusion([(reviewer[c]['outcome'], judged[arm].get(c)) for c in cases])
        out['per_call'][arm] = confusion([(reviewer[r['case']]['outcome'], r['output']['outcome_guess'])
                                          for r in lab_recs if r['arm'] == arm and r['ok']])
    for name, verdicts in judged.items():
        wrong = [c for c in cases if verdicts.get(c) != reviewer[c]['outcome']]
        out['errors'][name] = {'false_failure': [short(c) for c in wrong if reviewer[c]['outcome'] == 'success'],
                               'false_success': [short(c) for c in wrong if reviewer[c]['outcome'] == 'failure'],
                               'other': [short(c) for c in wrong if verdicts.get(c) not in CLASSES]}
        out['by_robot'][name] = {rb: [sum(verdicts.get(c) == reviewer[c]['outcome'] for c in cases if c.split('_')[1] == rb),
                                      sum(c.split('_')[1] == rb for c in cases)] for rb in ('A', 'B')}
    return out


def headline(o):
    """Plain headline from the numbers; the wording follows the comparison, it is not fixed text."""
    if 'blind' not in o['majority_of_repeats']:
        return None
    b, base, v = o['majority_of_repeats']['blind'], o['baseline'], o['v5_1']
    n, (sh, sn), (fh, fn), always = b['n'], b['recall']['success'], b['recall']['failure'], base['always']
    if b['correct'] <= base['correct']:
        s = (f"**Headline.** From text timelines, Nemotron's blind outcome agreement did not beat a trivial "
             f"always-\"{always}\" baseline on these {n} episodes: its majority vote matched my reviewer on "
             f"{b['correct']}/{n}, and always answering \"{always}\" scores {base['correct']}/{n}.")
    else:
        s = (f"**Headline.** From text timelines, Nemotron's blind majority vote matched my reviewer on {b['correct']}/{n} "
             f"episodes, {b['correct'] - base['correct']} more than always answering \"{always}\" ({base['correct']}/{n}).")
    s += (f" Balanced accuracy was {b['balanced_accuracy']:.2f} (success recall {sh}/{sn}, failure recall {fh}/{fn}) "
          f"against {base['balanced_accuracy']:.2f} for the baseline. My locked v5.1 scorer alone matched the reviewer on "
          f"{v['correct']}/{n}" + (', so it is far more reliable here.' if v['correct'] > b['correct'] else '.'))
    if 'with_v51' in o['majority_of_repeats']:
        w = o['majority_of_repeats']['with_v51']
        s += (f" Shown the scorer's verdict, Nemotron matched the reviewer on {w['correct']}/{n}"
              + (', fewer than the scorer it was shown.' if w['correct'] < v['correct'] else '.'))
    s += (' This is a measured negative result, and worth knowing.' if b['correct'] <= base['correct'] else
          ' At this sample size that alone is not evidence of skill; PREREG_after_v2.md sets the claim rule.')
    return s


def ledger_window(rows, purposes, start_utc, end_utc):
    """Ledger totals before, during and through this run. The ledger is append-only, so file order is call order; a
    row belongs to the run when its purpose is one of the run's request ids and its time falls inside the run."""
    idx = [i for i, r in enumerate(rows) if r['purpose'] in purposes and start_utc <= r['utc'] <= end_utc]
    if not idx:
        return None
    usd = lambda xs: sum(r['usd'] for r in xs)  # noqa: E731
    return {'before_run_usd': usd(rows[:idx[0]]), 'before_run_purposes': dict(Counter(r['purpose'] for r in rows[:idx[0]])),
            'run_usd': usd(rows[i] for i in idx), 'run_rows': len(idx), 'through_run_usd': usd(rows[:idx[-1] + 1]),
            'note': 'estimated usage cost from returned token counts and recorded prices; billing deduction not verified'}


def render(run):
    """(analysis dict, report.md text) for a finished run directory. Reads only; writes nothing."""
    run = Path(run).resolve()
    recs = [json.loads(x) for x in (run / 'outputs/annotations.jsonl').read_text().splitlines()]
    raws = {p.stem: json.loads(p.read_text()) for p in (run / 'outputs/raw').glob('*.json')}
    reqs = [json.loads(p.read_text()) for p in (run / 'inputs/requests').glob('*.json')]
    report_json = json.loads((run / 'report.json').read_text())
    start = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.strptime(run.name[:16], '%Y%m%dT%H%M%SZ'))
    all_rows = tf.ledger_rows()
    window = ledger_window(all_rows, set(raws), start, max(r['utc'] for r in recs))
    ledger = [r for r in all_rows if r['purpose'] in raws and start <= r['utc'] <= max(x['utc'] for x in recs)]
    lab = [r for r in recs if r['kind'] == 'failure_label']
    intent = [r for r in recs if r['kind'] == 'intent_contract']

    first_try = sum(1 for k, v in raws.items() if v['ok'] and len(v['attempts']) == 1)
    modes = Counter(a.get('mode') for v in raws.values() for a in v['attempts'])
    errors = Counter((a.get('error') or '').split(':')[0] for v in raws.values() for a in v['attempts'] if a.get('error'))
    lat = [r['latency_s'] for r in ledger if r.get('latency_s') is not None]
    tf_stats = {'calls_logged': len(ledger), 'http_ok': sum(r['status'] == 'ok' for r in ledger),
                'transport_errors': sum(r['status'] != 'ok' for r in ledger),
                'estimated_usage_usd': sum(r['usd'] for r in ledger), 'prompt_tokens': sum(r['prompt_tokens'] for r in ledger),
                'completion_tokens': sum(r['completion_tokens'] for r in ledger),
                'reasoning_tokens': sum(r.get('reasoning_tokens') or 0 for r in ledger),
                'calls_reporting_zero_reasoning_tokens': sum(r.get('reasoning_tokens') == 0 for r in ledger),
                'requests_with_thinking_disabled': sum((q.get('chat_template_kwargs') or {}).get('enable_thinking') is False
                                                       for q in reqs), 'requests': len(reqs),
                'latency_s': {'median': statistics.median(lat), 'p90': sorted(lat)[int(.9 * (len(lat) - 1))], 'max': max(lat)} if lat else None,
                'served_models': dict(Counter(r.get('served_model') for r in ledger)),
                'response_format_attempts': dict(modes), 'attempt_errors': dict(errors),
                'valid': sum(r['ok'] for r in recs), 'records': len(recs), 'valid_first_try': first_try,
                'ledger': window}

    reviewer, locked = fl.reviewer_reference(), fl.locked_episodes()
    per_case = []
    for case in fl.cases():
        row = {'case': case, 'reviewer': reviewer[case]['outcome'], 'reviewer_modes': reviewer[case]['modes'],
               'v5_1': 'success' if locked[case]['v5_1_pass'] else 'failure', 'v5_1_fail_reason': locked[case]['fail_reason']}
        for arm in ('blind', 'with_v51'):
            ok = [r for r in lab if r['case'] == case and r['arm'] == arm and r['ok']]
            outs = [r['output']['outcome_guess'] for r in ok]
            ms = [r['output']['failure_mode'] for r in ok]
            row[arm] = {'n_ok': len(ok), 'outcomes': outs, 'modes': ms,
                        'outcome_agree_reviewer': f"{sum(o == reviewer[case]['outcome'] for o in outs)}/{len(ok)}",
                        'mode_agree_reviewer': None if reviewer[case]['modes'] is None else
                        f"{sum(m in reviewer[case]['modes'] for m in ms)}/{len(ok)}",
                        'consistent_outcome': len(set(outs)) == 1, 'consistent_mode': len(set(ms)) == 1,
                        'citations': f"{sum(len(r['grounding']['grounded']) for r in ok)}/{sum(len(r['grounding']['cited']) for r in ok)}",
                        'explanations': [r['output']['one_line_explanation'] for r in ok]}
        per_case.append(row)
    arms = {}
    for arm in ('blind', 'with_v51'):
        ok = [r for r in lab if r['arm'] == arm and r['ok']]
        cited = sum(len(r['grounding']['cited']) for r in ok)
        arms[arm] = {'calls': sum(r['arm'] == arm for r in lab), 'valid': len(ok),
                     'per_call_outcome_agree_reviewer': sum(r['output']['outcome_guess'] == reviewer[r['case']]['outcome'] for r in ok),
                     'per_call_outcome_agree_v5_1': sum(r['output']['outcome_guess'] == ('success' if locked[r['case']]['v5_1_pass'] else 'failure') for r in ok),
                     'per_call_mode_agree_reviewer': sum(r['output']['failure_mode'] in reviewer[r['case']]['modes'] for r in ok if reviewer[r['case']]['modes']),
                     'per_call_mode_scorable': sum(1 for r in ok if reviewer[r['case']]['modes']),
                     'cases_consistent_outcome': sum(c[arm]['consistent_outcome'] for c in per_case),
                     'cases_consistent_mode': sum(c[arm]['consistent_mode'] for c in per_case),
                     'citations_grounded': sum(len(r['grounding']['grounded']) for r in ok), 'citations': cited,
                     'calls_all_citations_grounded': sum(not r['grounding']['ungrounded'] for r in ok),
                     'uncertain': sum(r['output']['outcome_guess'] == 'uncertain' for r in ok),
                     'mode_histogram': dict(Counter(r['output']['failure_mode'] for r in ok))}

    insts = {x['id']: x for x in ic.instructions()[1]}
    intent_rows, faithful = [], {}
    for iid, inst in insts.items():
        ok = [r for r in intent if r['instruction_id'] == iid and r['ok']]
        hits = {f: sum(r['comparison']['vs_faithful'][f] for r in ok) for f in inst['faithful']}
        g = faithful.setdefault(inst['group'], {'instructions': 0, 'fields': 0, 'checks': 0, 'pass': 0})
        g['instructions'] += 1
        g['fields'] += len(inst['faithful'])
        g['checks'] += len(inst['faithful']) * len(ok)
        g['pass'] += sum(hits.values())
        intent_rows.append({'id': iid, 'text': inst['text'], 'group': inst['group'], 'n_ok': len(ok),
                            'fields': {f: Counter(str(ic.flat(r['output']).get(f)) for r in ok) for f in ic.FIELDS},
                            'faithful': {f: f'{h}/{len(ok)}' for f, h in hits.items()},
                            'ambiguity': f"{sum(r['comparison']['flagged_ambiguity'] for r in ok)}/{len(ok)}",
                            'ambiguity_expected': inst['ambiguous'],
                            'ambiguities_r0': ok[0]['output']['ambiguities'] if ok else None,
                            'constraints_r0': ok[0]['output']['constraints'] if ok else None})

    lab_report = run_all.labeller_report(lab) if lab else None
    intent_report = run_all.intent_report(intent) if intent else None
    o = outcomes(lab_report[0], lab) if lab_report else None
    out = {'tf': tf_stats, 'outcome_vs_reviewer': o, 'arms': arms, 'per_case': per_case, 'intent': intent_rows,
           'intent_faithful_by_group': faithful}

    head = headline(o) if o else None
    text = run_all.markdown(run.name, report_json['estimate'], lab_report, intent_report, False,
                            window['before_run_usd'] if window else tf.spent(), head)
    text += '\n'.join(outcome_lines(o) + stats_lines(tf_stats, arms, per_case, intent_rows, faithful)) + '\n'
    notes = run / 'notes.md'
    if notes.exists():
        text += '\n' + notes.read_text()
    return out, text


def _f(x):
    return f'{x:.2f}' if x is not None else '—'


def outcome_lines(o):
    if not o:
        return []
    judges = [*o['majority_of_repeats'], 'v5_1']
    cols = {**{a: o['majority_of_repeats'][a] for a in o['majority_of_repeats']}, 'v5_1': o['v5_1'], 'baseline': o['baseline']}
    names = {**ARM_NAMES, 'baseline': f"Always \"{o['baseline']['always']}\""}
    n = o['n_cases']
    L = ['', '## Outcome agreement with the reviewer (majority of 3 repeats; reference = my independent reviewer)', '',
         'Each judge is scored against the same reviewer verdicts. Balanced accuracy is the mean of the two per-class '
         'recalls, so always answering one class scores 0.50 whatever the class balance.', '',
         '| | ' + ' | '.join(names[k] for k in cols) + ' |', '|---|' + '---|' * len(cols),
         '| Agrees with the reviewer | ' + ' | '.join(f"{c['correct']}/{n}" for c in cols.values()) + ' |',
         '| Reviewer success, judged success | ' + ' | '.join('{}/{}'.format(*c['recall']['success']) for c in cols.values()) + ' |',
         '| Reviewer failure, judged failure | ' + ' | '.join('{}/{}'.format(*c['recall']['failure']) for c in cols.values()) + ' |',
         '| Balanced accuracy | ' + ' | '.join(_f(c['balanced_accuracy']) for c in cols.values()) + ' |',
         '| Agrees, by robot | ' + ' | '.join(
             'A {}/{}, B {}/{}'.format(*o['by_robot'][k]['A'], *o['by_robot'][k]['B']) if k in o['by_robot'] else '—' for k in cols) + ' |',
         '', '| Judge | Reviewer says | Judged success | Judged failure | Uncertain, split or none |', '|---|---|---|---|---|']
    for k in judges:
        c = cols[k]
        for t in CLASSES:
            L.append(f"| {names[k] if t == 'success' else ''} | {t} ({sum(c['matrix'][t].values())}) | "
                     f"{c['matrix'][t]['success']} | {c['matrix'][t]['failure']} | {c['matrix'][t]['other']} |")
    L.append('')
    for arm, c in o['per_call'].items():
        e = o['errors'][arm]
        L.append(f"- **{arm}**, per call (secondary): {c['correct']}/{c['n']}, balanced accuracy {_f(c['balanced_accuracy'])} "
                 f"(success {'{}/{}'.format(*c['recall']['success'])}, failure {'{}/{}'.format(*c['recall']['failure'])}). "
                 f"Majority disagreements: called failure where the reviewer says success: {', '.join(e['false_failure']) or 'none'}; "
                 f"called success where the reviewer says failure: {', '.join(e['false_success']) or 'none'}"
                 + (f"; no class: {', '.join(e['other'])}" if e['other'] else '') + '.')
    v, e = o['v5_1'], o['errors']['v5_1']
    L.append(f"- **Locked v5.1 alone** disagrees with the reviewer on {v['n'] - v['correct']} of {v['n']}: "
             f"{', '.join(e['false_failure'] + e['false_success']) or 'none'}.")
    return L


def stats_lines(tf_stats, arms, per_case, intent_rows, faithful):
    w = tf_stats['ledger']
    L = ['', '## Run statistics', '',
         f"Token Factory: {tf_stats['calls_logged']} HTTP calls, {tf_stats['http_ok']} OK, {tf_stats['transport_errors']} transport errors. "
         f"Estimated usage cost for this run ${tf_stats['estimated_usage_usd']:.6f}, from returned token counts and recorded prices "
         f"({tf_stats['prompt_tokens']:,} prompt + {tf_stats['completion_tokens']:,} completion tokens); the actual billing "
         'deduction was not verified.'
         + (f" Ledger total before this run ${w['before_run_usd']:.6f} ({', '.join(f'{k} ×{v}' for k, v in w['before_run_purposes'].items()) or 'empty'}), "
            f"through this run ${w['through_run_usd']:.6f}." if w else '')
         + f" Reasoning tokens: {tf_stats['calls_reporting_zero_reasoning_tokens']}/{tf_stats['calls_logged']} responses reported 0, "
         f"with thinking disabled in {tf_stats['requests_with_thinking_disabled']}/{tf_stats['requests']} requests. "
         f"Latency median {tf_stats['latency_s']['median']:.2f} s, p90 {tf_stats['latency_s']['p90']:.2f} s, max "
         f"{tf_stats['latency_s']['max']:.2f} s; served models {tf_stats['served_models']}; requested response_format per "
         f"attempt {tf_stats['response_format_attempts']}; attempt errors {tf_stats['attempt_errors'] or 'none'}.",
         f"Validity: {tf_stats['valid']}/{tf_stats['records']} responses passed local schema validation, "
         f"{tf_stats['valid_first_try']} on the first attempt. That checks shape only, not correctness.", '']
    for arm, a in arms.items():
        L.append(f"- **{arm}** (per call, n={a['valid']}): outcome = reviewer {a['per_call_outcome_agree_reviewer']}, = v5.1 "
                 f"{a['per_call_outcome_agree_v5_1']}; mode = reviewer's stated mechanism {a['per_call_mode_agree_reviewer']}/"
                 f"{a['per_call_mode_scorable']}; same outcome in all repeats {a['cases_consistent_outcome']}/{len(per_case)} cases, same mode "
                 f"{a['cases_consistent_mode']}/{len(per_case)}; cited controls present in the summary {a['citations_grounded']}/{a['citations']} "
                 f"({a['calls_all_citations_grounded']}/{a['valid']} calls fully grounded; presence is not semantic support); "
                 f"uncertain {a['uncertain']}; modes {a['mode_histogram']}")
    L += ['', '| Case | Reviewer | v5.1 | Blind outcomes | Blind modes | With v5.1 outcomes | With v5.1 modes |', '|---|---|---|---|---|---|---|']
    for c in per_case:
        L.append(f"| {c['case'][6:16]} | {c['reviewer']}{' (' + '/'.join(c['reviewer_modes']) + ')' if c['reviewer_modes'] else ''} | "
                 f"{c['v5_1']} | {', '.join(c['blind']['outcomes'])} | {', '.join(c['blind']['modes'])} | "
                 f"{', '.join(c['with_v51']['outcomes'])} | {', '.join(c['with_v51']['modes'])} |")
    L += ['', '| Instruction | Group | object | target | relation | settle | release | rim | Faithful | Ambiguity flagged (expected) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in intent_rows:
        L.append(f"| {r['text']} | {r['group']} | " + ' | '.join(
            ', '.join(f'{k}×{v}' for k, v in r['fields'][f].items()) for f in ic.FIELDS)
            + f" | {r['faithful']} | {r['ambiguity']} ({'yes' if r['ambiguity_expected'] else 'no'}) |")
    L += ['', 'Faithful-reading field checks by group (fields × valid repeats): ' + '; '.join(
        f"{g}: {v['pass']}/{v['checks']} ({v['fields']} fields across {v['instructions']} instructions)" for g, v in faithful.items()) + '.']
    return L


def main(run):
    run = Path(run).resolve()
    out, text = render(run)
    write_json(run / 'analysis.json', out)
    write_text(run / 'report.md', text)
    print(json.dumps({'tf': out['tf'], 'outcome_vs_reviewer': out['outcome_vs_reviewer']}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
