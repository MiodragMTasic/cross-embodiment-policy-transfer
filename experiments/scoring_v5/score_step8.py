"""Score step 8 runs offline with the locked v5.1 gate (release_gate_v5_1.py) and stage metrics revision 2 (stages_r2.py).

python score_step8.py --phase {before,after,mock} --out DIR ROOT [ROOT ...]
Each ROOT is searched recursively for directories named as step 8 cases (experiments/bowl_step8/STEP8_RUNNERS.md):
  A  step8_A_{P0,P1,P0canary}_l{L}_s0_v1/<task>/strict-metric.json
  B  step8_B_{P0,P1,P0canary}_l{L}_s0_v1/{gate_trace.jsonl, tracking.npz, checks.json}
Smoke cases are never scored. Canary cases are scored but kept out of the paired comparison.
--phase mock reads the sealed step 6 live B episodes (painted vs unpainted) and the step 5 A traces instead.

Scored = a complete run: a positive exit record (episodes.tsv status 0 and failure no, else every exit-code.txt 0), no failure.txt,
a trace, and for B checks.json (complete true) and tracking.npz; OR a B safety abort (reviewer B2): abort-classification.json
(step8-abort-classification-v1) with class safety_abort, matching failure.txt, an agreeing episodes.tsv row (class and exit status),
and the same class and kind when the locked classify_abort.py re-derives them from failure.txt, that exit status and the gate rows
on disk (else marker_disagrees_with_classifier). A safety abort is a task outcome:
v5.1 and stages_r2 score the partial trace (a first rest before the abort is a win; otherwise fail_reason safety_abort:<kind>),
live v4 is "not reached", and it is paired; its class and kind sit next to the verdict. A :298 abort's gate_trace.jsonl is one row
longer than tracking.npz and that aborted row is dropped (P1); every other kind must have equal lengths. Under 3 scored rows it is a fail with no stage, labelled empty_trace. A crash, or
failure.txt without the marker, is unscored (R4). Pass roots holding both phases' runs for --phase after (P1 pairs with before's P0).
Per episode: v5.1 verdict (first rest primary, at end alongside), win control, M2, furthest stage and its controls, review flags
and the LIVE v4 result (B checks.json, A strict-metric stable_success_s) - never a replay of a truncated trace (N4).
The headline outcome is score()['pass'], never stages()['win']; a disagreement is a review flag, not an assert (S5).
Times are control/15 on both robots (S4): B's recorded simulation_s is (control+1)/15 because of its reset step, so it is not used.
M2 is descriptive only, among winners, with no test (S6).
Per condition: win k/n with a Wilson 95% interval and stage distributions. Per robot, paired: discordant layouts and the exact
two-sided McNemar p, over layouts scored in both conditions only (R4); the rest are listed, never imputed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os
import re
import sys
import tempfile
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'bowl_step5'), str(HERE.parent / 'bowl_step8')]
from release_gate_v5_1 import load, review_flags, score
import stages_r2
import classify_abort  # locked, bowl_step8/classify_abort.py

SIM_HZ = 15  # control/15 s on both robots (S4)
ABORT_SCHEMA = 'step8-abort-classification-v1'
A_LIMIT_CAVEAT = ("A has no target-limit check, so the same out-of-limit target ends a B episode but not an A episode. "
                  "Within-robot ΔB is unaffected.")
CONTACT_CAVEAT = ("The contact flag is not shared: A's 'gripper' sensor is one finger, B's flag is either of two tip pads. "
                  "Within-robot comparisons are unaffected; A-vs-B stage rates carry a B-lenient bias.")
STEP8 = [re.compile(r'^step8_(?P<robot>[AB])_(?P<cond>P0|P1|P0canary)_(?P<layout>l40(?:0\d|1[01]))_s0_v1$')]
PHASES = {
    'before': {'names': STEP8, 'conditions': ('P0',), 'pairs': ()},
    'after': {'names': STEP8, 'conditions': ('P0', 'P1', 'P0canary'), 'pairs': (('P1', 'P0'),)},
    'mock': {'names': [re.compile(r'^step6_(?P<robot>B)_paint_live_(?P<layout>l2002_s0|base_s[01])_(?P<cond>painted|unpainted)_v1$'),
                       re.compile(r'^step5_(?:controlled|training)_(?P<layout>(?:base_s[01]|layout\d{4})_v[234](?:_retry1)?)$')],
             'conditions': ('painted', 'unpainted', 'P0'), 'pairs': (('painted', 'unpainted'),)},
}


def discover(roots, phase):
    found, status, problems = {}, {}, []
    for root in roots:
        for tsv in sorted(Path(root).rglob('episodes.tsv')):
            head, *rows = [l.split('\t') for l in tsv.read_text().splitlines()]
            for r in rows:
                r = dict(zip(head, r))
                if r['case'] in status:
                    problems.append(f"{r['case']}: more than one episodes.tsv row")
                # step 8: status/failure; step 6: bridge_status/failure_txt
                status[r['case']] = {'tsv': str(tsv), 'exit': r.get('status', r.get('bridge_status')),
                                     'failure': r.get('failure', r.get('failure_txt')), 'v4': r.get('v4_success', r.get('gate_success'))}
        for d in sorted(Path(root).rglob('step*_*')):
            m = next((m for p in PHASES[phase]['names'] if d.is_dir() and (m := p.match(d.name))), None)
            if not m:
                continue
            robot, cond = m.groupdict().get('robot', 'A'), m.groupdict().get('cond', 'P0')
            if cond not in PHASES[phase]['conditions']:
                problems.append(f'{d.name}: condition {cond} not scored in phase {phase}')
                continue
            # A: the supervisor's runs/<case> and RoboLab/output/<case>/<task> share the name; B writes one runs/<case>
            entry = found.setdefault((robot, cond, m['layout']), {'case': d.name, 'dirs': []})
            entry['dirs'].append(d)
            trace = d / 'gate_trace.jsonl' if robot == 'B' else next(iter(sorted(d.glob('*/strict-metric.json'))), None)
            if trace is None or not trace.exists():
                continue
            if entry.get('trace') and entry['trace'].read_bytes() != trace.read_bytes():
                problems.append(f'{d.name}: two different traces ({entry["trace"]}, {trace})')
            entry.setdefault('trace', trace)
    for entry in found.values():
        entry['status'] = status.get(entry['case'])
    return found, problems


def live_v4(robot, trace):
    if robot == 'B':
        c = json.loads(trace.with_name('checks.json').read_text())
        return {'gate': 'v4 (B port), live', 'pass': bool(c['automatic_v4B_success']), 'success_control': c['success_step'],
                'complete': bool(c['complete'])}
    d = json.loads(trace.read_text())
    return {'gate': f"{d['schema']}, live", 'pass': d['stable_success_s'] is not None, 'success_s': d['stable_success_s'],
            'complete': bool(d.get('trace'))}


def abort_marker(robot, entry):
    """B only: (reasons it cannot be scored, the safety_abort marker or None, the marker or None).
    The class and kind are re-derived with the locked classifier (bowl_step8/classify_abort.py) from failure.txt, the exit status
    and the gate rows on disk; a scored safety abort also needs an agreeing episodes.tsv row."""
    found = [d / 'abort-classification.json' for d in entry['dirs'] if (d / 'abort-classification.json').exists()] if robot == 'B' else []
    if not found:
        return [], None, None
    m, why, st = json.loads(found[0].read_text()), [], entry['status']
    failure = [d / 'failure.txt' for d in entry['dirs'] if (d / 'failure.txt').exists()]
    if m.get('schema') != ABORT_SCHEMA or m.get('case') != entry['case'] or m.get('class') not in ('safety_abort', 'crash'):
        why.append(f'{found[0]}: not a {ABORT_SCHEMA} record for this case')
    if st is None:
        why.append('no_episode_row')
    else:
        if st['failure'] != m.get('class'):
            why.append(f"episodes.tsv failure {st['failure']} but abort-classification.json class {m.get('class')}")
        if st['exit'] != str(m.get('exit_status')):
            why.append(f"exit status: abort-classification.json {m.get('exit_status')}, episodes.tsv {st['exit']}")
    if not failure:
        return why + ['abort-classification.json without failure.txt'], None, m
    text = failure[0].read_text()
    if hashlib.sha256(text.encode()).hexdigest() != m.get('failure_txt_sha256'):  # hashed as the classifier does
        why.append('failure.txt does not match abort-classification.json')
    gate = found[0].with_name('gate_trace.jsonl')
    rows = sum(1 for l in gate.read_text().splitlines() if l.strip()) if gate.exists() else 0
    kind = classify_abort.classify(*classify_abort.parse(text), m.get('exit_status'), rows)
    m['classifier'] = {'class': 'safety_abort' if kind else 'crash', 'kind': kind, 'gate_rows_on_disk': rows}
    if (m['classifier']['class'], kind) != (m.get('class'), m.get('kind')):
        why.insert(0, f"marker_disagrees_with_classifier: marker {m.get('class')}/{m.get('kind')}, classifier {m['classifier']['class']}/{kind}")
    elif kind is None:
        why.insert(0, f"crash: exit {m.get('exit_status')}, {m.get('exception') or 'no traceback'}")  # R4: infrastructure, never scored
    return why, None if why else m, m


def unscorable(robot, entry):
    """Reasons a non-abort episode cannot be scored; scoring needs a positive exit record (R4: listed, never imputed)."""
    why, st, trace = [], entry['status'], entry.get('trace')
    if st:
        if st['exit'] != '0' or st['failure'] != 'no':
            why.append(f"episodes.tsv: exit {st['exit']}, failure {st['failure']}")
    else:
        codes = [f.read_text().strip() for d in entry['dirs'] for f in (d / 'exit-code.txt',) if f.exists()]
        if not codes or any(c != '0' for c in codes):
            why.append(f'no episodes.tsv row; exit-code.txt {codes or "missing"}')
    why += [f'{f} present without abort-classification.json' for d in entry['dirs'] for f in (d / 'failure.txt',) if f.exists()]
    if trace is None:
        return why + ['no trace found']
    if robot == 'B':
        why += [f'{n} missing' for n in ('checks.json', 'tracking.npz') if not trace.with_name(n).exists()]
    return why


def scored(rec, trace, v4, abort=None, src=None):
    src = src or trace
    s, st = score(trace), stages_r2.stages(trace)
    _, rows, _, _ = load(trace)
    flags = review_flags(s, bool(v4['pass'])) + (['fail, upstream predicate true at end'] if not s['pass'] and rows[-1]['upstream_predicate'] else [])
    if st['win'] != s['pass'] or st.get('win_control') != s['success_control']:
        flags.append(f"stages win {st['win']} at {st.get('win_control')} disagrees with v5.1 pass {s['pass']} at {s['success_control']}")
    best = st['best_attempt']
    m2 = {k: st[k] for k in ('qualifying_open_control', 'time_to_win_P0_s', 'win_control') if k in st}
    if m2:
        m2.update(qualifying_open_s=round(m2['qualifying_open_control'] / SIM_HZ, 3), win_s=round(m2['win_control'] / SIM_HZ, 3))
    return {**rec, 'scored': True, 'complete': abort is None, 'trace': str(src), 'trace_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
            'live_v4': v4, 'v5_1_pass': s['pass'], 'v5_1_pass_at_end': s['pass_at_end'], 'win_control': s['success_control'],
            'fail_reason': s['fail_reason'] if abort is None or s['pass'] else f"safety_abort:{abort['kind']}", 'gate_fail_reason': s['fail_reason'],
            'qualifying_open': s['qualifying_open'], 'M2': m2,
            'furthest': st['furthest'], 'stage_controls': {k: best[k] for k in stages_r2.STAGES if k in best},
            'attempts': st['attempts'], 'stages': {k: st[k] for k in stages_r2.STAGES}, 'review_flags': flags}


def aborted(rec, entry, m):
    """B2: a safety abort is a task outcome, scored from the partial trace files on disk that the runner's finally block saved (P2).
    P1: a :298 soft-limit abort appends the gate row (:297) but not the tracking row, so gate_trace.jsonl is one row longer than
    tracking.npz; that row (the aborted control) is dropped. :237, :273 and :282 aborts leave equal lengths. At step 0 a :298 abort
    leaves one gate row and no tracking.npz. Under 3 scored rows the episode is a fail with no stage (grasp needs 3, a win 16)."""
    d = next(d for d in entry['dirs'] if (d / 'abort-classification.json').exists())
    gate, track = d / 'gate_trace.jsonl', d / 'tracking.npz'
    lines = [l for l in gate.read_text().splitlines() if l.strip()] if gate.exists() else []
    steps = np.load(track)['step'].tolist() if track.exists() else []
    info = {k: m[k] for k in ('kind', 'exit_status', 'exception', 'message')}
    info.update(gate_trace_rows=len(lines), tracking_rows=len(steps), last_gate_trace_step=json.loads(lines[-1])['step'] if lines else None,
                dropped_gate_rows=len(lines) - len(steps), scored_rows=len(steps), marker_partial_trace=m['partial_trace'])
    rec = {**rec, 'abort_class': m['class'], 'abort_kind': m['kind'], 'abort_classifier': m['classifier'], 'safety_abort': info}
    surplus = 1 if m['kind'] == 'B_joint_beyond_soft_limit' else 0  # only a :298 abort leaves the gate row without its tracking row
    if len(lines) - len(steps) != surplus or [json.loads(l)['step'] for l in lines[:len(steps)]] != steps:
        return {**rec, 'scored': False, 'complete': False,
                'unscored_reason': [f"trace shape: {len(lines)} gate rows, {len(steps)} tracking rows; {m['kind']} allows a surplus of {surplus}"]}
    info['empty_trace'] = len(steps) < 3
    v4 = {'gate': 'v4 (B port), live', 'pass': None, 'reached': False, 'note': 'not reached: safety abort, no complete checks.json'}
    flags = ['safety_abort'] + ([] if m['partial_trace']['gate_trace_rows'] == len(lines) else
                                [f"marker says {m['partial_trace']['gate_trace_rows']} gate rows, disk has {len(lines)}"])
    flags += ['partial trace records a live v4 pass'] if any(json.loads(l).get('stable_success') for l in lines) else []
    if info['empty_trace']:
        return {**rec, 'scored': True, 'complete': False, 'trace': str(gate) if lines else None, 'live_v4': v4, 'v5_1_pass': False,
                'v5_1_pass_at_end': False, 'win_control': None, 'fail_reason': f"safety_abort:{m['kind']}", 'gate_fail_reason': None,
                'qualifying_open': None, 'M2': {}, 'furthest': None, 'stage_controls': {}, 'attempts': 0,
                'stages': {k: False for k in stages_r2.STAGES}, 'review_flags': flags}
    with tempfile.TemporaryDirectory() as t:
        trace = gate
        if info['dropped_gate_rows']:
            trace = Path(t) / 'gate_trace.jsonl'
            trace.write_text(''.join(l + '\n' for l in lines[:len(steps)]))
            os.symlink(track, Path(t) / 'tracking.npz')
        out = scored(rec, trace, v4, m, src=gate)
    out['review_flags'] += flags
    return out


def episode(key, entry):
    robot, cond, layout = key
    rec = {'case': entry['case'], 'robot': robot, 'condition': cond, 'layout': layout, 'dirs': [str(d) for d in entry['dirs']],
           'episodes_tsv': entry['status']}
    why, abort, marker = abort_marker(robot, entry)
    if abort:
        return aborted(rec, entry, abort)
    if marker:
        rec.update(abort_class=marker.get('class'), abort_kind=marker.get('kind'), abort_classifier=marker.get('classifier'))
    why = why if marker else unscorable(robot, entry)
    v4 = None if why else live_v4(robot, entry['trace'])
    if why or not v4['complete']:
        return {**rec, 'scored': False, 'complete': False, 'unscored_reason': why or ['trace marked incomplete'], 'live_v4': v4}
    return scored(rec, entry['trace'], v4)


def m2_among_winners(xs):
    ts = sorted(e['M2']['time_to_win_P0_s'] for e in xs if e['v5_1_pass'])
    return {'n': len(ts), 'time_to_win_P0_s': ts, 'median': float(np.median(ts)) if ts else None}


def aggregate(episodes, phase):
    done = [e for e in episodes if e['scored']]
    per = {}
    for robot in 'AB':
        for cond in PHASES[phase]['conditions']:
            xs = [e for e in done if e['robot'] == robot and e['condition'] == cond]
            lost = [e['case'] for e in episodes if e['robot'] == robot and e['condition'] == cond and not e['scored']]
            if not xs and not lost:
                continue
            k = sum(e['v5_1_pass'] for e in xs)
            aborts = [e for e in xs if 'safety_abort' in e]
            per[f'{robot}/{cond}'] = {'n': len(xs), 'unscored': lost, 'wins': k, 'wilson95': stages_r2.wilson(k, len(xs)),
                                      'wins_at_end': sum(e['v5_1_pass_at_end'] for e in xs),
                                      'safety_aborts': {'n': len(aborts), 'wins': sum(e['v5_1_pass'] for e in aborts),
                                                        'by_kind': {kind: sum(e['safety_abort']['kind'] == kind for e in aborts)
                                                                    for kind in sorted({e['safety_abort']['kind'] for e in aborts})},
                                                        'empty_trace': sum(e['safety_abort']['empty_trace'] for e in aborts),
                                                        'aborted_row_dropped': sum(e['safety_abort']['dropped_gate_rows'] for e in aborts)},
                                      'crashes': sum(e.get('abort_class') == 'crash' for e in episodes
                                                     if e['robot'] == robot and e['condition'] == cond and not e['scored']),
                                      'abort_markers_unscored_other': sum('abort_class' in e and e['abort_class'] != 'crash' for e in episodes
                                                                          if e['robot'] == robot and e['condition'] == cond and not e['scored']),
                                      'live_v4_passes': sum(bool(e['live_v4']['pass']) for e in xs),
                                      'live_v4_not_reached': len(aborts),
                                      'M2_among_winners_descriptive': m2_among_winners(xs),
                                      'furthest': {st: sum(e['furthest'] == st for e in xs) for st in (None, *stages_r2.STAGES)},
                                      'stage_rates': stages_r2.rates([e['stages'] for e in xs])}
    paired = {}
    for robot in 'AB':
        for first, second in PHASES[phase]['pairs']:
            a = {e['layout']: e for e in done if e['robot'] == robot and e['condition'] == first}
            b = {e['layout']: e for e in done if e['robot'] == robot and e['condition'] == second}
            seen = {e['layout'] for e in episodes if e['robot'] == robot and e['condition'] in (first, second)}
            if not seen:
                continue
            both = a.keys() & b.keys()
            win = stages_r2.paired({l: {'win': a[l]['v5_1_pass']} for l in both}, {l: {'win': b[l]['v5_1_pass']} for l in both}, 'win')
            paired[f'{robot}: {first} vs {second}'] = {
                'scored_layouts': sorted(both), 'unscored_layouts': sorted(seen - both), 'win': win,
                'stages': [stages_r2.paired({l: a[l]['stages'] for l in both}, {l: b[l]['stages'] for l in both}, st) for st in stages_r2.STAGES]}
    return per, paired


def markdown(episodes, per, paired, phase, stamp):
    out = [f'# Step 8 offline score, phase `{phase}` ({stamp})', '',
           '| Case | Live gate | Safety abort | v5.1 first rest | v5.1 at end | Win control (s = control/15) | M2: P0 time to open | Furthest | '
           'Stage controls | Review flags |', '|---|---|---|---|---|---|---|---|---|---|']
    for e in sorted(episodes, key=lambda e: (e['robot'], e['condition'], e['layout'])):
        if not e['scored']:
            out.append(f"| {e['case']} | unscored: {'; '.join(e['unscored_reason'])} | | | | | | | | |")
            continue
        a = e.get('safety_abort')
        live = 'not reached' if a else ('pass' if e['live_v4']['pass'] else 'fail') + ('' if 'v4' in e['live_v4']['gate'] else f" ({e['live_v4']['gate'].split(',')[0]})")
        abort = '—' if not a else f"{a['kind']} at {a['last_gate_trace_step']}" + (f" (empty trace, {a['gate_trace_rows']} gate row{'' if a['gate_trace_rows'] == 1 else 's'})" if a['empty_trace'] else '')
        out.append(f"| {e['case']} | {live} | {abort} | {'pass' if e['v5_1_pass'] else 'fail'} | "
                   f"{'pass' if e['v5_1_pass_at_end'] else 'fail'} | {f"{e['win_control']} ({e['M2']['win_s']} s)" if e['win_control'] else '—'} | "
                   f"{str(e['M2'].get('time_to_win_P0_s', '—')) + (' s' if 'time_to_win_P0_s' in e['M2'] else '')} | {e['furthest'] or 'none'} | "
                   f"{', '.join(f'{k} {v}' for k, v in e['stage_controls'].items()) or '—'} | {'; '.join(e['review_flags']) or '—'} |")
    out += ['', '| Condition | v5.1 wins | Wilson 95% | Live gate passes | Furthest stage (none/grasp/lift/over/open/landed/win) | '
            'M2 median among winners (descriptive) | Unscored |', '|---|---|---|---|---|---|---|']
    for c, x in per.items():
        m2 = x['M2_among_winners_descriptive']
        out.append(f"| {c} | {x['wins']}/{x['n']} | {x['wilson95'][0]:.2f}–{x['wilson95'][1]:.2f} | {x['live_v4_passes']}/{x['n']} | "
                   f"{'/'.join(str(v) for v in x['furthest'].values())} | {f"{m2['median']:.3f} s (n = {m2['n']})" if m2['n'] else '—'} | "
                   f"{', '.join(x['unscored']) or '—'} |")
    if len({c[0] for c in per}) > 1:
        out += ['', f'Caveat: {CONTACT_CAVEAT}']
    arms = {c: x for c, x in per.items() if c.startswith('B/')}
    if arms:
        out += ['', '| B arm | Safety aborts | By kind | v5.1 wins among them | Empty trace | Aborted control row dropped (:298) | Crashes (unscored) | '
                'Other unscored abort markers |', '|---|---|---|---|---|---|---|---|']
        for c, x in arms.items():
            ab = x['safety_aborts']
            out.append(f"| {c} | {ab['n']} | {', '.join(f'{k} {v}' for k, v in ab['by_kind'].items()) or '—'} | {ab['wins']} | "
                       f"{ab['empty_trace']} | {ab['aborted_row_dropped']} | {x['crashes']} | {x['abort_markers_unscored_other']} |")
        out += ['', f'Note: {A_LIMIT_CAVEAT}']
    if paired:
        out += ['', '| Paired | Scored layouts | Win discordant (first only / second only) | Exact McNemar p | Unscored layouts |', '|---|---|---|---|---|']
        for c, x in paired.items():
            out.append(f"| {c} | {len(x['scored_layouts'])} | {x['win']['only_first']} / {x['win']['only_second']} | "
                       f"{x['win']['p_exact_mcnemar']:.4g} | {', '.join(x['unscored_layouts']) or '—'} |")
        out += ['', 'Safety aborts are task outcomes (B2): scored on the partial trace and paired. Crashes are unscored (R4).']
        out += ['', 'p < 0.05 needs at least 6 discordant layouts, all in one direction.']
    return '\n'.join(out) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=sorted(PHASES))
    ap.add_argument('--out', required=True)
    ap.add_argument('--stamp', default='', help='UTC timestamp to print in the report (from date -u)')
    ap.add_argument('roots', nargs='+')
    a = ap.parse_args(argv)
    found, problems = discover(a.roots, a.phase)
    episodes = [episode(k, v) for k, v in sorted(found.items())]
    per, paired = aggregate(episodes, a.phase)
    code = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ('release_gate_v5_1.py', 'stages_r2.py', 'stages.py', 'score_step8.py')}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {'phase': a.phase, 'stamp_utc': a.stamp, 'roots': a.roots, 'code_sha256': code, 'problems': problems,
              'notes': {'contact_flag': CONTACT_CAVEAT, 'target_limits': A_LIMIT_CAVEAT,
                        'safety_abort': 'scored from the files on disk; a :298 abort leaves one more gate row than tracking rows, and that '
                                        'row (the aborted control) is dropped; under 3 scored rows it is a fail with no stage (empty_trace)',
                        'times': 'control/15 s on both robots; B simulation_s = (control+1)/15 is not used',
                        'M2': 'descriptive only, among winners, no test', 'headline': "score()['pass']; stages win only cross-checked"},
              'episodes': episodes, 'per_condition': per, 'paired': paired}
    (out / f'score_step8-{a.phase}.json').write_text(json.dumps(result, indent=1, default=str))
    (out / f'score_step8-{a.phase}.md').write_text(markdown(episodes, per, paired, a.phase, a.stamp))
    return result


if __name__ == '__main__':
    r = main()
    print(json.dumps({'episodes': len(r['episodes']), 'scored': sum(e['scored'] for e in r['episodes']), 'problems': r['problems'],
                      'per_condition': {k: f"{v['wins']}/{v['n']} {v['wilson95']}" for k, v in r['per_condition'].items()},
                      'paired': {k: v['win'] for k, v in r['paired'].items()}}, indent=1))
