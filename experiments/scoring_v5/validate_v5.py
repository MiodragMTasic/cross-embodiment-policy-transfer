"""Offline check of v5 against every saved trace that has an independent human/reviewer verdict. CPU only.

First proves the replay is faithful (v4 replayed here == v4 as recorded), then scores v5. Writes validation-v5.json.
"""
import hashlib
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'bowl_step5')]
from release_gate_v3 import ReleaseGateV3
from release_gate_v4 import ReleaseGateV4
from release_gate_v5 import ReleaseGateV5, score

E = HERE.parents[1] / 'assets/report/evidence'
A3, A4 = '2026-10-21/bowl-controlled-v3', '2026-10-21/bowl-training-feasibility-v4'
LIVE, DEV = '2026-10-23/bowl-step6/paint_live_v1/runs', '2026-10-23/bowl-step6/paint_dev_v1/runs'
# (case, trace, human verdict, source of the verdict). Pass = deliberate release that ended on/in the bowl (rim counts).
CASES = [
    ('A base s0 v3', f'{A3}/step5_controlled_base_s0_v3/BananaInBowlTask/strict-metric.json', False, 'BOWL_BASELINE_AND_DATA_RESULT.md: closed-jaw slip'),
    ('A base s1 v3', f'{A3}/step5_controlled_base_s1_v3/BananaInBowlTask/strict-metric.json', True, 'BOWL_BASELINE_AND_DATA_RESULT.md + base1-reviewed-manifest.json: supported rim placement'),
    *[(f'A layout {n}{" retry1" if n == 1000 else ""}', f'{A3}/step5_controlled_layout{n}_v3{"_retry1" if n == 1000 else ""}/BananaInBowlUniformInitPose10cmTask/strict-metric.json', v,
       'pilot-final-adjudication.json') for n, v in ((1000, True), (1001, False), (1002, False), (1003, False), (1004, False), (1005, True), (1006, False), (1007, False))],
    *[(f'A layout {n}', f'{A4}/step5_training_layout{n}_v4/BananaInBowlUniformInitPose10cmTask/strict-metric.json', v,
       'BOWL_BASELINE_AND_DATA_RESULT.md, training-feasibility table') for n, v in ((2000, False), (2001, True), (2002, True), (2003, False))],
    *[(f'B live {c}', f'{LIVE}/step6_B_paint_live_{c}_v1/gate_trace.jsonl', False, f'review/step6_live/adjudication/{e}.md')
      for c, e in (('l2002_s0_painted', 'ep1_l2002_s0_painted'), ('l2002_s0_unpainted', 'ep2_l2002_s0_unpainted'), ('base_s0_painted', 'ep3_base_s0_painted'),
                   ('base_s0_unpainted', 'ep4_base_s0_unpainted'), ('base_s1_painted', 'ep5_base_s1_painted'), ('base_s1_unpainted', 'ep6_base_s1_unpainted'))],
    ('B dev lift l2002 s0', f'{DEV}/step6_B_paint_dev_lift_l2002_s0_v1/gate_trace.jsonl', True, 'review/step6_dev/adjudication/lift_l2002_s0.md'),
]
# Saved trace, no explicit physical verdict: scored for information only.
INFO = [('A base s0 v2', '2026-10-21/bowl-controlled-v2/step5_controlled_base_s0_v2/BananaInBowlTask/strict-metric.json')]


def recorded_v4(path):
    """v4 as recorded: in the trace for v4 runs, in the reviewed retrospective replay for the v2/v3 A captures."""
    if path.name == 'gate_trace.jsonl':
        c = json.loads(path.with_name('checks.json').read_text())
        return c['success_step'], c['release_events']
    d = json.loads(path.read_text())
    if d['schema'].endswith('v4'):
        first = next((r['step'] for r in d['trace'] if r['stable_success']), None)
        return first, d['release_events']
    replay = json.loads((E / A3 / 'v4-development-replays.json').read_text())
    hit = next((c for c in replay['cases'] if Path(c['source_metric']).parts[-3] == path.parts[-3]), None)
    if hit is None:  # v3 runs outside the reviewed v4 replay: check the feed against their recorded v3 instead
        return 'v3', (round(d['stable_success_s'] * 15) if d['stable_success_s'] else None, d['release_events'])
    t = hit['retrospective_v4_first_pass_s']
    return (round(t * 15) if t else None), hit['events']


out = []
for case, rel, human, source in CASES + [(c, r, None, 'none') for c, r in INFO]:
    path = E / rel
    v4 = score(path, ReleaseGateV4)
    rec_step, rec_events = recorded_v4(path)
    check = v4
    if rec_step == 'v3':
        check = score(path, ReleaseGateV3); rec_step, rec_events = rec_events
    upto = [e for e in rec_events if check['success_control'] is None or e['time_s'] <= check['events'][-1]['time_s'] + 1e-6]  # compare up to first pass
    faithful = check['success_control'] == rec_step and len(upto) == len(check['events']) and all(
        abs(a['time_s'] - b['time_s']) < 1e-6 and {**a, 'time_s': 0} == {**b, 'time_s': 0} for a, b in zip(upto, check['events']))
    v5 = score(path)
    q = v5['qualifying_open']
    out.append({'case': case, 'human_pass': human, 'verdict_source': source, 'v4_pass': v4['pass'],
                'v5_pass_first_rest': v5['pass_first_rest'], 'v5_pass_at_end': v5['pass_at_end'],
                'agree_first_rest': None if human is None else v5['pass_first_rest'] == human,
                'agree_at_end': None if human is None else v5['pass_at_end'] == human,
                'replay_matches_record': faithful, 'replay_checked_against': 'v3 record' if check is not v4 else 'v4 record',
                'v5_success_control': v5['success_control'], 'rows': v5['rows'], 'qualifying_open': q, 'v5_fail_reason': v5['fail_reason'],
                'end_in_or_on_bowl': v5['end_in_or_on_bowl'], 'end_fingertip_contact': v5['end_fingertip_contact'],
                'v5_events': v5['events'], 'v5_events_after_first_rest': v5['after_first_rest'],
                'trace': rel, 'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    print(f"{case:25} human={human!s:5} v4={v4['pass']!s:5} v5rest={v5['pass_first_rest']!s:5} v5end={v5['pass_at_end']!s:5} "
          f"replay={faithful!s:5} end_bowl={v5['end_in_or_on_bowl']!s:5} rows={v5['rows']:4} "
          f"{'open@%d win@%d dxy=%.3f h=%.3f after=%d' % (q['control'], v5['success_control'], q['dxy_root_m'], q['centre_above_bowl_top_m'], len(v5['after_first_rest'])) if q else v5['fail_reason']}")
code = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ('release_gate_v5.py', 'validate_v5.py')}
code.update({n: hashlib.sha256((HERE.parent / 'bowl_step5' / n).read_bytes()).hexdigest()
             for n in ('release_gate_v3.py', 'release_gate_v4.py', 'rim_support.py', 'stability.py')})
(HERE / 'validation-v5.json').write_text(json.dumps({'classification': 'offline development validation on past adjudicated traces; no new trials',
                                                     'code_sha256': code, 'cases': out}, indent=1))
