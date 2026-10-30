"""Offline check of v5.1 and the step 8 stage metrics on the 22 traces of validation-v5.json. CPU only.

Writes validation-v5_1.json: per trace v5 vs v5.1 (V1/V3 effects), review flags (V2) and stages.
"""
import hashlib
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'bowl_step5')]
from release_gate_v5_1 import score, review_flags
from stages import stages, rates, paired, STAGES

E = HERE.parents[1] / 'assets/report/evidence'
v5 = json.loads((HERE / 'validation-v5.json').read_text())['cases']
out, live = [], {}
for c in v5:
    path = E / c['trace']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == c['trace_sha256'], c['case']
    s = score(path)
    st = stages(path)
    before = [(round(e['time_s'], 6), e['event'], e.get('reason')) for e in c['v5_events']]
    after = [(round(e['time_s'], 6), e['event'], e.get('reason')) for e in s['events']]
    row = {'case': c['case'], 'human_pass': c['human_pass'], 'v4_pass': c['v4_pass'], 'v5_pass': c['v5_pass_first_rest'],
           'v5_1_pass': s['pass'], 'v5_1_pass_at_end': s['pass_at_end'], 'agree': None if c['human_pass'] is None else s['pass'] == c['human_pass'],
           'win_control_v5': c['v5_success_control'], 'win_control_v5_1': s['success_control'],
           'events_changed_before_first_rest': before != after, 'events_v5': c['v5_events'], 'events_v5_1': s['events'],
           'after_first_rest_v5': c['v5_events_after_first_rest'], 'after_first_rest_v5_1': s['after_first_rest'],
           'end_on_bowl_v5_1': s['end_on_bowl'], 'fail_reason': s['fail_reason'], 'qualifying_open': s['qualifying_open'],
           'review_flags': review_flags(s, c['v4_pass']), 'stages': st, 'trace': c['trace'], 'trace_sha256': c['trace_sha256']}
    out.append(row)
    if c['case'].startswith('B live'):
        layout, arm = c['case'][7:].rsplit('_', 1)
        live.setdefault(arm, {})[layout] = st
    flags = ', '.join(row['review_flags']) or '-'
    m2 = f"open {st['qualifying_open_control']} P0 {st['time_to_win_P0_s']} s, win {st['win_control']} ({st['win_simulation_s']} s)" if s['pass'] else ''
    print(f"{c['case']:25} human={c['human_pass']!s:5} v5={c['v5_pass_first_rest']!s:5} v5.1={s['pass']!s:5} "
          f"win {c['v5_success_control']}->{s['success_control']} ev_changed={before != after!s:5} | furthest={st['furthest']!s:9} "
          f"attempts={st['attempts']:3} {m2} | review: {flags}")
demo = {'note': 'illustration of M5 only: 3 B live layouts, painted vs unpainted, from the sealed live comparison',
        'rates_painted': rates(list(live['painted'].values())), 'rates_unpainted': rates(list(live['unpainted'].values())),
        'paired_painted_vs_unpainted': [paired(live['painted'], live['unpainted'], st) for st in STAGES]}
code = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ('release_gate_v5.py', 'release_gate_v5_1.py', 'stages.py', 'validate_v5_1.py')}
code.update({n: hashlib.sha256((HERE.parent / 'bowl_step5' / n).read_bytes()).hexdigest()
             for n in ('release_gate_v3.py', 'release_gate_v4.py', 'rim_support.py', 'stability.py')})
(HERE / 'validation-v5_1.json').write_text(json.dumps({'classification': 'offline development validation on past adjudicated traces; no new trials',
                                                       'code_sha256': code, 'cases': out, 'm5_demo': demo}, indent=1))
for st in STAGES:
    p = next(x for x in demo['paired_painted_vs_unpainted'] if x['stage'] == st)
    print(f"M5 demo {st:9} painted {demo['rates_painted'][st]['k']}/3 {demo['rates_painted'][st]['wilson95']}  "
          f"unpainted {demo['rates_unpainted'][st]['k']}/3 {demo['rates_unpainted'][st]['wilson95']}  "
          f"discordant {p['only_first']}/{p['only_second']} p={p['p_exact_mcnemar']}")
