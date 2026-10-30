"""Stage metrics revision 2 vs revision 1 on the 22 traces of validation-v5_1.json; v5.1 verdicts re-checked. CPU only.

Writes validation-stages-r2.json.
"""
import hashlib
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'bowl_step5')]
from release_gate_v5_1 import score
import stages as r1
import stages_r2 as r2

E = HERE.parents[1] / 'assets/report/evidence'
cases = json.loads((HERE / 'validation-v5_1.json').read_text())['cases']
out, live = [], {}
for c in cases:
    path = E / c['trace']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == c['trace_sha256'], c['case']
    s, a, b = score(path), r1.stages(path), r2.stages(path)
    verdict_same = (s['pass'], s['success_control']) == (c['v5_1_pass'], c['win_control_v5_1'])
    ctl = lambda x: [x['best_attempt'].get(st) for st in r1.STAGES[:x['best_attempt']['depth']]]
    out.append({'case': c['case'], 'human_pass': c['human_pass'], 'v5_1_pass': s['pass'], 'v5_1_verdict_unchanged': verdict_same,
                'furthest_r1': a['furthest'], 'furthest_r2': b['furthest'], 'controls_r1': ctl(a), 'controls_r2': ctl(b),
                'changed': (a['furthest'], ctl(a)) != (b['furthest'], ctl(b)), 'stages_r2': b})
    if c['case'].startswith('B live'):
        layout, arm = c['case'][7:].rsplit('_', 1)
        live.setdefault(arm, {})[layout] = b
    print(f"{c['case']:25} v5.1={s['pass']!s:5} unchanged={verdict_same!s:5} | r1 {a['furthest']!s:9} {ctl(a)} -> r2 {b['furthest']!s:9} {ctl(b)}"
          f"{'  CHANGED' if out[-1]['changed'] else ''}")
demo = {'note': 'illustration of M5 only: 3 B live layouts, painted vs unpainted', 'rates_painted': r2.rates(list(live['painted'].values())),
        'rates_unpainted': r2.rates(list(live['unpainted'].values())),
        'paired_painted_vs_unpainted': [r2.paired(live['painted'], live['unpainted'], st) for st in r2.STAGES]}
code = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()
        for n in ('release_gate_v5_1.py', 'stages.py', 'stages_r2.py', 'validate_stages_r2.py')}
(HERE / 'validation-stages-r2.json').write_text(json.dumps({'classification': 'offline development validation; no new trials',
                                                           'code_sha256': code, 'cases': out, 'm5_demo': demo}, indent=1))
for x in demo['paired_painted_vs_unpainted'][:3]:
    print(f"M5 demo {x['stage']:9} painted {demo['rates_painted'][x['stage']]['k']}/3 unpainted {demo['rates_unpainted'][x['stage']]['k']}/3 "
          f"discordant {x['only_first']}/{x['only_second']} p={x['p_exact_mcnemar']}")
