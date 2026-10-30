"""Validate the EXPLORATORY pose-based first rest on the 32 step 7 first-pass traces (A), and show B field access on sealed step 6 B.

python validate_pose_variant.py [STAMP]  ->  validation-pose-variant.json. Step 7 and step 6 inputs only.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pose_variant as pv
from release_gate_v5_1 import load, score

ROOT = HERE.parent.parent
STEP7 = ROOT / 'review/step7_first'
B = [ROOT / 'assets/report/evidence/2026-10-23/bowl-step6/paint_live_v1/runs' / f'step6_B_paint_live_{c}_v1'
     for c in ('l2002_s0_painted', 'l2002_s0_unpainted', 'base_s0_painted', 'base_s0_unpainted', 'base_s1_painted', 'base_s1_unpainted')]
B.append(ROOT / 'assets/report/evidence/2026-10-23/bowl-step6/paint_dev_v1/runs/step6_B_paint_dev_lift_l2002_s0_v1')
# the reviewer's expected union changes (layout: (v5.1 control, union control)); every other episode keeps its v5.1 result
CHANGES = {2006: (None, 195), 2017: (None, 166), 2022: (543, 153), 2002: (301, 297), 2024: (445, 444)}


def layout(case):
    return int(case.split('layout')[1][:4])


def pose_with_infinite_first_row(path):
    """The same pose replay with a non-finite first-row rate, to show the sentinel choice changes nothing here."""
    robot, rows, _, _ = load(path)
    rates = pv.pose_rates(pv.quaternions(robot, path, rows), [r['simulation_s'] for r in rows])
    return score(path, gate_cls=pv.pose_gate([math.inf, *rates[1:]]))['success_control']


def b_fields(d):
    """Evidence that B's quaternion field is read: norms, and pose-derived vs reported angular speed over the trace."""
    path = d / 'gate_trace.jsonl'
    robot, rows, _, _ = load(path)
    q = pv.quaternions(robot, path, rows)
    rates = pv.pose_rates(q, [r['simulation_s'] for r in rows])[1:]
    norm = np.linalg.norm(q, axis=1)
    reported = [r['angular_rad_s'] for r in rows[1:]]
    return {'case': d.name, 'rows': len(rows), 'quaternion_norm_range': [round(float(norm.min()), 6), round(float(norm.max()), 6)],
            'pose_rate_median_rad_s': round(float(np.median(rates)), 4), 'reported_angular_median_rad_s': round(float(np.median(reported)), 4),
            'rows_pose_rate_le_0_3': int(sum(x <= .3 for x in rates)), 'rows_reported_le_0_3': int(sum(x <= .3 for x in reported))}


def main(stamp=''):
    recorded = {c['case']: c['win_control'] for c in json.loads((STEP7 / 'candidates-first-v1.json').read_text())['cases']}
    cases = sorted(p for p in (STEP7 / 'episodes').iterdir() if p.is_dir())
    a = [pv.episode(c) for c in cases]
    diffs = []
    for r in a:
        want_v51, want_union = CHANGES.get(layout(r['case']), (recorded[r['case']], recorded[r['case']]))
        got = (r['v5_1_win_control'], r['union_win_control'])
        if got != (want_v51, want_union):
            diffs.append({'case': r['case'], 'got (v5.1, union)': got, 'expected': (want_v51, want_union)})
        if r['v5_1_win_control'] != recorded[r['case']]:
            diffs.append({'case': r['case'], 'v5.1 differs from candidates-first-v1.json': (r['v5_1_win_control'], recorded[r['case']])})
        if r['not_computable']:
            diffs.append({'case': r['case'], 'not_computable': r['not_computable']})
    inf_first = {r['case']: pose_with_infinite_first_row(Path(r['trace'])) for r in a}
    unchanged = [r for r in a if layout(r['case']) not in CHANGES]
    counts = {'episodes': len(a), 'v5_1_passes': sum(r['v5_1_pass'] for r in a), 'union_passes': sum(r['union_pass'] for r in a),
              'changed': sorted(layout(r['case']) for r in a if r['union_win_control'] != r['v5_1_win_control']),
              'unchanged_passes': sum(r['v5_1_pass'] for r in unchanged), 'unchanged_fails': sum(not r['v5_1_pass'] for r in unchanged),
              'brief_said': '8 unchanged passes, 19 unchanged fails',
              'won_by': {k: sum(r['won_by'] == k for r in a) for k in ('v5.1', 'pose', 'both', None)}}
    res = {'stamp_utc': stamp, 'scope': 'EXPLORATORY pose-based first rest (EXPERIMENT_REPORT §22.11); step 7 first pass (A) and sealed step 6 B only',
           'code_sha256': {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ('pose_variant.py', 'release_gate_v5_1.py')},
           'step7_candidates_sha256': hashlib.sha256((STEP7 / 'candidates-first-v1.json').read_bytes()).hexdigest(),
           'A_step7_first': a, 'A_diffs_from_expected': diffs, 'A_counts': counts,
           'first_row_infinite_gives_same_pose_controls': all(inf_first[r['case']] == r['pose_win_control'] for r in a),
           'B_step6_field_access': [pv.episode(d) for d in B], 'B_step6_fields': [b_fields(d) for d in B]}
    (HERE / 'validation-pose-variant.json').write_text(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    r = main(sys.argv[1] if len(sys.argv) > 1 else '')
    print(json.dumps({k: r[k] for k in ('A_diffs_from_expected', 'A_counts', 'first_row_infinite_gives_same_pose_controls')}, indent=1))
    for e, f in zip(r['B_step6_field_access'], r['B_step6_fields']):
        print(e['case'], e['v5_1_win_control'], e['pose_win_control'], e['union_win_control'], e['won_by'], e['not_computable'], f)
    sys.exit(1 if r['A_diffs_from_expected'] or not r['first_row_infinite_gives_same_pose_controls'] else 0)
