"""Check score_step8.py --phase mock against validation-v5_1.json and validation-stages-r2.json (sealed step 6 live B + step 5 A traces)."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_step8

E = HERE.parent.parent / 'assets/report/evidence'
ROOTS = [E / '2026-10-23/bowl-step6/paint_live_v1/runs', E / '2026-10-21/bowl-controlled-v2', E / '2026-10-21/bowl-controlled-v3',
         E / '2026-10-21/bowl-training-feasibility-v4']
N5 = 'fail, upstream predicate true at end'


def main(out, stamp):
    r = score_step8.main(['--phase', 'mock', '--out', str(out), '--stamp', stamp, *map(str, ROOTS)])
    v51 = {c['trace_sha256']: c for c in json.loads((HERE / 'validation-v5_1.json').read_text())['cases']}
    st = json.loads((HERE / 'validation-stages-r2.json').read_text())
    stv = {v51c['trace_sha256']: c['stages_r2'] for c in st['cases'] for v51c in v51.values() if v51c['case'] == c['case']}
    checked, diffs = [], []
    for e in r['episodes']:
        if not e['scored']:
            continue
        v, s = v51[e['trace_sha256']], stv[e['trace_sha256']]
        got = {'pass': e['v5_1_pass'], 'at_end': e['v5_1_pass_at_end'], 'win': e['win_control'], 'qopen': e['qualifying_open'],
               'flags': [f for f in e['review_flags'] if f != N5], 'furthest': e['furthest'], 'controls': e['stage_controls'], 'M2': e['M2']}
        want = {'pass': v['v5_1_pass'], 'at_end': v['v5_1_pass_at_end'], 'win': v['win_control_v5_1'], 'qopen': v['qualifying_open'],
                'flags': v['review_flags'], 'furthest': s['furthest'], 'controls': {k: s['best_attempt'][k] for k in score_step8.stages_r2.STAGES if k in s['best_attempt']},
                'M2': {k: s[k] for k in ('qualifying_open_control', 'time_to_win_P0_s', 'win_control') if k in s}}
        got['M2'] = {k: x for k, x in e['M2'].items() if k in want['M2']}
        if 'win_simulation_s' in s:  # S4: A's recorded simulation_s is already control/15, the time reported for both robots
            got['S4'] = e['M2']['win_s']
            want['S4'] = s['win_simulation_s'] if e['robot'] == 'A' else round(s['win_control'] / 15, 3)
        diffs += [(v['case'], k, got[k], want[k]) for k in got if got[k] != want[k]]
        checked.append(v['case'])
    m5 = st['m5_demo']
    r = json.loads(json.dumps(r, default=str))  # compare as written (tuples become lists)
    pair = r['paired']['B: painted vs unpainted']
    m5_ok = (r['per_condition']['B/painted']['stage_rates'] == m5['rates_painted'] and r['per_condition']['B/unpainted']['stage_rates'] == m5['rates_unpainted']
             and pair['stages'] == m5['paired_painted_vs_unpainted'])
    unchecked = sorted(c['case'] for h, c in v51.items() if c['case'] not in checked)
    md = (Path(out) / 'score_step8-mock.md').read_text()
    return {'episodes_checked': len(checked), 'diffs': diffs, 'm5_demo_reproduced': m5_ok, 'validation_cases_not_in_mock': unchecked,
            'S3_caveat_printed': score_step8.CONTACT_CAVEAT in md,
            'S5_disagreement_flags': [f for e in r['episodes'] for f in e.get('review_flags', []) if f.startswith('stages win')],
            'unscored_listed': [(e['case'], e['unscored_reason']) for e in r['episodes'] if not e['scored']],
            'n5_flagged': [e['case'] for e in r['episodes'] if N5 in e.get('review_flags', [])],
            'per_condition': {k: f"{x['wins']}/{x['n']} {x['wilson95']}" for k, x in r['per_condition'].items()}}


RUNNER = '/home/miodrag/robot-transfer/bowl/experiments/bowl_step6/step8_B_runner.py'
LOCAL_IK = '/home/miodrag/robot-transfer/bowl/experiments/bowl_step6/local_ik.py'
B = E / '2026-10-23/bowl-step6/paint_live_v1/runs'
LIFT = E / '2026-10-23/bowl-step6/paint_dev_v1/runs/step6_B_paint_dev_lift_l2002_s0_v1'
# hand-written tracebacks in the runner's real frames; the locked classifier must give each its kind (checked in synthetic_after)
SOLVE = (85, 'log_solve', "assert result['success'], (step,direction,result['reason'])", RUNNER)
FAILURES = {
    'B_joint_beyond_soft_limit': ('AssertionError', lambda step: '', [(298, '<module>', 'postq,postqd,postroot,posttool=state()', RUNNER),
                                                                    (201, 'state', 'assert (q>=soft[:,0]-.03).all() and (q<=soft[:,1]+.03).all()', RUNNER)]),
    'ik_action_A_to_B': ('AssertionError', lambda step: f"({step}, 'action_A_to_B', 'no converged solution')",
                         [(282, '<module>', "forward=solve(B,targetB,q[ids],Blim);log_solve(step,'action_A_to_B',forward,targetB,q[ids])", RUNNER), SOLVE]),
    None: ('RuntimeError', lambda step: 'CUDA error: an illegal memory access was encountered', [(286, '<module>', 'obs,_,term,trunc,_=env.step(action)', RUNNER)]),
}
TSV_HEAD = 'phase\trobot\tpolicy\tlayout\tcase\tstatus\tfailure\tv4_success\tseconds'


def tsv_row(ph, robot, pol, layout, case, status='0', failure='no'):
    return '\t'.join((ph, robot, pol, layout, case, status, failure, 'x', '1'))


def write_abort(runs, case, src, rows, kind, short=0, marker=None, status=1):
    """What the runner's finally block and classify_abort.py leave: failure.txt, the marker, and the partial trace if any.
    short=1 is a :298 abort, where gate_trace.jsonl has one row more than tracking.npz. marker overrides the class/kind written."""
    d = runs / case
    d.mkdir(parents=True)
    exc, message, frames = FAILURES[kind]
    message = message(rows)  # the loop step at :282 equals the gate rows
    text = 'Traceback (most recent call last):\n' + ''.join(f'  File "{f}", line {l}, in {fn}\n    {src_}\n' for l, fn, src_, f in frames)
    text += f'{exc}: {message}\n' if message else f'{exc}\n'
    (d / 'failure.txt').write_text(text)
    if rows:
        with open(src / 'gate_trace.jsonl') as f:
            (d / 'gate_trace.jsonl').write_text(''.join(next(f) for _ in range(rows)))
    if rows - short:  # the runner's finally block writes tracking.npz only when it has rows
        z = np.load(src / 'tracking.npz')
        np.savez_compressed(d / 'tracking.npz', complete=np.asarray(False), step=z['step'][:rows - short],
                            objects_banana_bowl=z['objects_banana_bowl'][:rows - short])
    cls, kind = marker or (('safety_abort' if kind else 'crash'), kind)
    (d / 'abort-classification.json').write_text(json.dumps({
        'schema': score_step8.ABORT_SCHEMA, 'case': case, 'class': cls, 'kind': kind, 'exit_status': status, 'exception': exc, 'message': message,
        'frames': [{'file': f, 'line': l, 'function': fn} for l, fn, _, f in frames],
        'failure_txt_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'partial_trace': {'gate_trace_rows': rows, 'last_gate_trace_step': rows or None, 'tracking_npz': bool(rows - short),
                          'failed_solve_json': False, 'checks_json': False}}, indent=2))
    return d


def locked_classifier(d):
    """The locked classifier's (class, kind) for a written abort, on its own failure.txt, exit status and gate rows."""
    m = json.loads((d / 'abort-classification.json').read_text())
    gate = d / 'gate_trace.jsonl'
    rows = sum(1 for l in gate.read_text().splitlines() if l.strip()) if gate.exists() else 0
    kind = score_step8.classify_abort.classify(*score_step8.classify_abort.parse((d / 'failure.txt').read_text()), m['exit_status'], rows)
    return ('safety_abort' if kind else 'crash', kind, rows)


def synthetic_after():
    """Step 8 names, R4, N4 and B2 safety aborts on a fake after-phase tree built from sealed traces (symlinks and truncated copies)."""
    A = E / '2026-10-21/bowl-training-feasibility-v4/step5_training_layout2002_v4'
    task = 'BananaInBowlUniformInitPose10cmTask'
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        def a_case(case, rows=None):
            (t / 'runs' / case).mkdir(parents=True)
            o = t / 'RoboLab/output' / case / task
            o.mkdir(parents=True)
            d = json.loads((A / task / 'strict-metric.json').read_text())
            if rows:
                d['trace'] = d['trace'][:rows]  # ends before the v5.1 window closes; the live v4 record stays a pass
            (o / 'strict-metric.json').write_text(json.dumps(d))
        def b_case(case, src):
            (t / 'runs' / case).mkdir(parents=True)
            for n in ('gate_trace.jsonl', 'tracking.npz', 'checks.json'):
                os.symlink(B / src / n, t / 'runs' / case / n)
        a_case('step8_A_smoke_l2000_s0_v1')
        a_case('step8_A_P0_l4000_s0_v1'); a_case('step8_A_P1_l4000_s0_v1', rows=295); a_case('step8_A_P0canary_l4000_s0_v1')
        a_case('step8_A_P0_l4001_s0_v1'); a_case('step8_A_P1_l4001_s0_v1')
        p0 = {'4000': 'base_s0_painted', '4001': 'l2002_s0_painted', '4002': 'l2002_s0_unpainted', '4003': 'base_s1_painted',
              '4004': 'base_s0_unpainted', '4005': 'l2002_s0_unpainted', '4006': 'base_s1_unpainted'}
        for l, src in p0.items():
            b_case(f'step8_B_P0_l{l}_s0_v1', f'step6_B_paint_live_{src}_v1')
        b_case('step8_B_P1_l4000_s0_v1', 'step6_B_paint_live_base_s1_unpainted_v1')
        runs = t / 'runs'
        written = {
            # mid-episode, just after the grasp at 1248: unaligned, stages_r2 raises IndexError on this shape (touch at the last gate row)
            '4002': write_abort(runs, 'step8_B_P1_l4002_s0_v1', B / 'step6_B_paint_live_base_s0_painted_v1', 1250, 'B_joint_beyond_soft_limit', short=1),
            # a pre-C1 marker: safety_abort on the first control with no trace files; under C1 0 gate rows is always a crash
            '4003': write_abort(runs, 'step8_B_P1_l4003_s0_v1', None, 0, 'ik_action_A_to_B'),
            '4004': write_abort(runs, 'step8_B_P1_l4004_s0_v1', B / 'step6_B_paint_live_base_s1_painted_v1', 900, None),
            '4005': write_abort(runs, 'step8_B_P1_l4005_s0_v1', LIFT, 1700, 'ik_action_A_to_B'),  # equal lengths, after the first rest at 1554
            '4006': write_abort(runs, 'step8_B_P1_l4006_s0_v1', B / 'step6_B_paint_live_base_s0_painted_v1', 1, 'B_joint_beyond_soft_limit', short=1),  # step 0
        }
        classifier = {l: locked_classifier(d) for l, d in written.items()}
        for ph, rows in (('before', [tsv_row('before', 'A', 'P0', l, f'step8_A_P0_l{l}_s0_v1') for l in ('4000', '4001')]
                                    + [tsv_row('before', 'B', 'P0', l, f'step8_B_P0_l{l}_s0_v1') for l in p0]),
                         ('after', [tsv_row('after', 'A', 'P0', '4000', 'step8_A_P0canary_l4000_s0_v1'), tsv_row('after', 'A', 'P1', '4000', 'step8_A_P1_l4000_s0_v1'),
                                    tsv_row('after', 'B', 'P1', '4000', 'step8_B_P1_l4000_s0_v1'), tsv_row('after', 'A', 'P1', '4001', 'step8_A_P1_l4001_s0_v1', '1', 'capture'),
                                    *(tsv_row('after', 'B', 'P1', l, f'step8_B_P1_l{l}_s0_v1', '1', json.loads((d / 'abort-classification.json').read_text())['class'])
                                      for l, d in written.items())])):
            (t / 'runs' / f'step8_{ph}_v1-supervisor').mkdir()
            (t / 'runs' / f'step8_{ph}_v1-supervisor/episodes.tsv').write_text('\n'.join([TSV_HEAD, *rows]) + '\n')
        r = score_step8.main(['--phase', 'after', '--out', str(t / 'out'), str(t / 'runs'), str(t / 'RoboLab/output')])
        md = (t / 'out/score_step8-after.md').read_text()
    ep = {e['case']: e for e in r['episodes']}
    a1, mid, zero_rows, crash, late, step0 = (ep[f'step8_{c}_s0_v1'] for c in ('A_P1_l4000', 'B_P1_l4002', 'B_P1_l4003', 'B_P1_l4004', 'B_P1_l4005', 'B_P1_l4006'))
    pa, pb = r['paired']['A: P1 vs P0'], r['paired']['B: P1 vs P0']
    want = {'cases': sorted(ep) == sorted(['step8_A_P0_l4000_s0_v1', 'step8_A_P1_l4000_s0_v1', 'step8_A_P0canary_l4000_s0_v1', 'step8_A_P0_l4001_s0_v1',
                                           'step8_A_P1_l4001_s0_v1', 'step8_B_P1_l4000_s0_v1', *(f'step8_B_P0_l{l}_s0_v1' for l in p0),
                                           *(f'step8_B_P1_l{l}_s0_v1' for l in written)]),
            'smoke_ignored': not any('smoke' in c for c in ep),
            'N4_truncated_A_flagged': a1['live_v4']['pass'] and not a1['v5_1_pass'] and 'v4 pass, v5.1 fail' in a1['review_flags'],
            'failed_A_unscored': not ep['step8_A_P1_l4001_s0_v1']['scored'],
            'R4_A': (pa['scored_layouts'], pa['unscored_layouts']) == (['l4000'], ['l4001']),
            'A_win_discordant': (pa['win']['only_first'], pa['win']['only_second']) == (0, 1),
            'canary_scored_not_paired': r['per_condition']['A/P0canary']['n'] == 1,
            'locked_classifier_gives_each_kept_abort_its_kind': classifier == {
                '4002': ('safety_abort', 'B_joint_beyond_soft_limit', 1250), '4003': ('crash', None, 0), '4004': ('crash', None, 900),
                '4005': ('safety_abort', 'ik_action_A_to_B', 1700), '4006': ('safety_abort', 'B_joint_beyond_soft_limit', 1)},
            'P1_one_row_longer_scored': mid['scored'] and not mid['complete'] and not mid['v5_1_pass'] and mid['furthest'] == 'grasp'
                                        and mid['fail_reason'] == 'safety_abort:B_joint_beyond_soft_limit' and mid['live_v4']['reached'] is False
                                        and (mid['safety_abort']['gate_trace_rows'], mid['safety_abort']['scored_rows'], mid['safety_abort']['dropped_gate_rows']) == (1250, 1249, 1),
            'P1_step0_no_tracking_scored': step0['scored'] and step0['safety_abort']['empty_trace'] and step0['furthest'] is None and not step0['v5_1_pass']
                                           and (step0['safety_abort']['gate_trace_rows'], step0['safety_abort']['scored_rows']) == (1, 0),
            'P1_equal_lengths_scored': late['safety_abort']['dropped_gate_rows'] == 0 and late['safety_abort']['scored_rows'] == 1700,
            'P2_class_kind_next_to_verdict': all((e['abort_class'], e['abort_kind']) == k for e, k in (
                (mid, ('safety_abort', 'B_joint_beyond_soft_limit')), (crash, ('crash', None)),
                (late, ('safety_abort', 'ik_action_A_to_B')), (step0, ('safety_abort', 'B_joint_beyond_soft_limit')))),
            'C1_zero_rows_unscored': not zero_rows['scored'] and zero_rows['unscored_reason'][0].startswith('marker_disagrees_with_classifier')
                                     and zero_rows['abort_classifier'] == {'class': 'crash', 'kind': None, 'gate_rows_on_disk': 0},
            'B2_win_before_abort_is_a_win': late['v5_1_pass'] and late['win_control'] == 1554 and late['fail_reason'] is None,
            'safety_abort_flag_on_every_scored_abort': all('safety_abort' in e['review_flags'] for e in (mid, late, step0)),
            'crash_unscored': not crash['scored'] and crash['unscored_reason'][0].startswith('crash'),
            'R4_B': (pb['scored_layouts'], pb['unscored_layouts']) == (['l4000', 'l4002', 'l4005', 'l4006'], ['l4001', 'l4003', 'l4004']),
            'B_win_discordant': (pb['win']['only_first'], pb['win']['only_second']) == (1, 0),
            'abort_counts': r['per_condition']['B/P1']['safety_aborts'] == {'n': 3, 'wins': 1, 'empty_trace': 1, 'aborted_row_dropped': 2,
                                                                             'by_kind': {'B_joint_beyond_soft_limit': 2, 'ik_action_A_to_B': 1}}
                            and (r['per_condition']['B/P1']['crashes'], r['per_condition']['B/P1']['abort_markers_unscored_other']) == (1, 1),
            'abort_counts_in_markdown': '| B/P1 | 3 | B_joint_beyond_soft_limit 2, ik_action_A_to_B 1 | 1 | 1 | 2 | 1 | 1 |' in md
                                        and '| B/P0 | 0 | — | 0 | 0 | 0 | 0 | 0 |' in md and 'empty trace, 1 gate row)' in md
                                        and score_step8.A_LIMIT_CAVEAT in md}
    return {'checks': want, 'locked_classifier': classifier, 'flags_truncated_A': a1['review_flags'], 'fail_reason_truncated_A': a1['fail_reason'],
            'mid_abort': {k: mid[k] for k in ('safety_abort', 'furthest', 'stage_controls', 'gate_fail_reason', 'review_flags')},
            'step0_abort': {k: step0[k] for k in ('safety_abort', 'furthest', 'fail_reason', 'review_flags')},
            'abort_after_win': {k: late[k] for k in ('safety_abort', 'win_control', 'M2', 'stage_controls', 'review_flags')},
            'zero_rows': zero_rows['unscored_reason'], 'crash': crash['unscored_reason'], 'B_P1_safety_aborts': r['per_condition']['B/P1']['safety_aborts']}


if __name__ == '__main__':
    res = main(sys.argv[1] if len(sys.argv) > 1 else HERE / 'score_step8_mock', sys.argv[2] if len(sys.argv) > 2 else '')
    res['synthetic_after'] = synthetic_after()
    print(json.dumps(res, indent=1))
    sys.exit(0 if not res['diffs'] and res['m5_demo_reproduced'] and res['S3_caveat_printed'] and not res['S5_disagreement_flags']
             and all(res['synthetic_after']['checks'].values()) else 1)
