"""#1 Failure labeller: per-episode TEXT evidence summary from telemetry -> Nemotron -> schema-checked failure label.

Post-hoc only. Reads the sealed step 8 "before" evidence copies and the locked v5.1 score file, never writes to them,
and keeps its labels apart from the locked scores and the reviewer's verdicts (compared, never merged).

Evidence arms:
  blind     telemetry only: no scorer verdict, no gate events, no reviewer text (the primary arm)
  with_v51  the same summary plus the locked v5.1 fields (verdict, win control, fail_reason, stages, review flags);
            measures how much the label leans on the scorer instead of the trace
Summary thresholds below are descriptive only (they shape the narrative, they score nothing).

  python failure_labeller.py --summary step8_A_P0_l4000_s0_v1     print one evidence summary (offline)
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVIDENCE, LOCKED_SCORE, REVIEW_LOG, TASK_NAME, ref, text_sha  # noqa: E402

PROMPT_VERSION = 'failure-labeller-v1'
HZ = 15                # control/15 s on both robots (locked scorer note "times")
DEBOUNCE = 3           # samples (0.2 s) a flag must hold before a transition is reported
LIFT_CLEAR_M = 0.02    # banana root this far above its start, fingertip contact, touching neither table nor bowl
OVER_BOWL_R_M = 0.081  # horizontal banana-bowl root distance read as "over the bowl" (bowl outer half-width)
REST_V, REST_W, REST_N = 0.03, 0.3, 15   # free banana at rest: |v| <= 0.03 m/s and |w| <= 0.3 rad/s for 1 s
MAX_EVENTS = 40       # timeline lines
MOVED_M = 0.02         # a gripper cycle that neither lifts nor moves the banana this far is collapsed

FAILURE_MODES = ['slip_during_carry', 'squeezed_out', 'closed_gripper_set_down', 'missed_grasp',
                 'regrasp_after_release', 'ik_stuck_at_joint_limit', 'soft_limit_abort', 'ends_on_table',
                 'ends_on_rim', 'success', 'other']
OUTCOMES = ['success', 'failure', 'uncertain']


def schema(order_seed=None):
    """Label schema; order_seed shuffles the enum order (answer-order bias check across repeats)."""
    modes, outcomes = FAILURE_MODES[:], OUTCOMES[:]
    if order_seed is not None:
        rng = random.Random(order_seed)
        rng.shuffle(modes)
        rng.shuffle(outcomes)
    return {'type': 'object', 'additionalProperties': False,
            'required': ['outcome_guess', 'failure_mode', 'evidence_controls', 'one_line_explanation', 'confidence'],
            'properties': {
                'outcome_guess': {'type': 'string', 'enum': outcomes},
                'failure_mode': {'type': 'string', 'enum': modes},
                'evidence_controls': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}, 'minItems': 1, 'maxItems': 8},
                'one_line_explanation': {'type': 'string', 'maxLength': 300},
                'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}}}


SYSTEM = """You label one simulated robot episode from a telemetry summary. Task: pick up the banana and place it in the bowl.
Success means: the robot deliberately commands the gripper OPEN while holding the lifted banana; the banana's first support after that release is the original bowl (resting on the bowl's rim counts) and it then comes to rest supported by the bowl alone. A banana that slips out of a still-closed gripper never counts, even if it lands in the bowl.
Failure modes (pick the one that best explains the episode; pick "success" if it succeeded):
- slip_during_carry: banana lost from a closed gripper while carried
- squeezed_out: banana ejected by the closing fingers during or right after the grasp
- closed_gripper_set_down: banana put down while the gripper stayed closed (no deliberate open)
- missed_grasp: gripper closes but never holds the banana clear of the table
- regrasp_after_release: banana released, then grasped again before it settled
- ik_stuck_at_joint_limit: episode stopped by an inverse-kinematics failure with a joint pinned at its limit
- soft_limit_abort: episode stopped because a joint passed its soft limit
- ends_on_table: released, but the banana's first support is the table or it settles on the table
- ends_on_rim: released, but the banana ends straddling the bowl rim and the table instead of resting on the bowl alone
- other: none of these
Use only the evidence given. Cite in evidence_controls the control indices (the c### numbers) that prove your label. Reply with one JSON object matching this schema:
"""


def _debounce(flags, n=DEBOUNCE):
    """[(index, value)] state changes, ignoring flickers shorter than n samples; first entry is the initial state."""
    out, cur, i = [(0, flags[0])], flags[0], 0
    while i < len(flags):
        if flags[i] != cur and i + n <= len(flags) and all(flags[j] == flags[i] for j in range(i, i + n)):
            cur = flags[i]
            out.append((i, cur))
        i += 1
    return out


def case_dirs(case):
    robot = case.split('_')[1]
    if robot == 'A':
        return robot, EVIDENCE / 'RoboLab/output' / case / TASK_NAME, EVIDENCE / 'runs' / case
    return robot, None, EVIDENCE / 'runs' / case


def load(case):
    """Per-control rows in one shape for both robots, plus the evidence files used (path + sha256)."""
    import numpy as np
    robot, adir, rdir = case_dirs(case)
    refs, extra = [], {}
    if robot == 'A':
        path = adir / 'strict-metric.json'
        raw = json.loads(path.read_text())['trace']
        refs.append(ref(path))
        where = np.array([[r['post']['poses_env_local'][n][0] for n in ('banana', 'bowl')] for r in raw], float)
        limit = 750
    else:
        path = rdir / 'gate_trace.jsonl'
        raw = [json.loads(x) for x in path.read_text().splitlines()]
        refs.append(ref(path))
        npz = rdir / 'tracking.npz'
        where = np.load(npz)['objects_banana_bowl'][:, :, :3].astype(float)
        refs.append(ref(npz))
        raw = raw[:len(where)]  # a :298 abort leaves one gate row without a tracking row (the aborted control)
        limit = 9000
        for name in ('checks.json', 'abort-classification.json', 'failure.txt', 'failed_solve.json'):
            if (rdir / name).exists():
                refs.append(ref(rdir / name))
                extra[name] = (rdir / name).read_text()
    rise = where[:, 0, 2] - where[0, 0, 2]
    dxy = np.linalg.norm(where[:, 0, :2] - where[:, 1, :2], axis=1)
    rows = [{'c': r['step'], 'closed': r['command'] > .5, 'closure': r['post']['closure'], 'tip': bool(r['post']['contact']),
             'table': bool(r['post']['table_contact']), 'bowl': bool(r['post']['bowl_contact']),
             'rim': bool(r['post']['rim_support']), 'bsup': bool(r['post']['bowl_support']),
             'above': r['post']['centroid_height_above_bowl_m'], 'upred': bool(r['upstream_predicate']),
             'v': r['linear_m_s'], 'w': r['angular_rad_s'], 'rise': float(rise[k]), 'dxy': float(dxy[k])}
            for k, r in enumerate(raw)]
    return {'case': case, 'robot': robot, 'rows': rows, 'limit': limit, 'refs': refs, 'extra': extra}


EVENT_TEXT = {('cmd', True): 'CLOSE cmd', ('cmd', False): 'OPEN cmd', ('tip', True): 'tip contact on',
              ('tip', False): 'tip contact off', ('table', True): 'touches table', ('table', False): 'leaves table',
              ('bowl', True): 'touches bowl', ('bowl', False): 'leaves bowl', ('rim', True): 'rim support on',
              ('rim', False): 'rim support off', ('clear', True): 'held clear', ('clear', False): 'held-clear ends',
              ('rest', True): 'free banana at rest 1 s'}
MINOR = {('tip', True), ('table', False), ('bowl', False), ('rim', True), ('rim', False)}  # dropped first over budget,
# together with fingertip contact changes while the gripper is commanded open (pads brushing a released banana)
LEGEND = ('State [..]: cl = gripper closure (0 open, 1 closed); tip / no-tip = fingertip contact; z = banana rise above '
          f'its start (cm); d = horizontal banana-bowl distance (cm, <= {100 * OVER_BOWL_R_M:.1f} is over the bowl); '
          'T / B = banana touching table / bowl, R = supported on the bowl rim, - = touching nothing.')


def _state(r):
    touch = ''.join(x for x, k in (('T', 'table'), ('B', 'bowl'), ('R', 'rim')) if r[k]) or '-'
    return (f"[cl{max(r['closure'], 0.0):.2f} {'tip' if r['tip'] else 'no-tip'} z{100 * r['rise']:+.1f} "
            f"d{100 * r['dxy']:.1f} {touch}]")


def events(ep):
    """[(row index, (signal, value))] in time order, plus the per-control held-clear flags."""
    rows = ep['rows']
    ev = [(i, ('cmd', rows[i]['closed'])) for i in range(1, len(rows)) if rows[i]['closed'] != rows[i - 1]['closed']]
    clear = [r['tip'] and not r['table'] and not r['bowl'] and r['rise'] >= LIFT_CLEAR_M for r in rows]
    free_rest = [not r['tip'] and abs(r['v']) <= REST_V and abs(r['w']) <= REST_W for r in rows]
    for key in ('tip', 'table', 'bowl', 'rim'):
        ev += [(i, (key, v)) for i, v in _debounce([r[key] for r in rows])[1:]]
    ev += [(i, ('clear', v)) for i, v in _debounce(clear)[1:]]
    ev += [(i, ('rest', True)) for i, v in _debounce(free_rest, REST_N)[1:] if v]
    order = list(EVENT_TEXT)
    return sorted(ev, key=lambda e: (e[0], order.index(e[1]))), clear


def _items(ev, rows, clear, limit=MAX_EVENTS):
    """Timeline items, fitted to `limit` lines. Gripper cycles (CLOSE to next CLOSE) in which the banana is neither
    held clear nor moved > MOVED_M are collapsed into one line per run; then MINOR signals go; the middle goes last."""
    closes = [i for i, e in ev if e == ('cmd', True)]
    bounds = [0] + closes + [len(rows)]
    quiet = set()
    for a, b in zip(bounds[1:-1], bounds[2:]):
        d = [r['dxy'] for r in rows[a:b]]
        if not any(clear[a:b]) and max(d) - min(d) <= MOVED_M:
            quiet.add(a)
    items, run = [], []
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if a in quiet:
            run.append((a, b))
            continue
        if run:
            items.append(('quiet', run))
            run = []
        groups = {}
        for i, e in ev:
            if a <= i < b:
                groups.setdefault(i, []).append(e)
        items += [('events', i, es) for i, es in groups.items()]
    if run:
        items.append(('quiet', run))
    notes = []
    if len(items) > limit:
        kept, dropped = [], 0
        for x in items:
            if x[0] == 'events':
                es = [e for e in x[2] if e not in MINOR and not (e[0] == 'tip' and not rows[x[1]]['closed'])]
                dropped += len(x[2]) - len(es)
                x = ('events', x[1], es) if es else None
            if x:
                kept.append(x)
        items = kept
        notes.append(f'{dropped} minor events dropped (tip contact on, tip contact changes while commanded open, '
                     'leaves table/bowl, rim on/off)')
    if len(items) > limit:
        head, tail = items[:limit - 15], items[-14:]
        notes.append(f'{len(items) - len(head) - len(tail)} timeline lines omitted in the middle')
        items = head + [('omitted', len(items) - len(head) - len(tail))] + tail
    return items, notes


def _quiet_line(rows, run):
    a, b = run[0][0], run[-1][1] - 1
    cs = [rows[x]['c'] for x, _ in run]
    listed = ', '.join(f'c{c}' for c in cs[:6]) + (f', ... c{cs[-1]}' if len(cs) > 6 else '')
    return (f'  c{rows[a]["c"]}..c{rows[b]["c"]} {len(cs)} gripper cycle(s) (CLOSE at {listed}) without lifting the '
            f'banana or moving it > {100 * MOVED_M:.0f} cm; at c{rows[b]["c"]} {_state(rows[b])}')


def summary(case, arm='blind', locked=None):
    """Compact text evidence. Returns (text, refs)."""
    ep = load(case)
    rows, robot = ep['rows'], ep['robot']
    ev, clear = events(ep)
    items, notes = _items(ev, rows, clear)
    lines = [f'Episode {case}. Robot {robot}: ' + (
        'Franka Panda with its own gripper, running the policy natively. Fingertip contact flag = one finger.'
        if robot == 'A' else
        'Kinova Gen3 with a Robotiq 2F-85, driven by the Franka-trained policy through a fixed adapter; each policy '
        'action is held for 12 controls, so events are spread over more controls than on robot A. Fingertip contact '
        'flag = either of two finger pads.'),
        f'Controls run at {HZ} per second. Trace: controls {rows[0]["c"]}..{rows[-1]["c"]} '
        f'(max {ep["limit"]}, {rows[-1]["c"] / HZ:.1f} s).', LEGEND,
        f'Start c{rows[0]["c"]}: {_state(rows[0])}', 'Events (cN = control index; state after that control):']
    for x in items:
        lines.append(_quiet_line(rows, x[1]) if x[0] == 'quiet' else f'  ... {x[1]} lines omitted ...'
                     if x[0] == 'omitted' else f"  c{rows[x[1]]['c']} {'; '.join(EVENT_TEXT[e] for e in x[2])} {_state(rows[x[1]])}")
    if notes:
        lines.append('Note: ' + '; '.join(notes) + '.')
    held = sum(clear)
    lines.append(f"Totals: {sum(e == ('cmd', True) for _, e in ev)} close and {sum(e == ('cmd', False) for _, e in ev)} "
                 f"open commands; max rise {100 * max(r['rise'] for r in rows):.1f} cm; held clear {held} controls "
                 f'({held / HZ:.1f} s).')
    end = rows[-1]
    lines.append(f"End c{end['c']}: {_state(end)}; banana centre {100 * end['above']:+.1f} cm vs bowl top; "
                 f"bowl-supported {'yes' if end['bsup'] else 'no'}; free at rest "
                 f"{'yes' if not end['tip'] and abs(end['v']) <= REST_V and abs(end['w']) <= REST_W else 'no'}; "
                 f"RoboLab object-in-bowl predicate {'true' if end['upred'] else 'false'}.")
    lines += _abort_lines(ep)
    if arm == 'with_v51':
        e = (locked or locked_episodes())[case]
        win = f" at c{e['win_control']}" if e['win_control'] else ''
        lines.append(f"Locked v5.1 scorer (offline): first-rest {'PASS' if e['v5_1_pass'] else 'FAIL'}{win}; "
                     f"fail_reason {e['fail_reason']}; furthest stage {e['furthest']}; stage controls "
                     f"{e['stage_controls']}; review flags {e['review_flags']}.")
    elif arm != 'blind':
        raise ValueError(arm)
    return '\n'.join(lines), ep['refs']


def _abort_lines(ep):
    x = ep['extra']
    if 'abort-classification.json' not in x:
        return [] if ep['robot'] == 'A' else ['No safety abort: the episode ran to its time limit or the runner stop rule.']
    a = json.loads(x['abort-classification.json'])
    out = [f"SAFETY ABORT ({a['kind']}): runner stopped at control {a['partial_trace']['last_gate_trace_step']} with "
           f"{a['exception']} {a['message'] or '(soft joint-limit assertion in the state read)'}."]
    if 'failed_solve.json' in x:
        s = json.loads(x['failed_solve.json'])
        pinned = [f"joint {j + 1} q={q:.4f} at {'lower' if abs(q - lo) < 1e-3 else 'upper'} bound {lo if abs(q - lo) < 1e-3 else hi:.4f}"
                  for j, (q, lo, hi) in enumerate(zip(s['q'], s['local_lower'], s['local_upper']))
                  if abs(q - lo) < 1e-3 or abs(q - hi) < 1e-3]
        out.append(f"Failed IK solve at control {s['step']} ({s['direction']}): reason {s['reason']}, {s['iterations']} "
                   f"iterations, residual {1000 * s['position_error_m']:.2f} mm / {s['rotation_error_rad']:.4f} rad; "
                   f"joints at a local bound: {', '.join(pinned) or 'none'}.")
    return out


def locked_episodes():
    return {e['case']: e for e in json.loads(LOCKED_SCORE.read_text())['episodes']}


def cases():
    return list(locked_episodes())


def messages(text, sch):
    return [{'role': 'system', 'content': SYSTEM + json.dumps(sch)},
            {'role': 'user', 'content': text + '\nLabel this episode.'}]


def grounding(data, text):
    """Cited controls that appear as c### in the summary (a deterministic citation check)."""
    import re
    present = {int(x) for x in re.findall(r'\bc(\d+)\b', text)} | {int(x) for x in re.findall(r'control (\d+)', text)}
    cited = data.get('evidence_controls', [])
    return {'cited': cited, 'grounded': [c for c in cited if c in present], 'ungrounded': [c for c in cited if c not in present]}


# Reviewer adjudication, transcribed from STEP_REVIEW_LOG.md "Step 8 "before" half adjudication". Every quote must
# appear verbatim in the log (test_failure_labeller checks it). 'modes' lists the failure_mode labels the reviewer's
# text supports; None where the reviewer agreed with a fail without naming a mechanism.
_Q_PASS = '14/14 v5.1 passes CONFIRMED (A 4004–4010; B 4000, 4002, 4003, 4006, 4008, 4010, 4011)'
_Q_FLAG = '4 agree-fail (A 4001, 4002, 4011; B 4001)'
_Q_A4000 = ('In A 4000, P0 regrasped after a slip, carried the banana dangling, opened at 193 while holding it, '
            'and it dropped into the bowl and was still 203–243.')
_Q_UNFL = 'A 4003 and B 4009 agree-fail.'
_Q_IK = ('B 4004, 4005 and 4007 are correctly classified ik_action_A_to_B, all at :282, with joint 6 stuck at its '
         'local lower bound (−2.09 rad).')
_Q_4006 = 'B 4006 is correctly classified B_joint_beyond_soft_limit at :298, after the win.'


def reviewer_reference():
    r = {}
    for n in range(4004, 4011):
        r[f'step8_A_P0_l{n}_s0_v1'] = ('success', ['success'], _Q_PASS)
    for n in (4000, 4002, 4003, 4008, 4010, 4011):
        r[f'step8_B_P0_l{n}_s0_v1'] = ('success', ['success'], _Q_PASS)
    r['step8_B_P0_l4006_s0_v1'] = ('success', ['success', 'soft_limit_abort'], _Q_4006)
    for c in ('A_P0_l4001', 'A_P0_l4002', 'A_P0_l4011', 'B_P0_l4001'):
        r[f'step8_{c}_s0_v1'] = ('failure', None, _Q_FLAG)
    r['step8_A_P0_l4000_s0_v1'] = ('success', ['success'], _Q_A4000)
    for c in ('A_P0_l4003', 'B_P0_l4009'):
        r[f'step8_{c}_s0_v1'] = ('failure', None, _Q_UNFL)
    for n in (4004, 4005, 4007):
        r[f'step8_B_P0_l{n}_s0_v1'] = ('failure', ['ik_stuck_at_joint_limit'], _Q_IK)
    return {k: {'outcome': o, 'modes': m, 'quote': q} for k, (o, m, q) in r.items()}


def reviewer_source():
    return ref(REVIEW_LOG)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', metavar='CASE', required=True)
    ap.add_argument('--arm', default='blind', choices=['blind', 'with_v51'])
    a = ap.parse_args(argv)
    text, refs = summary(a.summary, a.arm)
    print(text)
    print(f'\n[{len(text)} chars, ~{len(text) // 4}-{len(text) // 3} tokens; sha256 {text_sha(text)[:12]}; refs '
          f"{[r['file'] for r in refs]}]")


if __name__ == '__main__':
    main()
