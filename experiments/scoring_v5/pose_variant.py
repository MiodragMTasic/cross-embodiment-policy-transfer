"""EXPLORATORY: pose-based first rest (EXPERIMENT_REPORT §22.11, tightened). Not a step 8 scorer; the locked v5.1 stays the outcome.

python pose_variant.py [--out FILE.json] CASE_DIR [CASE_DIR ...]
  Traces outside step 8: an A case dir holds <task>/strict-metric.json, a B case dir gate_trace.jsonl and tracking.npz of equal
  length. Step 8 paths are refused in this mode.
python pose_variant.py --step8 --phase {before,after} [--headline score_step8-<phase>.json] [--out FILE.json] ROOT [ROOT ...]
  Step 8, explicit opt-in (F1): the only way to read step 8, and only step 8 cases (score_step8's discovery for those phases).
  Driven by the locked score_step8 (F2): its episode set, scored or not, and the exact trace it scored each episode on (the
  :298 row dropped, B aligned by its surplus rule) - taken at its scored() call, since score_step8 keeps that trace only in a
  TemporaryDirectory and its output records just the original path and row counts. The headline win is score_step8's
  win_control; the variant adds only the pose win on that trace, re-runs v5.1 there and reports any mismatch as a disagreement.
  Unscored stays unscored; an empty_trace fail stays a fail. --headline cross-checks against a score_step8 JSON output.

An episode wins if the locked v5.1 wins OR the pose-rate replay wins; the win control is the earlier of the two.
The pose-rate replay is the locked replay() with ReleaseGateV51, where after()'s angular-speed argument (the one the
StableWindow uses) is replaced row by row by the banana's pose-derived rate; linear speed and everything else are unchanged.
  quaternion  A: trace[i].post.poses_env_local.banana[1]; B: tracking.npz objects_banana_bowl[i, 0, 3:7] (Isaac root_state_w)
  rate        2 arccos(min(1, |q_i . q_i-1|)) / (t_i - t_i-1), unit quaternions, consecutive post times (simulation_s);
              every step must be 1/15 s within 1e-6, else the pose replay is not computable
  first row   no predecessor: a finite rate above any limit, so it never qualifies. A non-finite value would also trip the
              gate's nonfinite_post_action check, which v5.1 does not see on that row.
Not computable leaves the v5.1 result as the union result, so the union never removes a v5.1 win.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest import mock
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'bowl_step5')]
from release_gate_v5_1 import ReleaseGateV51, load, score
import score_step8

DT = 1 / 15
DT_TOLERANCE = 1e-6
FIRST_ROW = sys.float_info.max


class NotComputable(Exception):
    pass


def pose_rates(q, t):
    """Pose-derived angular rate (rad/s) per post row, from quaternions q (N, 4) and post times t (N,)."""
    q, t = np.asarray(q, float), np.asarray(t, float)
    if q.ndim != 2 or q.shape[1] != 4 or len(q) != len(t):
        raise NotComputable(f'{len(q)} quaternions for {len(t)} post times')
    norm = np.linalg.norm(q, axis=1)
    if not (np.isfinite(q).all() and (norm > 0).all()):
        raise NotComputable('non-finite or zero quaternion')
    q = q / norm[:, None]
    dt = np.diff(t)
    bad = np.flatnonzero(np.abs(dt - DT) > DT_TOLERANCE)
    if bad.size:
        raise NotComputable(f'post time step {dt[bad[0]]!r} s at row {bad[0] + 1}, not 1/15')
    dot = np.minimum(1.0, np.abs(np.einsum('ij,ij->i', q[1:], q[:-1])))
    return [FIRST_ROW, *(2 * np.arccos(dot) / dt).tolist()]


def pose_gate(rates):
    """ReleaseGateV51 whose after() gets the pose rate as its angular speed; replay() calls after() once per row, in order."""
    class PoseRateGateV51(ReleaseGateV51):
        last_row = -1

        def after(self, t, sample, raw_success, linear_speed, angular_speed):
            self.pose_row = getattr(self, 'pose_row', -1) + 1
            type(self).last_row = self.pose_row
            return super().after(t, sample, raw_success, linear_speed, rates[self.pose_row])
    return PoseRateGateV51


def quaternions(robot, trace, rows):
    if robot == 'A':
        return [r['post']['poses_env_local']['banana'][1] for r in rows]
    return np.load(trace.with_name('tracking.npz'))['objects_banana_bowl'][:, 0, 3:7]


def pose_win(trace):
    """(pose-rate replay win control or None, rows) on this exact trace; raises NotComputable."""
    robot, rows, _, _ = load(trace)
    rates = pose_rates(quaternions(robot, trace, rows), [r['simulation_s'] for r in rows])
    gate = pose_gate(rates)
    control = score(trace, gate_cls=gate)['success_control']
    assert gate.last_row == len(rows) - 1, (gate.last_row, len(rows))  # one after() call per row
    return control, len(rows)


def combine(v51, pose):
    """Union of two win controls (None = no win): the earlier wins, and a v5.1 win is never removed."""
    wins = [c for c in (v51, pose) if c is not None]
    control = min(wins) if wins else None
    by = None if control is None else 'both' if v51 == pose else 'v5.1' if control == v51 else 'pose'
    return control, by


def with_pose(v51, trace):
    """The union fields for one scored trace, given its v5.1 win control."""
    try:
        pose, _ = pose_win(trace)
        reason = None
    except NotComputable as e:
        pose, reason = None, str(e)
    control, by = combine(v51, pose)
    return {'pose_pass': pose is not None, 'pose_win_control': pose, 'not_computable': reason,
            'union_pass': control is not None, 'union_win_control': control, 'won_by': by}


def trace_of(case_dir):
    d = Path(case_dir)
    if any(part.startswith('step8') for part in d.resolve().parts):
        raise SystemExit(f'refusing a step 8 path without --step8: {d}')
    if (d / 'gate_trace.jsonl').exists():
        return 'B', d / 'gate_trace.jsonl'
    found = sorted(d.glob('*/strict-metric.json'))
    if len(found) != 1:
        raise SystemExit(f'{len(found)} strict-metric.json under {d}')
    return 'A', found[0]


def episode(case_dir):
    """A trace outside step 8. B must have as many tracking rows as gate rows; abort traces go through --step8."""
    robot, trace = trace_of(case_dir)
    out = {'case': Path(case_dir).name, 'robot': robot, 'trace': str(trace), 'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest()}
    if robot == 'B':
        gate = sum(1 for l in trace.read_text().splitlines() if l.strip())
        track = trace.with_name('tracking.npz')
        n = len(np.load(track)['objects_banana_bowl']) if track.exists() else 0
        if gate != n:
            return {**out, 'scored': False, 'unscored_reason': f'{gate} gate rows, {n} tracking rows'}
    v51 = score(trace)['success_control']
    return {**out, 'scored': True, 'v5_1_pass': v51 is not None, 'v5_1_win_control': v51, **with_pose(v51, trace)}


def step8(roots, phase, headline=None):
    """--step8: score_step8's episodes, each with the pose win added on the exact trace score_step8 scored."""
    if phase not in ('before', 'after'):
        raise SystemExit('--step8 takes --phase before or after')
    found, problems = score_step8.discover(roots, phase)
    locked, added = score_step8.scored, {}

    def scored_and_pose(rec, trace, v4, abort=None, src=None):
        out = locked(rec, trace, v4, abort, src)
        _, rows, _, _ = load(trace)
        mine = score(trace)['success_control']
        added[out['case']] = {'scored_trace_rows': len(rows), 'v5_1_recomputed': mine, **with_pose(out['win_control'], Path(trace)),
                              'disagreement': None if mine == out['win_control'] else
                              f"v5.1 recomputed {mine} but score_step8 win_control {out['win_control']}"}
        return out

    with mock.patch.object(score_step8, 'scored', scored_and_pose):
        episodes = [score_step8.episode(k, v) for k, v in sorted(found.items())]
    res = []
    for e in episodes:
        rec = {k: e.get(k) for k in ('case', 'robot', 'condition', 'layout', 'scored', 'unscored_reason', 'abort_class', 'abort_kind')}
        if e['scored']:
            rec.update(headline_pass=e['v5_1_pass'], headline_win_control=e['win_control'], safety_abort=e.get('safety_abort'))
            if e['case'] in added:
                rec.update(added[e['case']])
            else:  # only an empty_trace abort skips scored(); it stays a fail, with no pose replay
                empty = (e.get('safety_abort') or {}).get('empty_trace')
                rec.update(scored_trace_rows=(e.get('safety_abort') or {}).get('scored_rows'), **combine_fields(e['win_control']),
                           not_computable='empty_trace' if empty else None,
                           disagreement=None if empty else 'score_step8 scored this episode without a trace the variant could see')
        res.append(rec)
    per = {}
    for x in res:
        c = per.setdefault(f"{x['robot']}/{x['condition']}", {'scored': 0, 'unscored': 0, 'headline_wins': 0, 'union_wins': 0, 'added_by_pose': 0})
        if not x['scored']:
            c['unscored'] += 1
            continue
        c['scored'] += 1
        c['headline_wins'] += x['headline_pass']
        c['union_wins'] += x['union_pass']
        c['added_by_pose'] += x['union_pass'] and not x['headline_pass']
    return {'phase': phase, 'roots': [str(r) for r in roots], 'problems': problems, 'episodes': res, 'per_condition': per,
            'disagreements': [x['case'] for x in res if x.get('disagreement')],
            'headline_check': None if headline is None else check_headline(res, headline)}


def combine_fields(v51):
    control, by = combine(v51, None)
    return {'pose_pass': False, 'pose_win_control': None, 'union_pass': control is not None, 'union_win_control': control, 'won_by': by}


def check_headline(res, headline):
    """Differences from a score_step8 output (its JSON, or the dict main() returns): episode set, scored flag, win_control."""
    h = headline if isinstance(headline, dict) else json.loads(Path(headline).read_text())
    want = {e['case']: (e['scored'], e.get('win_control')) for e in h['episodes']}
    got = {e['case']: (e['scored'], e.get('headline_win_control')) for e in res}
    return [{'case': c, 'variant': got.get(c), 'score_step8': want.get(c)} for c in sorted(want.keys() | got.keys()) if got.get(c) != want.get(c)]


def main(argv=None):
    ap = argparse.ArgumentParser(description='EXPLORATORY pose-based first rest: union of locked v5.1 and the pose-rate replay')
    ap.add_argument('--out')
    ap.add_argument('--step8', action='store_true', help='read step 8 (explicit opt-in): ROOTs as for score_step8.py')
    ap.add_argument('--phase')
    ap.add_argument('--headline', help='score_step8-<phase>.json to cross-check (with --step8)')
    ap.add_argument('paths', nargs='+')
    a = ap.parse_args(argv)
    if a.step8:
        res = step8(a.paths, a.phase, a.headline)
    elif a.phase or a.headline:
        raise SystemExit('--phase and --headline need --step8')
    else:
        res = [episode(c) for c in a.paths]
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    r = main()
    for e in (r['episodes'] if isinstance(r, dict) else r):
        v51 = e.get('headline_win_control', e.get('v5_1_win_control'))
        print(f"{e['case']}: " + (f"v5.1 {v51} pose {e['pose_win_control']} union {e['union_win_control']} ({e['won_by'] or 'no win'})"
                                  + (f"  not computable: {e['not_computable']}" if e['not_computable'] else '')
                                  + (f"  DISAGREEMENT: {e['disagreement']}" if e.get('disagreement') else '')
                                  if e['scored'] else f"unscored: {e['unscored_reason']}"))
