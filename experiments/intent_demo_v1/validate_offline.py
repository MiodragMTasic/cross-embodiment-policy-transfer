"""Offline validation of checker.py against the sealed step 8 "before" evidence (read-only), no model call, no VM.

1. "in" on all 24 episodes, compared with the locked v5.1 results in score_step8-before.json and with the reviewer's
   adjudication (nemotron/failure_labeller.reviewer_reference(), transcribed from STEP_REVIEW_LOG.md). Each trace's
   sha256 must equal the one the locked score recorded.
2. left_of / right_of / beside on the 12 A episodes, at the start and at the end, with the numbers needed to check them
   by hand, plus the camera reading compared with the banana (yellow) and bowl (red) pixel centroids detected in frame 0
   of each saved exterior video (needs ffmpeg; skipped if absent).
3. A mirror check: each A start state reflected across the bowl's forward axis must flip right_of to left_of.

  .venv/bin/python -B validate_offline.py      writes validation/checker-offline-v1.{json,md}
"""
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import checker  # noqa: E402  (also puts the locked scorer modules on sys.path, bytecode off)

PROJECT = checker.PROJECT
EVIDENCE = PROJECT / 'assets/report/evidence/2026-10-24/step8_before_v1'
LOCKED = PROJECT / 'experiments/scoring_v5/score_step8_before/score_step8-before.json'
TASK = 'BananaInBowlUniformInitPose10cmTask'
OUT = HERE / 'validation'
sys.path.insert(0, str(PROJECT / 'nemotron'))
from failure_labeller import reviewer_reference  # noqa: E402

BASE = {'object': 'banana', 'reference': 'bowl', 'rim_counts': True, 'max_gap_m': 0.05, 'require_released': True,
        'require_rest': True, 'require_commanded_release': False}
IN_END = BASE | {'relation': 'in'}                                   # final-state "in", the checker's own test
IN_END_NORIM = IN_END | {'rim_counts': False}                         # the literal reading Nemotron gave 18/18
IN_RULE = IN_END | {'require_commanded_release': True}                # plus the locked v5.1 verdict


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def trace_path(case, robot):
    if robot == 'A':
        return EVIDENCE / 'RoboLab/output' / case / TASK / 'strict-metric.json'
    return EVIDENCE / 'runs' / case / 'gate_trace.jsonl'


def frame0_centroids(case):
    """(banana_px, bowl_px) column centroids in the exterior half of frame 0, native 1280-wide scale; None if unavailable."""
    if not shutil.which('ffmpeg'):
        return None
    video = next((EVIDENCE / 'RoboLab/output' / case / TASK).glob('Pick_up_*_0.mp4'))
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-vf', r'select=eq(n\,0)', '-vframes', '1',
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    im = np.frombuffer(raw, np.uint8).reshape(360, 1280, 3)[:, :640].astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    masks = {'banana': (r > 200) & (g > 170) & (b < 80), 'bowl': (r > 90) & (g < 50) & (b < 50) & (r > 2 * g)}
    scale = checker.CAMERA['width'] / 640  # the saved video is the 640x360 policy view of the 1280x720 sensor
    return {k: round(float(np.nonzero(m)[1].mean()) * scale, 1) if m.any() else None for k, m in masks.items()}


def geometric_rows(case):
    path = trace_path(case, 'A')
    loaded = checker.load(path)
    out = {}
    for name, rel, gap in (('left_of', 'left_of', .05), ('right_of', 'right_of', .05), ('beside', 'beside', .05),
                           ('left_of_any', 'left_of', .3), ('right_of_any', 'right_of', .3)):
        r = checker.evaluate(path, BASE | {'relation': rel, 'max_gap_m': gap, 'require_rest': False, 'require_released': False},
                             loaded)
        out[name] = {'start': r['satisfied_at_start'], 'end': r['relation_at_end']}
    gs, ge = r['geometry_start'], r['geometry_end']
    return out, gs, ge, r['end_flags']


def mirrored_start(case):
    """Reflect the banana's start box across the line through the bowl centre along the robot's forward axis."""
    path = trace_path(case, 'A')
    robot, rows, extend, trimmed, yaw = checker.load(path)
    assert abs(yaw) < 1e-6, 'mirror check assumes the identity base seen in every capture'
    rows = copy.deepcopy(rows[:3])
    pre = rows[0]['pre']
    by = float(np.mean(np.asarray(pre['boxes_env_local']['bowl'])[:, 1]))
    pre['boxes_env_local']['banana'] = [[x, 2 * by - y, z] for x, y, z in pre['boxes_env_local']['banana']]
    r = checker.evaluate(path, BASE | {'relation': 'left_of', 'max_gap_m': .3, 'require_rest': False, 'require_released': False},
                         (robot, rows, extend, trimmed, yaw))
    g = r['geometry_start']
    return {'left_of_after_mirror': r['satisfied_at_start'], 'dy_left_m': g['dy_left_m'],
            'camera_banana_left_of_bowl': g['camera_px']['banana'] < g['camera_px']['bowl']}


def main():
    locked = json.loads(LOCKED.read_text())
    reviewer = reviewer_reference()
    episodes, geo = [], []
    for e in locked['episodes']:
        case, robot = e['case'], e['robot']
        path = trace_path(case, robot)
        assert sha256(path) == e['trace_sha256'], f'{case}: trace differs from the one the locked score read'
        loaded = checker.load(path)
        end = checker.evaluate(path, IN_END, loaded)
        norim = checker.evaluate(path, IN_END_NORIM, loaded)
        rule = checker.evaluate(path, IN_RULE, loaded)
        rv = reviewer[case]['outcome'] == 'success'
        episodes.append({'case': case, 'robot': robot, 'locked_v5_1_pass': e['v5_1_pass'],
                         'locked_v5_1_pass_at_end': e['v5_1_pass_at_end'], 'locked_fail_reason': e['fail_reason'],
                         'reviewer_success': rv, 'checker_in_end': end['holds'], 'checker_in_end_relation': end['relation_at_end'],
                         'checker_in_end_conditions': end['conditions'], 'checker_in_end_norim': norim['holds'],
                         'checker_in_rule': rule['holds'], 'recomputed_v5_1_pass': (end['v5_1_alongside'] or {}).get('pass'),
                         'end_flags': end['end_flags'], 'trimmed_abort_row': end['trimmed_abort_row']})
        if robot == 'A':
            rels, gs, ge, flags = geometric_rows(case)
            seen = frame0_centroids(case)
            geo.append({'case': case, 'relations': rels, 'start': gs, 'end': ge, 'end_flags': flags, 'frame0_px': seen,
                        'camera_side_agrees': None if not seen or None in seen.values() else
                        (seen['banana'] < seen['bowl']) == (gs['camera_px']['banana'] < gs['camera_px']['bowl']),
                        'mirror': mirrored_start(case)})

    def agree(key, ref, robot=None):
        xs = [x for x in episodes if robot is None or x['robot'] == robot]
        return sum(x[key] == x[ref] for x in xs), len(xs)

    summary = {}
    for key in ('checker_in_end', 'checker_in_end_norim', 'checker_in_rule'):
        summary[key] = {ref: {'all': agree(key, ref), 'A': agree(key, ref, 'A'), 'B': agree(key, ref, 'B')}
                        for ref in ('locked_v5_1_pass', 'locked_v5_1_pass_at_end', 'reviewer_success')}
    recomputed = [x for x in episodes if x['recomputed_v5_1_pass'] is not None]
    summary['recomputed_v5_1_equals_locked'] = (sum(x['recomputed_v5_1_pass'] == x['locked_v5_1_pass'] for x in recomputed),
                                                len(recomputed))
    summary['disagreements_in_end_vs_locked'] = [
        {k: x[k] for k in ('case', 'locked_v5_1_pass', 'locked_fail_reason', 'reviewer_success', 'checker_in_end',
                           'checker_in_end_relation', 'checker_in_end_conditions', 'end_flags')}
        for x in episodes if x['checker_in_end'] != x['locked_v5_1_pass']]
    summary['camera_frame0_agreement'] = (sum(bool(g['camera_side_agrees']) for g in geo),
                                          sum(g['camera_side_agrees'] is not None for g in geo))
    summary['mirror_flips_to_left_of'] = (sum(g['mirror']['left_of_after_mirror'] for g in geo), len(geo))
    OUT.mkdir(exist_ok=True)
    result = {'evidence': str(EVIDENCE.relative_to(PROJECT)), 'locked_score': {'file': str(LOCKED.relative_to(PROJECT)),
              'sha256': sha256(LOCKED)}, 'checker_sha256': sha256(HERE / 'checker.py'),
              'checks': {'in_end': IN_END, 'in_end_norim': IN_END_NORIM, 'in_rule': IN_RULE},
              'summary': summary, 'episodes': episodes, 'geometric_A': geo}
    (OUT / 'checker-offline-v1.json').write_text(json.dumps(result, indent=1) + '\n')
    (OUT / 'checker-offline-v1.md').write_text(markdown(result))
    print(json.dumps(summary, indent=1))


def markdown(r):
    s = r['summary']
    def f(t):
        return f'{t[0]}/{t[1]}'
    lines = ['# Checker offline validation v1', '',
             f"Evidence `{r['evidence']}` (read-only). Locked score `{r['locked_score']['file']}` sha256 "
             f"`{r['locked_score']['sha256'][:12]}…`. checker.py sha256 `{r['checker_sha256'][:12]}…`. Every trace's sha256 matched "
             'the one the locked score recorded.', '',
             '## "in", against the locked v5.1 results and the reviewer', '',
             '| Checker variant | vs locked v5.1 pass (first rest) | vs locked v5.1 pass_at_end | vs reviewer |', '|---|---|---|---|']
    names = {'checker_in_end': 'end state: in, rim counts, released, at rest',
             'checker_in_end_norim': 'end state, rim not counted',
             'checker_in_rule': 'end state plus locked v5.1 commanded release'}
    for k, label in names.items():
        row = s[k]
        lines.append(f'| {label} | ' + ' | '.join(f"{f(row[ref]['all'])} (A {f(row[ref]['A'])}, B {f(row[ref]['B'])})"
                                                  for ref in ('locked_v5_1_pass', 'locked_v5_1_pass_at_end', 'reviewer_success')) + ' |')
    lines += ['', f"v5.1 recomputed through the checker equals the locked pass on {f(s['recomputed_v5_1_equals_locked'])} "
              'episodes (B traces trimmed after an abort are not recomputed).', '',
              '**End-state "in" vs the locked v5.1 pass, disagreements:**', '',
              '| Case | v5.1 | v5.1 fail_reason | Reviewer | Checker | Relation at end | Rest / released | End flags |', '|---|---|---|---|---|---|---|---|']
    for d in s['disagreements_in_end_vs_locked']:
        c = d['checker_in_end_conditions']
        fl = ', '.join(k for k, v in d['end_flags'].items() if v)
        lines.append(f"| {d['case']} | {d['locked_v5_1_pass']} | {d['locked_fail_reason']} | {d['reviewer_success']} | "
                     f"{d['checker_in_end']} | {d['checker_in_end_relation']} | {c['rest']} / {c['released']} | {fl} |")
    lines += ['', '## left_of / right_of / beside on the 12 A episodes (robot frame; max_gap 0.05 m)', '',
              f"Camera side from the checker's projection matches the colour centroids in frame 0 on {f(s['camera_frame0_agreement'])}. "
              f"Mirroring each start state across the bowl's forward axis turns it into left_of (0.30 m cap) on {f(s['mirror_flips_to_left_of'])}. "
              'Columns "any" use the 0.30 m cap (side only); the others use 0.05 m.', '',
              '| Case | Start dx, dy (m) | Bearing (deg) | Gap (m) | Start: left / right / beside | Start, any cap: left / right | Camera px banana, bowl (frame 0 detected) | End dx, dy | End: left / right / beside | End flags |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for g in r['geometric_A']:
        st, en, rel = g['start'], g['end'], g['relations']
        seen = g['frame0_px'] or {}
        fl = ', '.join(k for k, v in g['end_flags'].items() if v)
        lines.append(f"| {g['case']} | {st['dx_forward_m']}, {st['dy_left_m']} | {st['bearing_deg']} | {st['gap_m']} | "
                     f"{rel['left_of']['start']} / {rel['right_of']['start']} / {rel['beside']['start']} | "
                     f"{rel['left_of_any']['start']} / {rel['right_of_any']['start']} | "
                     f"{st['camera_px']['banana']}, {st['camera_px']['bowl']} ({seen.get('banana')}, {seen.get('bowl')}) | "
                     f"{en['dx_forward_m']}, {en['dy_left_m']} | {rel['left_of']['end']} / {rel['right_of']['end']} / {rel['beside']['end']} | {fl} |")
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
