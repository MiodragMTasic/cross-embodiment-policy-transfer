"""Intent demo v1 checker: evaluates a spec's success_check on one saved episode, offline and read-only.

Traces are loaded with the locked scorer's own loader, release_gate_v5_1.load(), imported unchanged from
experiments/scoring_v5 (bytecode writing is turned off so nothing lands in the locked folders):
  A  RoboLab/output/<case>/<task>/strict-metric.json   rows with pre/post samples, including oriented bounding boxes
  B  runs/<case>/gate_trace.jsonl beside tracking.npz  rows with pre/post contact and support flags, no boxes
As in the locked scorer, a B gate trace one row longer than tracking.npz (a :298 abort) loses that last row.

Relations, evaluated on the final row's post sample (and on the first row's pre sample, for satisfied_at_start):
  in        banana supported by the bowl alone (bowl_support, no table contact) and inside it by the task's own predicate
            (upstream_predicate; geometric_bowl_predicate at the start), or resting on the rim when rim_counts is true.
            This is the end-state test of release_gate_v5_1.replay()'s end_on_bowl, with rim_counts made optional.
  on        banana on the bowl: supported by the bowl alone, inside or on the rim. Bowl on banana (A only, approximate):
            banana-bowl contact, the bowl's lowest corner 2 cm above the banana's, and overlapping footprints.
  left_of   A only. Frame: origin at the bowl's centre; axes of the robot base, x forward from the base, y to the robot's
  right_of  left (base yaw from initial_poses_world['robot'], identity in every capture checked). d = banana centre minus
            bowl centre in that frame. left_of: d_y > 0 and d_y >= |d_x| (within 45 deg of the robot's left axis),
            the banana on the table (table contact) and not supported by the bowl, and gap <= max_gap_m. right_of mirrors.
  beside    A only. gap <= max_gap_m, table contact, not supported by the bowl and not resting on its rim.
  gap       distance from the bowl centre to the banana's footprint (convex hull of its box corners on the table plane)
            minus the bowl's rim radius (half its box's horizontal side, 0.081 m); negative when the footprints overlap.
  camera    reported, never used for a verdict: the image column of the banana and bowl centres in the exterior camera
            GR00T sees (over_shoulder_left_camera, pinhole from env_cfg.json). Checked against frame 0 of the saved videos
            in validation/. Image left and right agree with the robot frame's left and right inside the 45 deg cone.

End conditions, each applied only when the spec asks for it:
  require_rest               every row in the final 1 s at or under 0.03 m/s and 0.3 rad/s (the v4/v5.1 StableWindow
                             limits, <= as in bowl_step5/stability.py), on the recorded speed channels. The rates derived
                             from consecutive post poses are reported beside them (the recorded angular channel spikes).
  require_released           no gripper-banana contact in the final 1 s
  require_commanded_release  banana in/on bowl: the locked v5.1 verdict (score()['pass']: a commanded release while held
                             and lifted, landing on the bowl, first stable rest). Other relations: at least one eligible
                             open in v5.1's replay (a commanded open of a held, lifted banana): the gripper was commanded
                             open during the episode.
holds = the relation at the end, every required condition, and the relation NOT already true at the start.

The demo floor, applied to every demo run whatever the spec asks: commanded_release (as above), released, and not true
at the start. demo_holds = holds and the floor. Rest is reported (with both rates), never part of the floor.
Also reported for banana relations: no gripper contact after the last eligible open, once the banana was first released.

  python checker.py --spec specs/x.json <strict-metric.json | gate_trace.jsonl> ...
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.dont_write_bytecode = True  # the locked modules below must not get __pycache__ writes from us
for _d in (PROJECT / 'experiments/scoring_v5', PROJECT / 'experiments/bowl_step5'):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
import release_gate_v5_1 as v51  # noqa: E402  locked, read-only

REST_S, LINEAR_LIMIT, ANGULAR_LIMIT = 1.0, .03, .3
ON_LIFT_M = .02
CAMERA = {'name': 'over_shoulder_left_camera', 'pos': (0.05, 0.57, 0.66), 'rot_wxyz': (-0.393, -0.195, 0.399, 0.805),
          'focal_length': 2.1, 'horizontal_aperture': 5.376, 'width': 1280, 'convention': 'opengl',
          'source': 'env_cfg.json scene.over_shoulder_left_camera (policies/gr00t/client.py:222 feeds it to GR00T)'}
GEOMETRIC = ('left_of', 'right_of', 'beside')


def quat_matrix(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def pixel_x(p, cam=CAMERA):
    """Image column (native width) of an env-local point. OpenGL camera: looks along -Z, image right is +X."""
    r = quat_matrix(*cam['rot_wxyz'])
    d = np.asarray(p, float) - np.asarray(cam['pos'], float)
    return cam['width'] / 2 + cam['focal_length'] / cam['horizontal_aperture'] * cam['width'] * (d @ r[:, 0]) / (d @ -r[:, 2])


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _hull(points):
    pts = sorted(map(tuple, points))
    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and _cross(np.subtract(out[-1], out[-2]), np.subtract(p, out[-2])) <= 0:
                out.pop()
            out.append(p)
        return out
    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def distance_to_polygon(p, poly):
    """0 inside the convex polygon (counter-clockwise), else the distance to its nearest edge."""
    p = np.asarray(p, float)
    inside = all(_cross(poly[(i + 1) % len(poly)] - poly[i], p - poly[i]) >= 0 for i in range(len(poly)))
    if inside:
        return 0.
    best = math.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        t = np.clip((p - a) @ (b - a) / max((b - a) @ (b - a), 1e-12), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * (b - a)))))
    return best


def box_sides(corners):
    """The three edge vectors at the corner nearest corners[0] (an oriented box's 8 corners, any order)."""
    c = np.asarray(corners, float)
    d = np.linalg.norm(c - c[0], axis=1)
    return [c[i] - c[0] for i in np.argsort(d)[1:4]]


def geometry(sample, base_yaw):
    """Bowl centre and radius, banana centre and footprint gap, the robot-frame offset and the camera columns."""
    banana = np.asarray(sample['boxes_env_local']['banana'], float)
    bowl = np.asarray(sample['boxes_env_local']['bowl'], float)
    centre = bowl.mean(0)
    flat = sorted(box_sides(bowl), key=lambda v: abs(v[2]) / max(np.linalg.norm(v), 1e-12))[:2]
    radius = float(np.mean([np.linalg.norm(v[:2]) for v in flat]) / 2)
    gap = distance_to_polygon(centre[:2], _hull(banana[:, :2])) - radius  # -radius when the bowl centre is under the banana
    c, s = math.cos(-base_yaw), math.sin(-base_yaw)
    off = banana.mean(0)[:2] - centre[:2]
    dx, dy = c * off[0] - s * off[1], s * off[0] + c * off[1]
    return {'bowl_centre': centre.round(4).tolist(), 'bowl_radius_m': round(radius, 4), 'banana_centre': banana.mean(0).round(4).tolist(),
            'dx_forward_m': round(float(dx), 4), 'dy_left_m': round(float(dy), 4),
            'bearing_deg': round(math.degrees(math.atan2(dy, dx)), 1), 'gap_m': round(float(gap), 4),
            'bowl_bottom_z': float(bowl[:, 2].min()), 'banana_bottom_z': float(banana[:, 2].min()),
            'camera_px': {'banana': round(float(pixel_x(banana.mean(0))), 1), 'bowl': round(float(pixel_x(centre)), 1)}}


def relation(rel, obj, ref, s, inside, check, g):
    """(holds, why) for one sample s; g is geometry(s) or None when the trace has no boxes."""
    bowl_alone = bool(s['bowl_support']) and not s['table_contact']
    if obj == 'banana' and ref == 'bowl':
        if rel == 'in':
            return bowl_alone and (bool(inside) or (check['rim_counts'] and bool(s['rim_support']))), 'support flags'
        if rel == 'on':
            return bowl_alone, 'support flags'
        if rel in GEOMETRIC:
            if g is None:
                return None, 'no bounding boxes in this trace (B): geometric relations need A'
            on_table = bool(s['table_contact']) and not s['bowl_support']
            near = g['gap_m'] <= check['max_gap_m']
            if rel == 'beside':
                return on_table and not s['rim_support'] and near, 'gap and support'
            side = g['dy_left_m'] if rel == 'left_of' else -g['dy_left_m']
            cone = side > 0 and side >= abs(g['dx_forward_m'])  # within 45 deg of the robot's left (right) axis
            return on_table and near and cone, 'robot-frame bearing, gap and support'
    if obj == 'bowl' and ref == 'banana' and rel in ('on', 'in'):
        if g is None:
            return None, 'no bounding boxes in this trace (B)'
        lifted = g['bowl_bottom_z'] >= g['banana_bottom_z'] + ON_LIFT_M
        return bool(s['bowl_contact']) and lifted and g['gap_m'] < 0, 'approximate: contact, height and footprint overlap'
    return None, f'no rule for {obj} {rel} {ref}'


def load(path):
    robot, rows, where, extend = v51.load(path)
    trimmed = robot == 'B' and len(rows) == len(where) + 1
    if trimmed:  # locked scorer rule P1: the aborted control's gate row has no tracking row
        rows = rows[:-1]
    base_yaw = 0.
    if robot == 'A':
        w, x, y, z = json.loads(Path(path).read_text())['initial_poses_world']['robot'][3:7]
        base_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return robot, rows, extend, trimmed, base_yaw


def end_window(rows):
    t_end = rows[-1]['simulation_s']
    window = [r for r in rows if r['simulation_s'] >= t_end - REST_S - 1e-9]
    dt = rows[-1]['simulation_s'] - rows[-2]['simulation_s'] if len(rows) > 1 else math.inf
    covered = t_end - window[0]['simulation_s'] >= REST_S - dt / 2
    return window, covered


def calm(r):
    """One row inside the v4/v5.1 StableWindow limits (inclusive, finite), on the recorded channels."""
    lin, ang = r['linear_m_s'], r['angular_rad_s']
    return math.isfinite(lin) and math.isfinite(ang) and lin <= LINEAR_LIMIT and ang <= ANGULAR_LIMIT


def pose_rates(rows, window):
    """Max linear (m/s) and angular (rad/s) banana rates over the window, from consecutive post poses (A only)."""
    idx = [i for i, r in enumerate(rows) if r in window and i > 0]
    if not idx or 'poses_env_local' not in rows[idx[0]]['post']:
        return None, None
    lin, ang = [], []
    for i in idx:
        (p0, q0), (p1, q1) = rows[i - 1]['post']['poses_env_local']['banana'], rows[i]['post']['poses_env_local']['banana']
        dt = rows[i]['simulation_s'] - rows[i - 1]['simulation_s']
        dot = min(1., abs(sum(a * b for a, b in zip(q0, q1))) / (math.sqrt(sum(a * a for a in q0)) * math.sqrt(sum(b * b for b in q1))))
        lin.append(math.dist(p0, p1) / dt)
        ang.append(2 * math.acos(dot) / dt)
    return round(max(lin), 4), round(max(ang), 4)


def no_recontact(rows, extend, eligible_steps):
    """After the last eligible open, once the gripper first loses the banana, it never touches it again.
    None when there was no eligible open."""
    start = next((i for i, r in enumerate(rows) if eligible_steps and r['step'] == eligible_steps[-1]), None)
    if start is None:   # no eligible open, or rows that are not the file's own (the validation's mirrored copies)
        return None
    contact = [bool(extend(r['post'])['contact']) for r in rows[start:]]
    if all(contact):
        return False   # never let go
    first_free = contact.index(False)
    return not any(contact[first_free:])


def evaluate(path, check, robot_rows=None, alongside=True):
    """Verdict for one trace and one success_check (spec['success_check']). alongside=False skips the v5.1 replay unless
    the check itself needs it."""
    path = Path(path)
    robot, rows, extend, trimmed, base_yaw = robot_rows or load(path)
    rel, obj, ref = check['relation'], check['object'], check['reference']
    out = {'trace': str(path), 'robot': robot, 'rows': len(rows), 'trimmed_abort_row': trimmed, 'success_check': check}
    if rel == 'none':
        return out | {'holds': None, 'floor_holds': None, 'demo_holds': None, 'why': 'relation none: nothing to check'}
    if len(rows) < 3:
        return out | {'holds': False, 'floor_holds': None, 'demo_holds': False, 'why': 'empty trace (under 3 rows)'}
    end, start = extend(rows[-1]['post']), extend(rows[0]['pre'])
    g_end = geometry(end, base_yaw) if 'boxes_env_local' in end else None
    g_start = geometry(start, base_yaw) if 'boxes_env_local' in start else None
    at_end, why = relation(rel, obj, ref, end, rows[-1]['upstream_predicate'], check, g_end)
    at_start, _ = relation(rel, obj, ref, start, start['geometric_bowl_predicate'], check, g_start)
    window, covered = end_window(rows)
    cond = {'rest': covered and all(calm(r) for r in window),
            'released': not any(extend(r['post'])['contact'] for r in window)}
    pose_lin, pose_ang = pose_rates(rows, window)
    rest_detail = {'limits': {'linear_m_s': LINEAR_LIMIT, 'angular_rad_s': ANGULAR_LIMIT, 'inclusive': True},
                   'window_covered': covered, 'recorded_max_linear_m_s': round(max(r['linear_m_s'] for r in window), 4),
                   'recorded_max_angular_rad_s': round(max(r['angular_rad_s'] for r in window), 4),
                   'pose_max_linear_m_s': pose_lin, 'pose_max_angular_rad_s': pose_ang}
    v, recontact = None, None
    if obj == 'banana' and not (robot == 'B' and trimmed) and (alongside or check['require_commanded_release']):
        sc = v51.score(path)
        v = {'pass': sc['pass'], 'pass_at_end': sc['pass_at_end'], 'end_on_bowl': sc['end_on_bowl'],
             'fail_reason': sc['fail_reason'], 'eligible_opens': sc['eligible_opens']}
        cond['commanded_release'] = sc['pass'] if (ref == 'bowl' and rel in ('in', 'on')) else bool(sc['eligible_opens'])
        recontact = no_recontact(rows, extend, sc['eligible_opens'])
    required = {'rest': check['require_rest'], 'released': check['require_released'],
                'commanded_release': check['require_commanded_release']}
    missing = [k for k, need in required.items() if need and k not in cond]
    holds = None if at_end is None or missing else bool(at_end and all(cond[k] for k, need in required.items() if need))
    if holds and at_start:
        holds = False   # already true before the robot moved: not a success
    floor = {'commanded_release': cond.get('commanded_release'), 'released': cond['released'],
             'not_true_at_start': None if at_start is None else not at_start}
    floor_holds = None if None in floor.values() else all(floor.values())
    demo_holds = None if holds is None or floor_holds is None else bool(holds and floor_holds)
    end_flags = {k: end[k] for k in ('bowl_support', 'rim_support', 'table_contact', 'bowl_contact', 'contact')}
    return out | {'holds': holds, 'relation_at_end': at_end, 'why': why, 'conditions': cond,
                  'not_evaluable': missing, 'satisfied_at_start': at_start, 'floor': floor, 'floor_holds': floor_holds,
                  'demo_holds': demo_holds, 'rest_detail': rest_detail, 'no_recontact_after_last_open': recontact,
                  'end_flags': end_flags, 'end_upstream_predicate': rows[-1]['upstream_predicate'],
                  'geometry_end': g_end, 'geometry_start': g_start, 'v5_1_alongside': v}


def describe_end(path, robot_rows=None):
    """Where the banana ended, in plain words, for a verdict card. Same samples and frame as evaluate()."""
    robot, rows, extend, trimmed, base_yaw = robot_rows or load(path)
    s = extend(rows[-1]['post'])
    if s['contact']:
        return 'in the gripper'
    if s['bowl_support'] and not s['table_contact']:
        return 'in the bowl' if rows[-1]['upstream_predicate'] else 'on the bowl rim'
    if s['bowl_support']:
        return 'straddling the bowl rim and the table'
    if not s['table_contact']:
        return 'off the table'
    if 'boxes_env_local' not in s:
        return 'on the table'
    g = geometry(s, base_yaw)
    dx, dy = g['dx_forward_m'], g['dy_left_m']
    side = ('left of' if dy >= abs(dx) else 'right of' if -dy >= abs(dx) else 'beyond' if dx > 0 else 'in front of')
    return f'on the table, {side} the bowl'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spec', required=True, help='a spec file from intent_to_spec.py (its success_check is used)')
    ap.add_argument('traces', nargs='+')
    a = ap.parse_args(argv)
    check = json.loads(Path(a.spec).read_text())['success_check']
    print(json.dumps([evaluate(t, check) for t in a.traces], indent=1))


if __name__ == '__main__':
    main()
