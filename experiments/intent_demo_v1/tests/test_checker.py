"""checker.py on real sealed evidence (read-only) and on synthetic A traces built from a real start sample."""
import copy
import hashlib
import json

import numpy as np
import pytest

import checker
from conftest import DEMO, a_trace, b_trace, needs_evidence

BASE = {'object': 'banana', 'reference': 'bowl', 'rim_counts': True, 'max_gap_m': 0.05, 'require_released': True,
        'require_rest': True, 'require_commanded_release': False}


def chk(relation, **over):
    return dict(BASE, relation=relation, **over)


def shifted(sample, dxy):
    s = copy.deepcopy(sample)
    s['boxes_env_local']['banana'] = [[x + dxy[0], y + dxy[1], z] for x, y, z in s['boxes_env_local']['banana']]
    p = s['poses_env_local']['banana'][0]
    s['poses_env_local']['banana'][0] = [p[0] + dxy[0], p[1] + dxy[1], p[2]]
    return s


def synthetic(tmp_path, target_xy, n=30, moving=False, held=False):
    """A 4004's real start sample (banana on the table, gripper open, no contact) with the banana moved so its centre sits at
    target_xy relative to the bowl centre (robot frame) from row 1 on; row 0 keeps the true start."""
    src = json.loads(a_trace(4004).read_text())
    start = src['trace'][0]['pre']
    bb = np.mean(start['boxes_env_local']['banana'], axis=0)[:2]
    bw = np.mean(start['boxes_env_local']['bowl'], axis=0)[:2]
    moved = shifted(start, bw + np.asarray(target_xy) - bb)
    rows = []
    for k in range(1, n + 1):
        s = start if k == 1 else moved
        post = copy.deepcopy(moved)
        post['contact'] = held and k > n - 5
        rows.append({'step': k, 'simulation_s': k / 15, 'upstream_predicate': False, 'command': 0.0,
                     'linear_m_s': .2 if moving and k > n - 3 else 0., 'angular_rad_s': 0., 'pre': copy.deepcopy(s),
                     'post': post, 'release_status': {}, 'stable_success': False})
    path = tmp_path / 'strict-metric.json'
    path.write_text(json.dumps({k: v for k, v in src.items() if k != 'trace'} | {'trace': rows}))
    return path


@needs_evidence
def test_real_A_win_is_in():
    r = checker.evaluate(a_trace(4004), chk('in', require_commanded_release=True))
    assert r['holds'] and r['conditions'] == {'rest': True, 'released': True, 'commanded_release': True}
    assert r['satisfied_at_start'] is False and r['v5_1_alongside']['pass'] is True
    assert r['geometry_start']['dy_left_m'] < -0.2  # the banana starts on the robot's right of the bowl


@needs_evidence
def test_real_A_set_down_in_bowl_without_valid_release():
    end_only = checker.evaluate(a_trace(4001), chk('in'))
    rule = checker.evaluate(a_trace(4001), chk('in', require_commanded_release=True))
    assert end_only['holds'] is True and rule['holds'] is False and rule['v5_1_alongside']['fail_reason'] == 'never_lifted'


@needs_evidence
def test_real_A_straddle_is_neither_in_nor_beside():
    for rel in ('in', 'beside', 'left_of', 'right_of'):
        r = checker.evaluate(a_trace(4003), chk(rel, max_gap_m=.3))
        assert r['relation_at_end'] is False, rel
    assert r['end_flags']['table_contact'] and r['end_flags']['bowl_support']


@needs_evidence
def test_real_B_in_and_geometric_not_evaluable():
    assert checker.evaluate(b_trace(4008), chk('in'))['holds'] is True
    r = checker.evaluate(b_trace(4008), chk('left_of'))
    assert r['holds'] is None and 'no bounding boxes' in r['why']


@needs_evidence
def test_real_B_abort_row_is_trimmed_like_the_locked_scorer():
    r = checker.evaluate(b_trace(4006), chk('in'))
    assert r['trimmed_abort_row'] and r['v5_1_alongside'] is None and r['holds'] is False


@needs_evidence
def test_left_of_close(tmp_path):
    r = checker.evaluate(synthetic(tmp_path, (0.0, 0.20)), chk('left_of'))
    assert r['holds'] and not r['satisfied_at_start']
    g = r['geometry_end']
    assert 0 < g['gap_m'] <= .05 and g['dy_left_m'] == pytest.approx(.20, abs=1e-3)
    assert g['camera_px']['banana'] < g['camera_px']['bowl']  # also left in the exterior camera image
    for rel in ('right_of', 'in', 'on'):
        assert not checker.evaluate(synthetic(tmp_path, (0.0, 0.20)), chk(rel))['relation_at_end'], rel
    assert checker.evaluate(synthetic(tmp_path, (0.0, 0.20)), chk('beside'))['holds']


@needs_evidence
def test_left_of_far_needs_the_larger_cap(tmp_path):
    p = synthetic(tmp_path, (0.0, 0.30))
    assert not checker.evaluate(p, chk('left_of'))['holds']
    assert checker.evaluate(p, chk('left_of', max_gap_m=.3))['holds']
    assert not checker.evaluate(p, chk('beside'))['holds']


@needs_evidence
def test_outside_the_45_degree_cone_is_neither_side(tmp_path):
    p = synthetic(tmp_path, (0.16, 0.06))  # mostly in front of the bowl
    for rel in ('left_of', 'right_of'):
        assert not checker.evaluate(p, chk(rel, max_gap_m=.3))['relation_at_end'], rel
    r = checker.evaluate(p, chk('beside', max_gap_m=.3))
    assert r['relation_at_end'] and r['satisfied_at_start'] and r['holds'] is False   # 0.3 m reaches the start too (F2)


@needs_evidence
def test_right_of_is_already_true_at_the_start(tmp_path):
    r = checker.evaluate(synthetic(tmp_path, (0.0, -0.20)), chk('right_of', max_gap_m=.3))
    assert r['relation_at_end'] and r['satisfied_at_start']  # every layout starts that way
    assert r['holds'] is False and r['floor']['not_true_at_start'] is False and r['demo_holds'] is False   # F2


@needs_evidence
def test_end_conditions(tmp_path):
    assert checker.evaluate(synthetic(tmp_path, (0., .20), held=True), chk('left_of'))['conditions']['released'] is False
    assert checker.evaluate(synthetic(tmp_path, (0., .20), moving=True), chk('left_of'))['conditions']['rest'] is False
    r = checker.evaluate(synthetic(tmp_path, (0., .20)), chk('left_of', require_commanded_release=True))
    assert r['holds'] is False and r['conditions']['commanded_release'] is False  # it was never held and lifted
    assert checker.evaluate(synthetic(tmp_path, (0., .20)), chk('none'))['holds'] is None


@needs_evidence
def test_floor_applies_whatever_the_spec_asks(tmp_path):
    """F2: the spec may not ask for a commanded release; the demo verdict still needs one, plus released and not at start."""
    r = checker.evaluate(synthetic(tmp_path, (0., .20)), chk('left_of', require_commanded_release=False))
    assert r['holds'] is True   # the spec's own check holds: the banana sits left of the bowl
    assert r['floor'] == {'commanded_release': False, 'released': True, 'not_true_at_start': True}
    assert r['floor_holds'] is False and r['demo_holds'] is False
    assert r['no_recontact_after_last_open'] is None   # no eligible open at all
    held = checker.evaluate(synthetic(tmp_path, (0., .20), held=True), chk('left_of', require_released=False))
    assert held['floor']['released'] is False and held['demo_holds'] is False


@needs_evidence
def test_rest_limits_are_inclusive_and_both_rates_are_reported(tmp_path):
    """P5: <= 0.03 m/s and <= 0.3 rad/s as in bowl_step5/stability.py; the pose-derived rates are reported beside."""
    path = synthetic(tmp_path, (0., .20))
    d = json.loads(path.read_text())
    for row in d['trace']:
        row['linear_m_s'], row['angular_rad_s'] = checker.LINEAR_LIMIT, checker.ANGULAR_LIMIT
    path.write_text(json.dumps(d))
    r = checker.evaluate(path, chk('left_of'))
    assert r['conditions']['rest'] is True
    assert r['rest_detail']['recorded_max_angular_rad_s'] == checker.ANGULAR_LIMIT
    assert r['rest_detail']['pose_max_angular_rad_s'] == 0 and r['rest_detail']['pose_max_linear_m_s'] == 0
    for row in d['trace']:
        row['angular_rad_s'] = checker.ANGULAR_LIMIT + 1e-6
    path.write_text(json.dumps(d))
    assert checker.evaluate(path, chk('left_of'))['conditions']['rest'] is False


@needs_evidence
def test_real_A_recontact_line():
    """P4: shown line. 4004 lets go after its open and stays free; 4011 (opening_timeout) touches the banana again."""
    assert checker.evaluate(a_trace(4004), chk('in'))['no_recontact_after_last_open'] is True
    assert checker.evaluate(a_trace(4011), chk('in'))['no_recontact_after_last_open'] is False


def test_polygon_distance_and_camera_axes():
    sq = checker._hull([[0, 0], [1, 0], [1, 1], [0, 1], [.5, .5]])
    assert len(sq) == 4
    assert checker.distance_to_polygon([.5, .5], sq) == 0 and checker.distance_to_polygon([2, .5], sq) == pytest.approx(1)
    # the camera sits on the robot's left looking across: a point further to the robot's left projects further left
    assert checker.pixel_x([0.5, 0.3, 0.03]) < checker.pixel_x([0.5, 0.0, 0.03])


def test_stored_validation_matches_this_checker():
    v = json.loads((DEMO / 'validation/checker-offline-v1.json').read_text())
    assert v['checker_sha256'] == hashlib.sha256((DEMO / 'checker.py').read_bytes()).hexdigest(), 'rerun validate_offline.py'
    s = v['summary']
    assert s['checker_in_end']['locked_v5_1_pass'] == {'all': [16, 24], 'A': [8, 12], 'B': [8, 12]}
    assert s['checker_in_rule']['locked_v5_1_pass']['A'] == [12, 12]
    assert s['recomputed_v5_1_equals_locked'] == [23, 23]
    assert s['camera_frame0_agreement'] == [12, 12] and s['mirror_flips_to_left_of'] == [12, 12]


def test_early_returns_carry_demo_keys():
    """REVIEW_2 M1: collect() reads demo_holds; the relation-none and short-trace returns must carry it."""
    import inspect
    src = inspect.getsource(checker.evaluate)
    early = [l for l in src.splitlines() if "return out | {'holds'" in l and 'relation_at_end' not in l]
    assert len(early) == 2 and all("'demo_holds'" in l and "'floor_holds'" in l for l in early)
