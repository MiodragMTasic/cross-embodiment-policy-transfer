"""Geometry-derived frame/camera proposal and bounded CPU transfer feasibility.

No simulator, policy, training, collision-free claim or learned calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rigid_chain import RigidChain, rigid_inverse
from local_ik import physical_limits, pose_error, solve


parser = argparse.ArgumentParser()
parser.add_argument('--evidence', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
inputs = {}


def read(relative):
    path = args.evidence / relative
    inputs[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def write(name, value):
    (args.output / name).write_text(json.dumps(value, indent=2) + '\n')


graphs = read('kinematics_export_v1/joint_graphs.json')
chains = {r: RigidChain(graphs[r]) for r in ('A', 'B')}
limits = {r: physical_limits(graphs[r]) for r in ('A', 'B')}
inventories, snapshots, frames, measurements = {}, {}, {}, {}
for robot, case in [('A', 'tracking_A_v3/step6_A_tracking_v3'),
                    ('B', 'tracking_B_v1/step6_B_tracking_v1')]:
    inventories[robot] = read(case + '/inventory.json')
    snapshots[robot] = read(case + '/snapshots.json')
    geometry = read(case + '/geometry.json')
    # Canonical +X points out of the fingers; +Y points toward the left finger.
    # +Z completes a right-handed frame. Signs follow measured named geometry.
    rotation = np.eye(3) if robot == 'A' else np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    assert np.linalg.det(rotation) == 1
    measurements[robot] = {}
    for phase in ('settle', 'close', 'reopen'):
        measurements[robot][phase] = {}
        for body, meshes in geometry.items():
            if robot == 'A':
                selected = [m for m in meshes if 'fingertipsstep' in m['path']]
            else:
                selected = meshes
            assert len(selected) == 1
            mesh = selected[0]
            vertices = np.asarray(mesh['body_vertices'])
            transform = np.asarray(snapshots[robot][phase]['tip_to_raw'][body])
            raw = (np.c_[vertices, np.ones(len(vertices))] @ transform.T)[:, :3]
            canonical_axes = raw @ rotation
            lo, hi = canonical_axes.min(axis=0), canonical_axes.max(axis=0)
            assert lo[0] > .09 and hi[0] < .18
            assert (lo[1] > 0) if 'left' in body else (hi[1] < 0)
            measurements[robot][phase][body] = {'mesh': mesh['path'], 'vertices': len(vertices),
                'min_in_canonical_axes_m': lo.tolist(), 'max_in_canonical_axes_m': hi.tolist()}
    # Fixed origin: midpoint of the two distal pad planes when fully open,
    # on the mechanical symmetry axis. Not adjusted during closure or fitted to success.
    tip_depth = float(np.mean([m['max_in_canonical_axes_m'][0]
                              for m in measurements[robot]['settle'].values()]))
    frame = np.eye(4)
    frame[:3, :3] = rotation
    frame[:3, 3] = rotation @ np.array([tip_depth, 0., 0.])
    rigid_inverse(frame)
    frames[robot] = frame

a_camera = np.asarray(snapshots['A']['settle']['wrist_to_raw_opengl'])
b_camera = frames['B'] @ rigid_inverse(frames['A']) @ a_camera
quat_xyzw = Rotation.from_matrix(b_camera[:3, :3]).as_quat()
a_camera_cfg = inventories['A']['camera_cfg']['wrist_cam']
a_spawn = a_camera_cfg['spawn']
a_k = np.asarray(snapshots['A']['settle']['wrist_k'])
assert np.allclose(np.diag(a_k)[:2], [
    a_camera_cfg['width'] * a_spawn['focal_length'] / a_spawn['horizontal_aperture'],
    a_camera_cfg['height'] * a_spawn['focal_length'] / a_spawn['vertical_aperture']], atol=1e-4)
proposal = {
    'schema': 'step6-fixed-frame-proposal-v1',
    'canonical_axes': '+X distal approach; +Y toward left finger; +Z right-handed cross product.',
    'origin': 'Open-state mean distal pad plane on mechanical symmetry axis; fixed during closure.',
    'geometry_scope': 'Authored mesh inventory, not cooked collision shape validation.',
    'matrix_convention': 'Column vectors: p_raw = raw_T_canonical @ p_canonical; p_raw = raw_T_camera @ p_camera.',
    'raw_T_canonical': {r: frames[r].tolist() for r in frames},
    'A_raw_T_camera_opengl': a_camera.tolist(),
    'B_raw_T_camera_opengl': b_camera.tolist(),
    'B_camera_cfg': {'prim_path': '{ENV_REGEX_NS}/robot/robotiq_85_base_link/wrist_cam',
        'offset': {'pos': b_camera[:3, 3].tolist(),
                   'rot': quat_xyzw[[3, 0, 1, 2]].tolist(), 'convention': 'opengl'},
        'width': a_camera_cfg['width'], 'height': a_camera_cfg['height'],
        'focal_length': a_spawn['focal_length'],
        'horizontal_aperture': a_spawn['horizontal_aperture'],
        'vertical_aperture': a_spawn['vertical_aperture']},
    'expected_camera_K': a_k.tolist(),
    'camera_scope': 'Explicit simulated remount and intrinsics match, part of fixed interface; no image synthesis.',
    'root_world': {r: inventories[r]['root_initial'] for r in frames},
    'input_sha256': inputs,
}
write('frame_proposal.json', proposal)
write('geometry_measurements.json', measurements)

# Algebraic checks at each saved phase: mapping preserves the same canonical
# pose and camera world pose. Inverse uses the distinct robot-root transforms.
checks = []
for source, dest in [('A', 'B'), ('B', 'A')]:
    src_cam = a_camera if source == 'A' else b_camera
    dst_cam = a_camera if dest == 'A' else b_camera
    for phase, snapshot in snapshots[source].items():
        src_world = np.asarray(snapshot['raw_tool_world'])
        dst_world = src_world @ frames[source] @ rigid_inverse(frames[dest])
        back = dst_world @ frames[dest] @ rigid_inverse(frames[source])
        frame_error = float(np.max(np.abs(src_world - back)))
        camera_error = float(np.max(np.abs(src_world @ src_cam - dst_world @ dst_cam)))
        assert frame_error < 1e-10 and camera_error < 1e-10
        checks.append({'source': source, 'phase': phase, 'roundtrip_max': frame_error,
                       'matched_camera_max': camera_error})
exterior_error = float(np.max(np.abs(np.asarray(snapshots['A']['settle']['exterior_world_opengl']) -
                                         snapshots['B']['settle']['exterior_world_opengl'])))
assert exterior_error < 1e-6
assert np.allclose(snapshots['A']['settle']['exterior_k'], snapshots['B']['settle']['exterior_k'], atol=1e-6)
write('algebra_checks.json', {'cases': checks, 'exterior_pose_difference': exterior_error})

# Declare the feasibility bank before solving. This existing A trajectory was
# used for measured FK checks; it is development/training-route case 2002,
# never held-out validation/final/FGH data. No parameter fitting occurs.
trajectory_path = args.evidence / 'recorded_fk_v1/pose_comparisons.npz'
trajectory_sha = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
trajectory = np.load(trajectory_path)['A_q']
assert trajectory.shape == (301, 7)
a_root = np.asarray(inventories['A']['root_initial'])
b_root = np.asarray(inventories['B']['root_initial'])
bank = [{'name': 'matched_A_home', 'A_q': inventories['A']['default_arm']}]
bank += [{'name': f'A_development_row_{i}', 'A_q': trajectory[i].tolist()} for i in range(0, 301, 10)]
for item in bank:
    world_canonical = a_root @ chains['A'].fk(item['A_q']) @ frames['A']
    item['B_raw_target_root'] = (rigid_inverse(b_root) @ world_canonical @ rigid_inverse(frames['B'])).tolist()
write('ik_bank.json', {'cases': bank, 'trajectory_sha256': trajectory_sha,
    'seed_policy': 'Native B home for first; previous successful B solution thereafter. Failed solves do not update seed.',
    'cap': '200 iterations per case, no retries; at most 32 forward and 32 reverse solves.',
    'scope': 'CPU reachability diagnosis only. No motion, policy, fitting or collision-free-path proof.'})
seed_b = np.asarray(inventories['B']['default_arm'])
seed_a = np.asarray(inventories['A']['default_arm'])
results = []
for item in bank:
    forward = solve(chains['B'], np.asarray(item['B_raw_target_root']), seed_b, limits['B'])
    row = {'name': item['name'], 'forward': forward}
    if forward['success']:
        seed_b = np.asarray(forward['q'])
        a_target = (rigid_inverse(a_root) @ b_root @ chains['B'].fk(seed_b) @ frames['B'] @ rigid_inverse(frames['A']))
        reverse = solve(chains['A'], a_target, seed_a, limits['A'])
        row['reverse'] = reverse
        if reverse['success']:
            seed_a = np.asarray(reverse['q'])
        # Compare canonical poses independently of the reverse IK branch.
        a_world_c = a_root @ chains['A'].fk(item['A_q']) @ frames['A']
        b_world_c = b_root @ chains['B'].fk(seed_b) @ frames['B']
        row['canonical_residual'] = pose_error(a_world_c, b_world_c).tolist()
    results.append(row)
    write('ik_results.json', results)
write('summary.json', {'complete': True, 'cases': len(bank),
    'forward_success': sum(r['forward']['success'] for r in results),
    'reverse_attempted': sum('reverse' in r for r in results),
    'reverse_success': sum(r.get('reverse', {}).get('success', False) for r in results),
    'scope': 'Geometry proposal and CPU feasibility only. Any failed solve remains explicit; no policy approval.'})
print((args.output / 'summary.json').read_text())
