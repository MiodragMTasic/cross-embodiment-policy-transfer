# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES (portions)
# SPDX-FileCopyrightText: Copyright 2026 Miodrag Tasic (modifications)
# SPDX-License-Identifier: Apache-2.0
# Portions based on NVIDIA RoboLab examples/run_kinova_jointpos.py (commit b6cfcf6ccd2113c4cd1216d35b33024a7d5691da),
# Copyright NVIDIA CORPORATION, licensed under Apache-2.0; modified.
"""Bounded native-B mechanics/camera inspection; no policy, training or task claim.

Based on pinned RoboLab examples/run_kinova_jointpos.py. Upstream stays intact.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path.home() / 'robot-transfer/bowl'
REPO = ROOT / 'RoboLab'
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import cv2  # must precede Isaac imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--case', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
assert args.case.startswith('step6_B_mechanics_') and all(c.isalnum() or c == '_' for c in args.case)
output = ROOT / 'runs' / args.case
output.mkdir(parents=True, exist_ok=False)
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
from robolab.constants import set_output_dir
from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams
from robolab.core.sensors.contact_sensor_utils import get_contact_sensor_with_order
from robolab.core.utils.video_utils import VideoWriter
from robolab.registrations.kinova.auto_env_registrations_jointpos import auto_register_kinova_envs
from robolab.robots.kinova_gen3 import ARM_JOINT_NAMES, GRIPPER_JOINT_COMMANDS, _to_torch


def numpy(value):
    return _to_torch(value).detach().cpu().numpy().copy()


def main():
    set_output_dir(str(output))
    auto_register_kinova_envs(task='BananaInBowlTask')
    names = get_envs(task='BananaInBowlTask')
    assert len(names) == 1, names
    env, cfg = create_env(names[0], device=args.device, seed=0, num_envs=1, use_fabric=True)
    video = VideoWriter(str(output/'three_views.mp4'), 15)
    rows = []
    complete = False
    try:
        obs, _ = env.reset()
        robot = env.scene['robot']
        joints = list(robot.data.joint_names)
        arm_indices = [joints.index(n) for n in ARM_JOINT_NAMES]
        term = env.action_manager.get_term('arm')
        resolved_names = [joints[i] for i in term._joint_ids]
        assert resolved_names == ARM_JOINT_NAMES, resolved_names
        assert env.action_manager.total_action_dim == 8
        assert np.isclose(env.step_dt, 1/15)
        assert cfg.scene.robot.spawn.usd_path.endswith('kinova_gen3_7dof_robotiq_2f85.usd')
        home = _to_torch(robot.data.default_joint_pos)[0, arm_indices].clone()
        root_initial = numpy(robot.data.root_state_w[0, :7])
        limits = numpy(robot.data.soft_joint_pos_limits)[0]
        sensor_info = {}
        for name in ('gripper', 'bowl', 'table'):
            sensor, reversed_order = get_contact_sensor_with_order(env.scene, 'banana', name)
            assert sensor.data.force_matrix_w is not None, name
            matrix = sensor.data.force_matrix_w
            assert matrix.ndim == 4 and matrix.shape[0] == 1 and matrix.shape[-1] == 3, (name, matrix.shape)
            assert matrix.shape[1] > 0 and matrix.shape[2] == 1, (name, matrix.shape)
            assert len(sensor.body_names) == matrix.shape[1], (name, sensor.body_names, matrix.shape)
            sensor_info[name] = {'reversed': bool(reversed_order),
                'matrix_shape': list(sensor.data.force_matrix_w.shape),
                'body_names': list(sensor.body_names),
                'prim_path': sensor.cfg.prim_path,
                'filter_paths': sensor.cfg.filter_prim_paths_expr}
        frame_names = list(env.scene['frames'].data.target_frame_names)
        assert frame_names.count('eef_frame') == 1, frame_names
        eef_index = frame_names.index('eef_frame')
        inventory = {'purpose': 'Native Kinova mechanics and image inspection; no policy or placement attempt',
            'environment': names[0], 'seed': 0, 'joint_names': joints, 'arm_indices': arm_indices,
            'action_joint_names': resolved_names, 'body_names': list(robot.data.body_names),
            'soft_joint_limits': limits.tolist(), 'default_arm': home.cpu().tolist(),
            'robot_initial_root': root_initial.tolist(), 'contact_sensors': sensor_info,
            'gripper_close_targets': GRIPPER_JOINT_COMMANDS,
            'thresholds': {'gripper_endpoint_error_rad':.005,'arm_endpoint_error_rad':.02,
                           'joint_limit_margin_rad':.03,'fixed_root_position_drift_m':1e-4},
            'robot_asset':cfg.scene.robot.spawn.usd_path,
            'scene_asset':cfg.scene.scene.spawn.usd_path,
            'table_fixture':cfg.scene.table_fixture.to_dict(),
            'wrist_camera':cfg.scene.wrist_cam.to_dict(),
            'frame_configuration':cfg.scene.frames.to_dict(),
            'contact_limitation':'Native gripper regex may select multiple bodies. Nonempty tensor is not proof of valid filtering. No contact-scored policy run until per-finger positive/negative controls pass.',
            'frame_names': frame_names, 'eef_index': eef_index,
            'step_dt': float(env.step_dt), 'planned_steps': 180,
            'object_initial_states': {n:numpy(env.scene[n].data.root_state_w[0]).tolist() for n in ('banana','bowl','table')}}
        (output/'inventory.json').write_text(json.dumps(inventory, indent=2, default=str)+'\n')
        checks = {}
        for step in range(180):
            target = home.clone()
            if 120 <= step < 150:
                target[0] += .05
                target[5] -= .05
            close = 1. if 30 <= step < 75 else 0.
            action = torch.cat((target, torch.tensor([close], device=env.device))).unsqueeze(0)
            assert bool(torch.isfinite(action).all())
            pre = numpy(robot.data.joint_pos)[0]
            obs, _, terminated, truncated, _ = env.step(action)
            assert not bool(terminated.any()) and not bool(truncated.any()), 'Unexpected reset/termination'
            assert int(env.episode_length_buf[0]) == step+1
            q = numpy(robot.data.joint_pos)[0]
            qd = numpy(robot.data.joint_vel)[0]
            root = numpy(robot.data.root_state_w)[0]
            assert np.isfinite(q).all() and np.isfinite(qd).all() and np.isfinite(root).all()
            assert np.all(q >= limits[:,0]-.03) and np.all(q <= limits[:,1]+.03), 'Joint-limit excursion'
            assert np.linalg.norm(root[:3]-root_initial[:3]) < 1e-4, 'Fixed base moved'
            forces = {}
            for name in ('gripper', 'bowl', 'table'):
                sensor, _ = get_contact_sensor_with_order(env.scene, 'banana', name)
                forces[name] = numpy(sensor.data.force_matrix_w)[0]
                assert np.isfinite(forces[name]).all()
            images = unpack_image_obs(obs)
            viewport = unpack_viewport_cams(obs)['combined_image']
            views = [images['over_shoulder_left_camera'], images['wrist_cam'], viewport]
            panels = []
            for label, image in zip(('EXTERIOR','WRIST','VIEWPORT'), views):
                assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
                assert image.max() > image.min(), label+' is constant'
                frame = cv2.resize(image, (640,360))
                cv2.putText(frame, f'{label} | step {step+1} | command {close:.0f}', (12,24),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1, cv2.LINE_AA)
                panels.append(frame)
            video.write(cv2.hconcat(panels))
            frames = env.scene['frames'].data
            eef_position = numpy(frames.target_pos_w)[0,eef_index]
            eef_quaternion = numpy(frames.target_quat_w)[0,eef_index]
            assert np.isfinite(eef_position).all() and np.isfinite(eef_quaternion).all()
            assert abs(np.linalg.norm(eef_quaternion)-1.) < 1e-4, eef_quaternion
            root_quaternion = root[3:7] / np.linalg.norm(root[3:7])
            initial_quaternion = root_initial[3:7] / np.linalg.norm(root_initial[3:7])
            root_orientation_drift = 2*np.arccos(np.clip(abs(np.dot(root_quaternion, initial_quaternion)), 0., 1.))
            rows.append({'step': step+1, 'action': action.cpu().numpy()[0], 'pre_q':pre,
                'post_q':q,'post_qd':qd,'root':root,
                'root_orientation_drift_rad':root_orientation_drift,
                'eef_position_world':eef_position,
                'eef_quaternion_wxyz':eef_quaternion,
                'gripper_force':forces['gripper'], 'bowl_force':forces['bowl'], 'table_force':forces['table']})
            if step in (74,119,149,179):
                grip_error = max(abs(q[joints.index(n)]-(t if close else 0.)) for n,t in GRIPPER_JOINT_COMMANDS.items())
                arm_error = float(np.max(np.abs(q[arm_indices]-target.cpu().numpy())))
                checks[str(step+1)] = {'gripper_command':close,'max_gripper_error_rad':float(grip_error),
                                      'max_arm_error_rad':arm_error}
                assert grip_error <= .005, checks[str(step+1)]
                assert arm_error <= .02, checks[str(step+1)]
        complete = True
        (output/'checks.json').write_text(json.dumps({'checks':checks,'complete':True,'steps':180,
            'max_root_orientation_drift_rad':float(max(r['root_orientation_drift_rad'] for r in rows)),
            'scope':'Tracking/open-close only; no grasp, policy or bowl success established'},indent=2)+'\n')
    finally:
        if rows:
            arrays = {k:np.stack([r[k] for r in rows]) for k in rows[0]}
            np.savez_compressed(output/'mechanics.npz', complete=np.asarray(complete), **arrays)
        video.release()
        env.close()
        app.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        app.close()
        sys.exit(1)
