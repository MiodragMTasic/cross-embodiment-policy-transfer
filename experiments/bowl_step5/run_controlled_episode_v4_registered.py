"""External RoboLab evaluation instrumentation; pinned upstream stays unchanged.

Run with RoboLab's Python and its normal policies/gr00t/run.py arguments.
Only the success termination and logging are changed. No policy input is added.
"""
import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path.home() / 'robot-transfer/bowl'
REPO = ROOT / 'RoboLab'
sys.path.insert(0, str(REPO))
os.chdir(REPO)

# The upstream entry point parses its normal arguments and starts Isaac Sim.
spec = importlib.util.spec_from_file_location('step5_upstream_entry', REPO / 'policies/gr00t/run.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

import h5py
import numpy as np
import torch
from release_gate_v4 import ReleaseGateV4 as ReleaseGate
from rim_support import support_signals
from robolab.core.world.world_state import get_world
from robolab.core.sensors.contact_sensor_utils import get_contact_sensor_with_order
from robolab.core.task.conditionals import object_in_container
from policies.gr00t.client import GR00TDroidJointposClient, _as_action_chunk, _get_action_value
from robolab.constants import get_output_dir
from robolab.core.environments import runtime
from robolab.eval import episode as upstream_episode
from robolab.eval.runner import run_evaluation

TRACE = []
CAPTURE = None
ORIGINAL_SUCCESS = None
ORIGINAL_SUCCESS_PARAMS = None
GATE = ReleaseGate()
PRE_SAMPLE = None
INITIAL_POSES = None
RESOLVED_ENV = None


def release_signals(env):
    world = get_world(env)
    boxes = {n: world.get_bbox(n)[0][0].detach().cpu().numpy() for n in ('banana', 'bowl', 'table')}
    sensors = {n: get_contact_sensor_with_order(env.scene, 'banana', n) for n in ('gripper', 'bowl', 'table')}
    forces = {n: sensor.data.force_matrix_w[0].detach().cpu().numpy() for n, (sensor, _) in sensors.items()}
    support = support_signals(boxes['banana'], boxes['bowl'], forces['bowl'], sensors['bowl'][1])
    idx = env.scene['robot'].data.joint_names.index('finger_joint')
    closure = float(env.scene['robot'].data.joint_pos[0, idx].item()) / (np.pi/4)
    poses = {n: [x.detach().cpu().tolist() for x in world.get_pose(n, env_id=0)]
             for n in ('banana', 'bowl', 'table')}
    finite = bool(all(np.isfinite(v).all() for v in list(boxes.values()) + list(forces.values()))
                  and np.isfinite(closure) and all(np.isfinite(x).all() for v in poses.values() for x in v))
    contacts = {n: bool(np.any(np.abs(f) > .1)) for n, f in forces.items()}
    clearance = float(boxes['banana'][:, 2].min() - boxes['table'][:, 2].max())
    height_above_bowl = float(boxes['banana'][:, 2].mean() - boxes['bowl'][:, 2].max())
    geometric = bool(object_in_container(env, object='banana', container='bowl',
                      require_contact_with=False, require_gripper_detached=False, env_id=0)) if finite else False
    return {**support, 'finite': finite, 'contact': contacts['gripper'], 'table_contact': contacts['table'],
            'bowl_contact': contacts['bowl'], 'closure': closure, 'clearance_m': clearance,
            'near_bowl': geometric and height_above_bowl <= .15,
            'geometric_bowl_predicate': geometric, 'centroid_height_above_bowl_m': height_above_bowl,
            'boxes_env_local': {k: v.tolist() for k,v in boxes.items()},
            'forces_world': {k:v.tolist() for k,v in forces.items()},
            'force_sensor_reversed': {k: bool(v[1]) for k,v in sensors.items()}, 'poses_env_local': poses}


def stable_success(env):
    if PRE_SAMPLE is None:
        assert int(env.episode_length_buf[0].item()) == 0
        return torch.tensor([False], dtype=torch.bool, device=env.device)
    raw = ORIGINAL_SUCCESS(env, **ORIGINAL_SUCCESS_PARAMS)
    assert env.num_envs == 1
    obj = env.scene['banana']
    linear = float(torch.linalg.vector_norm(obj.data.root_lin_vel_w[0]).item())
    angular = float(torch.linalg.vector_norm(obj.data.root_ang_vel_w[0]).item())
    step = int(env.episode_length_buf[0].item())
    t = step * float(env.step_dt)
    post = release_signals(env)
    passed = GATE.after(t, post, bool(raw[0].item()), linear, angular)
    TRACE.append({'step':step, 'simulation_s':t, 'upstream_predicate':bool(raw[0].item()),
                  'linear_m_s':linear, 'angular_rad_s':angular, 'command':GATE.command,
                  'pre':PRE_SAMPLE, 'post':post, 'release_status':GATE.status(), 'stable_success':passed})
    return torch.tensor([passed], dtype=torch.bool, device=env.device)


original_parse = runtime.parse_env_cfg
def parse_with_metric(*args, **kwargs):
    global ORIGINAL_SUCCESS, ORIGINAL_SUCCESS_PARAMS
    cfg = original_parse(*args, **kwargs)
    if entry.args_cli.task == ['BananaInBowlUniformInitPose10cmTask']:
        reset = cfg.events.randomize_init_pose
        assert reset.mode == 'reset'
        assert reset.func.__module__ == 'robolab.core.events.reset_pose'
        assert reset.func.__name__ == 'reset_pose_uniform'
        assert reset.params['asset_cfg'] == ['banana', 'bowl']
        assert reset.params['pose_range'] == {'x': (-.1, .1), 'y': (-.1, .1), 'z': (0., 0.)}
        assert reset.params['use_collision_check'] is True
        assert reset.params['reset_to_default_otherwise'] is True
        assert cfg.episode_length_s == 50
        print('VERIFIED_RELOCATED_RESET_CONFIG', reset.params, flush=True)
    ORIGINAL_SUCCESS = cfg.terminations.success.func
    ORIGINAL_SUCCESS_PARAMS = dict(cfg.terminations.success.params)
    cfg.terminations.success.func = stable_success
    cfg.terminations.success.params = {}
    return cfg
runtime.parse_env_cfg = parse_with_metric


def append_row(key, value):
    arr = np.asarray(value)
    if key not in CAPTURE:
        kwargs = {'compression': 'lzf', 'chunks': (1, *arr.shape)} if arr.ndim >= 2 else {}
        CAPTURE.create_dataset(key, shape=(0, *arr.shape), maxshape=(None, *arr.shape), dtype=arr.dtype, **kwargs)
    ds = CAPTURE[key]
    ds.resize(ds.shape[0] + 1, axis=0)
    ds[-1] = arr


class RecordingClient(GR00TDroidJointposClient):
    def _unpack_response(self, response):
        self._aux_eef = _as_action_chunk(_get_action_value(response[0], 'eef_9d'), name='eef_9d').copy()
        return super()._unpack_response(response)

    def infer(self, obs, instruction, *, env_id=0):
        assert env_id == 0
        refreshed = self._needs_refresh(env_id)
        # No stepping occurs between this pre-action read and upstream inference.
        packed = self._pack_request(self._extract_observation(obs, env_id=env_id), instruction)
        ret = super().infer(obs, instruction, env_id=env_id)
        row = self._counters[env_id] - 1
        for key, value in packed.items():
            if key.startswith(('video.', 'state.')):
                append_row(key, value[0, 0])
        append_row('action.executed_joint_gripper', ret['action'])
        append_row('action.auxiliary_predicted_eef_9d', self._aux_eef[row])
        append_row('fresh_model_request', np.bool_(refreshed))
        append_row('chunk_row', np.int32(row))
        CAPTURE.attrs['instruction'] = packed['annotation.language.language_instruction'][0]
        return ret


# Fix the previously reviewed recording-only omission of the terminal frame.
# Compile a copy in memory; preserve the upstream source on disk.
episode_source = inspect.getsource(upstream_episode)
old = 'if env._frozen_envs[env_id]:'
assert episode_source.count(old) == 1
episode_source = episode_source.replace(old, 'if env_id not in active_env_ids:')
episode_copy = types.ModuleType('step5_episode_copy')
exec(compile(episode_source, str(Path(__file__).with_name('episode_recording_copy.py')), 'exec'), episode_copy.__dict__)


def run_episode(**kwargs):
    global CAPTURE, INITIAL_POSES, GATE, PRE_SAMPLE, TRACE
    env = kwargs['env']
    assert env.num_envs == 1 and kwargs['episode'] == 0
    TRACE = []
    GATE = ReleaseGate()
    PRE_SAMPLE = None
    INITIAL_POSES = None
    output = Path(get_output_dir())
    assert not (output / 'synchronized.hdf5').exists(), 'Never overwrite a captured run'
    CAPTURE = h5py.File(output / 'synchronized.hdf5', 'x')
    CAPTURE.attrs.update(schema='step5-pre-action-controlled-release-v4', step_dt=float(env.step_dt),
                         layout_seed=entry.args_cli.seed, policy_seed=int(os.environ.get('POLICY_SEED', '0')),
                         task_class=','.join(entry.args_cli.task),
                         requested_task_dirs=','.join(entry.args_cli.task_dirs),
                         registered_environment=RESOLVED_ENV,
                         eef_label='auxiliary P0 prediction, not executed or achieved EEF target')
    actual_step = env.step

    def checked_step(action):
        global INITIAL_POSES, PRE_SAMPLE
        assert tuple(action.shape) == (1, 8), tuple(action.shape)
        assert bool(torch.isfinite(action).all().item()), 'Nonfinite action: do not execute'
        PRE_SAMPLE = release_signals(env)
        GATE.before(int(env.episode_length_buf[0].item()) * float(env.step_dt),
                    float(action[0, -1].item()), PRE_SAMPLE)
        if INITIAL_POSES is None:
            INITIAL_POSES = {name: env.scene[name].data.root_state_w[0].detach().cpu().tolist()
                             for name in ('banana', 'bowl', 'robot')}
        saved = CAPTURE['action.executed_joint_gripper'][-1]
        np.testing.assert_array_equal(saved, action[0].detach().cpu().numpy())
        result = actual_step(action)
        append_row('executed_verified', np.bool_(True))
        CAPTURE.flush()
        return result

    env.step = checked_step
    try:
        result = episode_copy.run_episode(**kwargs)
        CAPTURE.attrs['complete'] = True
        return result
    finally:
        env.step = actual_step
        CAPTURE.close()
        CAPTURE = None
        raw = next((x for x in TRACE if x['upstream_predicate']), None)
        strict = next((x for x in TRACE if x['stable_success']), None)
        (output / 'strict-metric.json').write_text(json.dumps({
            'schema': 'command-linked-supported-release-v4',
            'release_events': GATE.events,
            'lift_duration_s': .2, 'lift_clearance_m': .01, 'opening_timeout_s': .5,
            'open_closure_threshold': .2, 'minimum_opening': .05, 'release_height_cap_m': .15,
            'duration_s': 1., 'linear_limit_m_s': .03, 'angular_limit_rad_s': .3,
            'support_vertical_force_exclusive_N': .1, 'support_cone_inclusive_deg': 45.,
            'rim_height_tolerance_m': .005, 'footprint_projection_overlap_exclusive_m': 1e-6,
            'initial_poses_world': INITIAL_POSES,
            'layout_seed': entry.args_cli.seed, 'policy_seed': int(os.environ.get('POLICY_SEED', '0')),
            'task_class': entry.args_cli.task,
            'requested_task_dirs': entry.args_cli.task_dirs,
            'registered_environment': RESOLVED_ENV,
            'task_source': str(REPO / 'robolab/tasks/randomize_initial_pose/banana_in_bowl_uniform_10cm.py'),
            'first_upstream_predicate_s': raw['simulation_s'] if raw else None,
            'stable_success_s': strict['simulation_s'] if strict else None,
            'trace': TRACE,
        }, indent=2) + '\n')


try:
    evaluation_args = copy.copy(entry.args_cli)
    if entry.args_cli.task == ['BananaInBowlUniformInitPose10cmTask']:
        assert entry.args_cli.task_dirs == ['randomize_initial_pose']
        from robolab.core.environments.factory import get_global_env_factory, get_envs
        registered = get_envs(task='BananaInBowlTask')
        assert registered == ['BananaInBowlUniformInitPose10cmTask'], registered
        info = get_global_env_factory().get_env_info(registered[0])
        assert info['task_file_name'] == 'BananaInBowlUniformInitPose10cmTask'
        assert info['config_class'].__name__ == 'BananaInBowlUniformInitPose10cmTaskEnvCfg'
        RESOLVED_ENV = registered[0]
        evaluation_args.task = ['BananaInBowlTask']
        print('VERIFIED_TASK_ALIAS', entry.args_cli.task, registered, flush=True)
    else:
        raise ValueError('This registered launcher is restricted to the approved 10cm variant.')
    
    run_evaluation(evaluation_args, policy='gr00t',
                   client_factory=lambda args: RecordingClient(remote_host=args.remote_host,
                         remote_port=args.remote_port, open_loop_horizon=args.open_loop_horizon),
                   episode_runner=run_episode)
finally:
    entry.simulation_app.close()
