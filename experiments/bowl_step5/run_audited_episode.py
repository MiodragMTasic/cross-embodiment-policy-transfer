"""External RoboLab evaluation instrumentation; pinned upstream stays unchanged.

Run with RoboLab's Python and its normal policies/gr00t/run.py arguments.
Only the success termination and logging are changed. No policy input is added.
"""
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
from stability import StableWindow
from policies.gr00t.client import GR00TDroidJointposClient, _as_action_chunk, _get_action_value
from robolab.constants import get_output_dir
from robolab.core.environments import runtime
from robolab.eval import episode as upstream_episode
from robolab.eval.runner import run_evaluation

TRACE = []
CAPTURE = None
ORIGINAL_SUCCESS = None
ORIGINAL_SUCCESS_PARAMS = None
WINDOW = StableWindow()
INITIAL_POSES = None


def stable_success(env):
    raw = ORIGINAL_SUCCESS(env, **ORIGINAL_SUCCESS_PARAMS)
    assert env.num_envs == 1, 'This audited pilot intentionally supports one env'
    obj = env.scene['banana']
    linear = float(torch.linalg.vector_norm(obj.data.root_lin_vel_w[0]).item())
    angular = float(torch.linalg.vector_norm(obj.data.root_ang_vel_w[0]).item())
    step = int(env.episode_length_buf[0].item())
    t = step * float(env.step_dt)
    passed = WINDOW.update(t, bool(raw[0].item()), linear, angular)
    TRACE.append({'step': step, 'simulation_s': t, 'upstream_predicate': bool(raw[0].item()),
                  'linear_m_s': linear, 'angular_rad_s': angular,
                  'qualifying_since_s': WINDOW.since, 'stable_success': passed})
    return torch.tensor([passed], dtype=torch.bool, device=env.device)


original_parse = runtime.parse_env_cfg
def parse_with_metric(*args, **kwargs):
    global ORIGINAL_SUCCESS, ORIGINAL_SUCCESS_PARAMS
    cfg = original_parse(*args, **kwargs)
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
    global CAPTURE, INITIAL_POSES, WINDOW, TRACE
    env = kwargs['env']
    assert env.num_envs == 1 and kwargs['episode'] == 0
    TRACE = []
    WINDOW = StableWindow()
    output = Path(get_output_dir())
    assert not (output / 'synchronized.hdf5').exists(), 'Never overwrite a captured run'
    CAPTURE = h5py.File(output / 'synchronized.hdf5', 'x')
    CAPTURE.attrs.update(schema='step5-pre-action-v1', step_dt=float(env.step_dt),
                         layout_seed=entry.args_cli.seed, policy_seed=int(os.environ.get('POLICY_SEED', '0')),
                         task_class=','.join(entry.args_cli.task),
                         eef_label='auxiliary P0 prediction, not executed or achieved EEF target')
    actual_step = env.step

    def checked_step(action):
        global INITIAL_POSES
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
            'schema': 'stable-upstream-predicate-v1',
            'duration_s': 1., 'linear_limit_m_s': .03, 'angular_limit_rad_s': .3,
            'initial_poses_world': INITIAL_POSES,
            'layout_seed': entry.args_cli.seed, 'policy_seed': int(os.environ.get('POLICY_SEED', '0')),
            'task_class': entry.args_cli.task,
            'first_upstream_predicate_s': raw['simulation_s'] if raw else None,
            'stable_success_s': strict['simulation_s'] if strict else None,
            'trace': TRACE,
        }, indent=2) + '\n')


try:
    run_evaluation(entry.args_cli, policy='gr00t',
                   client_factory=lambda args: RecordingClient(remote_host=args.remote_host,
                         remote_port=args.remote_port, open_loop_horizon=args.open_loop_horizon),
                   episode_runner=run_episode)
finally:
    entry.simulation_app.close()
