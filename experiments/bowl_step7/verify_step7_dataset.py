"""CPU-only loader check of a step 7 dataset against its export npz files. No model weights, no optimization.

python verify_step7_dataset.py EXPORT_DIR DATASET_DIR   (Isaac-GR00T/.venv, CUDA_VISIBLE_DEVICES='', HF offline)
Every episode, every row: the pinned LeRobotEpisodeLoader returns exactly the exported 17-D state and action and the exact RGB
frames; lengths equal the approved win control; the last row is the trace row of the win control. Then 40-row samples at
rows 0, middle and n-40 of each episode go through the pinned training processor and collator, with all 17 normalized
dimensions recomputed independently (same method as bowl_step5/verify_training_capture_all17.py).
"""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'run with CUDA_VISIBLE_DEVICES= (CPU only)'
ROOT = Path.home() / 'robot-transfer/bowl'
sys.path[:0] = [str(ROOT / 'Isaac-GR00T'), str(ROOT / 'experiments/bowl_step5')]
import numpy as np
import torch
from transformers import Qwen3VLProcessor
import checkpoint_modalities  # noqa: F401  registers P0's own modality contract
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.types import EmbodimentTag, MessageType
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

export, dataset = map(Path, sys.argv[1:3])
manifest = json.loads((export / 'export-manifest.json').read_text())
provenance = json.loads((dataset / 'PROVENANCE.json').read_text())
assert provenance['export_manifest_sha256'] == hashlib.sha256((export / 'export-manifest.json').read_bytes()).hexdigest()
models = json.loads((ROOT / 'models/model-manifest.json').read_text())
snapshots = {m['repo']: m['snapshot'] for m in models['models']}
p0 = Path(snapshots['nvidia/GR00T-N1.7-DROID'])
original_loader = Qwen3VLProcessor.from_pretrained
def pinned_processor(path, *args, **kwargs):
    if str(path) == 'nvidia/Cosmos-Reason2-2B':
        path = snapshots['nvidia/Cosmos-Reason2-2B']
    return original_loader(path, *args, **kwargs)
Qwen3VLProcessor.from_pretrained = staticmethod(pinned_processor)

tag = EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
cfg = MODALITY_CONFIGS[tag.value]
assert cfg['video'].delta_indices == cfg['state'].delta_indices == [0] and cfg['action'].delta_indices == list(range(40))
keys = ['eef_9d', 'gripper_position', 'joint_position']
assert cfg['state'].modality_keys == cfg['action'].modality_keys == keys
loader = LeRobotEpisodeLoader(dataset, cfg)
assert len(loader) == len(manifest['episodes']), 'episode count'
info = json.loads((dataset / 'meta/info.json').read_text())
assert info['total_episodes'] == len(manifest['episodes']) and info['total_frames'] == sum(e['kept_rows'] for e in manifest['episodes'])

processor = Gr00tN1d7Processor.from_pretrained(p0, state_dropout_prob=0.)
processor.train()
assert processor.use_relative_action and processor.use_percentiles and processor.clip_outliers
assert not processor.use_mean_std and not processor.apply_sincos_state_encoding
collator = processor.data_collator_class(model_name=snapshots['nvidia/Cosmos-Reason2-2B'])
stats = json.loads((p0 / 'statistics.json').read_text())[tag.value]  # P0's own normalization, unchanged


def normalize(values, statistics, relative=False):
    lo_key, hi_key = ('min', 'max') if relative else ('q01', 'q99')
    lo, hi = np.asarray(statistics[lo_key]), np.asarray(statistics[hi_key])
    result = np.zeros_like(values)
    varying = ~np.isclose(hi, lo)
    result[..., varying] = 2 * (values[..., varying] - lo[varying]) / (hi[varying] - lo[varying]) - 1
    return np.clip(result, -1, 1)


def rotation_matrix(rot6d):
    rows = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    x = rows[0] / np.linalg.norm(rows[0])
    y = rows[1] - np.dot(rows[1], x) * x
    y /= np.linalg.norm(y)
    return np.stack((x, y, np.cross(x, y)))


def relative_eef(actions, reference):
    rotation = rotation_matrix(reference[3:])
    translations = (np.asarray(actions[:, :3], dtype=np.float64) - reference[:3]) @ rotation
    rotations = np.stack([rotation.T @ rotation_matrix(a[3:]) for a in actions])
    return np.concatenate((translations, rotations[:, :2, :].reshape(-1, 6)), axis=1)


def hwc(frame):
    frame = np.asarray(frame)
    return frame.transpose(1, 2, 0) if frame.shape[0] == 3 else frame


report = []
for e in manifest['episodes']:
    i, n = e['episode_index'], e['kept_rows']
    data = np.load(export / f"{e['case']}.npz", allow_pickle=False)
    df = loader[i]
    assert len(df) == n == loader.get_episode_length(i) == e['win_control'], f"{e['case']}: length"
    state = {'eef_9d': data['state.eef_9d'], 'gripper_position': data['state.gripper_position'], 'joint_position': data['state.joint_position']}
    executed = data['action.executed_joint_gripper']
    action = {'eef_9d': data['action.auxiliary_predicted_eef_9d'], 'gripper_position': executed[:, 7:8], 'joint_position': executed[:, :7]}
    dims = 0
    for group, source in (('state', state), ('action', action)):
        for key in keys:
            loaded = np.stack(df[f'{group}.{key}'].to_numpy())
            np.testing.assert_array_equal(loaded, source[key])
            assert np.isfinite(loaded).all()
            dims += loaded.shape[1] if group == 'state' else 0
    assert dims == 17
    frames_checked = 0
    for camera in cfg['video'].modality_keys:
        decoded = np.stack([hwc(f) for f in df[f'video.{camera}']])
        np.testing.assert_array_equal(decoded, data[f'video.{camera}'])
        frames_checked += len(decoded)
    # the cut: last kept row is the trace row of the win control
    assert int(data['trace_step'][-1]) == e['win_control'] and len(data['trace_step']) == n
    last_grip = float(np.stack(df['state.gripper_position'].to_numpy())[-1, 0])
    assert abs(last_grip - float(data['trace_pre_closure'][-1])) <= 1e-6
    assert float(np.stack(df['action.gripper_position'].to_numpy())[-1, 0]) == float(data['trace_command'][-1])
    samples = []
    for anchor in sorted({0, (n - 40) // 2, n - 40}):
        step = extract_step_data(df, anchor, cfg, tag, allow_padding=False)
        inputs = processor([{'type': MessageType.EPISODE_STEP.value, 'content': step}])
        window = slice(anchor, anchor + 40)
        np.testing.assert_allclose(inputs['action'][:, :9], normalize(relative_eef(action['eef_9d'][window], state['eef_9d'][anchor]),
                                                                       stats['relative_action']['eef_9d'], relative=True), atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(inputs['action'][:, 9:10], normalize(action['gripper_position'][window], stats['action']['gripper_position']),
                                   atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(inputs['action'][:, 10:17], normalize(action['joint_position'][window] - state['joint_position'][anchor],
                                                                         stats['relative_action']['joint_position'], relative=True), atol=2e-6, rtol=2e-6)
        for key, a, b in (('eef_9d', 0, 9), ('gripper_position', 9, 10), ('joint_position', 10, 17)):
            np.testing.assert_allclose(inputs['state'][:, a:b], normalize(state[key][anchor:anchor + 1], stats['state'][key]), atol=2e-6, rtol=2e-6)
        assert tuple(inputs['action'].shape) == (40, 132)
        assert torch.all(inputs['action_mask'][:, :17] == 1) and torch.all(inputs['action_mask'][:, 17:] == 0)
        batch = collator([inputs])['inputs']
        assert all(torch.isfinite(v).all() for v in batch.values() if isinstance(v, torch.Tensor))
        samples.append(anchor)
    report.append({'case': e['case'], 'episode_index': i, 'rows': n, 'win_control': e['win_control'], 'original_rows': e['original_rows'],
                   'cut': e['cut'], 'frames_exact': frames_checked, 'all_rows_17d_exact': True, 'processor_anchors_all17': samples})
    print(json.dumps(report[-1]), flush=True)

result = {'dataset': str(dataset), 'episodes': len(report), 'frames': sum(r['rows'] for r in report), 'per_episode': report,
          'statistics': 'original P0, unchanged (processor checked against P0 statistics.json)', 'optimization_steps': 0,
          'device': 'cpu', 'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (dataset / 'step7-loader-check.json').open('x') as stream:
    stream.write(json.dumps(result, indent=1) + '\n')
print('STEP7_LOADER_CHECK_OK', result['episodes'], 'episodes', result['frames'], 'frames')
