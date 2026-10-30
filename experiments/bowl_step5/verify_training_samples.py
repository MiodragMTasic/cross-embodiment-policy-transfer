"""Read-only validation of real 40-row samples through pinned training processor.

Use GR00T's environment. Does not load model weights or perform optimization.
Auxiliary EEF values remain explicitly unexecuted P0 teacher predictions.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
ROOT = Path.home() / 'robot-transfer/bowl'
sys.path.insert(0, str(ROOT / 'Isaac-GR00T'))
import numpy as np
import torch
from transformers import Qwen3VLProcessor
import checkpoint_modalities
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.types import EmbodimentTag, MessageType
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

source, dataset = map(Path, sys.argv[1:])
capture = np.load(source, allow_pickle=False)
manifest = json.loads((ROOT / 'models/model-manifest.json').read_text())
snapshots = {m['repo']: m['snapshot'] for m in manifest['models']}
p0 = Path(snapshots['nvidia/GR00T-N1.7-DROID'])
original_loader = Qwen3VLProcessor.from_pretrained
def pinned_processor(path, *args, **kwargs):
    if str(path) == 'nvidia/Cosmos-Reason2-2B':
        path = snapshots['nvidia/Cosmos-Reason2-2B']
    return original_loader(path, *args, **kwargs)
Qwen3VLProcessor.from_pretrained = staticmethod(pinned_processor)

tag = EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
cfg = MODALITY_CONFIGS[tag.value]
assert cfg['video'].delta_indices == cfg['state'].delta_indices == [0]
assert cfg['action'].delta_indices == list(range(40))
loader = LeRobotEpisodeLoader(dataset, cfg)
episode = loader[0]
n = len(episode)
assert n == len(capture['action.executed_joint_gripper']) and n >= 40
anchors = sorted({0, (n-40)//2, n-40})
processor = Gr00tN1d7Processor.from_pretrained(p0, state_dropout_prob=0.)
processor.train()
assert processor.training and processor.state_action_processor.training
assert processor.state_dropout_prob == 0 and not processor.exclude_state
assert processor.use_relative_action and processor.use_percentiles and processor.clip_outliers
assert not processor.use_mean_std and not processor.apply_sincos_state_encoding
for group in ('state', 'action'):
    assert processor.modality_configs[tag.value][group].modality_keys == ['eef_9d', 'gripper_position', 'joint_position']
assert processor.modality_configs[tag.value]['video'].modality_keys == cfg['video'].modality_keys
collator = processor.data_collator_class(model_name=snapshots['nvidia/Cosmos-Reason2-2B'])
stats = json.loads((p0 / 'statistics.json').read_text())[tag.value]

def independently_normalize(values, statistics, relative=False):
    # Pinned upstream replaces the relative-action norm dict with raw relative
    # statistics, so it uses min/max there despite use_percentiles=True.
    lo_key, hi_key = ('min', 'max') if relative else ('q01', 'q99')
    lo, hi = np.asarray(statistics[lo_key]), np.asarray(statistics[hi_key])
    result = np.zeros_like(values)
    varying = ~np.isclose(hi, lo)
    result[..., varying] = 2 * (values[..., varying]-lo[varying]) / (hi[varying]-lo[varying]) - 1
    return np.clip(result, -1, 1)

reports = []
for anchor in anchors:
    step = extract_step_data(episode, anchor, cfg, tag, allow_padding=False)
    for key in cfg['state'].modality_keys:
        np.testing.assert_array_equal(step.states[key], capture[f'state.{key}'][anchor:anchor+1])
    for key in cfg['video'].modality_keys:
        np.testing.assert_array_equal(np.asarray(step.images[key][0]), capture[f'video.{key}'][anchor])
    executed = capture['action.executed_joint_gripper'][anchor:anchor+40]
    np.testing.assert_array_equal(step.actions['joint_position'], executed[:, :7])
    np.testing.assert_array_equal(step.actions['gripper_position'], executed[:, 7:8])
    np.testing.assert_array_equal(step.actions['eef_9d'], capture['action.auxiliary_predicted_eef_9d'][anchor:anchor+40])
    inputs = processor([{'type': MessageType.EPISODE_STEP.value, 'content': step}])
    expected_joint = independently_normalize(executed[:, :7] - capture['state.joint_position'][anchor],
                                            stats['relative_action']['joint_position'], relative=True)
    expected_grip = independently_normalize(executed[:, 7:8], stats['action']['gripper_position'])
    np.testing.assert_allclose(inputs['action'][:, 10:17], expected_joint, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(inputs['action'][:, 9:10], expected_grip, atol=2e-6, rtol=2e-6)
    expected_state = independently_normalize(capture['state.joint_position'][anchor:anchor+1],
                                            stats['state']['joint_position'])
    np.testing.assert_allclose(inputs['state'][:, 10:17], expected_state, atol=2e-6, rtol=2e-6)
    assert tuple(inputs['action'].shape) == (40, 132)
    assert torch.all(inputs['action_mask'][:, :17] == 1)
    assert torch.all(inputs['action_mask'][:, 17:] == 0)
    assert torch.all(inputs['action'][:, 17:] == 0)
    assert torch.all(inputs['state'][:, 17:] == 0)
    batch = collator([inputs])['inputs']
    for k, v in batch.items():
        if isinstance(v, torch.Tensor): assert torch.isfinite(v).all(), k
    reports.append({'anchor': anchor, 'action_rows': [anchor, anchor+39],
                    'state_shape': list(batch['state'].shape), 'action_shape': list(batch['action'].shape),
                    'pixel_values_shape': list(batch['pixel_values'].shape),
                    'manual_joint_gripper_normalization_matches': True})

report = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'dataset': str(dataset), 'frames': n, 'eligible_anchors': n-39,
          'anchors_verified': reports, 'processor_training_mode': True, 'state_dropout': 0,
          'statistics': 'original P0, unchanged',
          'normalization_detail': 'Pinned upstream uses relative_action min/max for joints; q01/q99 for absolute gripper and state. Preserved, not changed.',
          'eef_labels': 'unexecuted auxiliary P0 predictions; teacher distillation, not independent corrective targets',
          'tail_limitation': 'last 39 observations excluded as anchors; their actions can appear in earlier chunks',
          'optimization_steps': 0}
report['image_verification'] = 'Source RGB pixels match before augmentation; training images intentionally cropped/augmented, then checked finite.'
report['padding'] = 'Action/state padded values and action-mask padding verified zero.'
(dataset / 'training-sample-verification.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
