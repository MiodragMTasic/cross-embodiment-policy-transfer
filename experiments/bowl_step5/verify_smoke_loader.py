"""Load through the actual GR00T loader and compare every frame and numeric row."""
import json
from pathlib import Path
import sys
import numpy as np
import checkpoint_modalities
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

source, dataset = map(Path, sys.argv[1:])
capture = np.load(source, allow_pickle=False)
configs = MODALITY_CONFIGS['oxe_droid_relative_eef_relative_joint']
loader = LeRobotEpisodeLoader(dataset, configs)
frame = loader[0]
for camera in configs['video'].modality_keys:
    decoded = np.stack(frame[f'video.{camera}'])
    np.testing.assert_array_equal(decoded, capture[f'video.{camera}'])
for key in configs['state'].modality_keys:
    np.testing.assert_array_equal(np.stack(frame[f'state.{key}']), capture[f'state.{key}'])
executed = capture['action.executed_joint_gripper']
np.testing.assert_array_equal(np.stack(frame['action.joint_position']), executed[:, :7])
np.testing.assert_array_equal(np.stack(frame['action.gripper_position']), executed[:, 7:8])
np.testing.assert_array_equal(np.stack(frame['action.eef_9d']), capture['action.auxiliary_predicted_eef_9d'])
assert all(frame['language.annotation.language.language_instruction'] == str(capture['instruction']))
result = {'loaded_frames': len(frame), 'video': 'all RGB pixels exact', 'state': 'all rows exact',
          'action': 'executed q/grip and explicitly auxiliary EEF exact',
          'video_indices': configs['video'].delta_indices, 'action_horizon': len(configs['action'].delta_indices),
          'relative_statistics_present': 'relative_action' in loader.get_dataset_statistics()}
(dataset / 'loader-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
