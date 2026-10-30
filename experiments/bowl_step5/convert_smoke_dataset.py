"""Convert synchronized NPZ to pinned GR00T's LeRobot-v2-style local dataset.

Smoke data only: auxiliary EEF labels are policy predictions, not expert targets.
Run in the GR00T environment. No downloads or uploads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = np.load(args.source, allow_pickle=False)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    meta = output / 'meta'
    meta.mkdir()
    n = len(data['action.executed_joint_gripper'])
    assert n > 40 and bool(data['complete'])
    assert np.all(data['executed_verified'])
    state = np.concatenate([data['state.eef_9d'], data['state.gripper_position'], data['state.joint_position']], axis=1)
    executed = data['action.executed_joint_gripper']
    action = np.concatenate([data['action.auxiliary_predicted_eef_9d'], executed[:, 7:8], executed[:, :7]], axis=1)
    assert state.shape == action.shape == (n, 17)
    assert np.isfinite(state).all() and np.isfinite(action).all()
    fps = round(1 / float(data['step_dt']))
    assert fps == 15
    instruction = str(data['instruction'])
    frame = pd.DataFrame({'observation.state': list(state), 'action': list(action),
                          'timestamp': np.arange(n, dtype=np.float64) / fps,
                          'frame_index': np.arange(n), 'episode_index': np.zeros(n, dtype=np.int64),
                          'index': np.arange(n), 'task_index': np.zeros(n, dtype=np.int64)})
    table = output / 'data/chunk-000/episode_000000.parquet'
    table.parent.mkdir(parents=True)
    frame.to_parquet(table, index=False)
    modalities = {}
    for modality in ('state', 'action'):
        modalities[modality] = {k: {'start': start, 'end': end}
                               for k, start, end in [('eef_9d', 0, 9), ('gripper_position', 9, 10), ('joint_position', 10, 17)]}
    modalities['video'] = {}
    modalities['annotation'] = {'language.language_instruction': {'original_key': 'task_index'}}
    features = {'observation.state': {'dtype': 'float32', 'shape': [17]},
                'action': {'dtype': 'float32', 'shape': [17]}}
    for camera in ('exterior_image_1_left', 'wrist_image_left'):
        key = f'observation.images.{camera}'
        modalities['video'][camera] = {'original_key': key}
        features[key] = {'dtype': 'video', 'shape': [180, 320, 3],
                         'names': ['height', 'width', 'channels'],
                         'video_info': {'video.fps': fps, 'video.codec': 'h264', 'video.pix_fmt': 'gbrp'}}
        frames = data[f'video.{camera}']
        assert frames.shape == (n, 180, 320, 3) and frames.dtype == np.uint8
        destination = output / f'videos/chunk-000/{key}/episode_000000.mp4'
        destination.parent.mkdir(parents=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-s', '320x180', '-r', str(fps), '-i', 'pipe:0', '-an',
                        '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'fast', str(destination)],
                       input=frames.tobytes(), check=True)
    info = {'codebase_version': 'v2.0', 'robot_type': 'droid_franka_robotiq_sim',
            'total_episodes': 1, 'total_frames': n, 'total_tasks': 1, 'total_videos': 2,
            'total_chunks': 1, 'chunks_size': 1000, 'fps': fps, 'splits': {'train': '0:1'},
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
            'features': features}
    for name, value in [('info', info), ('modality', modalities)]:
        (meta / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
    (meta / 'tasks.jsonl').write_text(json.dumps({'task_index': 0, 'task': instruction}) + '\n')
    (meta / 'episodes.jsonl').write_text(json.dumps({'episode_index': 0, 'tasks': [instruction], 'length': n}) + '\n')
    (output / 'PROVENANCE.json').write_text(json.dumps({
        'source': str(args.source), 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'purpose': 'disposable pipeline smoke only, exclude from P1/results',
        'state': 'pre-action model-format state', 'joint_gripper_labels': 'actual executed absolute targets',
        'eef_labels': 'auxiliary absolute P0 prediction, not independently verified controller targets',
        'video': 'pre-action RGB resized by stock client; lossless RGB H264',
    }, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'frames': n, 'fps': fps}))


if __name__ == '__main__':
    main()
