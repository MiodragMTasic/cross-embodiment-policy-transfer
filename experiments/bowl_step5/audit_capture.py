"""Read-only integrity summary for a finished controlled trial (RoboLab Python)."""
import json
from pathlib import Path
import sys
import subprocess

import h5py
import numpy as np

directory = Path(sys.argv[1])
metric = json.loads((directory / 'strict-metric.json').read_text())
trace = metric.pop('trace')
assert trace and [x['step'] for x in trace] == list(range(1, len(trace) + 1))
with h5py.File(directory / 'synchronized.hdf5', 'r') as capture:
    assert capture.attrs['complete']
    assert all(len(capture[k]) == len(trace) for k in capture)
    assert capture['executed_verified'][:].all()
    actions = capture['action.executed_joint_gripper'][:]
    assert np.isfinite(actions).all()
    np.testing.assert_array_equal(actions[:, -1], [x['command'] for x in trace])
    assert all(x['pre']['finite'] and x['post']['finite'] for x in trace)
    for key in capture:
        if key.startswith(('state.', 'action.')):
            assert np.isfinite(capture[key][:]).all(), key
    metric['capture_integrity'] = {
        'complete': True,
        'equal_stream_lengths': len(trace),
        'execution_verified': True,
        'command_trace_exact': True,
        'all_numeric_states_actions_finite': True,
        'fresh_model_requests': int(capture['fresh_model_request'][:].sum()),
    }
metric['trace_rows'] = len(trace)
metric['raw_predicate_true_rows'] = sum(x['upstream_predicate'] for x in trace)
metric['simulated_seconds'] = trace[-1]['simulation_s']
metric['passed'] = metric['stable_success_s'] is not None
assert bool(trace[-1]['stable_success']) == metric['passed']
assert len(trace) <= 750 and (metric['passed'] or len(trace) == 750)
videos = []
for video in sorted(directory.glob('*.mp4')):
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames,avg_frame_rate', '-of', 'json', str(video)]))
    stream = probe['streams'][0]
    frames = int(stream['nb_read_frames'])
    num, den = map(int, stream['avg_frame_rate'].split('/'))
    assert frames == len(trace) and num / den == 15, (video, frames, stream)
    videos.append({'file': video.name, 'frames': frames, 'fps': num / den})
assert len(videos) == 2, videos
metric['videos'] = videos
print(json.dumps(metric, indent=2))
