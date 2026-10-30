"""Approach-axis (canonical +X) direction over saved runs, in each robot's root frame."""
import sys, numpy as np
ev = sys.argv[1]
runs = {'A replay of A2002': ('replay_A_v1/step6_A_recorded_replay_v1', 0),
        'B replay (15Hz)': ('replay_B_v1/step6_B_recorded_replay_v1', 2),
        'B replay 12x': ('temporal_B_v1/step6_B_temporal_replay_v1', 2),
        'B live P0 v2 (15Hz)': ('P0_B_v2/step6_B_P0_v2', 2),
        'B live temporal_v1 (h=8)': ('P0_B_temporal_v1/step6_B_P0_temporal_v1', 2),
        'B live feedback_v1 (h=1)': ('P0_B_feedback_v1/step6_B_P0_feedback_v1', 2)}
for name, (p, col) in runs.items():
    d = np.load(f'{ev}/{p}/tracking.npz')
    key = 'tool_root' if 'tool_root' in d else [k for k in d if 'tool' in k][0]
    T = d[key]
    a = T[:, :3, col]                      # approach axis in root
    down = np.degrees(np.arccos(np.clip(-a[:, 2], -1, 1)))   # 0 = pointing straight down
    horiz = np.degrees(np.arctan2(a[:, 1], a[:, 0]))
    pos = T[:, :3, 3]
    i = np.argmax(down)
    print(f'{name:28s} rows {len(T):5d} key {key:9s} tilt-from-down median {np.median(down):5.1f} max {down.max():5.1f} deg '
          f'(at row {i}, approach xy-heading {horiz[i]:+.0f} deg, raw-tool pos {np.round(pos[i],3)}); final tilt {down[-1]:.1f}')
