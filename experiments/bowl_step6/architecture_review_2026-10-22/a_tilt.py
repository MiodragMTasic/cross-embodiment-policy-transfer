"""Gripper tilt-from-vertical (canonical approach axis) for every saved A episode vs B live runs."""
import glob, json, sys
import h5py, numpy as np
src, ev = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
from rigid_chain import RigidChain
g = json.load(open(f'{ev}/bowl-step6/kinematics_export_v1/joint_graphs.json'))
A = RigidChain(g['A'])
def tilt(approach): return np.degrees(np.arccos(np.clip(-approach[:, 2], -1, 1)))
def stats(t): return 'median %5.1f  p90 %5.1f  max %5.1f' % (np.median(t), np.percentile(t, 90), t.max())
print('A episodes (FK of measured joints, unchanged P0 on Franka):')
for p in sorted(glob.glob(f'{ev}/bowl-*/*/*/synchronized.hdf5')):
    q = h5py.File(p, 'r')['state.joint_position'][:]
    t = tilt(np.stack([A.fk(x.astype(float))[:3, 0] for x in q]))
    print(f'  {p.split("/")[-3]:42s} rows {len(q):4d}  {stats(t)}')
print('B live runs (approach of measured B tool; and of P0-commanded A joint targets):')
for name in ('P0_B_v2/step6_B_P0_v2', 'P0_B_temporal_v1/step6_B_P0_temporal_v1', 'P0_B_feedback_v1/step6_B_P0_feedback_v1'):
    d = np.load(f'{ev}/bowl-step6/{name}/tracking.npz')
    tb = tilt(d['tool_root'][:, :3, 2])
    tc = tilt(np.stack([A.fk(x[:7].astype(float))[:3, 0] for x in d['P0_action']]))
    print(f'  {name.split("/")[1]:42s} rows {len(tb):4d}  measured B {stats(tb)} | P0 command {stats(tc)}')
