"""First-request comparison: same scene/home, A vs B. Does P0's tilt diverge before B has moved?"""
import glob, json, sys
import h5py, numpy as np
src, ev = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
from rigid_chain import RigidChain
g = json.load(open(f'{ev}/bowl-step6/kinematics_export_v1/joint_graphs.json'))
A = RigidChain(g['A'])
tilt = lambda q: float(np.degrees(np.arccos(np.clip(-A.fk(np.asarray(q, float))[:3, 0][2], -1, 1))))
for p in sorted(glob.glob(f'{ev}/bowl-controlled-v3/step5_controlled_base_s*_v3/*/synchronized.hdf5')):
    h = h5py.File(p, 'r'); s = h['state.joint_position'][:]; a = h['action.executed_joint_gripper'][:]
    print(f'A {p.split("/")[-3]:30s} state0 {np.round(s[0],3)} grip0 {h["state.gripper_position"][0]}')
    print('   commanded tilt rows 0..39 step 8:', [round(tilt(a[i, :7]), 1) for i in range(0, 40, 8)], ' row 60/90/120:', [round(tilt(a[i, :7]), 1) for i in (60, 90, 120)])
for name in ('P0_B_v2/step6_B_P0_v2', 'P0_B_temporal_v1/step6_B_P0_temporal_v1', 'P0_B_feedback_v1/step6_B_P0_feedback_v1'):
    base = f'{ev}/bowl-step6/{name}'
    d = np.load(f'{base}/tracking.npz'); r0 = np.load(sorted(glob.glob(f'{base}/responses/*.npz'))[0])['joint_position']
    print(f'B {name.split("/")[1]:30s} virtualA0 {np.round(d["virtual_A_q"][0],3)} grip0 {d["gripper_observation"][0]}')
    print('   FIRST response chunk tilt rows 0..39 step 8:', [round(tilt(r0[i]), 1) for i in range(0, 40, 8)])
    rs = sorted(glob.glob(f'{base}/responses/*.npz'))[:6]
    print('   first-action tilt of first 6 responses:', [round(tilt(np.load(f)['joint_position'][0]), 1) for f in rs])
