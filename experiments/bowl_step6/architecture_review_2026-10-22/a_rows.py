import glob, json, sys
import h5py, numpy as np
src, ev = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
from rigid_chain import RigidChain
A = RigidChain(json.load(open(f'{ev}/bowl-step6/kinematics_export_v1/joint_graphs.json'))['A'])
tilt = lambda q: round(float(np.degrees(np.arccos(np.clip(-A.fk(np.asarray(q, float))[:3, 0][2], -1, 1)))), 1)
for p in sorted(glob.glob(f'{ev}/bowl-controlled-v3/step5_controlled_base_s*_v3/*/synchronized.hdf5')):
    h = h5py.File(p, 'r'); a = h['action.executed_joint_gripper'][:]; fr = np.flatnonzero(h['fresh_model_request'][:])[:6]
    print(p.split('/')[-3], 'request rows', fr.tolist(), 'first-action tilt', [tilt(a[i, :7]) for i in fr])
