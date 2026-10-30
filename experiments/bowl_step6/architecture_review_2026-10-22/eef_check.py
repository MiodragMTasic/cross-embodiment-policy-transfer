"""Does the bridge's eef_9d state equal what A's own pipeline feeds P0 for the same joints?"""
import glob, json, sys
import h5py, numpy as np
src, ev = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
from rigid_chain import RigidChain
g = json.load(open(f'{ev}/bowl-step6/kinematics_export_v1/joint_graphs.json'))
A = RigidChain(g['A'])
def rot_from6(r6):
    a, b = r6[:3], r6[3:6]; return np.stack([a, b, np.cross(a, b)])
d = np.load(f'{ev}/bowl-step6/P0_B_v2/step6_B_P0_v2/tracking.npz')
Rt = d['virtual_A_tool'][0][:3, :3]; M = rot_from6(d['state_eef_9d'][0][3:].astype(float))
C = Rt.T @ M                                           # recover DROID_EEF_ROTATION_CORRECT
print('recovered correction (rows):', np.round(C, 4).tolist(), 'orthonormal', np.allclose(C @ C.T, np.eye(3), atol=1e-5))
# consistency within B runs
for name in ('P0_B_v2/step6_B_P0_v2',):
    d = np.load(f'{ev}/bowl-step6/{name}/tracking.npz')
    e = [np.abs(np.r_[t[:3, 3], (t[:3, :3] @ C)[:2, :].reshape(-1)] - s).max() for t, s in zip(d['virtual_A_tool'], d['state_eef_9d'])]
    print('bridge formula self-consistency max abs', max(e))
# apply bridge formula to A's measured joints and compare with A's recorded state.eef_9d
for p in sorted(glob.glob(f'{ev}/bowl-controlled-v3/step5_controlled_base_s*_v3/*/synchronized.hdf5')) + \
         sorted(glob.glob(f'{ev}/bowl-training-feasibility-v4/step5_training_layout2002_v4/*/synchronized.hdf5')):
    h = h5py.File(p, 'r'); q = h['state.joint_position'][:].astype(float); s = h['state.eef_9d'][:].astype(float)
    pred = np.stack([np.r_[A.fk(x)[:3, 3], (A.fk(x)[:3, :3] @ C)[:2, :].reshape(-1)] for x in q])
    dp = np.linalg.norm(pred[:, :3] - s[:, :3], axis=1); dr = np.abs(pred[:, 3:] - s[:, 3:]).max(1)
    print(f'{p.split("/")[-3]:34s} pos diff mm median {1e3*np.median(dp):8.3f} max {1e3*dp.max():8.3f} | rot6 diff median {np.median(dr):.4f} max {dr.max():.4f}')
    print('   row0 recorded', np.round(s[0], 4).tolist())
    print('   row0 bridge  ', np.round(pred[0], 4).tolist())
