"""Read-only CPU check: does the feedback_v1 failed target have an in-limit B solution on another branch?
Uses the unchanged sealed solver and tolerances; varies only the seed."""
import json, sys
import numpy as np
src, ev = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
from rigid_chain import RigidChain
from local_ik import solve, physical_limits
g = json.load(open(f'{ev}/kinematics_export_v1/joint_graphs.json'))
B = RigidChain(g['B']); lim = physical_limits(g['B'])
f = json.load(open(f'{ev}/P0_B_feedback_v1/step6_B_P0_feedback_v1/failed_solve.json'))
target = np.array(f['target']); seed0 = np.array(f['seed'])
home = np.load(f'{ev}/P0_B_feedback_v1/step6_B_P0_feedback_v1/tracking.npz')['pre_q'][0][:7]
rng = np.random.default_rng(0)
finite = np.where(np.isfinite(lim), lim, np.c_[-np.pi*np.ones(7), np.pi*np.ones(7)])
seeds = [('measured_seed', seed0), ('episode_start', home)]
seeds += [(f'rand{i}', rng.uniform(finite[:,0]*0.9, finite[:,1]*0.9)) for i in range(400)]
ok = []
for name, s in seeds:
    r = solve(B, target, s, lim)
    if r['success']:
        q = np.array(r['q']); m = np.min(np.abs(lim[[1,3,5]]) - np.abs(q[[1,3,5]]))
        ok.append((name, m, q, np.linalg.norm(np.angle(np.exp(1j*(q-seed0))))))
    elif name in ('measured_seed', 'episode_start'):
        print(name, 'FAIL', r['reason'], r['position_error_m'], r['rotation_error_rad'], 'j6', r['q'][5])
print('converged seeds', len(ok), 'of', len(seeds))
ok.sort(key=lambda t: -t[1])
for name, m, q, d in ok[:3]:
    print(name, 'min bounded-joint margin %.3f rad' % m, 'j6 %.3f' % q[5], 'dist-from-seed %.2f rad' % d)
near = sorted(ok, key=lambda t: t[3])[:3]
for name, m, q, d in near:
    print('nearest', name, 'margin %.3f' % m, 'j6 %.3f' % q[5], 'dist %.2f' % d, np.round(q, 2))
