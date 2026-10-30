"""Exploratory: where does P0's action-8 target sit relative to the banana, A/A vs B-exterior, in world frame."""
import json,sys
import h5py,numpy as np
ev,graphs,src=sys.argv[1:4];sys.path.insert(0,src)
from rigid_chain import RigidChain
A=RigidChain(json.load(open(graphs))['A'])
with h5py.File(f'{ev}/step6_B_shadow_query_v1/responses.hdf5') as f:
    jp=f['action.joint_position'][:];row=f['meta.row'][:];cond=np.array([c.decode() for c in f['meta.condition'][:]])
cap=h5py.File(f'{ev}/step6_B_shadow_capture_v1/captures.hdf5')
print('row | gripper->banana xy now | A/A target->banana xy | B-ext target->banana xy | B/B target->banana xy  (mm, median of 8)')
for r in (8,16,24,32,40,48,56,64,72):
    g=cap[str(r)];W=g['A2002_canonical_world'][:]@np.linalg.inv(A.fk(g['A2002_joint_position'][:].astype(float)))
    banana=g['objects_banana_bowl_table'][0,:3]
    def dist(c):
        ix=np.flatnonzero((row==r)&(cond==c));p=np.array([(W@A.fk(jp[i,7].astype(float)))[:3,3] for i in ix])
        return 1e3*np.median(np.linalg.norm(p[:,:2]-banana[:2],axis=1))
    now=1e3*np.linalg.norm(g['A2002_canonical_world'][:2,3]-banana[:2])
    print(f'{r:3d} | {now:8.0f} | {dist("stateA_imgA"):8.0f} | {dist("stateA_extB"):8.0f} | {dist("stateB_imgB"):8.0f}')
