"""Exploratory (not pre-registered): does a B observation move P0's action-8 target pose beyond query noise?
Per row and condition: distance of each query's action-8 canonical pose from the stateA_imgA median pose,
compared with A/A queries' own distance (leave-one-out median of the other 7) as the noise floor."""
import json,sys
import h5py,numpy as np
ev,graphs,src=sys.argv[1:4];sys.path.insert(0,src)
from rigid_chain import RigidChain
A=RigidChain(json.load(open(graphs))['A'])
with h5py.File(f'{ev}/step6_B_shadow_query_v1/responses.hdf5') as f:
    jp=f['action.joint_position'][:];row=f['meta.row'][:];cond=np.array([c.decode() for c in f['meta.condition'][:]]);state=f['sent.state.joint_position'][:]
C=('stateA_imgA','stateB_imgA','stateA_wristB','stateA_extB','stateA_imgB','stateB_imgB')
P=np.array([A.fk(q[7].astype(float)) for q in jp]);S=np.array([A.fk(q.astype(float)) for q in state])
def ang(R1,R2):return np.degrees(np.arccos(np.clip((np.trace(R1.T@R2)-1)/2,-1,1)))
out={}
print('row  cond          pos_mm(med)  rot_deg(med)  | dz_mm  dxy_mm  heading_deg   (vs A/A median target; A/A row = leave-one-out noise)')
for r in (8,16,24,32,40,48,56,64,72):
    aa=np.flatnonzero((row==r)&(cond=='stateA_imgA'));ref=np.median(P[aa][:,:3,3],axis=0)
    # rotation reference: the A/A query closest to the positional median
    Rref=P[aa][np.argmin(np.linalg.norm(P[aa][:,:3,3]-ref,axis=1))][:3,:3]
    for c in C:
        ix=np.flatnonzero((row==r)&(cond==c))
        if c=='stateA_imgA':
            pos=[np.linalg.norm(P[i][:3,3]-np.median(P[[j for j in aa if j!=i]][:,:3,3],axis=0)) for i in ix]
        else:pos=[np.linalg.norm(P[i][:3,3]-ref) for i in ix]
        rot=[ang(P[i][:3,:3],Rref) for i in ix]
        d=np.median(P[ix][:,:3,3],axis=0)-ref
        # approach heading of action-8 axis in the horizontal plane
        hd=np.median([np.degrees(np.arctan2(P[i][1,0],P[i][0,0])) for i in ix])
        out.setdefault(r,{})[c]={'pos_mm':1e3*np.median(pos),'rot_deg':np.median(rot),'dz_mm':1e3*d[2],'dxy_mm':1e3*np.hypot(d[0],d[1]),'heading':hd}
        o=out[r][c];print(f'{r:3d}  {c:13s} {o["pos_mm"]:8.1f}     {o["rot_deg"]:7.1f}      | {o["dz_mm"]:6.1f} {o["dxy_mm"]:6.1f}  {hd:7.1f}')
    sp=S[aa[0]][:3,3];print(f'     sent state z {1e3*sp[2]:.0f} mm; A/A action-8 travel from state {1e3*np.linalg.norm(ref-sp):.0f} mm')
json.dump({str(r):v for r,v in out.items()},open('pose_shift.json','w'),indent=1)
