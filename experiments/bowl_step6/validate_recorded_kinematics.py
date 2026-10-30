"""CPU-only FK vs already recorded A/B poses; finite-difference Jacobian checks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
from rigid_chain import RigidChain,pose_matrix,rigid_inverse

parser=argparse.ArgumentParser()
parser.add_argument('--workspace',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
base=args.workspace/'assets/report/evidence/2026-10-21'
paths={
    'graphs':base/'bowl-step6/kinematics_export_v1/joint_graphs.json',
    'A_recording':base/'bowl-training-feasibility-v4/dataset-proof/datasets/training_layout2002_v4.npz',
    'B_recording':base/'bowl-step6/mechanics_v1/mechanics.npz',
    'B_inventory':base/'bowl-step6/mechanics_v1/inventory.json'}

def angle_error(r1,r2):
    r=r1@r2.T
    sine=np.linalg.norm([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/2
    return float(np.arctan2(sine,np.clip((np.trace(r)-1)/2,-1,1)))

try:
    graphs=json.loads(paths['graphs'].read_text());assert graphs['complete']
    a=np.load(paths['A_recording']);b=np.load(paths['B_recording'])
    assert a['complete'] and b['complete']
    bi=json.loads(paths['B_inventory'].read_text())
    aq=a['state.joint_position'];astate=a['state.eef_9d']
    assert aq.shape==(301,7) and astate.shape==(301,9)
    # Invert the pinned client's root-relative 9D representation; do not use auxiliary action predictions.
    correction=np.array([[0,0,-1],[-1,0,0],[0,1,0.]])
    aref=[]
    for row in astate:
        rawrows=row[3:].reshape(2,3).astype(float)
        assert np.allclose(rawrows@rawrows.T,np.eye(2),atol=1e-5)
        x=rawrows[0]/np.linalg.norm(rawrows[0]);y=rawrows[1]-x*np.dot(x,rawrows[1]);y/=np.linalg.norm(y)
        rot=np.stack([x,y,np.cross(x,y)])@correction.T
        t=np.eye(4);t[:3,:3]=rot;t[:3,3]=row[:3];aref.append(t)
    bq=b['post_q'][:,bi['arm_indices']]
    bref=[rigid_inverse(pose_matrix(root[:3],root[3:7]))@pose_matrix(p,q)
          for root,p,q in zip(b['root'],b['eef_position_world'],b['eef_quaternion_wxyz'])]
    traces={};summary={}
    for label,qs,refs in [('A',aq,np.array(aref)),('B',bq,np.array(bref))]:
        chain=RigidChain(graphs[label])
        pred=np.stack([chain.fk(q) for q in qs])
        pos=np.linalg.norm(pred[:,:3,3]-refs[:,:3,3],axis=-1)
        ang=np.array([angle_error(p[:3,:3],r[:3,:3]) for p,r in zip(pred,refs)])
        assert np.isfinite(pred).all() and np.isfinite(refs).all()
        traces.update({label+'_q':qs,label+'_predicted':pred,label+'_reference':refs,label+'_position_error_m':pos,label+'_rotation_error_rad':ang})
        summary[label]={'rows':len(qs),'max_position_error_m':float(pos.max()),'max_rotation_error_rad':float(ang.max())}
        # Save measured disagreements even if tolerance fails.
        np.savez_compressed(args.output/'pose_comparisons.npz',**traces)
        assert pos.max()<1e-4 and ang.max()<1e-3,(label,summary[label])
        sample_ids=np.linspace(0,len(qs)-1,5,dtype=int)
        jd=[];reversed_errors=[]
        rev=copy.deepcopy(graphs[label]);rev['base_body'],rev['raw_tool_body']=rev['raw_tool_body'],rev['base_body']
        rev['chain']=[{'joint_index':e['joint_index'],'direction':-e['direction']} for e in rev['chain'][::-1]]
        rev['arm_joint_names']=rev['arm_joint_names'][::-1];reverse_chain=RigidChain(rev)
        offset=pose_matrix([.02,.03,.04],[np.sqrt(.5),0,0,np.sqrt(.5)])
        for idx in sample_ids:
            q=qs[idx].astype(float)
            reversed_errors.append(float(np.max(np.abs(reverse_chain.fk(q[::-1])-rigid_inverse(chain.fk(q))))))
            for model,values in [(chain,q),(reverse_chain,q[::-1])]:
                for tool_offset in (None,offset):
                    t,j=model.fk_jacobian(values,tool_offset);numerical=np.zeros_like(j);eps=1e-6
                    for k in range(7):
                        plus=values.copy();minus=values.copy();plus[k]+=eps;minus[k]-=eps
                        tp=model.fk(plus,tool_offset);tm=model.fk(minus,tool_offset)
                        numerical[:3,k]=(tp[:3,3]-tm[:3,3])/(2*eps)
                        skew=((tp[:3,:3]-tm[:3,:3])/(2*eps))@t[:3,:3].T
                        numerical[3:,k]=[skew[2,1],skew[0,2],skew[1,0]]
                    jd.append(float(np.max(np.abs(numerical-j))))
        assert max(jd)<1e-6 and max(reversed_errors)<1e-8,(label,jd,reversed_errors)
        summary[label].update(jacobian_sample_rows=sample_ids.tolist(),max_jacobian_component_error=max(jd),
            max_reversed_chain_component_error=max(reversed_errors))
    provenance={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()}
    for name in ('rigid_chain.py','validate_recorded_kinematics.py'):
        p=Path(__file__).with_name(name);provenance[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    result={'complete':True,'summary':summary,'position_tolerance_m':1e-4,'rotation_tolerance_rad':1e-3,
        'provenance':provenance,'scope':'FK agrees with these existing measured poses; Jacobians match numerical derivatives. No live pose-control, IK feasibility or transfer success claim.'}
    (args.output/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
except BaseException:
    (args.output/'failure.txt').write_text(traceback.format_exc());raise
