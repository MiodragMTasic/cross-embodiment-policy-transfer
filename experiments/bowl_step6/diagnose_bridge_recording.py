"""Read-only CPU diagnosis of immutable B/P0 v2 and existing A2002 recordings."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from rigid_chain import RigidChain

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
base=Path(__file__).resolve().parents[2]/'assets/report/evidence/2026-10-21'
paths={'B':base/'bowl-step6/P0_B_v2/step6_B_P0_v2/tracking.npz',
       'inventory':base/'bowl-step6/P0_B_v2/step6_B_P0_v2/inventory.json',
       'graphs':base/'bowl-step6/kinematics_export_v1/joint_graphs.json',
       'A':base/'bowl-training-feasibility-v4/dataset-proof/datasets/training_layout2002_v4.npz'}
b=np.load(paths['B']);a=np.load(paths['A']);inv=json.loads(paths['inventory'].read_text());graphs=json.loads(paths['graphs'].read_text())
assert b['complete'] and a['complete'];A=RigidChain(graphs['A']);B=RigidChain(graphs['B'])
CA,CB=[np.array(inv['fixed_frames']['raw_T_canonical'][x]) for x in ('A','B')]
traces={};summary={'scope':'CPU diagnostic only. Different layouts and robot dynamics; A2002 is not a matched statistical control.',
 'metadata_correction':'Sealed inventory description says left inner knuckle; executed observation uses robotiq_85_left_knuckle_joint (index 8). Sealed files unchanged.',
 'inputs_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}}

def stats(x):
 x=np.asarray(x);assert np.isfinite(x).all()
 return {'n':len(x),'median':float(np.median(x)),'p95':float(np.percentile(x,95)),'max':float(np.max(x)),'max_row_0based':int(np.argmax(x))}

def errors(name,x,y):
 pos=np.linalg.norm(x[:,:3,3]-y[:,:3,3],axis=1)
 r=x[:,:3,:3]@y[:,:3,:3].transpose(0,2,1)
 sine=np.linalg.norm(np.stack([r[:,2,1]-r[:,1,2],r[:,0,2]-r[:,2,0],r[:,1,0]-r[:,0,1]],axis=1),axis=1)/2
 ang=np.arctan2(sine,np.clip((np.trace(r,axis1=1,axis2=2)-1)/2,-1,1))
 traces[name+'_m']=pos;traces[name+'_rad']=ang
 summary[name]={'position_m':stats(pos),'rotation_rad':stats(ang)}

def fk(chain,qs,offset):return np.stack([chain.fk(q,offset) for q in qs])

def runs(mask):
 d=np.diff(np.r_[False,mask,False].astype(int));return [(int(s),int(e)) for s,e in zip(np.where(d==1)[0],np.where(d==-1)[0])]

# Same-row comparisons use post-action measured pose and commanded pose.
target=b['target_B']@CB;goal=fk(B,b['B_goal'],CB);cmd=fk(B,b['B_action'][:,:7],CB);actual=b['tool_root']@CB
for name,x,y in [('B_IK_target_goal',target,goal),('B_limiter_goal_command',goal,cmd),('B_controller_command_measured',cmd,actual),('B_total_target_measured',target,actual)]:errors(name,x,y)
# At observation i, action i-1 is the preceding command: never compare pre-state with current action.
for label,qs,actions,chain,offset in [('B_virtual_A',b['virtual_A_q'],b['P0_action'],A,CA),('A_native',a['state.joint_position'],a['action.executed_joint_gripper'],A,CA)]:
 delta=qs[1:]-actions[:-1,:7];norm=np.linalg.norm(delta,axis=1)
 traces[label+'_q_delta_rad']=delta;summary[label+'_q_previous_action_l2_rad']=stats(norm)
 errors(label+'_previous_action_pose',fk(chain,qs[1:],offset),fk(chain,actions[:-1,:7],offset))
 # Linearized local nullspace projection only, not an exact same-pose decomposition.
 null=[]
 for q,d in zip(qs[1:],delta):
  _,jac=chain.fk_jacobian(q,offset);_,_,vh=np.linalg.svd(jac,full_matrices=True)
  n=vh[-1];null.append(abs(float(n@d)))
 traces[label+'_local_null_component_rad']=np.array(null);summary[label+'_local_null_component_rad']=stats(null)
 summary[label+'_large_joint_small_pose_rows']=int(np.sum((norm>.1)&(traces[label+'_previous_action_pose_m']<.005)&(traces[label+'_previous_action_pose_rad']<.02)))
# Gripper state sampled before action; force/physical joint sampled after action.
gi=inv['joint_names'].index('robotiq_85_left_knuckle_joint')
assert np.allclose(b['gripper_observation'][:,0],np.clip(b['pre_q'][:,gi]/.8,0,1),atol=1e-7)
force=np.linalg.norm(b['banana_force_on_tips'],axis=2)
for label,action,obs in [('B',b['P0_action'][:,7],b['gripper_observation'][:,0]),('A',a['action.executed_joint_gripper'][:,7],a['state.gripper_position'][:,0])]:
 close=action>.5
 summary[label+'_gripper']={'close_steps':int(close.sum()),'max_observed_closure':float(obs.max()),'bursts':[
  {'start_row_0based':s,'end_exclusive':e,'seconds':(e-s)/15,'max_observed_closure_during':float(obs[s:e].max())} for s,e in runs(close)]}
 if label=='B':
  summary[label+'_gripper'].update({'any_tip_over_0_1N_rows':int((force>.1).any(axis=1).sum()),'both_tip_over_0_1N_rows':int((force>.1).all(axis=1).sum()),'close_and_any_contact_rows':int((close&(force>.1).any(axis=1)).sum()),'max_post_closure':float(np.clip(b['q'][:,gi]/.8,0,1).max())})
summary['B_rate_limited_rows']=int(b['rate_limited'].sum())
summary['B_banana_max_z_m']=float(b['objects_banana_bowl'][:,0,2].max())
summary['B_banana_first_last_xyz_m']=b['objects_banana_bowl'][[0,-1],0,:3].tolist()
# Stable targeting intervals isolate end-of-block error without claiming steady state.
for s,e in [(0,50),(50,135),(135,225),(225,300),(300,750)]:
 summary[f'B_window_{s}_{e}']={'controller_error_m':stats(traces['B_controller_command_measured_m'][s:e]),'total_error_m':stats(traces['B_total_target_measured_m'][s:e]),'rate_limited':int(b['rate_limited'][s:e].sum())}
# Distributions of all predicted gripper rows, versus the first eight actually considered.
response_summary=[]
for path in sorted((paths['B'].parent/'responses').glob('*.npz')):
 r=np.load(path)['gripper_position'][:,0];i=int(path.stem)
 response_summary.append({'request_row':i,'close_rows_first8':int((r[:min(8,750-i)]>.5).sum()),'close_rows_all40':int((r>.5).sum()),'first_close_row':int(np.where(r>.5)[0][0]) if (r>.5).any() else None})
summary['response_gripper_horizons']=response_summary
np.savez_compressed(args.output/'traces.npz',**traces)
(args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(args.output/'diagnose_bridge_recording.py').write_bytes(Path(__file__).read_bytes())
(args.output/'rigid_chain.py').write_bytes(Path(__file__).with_name('rigid_chain.py').read_bytes())
print(json.dumps({k:v for k,v in summary.items() if k!='response_gripper_horizons'},indent=2))
