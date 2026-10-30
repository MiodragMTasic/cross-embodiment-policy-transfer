"""CPU-only summaries of all original-scene A attempts; no simulation or writes to evidence."""
import hashlib,json
from pathlib import Path
import h5py,numpy as np
root=Path('/home/miodrag/robot-transfer/bowl/RoboLab/output')
cases=['a_p0_first_bowl_seed0','a_p0_terminalframe_seed0','step5_controlled_base_s0_v3','step5_controlled_base_s1_v3']
def bursts(x):
 d=np.diff(np.r_[False,x,False].astype(int));return [[int(s),int(e)] for s,e in zip(np.where(d==1)[0],np.where(d==-1)[0])]
out={'scope':'Existing original-scene A runs, no new inference. Upstream first demos have weak immediate stopping; controlled cases have50s observation. No matched success-rate claim.','cases':{}}
for case in cases:
 folder=root/case/'BananaInBowlTask';p=folder/'run_0.hdf5'
 with h5py.File(p) as f:
  g=f['data/demo_0'];acts=g['actions'][:];banana=g['states/rigid_object/banana/root_pose'][:]
  record={'recording_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'actions':len(acts),'close_ticks':int((acts[:,7]>.5).sum()),'close_bursts_row_start_end_exclusive':bursts(acts[:,7]>.5),'banana_max_z':float(banana[:,2].max()),'banana_first_last_xyz':banana[[0,-1],:3].tolist(),'initial_state':{}}
  g['initial_state'].visititems(lambda n,x:record['initial_state'].update({n:x[:].tolist()}) if isinstance(x,h5py.Dataset) else None)
 sync=folder/'synchronized.hdf5'
 if sync.exists():
  with h5py.File(sync) as f:
   record['sync_sha256']=hashlib.sha256(sync.read_bytes()).hexdigest();record['max_pre_action_gripper_closure']=float(f['state.gripper_position'][:].max())
 out['cases'][case]=record
p=root/'step5_training_layout2002_v4/BananaInBowlUniformInitPose10cmTask/run_0.hdf5'
with h5py.File(p) as f:
 g=f['data/demo_0'];record={'recording_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'initial_state':{}}
 g['initial_state'].visititems(lambda n,x:record['initial_state'].update({n:x[:].tolist()}) if isinstance(x,h5py.Dataset) else None)
 record['first_two_post_action_robot_q']=g['states/articulation/robot/joint_position'][:2].tolist()
 out['A2002_setup']=record
print(json.dumps(out,indent=2))
