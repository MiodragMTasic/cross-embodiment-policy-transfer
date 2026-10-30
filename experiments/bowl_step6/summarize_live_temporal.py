"""Describe a completed live B temporal diagnostic; never assign task success."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();r=args.run
x=np.load(r/'tracking.npz');c=json.loads((r/'checks.json').read_text())
assert bool(x['complete']) and c['complete'] and len(x['step'])==3612
assert np.array_equal(x['logical_index_0based'],np.repeat(np.arange(301),12))
assert np.array_equal(x['repeat_index_0based'],np.tile(np.arange(12),301))
assert np.array_equal(np.flatnonzero(x['fresh_request']),np.arange(0,3612,96))
assert np.array_equal(x['P0_action'],np.repeat(x['P0_action'][::12],12,axis=0))
assert np.allclose(x['physical_time_s'],np.arange(1,3613)/15,atol=1e-12,rtol=0)
assert x['simulator_step'][-1]==3613
force=np.linalg.norm(x['banana_force_on_tips'],axis=2);contact=force>.1
banana=x['objects_banana_bowl'][:,0];bowl=x['objects_banana_bowl'][:,1]
closed=x['P0_action'][:,7]>.5
# Closure uses the recorded actual observation, measured BEFORE each step.
closure=x['gripper_observation'].reshape(-1)
def intervals(mask):
 ids=np.flatnonzero(mask)
 return [a for a in np.split(ids,np.where(np.diff(ids)>1)[0]+1) if len(a)]
def desc(ids):
 return {'first_row_0based':int(ids[0]),'last_row_0based':int(ids[-1]),'start_physical_s':float(x['physical_time_s'][ids[0]]),'end_physical_s':float(x['physical_time_s'][ids[-1]]),'ticks':len(ids),'max_banana_center_z_m':float(banana[ids,2].max()),'bilateral_contact_ticks':int(contact[ids].all(axis=1).sum())}
summary={'scope':'Unscored single closed-loop P0/native B diagnostic; no success-rate or causal-isolation claim','controls':3612,'logical_actions':301,'requests':int(x['fresh_request'].sum()),'policy_physical_duration_s':240.8,'total_simulator_duration_s':3613/15,'replan_interval_s':6.4,'task_score':None,'max_banana_center_z_m':float(banana[:,2].max()),'final_banana_state':banana[-1].tolist(),'final_bowl_state':bowl[-1].tolist(),'max_pre_step_normalized_closure':float(closure.max()),'any_tip_contact_ticks':int(contact.any(axis=1).sum()),'bilateral_contact_ticks':int(contact.all(axis=1).sum()),'rate_limited_ticks':int(x['rate_limited'].sum()),'close_bursts':[desc(a) for a in intervals(closed)],'bilateral_contact_intervals':[desc(a) for a in intervals(contact.all(axis=1))],'release_transition_rows_0based':(np.flatnonzero(closed[:-1]&~closed[1:])+1).tolist(),'limitations':['Single development episode; not independent trials.','Objects and contact telemetry are review-only, not policy inputs.','Feedback is only every 6.4 simulated seconds.','Maximum object center height alone does not prove stable grasp or placement.','Controlled release and support require video/trace review.']}
print(json.dumps(summary,indent=2))
