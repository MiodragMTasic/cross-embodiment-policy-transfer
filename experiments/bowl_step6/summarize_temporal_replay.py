"""Read completed temporal diagnostic and produce descriptive, explicitly unscored results."""
import json,hashlib
from pathlib import Path
import numpy as np
BASE=Path('assets/report/evidence/2026-10-21/bowl-step6')
p=BASE/'temporal_B_v1/step6_B_temporal_replay_v1'
x=np.load(p/'tracking.npz');trace=json.loads((p/'contact_trace.json').read_text())
a=np.load(BASE/'replay_A_v1/step6_A_recorded_replay_v1/tracking.npz')
assert x['complete'] and len(trace)==3612
assert np.array_equal(x['action_A'],np.repeat(a['action'],12,axis=0))
assert np.array_equal(x['logical_index_0based'],np.repeat(np.arange(301),12))
assert np.array_equal(x['repeat_index_0based'],np.tile(np.arange(12),301))
assert np.allclose(x['physical_time_s'],np.arange(1,3613)/15,rtol=0,atol=1e-12)
assert np.allclose(x['logical_elapsed_s'],np.arange(1,3613)/180,rtol=0,atol=1e-12)
forces=np.array([[r['post']['forces_world']['banana_on_'+s+'_tip'] for s in ('left','right')] for r in trace]);contact=np.linalg.norm(forces,axis=2)>.1
q=x['gripper_closure'];banana=x['objects'][:,0];closed=x['action_A'][:,7]>.5
indices=np.flatnonzero(closed);bursts=np.split(indices,np.where(np.diff(indices)>1)[0]+1)
summary={'scope':'Descriptive prerecorded temporal diagnostic; no policy success score or independent-repeat claim','steps':3612,'logical_steps':301,'duration_s':240.8,'task_score':None,'source_action_exact':True,
 'max_banana_center_z_m':float(banana[:,2].max()),'final_banana_state':banana[-1].tolist(),
 'max_closure':float(q.max()),'any_tip_contact_rows':int(contact.any(axis=1).sum()),'bilateral_tip_contact_rows':int(contact.all(axis=1).sum()),
 'rate_limited_rows':int(x['limited'].sum()),'close_bursts':[],
 'tracking_canonical_command_m':{k:float(v) for k,v in zip(('median','p95','max'),np.quantile(np.linalg.norm(x['command_tracking_error'][:,:3],axis=1),[.5,.95,1]))},
 'inputs_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (p/'tracking.npz',p/'contact_trace.json')},
 'limitations':['Initial pre-contact sample excluded; only post-step samples counted.','Contacts are control-rate samples, not all physics substeps.','Normalized closure does not equal grip force or aperture.','Later attempts after object divergence are not independent trials.','No automated placement score assigned; video/support/release review required.']}
for ids in bursts:
 if not len(ids):continue
 summary['close_bursts'].append({'physical_first_row_0based':int(ids[0]),'physical_last_row_0based':int(ids[-1]),'logical_first_row_0based':int(x['logical_index_0based'][ids[0]]),'ticks':len(ids),'physical_seconds':len(ids)/15,'max_closure':float(q[ids].max()),'any_tip_rows':int(contact[ids].any(axis=1).sum()),'bilateral_tip_rows':int(contact[ids].all(axis=1).sum()),'max_banana_center_z_m':float(banana[ids,2].max())})
print(json.dumps(summary,indent=2))
