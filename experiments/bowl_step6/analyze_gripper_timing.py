"""Read-only timing analysis of sealed stationary captures; no simulator or fitting."""
from pathlib import Path
import json,hashlib,numpy as np
BASE=Path('assets/report/evidence/2026-10-21/bowl-step6')
SOURCES={'A':('tracking_A_v3','step6_A_tracking_v3','finger_joint',np.pi/4),'B_native':('tracking_B_v1','step6_B_tracking_v1','robotiq_85_left_knuckle_joint',.8),'B_mapped':('mapped_tracking_B_v1','step6_B_mapped_v1','robotiq_85_left_knuckle_joint',.8)}
result={};curves={}
for label,(folder,case,joint,closed) in SOURCES.items():
 root=BASE/folder;manifest=json.loads((root/'manifest.json').read_text())
 for entry in manifest['files']:
  f=root/entry['path'];assert hashlib.sha256(f.read_bytes()).hexdigest()==entry['sha256'],str(f)
 p=root/case;x=np.load(p/'tracking.npz');inv=json.loads((p/'inventory.json').read_text());jid=inv['joint_names'].index(joint)
 assert x['complete'] and len(x['q'])==300
 rows=np.flatnonzero(x['phase']=='close');assert len(rows) and np.all(np.diff(rows)==1) and np.all(x['action'][rows,7]==1)
 values=x['q'][rows,jid]/closed;pre=float(x['pre_q'][rows[0],jid]/closed);assert abs(pre)<.01
 hits=np.flatnonzero(values>=.95);assert len(hits)
 ticks=int(hits[0])+1
 result[label]={'joint':joint,'closed_target_rad':closed,'pre_closure':pre,'first_tick_closure':float(values[0]),'ticks_to_95pct':ticks,'seconds_to_95pct':ticks/15,'close_row_start_0based':int(rows[0]),'tracking_sha256':hashlib.sha256((p/'tracking.npz').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
 curves[label]=values.tolist()
factor=int(np.ceil(max(result[k]['ticks_to_95pct'] for k in ('B_native','B_mapped'))/result['A']['ticks_to_95pct']))
print(json.dumps({'measurements':result,'curves_normalized_joint':curves,'candidate_uniform_repeat_factor':factor,'factor_rule':'ceil(max(B native,B mapped 95pct closing ticks) / A 95pct closing ticks), computed only from existing stationary captures','limitations':['Normalized joint position is not aperture, gripping force or task success.','Single existing capture per condition; no statistical precision claim.','Uniform repetition changes arm dynamics and contact timing as well as gripper timing.','No new physics or implementation approval follows this analysis.']},indent=2))
