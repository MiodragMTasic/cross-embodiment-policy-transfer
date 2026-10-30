from pathlib import Path
import importlib.util,json,shutil,base64,io
import numpy as np
from PIL import Image
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('screen',BASE/'check_saved_frames.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.HERE=BASE.parent/'saved_frame_check_v3';m.HERE.mkdir(exist_ok=True)
dev=[(4003,t,v) for t,v in [(750,False),(900,False),(1000,False),(1200,True),(1500,True),(1870,False),(2000,False)]]+[(4009,1500,False)]
holdout=[]
for layout in [4000,4001,4008]:
 run=m.RUNS/f'step8_B_P1_l{layout}_s0_v1';r=[json.loads(x) for x in (run/'gate_trace.jsonl').read_text().splitlines()];o=np.load(run/'tracking.npz')['objects_banana_bowl'];n=min(len(r),len(o));d=np.linalg.norm(o[:n,0,:2]-o[:n,1,:2],axis=1)
 def eligible(t,kind):
  for k in (12,6,0):
   i=t-k;p=r[i]['post'];held=p['contact'] and r[i]['command']>.5
   if kind=='central' and not (held and d[i]<.025):return False
   if kind=='approach' and not (held and d[i]>.12):return False
   if kind=='rest' and not (not p['contact'] and (p['bowl_contact'] or p['table_contact'])):return False
  return True
 for kind in ['approach','central','rest']:
  candidates=[t for t in range(30,n-6,3) if eligible(t,kind)]
  if not candidates:raise RuntimeError(f'no {kind} for {layout}')
  holdout.append((layout,candidates[len(candidates)//2],kind=='central'))
(m.HERE/'PREDECLARATION.json').write_text(json.dumps({'dev':dev,'heldout_layout_cases':holdout,'crop_pixels':[160,120,560,345],'selection':'Compare full v2 dev responses against crop v2 prompt on same 8 cases; minimize false READY first, then missed positives; tie keeps full. Then freeze selected view and query all 9 new-layout cases. Zero false READY and all 3 candidate positives required to pass small screening. No live approval implied.','max_calls':17,'cap_usd':1,'note':'Cross-layout validation uses existing known experiment recordings but images not used in previous prompt trials; labels based on contact and distance only. No physical drop oracle.'},indent=2))
m.CASES=dev+holdout;m.prepare()
prompt=(BASE.parent/'saved_frame_check_v2/prompt_v2.txt').read_text()
jobs=[j for j in json.loads((m.HERE/'jobs.json').read_text()) if j['resolution']==640]
for i,j in enumerate(jobs):
 j['split']='development' if i<8 else 'heldout_layouts'
 folder=m.HERE/j['folder'];p=folder/'request.json';b=json.loads(p.read_text());b['messages'][0]['content']=prompt
 (folder/'full_request.json').write_text(json.dumps(b))
 parts=[x for x in b['messages'][1]['content'] if x['type']=='image_url']
 for part in parts[::2]:
  im=Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',')[1])))
  im=im.crop((160,120,560,345)).resize((640,360),Image.Resampling.LANCZOS)
  buf=io.BytesIO();im.save(buf,format='PNG');part['image_url']['url']='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
 (folder/'crop_request.json').write_text(json.dumps(b))
 p.write_text(json.dumps(b))
(m.HERE/'all_jobs.json').write_text(json.dumps(jobs,indent=2));(m.HERE/'jobs.json').write_text(json.dumps(jobs[:8],indent=2))
for name in ['model_metadata.json','current_catalog.json']:shutil.copyfile(BASE/name,m.HERE/name)
print('Prepared crop development and untouched-layout validation:',holdout)
