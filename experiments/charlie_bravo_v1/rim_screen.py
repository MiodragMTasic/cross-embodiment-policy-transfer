"""Frozen visual candidate: descriptive distance-band screening, no VM."""
from pathlib import Path
import json,io,hashlib,subprocess,importlib.util,shutil
import numpy as np
from PIL import Image
from vision_images import encode_views
from controller import Frame
from transport import payload
P=Path(__file__).resolve().parent
OUT=P/'rim_screen_v1'
ROOT=P.parents[1]
RUNS=ROOT/'assets/report/evidence/2026-10-25/step8_after_v1/runs'
BANDS=[('under_2.5cm',0,.025),('2.5_to_5cm',.025,.05),('5_to_8cm',.05,.08),('8_to_12cm',.08,.12)]
def prepare():
 OUT.mkdir(exist_ok=True);selected=[]
 for name,lo,hi in BANDS:
  options=[]
  for layout in range(4000,4012):
   run=RUNS/f'step8_B_P1_l{layout}_s0_v1';rs=[json.loads(x) for x in (run/'gate_trace.jsonl').read_text().splitlines()];ob=np.load(run/'tracking.npz')['objects_banana_bowl'];n=min(len(rs),len(ob));ds=np.linalg.norm(ob[:n,0,:2]-ob[:n,1,:2],axis=1)
   ts=[t for t in range(30,n-6,3) if all(rs[t-k]['post']['contact'] and rs[t-k]['command']>.5 and lo<=ds[t-k]<hi for k in [0,6,12])]
   if ts:
    t=ts[len(ts)//2];options.append({'layout':layout,'tick':t,'band':name,'distances_m':[float(ds[t-k]) for k in [12,6,0]],'root_z_delta_m':[float(ob[t-k,0,2]-ob[t-k,1,2]) for k in [12,6,0]]})
  selected+=options[:4]
 assert len(selected)==16
 (OUT/'PREDECLARATION.json').write_text(json.dumps({'selection':'Scan layouts ascending, median eligible history per layout, first four available layouts per band. All three frames contact+closed and in band. No prompt/crop changes after results.','cases':selected,'max_calls':16,'cap_usd':1,'purpose':'READY rates by distance band, not physical-drop success or calibrated safety labels','prompt_sha256':hashlib.sha256((P/'vision_prompt.txt').read_bytes()).hexdigest()},indent=2))
 jobs=[]
 for case in selected:
  layout,t=case['layout'],case['tick'];run=RUNS/f'step8_B_P1_l{layout}_s0_v1';folder=OUT/f'{case["band"]}_l{layout}_t{t}';folder.mkdir(exist_ok=True);fs=[]
  for tick in [t-12,t-6,t]:
   png=subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-ss',str(tick/15),'-i',str(run/'views.mp4'),'-frames:v','1','-f','image2pipe','-vcodec','png','-'],capture_output=True,check=True).stdout
   im=Image.open(io.BytesIO(png)).convert('RGB');assert im.size==(1920,360)
   ext,wri=encode_views(np.asarray(im.crop((0,0,640,360))),np.asarray(im.crop((1280,0,1920,360))))
   for name,data in [('exterior',ext),('wrist',wri)]: (folder/f'{tick}_{name}.png').write_bytes(data)
   fs.append(Frame(tick,ext,wri))
  body=payload((P/'vision_prompt.txt').read_text(),fs,{'phase':'POLICY','tick':t,'hold_since_tick':None})
  (folder/'request.json').write_text(json.dumps(body));(folder/'labels_not_sent.json').write_text(json.dumps(case,indent=2))
  jobs.append({'folder':folder.name,'band':case['band'],'positive':case['band']=='under_2.5cm','resolution':640})
 (OUT/'jobs.json').write_text(json.dumps(jobs,indent=2))
 for n in ['model_metadata.json','current_catalog.json']:shutil.copyfile(P/'saved_frame_check'/n,OUT/n)
 print('Prepared sixteen frozen cases. No calls made.')
def run():
 spec=importlib.util.spec_from_file_location('screen',P/'saved_frame_check/check_saved_frames.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.HERE=OUT;m.CAP=1.;m.run()
if __name__=='__main__':
 import sys
 if sys.argv[1:]==['prepare']:prepare()
 elif sys.argv[1:]==['run']:run()
 else:raise SystemExit('prepare or run')
