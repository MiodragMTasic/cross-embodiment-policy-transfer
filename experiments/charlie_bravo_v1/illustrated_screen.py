"""Visual-reference candidate, frozen before new boundary responses. No VM."""
from pathlib import Path
import json,io,hashlib,subprocess,importlib.util,shutil,copy
import numpy as np
from PIL import Image
from vision_images import encode_views
from controller import Frame
from transport import payload
P=Path(__file__).resolve().parent;OUT=P/'illustrated_screen_v1';ROOT=P.parents[1]
RUNS=ROOT/'assets/report/evidence/2026-10-25/step8_after_v1/runs'
BANDS=[('under_2.5cm',0,.025),('2.5_to_5cm',.025,.05),('5_to_8cm',.05,.08),('8_to_12cm',.08,.12)]
def prepare():
 OUT.mkdir(exist_ok=True)
 old=json.loads((P/'rim_screen_v1/PREDECLARATION.json').read_text())['cases'];used={(x['layout'],x['tick']) for x in old}
 references=[]
 # These are desired conservative trigger examples, not physics-based drop labels.
 for band,verdict,held,central,reason in [('under_2.5cm','READY_TO_RELEASE',True,True,'Banana is pinched and suspended in the central release position.'),('5_to_8cm','WAIT',True,False,'Banana is held but offset toward the rim; wait for a more central position.'),('8_to_12cm','WAIT',True,False,'Banana is held beside the acceptable central release position; wait.')]:
  case=next(x for x in old if x['band']==band and x['layout']==4003)
  folder=P/'rim_screen_v1'/f'{band}_l4003_t{case["tick"]}'
  b=json.loads((folder/'request.json').read_text());images=[x for x in b['messages'][1]['content'] if x['type']=='image_url'][-2:]
  references.append({'source':str(folder.relative_to(P)),'images':images,'answer':dict(verdict=verdict,held=held,central=central,clearance=central,views_agree=True,motion='unknown',reason=reason)})
 folder=P/'saved_frame_check_v3/l4003_t2000/640x360';b=json.loads((folder/'crop_request.json').read_text())
 references.append({'source':str(folder.relative_to(P)),'images':[x for x in b['messages'][1]['content'] if x['type']=='image_url'][-2:],'answer':dict(verdict='WAIT',held=False,central=True,clearance=False,views_agree=True,motion='still',reason='Banana already rests on the bowl; do not request a release.')})
 selected=[]
 for name,lo,hi in BANDS:
  options=[]
  for layout in range(4000,4012):
   run=RUNS/f'step8_B_P1_l{layout}_s0_v1';rs=[json.loads(x) for x in (run/'gate_trace.jsonl').read_text().splitlines()];ob=np.load(run/'tracking.npz')['objects_banana_bowl'];n=min(len(rs),len(ob));ds=np.linalg.norm(ob[:n,0,:2]-ob[:n,1,:2],axis=1)
   ts=[t for t in range(30,n-6,3) if (layout,t) not in used and all(rs[t-k]['post']['contact'] and rs[t-k]['command']>.5 and lo<=ds[t-k]<hi for k in [0,6,12])]
   if ts:
    t=ts[len(ts)//4];options.append({'layout':layout,'tick':t,'band':name,'distances_m':[float(ds[t-k]) for k in [12,6,0]]})
  selected+=options[:4]
 assert len(selected)==16
 declaration={'cases':selected,'selection':'First four available layouts per band, lower quartile of eligible timestamps; excludes exact prior median cases. Same episodes, correlated; not independent generalization.','references':[{'source':x['source'],'answer':x['answer']} for x in references],'criterion':'under-2.5cm at least 3/4 READY; zero READY at 5–8 and 8–12cm; middle band descriptive. Conservative trigger screen, not physical safety proof.','max_calls':16,'cap_usd':1,'no_tuning_after_responses':True}
 (OUT/'PREDECLARATION.json').write_text(json.dumps(declaration,indent=2))
 jobs=[];prompt=(P/'vision_prompt.txt').read_text()+'\nThe following labelled reference images define the requested CONSERVATIVE trigger. Use them to distinguish central from offset/rim positions, not to predict the final landing. Apply the same visual standard to the final query, which can have a different layout. Do not call an offset position central merely because the banana overlaps the bowl in perspective. If uncertain, WAIT or UNCERTAIN. Reference timestamps are not part of the query history.'
 (OUT/'candidate_prompt.txt').write_text(prompt)
 for case in selected:
  layout,t=case['layout'],case['tick'];folder=OUT/f'{case["band"]}_l{layout}_t{t}';folder.mkdir(exist_ok=True);frames=[]
  for tick in [t-12,t-6,t]:
   png=subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-ss',str(tick/15),'-i',str(RUNS/f'step8_B_P1_l{layout}_s0_v1/views.mp4'),'-frames:v','1','-f','image2pipe','-vcodec','png','-'],capture_output=True,check=True).stdout
   im=Image.open(io.BytesIO(png)).convert('RGB');ext,wri=encode_views(np.asarray(im.crop((0,0,640,360))),np.asarray(im.crop((1280,0,1920,360))))
   for name,data in [('exterior',ext),('wrist',wri)]:(folder/f'{tick}_{name}.png').write_bytes(data)
   frames.append(Frame(tick,ext,wri))
  body=payload(prompt,frames,{'phase':'POLICY','tick':t,'hold_since_tick':None});messages=[body['messages'][0]]
  for ref in references:
   messages.extend([{'role':'user','content':[{'type':'text','text':'REFERENCE example: actual exterior, then wrist.'},*copy.deepcopy(ref['images'])]},{'role':'assistant','content':json.dumps(ref['answer'])}])
  messages.append(body['messages'][1]);body['messages']=messages;encoded=json.dumps(body);assert len(encoded.encode())<12000000
  (folder/'request.json').write_text(encoded);(folder/'labels_not_sent.json').write_text(json.dumps(case,indent=2));jobs.append({'folder':folder.name,'band':case['band'],'positive':case['band']=='under_2.5cm','resolution':640})
 (OUT/'jobs.json').write_text(json.dumps(jobs,indent=2))
 for name in ['model_metadata.json','current_catalog.json']:shutil.copyfile(P/'saved_frame_check'/name,OUT/name)
 print('Prepared',len(jobs),'calls; largest request bytes',max((OUT/j['folder']/'request.json').stat().st_size for j in jobs))
def run():
 spec=importlib.util.spec_from_file_location('screen',P/'saved_frame_check/check_saved_frames.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.HERE=OUT;m.CAP=1.;m.run()
if __name__=='__main__':
 import sys
 if sys.argv[1:]==['prepare']:prepare()
 elif sys.argv[1:]==['run']:run()
 else:raise SystemExit('prepare or run')
