"""Synthetic stationary-image contract test; cannot validate a physical robot hold."""
from pathlib import Path
import json,copy,shutil,importlib.util
P=Path(__file__).resolve().parent;OUT=P/'confirmation_static_screen_v1'
def prepare():
 OUT.mkdir(exist_ok=True)
 rim=json.loads((P/'rim_screen_v1/jobs.json').read_text());cases=[(P/'rim_screen_v1'/j['folder'],True) for j in rim if j['positive']]
 v=P/'saved_frame_check_v3/validation';jobs=json.loads((v/'jobs.json').read_text());cases += [(v/j['folder'],False) for j in jobs if j['folder'].split('/')[0] in ['l4000_t1833','l4001_t5169','l4008_t4278']]
 cases += [(P/'saved_frame_check_v3/l4003_t2000/640x360',False)]
 assert len(cases)==8
 (OUT/'PREDECLARATION.json').write_text(json.dumps({'purpose':'Synthetic repeated-image HOLD-format check: 3 frames at +3,+6,+9; no scene movement by construction. Not recorded post-hold footage or real hold validation.','expected':'held/central candidates READY and still; already-supported WAIT/UNCERTAIN with held=false','cases':[{'source':str(p.relative_to(P)),'candidate_positive':pos} for p,pos in cases],'max_calls':8,'cap_usd':1},indent=2))
 rows=[]
 for i,(source,pos) in enumerate(cases):
  folder=OUT/f'case_{i:02d}';folder.mkdir(exist_ok=True);body=json.loads((source/'request.json').read_text());content=body['messages'][1]['content'];imgs=[x for x in content if x['type']=='image_url'][-2:]
  content=[{'type':'text','text':'Phase: HOLD. Hold since tick 0. Ordered images follow.'}]
  for tick in [3,6,9]:
   for view,img in zip(['actual exterior','actual wrist'],imgs):content.extend([{'type':'text','text':f'tick {tick}, {view}'},copy.deepcopy(img)])
  body['messages'][1]['content']=content;(folder/'request.json').write_text(json.dumps(body));rows.append({'folder':folder.name,'positive':pos,'resolution':640})
 (OUT/'jobs.json').write_text(json.dumps(rows,indent=2))
 for n in ['model_metadata.json','current_catalog.json']:shutil.copyfile(P/'saved_frame_check'/n,OUT/n)
 print('Prepared 8 synthetic HOLD-format checks; no calls made.')
def run():
 spec=importlib.util.spec_from_file_location('screen',P/'saved_frame_check/check_saved_frames.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.HERE=OUT;m.CAP=1.;m.run()
if __name__=='__main__':
 import sys
 if sys.argv[1:]==['prepare']:prepare()
 elif sys.argv[1:]==['run']:run()
 else:raise SystemExit('prepare or run')
