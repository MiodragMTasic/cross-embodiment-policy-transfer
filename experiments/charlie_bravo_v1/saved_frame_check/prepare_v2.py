"""Prepare fixed second-round prompt and fresh holdout cases before any calls."""
import importlib.util,json,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('screen',BASE/'check_saved_frames.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.HERE=BASE.parent/'saved_frame_check_v2';m.HERE.mkdir(exist_ok=True)
dev=m.CASES.copy()
holdout=[(4003,800,False),(4003,1000,False),(4003,1350,True),(4003,1650,True),(4003,1900,False),(4003,2100,False),(4009,1600,False)]
m.CASES=dev+holdout
prompt='''You are a conservative visual release observer for a simulated robot. You receive three timestamps, each with an ACTUAL EXTERIOR view followed by a WRIST view. The two cameras have different perspectives. Use only these images; no hidden state is available.

Your task is to identify a banana that is CURRENTLY PINCHED AND SUSPENDED by the gripper over the bowl, BEFORE it is released. A banana already resting in/on the bowl is NEVER READY, even if fingers hover above it or appear near it in the wrist image. The presence of a banana in the bowl is not evidence that the gripper holds it. Specifically inspect whether the gripping finger surfaces visibly pinch opposite sides of the banana, or whether the banana has separated and is supported by the bowl. On uncertainty about contact or support, set held=false and verdict=UNCERTAIN. Do not infer contact from proximity, 2D overlap, closed-looking fingers, or the instruction to place it.

Use the exterior view to judge whether the suspended banana is above the bowl opening with an air gap; use the wrist view as a second perspective for grasp and rim alignment. The wrist view is attached to the arm: a banana can extend down toward the bottom of that image even when it is over the bowl. Do not require the banana to coincide with the image centre. Central means a plausible downward release into the interior rather than an edge or table landing, accounting for both views. If depth or whole-object clearance cannot be resolved, say UNCERTAIN. Do not invent certainty.

For POLICY phase: READY_TO_RELEASE requires held=true, central=true, clearance=true, views_agree=true. Motion is allowed if already centrally aligned; being near an edge and moving inward is WAIT. For HOLD phase: additionally require motion=still using only the new post-hold images. Judge the latest timestamp, using earlier timestamps only for motion/contact changes.

Return exactly one JSON object, no prose: verdict (WAIT|READY_TO_RELEASE|UNCERTAIN), held (boolean), central (boolean), clearance (boolean), views_agree (boolean), motion (inward|outward|still|unknown), reason (one sentence, at most 240 characters). In reason describe actual contact/support evidence, not generic requirements. A stationary banana resting on the bowl must have held=false, regardless of how well centred it is.
'''
(m.HERE/'prompt_v2.txt').write_text(prompt)
(m.HERE/'PREDECLARATION.json').write_text(json.dumps({'dev_cases':dev,'fresh_holdout_cases':holdout,'image_size':[640,360],'cap_usd':1.0,'max_calls':14,'criterion':'0 false READY negatives and both positive candidates READY on fresh holdout; diagnostic only, not live validation','authorization':'User authorized reasonable further Token Factory use; VM stays off','limitations':'Holdout timestamps share recorded episodes and are correlated, not independent trials'},indent=2))
m.prepare()
jobs=json.loads((m.HERE/'jobs.json').read_text());jobs=[j for j in jobs if j['resolution']==640]
for i,j in enumerate(jobs):
 j['split']='development' if i<7 else 'fresh_timestamps'
 p=m.HERE/j['folder']/'request.json';b=json.loads(p.read_text());b['messages'][0]['content']=prompt;p.write_text(json.dumps(b))
(m.HERE/'jobs.json').write_text(json.dumps(jobs,indent=2))
shutil.copyfile(BASE/'current_catalog.json',m.HERE/'current_catalog.json')
shutil.copyfile(BASE/'model_metadata.json',m.HERE/'model_metadata.json')
print('Prepared fixed v2:',len(jobs),'640px calls; no calls made')
