from pathlib import Path
import json,shutil,importlib.util,hashlib
HERE=Path(__file__).resolve().parent
crop=json.loads((HERE/'results.json').read_text())['results']; full=json.loads((HERE.parent/'saved_frame_check_v2/results.json').read_text())['results']
keys={x['folder'] for x in crop};full=[x for x in full if x['folder'] in keys];assert len(crop)==len(full)==8
assert all(x['valid'] for x in crop+full)
def errors(xs):return (sum(x['ready'] and not x['positive'] for x in xs),sum(not x['ready'] and x['positive'] for x in xs))
choice='crop' if errors(crop)<errors(full) else 'full'
selected={'choice':choice,'crop_errors_false_then_missed':errors(crop),'full_errors_false_then_missed':errors(full),'prompt_sha256':hashlib.sha256((HERE.parent/'saved_frame_check_v2/prompt_v2.txt').read_bytes()).hexdigest(),'heldout_results_not_yet_seen':True}
(HERE/'SELECTION.json').write_text(json.dumps(selected,indent=2))
validation=HERE/'validation';validation.mkdir(exist_ok=False)
jobs=[j for j in json.loads((HERE/'all_jobs.json').read_text()) if j['split']=='heldout_layouts']
for j in jobs:
 src=HERE/j['folder'];dst=validation/j['folder'];shutil.copytree(src,dst)
 shutil.copyfile(dst/f'{choice}_request.json',dst/'request.json')
for n in ['model_metadata.json','current_catalog.json']:shutil.copyfile(HERE/n,validation/n)
(validation/'jobs.json').write_text(json.dumps(jobs,indent=2))
spec=importlib.util.spec_from_file_location('screen',HERE.parent/'saved_frame_check/check_saved_frames.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.HERE=validation;m.CAP=1.;print('Frozen selection:',choice,flush=True);m.run()
