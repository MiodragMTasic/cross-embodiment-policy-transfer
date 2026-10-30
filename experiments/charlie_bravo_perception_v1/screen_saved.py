"""RGB feasibility inventory. Does not claim detection accuracy or test depth."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from perception import observe
P=Path(__file__).resolve().parent
SOURCE=P.parent/'charlie_bravo_v1/rim_screen_v1'
def main():
 rows=[]
 for job in json.loads((SOURCE/'jobs.json').read_text()):
  folder=SOURCE/job['folder']; label=json.loads((folder/'labels_not_sent.json').read_text())
  tick=label['tick']
  for view in ['exterior','wrist']:
   path=folder/f'{tick}_{view}.png'
   rows.append({'case':job['folder'],'view':view,'image_path':str(path),**observe(np.asarray(Image.open(path).convert('RGB')))})
 (P/'saved_rgb_screen.json').write_text(json.dumps({'purpose':'Colour-mask feasibility only, not correctness, held state, depth or release validation.','frames':len(rows),'rows':rows},indent=2))
 print('Frames:',len(rows))
 for view in ['exterior','wrist']:
  selected=[r for r in rows if r['view']==view]
  print(view,'candidate masks in both objects',sum(all(o['detected'] for o in r['objects'].values()) for r in selected),'/',len(selected))
 print('No depth; no release authorization on any case.')
if __name__=='__main__':main()
