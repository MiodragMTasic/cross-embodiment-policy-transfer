import unittest,tempfile,subprocess,sys,json
from pathlib import Path
import numpy as np
from PIL import Image
from validate_capture import evaluate
class OfflineTests(unittest.TestCase):
 def test_missing_capture_is_no_go(self):
  with tempfile.TemporaryDirectory() as tmp:self.assertFalse(evaluate(Path(tmp))['pilot_ready'])
 def test_replay_prepares_without_api_key_or_model(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);capture=root/'capture';capture.mkdir()
   for tick in (0,6,12):
    (capture/f'{tick:05d}.json').write_text(json.dumps({'tick':tick,'geometry_allows_request':True,'measurement':{'banana_surface_offset_m':.02}}))
    for view in ('exterior','wrist'):Image.fromarray(np.full((24,24,3),[240,220,0],np.uint8)).save(capture/f'{tick:05d}_{view}.png')
   # Claude round 3: replay selects ticks whose executed gripper command was closed.
   (root/'gate_trace.jsonl').write_text(''.join(json.dumps({'step':t+1,'command':1.0})+'\n' for t in (0,6,12)))
   result=subprocess.run([sys.executable,str(Path(__file__).with_name('replay_judge.py')),str(capture),str(root/'replay')],capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stderr)
   report=json.loads((root/'replay/summary.json').read_text());self.assertFalse(report['executed_api'])
   prompt=json.loads((root/'replay/12/request.json').read_text())['prompt']
   self.assertIn('banana_surface_offset_m',prompt);self.assertNotIn('geometry_allows_request',prompt)
if __name__=='__main__':unittest.main()
