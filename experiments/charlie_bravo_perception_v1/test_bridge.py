import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from bridge import Collector,PerceptionJudge,optical_from_opengl
from controller import Frame
from build_runner import build
READY=json.dumps(dict(verdict='READY_TO_RELEASE',held=True,central=True,clearance=True,views_agree=True,motion='still',reason='Held above center'))
class BridgeTests(unittest.TestCase):
 def invoke(self,validated=True,missing=False,allowed=True):
  with tempfile.TemporaryDirectory() as tmp:
   c=SimpleNamespace(path=Path(tmp),records={t:{'tick':t,'geometry_allows_request':allowed} for t in (3,6,9)})
   if missing:del c.records[6]
   seen=[]
   def judge(prompt,frames,request):seen.append(prompt);return READY
   response=PerceptionJudge(judge,c,validated)("test",tuple(Frame(t,b'',b'') for t in (3,6,9)),{'tick':9})
   logs=json.loads((Path(tmp)/'judge_events.jsonl').read_text())
   self.assertEqual(logs['raw_model_response'],READY)
   return json.loads(response),seen
 def test_model_ready_and_validated_geometry(self):self.assertEqual(self.invoke()[0]['verdict'],'READY_TO_RELEASE')
 def test_unreviewed_blocks_open(self):self.assertEqual(self.invoke(validated=False)[0]['verdict'],'WAIT')
 def test_missing_history_blocks_open(self):self.assertEqual(self.invoke(missing=True)[0]['verdict'],'WAIT')
 def test_off_center_blocks_open(self):self.assertEqual(self.invoke(allowed=False)[0]['verdict'],'WAIT')
 def test_bad_reply_passes_through_and_is_logged(self):
  # Claude round 3: Runtime then records SchemaError with the raw text instead of losing it as a ValueError.
  with tempfile.TemporaryDirectory() as tmp:
   c=SimpleNamespace(path=Path(tmp),records={})
   self.assertEqual(PerceptionJudge(lambda *a:'garbage',c)("",(),{'tick':1}),'garbage')
   self.assertTrue(json.loads((Path(tmp)/'judge_events.jsonl').read_text())['parse_error'])
 def test_model_not_given_geometry_verdict(self):
  _,prompts=self.invoke()
  self.assertNotIn('geometry_allows_request',prompts[0]);self.assertNotIn('geometry_validated',prompts[0])
 def test_camera_convention(self):np.testing.assert_array_equal(optical_from_opengl(np.eye(4)),np.diag([1,-1,-1,1]))
 def test_collector_missing_depth_shape_refuses(self):
  with tempfile.TemporaryDirectory() as tmp:
   c=Collector(Path(tmp)/'p');v={'rgb':np.zeros((20,20,3),np.uint8),'depth':np.zeros((10,10)),'K':np.eye(3),'world_from_opengl':np.eye(4)}
   with self.assertRaises(ValueError):c.capture(0,{'exterior':v,'wrist':v})
 def test_generated_capture_precedes_paint_and_is_guarded(self):
  with tempfile.TemporaryDirectory() as tmp:
   s=build(tmp);compile(s,'runner','exec')
   self.assertLess(s.index('depth_capture.capture(step,views)'),s.index('painted,written=painter.paint'))
   self.assertIn("if args.mode=='pilot' and judge.permit.get('perception_review_status')!='OK'",s)
   self.assertIn('RUN_TICKS=3600',s)
   self.assertIn('CaptureJudge(args.permit)',s)
   self.assertIn('data.quat_w_ros',s)
   self.assertIn('camera.frame',s)
if __name__=='__main__':unittest.main()
