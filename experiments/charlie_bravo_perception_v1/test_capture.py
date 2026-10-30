import unittest,tempfile,json
from pathlib import Path
import numpy as np
from bridge import Collector
from launch_capture import seal,verify_load
from verify_copy import verify
class CaptureTests(unittest.TestCase):
 def view(self,frame=0):
  return {'rgb':np.full((24,24,3),[240,220,0],np.uint8),'depth':np.full((24,24),.71234,np.float32),
     'K':np.array([[100,0,12],[0,100,12],[0,0,1]]),'world_from_opengl':np.eye(4),
     'world_from_ros':np.diag([1,-1,-1,1]),'frame_counter':frame}
 def test_capture_roundtrip_and_stale_frame(self):
  with tempfile.TemporaryDirectory() as tmp:
   c=Collector(Path(tmp)/'capture');v=self.view();r=c.capture(0,{'exterior':v,'wrist':v})
   a=np.load(c.path/'00000.npz');self.assertEqual(a['exterior_depth_mm'].dtype,np.uint16)
   self.assertLess(np.max(abs(a['exterior_depth_mm']/1000-v['depth'])),.00051)
   self.assertTrue((c.path/'00000_exterior.png').exists());self.assertEqual(r['views']['exterior']['frame_counter'],0)
   with self.assertRaises(ValueError):c.capture(6,{'exterior':v,'wrist':v})
 def test_pose_mismatch_and_cap_fail_before_write(self):
  for kind in ('pose','cap'):
   with tempfile.TemporaryDirectory() as tmp:
    c=Collector(Path(tmp)/'capture',byte_cap=1 if kind=='cap' else 4000000);v=self.view()
    if kind=='pose':v['world_from_ros']=np.eye(4)
    with self.assertRaises((ValueError,RuntimeError)):c.capture(0,{'exterior':v,'wrist':v})
    self.assertEqual(list(c.path.iterdir()),[])
 def test_invalid_depth_retained_as_zero_and_reported(self):
  with tempfile.TemporaryDirectory() as tmp:
   c=Collector(Path(tmp)/'capture');v=self.view();v['depth'][0]=np.nan
   r=c.capture(0,{'exterior':v,'wrist':v})
   self.assertLess(r['views']['exterior']['masks']['banana']['valid_fraction'],1)
   self.assertTrue((np.load(c.path/'00000.npz')['exterior_depth_mm'][0]==0).all())
 def test_copy_hash_detects_video_corruption_and_extra_files(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'video.mp4').write_bytes(b'video');seal(p);self.assertEqual(verify(p),[])
   (p/'video.mp4').write_bytes(b'wrong');self.assertIn('video.mp4',verify(p))
   (p/'extra').write_text('x');self.assertIn('extra',verify(p))
 def test_early_storage_forecast_preserves_tenth_capture(self):
  with tempfile.TemporaryDirectory() as tmp:
   c=Collector(Path(tmp)/'capture',byte_cap=1_000_000)
   for tick in range(9):
    v=self.view(tick);r=c.capture(tick,{'exterior':v,'wrist':v})
    self.assertIn('median_world_m',r['views']['wrist']['banana_surface'])
   v=self.view(9)
   with self.assertRaisesRegex(RuntimeError,'forecast'):c.capture(9,{'exterior':v,'wrist':v})
   r=json.loads((c.path/'00009.json').read_text())
   self.assertGreater(r['storage_projection']['bytes'],r['storage_projection']['abort_threshold_bytes'])
   self.assertTrue((c.path/'00009.npz').exists())
 def test_model_load_rejects_mismatched_weights(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'loading.json';cp=Path('/checkpoint');r={'seed':0,'policy_label':'disposable copy','effective_checkpoint':str(cp),
     'p0_snapshot':'/models/05e7cc97e40dbd33b0890c35cc0214fcb0547ab5','cosmos_revision':'9ce19a195e423419c349abfc86fd07178b230561',
     'loads':[{'label':'disposable copy','path':str(cp),'loading_info':{}},{'label':'Cosmos','loading_info':{}}]}
   p.write_text(json.dumps(r));verify_load(p,cp,0)
   r['loads'][0]['loading_info']['missing_keys']=['action_head'];p.write_text(json.dumps(r))
   with self.assertRaises(ValueError):verify_load(p,cp,0)
if __name__=='__main__':unittest.main()
