import unittest,ast
from pathlib import Path
import numpy as np
from perception import geometry_evidence,camera_to_world,BowlReference,fit_bowl
class GeometryTests(unittest.TestCase):
 def fixture(self,offset=0):
  t=np.linspace(-np.pi,np.pi,400,endpoint=False)
  rim=np.column_stack([.08*np.cos(t),.08*np.sin(t),np.full(len(t),.1)])
  banana=np.column_stack([np.linspace(-.089,.089,200)+offset,np.zeros(200),np.full(200,.16)])
  return banana,rim
 def test_real_size_centered_flat_banana_passes(self):
  a,b=self.fixture();self.assertTrue(geometry_evidence(a,b)['geometry_allows_request'])
 def test_all_near_rim_offsets_veto(self):
  for offset in (.05,.07,.08,.10,.12):
   for sign in (-1,1):
    a,b=self.fixture(offset*sign);self.assertFalse(geometry_evidence(a,b)['geometry_allows_request'])
 def test_resting_and_high_veto(self):
  for z in (.10,.11,.22):
   a,b=self.fixture();a[:,2]=z;self.assertFalse(geometry_evidence(a,b)['geometry_allows_request'])
 def test_partial_arc_cannot_initialize(self):
  a,b=self.fixture();self.assertEqual(fit_bowl(b[:140])['status'],'unknown')
 def test_partial_view_after_initialization_can_measure(self):
  a,b=self.fixture();ref=BowlReference();ref.measure(0,a,b)
  self.assertTrue(ref.measure(100,a,b[:140])['geometry_allows_request'])
 def test_moved_bowl_veto_and_no_recalibration(self):
  a,b=self.fixture();ref=BowlReference();ref.measure(0,a,b)
  moved=b.copy();moved[:,0]+=.04
  self.assertFalse(ref.measure(12,a,moved)['geometry_allows_request'])
  self.assertEqual(ref.reference['calibration_tick'],0)
 def test_late_calibration_forbidden(self):
  a,b=self.fixture();self.assertFalse(BowlReference().measure(61,a,b)['geometry_allows_request'])
 def test_lip_band_excludes_wall(self):
  a,b=self.fixture();wall=b.copy();wall[:,:2]*=.8;wall[:,2]-=.02
  f=fit_bowl(np.concatenate([b,wall]));self.assertAlmostEqual(f['rim_height_m'],.1);self.assertAlmostEqual(f['rim_radius_m'],.08)
 def test_single_high_red_outlier_does_not_destroy_fit(self):
  a,b=self.fixture()
  for height in (.11,.15,.4):
   fit=fit_bowl(np.vstack([b,[.4,.4,height]]))
   self.assertEqual(fit['status'],'fitted');self.assertAlmostEqual(fit['rim_height_m'],.1)
 def test_empty(self):self.assertFalse(geometry_evidence([],[])['geometry_allows_request'])
 def test_transform_translation_and_reflection(self):
  t=np.eye(4);t[:3,3]=[1,2,3];np.testing.assert_allclose(camera_to_world([[0,0,1]],t),[[1,2,4]])
  t[0,0]=-1
  with self.assertRaises(ValueError):camera_to_world([[0,0,1]],t)
 def test_runtime_perception_has_no_ground_truth_dependency(self):
  for name in ('perception.py','bridge.py'):
   source=Path(__file__).with_name(name).read_text();tree=ast.parse(source)
   imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
   self.assertFalse(any('isaac' in n or 'scor' in n or 'validate_capture' in n for n in imports))
   for forbidden in ('root_state_w','objects_banana_bowl','banana_force_on_tips','gate_trace'):
    self.assertNotIn(forbidden,source)
if __name__=='__main__':unittest.main()
