import unittest
import numpy as np
from perception import backproject, observe
class Tests(unittest.TestCase):
 def test_principal_point_and_off_axis(self):
  d=np.array([[2.,2.],[2.,2.]])
  k=np.array([[2.,0,0],[0,2.,0],[0,0,1.]])
  p=backproject(d,k,np.ones((2,2),bool))
  np.testing.assert_allclose(p,[[0,0,2],[1,0,2],[0,1,2],[1,1,2]])
 def test_invalid_depth_excluded(self):
  d=np.array([[np.nan,0],[np.inf,-1.]])
  self.assertEqual(len(backproject(d,np.eye(3),np.ones((2,2),bool))),0)
 def test_calibration_rejected(self):
  with self.assertRaises(ValueError):backproject(np.ones((2,2)),np.zeros((3,3)),np.ones((2,2),bool))
 def test_rgb_never_claims_height_or_release(self):
  a=np.zeros((20,20,3),dtype=np.uint8);a[:10]=[240,220,0];a[10:]=[130,0,0]
  r=observe(a)
  self.assertTrue(r['objects']['banana']['detected']);self.assertTrue(r['objects']['bowl']['detected'])
  self.assertIsNone(r['height_above_rim_m']);self.assertFalse(r['release_authorized'])
 def test_depth_does_not_imply_held_or_release(self):
  a=np.full((20,20,3),[240,220,0],dtype=np.uint8)
  r=observe(a,np.ones((20,20)),np.eye(3))
  self.assertEqual(r['held'],'unknown');self.assertFalse(r['release_authorized'])
if __name__=='__main__':unittest.main()
