import itertools
import unittest
import numpy as np
from rim_support import support_signals, footprints_overlap
from release_gate_v4 import ReleaseGateV4


def box(x=(-.05,.05), y=(-.05,.05), z=(0,.06)):
    return np.array(list(itertools.product(x,y,z)))


class SupportTests(unittest.TestCase):
    def test_supported_rim_and_reversed_sensor(self):
        for force, reversed_ in (([[0,0,1]],False),([[0,0,-1]],True)):
            d=support_signals(box(z=(.02,.12)),box(),force,reversed_)
            self.assertTrue(d['rim_support'])
            self.assertEqual(d['support_force_on_banana_w'],[0,0,1])
    def test_noise_boundary_and_lateral_side_touch(self):
        for f in ([[0,0,.1]], [[1,0,.1]], [[0,0,-1]], [[float('nan'),0,1]]):
            self.assertFalse(support_signals(box(z=(0,.1)),box(),f)['rim_support'])
    def test_strict_vertical_and_inclusive_cone(self):
        self.assertTrue(support_signals(box(z=(0,.1)),box(),[[1,0,1]])['rim_support'])
        self.assertFalse(support_signals(box(z=(0,.1)),box(),[[1.001,0,1]])['rim_support'])
    def test_low_side_contact_and_disjoint(self):
        self.assertFalse(support_signals(box(z=(0,.02)),box(),[[0,0,1]])['rim_support'])
        self.assertFalse(support_signals(box(x=(.2,.3)),box(),[[0,0,1]])['rim_support'])
    def test_rotated_hulls_and_touch_only(self):
        a=np.array([[-1,-1],[-1,1],[1,1],[1,-1]])
        rot=np.array([[1,-1],[1,1]])/np.sqrt(2)
        self.assertTrue(footprints_overlap(a,a@rot))
        self.assertFalse(footprints_overlap(a,a+[2,0]))
        self.assertFalse(footprints_overlap(a,a@rot+[4,0]))


def sample(**kw):
    d=dict(finite=True,contact=True,table_contact=False,bowl_contact=False,
           clearance_m=.03,near_bowl=True,closure=.7,bowl_support=False,rim_support=False)
    d.update(kw);return d


class RimGateTests(unittest.TestCase):
    def setUp(self): self.g=ReleaseGateV4(); self.t=0.
    def tick(self,cmd,pre=None,post=None,raw=False,speed=0.):
        pre=sample() if pre is None else pre; post=pre if post is None else post
        self.g.before(self.t,cmd,pre);self.t+=1/15
        return self.g.after(self.t,post,raw,speed,0.)
    def release(self):
        for _ in range(5):self.tick(1.)
        self.assertTrue(self.g.armed)
        self.tick(0.,sample(),sample(contact=False,closure=.1,bowl_contact=True,bowl_support=True),raw=True)
        self.assertEqual(self.g.phase,'released')
    def rim(self,**kw):
        d=sample(contact=False,closure=.1,bowl_contact=True,bowl_support=True,rim_support=True,near_bowl=False)
        d.update(kw);return d
    def test_rim_settles_without_centroid_containment(self):
        self.release()
        for _ in range(14):self.assertFalse(self.tick(0.,self.rim()))
        self.assertTrue(self.tick(0.,self.rim()))
    def test_contact_without_support_never_qualifies(self):
        self.release()
        for _ in range(40):self.assertFalse(self.tick(0.,self.rim(bowl_support=False,rim_support=False,near_bowl=True),raw=True))
    def test_uncommanded_slip_not_rescued_by_rim(self):
        for _ in range(5):self.tick(1.)
        self.tick(1.,sample(),self.rim())
        for _ in range(40):self.assertFalse(self.tick(0.,self.rim(),raw=True))
    def test_bounce_and_brush_reset_duration(self):
        self.release()
        for _ in range(10):self.tick(0.,self.rim())
        self.assertFalse(self.tick(0.,self.rim(),speed=.1))
        self.assertIsNone(self.g.stable.since)
        self.assertFalse(self.tick(0.,self.rim(contact=True)))
        for _ in range(15):self.assertFalse(self.tick(0.,self.rim()))
        self.assertTrue(self.tick(0.,self.rim()))
    def test_regrasp_and_complete_departure_revoke(self):
        for cmd,s in ((1.,self.rim(contact=True)),(0.,self.rim(rim_support=False,bowl_support=False,bowl_contact=False))):
            self.setUp();self.release();self.tick(cmd,s)
            self.assertEqual(self.g.phase,'idle')
            for _ in range(30):self.assertFalse(self.tick(0.,self.rim()))
    def test_supported_native_containment_below_rim(self):
        self.release()
        s=self.rim(rim_support=False,near_bowl=True)
        for _ in range(14):self.assertFalse(self.tick(0.,s,raw=True))
        self.assertTrue(self.tick(0.,s,raw=True))
    def test_nonfinite_revokes_and_no_prior_release(self):
        for _ in range(30):self.assertFalse(self.tick(0.,self.rim(),raw=True))
        self.setUp();self.release();self.tick(0.,self.rim(finite=False),raw=True)
        self.assertEqual(self.g.phase,'idle')
    def test_pre_release_rim_does_not_bypass_initial_geometry(self):
        for _ in range(5):self.tick(1.)
        self.tick(0.,self.rim(contact=True,closure=.7),self.rim())
        self.assertEqual(self.g.phase,'idle')

if __name__=='__main__':unittest.main()
