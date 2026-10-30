import json
from pathlib import Path
import tempfile
import unittest
from test_controller import READY, WAIT, POSE, TARGET, frames
from controller import Config, State, Supervisor
from runtime import Runtime

class ReviewRegressions(unittest.TestCase):
    def runtime(self, folder, judge, enabled=True):
        return Runtime(Path(folder)/'run', judge, enabled=enabled, config=Config(max_calls=300,api_cap=1), clock=lambda:0.)

    def test_hold_uses_command_and_confirmation_images_are_all_new(self):
        seen=[]
        def judge(prompt, fs, request):
            seen.append((request.copy(),[f.tick for f in fs]))
            return READY
        with tempfile.TemporaryDirectory() as d:
            r=self.runtime(d,judge)
            for t in range(16): result=r.step(t,POSE,TARGET,1,lambda:(b'e',b'w'))
            self.assertEqual(result.arm,TARGET)
            self.assertEqual(result.grip,0)
            self.assertEqual(seen[1][1],[9,12,15])
            self.assertTrue(all(t>seen[0][0]['tick'] for t in seen[1][1]))

    def test_open_never_recloses_and_arm_resumes_after_settling(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.runtime(d,lambda *a:READY)
            for t in range(150):
                cmd=r.step(t,POSE,TARGET,1,lambda:(b'e',b'w'))
                if t>=15:self.assertEqual(cmd.grip,0)
            self.assertEqual(r.controller.state,State.DONE)
            self.assertFalse(cmd.stop)

    def test_moving_initial_ready_can_hold_but_not_confirm(self):
        c=Supervisor()
        moving=READY.replace('"still"','"inward"')
        c.advance(1,0,POSE,TARGET,1);c.request(frames(1),1,0);c.receive(moving,1,1,TARGET)
        self.assertEqual(c.state,State.HOLD)
        c.advance(10,2,POSE,TARGET,1);c.request(frames(10),1,2);c.receive(moving,3,10,TARGET)
        self.assertEqual(c.state,State.POLICY)
        self.assertFalse(c.used)

    def test_baseline_never_calls_judge(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.runtime(d,lambda *a:self.fail('baseline called network'),False)
            for t in range(40):
                cmd=r.step(t,POSE,TARGET,1,lambda:(b'e',b'w'))
                self.assertEqual(cmd.grip,1)
            self.assertGreater(len(list(r.output.glob('query_*/request.json'))),0)

    def test_permission_failure_stops_immediately(self):
        def fail(*a):raise PermissionError('SECRET')
        with tempfile.TemporaryDirectory() as d:
            r=self.runtime(d,fail)
            for t in range(7):cmd=r.step(t,POSE,TARGET,1,lambda:(b'e',b'w'))
            self.assertTrue(cmd.stop)
            self.assertEqual(cmd.reason,'transport_PermissionError')
            self.assertNotIn('SECRET',(r.output/'events.jsonl').read_text())

    def test_three_transport_errors_stop(self):
        def fail(*a):raise TimeoutError()
        with tempfile.TemporaryDirectory() as d:
            r=self.runtime(d,fail)
            for t in range(31):cmd=r.step(t,POSE,TARGET,1,lambda:(b'e',b'w'))
            self.assertTrue(cmd.stop)
            self.assertEqual(r.consecutive_errors,3)

    def test_old_confirmation_frame_is_rejected(self):
        c=Supervisor();c.advance(1,0,POSE,TARGET,1);c.request(frames(1),1,0);c.receive(READY,1,1,TARGET)
        c.advance(7,2,POSE,TARGET,1)
        self.assertIsNone(c.request([frames(1)[-1],frames(7)[-1]],1,2))

class RoundTwoRegressions(unittest.TestCase):
    def test_three_failed_confirmations_continue_valid_without_more_calls(self):
        calls=[]
        def judge(prompt,fs,request):
            calls.append(request.copy())
            return WAIT if request['phase']=='HOLD' else READY
        with tempfile.TemporaryDirectory() as d:
            r=Runtime(Path(d)/'run',judge,config=Config(max_calls=306,api_cap=1),clock=lambda:0.)
            for tick in range(500):
                cmd=r.step(tick,POSE,TARGET,1,lambda:(b'e',b'w'))
                self.assertFalse(cmd.stop)
                self.assertEqual(cmd.grip,1)
            self.assertTrue(r.controller.exhausted)
            self.assertFalse(r.controller.used) # must not activate permanent open
            self.assertIsNone(r.controller.invalid_reason)
            self.assertEqual(len(calls),6)
            self.assertTrue(all(x['hold_since_tick'] is None for x in calls if x['phase']=='POLICY'))

    def test_three_malformed_responses_are_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as d:
            r=Runtime(Path(d)/'run',lambda *a:'{}',clock=lambda:0.)
            for tick in range(31):cmd=r.step(tick,POSE,TARGET,1,lambda:(b'e',b'w'))
            self.assertTrue(cmd.stop)
            self.assertEqual(cmd.reason,'transport_SchemaError')

if __name__=='__main__':unittest.main()
