"""Claude round-3 tests: V1 availability, V2 calibration instant, V3 unverified, V4 POLICY replay, judge logging."""
import unittest,tempfile,subprocess,sys,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from validate_capture import evaluate
from bridge import PerceptionJudge
from score_judge_log import score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'charlie_bravo_v1'))
from controller import Frame

READY=json.dumps(dict(verdict='READY_TO_RELEASE',held=True,central=True,clearance=True,views_agree=True,motion='still',reason='ok'))
BOWL=np.array([.5,0.,.05])


def make_run(root,ticks,held,measured,calibration_tick=0,startup=True,bowl_shift=0.):
    run=root/'run';(run/'perception').mkdir(parents=True)
    steps=np.arange(1,max(ticks)+2)
    objects=np.zeros((len(steps),2,13));objects[:,1,:3]=BOWL;objects[:,1,0]+=bowl_shift
    objects[:,0,:3]=BOWL+[0,0,.12]
    np.savez(run/'tracking.npz',step=steps,objects_banana_bowl=objects)
    rows=[{'step':int(s),'command':1.0 if s-1 in held else 0.0,'policy_command':1.0 if s-1 in held else 0.0,
           'post':{'contact':s in held,'table_contact':False,'bowl_contact':False}} for s in steps]
    (run/'gate_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if startup:(run/'startup_after.json').write_text(json.dumps({'episode_step':1,'objects':{'bowl':BOWL.tolist()+[1,0,0,0]}}))
    for t in ticks:
        view={'pose_max_error':0.,'masks':{'banana':{'pixels':500,'valid_fraction':1.},'bowl':{'pixels':900,'valid_fraction':1.}},
              'banana_surface':{'points':400,'median_world_m':(BOWL+[0,0,.12]).tolist(),'low_z_p05_m':.15}}
        m={'status':'draft_measurement','calibration_tick':calibration_tick,'bowl_center_xy_m':BOWL[:2].tolist(),'rim_height_m':BOWL[2]+.0275,
           'banana_surface_median_world_m':(BOWL+[0,0,.12]).tolist(),'banana_low_surface_height_above_rim_m':.05,'geometry_allows_request':True} \
          if t in measured else {'status':'unknown','geometry_allows_request':False,'reason':'missing calibration or visible surfaces'}
        (run/'perception'/f'{t:05d}.json').write_text(json.dumps({'tick':t,'views':{'exterior':view,'wrist':view},'measurement':m,
            'geometry_allows_request':m['geometry_allows_request'],'capture_seconds':.2}))
    return run


class ValidationTests(unittest.TestCase):
    def test_a4_needs_thirty_measured_held_frames_and_reports_unknowns(self):
        ticks=list(range(0,240,6))
        with tempfile.TemporaryDirectory() as tmp:
            run=make_run(Path(tmp),ticks,held=set(ticks[1:]),measured=set(ticks[:20]))
            r=evaluate(run)
            self.assertFalse(r['checks']['A4_held_banana_centroid'])
            a=r['availability'];self.assertEqual(a['held_measured'],19);self.assertEqual(a['held_frames'],39)
            self.assertEqual(a['held_unknown_reasons'],{'missing calibration or visible surfaces':20})
        with tempfile.TemporaryDirectory() as tmp:
            run=make_run(Path(tmp),ticks,held=set(ticks[1:]),measured=set(ticks))
            r=evaluate(run);self.assertTrue(r['checks']['A4_held_banana_centroid']);self.assertEqual(r['availability']['held_measured_fraction'],1.)

    def test_a3_uses_calibration_instant_not_later_pose(self):
        ticks=list(range(0,60,6))
        with tempfile.TemporaryDirectory() as tmp:
            r=evaluate(make_run(Path(tmp),ticks,held=set(),measured=set(ticks),startup=False))
            self.assertIsNone(r['checks']['A3_bowl_calibration'])
            self.assertTrue(r['bowl_calibration']['status'].startswith('unknown'))
        with tempfile.TemporaryDirectory() as tmp:
            r=evaluate(make_run(Path(tmp),ticks,held=set(),measured=set(ticks)))
            self.assertTrue(r['checks']['A3_bowl_calibration']);self.assertEqual(r['bowl_calibration']['tick'],0)
        with tempfile.TemporaryDirectory() as tmp:
            # Tracked bowl sits 1 cm away from the startup pose: calibration (tick 0) still scored at tick 0; shift shows as drift.
            r=evaluate(make_run(Path(tmp),ticks,held=set(),measured=set(ticks),bowl_shift=.01))
            self.assertTrue(r['checks']['A3_bowl_calibration']);self.assertAlmostEqual(r['bowl_drift_after_calibration_max_m'],.01)

    def test_a2_unverified_without_reference_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=evaluate(make_run(Path(tmp),[0,6],held=set(),measured={0,6}))
            self.assertIsNone(r['checks']['A2_startup_image_comparison'])
            self.assertIn('A2_startup_image_comparison (unknown)',r['blockers']);self.assertFalse(r['pilot_ready'])


class ReplayTests(unittest.TestCase):
    def test_replay_selects_closed_executed_ticks_with_policy_phase(self):
        ticks=list(range(0,120,6));closed={t for t in ticks if t>=60}
        with tempfile.TemporaryDirectory() as tmp:
            run=make_run(Path(tmp),ticks,held=closed,measured=set(ticks))
            for t in ticks:
                for v in ('exterior','wrist'):Image.fromarray(np.full((360,640,3),[240,220,0],np.uint8)).save(run/'perception'/f'{t:05d}_{v}.png')
            out=Path(tmp)/'replay'
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('replay_judge.py')),str(run/'perception'),str(out),'--limit','5'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['phase'],'POLICY');self.assertFalse(summary['hold_confirmation']['tested'])
            self.assertTrue(all(row['tick'] in closed for row in summary['rows']));self.assertEqual(len(summary['rows']),5)
            request=json.loads((out/str(summary['rows'][0]['tick'])/'request.json').read_text())['request']
            self.assertEqual(request['phase'],'POLICY')

    def test_replay_refuses_without_executed_command_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture=Path(tmp)/'perception';capture.mkdir()
            (capture/'00000.json').write_text(json.dumps({'tick':0,'measurement':{},'geometry_allows_request':False}))
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('replay_judge.py')),str(capture),str(Path(tmp)/'out')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('gate_trace',result.stderr)


class JudgeLoggingTests(unittest.TestCase):
    def judge(self,tmp,inner):
        collector=SimpleNamespace(path=Path(tmp),records={6:{'tick':6,'geometry_allows_request':True,'measurement':{}}})
        return PerceptionJudge(inner,collector,validated=False),Path(tmp)/'judge_events.jsonl'

    def test_malformed_reply_is_logged_and_passed_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            j,log=self.judge(tmp,lambda *a:'```json not json')
            self.assertEqual(j('p',[Frame(6,b'e',b'w')],{'tick':6,'phase':'POLICY'}),'```json not json')
            event=json.loads(log.read_text());self.assertTrue(event['parse_error']);self.assertEqual(event['raw_model_response'],'```json not json')

    def test_transport_error_is_logged_then_raised(self):
        def late(*a):raise TimeoutError('late reply')
        with tempfile.TemporaryDirectory() as tmp:
            j,log=self.judge(tmp,late)
            with self.assertRaises(TimeoutError):j('p',[Frame(6,b'e',b'w')],{'tick':6,'phase':'HOLD'})
            self.assertEqual(json.loads(log.read_text())['error_type'],'TimeoutError')

    def test_unvalidated_ready_is_vetoed_and_marked(self):
        with tempfile.TemporaryDirectory() as tmp:
            j,log=self.judge(tmp,lambda *a:READY)
            self.assertEqual(json.loads(j('p',[Frame(6,b'e',b'w')],{'tick':6,'phase':'POLICY'}))['verdict'],'WAIT')
            self.assertTrue(json.loads(log.read_text())['vetoed'])


class ScoreTests(unittest.TestCase):
    def test_counts_errors_failed_confirmations_and_gripper_violations(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);drop=run/'drop';(run/'perception').mkdir();drop.mkdir()
            calls=[(1,100,'POLICY',None,'HOLD'),(2,109,'HOLD','TimeoutError','POLICY'),(3,120,'POLICY','SchemaError','POLICY'),
                   (4,132,'POLICY',None,'HOLD'),(5,141,'HOLD',None,'OPEN')]
            for call,tick,phase,error,state in calls:
                q=drop/f'query_{call:04d}';q.mkdir()
                (q/'request.json').write_text(json.dumps({'request':{'call':call,'tick':tick,'phase':phase}}))
                (q/'response.json').write_text(json.dumps({'raw':'','error_type':error,'state':state}))
            q=drop/'query_0006';q.mkdir();(q/'request.json').write_text(json.dumps({'request':{'call':6,'tick':300,'phase':'POLICY'}}))
            states={**{t:'POLICY' for t in range(0,100)},**{t:'HOLD' for t in range(100,109)},**{t:'POLICY' for t in range(109,132)},
                    **{t:'HOLD' for t in range(132,141)},**{t:'OPEN' for t in range(141,231)},**{t:'DONE' for t in range(231,300)}}
            (drop/'events.jsonl').write_text(''.join(json.dumps({'tick':t,'proposed_override':{'state':s,'grip':0,'stop':False}})+'\n' for t,s in states.items()))
            grip={t:(1.0 if 50<=t<141 else 0.0) for t in range(300)};grip[250]=1.0
            (run/'gate_trace.jsonl').write_text(''.join(json.dumps({'step':t+1,'command':g,'policy_command':1.0 if t>=50 else 0.0})+'\n' for t,g in grip.items()))
            (run/'perception'/'judge_events.jsonl').write_text(json.dumps({'tick':120,'parse_error':True,'raw_model_response':'x','vetoed':False})+'\n'
                +json.dumps({'tick':109,'error_type':'TimeoutError'})+'\n'+json.dumps({'tick':100,'parse_error':False,'raw_model_response':READY,'vetoed':False})+'\n')
            r=score(run)
            self.assertEqual(r['queries'],6);self.assertEqual(r['timeouts'],1);self.assertEqual(r['schema_errors'],1)
            self.assertEqual(r['response_errors']['no_response_file'],1)
            self.assertEqual(r['hold_entries'],2);self.assertEqual(r['failed_confirmations'],1);self.assertEqual(r['confirmed_opens'],1)
            self.assertEqual(r['model_verdicts'],{'malformed':1,'transport_error':1,'READY_TO_RELEASE':1})
            self.assertEqual(r['executed_gripper']['first_executed_open_after_close'],141)
            self.assertEqual(r['executed_gripper']['state_command_violations'],[250])


if __name__=='__main__':unittest.main()
