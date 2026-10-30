import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controller import Frame
from transport import MODEL,ENDPOINT,VisionJudge,payload
from build_runner import build, BASE

HERE=Path(__file__).resolve().parents[1]
PNG=b'\x89PNG\r\n\x1a\n'+b'\0'*8+struct.pack('>II',640,360) # serialization mock, not image decoder proof
FRAMES=(Frame(0,PNG,PNG),Frame(6,PNG,PNG))
REQUEST={'tick':6,'phase':'POLICY','deadline_s':10}
REPLY={'model':MODEL,'usage':{'prompt_tokens':1000,'completion_tokens':100},
       'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]}


class TransportTests(unittest.TestCase):
    def setup_judge(self,folder,sender,**overrides):
        p=dict(review_status='OK',human_paid_approval=True,model=MODEL,endpoint=ENDPOINT,
               prices_verified_date=time.strftime('%Y-%m-%d',time.gmtime()),expires_unix=time.time()+60,
               api_cap_usd=.003,reserve_per_call_usd=.003,input_usd_per_million=.3,output_usd_per_million=.9)
        p['ledger_path']=str((Path(folder)/'ledger.jsonl').resolve())
        p.update(overrides)
        permit=Path(folder)/'permit.json';permit.write_text(json.dumps(p))
        return VisionJudge(permit,Path(folder)/'ledger.jsonl',sender,require_key=False)

    def test_payload_actual_images_order_and_output_bound(self):
        p=payload('prompt',FRAMES,REQUEST)
        self.assertEqual(p['model'],MODEL)
        self.assertEqual(p['max_tokens'],300)
        c=p['messages'][1]['content']
        self.assertEqual(sum(x['type']=='image_url' for x in c),4)
        self.assertIn('tick 6, actual wrist',c[-2]['text'])
        with self.assertRaises(ValueError): payload('prompt',[Frame(0,b'bad',PNG)]*2,REQUEST)

    def test_no_approval_no_request(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(PermissionError): self.setup_judge(d,lambda *a:self.fail(),human_paid_approval=False)

    def test_verified_source_lock(self):
        with tempfile.TemporaryDirectory() as d:
            j=self.setup_judge(d,lambda *a:REPLY)
            f=Path(d)/'code.py';f.write_text('x=1')
            lock=Path(d)/'source-lock.json';lock.write_text(json.dumps({'code.py':hashlib.sha256(f.read_bytes()).hexdigest()}))
            j.permit['source_lock_sha256']=hashlib.sha256(lock.read_bytes()).hexdigest()
            j.verify_source_lock(lock)
            f.write_text('x=2')
            with self.assertRaises(PermissionError): j.verify_source_lock(lock)

    def test_usage_settlement(self):
        with tempfile.TemporaryDirectory() as d:
            j=self.setup_judge(d,lambda *a:REPLY)
            self.assertEqual(j('prompt',FRAMES,REQUEST),'{}')
            rows=[json.loads(l) for l in j.ledger.read_text().splitlines()]
            self.assertEqual(rows[0]['state'],'reserved')
            self.assertAlmostEqual(rows[1]['charge_usd'],.00039)
            self.assertNotIn('Authorization',j.ledger.read_text())

    def test_timeout_reservation_persists_across_restart(self):
        def fail(*a): raise TimeoutError()
        with tempfile.TemporaryDirectory() as d:
            j=self.setup_judge(d,fail)
            with self.assertRaises(TimeoutError): j('prompt',FRAMES,REQUEST)
            j=self.setup_judge(d,lambda *a:self.fail('must not call'))
            with self.assertRaises(PermissionError): j('prompt',FRAMES,REQUEST)

    def test_bad_model_or_usage_does_not_refund(self):
        for bad in [{**REPLY,'model':'other'}, {**REPLY,'usage':{}}]:
            with tempfile.TemporaryDirectory() as d:
                j=self.setup_judge(d,lambda *a:bad)
                with self.assertRaises(ValueError): j('prompt',FRAMES,REQUEST)
                self.assertEqual(len(j.ledger.read_text().splitlines()),1)

    def test_underestimated_charge_disables_transport(self):
        with tempfile.TemporaryDirectory() as d:
            r={**REPLY,'usage':{'prompt_tokens':100000,'completion_tokens':100}}
            j=self.setup_judge(d,lambda *a:r)
            with self.assertRaises(ValueError): j('prompt',FRAMES,REQUEST)
            with self.assertRaises(PermissionError): j('prompt',FRAMES,REQUEST)

    def test_worker_timeout_is_bounded(self):
        from unittest.mock import patch
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            j=self.setup_judge(d,None)
            with patch.dict(os.environ,{'NEBIUS_API_KEY':'TEST_ONLY'}), patch('transport.subprocess.run',side_effect=subprocess.TimeoutExpired('worker',.1)) as run:
                with self.assertRaises(subprocess.TimeoutExpired): j._send({},.1)
                self.assertEqual(run.call_args.kwargs['timeout'],.1)
                self.assertNotIn('TEST_ONLY',str(run.call_args.args))


class RunnerTests(unittest.TestCase):
    def built(self):
        with tempfile.TemporaryDirectory() as d:
            return build(d).read_text()

    def test_build_does_not_mutate_source(self):
        before=BASE.read_bytes();ast.parse(self.built())
        self.assertEqual(before,BASE.read_bytes())

    def test_executed_command_scored_and_original_logged(self):
        s=self.built()
        self.assertIn('gate.before(t_pre,float(executed_grip),pre)',s)
        self.assertIn("'command':float(executed_grip)",s)
        self.assertIn("'policy_command':float(predicted[7])",s)
        self.assertNotIn('gate.before(t_pre,float(predicted[7])',s)
        self.assertIn("'P0_action':predicted",s)

    def test_same_episode_limits_and_bounded_intervention(self):
        s=self.built()
        self.assertIn('RUN_TICKS=3600',s)
        self.assertIn("enabled=args.condition=='assisted'",s)
        self.assertIn('if passed:success_step=',s)
        self.assertIn('override=(tuple(command)',s)
        self.assertIn('if override is not None:',s)
        self.assertIn('force_policy_request or finishing_open',s)

    def test_real_frames_and_source_gate_before_simulator(self):
        s=self.built()
        self.assertLess(s.index('judge.verify_source_lock'),s.index('app=AppLauncher'))
        start=s.index('        def capture_drop():');end=s.index('        was_hold=',start)
        capture=s[start:end]
        self.assertNotIn('last_painted',capture.replace('# Current actual images; never last_painted. No extra env.step/render.',''))
        self.assertNotIn('env.step(',capture)
        self.assertIn("encode_views(images[",capture)

if __name__=='__main__': unittest.main()
