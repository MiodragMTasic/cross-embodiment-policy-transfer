import hashlib,json,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import preflight
from transport import VisionJudge,MODEL,ENDPOINT
P=Path(__file__).resolve().parents[1]
class AttemptTests(unittest.TestCase):
 def permit(self,lock):
  p=json.loads((P/'permit.example.json').read_text())
  p.update(review_status='OK',human_paid_approval=True,expires_unix=2000,balance_checked_unix=1000,current_cloud_balance_usd=2.50,hourly_compute_usd=1.8,vm_started_unix=999,console_stop_timer_confirmed=True,guest_halt_backstop_approved=True,source_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
  return p
 def test_scope_budget_and_lock_guard(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'source-lock.json').write_text('{}');p=self.permit(root/'source-lock.json')
   with patch.object(preflight,'P',root):
    self.assertTrue(preflight.check(p,1001)['launch_ready'])
    for update in ({'scope':'single_capture'},{'current_cloud_balance_usd':1},{'human_paid_approval':False},{'accept_unvalidated_vision':False},{'vm_started_unix':0},{'expires_unix':999},{'source_lock_sha256':'wrong'}):
     self.assertFalse(preflight.check({**p,**update},1001)['launch_ready'],update)
 def test_checking_transport_does_not_create_supervisor_directory(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ledger=root/'supervisor/api-ledger.jsonl';p=root/'permit.json'
   p.write_text(json.dumps(dict(review_status='OK',human_paid_approval=True,model=MODEL,endpoint=ENDPOINT,prices_verified_date=time.strftime('%Y-%m-%d',time.gmtime()),expires_unix=time.time()+60,api_cap_usd=1,reserve_per_call_usd=.003,input_usd_per_million=.3,output_usd_per_million=.9,ledger_path=str(ledger.resolve()))))
   VisionJudge(p,ledger,require_key=False)
   self.assertFalse(ledger.parent.exists())
 def test_runner_single_case_and_executed_release_marker(self):
  import ast
  source=(P/'runner_B.py').read_text();ast.parse(source)
  self.assertIn("assert args.condition=='assisted'",source)
  self.assertIn("'roger_B_P1_l4003_assisted_v1'",source)
  self.assertLess(source.index('obs,_,term,trunc,_=env.step(action)'),source.index("save('release-command.json'"))
  self.assertIn("gate.before(t_pre,float(executed_grip),pre)",source)
  self.assertIn("'policy_command':float(predicted[7])",source)
  self.assertIn("'task_success':False",source)
if __name__=='__main__':unittest.main()
