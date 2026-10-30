import unittest,json,tempfile
from pathlib import Path
from unittest.mock import patch
import preflight
class PreflightTests(unittest.TestCase):
 def valid(self,folder):
  (folder/'source-lock.json').write_text('{}')
  return {'scope':'single_capture','review_status':'OK','human_paid_approval':True,'expires_unix':5000,
   'api_cap_usd':0,'current_cloud_balance_usd':3.14,'balance_checked_unix':990,'cloud_spend_cap_usd':2,
   'hourly_compute_usd':1.8,'whole_session_seconds':3600,'stop_owner':'Miodrag','console_stop_timer_confirmed':True,
   'guest_halt_backstop_approved':True,'vm_id':preflight.VM,'vm_started_unix':950,
   'checkpoint_manifest_sha256':'a'*64,'checkpoint_manifest_path':'/tmp/p1.json','source_lock_sha256':preflight.digest(folder/'source-lock.json')}
 def test_capture_does_not_require_its_own_completed_review(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);v=self.valid(p)
   with patch.object(preflight,'P',p):self.assertTrue(preflight.check(v,1000)['launch_ready'])
 def test_every_material_guard_blocks(self):
  mutations={'scope':'twelve_runs','review_status':'PENDING','human_paid_approval':False,'expires_unix':0,
    'api_cap_usd':.01,'current_cloud_balance_usd':1,'balance_checked_unix':-4000,'hourly_compute_usd':4,
    'whole_session_seconds':4000,'stop_owner':'nobody','console_stop_timer_confirmed':False,
    'guest_halt_backstop_approved':False,'vm_id':'wrong','vm_started_unix':0,'source_lock_sha256':'wrong'}
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);v=self.valid(p)
   with patch.object(preflight,'P',p):
    for key,value in mutations.items():
     with self.subTest(key=key):self.assertFalse(preflight.check({**v,key:value},1000)['launch_ready'])
 def test_pilot_cannot_skip_capture_and_replay(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);v={**self.valid(p),'scope':'single_pilot'}
   with patch.object(preflight,'P',p):self.assertFalse(preflight.check(v,1000)['launch_ready'])
 def test_empty_and_malformed(self):
  for x in ({},{'hourly_compute_usd':'bad'},{'current_cloud_balance_usd':float('nan')}):self.assertFalse(preflight.check(x,1000)['launch_ready'])
if __name__=='__main__':unittest.main()
