"""Exercise supervisor sequencing with fake processes; never starts a VM or shell job."""
import unittest,tempfile,json,sys,time,subprocess
from verify_copy import verify,gpu_snapshot_errors
from pathlib import Path
from unittest.mock import patch
import launch
class LauncherTests(unittest.TestCase):
 def exercise(self,runner_code,snapshot_error=None,execute=True):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);helpers=root/'experiments/bowl_step6';helpers.mkdir(parents=True);[(helpers/n).write_text('') for n in ('run_painted_dev.py','paint_A.py','local_ik.py','rigid_chain.py')];(root/'runs').mkdir();(root/'models').mkdir();(root/'logs').mkdir()
   (root/'models/model-manifest.json').write_text('{"models":[]}')
   (root/'logs/model-loading.json').write_text('{}')
   permit=root/'permit.json';permit.write_text(json.dumps({'scope':'single_vision_attempt','vm_started_unix':time.time()-1,'whole_session_seconds':3600,'checkpoint_manifest_path':'dummy','checkpoint_manifest_sha256':'x','ledger_path':str(root/'runs'/(launch.CASE+'-supervisor')/'api-ledger.jsonl')}))
   calls=[];processes=[];original=Path.read_text
   def read(path,*a,**kw):
    if str(path)=='/proc/uptime':return '1 1'
    return original(path,*a,**kw)
   def command(argv,**kw):
    calls.append(argv)
    if argv==['hostname','-s']:return launch.VM
    if '--query-compute-apps=pid,process_name' in argv and snapshot_error:raise snapshot_error
    if 'rev-parse' in argv:return launch.PINS[Path(argv[2]).name]
    if 'import importlib.metadata' in ' '.join(argv):return '5.1.0.0 2.3.2.post1'
    return ''
   class Proc:
    pid=99999999
    def __init__(self,argv,**kw):
     processes.append((argv,kw));self.server=argv[0]!='timeout';self.returncode=None if self.server else runner_code
     if self.server:kw['stdout'].write('Server is ready and listening\n');kw['stdout'].flush()
     else:
      folder=root/'runs'/launch.CASE;folder.mkdir();(folder/'checks.json').write_text('{"complete":true}')
    def poll(self):return self.returncode
   with patch('transport.VisionJudge'),patch.dict('os.environ',{'NEBIUS_API_KEY':'TEST_ONLY'}),patch.object(launch,'ROOT',root),patch.object(launch,'check',return_value={'launch_ready':True,'blockers':[]}),patch.object(launch,'run',side_effect=command),patch.object(launch,'verify_checkpoint',return_value=root/'checkpoint'),patch.object(launch,'verify_load'),patch.object(launch,'verify_layout'),patch.object(launch,'stop_process'),patch.object(launch.subprocess,'Popen',Proc),patch.object(launch.signal,'signal') as register_signal,patch.object(Path,'home',return_value=Path('/home/miodrag')),patch.object(Path,'read_text',read),patch.object(sys,'argv',['launch',str(permit)]+(['--execute'] if execute else [])):
    if execute and (runner_code or snapshot_error):
     with self.assertRaises(RuntimeError) as caught:launch.main()
     if runner_code:self.assertIn(str(runner_code),str(caught.exception))
    else:self.assertEqual(launch.main(),0)
   if not execute:
    self.assertEqual(processes,[])
    self.assertIn(['sudo','-n','true'],calls)
    self.assertTrue(any('sha256sum' in c for c in calls))
    self.assertFalse(any('shutdown' in c for c in calls))
    self.assertEqual(list((root/'runs').iterdir()),[])
    return
   self.assertIn(launch.signal.SIGHUP,[c.args[0] for c in register_signal.call_args_list])
   # Claude round 3: cleanup runs with interrupts ignored.
   self.assertEqual([c.args for c in register_signal.call_args_list[-3:]],[(s,launch.signal.SIG_IGN) for s in (launch.signal.SIGTERM,launch.signal.SIGINT,launch.signal.SIGHUP)])
   sup=root/'runs'/(launch.CASE+'-supervisor')
   self.assertEqual(verify(sup),[])
   self.assertEqual(verify(root/'runs'/launch.CASE),[])
   self.assertEqual(bool(gpu_snapshot_errors(sup)),bool(snapshot_error))
   self.assertEqual((sup/'exit-code.txt').read_text().strip(),str(runner_code))
   self.assertEqual((sup/'ATTEMPT_COMPLETE').exists(),runner_code==0)
   self.assertTrue((sup/'copy-manifest.json').exists())
   self.assertTrue(any(c[:4]==['sudo','-n','shutdown','-h'] for c in calls))
   argv,kw=processes[-1];self.assertEqual(argv[:4],['timeout','-k','30','2400'])
   self.assertEqual(kw['env']['NEBIUS_API_KEY'],'TEST_ONLY');self.assertNotIn('NEBIUS_API_KEY',processes[0][1]['env']);self.assertEqual(kw['env']['POLICY_SEED'],'0')
 def test_layout_pin_rejects_modified_input(self):
  import layout_pin,hashlib
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);path=root/layout_pin.RELATIVE;path.parent.mkdir(parents=True);path.write_bytes(b'original')
   with patch.object(layout_pin,'EXPECTED',hashlib.sha256(b'original').hexdigest()):
    self.assertEqual(layout_pin.verify_layout(root),path)
    path.write_bytes(b'modified')
    with self.assertRaises(ValueError):layout_pin.verify_layout(root)
 def test_read_only_checks_do_not_launch(self):self.exercise(0,execute=False)
 def test_snapshot_failure_preserves_original(self):self.exercise(124,subprocess.CalledProcessError(1,['nvidia-smi']))
 def test_snapshot_timeout_preserves_original(self):self.exercise(124,subprocess.TimeoutExpired(['nvidia-smi'],60))
 def test_snapshot_failure_blocks_success_readiness(self):self.exercise(0,RuntimeError('snapshot failed'))
 def test_deadline_uses_billing_start_and_boot_tolerance(self):
  permit={'vm_started_unix':1000,'whole_session_seconds':3600}
  for boot in (940,1000,1600):self.assertEqual(launch.session_deadline(permit,boot),4600)
  for boot in (939,1601):
   with self.assertRaises(ValueError):launch.session_deadline(permit,boot)
 def test_completed_capture_is_sealed(self):self.exercise(0)
 def test_timeout_is_retained_as_failure(self):self.exercise(124)
if __name__=='__main__':unittest.main()
