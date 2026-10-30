"""VM-only, single capture supervisor. Default is read-only --check; --execute is explicit.
Never starts/restarts a cloud VM and never calls Token Factory. Console stop is human-owned.
"""
import argparse,json,os,signal,subprocess,time,math,shutil
from pathlib import Path
from preflight import check,digest,VM
from layout_pin import verify_layout
P=Path(__file__).resolve().parent
ROOT=Path('/home/miodrag/robot-transfer/bowl')
CASE='charlie_depth_B_P1_l4003_capture_v1'
PINS={'RoboLab':'b6cfcf6ccd2113c4cd1216d35b33024a7d5691da','Isaac-GR00T':'dac54b313160c201c450f97df0fd74ebd379c892'}

def run(argv,**kwargs):return subprocess.check_output(argv,text=True,timeout=kwargs.pop('timeout',60),**kwargs).strip()

def verify_checkpoint(path,expected):
    if digest(path)!=expected:raise ValueError('P1 manifest mismatch')
    m=json.loads(Path(path).read_text());d=Path(m['checkpoint_dir'])
    if d!=ROOT/'models/step7_train_v1/checkpoint-1500':raise ValueError('wrong P1 checkpoint')
    listed={f['path'] for f in m['files']}
    if listed!={str(p.relative_to(d)) for p in d.rglob('*') if p.is_file()}:raise ValueError('P1 file set mismatch')
    for f in m['files']:
        path=d/f['path']
        if not path.resolve().is_relative_to(d.resolve()) or path.stat().st_size!=f['size'] or digest(path)!=f['sha256']:raise ValueError('P1 file integrity failed')
    return d

def verify_load(path,checkpoint,started):
    if path.stat().st_mtime<started:raise ValueError('stale loading report')
    r=json.loads(path.read_text())
    if r['seed']!=0 or r['policy_label']!='disposable copy' or r['effective_checkpoint']!=str(checkpoint):raise ValueError('wrong loaded policy')
    if r['cosmos_revision']!='9ce19a195e423419c349abfc86fd07178b230561':raise ValueError('Cosmos revision mismatch')
    if Path(r['p0_snapshot']).name!='05e7cc97e40dbd33b0890c35cc0214fcb0547ab5':raise ValueError('base revision mismatch')
    if {x['label'] for x in r['loads']}!={'disposable copy','Cosmos'}:raise ValueError('load set mismatch')
    if [x['path'] for x in r['loads'] if x['label']=='disposable copy']!=[str(checkpoint)]:raise ValueError('loaded checkpoint mismatch')
    for item in r['loads']:
        if any(item['loading_info'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('model loading errors')

def stop_process(p):
    if p is None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait(timeout=5)

def seal(folder):
    files=[]
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():raise ValueError('symlink in evidence')
        if path.is_file() and path.name!='copy-manifest.json':
            files.append({'path':str(path.relative_to(folder)),'bytes':path.stat().st_size,'sha256':digest(path)})
    (folder/'copy-manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')

def session_deadline(permit,boot):
    started=permit['vm_started_unix']
    if not started-60 <= boot <= started+600:
        raise ValueError('VM start timestamp disagrees with uptime')
    return started+permit['whole_session_seconds']


def vm_preflight(permit):
    if Path.home()!=Path('/home/miodrag') or run(['hostname','-s'])!=VM:raise RuntimeError('wrong VM: refusing execution')
    run(['sudo','-n','true'])
    uptime=float(Path('/proc/uptime').read_text().split()[0]);boot=time.time()-uptime
    deadline=session_deadline(permit,boot)
    if deadline-time.time()<3000:raise RuntimeError('Less than 50 minutes remain: do not begin loading models')
    if run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']):raise RuntimeError('GPU already in use')
    if shutil.disk_usage(ROOT).free<8_000_000_000:raise RuntimeError('Need at least 8 GB free for evidence and copy staging')
    if (ROOT/'runs'/CASE).exists() or (ROOT/'runs'/(CASE+'-supervisor')).exists():raise RuntimeError('No reruns/overwrite under same case id')
    for repo,pin in PINS.items():
        if run(['git','-C',str(ROOT/repo),'rev-parse','HEAD'])!=pin:raise ValueError('repository pin mismatch')
        dirty=run(['git','-C',str(ROOT/repo),'status','--porcelain','--untracked-files=all'])
        if any(not line.startswith('?? output/') for line in dirty.splitlines()):raise ValueError('repository dirty')
    ignored=run(['git','-C',str(ROOT/'RoboLab'),'status','--porcelain','--ignored','--untracked-files=all','--','robolab/tasks'])
    if any('__pycache__/' not in line and not line.endswith('.pyc') for line in ignored.splitlines()):raise ValueError('unreviewed task registrations')
    py=ROOT/'RoboLab/.venv/bin/python';policy_py=ROOT/'Isaac-GR00T/.venv/bin/python'
    if run([str(py),'-c','import importlib.metadata as m;print(m.version("isaacsim"),m.version("isaaclab"))'])!='5.1.0.0 2.3.2.post1':raise ValueError('Isaac version mismatch')
    # Verify the frozen dependency lock, itself pinned by our source lock.
    run(['sha256sum','-c',str(ROOT/'experiments/bowl_step8/step8-source-lock.sha256')],timeout=120)
    # Model dependency integrity (Cosmos and the pinned base assets) as well as P1.
    base_manifest=json.loads((ROOT/'models/model-manifest.json').read_text())
    for model in base_manifest['models']:
        for item in model['files']:
            asset=Path(model['snapshot'])/item['path']
            if asset.stat().st_size!=item['size'] or digest(asset)!=item['sha256']:raise ValueError('base model integrity failed')
    verify_layout(ROOT)
    checkpoint=verify_checkpoint(permit['checkpoint_manifest_path'],permit['checkpoint_manifest_sha256'])
    remaining=deadline-time.time()
    if remaining<2700:raise RuntimeError('Not enough session time after integrity checks')
    return checkpoint,deadline,py,policy_py


def finish_evidence(out,run_folder,runner,server,status):
    """Best effort independent cleanup; never mask the original runner exception."""
    errors=[]
    for name,process in [('runner',runner),('server',server)]:
        try:stop_process(process)
        except Exception as exc:errors.append(name+': '+type(exc).__name__)
    (out/'exit-code.txt').write_text(str(status)+'\n')
    for folder in (run_folder,out):
        try:
            if folder.exists():seal(folder)
        except Exception as exc:errors.append('seal '+folder.name+': '+type(exc).__name__)
    try:
        snapshot=run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'])
        (out/'gpu-after.txt').write_text(snapshot)
        if snapshot:errors.append('GPU processes remain')
    except Exception as exc:
        (out/'gpu-after-error.txt').write_text(type(exc).__name__+'\n')
        errors.append('GPU snapshot: '+type(exc).__name__)
    if errors:(out/'cleanup-errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    try:seal(out)
    except Exception as exc:errors.append('reseal: '+type(exc).__name__)
    return errors


def main():
    ap=argparse.ArgumentParser();ap.add_argument('permit',type=Path);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    permit=json.loads(args.permit.read_text());result=check(permit)
    if permit.get('scope')!='single_capture':result['blockers'].append('This launcher implements capture only.');result['launch_ready']=False
    print(json.dumps(result,indent=2),flush=True)
    if not result['launch_ready']:return 2
    checkpoint,deadline,py,policy_py=vm_preflight(permit)
    if not args.execute:
        print('VM read-only checks passed; no shutdown scheduled and no models started.',flush=True)
        return 0
    remaining=deadline-time.time()
    # Backstop survives SSH loss; console Stopped still must be verified by Miodrag.
    run(['sudo','-n','shutdown','-h','+'+str(max(1,math.ceil(remaining/60)+5))])
    out=ROOT/'runs'/(CASE+'-supervisor');out.mkdir()
    (out/'permit.json').write_text(json.dumps(permit,indent=2))
    started=time.time();server=runner=None;status=1
    env=os.environ.copy();env.pop('NEBIUS_API_KEY',None)
    env.update(POLICY_SEED='0',POLICY_CHECKPOINT=str(checkpoint),OMNI_KIT_ACCEPT_EULA='Y',PYTHONPATH=str(ROOT/'RoboLab'))
    def interrupted(*unused):raise InterruptedError('supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGHUP,interrupted)
    try:
        with (out/'policy.log').open('w') as log:
            server=subprocess.Popen([str(policy_py),'-u',str(ROOT/'experiments/bowl_step5/run_policy.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            ready_by=min(started+300,deadline-2400)
            while time.time()<ready_by:
                if server.poll() is not None:raise RuntimeError('policy server exited')
                if 'Server is ready and listening' in (out/'policy.log').read_text():break
                time.sleep(1)
            else:raise TimeoutError('policy startup timeout')
            verify_load(ROOT/'logs/model-loading.json',checkpoint,started)
            (out/'model-loading.json').write_bytes((ROOT/'logs/model-loading.json').read_bytes())
            cmd=['timeout','-k','30','2400',str(py),'-u',str(P/'runner_B.py'),'--case',CASE,'--condition','baseline','--mode','capture','--permit',str(args.permit.resolve()),'--ledger',str(out/'unused-api-ledger.jsonl'),'--scene','layout','--arm','painted','--variant','lift','--policy','P1','--layout-seed','4003','--headless']
            warned=False
            with (out/'runner.log').open('w') as rlog:
                runner=subprocess.Popen(cmd,env=env,cwd=ROOT/'RoboLab',stdout=rlog,stderr=subprocess.STDOUT,start_new_session=True)
                while runner.poll() is None:
                    if time.time()>=deadline-600 and not warned:
                        print('STOP WARNING: 10 minutes until console-stop deadline; preserve/copy evidence now.',flush=True);warned=True
                    if time.time()>=deadline-300:raise TimeoutError('five-minute copy reserve reached')
                    time.sleep(1)
                status=runner.returncode
                if status!=0:raise RuntimeError('runner exited with status '+str(status))
            checks=json.loads((ROOT/'runs'/CASE/'checks.json').read_text())
            if not checks.get('complete'):raise RuntimeError('runner incomplete')
            if (out/'unused-api-ledger.jsonl').exists():raise RuntimeError('capture created unexpected API ledger')
            (out/'CAPTURE_COMPLETE').write_text('Capture only; not perception acceptance or task success.\n')
            status=0
    except BaseException as exc:
        status=status if status else 1
        (out/'failure-type.txt').write_text(type(exc).__name__+'\n')
        raise
    finally:
        # A second HUP/TERM (SSH drop, tmux kill) must not abort sealing; SIGKILL still works.
        for number in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(number,signal.SIG_IGN)
        cleanup_errors=finish_evidence(out,ROOT/'runs'/CASE,runner,server,status)
        print('Copy both evidence folders and verify hashes. Console stop deadline still applies; retain disk.',flush=True)
    if cleanup_errors:raise RuntimeError('Cleanup or GPU snapshot incomplete: '+', '.join(cleanup_errors))
    return status
if __name__=='__main__':raise SystemExit(main())
