"""Read-only permit validation. No cloud, SSH, simulator or API side effects."""
import argparse,hashlib,json,math,time
from pathlib import Path
P=Path(__file__).resolve().parent
VM='<vm-instance-id>'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def check(permit,now=None):
    now=time.time() if now is None else now;problems=[]
    def require(ok,message):
        if not ok:problems.append(message)
    def number(key):
        n=permit.get(key)
        return float(n) if type(n) in (int,float) and math.isfinite(n) else 0.
    mode=permit.get('scope')
    require(mode in ('single_capture','single_pilot'),'Only one capture or one pilot per permit.')
    require(permit.get('review_status')=='OK','Current build review is pending.')
    require(permit.get('human_paid_approval') is True,'Concrete paid execution approval missing.')
    require(number('expires_unix')>now,'Permit expired.')
    if mode=='single_pilot':
        require(permit.get('capture_smoke_reviewed') is True,'Capture acceptance has not passed.')
        require(permit.get('perception_review_status')=='OK' and permit.get('replay_review_status')=='OK','Geometry and augmented-prompt replay must pass before DROP.')
    else:require(number('api_cap_usd')==0,'Capture must have zero API allowance.')
    age=now-number('balance_checked_unix');balance=number('current_cloud_balance_usd')
    cap=number('cloud_spend_cap_usd');rate=number('hourly_compute_usd');seconds=number('whole_session_seconds')
    require(balance>0 and 0<=age<=3600,'Need balance checked within one hour.')
    require(0<seconds<=(3600 if mode=='single_capture' else 5400),'Invalid total session deadline.')
    require(0<rate and 0<cap<=(2 if mode=='single_capture' else 3),'Invalid rate or compute cap.')
    require(rate*seconds/3600<=cap<=balance-(.50 if mode=='single_capture' else 1),'Cap must cover entire VM session and preserve reserve.')
    require(permit.get('stop_owner')=='Miodrag' and permit.get('console_stop_timer_confirmed') is True,'Human console-stop owner and deadline timer must be confirmed; guest halt alone is not verified cloud stop.')
    require(permit.get('guest_halt_backstop_approved') is True,'Guest halt backstop must be approved.')
    require(permit.get('vm_id')==VM,'Wrong or missing VM identity.')
    require(len(str(permit.get('checkpoint_manifest_sha256','')))==64,'Checkpoint manifest SHA missing.')
    require(bool(permit.get('checkpoint_manifest_path')),'Checkpoint manifest path missing.')
    require(number('vm_started_unix')>0 and 0<=now-number('vm_started_unix')<seconds,'VM start timestamp must be fresh and inside session deadline.')
    lock=P/'source-lock.json'
    require(lock.is_file(),'Source lock missing.')
    if lock.is_file():
        require(digest(lock)==permit.get('source_lock_sha256'),'Permit does not pin current source lock.')
        for name,sha in json.loads(lock.read_text()).items():
            file=P/name;require(file.is_file() and digest(file)==sha,'Source changed: '+name)
    return {'launch_ready':not problems,'blockers':problems,'mode':mode,'actions_performed':'local reads only'}

class CaptureJudge:
    """Explicit no-network transport for capture; any attempted call is a hard error."""
    def __init__(self,permit):
        self.permit=json.loads(Path(permit).read_text())
        result=check(self.permit)
        if self.permit.get('scope')!='single_capture' or not result['launch_ready']:raise PermissionError(result['blockers'])
    def verify_source_lock(self,path):
        if digest(path)!=self.permit['source_lock_sha256']:raise PermissionError('source lock drift')
    def __call__(self,*args,**kwargs):raise AssertionError('API forbidden in capture')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('permit',type=Path);args=parser.parse_args()
    result=check(json.loads(args.permit.read_text()));print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['launch_ready'] else 2)
