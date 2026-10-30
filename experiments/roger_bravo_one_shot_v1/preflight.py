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
    require(mode=='single_vision_attempt','Only a single exploratory vision attempt is allowed.')
    require(permit.get('review_status')=='OK','Current build review is pending.')
    require(permit.get('human_paid_approval') is True,'Concrete paid execution approval missing.')
    require(permit.get('accept_unvalidated_vision') is True,'Exploratory vision limitation must be acknowledged.')
    require(number('expires_unix')>now,'Permit expired.')
    require(0<number('api_cap_usd')<=1,'API cap must be positive and at most $1.')
    require(permit.get('ledger_path')=='/home/miodrag/robot-transfer/bowl/runs/roger_B_P1_l4003_assisted_v1-supervisor/api-ledger.jsonl','Wrong evidence ledger.')
    age=now-number('balance_checked_unix');balance=number('current_cloud_balance_usd')
    cap=number('cloud_spend_cap_usd');rate=number('hourly_compute_usd');seconds=number('whole_session_seconds')
    require(balance>0 and 0<=age<=3600,'Need balance checked within one hour.')
    require(0<seconds<=3600,'Invalid total session deadline.')
    require(0<rate and 0<cap<=2,'Invalid rate or compute cap.')
    require(rate*seconds/3600<=cap<=balance-.50,'Cap must cover entire VM session and preserve reserve.')
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

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('permit',type=Path);args=parser.parse_args()
    result=check(json.loads(args.permit.read_text()));print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['launch_ready'] else 2)
