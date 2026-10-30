"""Offline saved-frame prompt screen. VM stays off. Default builds packets, no API.
Explicit --execute needs a separately reviewed transport permit. Never actuates a robot.
Replays POLICY-phase first queries at ticks where the executed gripper command was closed.
A baseline capture has no intervention-induced HOLD, so HOLD confirmation is NOT tested here.
"""
import argparse,json,sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from bridge import PerceptionJudge
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'charlie_bravo_v1'))
from controller import Config,Decision,Frame
from transport import VisionJudge
from vision_images import encode_views

HOLD_NOT_TESTED=('A baseline capture contains no intervention-induced HOLD (arm frozen at the last executed target). '
                 'Ordinary consecutive frames are not post-HOLD evidence, so confirmation behaviour is untested.')


def closed_ticks(gate_trace):
    """Executed command only: row step t+1 holds the command executed at tick t. Contact/object fields are never read."""
    rows=[json.loads(line) for line in Path(gate_trace).read_text().splitlines() if line.strip()]
    return {row['step']-1 for row in rows if float(row['command'])>.5}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('capture',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--gate-trace',type=Path,help='defaults to <capture>/../gate_trace.jsonl')
    ap.add_argument('--execute',action='store_true');ap.add_argument('--permit',type=Path);ap.add_argument('--ledger',type=Path);ap.add_argument('--limit',type=int,default=24);args=ap.parse_args()
    if not 1<=args.limit<=300:raise ValueError('bounded replay required')
    records={int(p.stem):json.loads(p.read_text()) for p in args.capture.glob('[0-9]*.json')}
    if not records:raise ValueError('No live capture records; no replay possible')
    gate_trace=args.gate_trace or args.capture.parent/'gate_trace.jsonl'
    if not gate_trace.is_file():raise ValueError('gate_trace.jsonl required to select executed closed-gripper ticks')
    if args.execute and (args.permit is None or args.ledger is None):raise ValueError('paid replay requires separate permit and ledger')
    args.output.mkdir(parents=True,exist_ok=False)
    judge=VisionJudge(args.permit,args.ledger) if args.execute else None
    if judge:
        if judge.permit.get('scope')!='offline_perception_replay':raise PermissionError('wrong replay scope')
        judge.verify_source_lock(Path(__file__).with_name('source-lock.json'))
    prompt=(Path(__file__).resolve().parents[1]/'charlie_bravo_v1/vision_prompt.txt').read_text()
    closed=closed_ticks(gate_trace);ticks=sorted(records)
    # Live POLICY queries see the last three captures; all three must exist and the newest must be closed.
    eligible=[i for i in range(2,len(ticks)) if ticks[i] in closed]
    selected=[eligible[i] for i in np.unique(np.linspace(0,len(eligible)-1,min(args.limit,len(eligible)),dtype=int))] if eligible else []
    summaries=[];counts=Counter();consecutive=0;stopped=None
    for index in selected:
        triplet=ticks[index-2:index+1];frames=[]
        for tick in triplet:
            ext=np.asarray(Image.open(args.capture/f'{tick:05d}_exterior.png').convert('RGB'))
            wrist=np.asarray(Image.open(args.capture/f'{tick:05d}_wrist.png').convert('RGB'))
            e,w=encode_views(ext,wrist);frames.append(Frame(tick,e,w))
        request={'tick':triplet[-1],'phase':'POLICY','hold_since_tick':None,'deadline_s':Config().timeout_s}
        def sender(augmented,frames,request):
            folder=args.output/str(request['tick']);folder.mkdir()
            (folder/'request.json').write_text(json.dumps({'prompt':augmented,'request':request,'ticks':triplet},indent=2))
            for frame in frames:
                (folder/f'{frame.tick}_exterior.png').write_bytes(frame.exterior);(folder/f'{frame.tick}_wrist.png').write_bytes(frame.wrist)
            if judge:return judge(augmented,frames,request)
            return json.dumps(dict(verdict='WAIT',held=False,central=False,clearance=False,views_agree=False,motion='unknown',reason='DRY RUN: no model called'))
        collector=SimpleNamespace(path=args.output,records=records)
        row={'tick':triplet[-1],'frames':triplet,'live_model_called':bool(judge)}
        try:
            # validated=True ONLY evaluates the candidate veto offline, never approves live use.
            returned=PerceptionJudge(sender,collector,validated=True)(prompt,frames,request)
        except Exception as exc:
            error=type(exc).__name__;counts['error_'+error]+=1;consecutive+=1
            summaries.append({**row,'error_type':error})
            if error=='PermissionError' or consecutive>=3:
                stopped=f'{error} after {consecutive} consecutive transport errors';break
            continue
        consecutive=0
        event=json.loads((args.output/'judge_events.jsonl').read_text().splitlines()[-1])
        try:decision=Decision.parse(returned);verdict=decision.verdict;ready=decision.ready
        except (ValueError,TypeError):verdict='MALFORMED';ready=False
        counts['schema_error' if event['parse_error'] else 'parsed']+=1;counts['vetoed']+=event['vetoed'];counts['returned_ready']+=ready
        summaries.append({**row,'returned_verdict':verdict,'returned_ready':ready,'parse_error':event['parse_error'],
                          'vetoed':event['vetoed'],'geometry_permits':event['geometry_permits']})
    (args.output/'summary.json').write_text(json.dumps({'executed_api':bool(judge),'phase':'POLICY','initial_decisions_only':True,
        'hold_confirmation':{'tested':False,'reason':HOLD_NOT_TESTED},
        'selection':{'rule':'newest of three consecutive captures has executed gripper command > 0.5','eligible':len(eligible),'selected':len(selected)},
        'counts':dict(counts),'stopped':stopped,'rows':summaries},indent=2))
if __name__=='__main__':main()
