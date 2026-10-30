"""Local verification of both copied run and supervisor folders; never contacts VM."""
import argparse,json
from pathlib import Path
from preflight import digest

def verify(folder):
    m=json.loads((folder/'copy-manifest.json').read_text());bad=[]
    listed=set()
    for f in m['files']:
        rel=f['path'];p=folder/rel;listed.add(rel)
        if not p.resolve().is_relative_to(folder.resolve()) or not p.is_file() or p.is_symlink() or p.stat().st_size!=f['bytes'] or digest(p)!=f['sha256']:bad.append(rel)
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='copy-manifest.json'}
    bad+=sorted(actual^listed)
    return bad
def gpu_snapshot_errors(folder):
    if (folder/'gpu-after-error.txt').exists() or (folder/'cleanup-errors.json').exists():
        return ['cleanup or GPU snapshot failed; inspect supervisor evidence']
    if not (folder/'gpu-after.txt').is_file():return ['GPU snapshot missing']
    if (folder/'gpu-after.txt').read_text().strip():return ['GPU processes remained at supervisor exit']
    return []

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('supervisor',type=Path);args=ap.parse_args()
    errors=verify(args.run)+verify(args.supervisor)
    for name in ('tracking.npz','gate_trace.jsonl','checks.json'):
        if not (args.run/name).is_file():errors.append('missing '+name)
    if not list(args.run.rglob('*.mp4')):errors.append('missing review video')
    if not list((args.run/'perception').glob('*.npz')):errors.append('missing depth capture')
    if (args.supervisor/'exit-code.txt').read_text().strip()!='0':errors.append('supervisor nonzero exit')
    if not (args.supervisor/'CAPTURE_COMPLETE').is_file():errors.append('no completion marker')
    errors+=gpu_snapshot_errors(args.supervisor)
    print(json.dumps({'verified':not errors,'errors':errors,'note':'Historical process snapshot only; verify no current copy/test before stopping.'},indent=2))
    raise SystemExit(bool(errors))
