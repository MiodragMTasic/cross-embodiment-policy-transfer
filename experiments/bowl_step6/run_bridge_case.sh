#!/bin/bash
set -euo pipefail
root=/home/miodrag/robot-transfer/bowl
code="$root/experiments/bowl_step6"
case_id=step6_B_P0_v2
run="$root/runs/$case_id"
supervisor="$root/runs/$case_id-supervisor"
[ ! -e "$run" ] && [ ! -e "$supervisor" ]
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
mkdir "$supervisor"
exec > >(tee "$supervisor/supervisor.log") 2>&1
server_pid=''
finish() {
 status=$?
 trap - EXIT
 if [ -n "$server_pid" ]; then
  kill -TERM "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
 fi
 printf '%s\n' "$status" > "$supervisor/exit-code.txt"
 date -u
 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$supervisor/gpu-after.txt"
 exit "$status"
}
trap finish EXIT
date -u
sha256sum -c "$code/bridge-source-lock.sha256"
cp "$code/bridge-source-lock.sha256" "$code/run_bridge_case.sh" "$supervisor/"
[ "$(git -C "$root/RoboLab" rev-parse HEAD)" = b6cfcf6ccd2113c4cd1216d35b33024a7d5691da ]
[ "$(git -C "$root/Isaac-GR00T" rev-parse HEAD)" = dac54b313160c201c450f97df0fd74ebd379c892 ]
[ -z "$(git -C "$root/RoboLab" status --porcelain --untracked-files=no)" ]
[ -z "$(git -C "$root/Isaac-GR00T" status --porcelain --untracked-files=no)" ]
"$root/Isaac-GR00T/.venv/bin/python" - "$root" "$supervisor" <<'PY'
import json,hashlib,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:]);manifest=json.loads((root/'models/model-manifest.json').read_text())
verified=[]
for model in manifest['models']:
 for f in model['files']:
  p=Path(model['snapshot'])/f['path'];h=hashlib.sha256()
  with p.open('rb') as stream:
   for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
  assert p.stat().st_size==f['size'] and h.hexdigest()==f['sha256'],str(p)
  verified.append({'repo':model['repo'],'revision':model['revision'],**f})
(out/'model-integrity.json').write_text(json.dumps(verified,indent=2)+'\n')
print('VERIFIED_ALL_PINNED_MODEL_FILES',len(verified),flush=True)
PY
unset POLICY_CHECKPOINT
export POLICY_SEED=0
timeout -k 20 1500 "$root/Isaac-GR00T/.venv/bin/python" -u "$root/experiments/bowl_step5/run_policy.py" > "$supervisor/server.log" 2>&1 &
server_pid=$!
ready=0
for attempt in $(seq 1 300); do
 if grep -q 'Server is ready and listening' "$supervisor/server.log"; then ready=1; break; fi
 kill -0 "$server_pid"
 sleep 1
done
[ "$ready" = 1 ]
cp "$root/logs/model-loading.json" "$supervisor/model-loading.json"
"$root/Isaac-GR00T/.venv/bin/python" - "$supervisor/model-loading.json" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['policy_label']=='P0' and r['seed']==0 and r['effective_checkpoint']==r['p0_snapshot']
assert Path(r['effective_checkpoint']).name=='05e7cc97e40dbd33b0890c35cc0214fcb0547ab5'
assert r['cosmos_revision']=='9ce19a195e423419c349abfc86fd07178b230561'
assert {x['label'] for x in r['loads']}=={'P0','Cosmos'}
for x in r['loads']:assert all(not x['loading_info'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
print('VERIFIED_P0_LOAD')
PY
cd "$root/RoboLab"
export OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH="$root/RoboLab"
set +e
timeout -k 20 1200 .venv/bin/python -u "$code/run_fixed_bridge.py" --case "$case_id" --headless > "$supervisor/evaluation.log" 2>&1
status=$?
set -e
[ ! -e "$run/failure.txt" ] || { cat "$run/failure.txt"; exit 10; }
[ "$status" = 0 ]
.venv/bin/python - "$run" <<'PY'
import json,sys,numpy as np,cv2,h5py
from pathlib import Path
p=Path(sys.argv[1]);c=json.loads((p/'checks.json').read_text());assert c['complete'] and c['steps']==750 and c['task_score'] is None
x=np.load(p/'tracking.npz');assert bool(x['complete']) and len(x['step'])==750
assert c['initialization_steps']==1 and c['total_simulator_steps']==751 and x['simulator_step'][-1]==751
assert json.loads((p/'startup_before.json').read_text())['episode_step']==0
assert json.loads((p/'startup_after.json').read_text())['episode_step']==1
assert len(list(p.glob('startup_*.png')))==6
with h5py.File(p/'requests.hdf5') as f:assert len(f)==94
assert len(list((p/'responses').glob('*.npz')))==94
v=cv2.VideoCapture(str(p/'three_views.mp4'));n=0
while True:
 ok,_=v.read()
 if not ok:break
 n+=1
v.release();assert n==750,n
print('VERIFIED_COMPLETE_UNSCORED_DIAGNOSTIC',n)
PY
