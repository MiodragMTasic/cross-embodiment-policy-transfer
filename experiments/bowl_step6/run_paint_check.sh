#!/bin/bash
# Painted shadow check. Stage 1: B pose-matched capture with exterior painting (simulator only). Stage 2: server-only queries.
# Nothing P0 returns is executed. Stages never overlap on the GPU.
set -euo pipefail
root=/home/miodrag/robot-transfer/bowl
code="$root/experiments/bowl_step6"
capture_id=step6_B_paint_capture_v1
query_id=step6_B_paint_query_v1
capture_run="$root/runs/$capture_id"
query_run="$root/runs/$query_id"
supervisor="$root/runs/step6_B_paint_check_v1-supervisor"
[ ! -e "$capture_run" ]
[ ! -e "$query_run" ]
[ ! -e "$supervisor" ]
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
sha256sum -c "$code/paint-check-source-lock.sha256"
cp "$code/paint-check-source-lock.sha256" "$code/run_paint_check.sh" "$supervisor/"
[ "$(git -C "$root/RoboLab" rev-parse HEAD)" = b6cfcf6ccd2113c4cd1216d35b33024a7d5691da ]
[ "$(git -C "$root/Isaac-GR00T" rev-parse HEAD)" = dac54b313160c201c450f97df0fd74ebd379c892 ]
[ -z "$(git -C "$root/RoboLab" status --porcelain --untracked-files=no)" ]
[ -z "$(git -C "$root/Isaac-GR00T" status --porcelain --untracked-files=no)" ]

echo STAGE1_CAPTURE_START
cd "$root/RoboLab"
export OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH="$root/RoboLab"
set +e
timeout -k 20 1200 .venv/bin/python -u "$code/shadow_capture_painted.py" --case "$capture_id" --headless > "$supervisor/capture.log" 2>&1
status=$?
set -e
[ ! -e "$capture_run/failure.txt" ] || { cat "$capture_run/failure.txt"; exit 10; }
[ "$status" = 0 ]
.venv/bin/python - "$capture_run" <<'PY'
import json,sys,numpy as np,cv2,h5py
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'result.json').read_text())
assert r['complete'] and r['steps']==1320 and r['logical_steps']==88 and r['repeat_factor']==12 and r['task_score'] is None
assert r['captured_rows']==list(range(8,89,8))
x=np.load(p/'tracking.npz');assert x['complete'] and np.array_equal(x['step'],np.arange(1,1321))
counts=[36 if (r+1)%8==0 else 12 for r in range(88)]
assert np.array_equal(x['logical_index_0based'],np.repeat(np.arange(88),counts))
assert json.loads((p/'initialization.json').read_text())['physics_warmup_steps']==0
with h5py.File(p/'captures.hdf5') as f:
 assert sorted(int(k) for k in f)==list(range(8,89,8))
 for k in f:
  g=f[k];assert g['video.exterior_image_1_left'].shape==(1,1,180,320,3) and g['video.wrist_image_left'].shape==(1,1,180,320,3)
  assert g['state.joint_position'].shape==(1,1,7) and g['state.eef_9d'].shape==(1,1,9) and g['state.gripper_position'].shape==(1,1,1)
  assert g['painted.video.exterior_image_1_left'].shape==(1,1,180,320,3) and g.attrs['exterior_reread_max_abs_diff']==0
v=cv2.VideoCapture(str(p/'three_views.mp4'));n=0
while True:
 ok,_=v.read()
 if not ok:break
 n+=1
v.release();assert n==1320,n
print('VERIFIED_PAINT_CAPTURE',json.dumps({k:r[k] for k in ('captured_rows','valid_rows','max_banana_reference_error_m')}))
PY
# Gate: the frozen analysis needs six valid rows; do not start the policy server without them.
.venv/bin/python - "$capture_run" <<'PY' || exit 12
import json,sys
from pathlib import Path
v=json.loads((Path(sys.argv[1])/'result.json').read_text())['valid_rows'];print('VALID_ROWS',v)
assert len(v)>=6,v
PY
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]

echo STAGE2_QUERY_START
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
timeout -k 20 1800 "$root/Isaac-GR00T/.venv/bin/python" -u "$root/experiments/bowl_step5/run_policy.py" > "$supervisor/server.log" 2>&1 &
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
set +e
timeout -k 20 1200 .venv/bin/python -u "$code/shadow_query_painted.py" --case "$query_id" > "$supervisor/query.log" 2>&1
status=$?
set -e
[ ! -e "$query_run/failure.txt" ] || { cat "$query_run/failure.txt"; exit 11; }
[ "$status" = 0 ]
.venv/bin/python - "$query_run" <<'PY'
import sys,numpy as np,h5py
from pathlib import Path
with h5py.File(Path(sys.argv[1])/'responses.hdf5') as f:
 assert f.attrs['complete'] and f['action.joint_position'].shape==(352,40,7) and np.isfinite(f['action.joint_position'][:]).all()
 c=[x.decode() for x in f['meta.condition'][:]]
 assert all(c.count(k)==88 for k in ('stateA_imgA','stateB_imgB','stateB_painted','stateA_painted'))
print('VERIFIED_PAINT_QUERIES',352)
PY
# Analysis runs afterwards on CPU with the hash-locked analyze_painted.py. No rerun on any outcome.
