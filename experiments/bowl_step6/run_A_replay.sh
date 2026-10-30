#!/bin/bash
set -euo pipefail
root=/home/miodrag/robot-transfer/bowl
code="$root/experiments/bowl_step6"
case_id=step6_A_recorded_replay_v1
run="$root/runs/$case_id"
supervisor="$root/runs/$case_id-supervisor"
[ ! -e "$run" ] && [ ! -e "$supervisor" ]
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
mkdir "$supervisor"
exec > >(tee "$supervisor/supervisor.log") 2>&1
finish() {
 status=$?
 trap - EXIT
 printf '%s\n' "$status" > "$supervisor/exit-code.txt"
 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$supervisor/gpu-after.txt"
 date -u
 exit "$status"
}
trap finish EXIT
date -u
sha256sum "$code/run_A_replay.sh" > "$supervisor/supervisor-sha256.txt"
sha256sum -c "$code/A-replay-source-lock.sha256"
cp "$code/A-replay-source-lock.sha256" "$code/run_A_replay.sh" "$supervisor/"
[ "$(git -C "$root/RoboLab" rev-parse HEAD)" = b6cfcf6ccd2113c4cd1216d35b33024a7d5691da ]
[ -z "$(git -C "$root/RoboLab" status --porcelain --untracked-files=no)" ]
cd "$root/RoboLab"
export OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH="$root/RoboLab"
set +e
timeout -k 20 600 .venv/bin/python -u "$code/replay_A_reference.py" --case "$case_id" --headless > "$supervisor/evaluation.log" 2>&1
status=$?
set -e
[ ! -e "$run/failure.txt" ] || { cat "$run/failure.txt"; exit 10; }
[ "$status" = 0 ]
.venv/bin/python - "$run" <<'PY'
import json,sys,numpy as np,cv2
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'result.json').read_text())
assert r['complete'] and r['steps']==301
x=np.load(p/'tracking.npz');assert x['complete'] and np.array_equal(x['step'],np.arange(1,302))
assert len(json.loads((p/'release_trace.json').read_text()))==301
assert json.loads((p/'initialization.json').read_text())['physics_warmup_steps']==0
v=cv2.VideoCapture(str(p/'three_views.mp4'));n=0
while True:
 ok,_=v.read()
 if not ok:break
 n+=1
v.release();assert n==301,n
print('COMPLETE_A_REPLAY_DIAGNOSTIC',json.dumps(r))
# Deliberately no success-based rerun or hidden data exclusion.
PY
