#!/bin/bash
# Bounded live B comparison: six predeclared episodes, painted vs unpainted exterior, fixed order, no reruns.
# Starts only if the frozen painted shadow check read PASS. A fresh P0 server (seeded) per episode.
# An episode whose bridge fails (guard, IK miss, timeout) is recorded and the next one still runs. Infrastructure
# failures (GPU not released, server not ready, stale or wrong model load) stop the whole batch (fail closed).
set -euo pipefail
root=/home/miodrag/robot-transfer/bowl
code="$root/experiments/bowl_step6"
supervisor="$root/runs/step6_B_paint_live_v1-supervisor"
EPISODES="l2002:0:painted l2002:0:unpainted base:0:painted base:0:unpainted base:1:painted base:1:unpainted"
[ ! -e "$supervisor" ]
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
mkdir "$supervisor"
exec > >(tee "$supervisor/supervisor.log") 2>&1
server_pid=''
stop_server() {
 if [ -n "$server_pid" ]; then
  kill -TERM "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  server_pid=''
 fi
}
finish() {
 status=$?
 trap - EXIT
 stop_server
 printf '%s\n' "$status" > "$supervisor/exit-code.txt"
 date -u
 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$supervisor/gpu-after.txt"
 exit "$status"
}
trap finish EXIT
date -u
sha256sum -c "$code/paint-live-source-lock.sha256"
cp "$code/paint-live-source-lock.sha256" "$code/run_paint_live.sh" "$supervisor/"
[ "$(git -C "$root/RoboLab" rev-parse HEAD)" = b6cfcf6ccd2113c4cd1216d35b33024a7d5691da ]
[ "$(git -C "$root/Isaac-GR00T" rev-parse HEAD)" = dac54b313160c201c450f97df0fd74ebd379c892 ]
[ -z "$(git -C "$root/RoboLab" status --porcelain --untracked-files=no)" ]
[ -z "$(git -C "$root/Isaac-GR00T" status --porcelain --untracked-files=no)" ]
# Precondition: the REAL policy shadow check passed. Recompute the frozen analysis here, on the VM, from the real
# P0 responses of step6_B_paint_query_v1 (verified complete by run_paint_check.sh), never from a copied or mock result.
check_sup="$root/runs/step6_B_paint_check_v1-supervisor"
[ "$(cat "$check_sup/exit-code.txt")" = 0 ]
grep -q '^VERIFIED_PAINT_QUERIES 352$' "$check_sup/supervisor.log"
grep -q 'Server is ready and listening' "$check_sup/server.log"
# Bind the analysed files to that check run: written after its start (lock copy) and before its exit record.
for f in "$root/runs/step6_B_paint_query_v1/responses.hdf5" "$root/runs/step6_B_paint_capture_v1/captures.hdf5"; do
 [ "$(stat -c %Y "$f")" -ge "$(stat -c %Y "$check_sup/paint-check-source-lock.sha256")" ]
 [ "$(stat -c %Y "$f")" -le "$(stat -c %Y "$check_sup/exit-code.txt")" ]
done
sha256sum "$root/runs/step6_B_paint_query_v1/responses.hdf5" "$root/runs/step6_B_paint_capture_v1/captures.hdf5" | tee "$supervisor/paint-check-outputs.sha256"
"$root/RoboLab/.venv/bin/python" "$code/analyze_painted.py" \
 "$root/runs/step6_B_paint_query_v1/responses.hdf5" "$root/runs/step6_B_paint_capture_v1/captures.hdf5" \
 "$root/runs/step6_B_shadow_capture_v1/captures.hdf5" \
 "$root/RoboLab/output/step5_training_layout2002_v4/BananaInBowlUniformInitPose10cmTask/synchronized.hdf5" \
 "$root/assets/report/evidence/2026-10-21/bowl-step6/kinematics_export_v1/joint_graphs.json" "$code" > "$supervisor/paint-check-analysis.json"
"$root/RoboLab/.venv/bin/python" - "$supervisor/paint-check-analysis.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));print('PAINT_CHECK_READING',r['reading'],r.get('failed'));assert r['reading']=='PASS'
PY
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
printf 'case\tseed\tbridge_status\tfailure_txt\tgate_success\n' > "$supervisor/episodes.tsv"
run_episode() {
 # run_episode CASE SCENE SEED ARM [extra bridge args]
 local case_id=$1 scene=$2 seed=$3 arm=$4; shift 4
 echo "EPISODE_START $case_id $(date -u +%H:%M:%S)"
 [ ! -e "$root/runs/$case_id" ]
 for wait in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 1; done
 [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
 server_start=$(date +%s)
 unset POLICY_CHECKPOINT
 export POLICY_SEED="$seed"
 timeout -k 20 5400 "$root/Isaac-GR00T/.venv/bin/python" -u "$root/experiments/bowl_step5/run_policy.py" > "$supervisor/server_$case_id.log" 2>&1 &
 server_pid=$!
 ready=0
 for attempt in $(seq 1 300); do
  if grep -q 'Server is ready and listening' "$supervisor/server_$case_id.log"; then ready=1; break; fi
  kill -0 "$server_pid"
  sleep 1
 done
 [ "$ready" = 1 ]
 [ "$(stat -c %Y "$root/logs/model-loading.json")" -ge "$server_start" ]  # written by this server, not a stale file
 cp "$root/logs/model-loading.json" "$supervisor/model-loading_$case_id.json"
 "$root/Isaac-GR00T/.venv/bin/python" - "$supervisor/model-loading_$case_id.json" "$seed" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['policy_label']=='P0' and r['seed']==int(sys.argv[2]) and r['effective_checkpoint']==r['p0_snapshot']
assert Path(r['effective_checkpoint']).name=='05e7cc97e40dbd33b0890c35cc0214fcb0547ab5'
assert r['cosmos_revision']=='9ce19a195e423419c349abfc86fd07178b230561'
assert {x['label'] for x in r['loads']}=={'P0','Cosmos'}
for x in r['loads']:assert all(not x['loading_info'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
print('VERIFIED_P0_LOAD seed',sys.argv[2])
PY
 cd "$root/RoboLab"
 export OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH="$root/RoboLab"
 bridge_start=$(date +%s)
 set +e
 timeout -k 20 4500 .venv/bin/python -u "$code/run_painted_bridge.py" --case "$case_id" --scene "$scene" --arm "$arm" "$@" --headless > "$supervisor/bridge_$case_id.log" 2>&1
 status=$?
 set -e
 bridge_seconds=$(( $(date +%s) - bridge_start ))
 stop_server
 failure=no; [ -e "$root/runs/$case_id/failure.txt" ] && failure=yes
 success=$("$root/RoboLab/.venv/bin/python" -c "import json,sys;print(json.load(open(sys.argv[1]))['automatic_v4B_success'])" "$root/runs/$case_id/checks.json" 2>/dev/null || echo none)
 printf '%s\t%s\t%s\t%s\t%s\n' "$case_id" "$seed" "$status" "$failure" "$success" | tee -a "$supervisor/episodes.tsv"
 echo "EPISODE_END $case_id $(date -u +%H:%M:%S) bridge_seconds=$bridge_seconds"
}
# Declared non-scoring integration smoke (reviewer requirement): 300 controls, base scene, seed 0, unpainted.
# It must complete cleanly, and its measured pace must project a full 9000-control episode under the 4500 s cap.
run_episode step6_B_paint_live_smoke_v1 base 0 unpainted --max-ticks 300
smoke_seconds=$bridge_seconds
"$root/RoboLab/.venv/bin/python" - "$root/runs/step6_B_paint_live_smoke_v1" "$smoke_seconds" <<'PY'
import json,sys,cv2
from pathlib import Path
p=Path(sys.argv[1]);seconds=int(sys.argv[2])
assert not (p/'failure.txt').exists()
c=json.loads((p/'checks.json').read_text());assert c['complete'] and c['smoke_non_scoring'] and c['steps']==300,c
lines=(p/'gate_trace.jsonl').read_text().splitlines();assert len(lines)==300
v=cv2.VideoCapture(str(p/'views.mp4'));n=0
while v.read()[0]:n+=1
assert n==100,n
wall=[json.loads(x)['wall_s'] for x in lines];pace=(wall[-1]-wall[0])/(len(wall)-1)
startup=seconds-(wall[-1]-wall[0]);projected=startup+pace*9000
print('SMOKE_OK seconds',seconds,'s_per_control',round(pace,4),'startup_s',round(startup),'projected_full_episode_s',round(projected))
assert projected<4200,('pace too slow for the 4500 s cap',projected)
PY
for spec in $EPISODES; do
 IFS=: read -r scene seed arm <<< "$spec"
 run_episode "step6_B_paint_live_${scene}_s${seed}_${arm}_v1" "$scene" "$seed" "$arm"
done
echo LIVE_COMPARISON_COMPLETE
