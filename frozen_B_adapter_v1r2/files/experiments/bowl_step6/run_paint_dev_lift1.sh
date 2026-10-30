#!/bin/bash
# Exploratory development round, single episode by Miodrag's decision (23 Oct 2026, ~17:35 UTC): variant lift only,
# one painted episode on layout 2002 seed 0 (the scene/seed of live painted episode 1), no probe, no further episodes.
# Otherwise identical to run_paint_dev.sh: same bridge script, same checks, fresh seeded P0 server, fail-closed infrastructure.
set -euo pipefail
root=/home/miodrag/robot-transfer/bowl
code="$root/experiments/bowl_step6"
supervisor="$root/runs/step6_B_paint_dev_lift1_v1-supervisor"
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
sha256sum -c "$code/paint-dev-lift1-source-lock.sha256"
cp "$code/paint-dev-lift1-source-lock.sha256" "$code/run_paint_dev_lift1.sh" "$supervisor/"
[ "$(git -C "$root/RoboLab" rev-parse HEAD)" = b6cfcf6ccd2113c4cd1216d35b33024a7d5691da ]
[ "$(git -C "$root/Isaac-GR00T" rev-parse HEAD)" = dac54b313160c201c450f97df0fd74ebd379c892 ]
[ -z "$(git -C "$root/RoboLab" status --porcelain --untracked-files=no)" ]
[ -z "$(git -C "$root/Isaac-GR00T" status --porcelain --untracked-files=no)" ]
# Precondition: the frozen six-episode live comparison finished before any development episode runs.
[ "$(cat "$root/runs/step6_B_paint_live_v1-supervisor/exit-code.txt")" = 0 ]
grep -q '^LIVE_COMPARISON_COMPLETE$' "$root/runs/step6_B_paint_live_v1-supervisor/supervisor.log"
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
 timeout -k 20 4500 .venv/bin/python -u "$code/run_painted_dev.py" --case "$case_id" --scene "$scene" --arm "$arm" "$@" --headless > "$supervisor/bridge_$case_id.log" 2>&1
 status=$?
 set -e
 bridge_seconds=$(( $(date +%s) - bridge_start ))
 stop_server
 failure=no; [ -e "$root/runs/$case_id/failure.txt" ] && failure=yes
 success=$("$root/RoboLab/.venv/bin/python" -c "import json,sys;print(json.load(open(sys.argv[1]))['automatic_v4B_success'])" "$root/runs/$case_id/checks.json" 2>/dev/null || echo none)
 printf '%s\t%s\t%s\t%s\t%s\n' "$case_id" "$seed" "$status" "$failure" "$success" | tee -a "$supervisor/episodes.tsv"
 echo "EPISODE_END $case_id $(date -u +%H:%M:%S) bridge_seconds=$bridge_seconds"
}
run_episode step6_B_paint_dev_lift_l2002_s0_v1 l2002 0 painted --variant lift
echo DEV_LIFT_SINGLE_COMPLETE
