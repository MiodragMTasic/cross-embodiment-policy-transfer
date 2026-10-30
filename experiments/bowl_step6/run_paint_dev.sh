#!/bin/bash
# Exploratory development round, painted arm only: variant lift, then variant grip; each is a declared 1000-control
# non-scoring probe on base seed 1, then the three painted scenes of the live comparison, fixed order, no reruns.
# A variant whose probe misses its declared criterion is skipped, and the next variant still runs. Starts only after
# the six-episode live comparison completed. A fresh P0 server (seeded) per episode.
# An episode whose bridge fails (guard, IK miss, timeout) is recorded and the next one still runs. Infrastructure
# failures (GPU not released, server not ready, stale or wrong model load) stop the whole batch (fail closed).
set -euo pipefail
root=/home/miodrag/robot-transfer/bowl
code="$root/experiments/bowl_step6"
supervisor="$root/runs/step6_B_paint_dev_v1-supervisor"
EPISODES="l2002:0 base:0 base:1"
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
sha256sum -c "$code/paint-dev-source-lock.sha256"
cp "$code/paint-dev-source-lock.sha256" "$code/run_paint_dev.sh" "$supervisor/"
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
probe_ok() {
 # probe_ok VARIANT SECONDS: the declared probe criterion for that variant (prints the reading, exits non-zero on a miss)
 "$root/RoboLab/.venv/bin/python" - "$root/runs/step6_B_paint_dev_$1_probe_v1" "$1" "$2" <<'PY'
import json,sys
import numpy as np
from pathlib import Path
p=Path(sys.argv[1]);variant=sys.argv[2];seconds=int(sys.argv[3])
assert not (p/'failure.txt').exists(),'probe bridge failed'
c=json.loads((p/'checks.json').read_text());assert c['complete'] and c['smoke_non_scoring'] and c['variant']==variant and (c['steps']==1000 or c['automatic_v4B_success']),c
rows=[json.loads(x) for x in (p/'gate_trace.jsonl').read_text().splitlines()];assert len(rows)==c['steps']
wall=[r['wall_s'] for r in rows];pace=(wall[-1]-wall[0])/(len(rows)-1);projected=seconds-(wall[-1]-wall[0])+pace*9000
z=np.load(p/'tracking.npz');held=np.array([r['post']['contact'] and not r['post']['table_contact'] for r in rows])
force=np.linalg.norm(z['banana_force_on_tips'],axis=2).sum(1)
reading={'variant':variant,'steps':c['steps'],'gate_success':c['automatic_v4B_success'],'projected_full_episode_s':round(projected),'held_controls':int(held.sum()),'max_lift_m':float(z['lift_m'].max()),
         'median_tip_force_held_N':float(np.median(force[held])) if held.any() else None}
print('PROBE_READING',json.dumps(reading),flush=True)
assert projected<4200,('pace too slow for the 4500 s cap',projected)
if variant=='lift':assert np.isclose(reading['max_lift_m'],.015),'lift never reached 1.5 cm'
else:assert held.sum()>=60 and reading['median_tip_force_held_N']>=12.,'grip probe below criterion (>=60 held controls, median >=12 N; live painted baseline 4.1-4.6 N)'
PY
}
for variant in lift grip; do
 run_episode "step6_B_paint_dev_${variant}_probe_v1" base 1 painted --variant "$variant" --max-ticks 1000
 if probe_ok "$variant" "$bridge_seconds"; then
  echo "PROBE_PASS $variant"
  for spec in $EPISODES; do
   IFS=: read -r scene seed <<< "$spec"
   run_episode "step6_B_paint_dev_${variant}_${scene}_s${seed}_v1" "$scene" "$seed" painted --variant "$variant"
  done
 else
  echo "PROBE_MISS $variant: its three episodes are skipped"
 fi
done
echo DEV_ROUND_COMPLETE
