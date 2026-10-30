#!/bin/bash
# Step 8 final test on reserved layouts 4000-4011 (EXPERIMENT_REPORT §22-22.4; frozen_B_adapter_v1r2 README, "Rule for step 8";
# STEP8_RUNNERS.md). One supervisor per phase:
#   run_step8.sh smoke                                   non-scoring runner check: A/P0 on training layout 2000, then B/P0 on 2000 from its poses, 300 controls
#   run_step8.sh before                                  per layout: A/P0, then B/P0 reading A/P0's initial poses
#   run_step8.sh after <P1-manifest.json> <sha256> [--b-canary]
#                                                        drift checks, A/P0 canary on 4000/4001 (+ optional B/P0 canary on 4000),
#                                                        then per layout, same order: A/P1 (initial poses must equal A/P0's), B/P1
# Infrastructure checks marked COPIED are byte-identical to run_paint_dev_lift1.sh (B) or run_controlled_case_v4_registered.sh (A).
set -euo pipefail
phase=${1:-}
root=/home/miodrag/robot-transfer/bowl
code="$root/experiments/bowl_step8"
s5="$root/experiments/bowl_step5"
s6="$root/experiments/bowl_step6"
layouts=(4000 4001 4002 4003 4004 4005 4006 4007 4008 4009 4010 4011)
task=BananaInBowlUniformInitPose10cmTask
b_canary=no
case "$phase" in
 smoke|before) [ $# = 1 ];;
 after) [ $# = 3 ] || [ $# = 4 ]; p1_manifest=$2; p1_sha=$3; if [ $# = 4 ]; then [ "$4" = --b-canary ]; b_canary=yes; fi;;
 *) echo 'usage: run_step8.sh smoke | before | after <P1-manifest.json> <sha256> [--b-canary]'; exit 2;;
esac
supervisor="$root/runs/step8_${phase}_v1-supervisor"
before_sup="$root/runs/step8_before_v1-supervisor"
[ ! -e "$supervisor" ]
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
mkdir "$supervisor"
exec > >(tee "$supervisor/supervisor.log") 2>&1
# COPIED (run_paint_dev_lift1.sh:13-30)
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
echo "PHASE $phase"
sha256sum -c "$code/step8-source-lock.sha256"
cp "$code/step8-source-lock.sha256" "$code/run_step8.sh" "$supervisor/"
# COPIED (run_paint_dev_lift1.sh:34-37)
[ "$(git -C "$root/RoboLab" rev-parse HEAD)" = b6cfcf6ccd2113c4cd1216d35b33024a7d5691da ]
[ "$(git -C "$root/Isaac-GR00T" rev-parse HEAD)" = dac54b313160c201c450f97df0fd74ebd379c892 ]
[ -z "$(git -C "$root/RoboLab" status --porcelain --untracked-files=no)" ]
[ -z "$(git -C "$root/Isaac-GR00T" status --porcelain --untracked-files=no)" ]
# Added for step 8, every phase: the runner rule, Isaac versions, platform record, no stray files where tasks are registered from.
python3 "$code/check_step8_diff.py" B "$s6/run_painted_dev.py" "$s6/step8_B_runner.py" | tail -1
python3 "$code/check_step8_diff.py" A "$s5/run_controlled_episode_v4_registered.py" "$root/runs/step5_training_layout2002_v4/source-hashes.txt" | tail -1
[ "$("$root/RoboLab/.venv/bin/python" -c 'import importlib.metadata as m;print(m.version("isaacsim"),m.version("isaaclab"))')" = '5.1.0.0 2.3.2.post1' ]
printf 'driver=%s\ncuda=%s\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)" "$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')" > "$supervisor/platform.txt"
cat "$supervisor/platform.txt"
for repo in RoboLab Isaac-GR00T; do
 [ -z "$(git -C "$root/$repo" status --porcelain --untracked-files=all | grep -v -E '^\?\? output/' || true)" ]
done
[ -z "$(git -C "$root/RoboLab" status --porcelain --ignored --untracked-files=all -- robolab/tasks | grep -v -E '__pycache__/|\.pyc$' || true)" ]
echo VERIFIED_PINS_VERSIONS_AND_CLEAN_TASK_TREES
# COPIED (run_paint_dev_lift1.sh:41-55)
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
# Phase preconditions.
if [ "$phase" = before ]; then
 [ "$(cat "$root/runs/step8_smoke_v1-supervisor/exit-code.txt")" = 0 ]
 grep -q '^STEP8_SMOKE_COMPLETE$' "$root/runs/step8_smoke_v1-supervisor/supervisor.log"
fi
if [ "$phase" = after ]; then
 [ "$(cat "$before_sup/exit-code.txt")" = 0 ]
 grep -q '^STEP8_BEFORE_COMPLETE$' "$before_sup/supervisor.log"
 cmp "$before_sup/platform.txt" "$supervisor/platform.txt"  # same driver and CUDA as the before phase
 [ "$(sha256sum "$p1_manifest" | cut -d' ' -f1)" = "$p1_sha" ]
 cp "$p1_manifest" "$supervisor/P1-manifest.json"
 p1_dir=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_dir"])' "$p1_manifest")
 echo "P1_PINNED $p1_sha $p1_dir"
fi
failures=0 abort_kind= abort_run=0
printf 'phase\trobot\tpolicy\tlayout\tcase\tstatus\tfailure\tv4_success\tseconds\n' > "$supervisor/episodes.tsv"
record() {
 # record ROBOT POLICY LAYOUT CASE STATUS FAILURE V4 SECONDS
 printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$@" | tee -a "$supervisor/episodes.tsv"
 # A B safety_abort is a task outcome (scored on its partial trace, in delta-B), never a count toward the 3-failure halt.
 # Declared stop trigger: 3 consecutive B safety_aborts of the same kind halt the phase for Miodrag. Any other B episode
 # (no abort, a crash, another kind) resets the run; A episodes don't. The halt changes no score.
 if [ "$1" = B ]; then
  if [ "$6" = safety_abort ]; then
   local kind
   kind=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["kind"])' "$root/runs/$4/abort-classification.json") || { echo "HALT: safety_abort without a readable kind for $4"; exit 1; }
   if [ "$kind" = "$abort_kind" ]; then abort_run=$((abort_run + 1)); else abort_kind=$kind abort_run=1; fi
   [ "$abort_run" -lt 3 ] || { echo "HALT: $abort_run consecutive B safety_aborts of kind $kind in this phase. Stop and ask Miodrag (STEP8_RUNNERS.md, stop trigger)."; exit 1; }
   return 0
  fi
  abort_kind= abort_run=0
 fi
 if [ "$5" != 0 ] || [ "$6" != no ]; then
  failures=$((failures + 1))
  [ "$failures" -lt 3 ] || { echo "HALT: $failures failed episodes in this phase"; exit 1; }
 fi
}
verify_p1() {
 # Re-hash every P1 file against the sealed manifest; the directory may hold nothing else.
 "$root/Isaac-GR00T/.venv/bin/python" - "$supervisor/P1-manifest.json" "$supervisor/p1-verify_$1.json" <<'PY'
import json,hashlib,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text());d=Path(m['checkpoint_dir'])
listed={f['path'] for f in m['files']};present={str(p.relative_to(d)) for p in d.rglob('*') if p.is_file()}
assert listed==present,(sorted(listed^present))
for f in m['files']:
 p=d/f['path'];h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
 assert p.stat().st_size==f['size'] and h.hexdigest()==f['sha256'],str(p)
Path(sys.argv[2]).write_text(json.dumps({'checkpoint_dir':str(d),'files':len(m['files'])})+'\n')
print('VERIFIED_P1_FILES',len(m['files']),flush=True)
PY
}
start_server() {
 # start_server P0|P1 CASE SERVER_TIMEOUT READY_ATTEMPTS
 local policy=$1 case_id=$2 limit=$3 attempts=$4
 for wait in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 1; done
 [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
 export POLICY_SEED=0
 if [ "$policy" = P1 ]; then verify_p1 "$case_id"; export POLICY_CHECKPOINT="$p1_dir"; else unset POLICY_CHECKPOINT; fi
 server_start=$(date +%s)
 timeout -k 20 "$limit" "$root/Isaac-GR00T/.venv/bin/python" -u "$s5/run_policy.py" > "$supervisor/server_$case_id.log" 2>&1 &
 server_pid=$!
 ready=0
 for attempt in $(seq 1 "$attempts"); do
  if grep -q 'Server is ready and listening' "$supervisor/server_$case_id.log"; then ready=1; break; fi
  kill -0 "$server_pid"
  sleep 1
 done
 [ "$ready" = 1 ]
 [ "$(stat -c %Y "$root/logs/model-loading.json")" -ge "$server_start" ]  # written by this server, not a stale file
 cp "$root/logs/model-loading.json" "$supervisor/model-loading_$case_id.json"
 if [ "$policy" = P0 ]; then
 # COPIED (run_paint_dev_lift1.sh:78-88)
 "$root/Isaac-GR00T/.venv/bin/python" - "$supervisor/model-loading_$case_id.json" "$POLICY_SEED" <<'PY'
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
 else
 # P1 is served by the unchanged run_policy.py through POLICY_CHECKPOINT, which labels it "disposable copy"; identity is the pin.
 "$root/Isaac-GR00T/.venv/bin/python" - "$supervisor/model-loading_$case_id.json" "$POLICY_SEED" "$p1_dir" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text());p1=sys.argv[3]
assert r['policy_label']=='disposable copy' and r['seed']==int(sys.argv[2]) and r['effective_checkpoint']==p1
assert Path(r['p0_snapshot']).name=='05e7cc97e40dbd33b0890c35cc0214fcb0547ab5'
assert r['cosmos_revision']=='9ce19a195e423419c349abfc86fd07178b230561'
assert {x['label'] for x in r['loads']}=={'disposable copy','Cosmos'}
assert [x['path'] for x in r['loads'] if x['label']=='disposable copy']==[p1]
for x in r['loads']:assert all(not x['loading_info'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
print('VERIFIED_P1_LOAD seed',sys.argv[2],p1)
PY
 fi
}
a_metric() { ls "$root/RoboLab/output/$1/$task/strict-metric.json"; }
run_A() {
 # run_A CASE LAYOUT POLICY: the approved step 5 episode, unchanged; sets a_ok=yes|no
 local case_id=$1 layout=$2 policy=$3 t0 status captured failure v4
 echo "EPISODE_START A $case_id $(date -u +%H:%M:%S)"
 [ ! -e "$root/runs/$case_id" ]
 [ ! -e "$root/RoboLab/output/$case_id" ]
 mkdir "$root/runs/$case_id"
 sha256sum "$s5"/*.py > "$root/runs/$case_id/source-hashes.txt"
 start_server "$policy" "$case_id" 900 120
 cd "$root/RoboLab"
 export OMNI_KIT_ACCEPT_EULA=Y
 export PYTHONPATH="$root/RoboLab"
 t0=$(date +%s)
 set +e
 # COPIED episode command (run_controlled_case_v4_registered.sh:64-68), case and layout as arguments
 timeout 600 .venv/bin/python -u "$s5/run_controlled_episode_v4_registered.py" \
  --task "$task" --task-dirs randomize_initial_pose --remote-host 127.0.0.1 --remote-port 15555 \
  --open-loop-horizon 8 --num-envs 1 --num-runs 1 --headless \
  --seed "$layout" --instruction-type default --video-mode all --enable-subtask \
  --output-folder-name "$case_id" > "$root/runs/$case_id/evaluation.log" 2>&1
 status=$?
 stop_server
 # COPIED capture-integrity check (run_controlled_case_v4_registered.sh:70-86)
 "$root/RoboLab/.venv/bin/python" - "$root" "$case_id" "$task" "$layout" "$POLICY_SEED" <<'PY'
import json, subprocess, sys
from pathlib import Path
root, case, task, layout, policy = sys.argv[1:]
root = Path(root)
outputs = list((root / 'RoboLab/output' / case).glob('*/strict-metric.json'))
assert len(outputs) == 1, outputs
raw = subprocess.check_output([str(root / 'RoboLab/.venv/bin/python'),
    str(root / 'experiments/bowl_step5/audit_capture.py'), str(outputs[0].parent)], text=True)
d = json.loads(raw)
assert d['schema'] == 'command-linked-supported-release-v4'
assert d['layout_seed'] == int(layout) and d['policy_seed'] == int(policy)
assert d['task_class'] == [task] and d['registered_environment'] == task
assert d['requested_task_dirs'] == ['randomize_initial_pose']
(root / 'runs' / case / 'capture-integrity.json').write_text(raw)
print('VERIFIED_FINISHED_CAPTURE_AND_BOTH_VIDEOS', d['trace_rows'], flush=True)
PY
 captured=$?
 set -e
 failure=no; [ "$captured" = 0 ] || failure=capture
 v4=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['stable_success_s'] is not None)" "$(a_metric "$case_id" 2>/dev/null)" 2>/dev/null || echo none)
 a_ok=no; [ "$status" = 0 ] && [ "$failure" = no ] && a_ok=yes
 record A "$policy" "$layout" "$case_id" "$status" "$failure" "$v4" "$(( $(date +%s) - t0 ))"
 echo "EPISODE_END A $case_id $(date -u +%H:%M:%S)"
}
run_B() {
 # run_B CASE POLICY SCENE LABEL [bridge args]: the frozen bridge through step8_B_runner.py (COPIED run_paint_dev_lift1.sh:57-102 otherwise, except the failure class)
 local case_id=$1 policy=$2 scene=$3 label=$4 status failure success bridge_start bridge_seconds; shift 4
 echo "EPISODE_START B $case_id $(date -u +%H:%M:%S)"
 [ ! -e "$root/runs/$case_id" ]
 start_server "$policy" "$case_id" 5400 300
 cd "$root/RoboLab"
 export OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH="$root/RoboLab"
 bridge_start=$(date +%s)
 set +e
 timeout -k 20 4500 .venv/bin/python -u "$s6/step8_B_runner.py" --case "$case_id" --scene "$scene" --arm painted --variant lift --policy "$policy" "$@" --headless > "$supervisor/bridge_$case_id.log" 2>&1
 status=$?
 set -e
 bridge_seconds=$(( $(date +%s) - bridge_start ))
 stop_server
 # failure: no | safety_abort (policy-driven exit, classify_abort.py) | crash; the class also lands in runs/CASE/abort-classification.json
 failure=no
 if [ -e "$root/runs/$case_id/failure.txt" ]; then failure=$(python3 "$code/classify_abort.py" "$root/runs/$case_id" "$status") || failure=crash; fi
 case $failure in no|safety_abort|crash) ;; *) failure=crash ;; esac
 success=$("$root/RoboLab/.venv/bin/python" -c "import json,sys;print(json.load(open(sys.argv[1]))['automatic_v4B_success'])" "$root/runs/$case_id/checks.json" 2>/dev/null || echo none)
 record B "$policy" "$label" "$case_id" "$status" "$failure" "$success" "$bridge_seconds"
 echo "EPISODE_END B $case_id $(date -u +%H:%M:%S) bridge_seconds=$bridge_seconds"
}
if [ "$phase" = smoke ]; then
 run_A step8_A_smoke_l2000_s0_v1 2000 P0
 [ "$a_ok" = yes ]
 run_B step8_B_smoke_l2000_s0_v1 P0 layout 2000 --layout-seed 2000 --max-ticks 300
 [ "$(tail -1 "$supervisor/episodes.tsv" | cut -f6,7)" = "$(printf '0\tno')" ]
 echo STEP8_SMOKE_COMPLETE
fi
if [ "$phase" = before ]; then
 for layout in "${layouts[@]}"; do
  run_A "step8_A_P0_l${layout}_s0_v1" "$layout" P0
  if [ "$a_ok" != yes ]; then echo "SKIP_B $layout: A/P0 produced no valid poses"; continue; fi
  sha256sum "$(a_metric "step8_A_P0_l${layout}_s0_v1")" | tee -a "$supervisor/A_P0_strict-metric.sha256"
  run_B "step8_B_P0_l${layout}_s0_v1" P0 layout "$layout" --layout-seed "$layout"
 done
 echo STEP8_BEFORE_COMPLETE
fi
if [ "$phase" = after ]; then
 # Drift canary: rerun A/P0 on 4000 and 4001 under new names; recorded for the reviewer, not a gate.
 for layout in 4000 4001; do
  run_A "step8_A_P0canary_l${layout}_s0_v1" "$layout" P0
  if [ "$a_ok" = yes ]; then python3 "$code/compare_canary.py" A "$(a_metric "step8_A_P0_l${layout}_s0_v1")" "$(a_metric "step8_A_P0canary_l${layout}_s0_v1")" "$supervisor/canary_A_l${layout}.json" || true; fi
 done
 if [ "$b_canary" = yes ]; then
  run_B step8_B_P0canary_l4000_s0_v1 P0 layout 4000 --layout-seed 4000
  python3 "$code/compare_canary.py" B "$root/runs/step8_B_P0_l4000_s0_v1/gate_trace.jsonl" "$root/runs/step8_B_P0canary_l4000_s0_v1/gate_trace.jsonl" "$supervisor/canary_B_l4000.json" || true
 fi
 for layout in "${layouts[@]}"; do
  if ! grep -q "/step8_A_P0_l${layout}_s0_v1/" "$before_sup/A_P0_strict-metric.sha256"; then echo "SKIP_LAYOUT $layout: no A/P0 pair from the before phase"; continue; fi
  run_A "step8_A_P1_l${layout}_s0_v1" "$layout" P1
  if [ "$a_ok" = yes ]; then
   # Same layout must mean the same scene: A/P1 initial poses equal A/P0's to 1e-6, or the batch halts.
   python3 - "$(a_metric "step8_A_P0_l${layout}_s0_v1")" "$(a_metric "step8_A_P1_l${layout}_s0_v1")" <<'PY'
import json,sys
a,b=(json.load(open(p))['initial_poses_world'] for p in sys.argv[1:])
assert a.keys()==b.keys()
for k in a:assert len(a[k])==len(b[k]) and all(abs(x-y)<=1e-6 for x,y in zip(a[k],b[k])),('initial pose mismatch',k)
print('VERIFIED_SAME_INITIAL_POSES',sorted(a),flush=True)
PY
  fi
  # B/P1 reads the same A/P0 poses B/P0 read; they must be byte-identical to what the before phase recorded.
  grep "/step8_A_P0_l${layout}_s0_v1/" "$before_sup/A_P0_strict-metric.sha256" | sha256sum -c -
  run_B "step8_B_P1_l${layout}_s0_v1" P1 layout "$layout" --layout-seed "$layout"
 done
 echo STEP8_AFTER_COMPLETE
fi
