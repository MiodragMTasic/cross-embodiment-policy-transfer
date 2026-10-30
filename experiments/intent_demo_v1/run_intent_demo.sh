#!/bin/bash
# Intent demo v1 (NOT RUN by its author; needs Miodrag's yes). A (Franka) / P0 episodes where the policy's instruction is
# a spec's policy_instruction instead of the task default. One supervisor, one plan:
#   run_intent_demo.sh <demo_plan.json> <its sha256>
# The episode is the approved step 5 A runner, unchanged, started through a_instruction_wrapper.py (instruction override in
# memory; see its docstring). Blocks marked COPIED are from experiments/bowl_step8/run_step8.sh (sha256 0e6bbc53…).
# Refuses to start while the step 8 after phase is unfinished or the GPU is busy. Halts on the first failed episode.
# Leaves RoboLab and Isaac-GR00T git-clean (checked before and after) and no process on the GPU (checked in the exit trap).
set -euo pipefail
[ $# = 2 ] || { echo 'usage: run_intent_demo.sh <demo_plan.json> <sha256>'; exit 2; }
plan=$(realpath "$1") plan_sha=$2
root=/home/miodrag/robot-transfer/bowl
demo="$root/experiments/intent_demo_v1"
s5="$root/experiments/bowl_step5"
code8="$root/experiments/bowl_step8"
task=BananaInBowlUniformInitPose10cmTask
after_sup="$root/runs/step8_after_v1-supervisor"
supervisor="$root/runs/intentdemo_v1-supervisor"
deadline_s=2700  # 45 min for the whole batch; no new episode starts after it
# Gate: the locked step 8 after run is finished and nothing is on the GPU or on the policy port.
[ -e "$after_sup/exit-code.txt" ] || { echo "REFUSE: $after_sup/exit-code.txt missing; step 8 after has not finished"; exit 3; }
[ "$(cat "$after_sup/exit-code.txt")" = 0 ] || [ "${INTENT_DEMO_AFTER_NONZERO_OK:-}" = yes ] || { echo "REFUSE: step 8 after exited $(cat "$after_sup/exit-code.txt"); ask Miodrag (set INTENT_DEMO_AFTER_NONZERO_OK=yes only on his word)"; exit 3; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo 'REFUSE: a process is on the GPU'; exit 3; }
[ -z "$(ss -ltnH 'sport = :15555')" ] || { echo 'REFUSE: something already listens on port 15555'; exit 3; }
[ ! -e "$supervisor" ] || { echo "REFUSE: $supervisor exists; supervisor dirs are never reused"; exit 3; }
[ "$(sha256sum "$plan" | cut -d' ' -f1)" = "$plan_sha" ] || { echo 'REFUSE: plan sha256 mismatch'; exit 3; }
mkdir "$supervisor"
exec > >(tee "$supervisor/supervisor.log") 2>&1
server_pid='' episode_pid=''
stop_pid() {
 if [ -n "$1" ]; then
  kill -TERM "$1" 2>/dev/null || true
  wait "$1" 2>/dev/null || true
 fi
}
stop_server() { stop_pid "$server_pid"; server_pid=''; }
stop_episode() { stop_pid "$episode_pid"; episode_pid=''; }
finish() {
 status=$?
 trap - EXIT
 stop_episode
 stop_server
 for wait in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 1; done
 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$supervisor/gpu-after.txt"
 if [ -s "$supervisor/gpu-after.txt" ]; then echo 'ORPHAN_GPU_PROCESS: see gpu-after.txt'; [ "$status" != 0 ] || status=4; fi
 printf '%s\n' "$status" > "$supervisor/exit-code.txt"
 date -u
 exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
date -u
echo PHASE intentdemo_v1
cp "$plan" "$supervisor/demo_plan.json"
[ "$(sha256sum "$supervisor/demo_plan.json" | cut -d' ' -f1)" = "$plan_sha" ]
sha256sum -c "$demo/intent-demo-lock.sha256"
sha256sum -c --quiet "$code8/step8-source-lock.sha256"  # A runner, audit_capture.py, run_policy.py, client.py unchanged
echo VERIFIED_STEP8_SOURCE_LOCK
cp "$demo/intent-demo-lock.sha256" "$demo/run_intent_demo.sh" "$supervisor/"
verify_trees() {
 # COPIED (run_step8.sh:54-57, 64-67)
 [ "$(git -C "$root/RoboLab" rev-parse HEAD)" = b6cfcf6ccd2113c4cd1216d35b33024a7d5691da ]
 [ "$(git -C "$root/Isaac-GR00T" rev-parse HEAD)" = dac54b313160c201c450f97df0fd74ebd379c892 ]
 [ -z "$(git -C "$root/RoboLab" status --porcelain --untracked-files=no)" ]
 [ -z "$(git -C "$root/Isaac-GR00T" status --porcelain --untracked-files=no)" ]
 for repo in RoboLab Isaac-GR00T; do
  [ -z "$(git -C "$root/$repo" status --porcelain --untracked-files=all | grep -v -E '^\?\? output/' || true)" ]
 done
 [ -z "$(git -C "$root/RoboLab" status --porcelain --ignored --untracked-files=all -- robolab/tasks | grep -v -E '__pycache__/|\.pyc$' || true)" ]
 echo "VERIFIED_PINS_AND_CLEAN_TREES $1"
}
verify_trees before
[ "$("$root/RoboLab/.venv/bin/python" -c 'import importlib.metadata as m;print(m.version("isaacsim"),m.version("isaaclab"))')" = '5.1.0.0 2.3.2.post1' ]
# COPIED (run_step8.sh:70-84)
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
python3 -I "$demo/demo_plan.py" check "$plan" > "$supervisor/episodes-planned.tsv"  # spec paths resolve beside the plan
mapfile -t episodes < "$supervisor/episodes-planned.tsv"
[ "${#episodes[@]}" -ge 1 ] && [ "${#episodes[@]}" -le 6 ]
for line in "${episodes[@]}"; do
 IFS=$'\t' read -r case_id layout spec spec_sha <<< "$line"
 [ ! -e "$root/runs/$case_id" ] && [ ! -e "$root/RoboLab/output/$case_id" ] || { echo "REFUSE: $case_id already exists"; exit 3; }
done
echo "PLAN_OK ${#episodes[@]} episodes"
start_server() {
 # start_server CASE: P0 only. COPIED (run_step8.sh:139-170) with the P1 branch removed.
 local case_id=$1 limit=900 attempts=120
 for wait in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 1; done
 [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
 export POLICY_SEED=0
 unset POLICY_CHECKPOINT
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
 [ -n "$(ss -ltnH 'src 127.0.0.1:15555')" ] && [ -z "$(ss -ltnH 'sport = :15555' | grep -v '127.0.0.1:15555' || true)" ] || { echo 'HALT: policy server is not bound to 127.0.0.1:15555 only'; exit 1; }
}
printf 'phase\trobot\tpolicy\tlayout\tcase\tstatus\tfailure\tv4_success\tinstruction_ok\tseconds\n' > "$supervisor/episodes.tsv"
run_demo_A() {
 # run_demo_A CASE LAYOUT SPEC SPEC_SHA: one A/P0 episode with the spec's instruction; halts the batch on any failure
 local case_id=$1 layout=$2 spec=$3 spec_sha=$4 t0 status captured checked failure v4 failed=()
 echo "EPISODE_START A $case_id $(date -u +%H:%M:%S)"
 mkdir "$root/runs/$case_id"
 cp "$spec" "$root/runs/$case_id/spec.json"
 [ "$(sha256sum "$root/runs/$case_id/spec.json" | cut -d' ' -f1)" = "$spec_sha" ]
 sha256sum "$s5"/*.py "$demo"/*.py > "$root/runs/$case_id/source-hashes.txt"
 start_server "$case_id"
 cd "$root/RoboLab"
 export OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH="$root/RoboLab"
 export INTENT_DEMO_SPEC="$root/runs/$case_id/spec.json" INTENT_DEMO_SPEC_SHA256="$spec_sha"
 t0=$(date +%s)
 set +e
 # The episode command of run_step8.sh:203-207 (arguments unchanged), launched through the wrapper and in the background
 # so the exit trap can stop it.
 timeout -k 20 600 .venv/bin/python -u "$demo/a_instruction_wrapper.py" \
  --task "$task" --task-dirs randomize_initial_pose --remote-host 127.0.0.1 --remote-port 15555 \
  --open-loop-horizon 8 --num-envs 1 --num-runs 1 --headless \
  --seed "$layout" --instruction-type default --video-mode all --enable-subtask \
  --output-folder-name "$case_id" > "$root/runs/$case_id/evaluation.log" 2>&1 &
 episode_pid=$!
 wait "$episode_pid"
 status=$?
 episode_pid=''
 stop_server
 # COPIED capture-integrity check (run_step8.sh:211-227)
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
 # The policy was sent the spec's text: one override marker in the log, and the runner's own record of the request.
 "$root/RoboLab/.venv/bin/python" - "$root/RoboLab/output/$case_id" "$root/runs/$case_id" <<'PY'
import h5py, json, sys
from pathlib import Path
out, run = map(Path, sys.argv[1:])
want = json.loads((run / 'spec.json').read_text())['policy_instruction']
caps = list(out.glob('*/synchronized.hdf5'))
assert len(caps) == 1, caps
with h5py.File(caps[0], 'r') as f:
    got = f.attrs['instruction']
got = got.decode() if isinstance(got, bytes) else str(got)
marks = [x for x in (run / 'evaluation.log').read_text().splitlines() if x.startswith('INTENT_OVERRIDE_APPLIED ')]
assert len(marks) == 1, marks
assert json.loads(marks[0].split(' ', 1)[1])['to'] == want, marks
assert got == want, (got, want)
(run / 'instruction-check.json').write_text(json.dumps({'policy_received': got, 'spec_policy_instruction': want,
                                                        'override_marker': marks[0]}, indent=1) + '\n')
print('VERIFIED_POLICY_RECEIVED_SPEC_INSTRUCTION', json.dumps(got), flush=True)
PY
 checked=$?
 set -e
 [ "$captured" = 0 ] || failed+=(capture)
 [ "$checked" = 0 ] || failed+=(instruction)
 failure=$(IFS=+; echo "${failed[*]:-no}")
 v4=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['stable_success_s'] is not None)" "$(ls "$root/RoboLab/output/$case_id/$task/strict-metric.json" 2>/dev/null)" 2>/dev/null || echo none)
 printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' intentdemo_v1 A P0 "$layout" "$case_id" "$status" "$failure" "$v4" \
  "$([ "$checked" = 0 ] && echo yes || echo no)" "$(( $(date +%s) - t0 ))" | tee -a "$supervisor/episodes.tsv"
 echo "EPISODE_END A $case_id $(date -u +%H:%M:%S)"
 [ "$status" = 0 ] && [ "$failure" = no ] || { echo "HALT: $case_id status=$status failure=$failure; no further episodes"; exit 1; }
}
batch_start=$(date +%s)
for line in "${episodes[@]}"; do
 IFS=$'\t' read -r case_id layout spec spec_sha <<< "$line"
 [ $(( $(date +%s) - batch_start )) -lt "$deadline_s" ] || { echo "HALT: batch deadline ${deadline_s}s reached before $case_id"; exit 1; }
 run_demo_A "$case_id" "$layout" "$spec" "$spec_sha"
done
verify_trees after
sha256sum -c --quiet "$code8/step8-source-lock.sha256"
echo INTENTDEMO_COMPLETE
