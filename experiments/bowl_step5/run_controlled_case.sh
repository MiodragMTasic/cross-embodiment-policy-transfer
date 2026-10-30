#!/usr/bin/env bash
set -euo pipefail
case_id="$1"
task="$2"
layout_seed="$3"
export POLICY_SEED="$4"
# This is exclusively a P0 baseline; never inherit the disposable smoke override.
unset POLICY_CHECKPOINT
root="$HOME/robot-transfer/bowl"
code="$root/experiments/bowl_step5"
run="$root/runs/$case_id"
[ ! -e "$run" ]
[ ! -e "$root/RoboLab/output/$case_id" ]
mkdir -p "$run"
exec > >(tee "$run/supervisor.log") 2>&1
server_pid=''
finish() {
  status=$?
  trap - EXIT
  if [ -n "$server_pid" ]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  printf '%s\n' "$status" > "$run/exit-code.txt"
  date -u
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$run/gpu-after.txt"
  exit "$status"
}
trap finish EXIT
date -u
printf 'case=%s task=%s layout_seed=%s policy_seed=%s\n' "$case_id" "$task" "$layout_seed" "$POLICY_SEED"
sha256sum "$code"/*.py "$code/PROTOCOL_V2.md" > "$run/source-hashes.txt"
timeout 900 "$root/Isaac-GR00T/.venv/bin/python" -u "$code/run_policy.py" > "$run/server.log" 2>&1 &
server_pid=$!
ready=0
for attempt in $(seq 1 120); do
  if grep -q 'Server is ready and listening' "$run/server.log"; then ready=1; break; fi
  kill -0 "$server_pid"
  sleep 1
done
[ "$ready" = 1 ]
cp "$root/logs/model-loading.json" "$run/model-loading.json"
"$root/Isaac-GR00T/.venv/bin/python" - "$run/model-loading.json" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r['policy_label'] == 'P0', r['policy_label']
assert r['effective_checkpoint'] == r['p0_snapshot']
assert Path(r['effective_checkpoint']).name == '05e7cc97e40dbd33b0890c35cc0214fcb0547ab5'
assert r['cosmos_revision'] == '9ce19a195e423419c349abfc86fd07178b230561'
assert {x['label'] for x in r['loads']} == {'P0', 'Cosmos'}
for x in r['loads']:
    assert all(not x['loading_info'].get(k) for k in
               ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
print('VERIFIED_ORIGINAL_P0_AND_COSMOS_IDENTITY', flush=True)
PY
cd "$root/RoboLab"
export OMNI_KIT_ACCEPT_EULA=Y
export PYTHONPATH="$root/RoboLab${PYTHONPATH:+:$PYTHONPATH}"
timeout 600 .venv/bin/python -u "$code/run_controlled_episode.py" \
  --task "$task" --remote-host 127.0.0.1 --remote-port 15555 \
  --open-loop-horizon 8 --num-envs 1 --num-runs 1 --headless \
  --seed "$layout_seed" --instruction-type default --video-mode all --enable-subtask \
  --output-folder-name "$case_id" > "$run/evaluation.log" 2>&1
