#!/usr/bin/env bash
set -euo pipefail
root="$HOME/robot-transfer/bowl"
code="$root/experiments/bowl_step5"
run="$root/runs/step5_reload_smoke_v1"
[ ! -e "$run" ]
mkdir "$run"
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
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$run/gpu-after.txt"
  exit "$status"
}
trap finish EXIT
export POLICY_CHECKPOINT="$root/models/disposable_smoke_v1/checkpoint-2"
export POLICY_SEED=0
printf 'Disposable checkpoint: %s\n' "$POLICY_CHECKPOINT"
sha256sum "$code/run_policy.py" "$code/reload_scene_smoke.py" "$code/run_reload_smoke.sh" > "$run/source-hashes.txt"
timeout 600 "$root/Isaac-GR00T/.venv/bin/python" -u "$code/run_policy.py" > "$run/server.log" 2>&1 &
server_pid=$!
ready=0
for attempt in $(seq 1 120); do
  if grep -q 'Server is ready and listening' "$run/server.log"; then ready=1; break; fi
  kill -0 "$server_pid"
  sleep 1
done
[ "$ready" = 1 ]
cp "$root/logs/model-loading.json" "$run/model-loading.json"
timeout 360 "$root/RoboLab/.venv/bin/python" -u "$code/reload_scene_smoke.py" > "$run/simulation.log" 2>&1
