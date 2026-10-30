#!/usr/bin/env bash
set -euo pipefail
case_id="$1"
task="$2"
layout_seed="$3"
export POLICY_SEED="$4"
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
sha256sum "$code"/*.py "$code/PROTOCOL.md" > "$run/source-hashes.txt"
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
cd "$root/RoboLab"
export OMNI_KIT_ACCEPT_EULA=Y
export PYTHONPATH="$root/RoboLab${PYTHONPATH:+:$PYTHONPATH}"
timeout 600 .venv/bin/python -u "$code/run_audited_episode.py" \
  --task "$task" --remote-host 127.0.0.1 --remote-port 15555 \
  --open-loop-horizon 8 --num-envs 1 --num-runs 1 --headless \
  --seed "$layout_seed" --instruction-type default --video-mode all --enable-subtask \
  --output-folder-name "$case_id" > "$run/evaluation.log" 2>&1
