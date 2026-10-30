#!/bin/bash
# Development probe for exterior painting (rows 1-24, Isaac only, no policy). Not a pre-registered check.
set -euo pipefail
root=/home/miodrag/robot-transfer/bowl
code="$root/experiments/bowl_step6"
case_id=step6_B_paint_probe_v2
supervisor="$root/runs/$case_id-supervisor"
[ ! -e "$root/runs/$case_id" ]
[ ! -e "$supervisor" ]
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
mkdir "$supervisor"
exec > >(tee "$supervisor/supervisor.log") 2>&1
finish() { status=$?; trap - EXIT; printf '%s\n' "$status" > "$supervisor/exit-code.txt"; date -u; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$supervisor/gpu-after.txt"; exit "$status"; }
trap finish EXIT
date -u
sha256sum "$code/paint_A.py" "$code/shadow_capture_painted.py" "$code/rigid_chain.py" "$code/local_ik.py" "$code/A_replay_inputs.json" | tee "$supervisor/source.sha256"
[ "$(git -C "$root/RoboLab" rev-parse HEAD)" = b6cfcf6ccd2113c4cd1216d35b33024a7d5691da ]
[ -z "$(git -C "$root/RoboLab" status --porcelain --untracked-files=no)" ]
cd "$root/RoboLab"
export OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH="$root/RoboLab"
timeout -k 20 900 .venv/bin/python -u "$code/shadow_capture_painted.py" --case "$case_id" --headless > "$supervisor/capture.log" 2>&1
[ ! -e "$root/runs/$case_id/failure.txt" ] || { tail -5 "$root/runs/$case_id/failure.txt"; exit 10; }
echo PAINT_PROBE_DONE
