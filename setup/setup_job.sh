#!/usr/bin/env bash
set -euo pipefail
root="$HOME/robot-transfer/bowl"
job="$1"
exec > >(tee -a "$root/logs/$job.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$root/logs/$job.exit"' EXIT
date -u
uv="$root/tools/uv/bin/uv"
case "$job" in
  groot-env)
    cd "$root/Isaac-GR00T"
    timeout 3600 "$uv" sync --locked --python 3.12
    "$uv" pip check --python .venv/bin/python
    ;;
  robolab-env)
    cd "$root/RoboLab"
    timeout 3600 "$uv" sync --locked --extra isaac51 --python 3.11
    "$uv" pip check --python .venv/bin/python
    ;;
  robolab-assets)
    cd "$root/RoboLab"
    git lfs install --local
    timeout 1800 git lfs pull
    git lfs fsck
    git status --short
    ;;
  *) echo "Unknown setup job: $job"; exit 2 ;;
esac
date -u
echo "COMPLETE: $job"
