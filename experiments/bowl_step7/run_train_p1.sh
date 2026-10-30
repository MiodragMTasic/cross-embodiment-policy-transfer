#!/bin/bash
# Step 7 P1 probe or training (STEP7_PLAN_2026-10-23.md §3; EXPERIMENT_REPORT §22.6, D1). Usage: run_train_p1.sh probe|train
# Pins as every step 7/8 supervisor: source lock, repo commits and clean trees, all P0/Cosmos files re-hashed, idle GPU.
# Dataset precondition: the verified build in runs/step7_dataset_v1. train additionally needs a finished probe.
set -euo pipefail
mode=${1:-}
[ "$mode" = probe ] || [ "$mode" = train ]
ver=v1; [ "$mode" = probe ] && ver=v2  # probe_v1 crashed at its max_steps save (run_train_p1.py docstring); never reused
root=/home/miodrag/robot-transfer/bowl
code="$root/experiments/bowl_step7"
supervisor="$root/runs/step7_${mode}_${ver}"
[ ! -e "$supervisor" ]
[ ! -e "$root/models/step7_${mode}_${ver}" ]
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
mkdir "$supervisor"
exec > >(tee "$supervisor/supervisor.log") 2>&1
sampler=''
finish() {
 status=$?
 trap - EXIT
 if [ -n "$sampler" ]; then kill "$sampler" 2>/dev/null || true; fi
 printf '%s\n' "$status" > "$supervisor/exit-code.txt"
 date -u
 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$supervisor/gpu-after.txt"
 exit "$status"
}
trap finish EXIT
date -u
echo "MODE $mode"
sha256sum -c "$code/step7-train-source-lock.sha256"
cp "$code/step7-train-source-lock.sha256" "$code/run_train_p1.sh" "$code/run_train_p1.py" "$supervisor/"
[ "$(git -C "$root/RoboLab" rev-parse HEAD)" = b6cfcf6ccd2113c4cd1216d35b33024a7d5691da ]
[ "$(git -C "$root/Isaac-GR00T" rev-parse HEAD)" = dac54b313160c201c450f97df0fd74ebd379c892 ]
[ -z "$(git -C "$root/RoboLab" status --porcelain --untracked-files=no)" ]
[ -z "$(git -C "$root/Isaac-GR00T" status --porcelain --untracked-files=no)" ]
[ "$(cat "$root/runs/step7_dataset_v1/exit-code.txt")" = 0 ]
grep -q '^STEP7_DATASET_COMPLETE$' "$root/runs/step7_dataset_v1/log.txt"
if [ "$mode" = train ]; then
 [ "$(cat "$root/runs/step7_probe_v2/exit-code.txt")" = 0 ]
 grep -q '^FINITE_TRAINING_CHECKS' "$root/runs/step7_probe_v2/supervisor.log"
fi
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
df -h "$root" | tail -1
if [ "$mode" = train ]; then [ "$(df --output=avail -BG "$root" | tail -1 | tr -dc 0-9)" -ge 70 ]; fi  # 4 checkpoints + upstream root save ≈ 60 GB
# Platform and training-stack versions (as run_step8.sh:62).
printf 'driver=%s\ncuda=%s\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)" "$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')" > "$supervisor/platform.txt"
"$root/Isaac-GR00T/.venv/bin/python" -c "import importlib.metadata as m
for p in ('torch','transformers','accelerate','flash-attn','tyro'):
 try: print(p+'='+m.version(p))
 except m.PackageNotFoundError: print(p+'=absent')" >> "$supervisor/platform.txt"
cat "$supervisor/platform.txt"
# The training data pinned by content: upstream may rewrite meta/stats on fingerprint mismatch (factory.py, stats.py), so
# the set must be byte-identical before and after this run, and train must see exactly the bytes the probe saw.
dataset_sha() { (cd "$root/datasets/step7_train_v1" && find . -type f -print0 | sort -z | xargs -0 sha256sum); }
dataset_sha > "$supervisor/dataset-sha256.txt"
if [ "$mode" = train ]; then cmp "$supervisor/dataset-sha256.txt" "$root/runs/step7_probe_v2/dataset-sha256.txt"; fi
echo "DATASET_PINNED $(wc -l < "$supervisor/dataset-sha256.txt") files $(sha256sum "$supervisor/dataset-sha256.txt" | cut -c1-16)"
# GPU memory every 10 s, for the peak the probe is meant to measure.
( while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader >> "$supervisor/gpu-samples.csv"; sleep 10; done ) &
sampler=$!
limit=1800; [ "$mode" = train ] && limit=18000  # probe 30 min; training 5 h (plan estimate 1-4 h)
timeout -k 60 "$limit" "$root/Isaac-GR00T/.venv/bin/python" -u "$code/run_train_p1.py" --mode "$mode"
dataset_sha > "$supervisor/dataset-sha256-after.txt"
cmp "$supervisor/dataset-sha256.txt" "$supervisor/dataset-sha256-after.txt"
echo VERIFIED_DATASET_UNCHANGED
# No stray files left where later supervisors require clean trees (run_select_v1.sh and run_step8.sh refuse to start).
[ -z "$(git -C "$root/Isaac-GR00T" status --porcelain --untracked-files=all | grep -v -E '^\?\? output/' || true)" ]
if [ "$mode" = train ]; then [ "$(ls -d "$root/models/step7_train_v1"/checkpoint-* | xargs -n1 basename | tr '\n' ' ')" = 'checkpoint-1000 checkpoint-1500 checkpoint-2000 checkpoint-500 ' ]; fi
ls -la "$root/models/step7_${mode}_${ver}"
du -sh "$root/models/step7_${mode}_${ver}"
echo "STEP7_${mode^^}_COMPLETE"
