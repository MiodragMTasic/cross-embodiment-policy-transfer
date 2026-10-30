#!/bin/bash
# Step 7 A-only training collection (STEP7_PLAN_2026-10-23.md, reviewer conditions in EXPERIMENT_REPORT §22.4).
# Usage: run_collect_v1.sh first|seed1. The unchanged step 5 v4 case runner does every episode; this file only adds the
# step 6 infrastructure checks plus an Isaac version check, then hands control to run_training_collection_v1.py.
set -euo pipefail
mode="$1"
[ "$mode" = first ] || [ "$mode" = seed1 ]
root=/home/miodrag/robot-transfer/bowl
code="$root/experiments/bowl_step7"
supervisor="$root/runs/step7_collect_v1_${mode}-supervisor"
[ ! -e "$supervisor" ]
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
mkdir "$supervisor"
exec > >(tee "$supervisor/supervisor.log") 2>&1
finish() {
 status=$?
 trap - EXIT
 printf '%s\n' "$status" > "$supervisor/exit-code.txt"
 date -u
 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$supervisor/gpu-after.txt"
 exit "$status"
}
trap finish EXIT
date -u
sha256sum -c "$code/step7-collect-source-lock.sha256"
cp "$code/step7-collect-source-lock.sha256" "$code/run_collect_v1.sh" "$code/run_training_collection_v1.py" "$supervisor/"
[ "$(git -C "$root/RoboLab" rev-parse HEAD)" = b6cfcf6ccd2113c4cd1216d35b33024a7d5691da ]
[ "$(git -C "$root/Isaac-GR00T" rev-parse HEAD)" = dac54b313160c201c450f97df0fd74ebd379c892 ]
[ -z "$(git -C "$root/RoboLab" status --porcelain --untracked-files=no)" ]
[ -z "$(git -C "$root/Isaac-GR00T" status --porcelain --untracked-files=no)" ]
"$root/RoboLab/.venv/bin/python" -c "import importlib.metadata as m;v=(m.version('isaacsim'),m.version('isaaclab'));assert v==('5.1.0.0','2.3.2.post1'),v;print('VERIFIED_ISAAC',*v)"
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
# Each case is bounded inside the unchanged runner (server 900 s, episode 600 s); this bounds the whole batch.
timeout -k 60 54000 "$root/RoboLab/.venv/bin/python" -u "$code/run_training_collection_v1.py" --mode "$mode"
echo "STEP7_COLLECTION_COMPLETE $mode"
