"""Collect only the four proposed training-layout v4 feasibility cases; stop on any anomaly."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

root = Path.home() / 'robot-transfer/bowl'
code = root / 'experiments/bowl_step5'
lock = json.loads((code / 'training-feasibility-source-lock-v4.json').read_text())
ledger = root / 'runs/training-feasibility-v4.jsonl'
assert not ledger.exists(), 'Never silently repeat a batch'

def record(**data):
    data['utc'] = datetime.now(timezone.utc).isoformat()
    with ledger.open('a') as stream:
        stream.write(json.dumps(data) + '\n')
    print(json.dumps(data), flush=True)

def verify_sources():
    for relative, expected in lock['files'].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
    for repo, commit in lock['commits'].items():
        assert subprocess.check_output(['git', '-C', str(root / repo), 'rev-parse', 'HEAD'], text=True).strip() == commit
        assert not subprocess.check_output(['git', '-C', str(root / repo), 'status', '--porcelain'], text=True).strip()

for seed in range(2000, 2004):
    case = f'step5_training_layout{seed}_v4'
    task = 'BananaInBowlUniformInitPose10cmTask'
    try:
        verify_sources()
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        record(case=case, stage='started', layout_seed=seed, policy_seed=0)
        subprocess.run(['bash', str(code / 'run_controlled_case_v4_registered.sh'), case, task, str(seed), '0'], check=True)
        result = json.loads((root / 'runs' / case / 'capture-integrity.json').read_text())
        assert result['layout_seed'] == seed and result['policy_seed'] == 0
        assert result['task_class'] == [task]
        assert not (root / 'runs' / case / 'gpu-after.txt').read_text().strip()
        verify_sources()
        record(case=case, stage='finished', infrastructure_ok=True, passed=result['passed'],
               steps=result['trace_rows'], stable_success_s=result['stable_success_s'])
    except Exception as error:
        record(case=case, stage='halted_for_review', error=repr(error))
        raise
record(stage='batch_finished', next='Review every attempt and adjudicate successful data; no P1 optimization or step6 until step5 approval.')
