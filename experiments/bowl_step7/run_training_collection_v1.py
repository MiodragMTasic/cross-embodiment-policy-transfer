"""Step 7 A-only training collection with the unchanged step 5 v4 runner; stop on any anomaly.

--mode first: layouts 2004-2031, policy seed 0 (2000-2003 are reused; S2 hashes matched).
--mode seed1: the single declared D3 pass over 2000-2031 with policy seed 1. Run only by explicit decision after the
first pass is v5-scored and reviewed with fewer than 6 wins; never automatic, never repeated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

TRAINING_LAYOUTS = range(2000, 2032)  # A training split only; validation 3000-3007 and test 4000-4023 never enter here
parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True, choices=('first', 'seed1'))
args = parser.parse_args()

root = Path.home() / 'robot-transfer/bowl'
code = root / 'experiments/bowl_step5'
step7 = root / 'experiments/bowl_step7'
lock = json.loads((code / 'training-feasibility-source-lock-v4.json').read_text())
first_ledger = root / 'runs/training-collection-v1.jsonl'
if args.mode == 'first':
    layouts, policy_seed, ledger = range(2004, 2032), 0, first_ledger
else:
    assert json.loads(first_ledger.read_text().splitlines()[-1])['stage'] == 'batch_finished', 'seed1 only after the first pass'
    layouts, policy_seed, ledger = range(2000, 2032), 1, root / 'runs/training-collection-v1-seed1.jsonl'
assert all(seed in TRAINING_LAYOUTS for seed in layouts), 'Only A training layouts 2000-2031'
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

def verify_step7():
    subprocess.run(['sha256sum', '--quiet', '-c', str(step7 / 'step7-collect-source-lock.sha256')], check=True)

def gpu_idle():
    for wait in range(60):
        if not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            return
        time.sleep(1)
    raise AssertionError('GPU still busy before the next case')

for seed in layouts:
    assert seed in TRAINING_LAYOUTS, seed
    case = f'step7_collect_layout{seed}_s{policy_seed}_v1'
    task = 'BananaInBowlUniformInitPose10cmTask'
    try:
        verify_sources()
        verify_step7()
        gpu_idle()
        record(case=case, stage='started', layout_seed=seed, policy_seed=policy_seed)
        subprocess.run(['bash', str(code / 'run_controlled_case_v4_registered.sh'), case, task, str(seed), str(policy_seed)], check=True)
        result = json.loads((root / 'runs' / case / 'capture-integrity.json').read_text())
        assert result['layout_seed'] == seed and result['policy_seed'] == policy_seed
        assert result['task_class'] == [task]
        assert not (root / 'runs' / case / 'gpu-after.txt').read_text().strip()
        verify_sources()
        record(case=case, stage='finished', infrastructure_ok=True, passed_v4=result['passed'],
               steps=result['trace_rows'], stable_success_s=result['stable_success_s'])
    except Exception as error:
        record(case=case, stage='halted_for_review', error=repr(error))
        raise
record(stage='batch_finished', mode=args.mode,
       next='Score every attempt offline with v5 (first-rest primary); reviewer checks v5 passes and the extended fail set; no training until review.')
