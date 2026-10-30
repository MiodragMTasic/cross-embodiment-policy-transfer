"""Mechanical check of the step 8 runner rule (frozen_B_adapter_v1r2 README, "Rule for step 8").

  check_step8_diff.py B <frozen run_painted_dev.py> <step8_B_runner.py>
      every changed or inserted line must sit inside the allowed frozen line ranges; anything else fails.
  check_step8_diff.py A <run_controlled_episode_v4_registered.py> <approved step 5 source-hashes.txt>
      A runs the step 5 v4 registered runner itself: no allowed ranges, so it must be byte-identical to the
      file hashed for the approved training-feasibility runs.
Exit 0 and STEP8_DIFF_OK on success; exit 1 with every offending hunk otherwise.
"""
import difflib
import hashlib
import json
import sys
from pathlib import Path

B_ALLOWED = sorted({*range(24, 29), 31, *range(32, 35), *range(98, 106), 164, 315, 322, 323})


def check_B(frozen, derived):
    old, new = Path(frozen).read_text().split('\n'), Path(derived).read_text().split('\n')
    allowed, bad, hunks = set(B_ALLOWED), [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        # replace/delete touch old lines i1+1..i2; an insert sits between old lines i1 and i1+1
        touched = list(range(i1 + 1, i2 + 1)) if tag != 'insert' else [i1, i1 + 1]
        ok = all(n in allowed for n in touched) if tag != 'insert' else any(n in allowed for n in touched)
        hunk = {'tag': tag, 'frozen_lines': [i1 + 1, i2], 'derived_lines': [j1 + 1, j2], 'allowed': ok}
        hunks.append(hunk)
        if not ok:
            bad.append(hunk)
    return hunks, bad


def check_A(runner, source_hashes):
    digest = hashlib.sha256(Path(runner).read_bytes()).hexdigest()
    name = Path(runner).name
    recorded = [l.split()[0] for l in Path(source_hashes).read_text().splitlines() if l.split()[-1].endswith('/' + name)]
    assert len(recorded) == 1, recorded
    return {'runner_sha256': digest, 'approved_sha256': recorded[0]}, digest == recorded[0]


if __name__ == '__main__':
    kind, a, b = sys.argv[1:]
    if kind == 'B':
        hunks, bad = check_B(a, b)
        print(json.dumps({'allowed_frozen_lines': B_ALLOWED, 'hunks': hunks}, indent=1))
        if bad:
            print('STEP8_DIFF_FAIL B', json.dumps(bad))
            sys.exit(1)
        print('STEP8_DIFF_OK B', len(hunks), 'hunks, all inside the allowed ranges')
    elif kind == 'A':
        record, same = check_A(a, b)
        print(json.dumps(record))
        if not same:
            print('STEP8_DIFF_FAIL A runner differs from the approved step 5 runner')
            sys.exit(1)
        print('STEP8_DIFF_OK A byte-identical to the approved step 5 runner')
    else:
        sys.exit(f'unknown kind {kind}')
