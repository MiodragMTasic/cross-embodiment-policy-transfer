"""Step 8 B abort classifier: is a B episode's exception a task outcome or an infrastructure crash?

  classify_abort.py <runs/CASE> <bridge exit status>
Reads failure.txt, writes abort-classification.json beside it (never overwrites) and prints the class.
safety_abort: the bridge stopped on what the policy commanded, through the frozen adapter, mid-episode. A task outcome:
  scored by v5.1 and stages_r2 on the partial trace the runner's finally block saved, included in delta-B, never counted
  toward the 3-failure halt. Every kind also needs at least one command executed: gate_trace.jsonl has >= 1 row.
crash: everything else, including any exception not matched exactly below, any abort before the first gate_trace row
  (step 0) and any timeout (exit 124/137).
Line numbers are step8_B_runner.py's (frozen run_painted_dev.py, +2 after line 28); the runner is locked by sha256.
"""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

RUNNER = '/experiments/bowl_step6/step8_B_runner.py'
LOCAL_IK = '/experiments/bowl_step6/local_ik.py'
SOLVE_ASSERT = "assert result['success'], (step,direction,result['reason'])"
# kind: (innermost frame (file, line, function, source), caller frame in the runner (line, function) or None,
#        exception type, message test)
ABORTS = {
    # B's measured joints beyond the simulator soft limits (+0.03 rad) after a step: what the commands drove the arm to.
    'B_joint_beyond_soft_limit': ((RUNNER, 201, 'state', 'assert (q>=soft[:,0]-.03).all() and (q<=soft[:,1]+.03).all()'),
                                  (298, '<module>'), 'AssertionError', lambda m: m == ''),
    # The policy's own A joint target (chunk row) outside A's physical limits.
    'policy_A_target_outside_limits': ((RUNNER, 273, '<module>', "assert (Aq>=Alim[:,0]).all() and (Aq<=Alim[:,1]).all(), 'P0 A joint target outside physical limits'"),
                                       None, 'AssertionError', lambda m: m == 'P0 A joint target outside physical limits'),
    # B cannot reach the policy's A tool target (adapter action map).
    'ik_action_A_to_B': ((RUNNER, 85, 'log_solve', SOLVE_ASSERT), (282, '<module>'), 'AssertionError',
                         lambda m: solve_message(m, 'action_A_to_B')),
    # A cannot represent where the commands took B (adapter observation map).
    'ik_observation_B_to_A': ((RUNNER, 85, 'log_solve', SOLVE_ASSERT), (237, '<module>'), 'AssertionError',
                              lambda m: solve_message(m, 'observation_B_to_A')),
    # B's measured joints past its physical limits, fed as the action IK seed: the same physical event as the soft limit.
    'B_joint_beyond_physical_limit': ((LOCAL_IK, 24, 'solve', "raise ValueError('Seed outside physical limits')"),
                                      (282, '<module>'), 'ValueError', lambda m: m == 'Seed outside physical limits'),
}
FRAME = re.compile(r'^  File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<fn>.+)$')


def solve_message(message, direction):
    try:
        step, d, reason = ast.literal_eval(message)
    except (ValueError, SyntaxError, TypeError):
        return False
    return type(step) is int and d == direction and isinstance(reason, str)


def parse(text):
    block = text.split('Traceback (most recent call last):')[-1].splitlines()
    frames, last = [], None
    for i, line in enumerate(block):
        m = FRAME.match(line)
        if m:
            source = block[i + 1].strip() if i + 1 < len(block) and block[i + 1].startswith('    ') else ''
            frames.append((m['file'], int(m['line']), m['fn'], source))
            last = i
    if last is None:
        return [], '', ''
    rest = [l for l in block[last + 1:] if not l.startswith('    ')]
    exc = '\n'.join(rest).strip()
    kind, _, message = exc.partition(': ')
    return frames, kind, message


def classify(frames, exc, message, status, gate_rows):
    # A gate_trace row is appended (:297) only after env.step: no row means no command reached the simulator yet.
    # Rows equal the loop step at :237/:273/:282, and step+1 at :298, so a soft-limit abort after the first step still counts.
    if status in (124, 137) or not frames or gate_rows < 1:
        return None
    runner = [f for f in frames if f[0].endswith(RUNNER)]
    for name, ((file, line, fn, source), caller, kind, test) in ABORTS.items():
        f = frames[-1]
        if not (f[0].endswith(file) and f[1:] == (line, fn, source) and exc == kind and test(message)):
            continue
        outer = [r for r in runner if r is not f]
        if caller is None or (outer and outer[-1][1:3] == caller):
            return name
    return None


def main():
    run, status = Path(sys.argv[1]), int(sys.argv[2])
    text = (run/'failure.txt').read_text()
    frames, exc, message = parse(text)
    gate = run/'gate_trace.jsonl'
    rows = [json.loads(l) for l in gate.read_text().splitlines() if l.strip()] if gate.exists() else []
    kind = classify(frames, exc, message, status, len(rows))
    record = {'schema': 'step8-abort-classification-v1', 'case': run.name, 'class': 'safety_abort' if kind else 'crash',
              'kind': kind, 'exit_status': status, 'exception': exc, 'message': message,
              'frames': [{'file': f, 'line': l, 'function': fn} for f, l, fn, _ in frames],
              'failure_txt_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'partial_trace': {'gate_trace_rows': len(rows), 'last_gate_trace_step': rows[-1]['step'] if rows else None,
                                'tracking_npz': (run/'tracking.npz').exists(), 'failed_solve_json': (run/'failed_solve.json').exists(),
                                'checks_json': (run/'checks.json').exists()},
              'rule': 'safety_abort is a task outcome: scored by v5.1 and stages_r2 on the partial trace, included in delta-B, '
                      'not counted toward the 3-failure halt. crash is infrastructure and counts toward the halt.'}
    with open(run/'abort-classification.json', 'x') as out:
        out.write(json.dumps(record, indent=2) + '\n')
    print(record['class'])


if __name__ == '__main__':
    main()
