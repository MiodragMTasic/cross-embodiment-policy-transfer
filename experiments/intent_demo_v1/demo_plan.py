"""Validate an intent demo plan and print one tab-separated line per episode: case, layout, spec path, spec sha256.
Standard library only, so the VM's system Python runs it (python3 -I demo_plan.py check <plan.json>).

Plan schema intent-demo-plan-v1:
  {"schema": "intent-demo-plan-v1", "episodes": [{"tag": "fruitaway", "layout": 4004,
    "spec": "specs/put_the_fruit_away.json", "spec_sha256": "<64 hex>"}, ...]}
Rules: 1-6 episodes; tag [a-z0-9]{1,24}; layout one of the step 8 layouts 4000-4011 (the before phase scored P0 on all of
them); spec path relative to the plan's folder and inside specs/; the file's sha256 equals spec_sha256; the spec says
decision.attempt is true and carries a one-line policy_instruction; case names intentdemo_A_P0_l{L}_{tag}_v1 are unique.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA = 'intent-demo-plan-v1'
MAX_EPISODES = 6
LAYOUTS = range(4000, 4012)
SPEC_SCHEMAS = ('intent-demo-spec-v1', 'intent-demo-literal-control-v1')


def case_name(layout, tag):
    return f'intentdemo_A_P0_l{layout}_{tag}_v1'


def check(plan_path):
    """[(case, layout, spec_path, sha256)] or raises ValueError naming every problem."""
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    problems, rows = [], []
    if plan.get('schema') != SCHEMA:
        problems.append(f"schema {plan.get('schema')!r} != {SCHEMA}")
    eps = plan.get('episodes')
    if not isinstance(eps, list) or not 1 <= len(eps) <= MAX_EPISODES:
        raise ValueError(problems + [f'episodes must be a list of 1-{MAX_EPISODES}'])
    specs_dir = (plan_path.parent / 'specs').resolve()
    for i, e in enumerate(eps):
        at = f'episode {i}'
        tag, layout, spec, want = e.get('tag'), e.get('layout'), e.get('spec'), e.get('spec_sha256')
        if not isinstance(tag, str) or not re.fullmatch(r'[a-z0-9]{1,24}', tag):
            problems.append(f'{at}: tag {tag!r}')
        if not isinstance(layout, int) or isinstance(layout, bool) or layout not in LAYOUTS:
            problems.append(f'{at}: layout {layout!r} not in 4000-4011')
        if not isinstance(want, str) or not re.fullmatch(r'[0-9a-f]{64}', want):
            problems.append(f'{at}: spec_sha256 {want!r}')
        path = (plan_path.parent / str(spec)).resolve()
        if not path.is_relative_to(specs_dir) or not path.is_file():
            problems.append(f'{at}: spec {spec!r} is not a file under specs/')
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            problems.append(f'{at}: {spec} sha256 {got} != {want}')
        s = json.loads(path.read_text())
        text = s.get('policy_instruction')
        if s.get('schema') not in SPEC_SCHEMAS:
            problems.append(f"{at}: spec schema {s.get('schema')!r}")
        if (s.get('decision') or {}).get('attempt') is not True:
            problems.append(f'{at}: spec {spec} is not to be attempted (decision.attempt)')
        if not isinstance(text, str) or not text.strip() or '\n' in text or len(text) > 160:
            problems.append(f'{at}: policy_instruction {text!r}')
        rows.append((case_name(layout, tag), layout, str(path), got))
    cases = [r[0] for r in rows]
    if len(set(cases)) != len(cases):
        problems.append(f'duplicate case names: {sorted(c for c in cases if cases.count(c) > 1)}')
    if problems:
        raise ValueError(problems)
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != 'check':
        print('usage: demo_plan.py check <plan.json>', file=sys.stderr)
        return 2
    try:
        rows = check(argv[1])
    except ValueError as e:
        for p in e.args[0]:
            print(f'PLAN_PROBLEM {p}', file=sys.stderr)
        return 1
    for r in rows:
        print('\t'.join(map(str, r)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
