"""a_instruction_wrapper.py (run under Python 3.11 like the VM's RoboLab venv, against a fake RoboLab), demo_plan.py,
and static checks of run_intent_demo.sh against run_step8.sh."""
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import a_instruction_wrapper as w
import demo_plan
from conftest import DEMO, env_without, write_literal_like

PY311 = shutil.which('python3.11') or sys.executable
PY312 = (shutil.which('python3.12') or next((str(p) for p in Path.home().glob('.local/share/uv/python/cpython-3.12*/bin/python3.12')), None)
         or sys.executable)
STEP8 = DEMO.parents[1] / 'experiments/bowl_step8/run_step8.sh'
OVERRIDE = 'Put the banana on the table to the left of the bowl'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fake_robolab(tmp_path, default=w.TESTED_DEFAULT):
    pkg = tmp_path / 'pkg/robolab/core/environments'
    pkg.mkdir(parents=True)
    for d in (pkg, pkg.parent, pkg.parent.parent):
        (d / '__init__.py').write_text('')
    (pkg / 'runtime.py').write_text(
        'def select_instruction_fields(instruction, requested=None):\n    return {"default": instruction}\n\n'
        'def create_env(task_instruction):\n    return select_instruction_fields(task_instruction, ["default"])\n')
    runner = tmp_path / 'step5/run_controlled_episode_v4_registered.py'
    runner.parent.mkdir()
    runner.write_text('import json, sys\nfrom robolab.core.environments import runtime\n'
                      f'print("RESULT", json.dumps(runtime.create_env({default!r})))\n'
                      'print("ARGV", json.dumps(sys.argv)); print("PATH0", sys.path[0]); print("FILE", __file__)\n')
    return runner


def run_wrapper(tmp_path, runner, spec, spec_sha, runner_sha=None):
    code = (f'import sys; sys.path.insert(0, {str(DEMO)!r}); import a_instruction_wrapper as w; '
            f'w.main(runner={str(runner)!r}, runner_sha={runner_sha or sha(runner)!r})')
    env = env_without('PYTHONPATH') | {'PYTHONPATH': str(tmp_path / 'pkg'), 'INTENT_DEMO_SPEC': str(spec),
                                      'INTENT_DEMO_SPEC_SHA256': spec_sha, 'PYTHONDONTWRITEBYTECODE': '1'}
    return subprocess.run([PY311, '-u', '-c', code, '--seed', '4004', '--headless'], env=env, capture_output=True, text=True)


def test_vm_side_files_parse_as_their_vm_python():
    ast.parse((DEMO / 'a_instruction_wrapper.py').read_text(), feature_version=(3, 11))
    ast.parse((DEMO / 'demo_plan.py').read_text(), feature_version=(3, 12))


def test_override_reaches_the_runner(tmp_path):
    runner = fake_robolab(tmp_path)
    spec = write_literal_like(tmp_path / 'spec.json', policy_instruction=OVERRIDE)
    p = run_wrapper(tmp_path, runner, spec, sha(spec))
    assert p.returncode == 0, p.stderr
    lines = dict(x.split(' ', 1) for x in p.stdout.splitlines() if ' ' in x)
    assert json.loads(lines['RESULT']) == {'default': OVERRIDE}
    assert json.loads(lines['INTENT_OVERRIDE_APPLIED']) == {'from': w.TESTED_DEFAULT, 'to': OVERRIDE}
    assert json.loads(lines['ARGV']) == [str(runner), '--seed', '4004', '--headless']  # the runner sees its own argv
    assert lines['PATH0'] == str(runner.parent) and lines['FILE'] == str(runner)
    assert p.stdout.count('INTENT_OVERRIDE_APPLIED') == 1


def test_wrapper_refuses_an_unexpected_task_instruction(tmp_path):
    runner = fake_robolab(tmp_path, default='Put the fruit in the bowl')
    spec = write_literal_like(tmp_path / 'spec.json')
    p = run_wrapper(tmp_path, runner, spec, sha(spec))
    assert p.returncode != 0 and 'expected the 10cm task default instruction' in p.stderr


@pytest.mark.parametrize('case', ['spec_sha', 'runner_sha', 'not_attempted', 'two_lines', 'too_long'])
def test_wrapper_refusals(tmp_path, case):
    runner = fake_robolab(tmp_path)
    over = {'not_attempted': {'decision': {'attempt': False}}, 'two_lines': {'policy_instruction': 'a\nb'},
            'too_long': {'policy_instruction': 'x' * 161}}.get(case, {})
    spec = write_literal_like(tmp_path / 'spec.json', **over)
    p = run_wrapper(tmp_path, runner, spec, '0' * 64 if case == 'spec_sha' else sha(spec),
                    runner_sha='0' * 64 if case == 'runner_sha' else None)
    assert p.returncode != 0 and 'REFUSE' in p.stderr and 'RESULT' not in p.stdout


def test_wrapper_pins_match_the_locks():
    lock = (DEMO.parents[1] / 'experiments/bowl_step8/step8-source-lock.sha256').read_text()
    assert f'{w.RUNNER_SHA256}  /home/miodrag/robot-transfer/bowl/experiments/bowl_step5/run_controlled_episode_v4_registered.py' in lock
    assert sha(DEMO.parents[1] / 'experiments/bowl_step5/run_controlled_episode_v4_registered.py') == w.RUNNER_SHA256


def make_plan(tmp_path, episodes, specs):
    for name, over in specs.items():
        write_literal_like(tmp_path / 'specs' / name, **over)
    eps = [dict(e, spec_sha256=e.get('spec_sha256') or sha(tmp_path / e['spec'])) if (tmp_path / e['spec']).is_file()
           else dict(e, spec_sha256='0' * 64) for e in episodes]
    plan = tmp_path / 'demo_plan.json'
    plan.write_text(json.dumps({'schema': demo_plan.SCHEMA, 'episodes': eps}))
    return plan


def test_plan_good(tmp_path):
    plan = make_plan(tmp_path, [{'tag': 'rawfruit', 'layout': 4004, 'spec': 'specs/a.json'},
                                {'tag': 'leftbeside', 'layout': 4006, 'spec': 'specs/b.json'}],
                     {'a.json': {}, 'b.json': {'policy_instruction': OVERRIDE}})
    rows = demo_plan.check(plan)
    assert [r[0] for r in rows] == ['intentdemo_A_P0_l4004_rawfruit_v1', 'intentdemo_A_P0_l4006_leftbeside_v1']
    p = subprocess.run([PY312, '-I', str(DEMO / 'demo_plan.py'), 'check', str(plan)], capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.splitlines()[1].split('\t')[:2] == ['intentdemo_A_P0_l4006_leftbeside_v1', '4006']


def test_plan_problems_are_all_reported(tmp_path):
    plan = make_plan(tmp_path, [{'tag': 'bad_tag', 'layout': 3000, 'spec': 'specs/a.json'},
                                {'tag': 'x', 'layout': 4004, 'spec': 'specs/no.json'},
                                {'tag': 'x', 'layout': 4004, 'spec': 'specs/a.json', 'spec_sha256': 'f' * 64},
                                {'tag': 'x', 'layout': 4004, 'spec': 'specs/a.json'},
                                {'tag': 'y', 'layout': 4005, 'spec': 'specs/c.json'},
                                {'tag': 'z', 'layout': 4005, 'spec': '../escape.json'}],
                     {'a.json': {}, 'c.json': {'decision': {'attempt': False}}})
    (tmp_path.parent / 'escape.json').write_text('{}')
    with pytest.raises(ValueError) as e:
        demo_plan.check(plan)
    text = '\n'.join(e.value.args[0])
    for needle in ("tag 'bad_tag'", 'layout 3000', "spec 'specs/no.json'", 'sha256', 'not to be attempted',
                   "'../escape.json'", 'duplicate case names'):
        assert needle in text, needle


def test_plan_size_limit(tmp_path):
    plan = make_plan(tmp_path, [{'tag': f't{i}', 'layout': 4004, 'spec': 'specs/a.json'} for i in range(7)], {'a.json': {}})
    with pytest.raises(ValueError, match='1-6'):
        demo_plan.check(plan)


def test_declared_tags_make_valid_unique_case_names():
    eps = json.loads((DEMO / 'declared_plan.json').read_text())['episodes']
    cases = [demo_plan.case_name(e['layout'], e['tag']) for e in eps]
    assert len(set(cases)) == len(cases) == len(eps) <= demo_plan.MAX_EPISODES
    assert all(re.fullmatch(r'[a-z0-9]{1,24}', e['tag']) and e['layout'] in demo_plan.LAYOUTS for e in eps)


def heredocs(text):
    return re.findall(r"<<'PY'\n(.*?)\nPY\n", text, flags=re.S)


def test_supervisor_copies_step8_blocks_verbatim():
    ours, theirs = (DEMO / 'run_intent_demo.sh').read_text(), STEP8.read_text()
    mine = heredocs(ours)
    for block in heredocs(theirs):
        if 'model-manifest.json' in block or "policy_label']=='P0'" in block or 'audit_capture.py' in block:
            assert block in mine, block[:60]
    assert subprocess.run(['bash', '-n', str(DEMO / 'run_intent_demo.sh')]).returncode == 0


def test_supervisor_gates_and_hygiene():
    s = (DEMO / 'run_intent_demo.sh').read_text()
    for gate in ('step8_after_v1-supervisor"', '/exit-code.txt" ] || { echo "REFUSE',
                 "nvidia-smi --query-compute-apps=pid --format=csv,noheader)\" ] || { echo 'REFUSE",
                 "ss -ltnH 'sport = :15555'", '--remote-host 127.0.0.1', 'trap finish EXIT', "trap 'exit 143' TERM",
                 'verify_trees after', 'ORPHAN_GPU_PROCESS'):
        assert gate in s, gate
    for forbidden in ('git checkout', 'git reset', 'git clean', 'git stash', 'pip install', 'rm -', 'sed -i', '> "$root/RoboLab/'):
        assert forbidden not in s, forbidden
    assert s.count('run_controlled_episode_v4_registered.py') == 0  # reached only through the wrapper's pinned path


def test_lock_file_matches_local_files():
    lock = (DEMO / 'intent-demo-lock.sha256').read_text().splitlines()
    names = []
    for line in lock:
        digest, path = line.split('  ', 1)
        assert path.startswith('/home/miodrag/robot-transfer/bowl/experiments/intent_demo_v1/')
        name = path.rsplit('/', 1)[1]
        names.append(name)
        assert sha(DEMO / name) == digest, f'{name} changed since the lock was written; regenerate it'
    assert sorted(names) == ['a_instruction_wrapper.py', 'demo_plan.py', 'run_intent_demo.sh']
