"""Run the approved A runner unchanged, with the policy's instruction replaced in memory by a demo spec's
policy_instruction. RoboLab, Isaac-GR00T and bowl_step5 are never edited. RoboLab's Python 3.11, VM only.

Why a wrapper: RoboLab has no free-text instruction flag. --instruction-type only selects among a task's own instruction
fields (robolab/eval/runner.py:95-98), and a plain-string task such as the 10cm banana task resolves any requested type to
its sole "default" field (robolab/core/task/instruction.py, select_instruction_fields). create_env() sets
env_cfg.instruction = select_instruction_fields(env_cfg.instruction, instruction_type) (robolab/core/environments/
runtime.py:170 and :203), and everything downstream reads env_cfg.instruction: the GR00T request
(policies/gr00t/client.py:249, instruction_text -> annotation.language.language_instruction), the run banner, the video
file names and the episode results. The A runner records what the policy was sent in synchronized.hdf5
attrs['instruction'] (bowl_step5/run_controlled_episode_v4_registered.py:146).

How: an import hook wraps runtime.py's own reference to select_instruction_fields when the runner first imports
runtime.py (after Isaac Sim has started, exactly as without the wrapper). The wrapped function calls the original,
requires its result to be exactly {"default": TESTED_DEFAULT}, prints one INTENT_OVERRIDE_APPLIED line and returns
{"default": <override>}. The runner itself is executed with runpy under its own path, sys.argv and sys.path[0], after
its sha256 is checked against the step 8 source lock.

  INTENT_DEMO_SPEC=<spec.json> INTENT_DEMO_SPEC_SHA256=<sha256> python -u a_instruction_wrapper.py <A runner arguments>
"""
import hashlib
import importlib.abc
import json
import os
import runpy
import sys
from pathlib import Path

RUNNER = Path.home() / 'robot-transfer/bowl/experiments/bowl_step5/run_controlled_episode_v4_registered.py'
RUNNER_SHA256 = 'a319119268b3eaf93934c7a609137d99be422a854c8300698f34d19d880531dd'  # bowl_step8/step8-source-lock.sha256
TESTED_DEFAULT = 'Pick up the banana and place it in the bowl'  # BananaInBowlUniformInitPose10cmTask.instruction
TARGET = 'robolab.core.environments.runtime'
SPEC_SCHEMAS = ('intent-demo-spec-v1', 'intent-demo-literal-control-v1')
MAX_CHARS = 160


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_override(spec_path, want_sha):
    """The instruction to send, from a spec file pinned by its sha256; raises on anything unexpected."""
    got = sha256(spec_path)
    if got != want_sha:
        raise SystemExit(f'REFUSE: {spec_path} sha256 {got} != {want_sha}')
    spec = json.loads(Path(spec_path).read_text())
    if spec.get('schema') not in SPEC_SCHEMAS:
        raise SystemExit(f"REFUSE: spec schema {spec.get('schema')!r}")
    if (spec.get('decision') or {}).get('attempt') is not True:
        raise SystemExit('REFUSE: spec decision.attempt is not true; this spec is not to be run')
    text = spec.get('policy_instruction')
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS or '\n' in text or not text.isprintable():
        raise SystemExit(f'REFUSE: policy_instruction must be one printable line of 1-{MAX_CHARS} characters: {text!r}')
    return text


def patch_runtime(module, override):
    original = module.select_instruction_fields

    def select_instruction_fields(instruction, requested=None):
        fields = original(instruction, requested)
        if dict(fields) != {'default': TESTED_DEFAULT}:
            raise RuntimeError(f'intent demo: expected the 10cm task default instruction, got {dict(fields)!r}')
        print('INTENT_OVERRIDE_APPLIED', json.dumps({'from': fields['default'], 'to': override}), flush=True)
        return {'default': override}

    module.select_instruction_fields = select_instruction_fields


class OverrideHook(importlib.abc.MetaPathFinder):
    """Finds runtime.py with the remaining finders, then patches it right after its own code has run."""

    def __init__(self, override):
        self.override = override

    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is not self and hasattr(finder, 'find_spec'):
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    break
        if spec is None or spec.loader is None:
            return None
        real_exec, override = spec.loader.exec_module, self.override

        def exec_module(module):
            real_exec(module)
            patch_runtime(module, override)
        spec.loader.exec_module = exec_module
        return spec


def main(runner=RUNNER, runner_sha=RUNNER_SHA256):
    override = load_override(os.environ['INTENT_DEMO_SPEC'], os.environ['INTENT_DEMO_SPEC_SHA256'])
    if sha256(runner) != runner_sha:
        raise SystemExit(f'REFUSE: {runner} does not match the step 8 source lock')
    if TARGET in sys.modules:
        raise SystemExit(f'REFUSE: {TARGET} already imported; the override could not be installed')
    print('INTENT_SPEC_VERIFIED', os.environ['INTENT_DEMO_SPEC_SHA256'], json.dumps(override), flush=True)
    sys.meta_path.insert(0, OverrideHook(override))
    sys.path[0] = str(Path(runner).parent)  # what `python <runner>` would have put first
    runpy.run_path(str(runner), run_name='__main__')


if __name__ == '__main__':
    main()
