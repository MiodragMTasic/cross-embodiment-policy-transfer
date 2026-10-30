"""Shared paths and guards for the Nemotron add-on.

Everything this package writes goes under nemotron/runs/. The project's evidence, locked scores and logs are read-only
inputs: write_text()/write_json() refuse any path outside RUNS, so a bug cannot touch them.
"""
import hashlib
import json
from pathlib import Path

PKG = Path(__file__).resolve().parent
PROJECT = PKG.parent
RUNS = PKG / 'runs'

EVIDENCE = PROJECT / 'assets/report/evidence/2026-10-24/step8_before_v1'
LOCKED_SCORE = PROJECT / 'experiments/scoring_v5/score_step8_before/score_step8-before.json'
REVIEW_LOG = PROJECT / 'STEP_REVIEW_LOG.md'
SCORING_DOC = PROJECT / 'experiments/scoring_v5/SCORING_V5.md'
TASK_DIR = PROJECT / 'assets/report/evidence/2026-10-21/bowl-feasibility'
TASK_10CM = TASK_DIR / 'robolab__robolab__tasks__randomize_initial_pose__banana_in_bowl_uniform_10cm.py'
TASK_BENCH = TASK_DIR / 'robolab__robolab__tasks__benchmark__banana_in_bowl_task.py'
TASK_NAME = 'BananaInBowlUniformInitPose10cmTask'

SEPARATION = ('Nemotron annotation, post-hoc. Not a score: never merged with the locked v5.1 scores or the independent '
              "reviewer's verdicts; compared against them only.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    """Evidence reference recorded with every annotation: project-relative path plus sha256."""
    path = Path(path).resolve()
    return {'file': str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path), 'sha256': sha256(path)}


def _guard(path):
    path = Path(path).resolve()
    if not path.is_relative_to(RUNS.resolve()):
        raise PermissionError(f'refusing to write outside {RUNS}: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path, text):
    _guard(path).write_text(text)


def write_json(path, obj):
    _guard(path).write_text(json.dumps(obj, indent=1, sort_keys=False) + '\n')


def append_jsonl(path, obj):
    with open(_guard(path), 'a') as f:
        f.write(json.dumps(obj) + '\n')


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()
