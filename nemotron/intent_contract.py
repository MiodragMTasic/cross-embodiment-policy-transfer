"""#2 Intent contract: natural-language instruction -> schema-checked task spec, scored against the project's success rule.

The scene's object list and RoboLab's instruction strings are parsed from the hash-checked evidence copies of the
RoboLab task files (NVlabs/RoboLab b6cfcf6c), so the enums are the scene's own objects. The project's success
definition (SCORING_V5.md, Miodrag's rule) is the reference; it is never shown to the model, and every quote below must
appear verbatim in SCORING_V5.md (test_intent_contract checks it).

  python intent_contract.py --list     print the instruction set with sources (offline)
"""
import argparse
import ast
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SCORING_DOC, TASK_BENCH, TASK_10CM, ref, sha256  # noqa: E402

PROMPT_VERSION = 'intent-contract-v1'
TASK_SHA = {TASK_10CM.name: 'bea6d8bfca1e32cd7fc5e05bf8b88af98ca782157bc59eda8462b9af5c6729fd',  # pilot-source-lock.json
            TASK_BENCH.name: 'ceaba5a9944cf6a9cd1ba6809a73251c01786059c0d4bd02604bf0c9c824ab4a'}  # bowl-feasibility manifest
RELATIONS = ['in_or_on', 'on_rim', 'near', 'other']
ABSENT = 'not_in_scene'

# Project success definition (the reference). Quotes are verbatim from SCORING_V5.md "## Rule".
PROJECT_RULE_QUOTES = [
    'the policy commands the gripper open while the banana is held and lifted;',
    'the banana lands on the original bowl (rim counts) directly from that release;',
    'it then settles there.',
    'Both a set-down and a drop count. A slip from a closed gripper never counts.',
]
PROJECT_SPEC = {'object': 'banana', 'target': 'bowl', 'relation': 'in_or_on',
                'success_predicate': {'settle_on_or_in_target': True, 'commanded_release_required': True, 'rim_counts': True}}
FIELDS = ['object', 'target', 'relation', 'settle_on_or_in_target', 'commanded_release_required', 'rim_counts']

# Hand-written paraphrases (fixtures, not RoboLab strings). 'faithful' is what a correct reading of THAT sentence gives,
# which for the adversarial ones deliberately differs from the project rule; 'ambiguous' marks sentences a careful
# compiler should flag.
PARAPHRASES = [
    ('put the yellow fruit in the dish', 'synonym', {'object': 'banana', 'target': 'bowl', 'relation': 'in_or_on'}, False),
    ('set the banana down in the bowl and let go of it', 'explicit_release',
     {'object': 'banana', 'target': 'bowl', 'relation': 'in_or_on'}, False),
    ('toss the banana into the bowl', 'throw', {'object': 'banana', 'target': 'bowl', 'relation': 'in_or_on'}, False),
    ('drop the banana near the bowl', 'adversarial_near', {'object': 'banana', 'target': 'bowl', 'relation': 'near'}, True),
    ('move the banana over to the bowl', 'adversarial_vague_relation', {'object': 'banana', 'target': 'bowl'}, True),
    ('balance the banana on the edge of the bowl', 'adversarial_rim',
     {'object': 'banana', 'target': 'bowl', 'relation': 'on_rim'}, False),
    ("put the banana in the bowl but don't let go of it", 'adversarial_no_release',
     {'object': 'banana', 'target': 'bowl', 'relation': 'in_or_on', 'commanded_release_required': False}, True),
    ('put the apple in the bowl', 'adversarial_absent_object', {'object': ABSENT, 'target': 'bowl'}, True),
]


def _class_attrs(path):
    """{class name: {attr: literal}} for literal class-level assignments in a task file (no import, no Isaac)."""
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.ClassDef):
            attrs = {}
            for st in node.body:
                target = st.targets[0] if isinstance(st, ast.Assign) else st.target if isinstance(st, ast.AnnAssign) else None
                if isinstance(target, ast.Name) and getattr(st, 'value', None) is not None:
                    try:
                        attrs[target.id] = ast.literal_eval(st.value)
                    except ValueError:
                        pass
            out[node.name] = attrs
    return out


def robolab_task():
    """Scene objects and the three RoboLab phrasings, with sources. Raises if a task copy's hash changed."""
    for p in (TASK_10CM, TASK_BENCH):
        if sha256(p) != TASK_SHA[p.name]:
            raise RuntimeError(f'{p} does not match its recorded sha256; refusing to use it')
    ten = _class_attrs(TASK_10CM)['BananaInBowlUniformInitPose10cmTask']
    bench = _class_attrs(TASK_BENCH)['BananaInBowlTask']
    insts = [
        {'id': 'robolab_default', 'text': ten['instruction'], 'group': 'robolab', 'level': 'default',
         'source': ref(TASK_10CM), 'note': 'the string every step 8 episode was run with (BananaInBowlUniformInitPose10cmTask)'},
        {'id': 'robolab_vague', 'text': bench['instruction']['vague'], 'group': 'robolab', 'level': 'vague',
         'source': ref(TASK_BENCH), 'note': 'parent BananaInBowlTask; the 10cm variant ships only the default string'},
        {'id': 'robolab_specific', 'text': bench['instruction']['specific'], 'group': 'robolab', 'level': 'specific',
         'source': ref(TASK_BENCH), 'note': 'parent BananaInBowlTask; says "red bowl", bowl colour not checked in the scene'},
    ]
    assert bench['instruction']['default'] == ten['instruction']
    return ten['contact_object_list'], insts


def instructions():
    scene, insts = robolab_task()
    for text, kind, faithful, ambiguous in PARAPHRASES:
        insts.append({'id': f'para_{kind}', 'text': text, 'group': 'adversarial' if kind.startswith('adversarial') else 'paraphrase',
                      'level': kind, 'source': {'file': 'nemotron/intent_contract.py', 'sha256': None},
                      'note': 'hand-written paraphrase', 'faithful': faithful, 'ambiguous': ambiguous})
    for x in insts[:3]:
        x.update(faithful={'object': 'banana', 'target': 'bowl', 'relation': 'in_or_on'}, ambiguous=False)
    return scene, insts


def schema(scene, order_seed=None):
    objs, rels = list(scene) + [ABSENT], RELATIONS[:]
    if order_seed is not None:
        rng = random.Random(order_seed)
        rng.shuffle(objs)
        rng.shuffle(rels)
    text = {'type': 'array', 'items': {'type': 'string', 'maxLength': 160}, 'maxItems': 6}
    return {'type': 'object', 'additionalProperties': False,
            'required': ['object', 'target', 'relation', 'success_predicate', 'constraints', 'ambiguities', 'confidence'],
            'properties': {
                'object': {'type': 'string', 'enum': objs},
                'target': {'type': 'string', 'enum': objs[:]},
                'relation': {'type': 'string', 'enum': rels},
                'success_predicate': {'type': 'object', 'additionalProperties': False,
                                      'required': ['settle_on_or_in_target', 'commanded_release_required', 'rim_counts'],
                                      'properties': {k: {'type': 'boolean'} for k in
                                                     ('settle_on_or_in_target', 'commanded_release_required', 'rim_counts')}},
                'constraints': text, 'ambiguities': text,
                'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}}}


SYSTEM = """You compile one natural-language instruction for a robot arm into a task contract for one fixed scene.
Scene objects: {scene}. Use "not_in_scene" if the instruction names an object that is not in this list.
Fields:
- object: the object to be moved. target: where it should end up.
- relation: in_or_on = the object ends inside or resting on the target; on_rim = deliberately on the target's rim or edge; near = beside the target, not on or in it; other = anything else.
- success_predicate.settle_on_or_in_target: true if success requires the object to come to rest on or in the target, not merely reach or touch it.
- success_predicate.commanded_release_required: true if success requires the robot to deliberately open its gripper to let go, so that an accidental slip or drop would not count.
- success_predicate.rim_counts: true if the object resting on the target's rim, supported by the target, should count as success.
- constraints: requirements the instruction places on how the task is done (short phrases; empty if none).
- ambiguities: anything a careful operator would need to clarify before running it (empty if none).
- confidence: 0 to 1.
Read the instruction literally; do not assume what the user "probably" meant. Reply with one JSON object matching this schema:
"""


def messages(text, scene, sch):
    return [{'role': 'system', 'content': SYSTEM.format(scene=', '.join(scene)) + json.dumps(sch)},
            {'role': 'user', 'content': f'Instruction: "{text}"'}]


def flat(spec):
    sp = spec.get('success_predicate', {})
    return {'object': spec.get('object'), 'target': spec.get('target'), 'relation': spec.get('relation'), **sp}


def compare(spec, inst):
    """Per-field agreement with the project rule, plus the faithful-reading check for this sentence."""
    got, rule = flat(spec), flat(PROJECT_SPEC)
    return {'vs_project': {f: got.get(f) == rule[f] for f in FIELDS},
            'vs_faithful': {f: got.get(f) == v for f, v in inst['faithful'].items()},
            'flagged_ambiguity': bool(spec.get('ambiguities')), 'ambiguity_expected': inst['ambiguous']}


def project_rule_source():
    return ref(SCORING_DOC)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true', required=True)
    ap.parse_args(argv)
    scene, insts = instructions()
    print(f'scene objects: {scene}')
    for x in insts:
        print(f"{x['id']:34} {x['text']!r:60} {x['source']['file']} {(x['source']['sha256'] or '')[:12]}")


if __name__ == '__main__':
    main()
