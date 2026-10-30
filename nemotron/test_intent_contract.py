"""Offline tests for the intent contract: sources, the project rule reference, schema and scoring."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intent_contract as ic  # noqa: E402
import tf_client as tf  # noqa: E402
from common import SCORING_DOC  # noqa: E402

GOOD = {'object': 'banana', 'target': 'bowl', 'relation': 'in_or_on', 'constraints': [], 'ambiguities': [],
        'success_predicate': {'settle_on_or_in_target': True, 'commanded_release_required': True, 'rim_counts': True},
        'confidence': 0.9}


class TestSources(unittest.TestCase):
    def test_robolab_strings(self):
        scene, insts = ic.robolab_task()
        self.assertEqual(scene, ['banana', 'bowl', 'table'])
        self.assertEqual([x['text'] for x in insts], [
            'Pick up the banana and place it in the bowl', 'Put the fruit in the bowl',
            'Grasp the yellow banana and place it inside the red bowl on the table'])
        self.assertEqual({x['source']['sha256'] for x in insts}, set(ic.TASK_SHA.values()))

    def test_changed_task_copy_refused(self):
        with mock.patch.dict(ic.TASK_SHA, {ic.TASK_BENCH.name: '0' * 64}):
            with self.assertRaises(RuntimeError):
                ic.robolab_task()

    def test_rule_quotes_verbatim(self):
        rule = SCORING_DOC.read_text().split('## How it is used')[0]
        for q in ic.PROJECT_RULE_QUOTES:
            self.assertIn(q, rule)

    def test_instruction_set(self):
        _, insts = ic.instructions()
        self.assertEqual(len(insts), 11)
        self.assertEqual(len({x['id'] for x in insts}), 11)
        self.assertIn('drop the banana near the bowl', [x['text'] for x in insts])
        self.assertIn('put the yellow fruit in the dish', [x['text'] for x in insts])

    def test_rule_not_leaked_into_prompt(self):
        sys_prompt = ic.messages('x', ['banana'], {})[0]['content']
        for q in ic.PROJECT_RULE_QUOTES + ['rim counts', 'original bowl']:
            self.assertNotIn(q, sys_prompt)


class TestScoring(unittest.TestCase):
    def test_schema(self):
        sch = ic.schema(['banana', 'bowl', 'table'], 3)
        self.assertEqual(tf.validate(sch, GOOD), [])
        self.assertTrue(tf.validate(sch, dict(GOOD, object='apple')))
        self.assertTrue(tf.validate(sch, dict(GOOD, success_predicate={'rim_counts': True})))

    def test_compare(self):
        _, insts = ic.instructions()
        near = next(x for x in insts if x['id'] == 'para_adversarial_near')
        c = ic.compare(dict(GOOD, relation='near', ambiguities=['near or in?']), near)
        self.assertFalse(c['vs_project']['relation'])
        self.assertTrue(c['vs_faithful']['relation'])
        self.assertTrue(c['flagged_ambiguity'] and c['ambiguity_expected'])
        self.assertTrue(all(ic.compare(GOOD, insts[0])['vs_project'].values()))


if __name__ == '__main__':
    unittest.main()
