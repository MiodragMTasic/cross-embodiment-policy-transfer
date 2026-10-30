"""Unit tests for convert_step7.py's input guards and cut mapping. Pure python + numpy; run anywhere: python test_convert_step7.py"""
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert_step7 as c

EVIDENCE = HERE.parents[1] / 'assets/report/evidence/2026-10-21/bowl-training-feasibility-v4'


def ledger(*cases, seed=0, last='finished'):
    rows = []
    for layout in cases:
        case = f'step7_collect_layout{layout}_s{seed}_v1'
        rows.append({'case': case, 'stage': 'started', 'layout_seed': layout, 'policy_seed': seed})
        rows.append({'case': case, 'stage': last, 'infrastructure_ok': True} if last == 'finished' else {'case': case, 'stage': last})
    return rows + [{'stage': 'batch_finished'}]


class Names(unittest.TestCase):
    def test_step8_smoke_never_enters(self):
        for name in ('step8_A_smoke_l2000_s0_v1', 'step8_A_P0_l4000_s0_v1', 'step8_collect_layout2004_s0_v1'):
            with self.assertRaisesRegex(AssertionError, 'step 8'):
                c.check_name(name)

    def test_layout_range_and_shape(self):
        self.assertEqual(c.check_name('step7_collect_layout2031_s1_v1'), (2031, 1))
        self.assertEqual(c.check_name('step5_training_layout2002_v4'), (2002, 0))
        for name in ('step7_collect_layout2032_s0_v1', 'step7_collect_layout3000_s0_v1', 'step7_collect_layout4000_s0_v1',
                     'step7_collect_layout1999_s0_v1', 'step5_training_layout2004_v4', 'step5_controlled_layout1000_v3',
                     'step7_collect_layout2004_s0_v2'):
            with self.assertRaises(AssertionError):
                c.check_name(name)


class CaseSet(unittest.TestCase):
    def test_explicit_set(self):
        rows = ledger(2004, 2005) + [{'case': 'step7_collect_layout2006_s0_v1', 'stage': 'started', 'layout_seed': 2006, 'policy_seed': 0}]
        rows.insert(-1, {'case': 'step7_collect_layout2007_s0_v1', 'stage': 'started', 'layout_seed': 2007, 'policy_seed': 0})
        rows.insert(-1, {'case': 'step7_collect_layout2007_s0_v1', 'stage': 'halted_for_review', 'error': 'x'})
        included, excluded = c.explicit_cases(rows)
        self.assertEqual(sorted(included), sorted(c.STEP5_REUSED + ('step7_collect_layout2004_s0_v1', 'step7_collect_layout2005_s0_v1')))
        self.assertEqual(set(excluded), {'step7_collect_layout2006_s0_v1', 'step7_collect_layout2007_s0_v1'})
        self.assertIn("'started'", excluded['step7_collect_layout2006_s0_v1'])

    def test_first_ledger_guards(self):
        for bad in (ledger(2003),                # 2000-2003 are reused, never recollected in the first pass
                    ledger(2004, seed=1),        # seed 1 belongs in the D3 ledger
                    ledger(2004) + ledger(2004), # repeated case
                    [{'case': 'step8_A_smoke_l2000_s0_v1', 'stage': 'started'}]):
            with self.assertRaises(AssertionError):
                c.explicit_cases(bad)
        mismatch = ledger(2004)
        mismatch[0]['layout_seed'] = 2005
        with self.assertRaises(AssertionError):
            c.explicit_cases(mismatch)

    def test_seed1_ledger(self):
        included, _ = c.explicit_cases(ledger(2004), ledger(2000, 2031, seed=1))
        self.assertEqual(included['step7_collect_layout2000_s1_v1']['policy_seed'], 1)
        with self.assertRaises(AssertionError):
            c.explicit_cases(ledger(2004), ledger(2005, seed=0))

    def test_unfinished_ledger_blocks_export(self):
        """C-b: export needs batch_finished with the right mode; candidates reads snapshots."""
        finished = ledger(2004)
        finished[-1]['mode'] = 'first'
        c.require_finished(finished, 'first', 'first')
        for bad in (finished[:-1], ledger(2004), [*finished[:-1], {'stage': 'batch_finished', 'mode': 'seed1'}], []):
            with self.assertRaisesRegex(AssertionError, 'batch_finished'):
                c.require_finished(bad, 'first', 'first')
        args = type('A', (), {})()
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            args.ledger, args.seed1_ledger = str(Path(d) / 'l.jsonl'), None
            Path(args.ledger).write_text(''.join(json.dumps(r) + '\n' for r in finished[:-1]))
            included, _ = c.read_ledgers(args)                # candidates: snapshot is fine
            self.assertIn('step7_collect_layout2004_s0_v1', included)
            with self.assertRaisesRegex(AssertionError, 'batch_finished'):
                c.read_ledgers(args, finished=True)           # export: refused

    def test_trace_rows_mismatch(self):
        """C-c: capture-integrity trace_rows must equal the metric trace length."""
        import tempfile
        case = 'step7_collect_layout2004_s0_v1'
        info = {'layout_seed': 2004, 'policy_seed': 0}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = c.case_paths(root, case)
            for k in ('hashes', 'integrity', 'metric'):
                p[k].parent.mkdir(parents=True, exist_ok=True)
            p['hashes'].write_text(f"{c.RUNNER[1]}  /home/x/robot-transfer/bowl/{c.RUNNER[0]}\n")
            ident = {**info, 'task_class': [c.TASK]}
            p['metric'].write_text(json.dumps({**ident, 'schema': 'command-linked-supported-release-v4', 'trace': [{}] * 750}))
            p['integrity'].write_text(json.dumps({**ident, 'trace_rows': 750}))
            c.identify(root, case, info)
            p['integrity'].write_text(json.dumps({**ident, 'trace_rows': 749}))
            with self.assertRaisesRegex(AssertionError, 'trace_rows'):
                c.identify(root, case, info)

    def test_runner_hash(self):
        text = (EVIDENCE / 'step5_training_layout2002_v4/source-hashes.txt').read_text()
        self.assertEqual(c.runner_hash(text), c.RUNNER[1])
        with self.assertRaises(AssertionError):
            c.runner_hash(text.replace('run_controlled_episode_v4_registered.py', 'other.py'))


class Approved(unittest.TestCase):
    cases = {'step5_training_layout2002_v4': {}, 'step7_collect_layout2004_s0_v1': {}}

    def approved(self, *episodes):
        return {'schema': c.APPROVED_SCHEMA, 'scorer_sha256': c.SCORER['scoring_v5/release_gate_v5_1.py'], 'episodes': list(episodes)}

    def test_ok_and_rejected(self):
        ok, rejected, pending = c.load_approved(self.approved(
            {'case': 'step5_training_layout2002_v4', 'win_control': 301, 'verdict': 'OK', 'reviewer': 'r'},
            {'case': 'step7_collect_layout2004_s0_v1', 'win_control': 400, 'verdict': 'REJECT', 'reviewer': 'r'}), self.cases)
        self.assertEqual(list(ok), ['step5_training_layout2002_v4'])
        self.assertEqual(list(rejected), ['step7_collect_layout2004_s0_v1'])
        self.assertEqual(pending, {})

    def test_pending_pass_blocks_export(self):
        """C-a: a PENDING pass has no verdict."""
        ok, rejected, pending = c.load_approved(self.approved(
            {'case': 'step5_training_layout2002_v4', 'win_control': 301, 'verdict': 'OK', 'reviewer': 'r'},
            {'case': 'step7_collect_layout2004_s0_v1', 'win_control': 400, 'verdict': 'PENDING', 'reviewer': 'r'}), self.cases)
        self.assertEqual(list(pending), ['step7_collect_layout2004_s0_v1'])
        self.assertEqual(c.verdict_gaps(set(self.cases), ok, rejected), ['step7_collect_layout2004_s0_v1'])

    def test_unlisted_pass_blocks_export(self):
        """C-a: a pass missing from the approvals file has no verdict; a REJECT is a verdict."""
        ok, rejected, _ = c.load_approved(self.approved(
            {'case': 'step5_training_layout2002_v4', 'win_control': 301, 'verdict': 'OK', 'reviewer': 'r'}), self.cases)
        self.assertEqual(c.verdict_gaps(set(self.cases), ok, rejected), ['step7_collect_layout2004_s0_v1'])
        ok, rejected, _ = c.load_approved(self.approved(
            {'case': 'step5_training_layout2002_v4', 'win_control': 301, 'verdict': 'OK', 'reviewer': 'r'},
            {'case': 'step7_collect_layout2004_s0_v1', 'win_control': 400, 'verdict': 'REJECT', 'reviewer': 'r'}), self.cases)
        self.assertEqual(c.verdict_gaps(set(self.cases), ok, rejected), [])

    def test_errors(self):
        good = {'case': 'step5_training_layout2002_v4', 'win_control': 301, 'verdict': 'OK', 'reviewer': 'r'}
        for bad in ({**good, 'case': 'step8_A_smoke_l2000_s0_v1'},       # step 8 name
                    {**good, 'case': 'step7_collect_layout2005_s0_v1'},  # not in the explicit case set
                    {**good, 'verdict': 'MAYBE'}, {**good, 'win_control': 0}, {**good, 'reviewer': ''}):
            with self.assertRaises(AssertionError):
                c.load_approved(self.approved(bad), self.cases)
        with self.assertRaises(AssertionError):
            c.load_approved(self.approved(good, good), self.cases)
        with self.assertRaises(AssertionError):
            c.load_approved({**self.approved(good), 'scorer_sha256': 'x'}, self.cases)


class Cut(unittest.TestCase):
    def test_win_row(self):
        self.assertEqual(c.win_row(range(1, 751), 308), 307)
        self.assertEqual(c.win_row(range(1, 302), 301), 300)
        for steps, win in ((range(0, 750), 308), ([1, 2, 4], 2), (range(1, 301), 301)):
            with self.assertRaises((AssertionError, ValueError)):
                c.win_row(steps, win)

    def test_shift_errors(self):
        import numpy as np
        a = np.sin(np.arange(50) / 3)
        e = c.shift_errors(a, a.copy())
        self.assertEqual(e[0], 0.0)
        self.assertTrue(all(e[s] > 0.1 for s in (-2, -1, 1, 2)))

    def test_real_traces_locked_scorer(self):
        """A 2001 (v5-only win, live v4 ran on to 750) and A 2002 (v4 stopped on its win): the cut rows."""
        score, _ = c.import_scorer(HERE.parent)
        for layout, win, rows in ((2001, 308, 750), (2002, 301, 301)):
            metric = EVIDENCE / f'step5_training_layout{layout}_v4/{c.TASK}/strict-metric.json'
            s = score(metric)
            self.assertTrue(s['pass'])
            self.assertEqual((s['success_control'], s['rows']), (win, rows))
            trace = json.loads(metric.read_text())['trace']
            row = c.win_row([r['step'] for r in trace], s['success_control'])
            self.assertEqual(row, win - 1)
            self.assertEqual(trace[row]['step'], win)


if __name__ == '__main__':
    unittest.main(verbosity=1)
