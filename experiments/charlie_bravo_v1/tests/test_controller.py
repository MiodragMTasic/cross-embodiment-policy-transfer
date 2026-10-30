import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller import Config, Decision, Frame, State, Supervisor
from runtime import Runtime

POSE = (0.,) * 7
TARGET = (.1,) * 7
READY = json.dumps(dict(verdict='READY_TO_RELEASE', held=True, central=True, clearance=True,
                        views_agree=True, motion='still', reason='Held centrally with clearance in both views.'))
WAIT = READY.replace('READY_TO_RELEASE', 'WAIT')


def frames(tick):
    return [Frame(tick-2,b'ext0',b'wrist0'), Frame(tick-1, b'ext1', b'wrist1'), Frame(tick, b'ext2', b'wrist2')]


class ControllerTests(unittest.TestCase):
    def start(self, enabled=True, config=None):
        c = Supervisor(config or Config(hold_ticks=6), enabled)
        c.advance(1, 0., POSE, TARGET, 1.)
        self.assertIsNotNone(c.request(frames(1), 1., 0.))
        return c

    def test_two_checks_then_latch_and_bounded_stop(self):
        c = self.start(); c.receive(READY, 1., 1, POSE)
        self.assertEqual(c.current_override(), (POSE, 1.))
        for tick in range(2, 7):
            cmd = c.advance(tick, 1., POSE, TARGET, 0.)
            self.assertEqual(cmd.arm, POSE)
            self.assertEqual(cmd.grip, 1.)
            self.assertIsNone(c.request(frames(tick), 0., 1.))
        c.advance(7, 1., POSE, TARGET, 1.)
        self.assertIsNotNone(c.request(frames(7), 1., 1.))
        c.receive(READY, 2., 7, POSE)
        self.assertEqual(c.current_override(), (POSE, 0.))
        cmd = c.advance(8, 2., POSE, TARGET, 1.)
        self.assertEqual(cmd.grip, 0.)  # cached close cannot override
        self.assertEqual(cmd.arm, POSE)
        self.assertIsNone(c.request(frames(8), 1., 2.))
        self.assertFalse(c.advance(97, 3., POSE, TARGET, 1.).stop)
        self.assertIsNone(c.current_override())
        self.assertTrue(c.used)

    def test_baseline_records_without_override(self):
        c = self.start(False); c.receive(READY, 1., 1, POSE)
        self.assertIsNone(c.current_override())
        self.assertEqual(c.state, State.POLICY)

    def test_bad_replies_never_open(self):
        variants = ['', '{}', '```json\n'+READY+'\n```', READY.replace('true', '"true"', 1),
                    READY.replace('"central": true', '"central": false'),
                    READY.replace('"held": true', '"held": false'),
                    READY.replace('"views_agree": true', '"views_agree": false'),
                    READY.replace('READY_TO_RELEASE', 'UNCERTAIN'), WAIT]
        for raw in variants:
            with self.subTest(raw=raw):
                c = self.start(); c.receive(raw, 1., 1, POSE)
                self.assertIsNone(c.current_override())

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            Decision.parse(READY[:-1] + ', "held": false}')

    def test_late_or_wrong_tick(self):
        for now, tick in [(10., 1), (1., 2), (-1., 1), (float('nan'), 1)]:
            c = self.start(); c.receive(READY, now, tick, POSE)
            self.assertIsNone(c.current_override())

    def test_no_physics_advance_while_request_pending(self):
        c = self.start()
        with self.assertRaises(RuntimeError):
            c.advance(2, 1., POSE, TARGET, 1.)

    def test_hold_failure_resumes_policy(self):
        c = self.start(); c.receive(READY, 1., 1, POSE)
        c.advance(10, 2., POSE, TARGET, 1.)
        c.request(frames(10), 1., 2.); c.receive(WAIT, 3., 10, POSE)
        self.assertIsNone(c.current_override())

    def test_hold_expiry(self):
        c = self.start(); c.receive(READY, 1., 1, POSE)
        self.assertEqual(c.advance(17, 2., POSE, TARGET, 1.).arm, TARGET)

    def test_budget_failure_releases_hold_without_forcing_open(self):
        c = self.start(config=Config(max_calls=1)); c.receive(READY, 1., 1, POSE)
        c.advance(10, 2., POSE, TARGET, 1.)
        self.assertIsNone(c.request(frames(10), 1., 2.))
        self.assertEqual(c.state, State.POLICY)
        self.assertEqual(c.calls, 1)

    def test_dollar_reservation_not_refunded_for_errors(self):
        c = self.start(config=Config(api_cap=.002))
        c.receive('', 1., 1, POSE)
        c.advance(13, 2., POSE, TARGET, 1.)
        self.assertIsNone(c.request(frames(13), 1., 2.))
        self.assertEqual(c.reserved, .002)

    def test_stale_painted_or_missing_images_rejected(self):
        for fs in [frames(0), [Frame(0,b'a',b'b'), Frame(1,b'a',b'b','painted')],
                   [Frame(0,b'a',b'b'), Frame(1,b'a',b'')], frames(1)[::-1], frames(1)[:1]]:
            c = Supervisor(); c.advance(1, 0., POSE, TARGET, 1.)
            self.assertIsNone(c.request(fs, 1., 0.))

    def test_wall_deadline(self):
        c = Supervisor(Config(max_wall_s=2))
        c.advance(0, 0., POSE, TARGET, 1.)
        self.assertTrue(c.advance(1, 2., POSE, TARGET, 1.).stop)

    def test_runtime_records_original_and_executed_and_keeps_images(self):
        with tempfile.TemporaryDirectory() as temp:
            rt = Runtime(Path(temp)/'run', lambda *a: READY, clock=lambda: 0.)
            for tick in range(16):
                result = rt.step(tick, POSE, TARGET, 1., lambda: (b'actual-exterior', b'actual-wrist'))
            self.assertEqual(result.grip, 0.)
            self.assertEqual(result.arm, TARGET)
            records = [json.loads(x) for x in (rt.output/'events.jsonl').read_text().splitlines()]
            self.assertEqual(records[-1]['original_grip'], 1.)
            self.assertEqual(records[-1]['executed']['grip'], 0.)
            self.assertEqual(len(list(rt.output.glob('query_*/request.json'))), 2)

    def test_runtime_transport_failure_and_resume(self):
        def broken(*a): raise TimeoutError('SECRET must not enter log')
        with tempfile.TemporaryDirectory() as temp:
            rt = Runtime(Path(temp)/'run', broken, clock=lambda: 0.)
            for tick in range(7):
                result = rt.step(tick, POSE, TARGET, 1., lambda: (b'e', b'w'))
            self.assertEqual(result.arm, TARGET)
            self.assertNotIn('SECRET', (rt.output/'query_0001/response.json').read_text())

    def test_runtime_deadline_includes_model_wait(self):
        clock = [0.]
        def delayed(*a): clock[0] = 3.; return READY
        with tempfile.TemporaryDirectory() as temp:
            rt = Runtime(Path(temp)/'run', delayed, config=Config(max_wall_s=2), clock=lambda: clock[0])
            for tick in range(7):
                result = rt.step(tick, POSE, TARGET, 1., lambda: (b'e', b'w'))
            self.assertTrue(result.stop)
            self.assertNotEqual(rt.controller.state, State.OPEN)


if __name__ == '__main__': unittest.main()
