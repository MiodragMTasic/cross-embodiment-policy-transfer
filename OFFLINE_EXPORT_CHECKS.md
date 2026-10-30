# Code export verification — 25 October 2026, ABBA rows added 26 October, integrity rows 27 October

These checks ran locally with no simulator, GPU, VM, model API or paid service. They establish only the scope below; they are not a live deployment or release-success result.

| Exported suite | Result | Limit |
| --- | --- | --- |
| Charlie perception `unittest discover` | 55 passed | Synthetic/local checks only |
| Charlie controller `unittest discover` | 37 passed, 1 skipped | Pixel comparison requires omitted licensed saved-image fixture |
| Nemotron `test_tf_client` + `test_analyse_run.TestMetrics` | 24 passed, 1 skipped | Optional OpenAI-compatible SDK absent; full evidence-dependent suites not run |
| Intent demo `tests/test_checker.py` | 3 passed, 13 skipped | Original episode evidence absent; copied validation summary checked |
| ABBA `roger_bravo_one_shot_v1` `unittest discover` (26 Oct) | 43 passed | Standard library only; fakes for Isaac, the policy server and Token Factory |
| ABBA v5.1 re-score of `evidence/abba_l4003/gate_trace.jsonl` (26 Oct) | PASS at control 1071, identical to `SCORE_V51.json` | Re-scores the shipped trace; does not re-run the episode |
| ABBA `evidence/abba_l4003/MANIFEST.sha256` (26 Oct) | 68 OK | Shipped evidence only; the VM copy manifests cover 64 of these files and all match |
| `tools/verify_locks.py` (27 Oct) | 855 lock entries in 36 lock files: 553 OK, 5 SANITIZED, 297 NOT_SHIPPED, 0 FAIL; exit 0 | SANITIZED = documented in `SANITIZATION.json` with the original sha256 equal to the lock entry. A copy with one locked file edited, one deleted, or one sanitized file edited exits 1 |
| `shasum -a 256 -c PUBLIC_MANIFEST.sha256` (27 Oct) | every shipped file OK | Describes this sanitized copy, not the experiment's bytes |

On 27 October every row above was re-run from a clean copy of this folder (no `.git`, no caches) with the same counts. The intent-demo row used pytest 9.1.1; the Charlie rows used Python 3.12 with numpy and Pillow; the others used the standard library, and numpy for the re-score.

All exported Python files parse. Export snapshot locks verify with zero mismatches: 13 controller entries and 28 perception entries. Original Charlie locks remain beside the export locks for provenance; against them the redacted perception `preflight.py` and the adjusted `tests/test_vision_images.py` report SANITIZED. Frozen experiment scores and source files were not edited; 21 files are sanitized public copies, listed in `SANITIZATION.md`. ABBA's `source-lock.json` is unchanged: 10 of its 11 entries are OK and `preflight.py` is SANITIZED, differing only by the VM-ID placeholder.

Example commands from repository root, using Python with numpy/Pillow and pytest where needed:

```sh
python -m unittest discover -s experiments/charlie_bravo_perception_v1 -p 'test_*.py' -v
PYTHONPATH=experiments/charlie_bravo_v1 python -m unittest discover -s experiments/charlie_bravo_v1/tests -p 'test_*.py' -v
PYTHONPATH=nemotron python -m unittest test_tf_client test_analyse_run.TestMetrics -v
python -m pytest experiments/intent_demo_v1/tests/test_checker.py -q -p no:cacheprovider
(cd experiments/roger_bravo_one_shot_v1 && python3 -m unittest discover -s tests -p 'test_*.py')
PYTHONPATH=experiments/bowl_step5 python3 experiments/scoring_v5/release_gate_v5_1.py evidence/abba_l4003/gate_trace.jsonl
(cd evidence/abba_l4003 && shasum -a 256 -c MANIFEST.sha256)
python3 tools/verify_locks.py
shasum -a 256 -c PUBLIC_MANIFEST.sha256
```

Reproducing the full historical evidence requires the external evidence bundle and upstream dependencies described in README. A clean public install and all launchers have not been executed from this export. Saved-image screening utilities also need image catalogs omitted from the code-only package.
