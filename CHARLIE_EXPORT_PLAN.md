# Charlie Bravo packages — prepared, never run

Updated 26 October 2026. Three packages here were prepared and never ran in the simulator:

- `experiments/charlie_bravo_v1/`: the DROP supervisor ("Charlie Bravo, Banana on the way"): controller, vision transport, saved-frame screens and synthetic tests.
- `experiments/charlie_bravo_perception_v1/`: the RGB-D capture and perception package. Its capture gate never ran, so no native depth capture exists.
- `experiments/intent_demo_v1/`: intent checking and the intent-controlled robot demo. Its offline checker validation ran on earlier episodes; the robot demo never ran.

A fourth package, ABBA (`experiments/roger_bravo_one_shot_v1/`), ran once on 26 October; see "ABBA" in `README.md`. It reuses `charlie_bravo_v1`'s `controller.py`, `runtime.py` and `vision_images.py` byte for byte, has its own transport, prompt and runner, and its launcher is adapted from the perception package's capture supervisor. ABBA's run does not count as a run of any of the three packages above, and none of them changes the frozen P0/P1 scores.

## Decision: export them, labelled

Recommended and done: keep all three in the export, each README opening with "Prepared, never run". The alternative, leaving them out, was not taken.

Why keep them: they are already redacted and scanned (25 October), their offline suites pass (`OFFLINE_EXPORT_CHECKS.md`), and they show where ABBA's code came from, including the capture gate it skipped. Removing them would hide that lineage and leave README statements about them without anything to check against. The cost is a larger surface that a reader could mistake for results; the labels, this file and the README's Results and ABBA sections say plainly that they never ran.

## Included and omitted

Included: authored Python code, synthetic test fixtures, prompts, example permits, runner provenance, original design locks and export snapshot locks, and the intent checker's offline validation summary. Saved-frame preparation code is included for inspection; its image catalogs, prepared payloads and licensed media are not.

Excluded: credentials, account and balance records, VM identity, raw API requests and responses and ledgers from the saved-frame screens, videos and images, model weights, private reviews and handoffs, full episode traces and task-source extracts. Evidence-dependent checks need the separately hosted evidence and the pinned upstream RoboLab source.

## Integrity and adaptations

`EXPORT_PROVENANCE.json` lists copied files and redactions. Each Charlie `source-lock.original.json` preserves the original project lock; `source-lock.json` describes the shipped export snapshot, not the original lock. Local Mac paths and the VM instance ID were replaced by placeholders (`<vm-instance-id>`), the same placeholder used in ABBA's public copy. One pixel-comparison test skips when its omitted image fixture is absent. The intent demo's SHA-256 list remains a historical lock, not a claim that all its external evidence is shipped.

The intent checker validation summary's score hash refers to the original score artifact; the public score copy may have its path metadata redacted. No numerical score is changed.

These scripts are research tooling, not a turnkey deployment. Configure paths, evidence, SDKs and run permissions before using any launcher; offline tests are not launch approval.
