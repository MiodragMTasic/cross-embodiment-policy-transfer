# Cross-Embodiment Policy Transfer

A Franka-trained GR00T policy drives a Kinova arm in Isaac Sim through a frozen, hand-written adapter. This repository holds the code, locks and small result files for that simulation-only study, built for the Nebius x NVIDIA Global AI Hackathon (Physical AI track).

This is a solo entry. Before 21 October I set up the Nebius VM and a first Isaac Sim and Isaac Lab install by hand. From about 21 October I directed AI coding agents, Claude (Anthropic) and Codex (OpenAI), which installed the stack described here and wrote and ran the code in this repository, while I made the calls on the task, the win rule, the test size, the order of steps, the freeze and the spending. AI agents, not people, reviewed each step and adjudicated outcomes (see Credits).

## The question

When further training improves a robot policy on the robot it was trained for, does the improvement show up on a second robot that runs the same policy through an adapter that never changes?

Robot A is a Franka Panda with a Robotiq 2F-85. Robot B is a Kinova Gen3 (7 joints) with a Robotiq 2F-85. The policy, P0, is NVIDIA's GR00T N1.7 DROID checkpoint. It sees an exterior camera, a wrist camera and the arm's state, and is told "Pick up the banana and place it in the bowl." P0 is measured on both robots, fine-tuned on Franka data only to get P1, and P1 is measured on both robots with the adapter unchanged. The pre-registered tests are within each robot: ΔA = A1 − A0 and ΔB = B1 − B0. A versus B is not the claim.

## The adapter

Hand-written, no trained weights, frozen on 23 October 2026 at 18:50 UTC as 19 hash-locked files (`frozen_B_adapter_v1r2/`). The frozen configuration is `experiments/bowl_step6/run_painted_dev.py --arm painted --variant lift`.

- **Arm mapping.** Local IK on a virtual Franka turns the Kinova's measured gripping-frame pose into the Franka joint state P0 expects. P0's Franka joint targets go through forward kinematics to a world pose, and IK on the Kinova gives its seven joint targets. Commands are rate-limited to 0.04 rad per control; a run aborts if a joint leaves its soft limits by more than 0.03 rad.
- **Timing.** Each P0 action is held for 12 controls at 15 Hz so the Kinova's gripper can close (95% closure takes 2 control steps on A and 23 on B in these simulator assets).
- **Lift.** After P0 has held a close command for more than 24 controls, the Kinova's tool target rises 15 mm, ramped over 24 controls, and comes back down when P0 opens. It applies only to a close that began with the banana away from the bowl, and that eligibility check reads privileged simulator state.
- **Exterior cross-painting.** Before each P0 request the simulator hides the Kinova's visuals, poses a rendering copy of the Franka at the virtual-Franka joints, renders the exterior camera and restores the scene. The copy is parked 20 m above the cell whenever physics advances, and its colliders stay on: it is kept out of the physics by placement, not made visual-only. The locked source's docstring (`experiments/bowl_step6/paint_A.py`) says "visual-only"; the description above is the accurate one. The idea is from Mirage (Chen et al., RSS 2024); this version renders from simulator ground truth instead of inpainting. The wrist camera is never painted.

## Honest limits

- **Simulation only.** There was no hardware access; nothing here has run on a physical robot.
- **Simulator ground truth.** Painting renders from simulator state, so P0 also sees what the Kinova's body would hide; a real camera would need mask-and-inpaint. The lift's eligibility reads banana–bowl contact and RoboLab's `object_in_container` predicate from the previous control. The Kinova's wrist camera is remounted to match the Franka's relative camera pose. Episode termination (the live v4 gate) reads simulator state on both robots; it ends episodes and does not steer them.
- **Twelve test layouts.** Wilson intervals are wide. Equal counts mean matched, not proven equal. The declared test is an exact two-sided McNemar per robot on the 12 paired layouts. It needs a lopsided split of the discordant layouts for p < 0.05: at least 6–0 (p = 0.031); 8–1 also qualifies (p = 0.039). P0 won 7/12 on both robots by the locked count, so P1 can gain at most 5 layouts on either, and even 5–0 gives p = 0.0625. No improvement can reach p < 0.05 in this design; a decline can, but only at 6–0 or 7–0 (p = 0.031 or 0.016). Crashed after-phase episodes are unscored and drop their pair, which lowers the ceiling further. A p-value above 0.05 here is not evidence of no effect. No one-sided or pooled test was declared; if one is shown, it is labelled exploratory and sits beside the declared test.
- **Contact sensing differs.** A's contact flag is one finger; B's is either of two pads. Cross-robot stage rates lean toward B. Within each robot, P0 and P1 go through the same measurement pipeline, but that does not make the measurement error equal: if P1 behaves differently, for example with more set-downs or different rests, the scorer's known blind spots can bias the measured change.
- **AI review.** Outcomes were adjudicated by AI agents. For the before half and checkpoint selection, an independent Claude agent judged from videos and traces; it did not see the scorer's numbers or reasoning, but knew which episodes the scorer had passed or flagged. For the after half, Codex reviewers judged from sampled video frames and event traces, not blinded and not continuous playback. No person adjudicated scored outcomes. Reviewer counts are reported beside the locked scorer, never pooled with it.
- **Safety aborts.** Four of the Kinova's twelve "before" test runs ended in a safety abort, all scored on their partial traces: three when local IK pinned wrist joint 6 at −2.09 rad after carrying the banana over the bowl, and one soft-limit abort long after a win. The twelve P1 runs had three IK safety aborts (4000, 4010 and 4011), all scored as failures.
- **Analysis timing.** The v5.1 scorer and the ΔA/ΔB tests were locked before any 4000–4011 test episode ran. One exploratory reading, the pose variant, was declared twelve minutes after the before phase started, before any outcome was inspected.
- **Tuning.** The lift was tuned on one development scene and tested once before the freeze. One fine-tuning run, one policy seed: a case study of this run, not of a method. Why P1 did worse is not established; the training-data gap under Results is a hypothesis, and no repaired training was run.

## Results

A win is a commanded release followed by a set-down or drop that settles on or in the original bowl; the rim counts. An uncommanded slip from a closed gripper is not a success. The locked scorer is v5.1 "first rest" (`experiments/scoring_v5/release_gate_v5_1.py`).

**Before training (P0, test layouts 4000–4011, policy seed 0)**

| Condition                           | v5.1 wins | Wilson 95% | AI reviewer | Live v4 gate | Safety aborts |
| ----------------------------------- | --------- | ---------- | ----------- | ------------ | ------------- |
| A/P0, Franka                        | 7/12      | 0.32–0.81  | 8/12        | 4/12         | —             |
| B/P0, Kinova via the frozen adapter | 7/12      | 0.32–0.81  | 7/12        | 1/12         | 4             |

Source: `experiments/scoring_v5/score_step8_before/score_step8-before.{json,md}`.

**Checkpoint selection.** Four checkpoints from one fine-tuning run were tried on the Franka alone, on 8 validation layouts (3000–3007), and my independent reviewer, a separate Claude session, adjudicated every v5.1 pass and every flagged fail. That adjudication is binding for the choice.

| Robot A, 8 validation layouts | P0  | c500 | c1000 | c1500 | c2000 |
| ----------------------------- | --- | ---- | ----- | ----- | ----- |
| Raw v5.1 wins                 | 4   | 0    | 1     | 3     | 3     |
| Adjudicated wins              | 6   | 0    | 1     | 3     | 3     |

No checkpoint beat P0. P1 is checkpoint 1500, the earlier of the two tied best, labelled **"P1, no A-validation improvement"**. Its validation count is selection-biased (best of four), and the candidate-versus-P0 p-values are descriptive only. Sources: `experiments/scoring_v5/score_select_v1_raw/` (raw) and `score_select_v1_adj/` (with the reviewer's verdicts from `select_v1_adjudication.json`; its `select-step7.json` records the binding choice).

**After training (P1 = checkpoint 1500; the after test launched on 24 October at 20:52 UTC and completed; all 24 P1 episodes scored, none unscored)**

| Condition                         | v5.1 wins | Wilson 95% | AI reviewer | Live v4 gate | Safety aborts |
| --------------------------------- | --------- | ---------- | ----------- | ------------ | ------------- |
| A/P1, Franka                      | 2/12      | 0.05–0.45  | 3/12        | 1/12         | —             |
| B/P1, Kinova via the same adapter | 0/12      | 0.00–0.24  | 1/12        | 0/12         | 3             |

| Pre-registered test | Δ (wins) | Discordant layouts (P1 only / P0 only) | Exact two-sided McNemar p (declared, per robot) |
| ------------------- | -------- | -------------------------------------- | ----------------------------------------------- |
| ΔA = A1 − A0        | −5       | 0 / 5                                  | 0.0625                                          |
| ΔB = B1 − B0        | −7       | 0 / 7                                  | 0.015625                                        |

Source: `experiments/scoring_v5/score_step8_after/score_step8-after.{json,md}`. The after-half "AI reviewer" counts come from Codex reviewers working from sampled video frames and event traces, not blinded; they found two physical placements the locked scorer missed (A 4007, rest with angular-speed noise; B 4001, a commanded reclose that never regrasped) and change neither the headline nor the paired tests. A rerun of A/P0 on 4000–4001 at the start of the after half gave 0/2, matching the before outcomes, but with different trajectories, so it does not rule out drift.

Pre-registered reading: ΔA and ΔB both positive is consistent with the improvement transferring under these conditions; ΔA positive with ΔB near zero or negative means it did not clearly transfer; no clear improvement on A means this run cannot assess transfer. Tiny samples or wide uncertainty: preliminary evidence, not a reliable general conclusion. A non-significant p is not evidence of no effect, and an improvement cannot reach p < 0.05 in this design at all. The locked v5.1 counts are the headline; the reviewer's counts are reported beside them and never replace them.

The selected checkpoint scored worse on both robots, so there was no improvement on A whose transfer could be assessed. These small, sequential before/after samples do not isolate fine-tuning as the only cause.

**Checks after the test.** None of these changes a locked score.

- **Grading.** A CPU replay of the locked scorer over all 48 saved before and after episodes reproduced every verdict and winning control. The scorer's source hashes and every trace hash matched the score records, so the automated grading was the same before and after. This covers the automated scores only, not the separate AI reviews.
- **Bowl support (exploratory, defined after the test).** Applied the same way to all four groups, it asks whether the banana rested on the bowl, rim included, with no table or fingertip contact, for at least one simulated second. Both B/P0 and B/P1 reached it in 8 of 12 runs; A fell from 12/12 to 8/12. It does not require a commanded release, so it is not the win rule, and it does not say where the banana ended up.
- **Observed failures.** On the Franka's first carry, by a trace proxy (contact, clearance and closure, not measured grip force), P1 lost a lifted banana under a closed command in 6 of 12 test runs against 3 of 12 for P0.
- **Training data (a hypothesis, not a proven cause).** Demonstrations were cut at the win, the loader only starts a sample where 39 more frames follow, and the client executes the first 8 of each 40 predicted actions. In 25 of the 27 demonstrations the successful opening never fell among those first eight targets, although it is present later in the 40-action targets. This is how samples were drawn, not missing video. A sampling repair was enumerated on CPU but neither trained nor shown to work. Source and configuration checks found no demonstrated action-column permutation or one-row shift between observations and actions; they did not rule out every pipeline issue. No further retraining was run.

A follow-up put a Nemotron vision supervisor beside P1 to decide when the Kinova lets go. It ran once, as ABBA (next section): a separate assisted system, not repaired training and not part of the frozen-adapter experiment. The two packages it grew from are included as prepared code that never ran in the simulator: the DROP supervisor ("Charlie Bravo, Banana on the way", `experiments/charlie_bravo_v1/`) and the RGB-D capture and perception package (`experiments/charlie_bravo_perception_v1/`). See `CHARLIE_EXPORT_PLAN.md` and `OFFLINE_EXPORT_CHECKS.md`.

## ABBA — Alpha Bravo Banana Away (one assisted episode)

On 26 October, Toronto time (23:50–00:04 UTC into 27 October), one Kinova episode ran with an added release supervisor: P1 (checkpoint 1500) through the same frozen painted adapter with its lift, on test layout 4003 at policy seed 0, for at most 3,600 controls (240 simulated seconds). The code is `experiments/roger_bravo_one_shot_v1/`, prepared and run by Codex with my approval; its case id is `roger_B_P1_l4003_assisted_v1`, its release banner "Roger Bravo, Banana away", and its video carries the burned-in label "CHARLIE P1 ASSISTED".

While P1 commanded a closed gripper, every 12 controls the runner sent NVIDIA Nemotron 3.5 Super VL (`nvidia/Nemotron-3.5-Super-VL` on Nebius Token Factory) the actual, unpainted exterior and wrist RGB images from three moments six controls apart, with `vision_prompt.txt`. Physics paused during each call. Nemotron got no depth, object coordinates or simulator state; P1 kept receiving its usual painted observation. A READY reply froze the arm at its last executed target. A second call, on three new frames taken after the freeze, three controls apart, had to return READY with the banana still before the gripper opened. Once open, the gripper stayed open; the arm held for 90 controls, then P1's arm motion resumed.

**What happened.** The 29 calls ran at zero-based ticks: 27 WAIT replies at ticks 672–984, READY at tick 996 (hold), and a confirming READY at tick 1005, which opened the gripper. The locked scorer numbers controls from 1, so the scored open is control 1006. The unchanged v5.1 scorer gives a **pass at control 1071** (71.4 s): the qualifying open at control 1006 came with the banana's root 0.0498 m from the bowl's in the horizontal plane and its centre 0.1053 m above the bowl top; it detached near the bowl, landed on it, and was still on the bowl at the end (`pass_at_end` and `end_on_bowl` true). Those measurements come from recorded simulator state, used only for scoring after the episode; they were never inputs to Nemotron. The live v4 gate, which only decides when an episode stops, reported no success: it marked that open as an ineligible open command. v5.1 is the primary rule for every locked result here, so the scorer's PASS is the reported verdict, with the v4 reading beside it.

The earlier unassisted B/P1 run on layout 4003 reached bowl support through a slip and failed the controlled-release criterion. In the separate ABBA run, Nemotron triggered a commanded release that passed the unchanged scorer. This demonstrates the assisted-release mechanism once; it does not establish a repeatable improvement or isolate Nemotron's contribution from the controller's hold and latched-open behavior.

**Limits.**

- One exploratory assisted demonstration on one selected layout and seed. Layout 4003 was picked because it was the only B/P1 test layout with a long window in which the banana was held centred over the bowl. It is not a success rate, not a controlled or paired causal comparison, and not a training repair.
- The original results are unchanged: A 7/12 → 2/12 and B 7/12 → 0/12. ABBA is not added to them. In that after test, B/P1 on 4003 (`step8_B_P1_l4003_s0_v1`) ended with the banana on the bowl after it slipped from the closed gripper at about 123 s, and failed v5.1 on `lost_contact_before_open`.
- After the forced open, P1's own gripper command also turned to open from control 1096.
- Nemotron's "central" judgement is not calibrated localization; the measured offset at the open was about 5 cm.
- The package had local CPU tests and a Codex self-review before the run, but no independent review and no simulator smoke test. A Codex check of sampled video frames from 66–75 s shows the open gripper and the banana coming to rest supported by the bowl; that is not a blinded or full-video review.
- The packaged `verify_copy.py` command line still requires the depth captures and `CAPTURE_COMPLETE` marker of the capture package it was adapted from, so it reports errors on this run. The copy check used its unchanged hash helpers directly with this run's own requirements; `evidence/abba_l4003/COPY_VERIFICATION.json` records the result (298 files, no errors). The tool was left unfixed so the source lock still verifies.
- The RGB-D capture package and the intent-controlled robot demo (`experiments/intent_demo_v1/`) never ran.

**Cost.** Nemotron: 29 calls, 68,040 input and 2,105 output tokens, **$0.0223065** at $0.30 per million input and $0.90 per million output tokens (Token Factory prices checked on 26 October); 56.5 s of total API wait. This is computed from the returned token counts; the billing deduction was not checked. The GPU VM ran for about 34 minutes, roughly $1.02 at $1.80 an hour, an estimate that leaves out disk and is not an invoice.

### Run the ABBA tests

The 43 tests use only the Python standard library and run on macOS or Linux. They exercise the supervisor, transport, permit checks and launcher with fakes; they do not start Isaac Sim or call Nemotron.

```bash
cd experiments/roger_bravo_one_shot_v1
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py'   # Ran 43 tests ... OK
```

Check the package against its source lock (`source-lock.json`, sha256 `166a1bb9…a1a91b83`, the hash the run's launch permit pinned). From the repository root:

```bash
python3 tools/verify_locks.py | grep -A1 roger_bravo_one_shot_v1/source-lock.json   # OK 10  SANITIZED 1  FAIL 0
```

Ten entries are OK. The eleventh, `preflight.py`, is SANITIZED: this public copy replaces the cloud VM's instance ID on its line 5 with `<vm-instance-id>`, and the lock pins the original file, whose sha256 (`818896a1…4e25a2`) `SANITIZATION.json` records beside the public one (see "Verify this copy"). The first entry is `../bowl_step8/step8-source-lock.sha256`, which pins the frozen adapter and gate files the runner imports from `experiments/bowl_step6/` (`run_painted_dev.py`, `paint_A.py`, `local_ik.py`, `rigid_chain.py`) and `experiments/bowl_step5/` (`release_gate_v4.py`, `rim_support.py`, the hashed `release_gate_v3.py` and `stability.py`, and the policy server `run_policy.py`), plus the pinned inputs under `frozen_B_adapter_v1r2/files/`. A live run also needs everything under "Reproduce" below and one file from the evidence bundle: the A/P0 layout-4003 scene record `RoboLab/output/step8_A_P0_l4003_s0_v1/BananaInBowlUniformInitPose10cmTask/strict-metric.json` (sha256 `61435857…0c8f83`, checked by `layout_pin.py`). `LAUNCH_DAY.md` is the launch checklist written for the run.

### Re-score the shipped trace (CPU only)

The scorer needs Python 3 and numpy. Run from the repository root; Python puts the scorer's own folder on the path, and `PYTHONPATH` adds `experiments/bowl_step5` for `release_gate_v4`. Don't add `-I`, which ignores `PYTHONPATH`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/bowl_step5 \
  python3 experiments/scoring_v5/release_gate_v5_1.py evidence/abba_l4003/gate_trace.jsonl > /tmp/abba_v51.json
python3 -c "import json;a,=json.load(open('/tmp/abba_v51.json')).values();b,=json.load(open('evidence/abba_l4003/SCORE_V51.json')).values();print('same verdict' if a==b else 'DIFFERENT')"
```

The scorer reads `tracking.npz` from the trace's folder. The output is keyed by the path you pass; `SCORE_V51.json` is keyed by the original local path, so the check compares the values. Re-run for this export on 26 October: `pass` true at control 1071, identical to `SCORE_V51.json`.

### Evidence shipped and held back

`evidence/abba_l4003/` (8.2 MB) holds what the scorer and the cost figures need. Check it with `cd evidence/abba_l4003 && shasum -a 256 -c MANIFEST.sha256` (68 files).

- `gate_trace.jsonl` and `tracking.npz`: the executed-command trace and object tracking the scorer reads. `policy_command` in the trace is P1's own gripper command.
- `SCORE_V51.json`: the v5.1 result. `checks.json`: the runner's own record, including the live v4 reading. `release-command.json`: the release marker (zero-based tick 1005). `drop_protocol.json`: the protocol the runner wrote at start, with `runner_B.py`'s sha256.
- `api-ledger.jsonl`: one reserved and one settled row per call, with token counts, charge and latency. It holds no key, header or account data.
- `drop/query_0001/` … `drop/query_0029/`: each call's `request.json` (prompt, phase, tick and the sha256 of each image sent) and `response.json` (Nemotron's raw reply). Text only; they contained no local paths.
- `COPY_VERIFICATION.json`, and `copy-manifest.run.json` and `copy-manifest.supervisor.json`, the manifests the VM wrote when it sealed the two output folders. Every shipped run file matches its entry in those manifests.

Held back. The full videos and the other large files go with the run evidence at https://github.com/MiodragMTasic/cross-embodiment-policy-transfer/releases/tag/evidence-v1:

- `views.mp4` (the full 4-minute video, 52 MB), the edited release clips, the 174 PNG frames sent to Nemotron and the six start-up renders: size, and they are renders of RoboLab assets (see NOTICE). The frames' hashes are in each `request.json`.
- `requests.hdf5` and `data.hdf5` (31 MB together), `ik_trace.jsonl` (13 MB), `drop/events.jsonl` (2.8 MB, the supervisor's per-control log), the 38 small policy response files, the two start-up camera records, and the supervisor's exit code, `ATTEMPT_COMPLETE` marker and empty GPU snapshot: size, or not needed to score or cost the run. Their hashes are in the two copy manifests.

Not in this repository; in the evidence release only after scrubbing:

- `env_cfg.json`, `inventory.json`, `model-loading.json`, `policy.log` and `runner.log`: they record the VM's home-directory and model-cache paths, and nothing in them is needed to score the run. The evidence release ships them with that home path replaced by `<vm-home>` (its `SANITIZATION.md` lists each file).

Kept private:

- The launch permit and the supervisor's copy of it (`permit.json`): they hold the cloud VM's instance ID and account balance figures. Their only reproduction-relevant field, the pinned source lock, is `166a1bb9…a1a91b83`, the hash of the shipped `source-lock.json`.
- The session monitor record: it holds the VM's public IP address.

## Nemotron add-on (annotation only)

With my approval, a Claude coding agent wrote the tooling and made the calls to NVIDIA Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) on Nebius Token Factory, under a hard $2 cap. Nemotron read a text timeline of each "before" test episode, built from its logs. It returned a schema-checked outcome and failure label with cited control steps. It never touches a robot, a policy or a score. Its labels are compared with the locked scorer and the reviewer and never merged into either.

The result is negative. From text timelines, its blind outcome agreement did not beat a trivial always-success baseline on these 24 episodes:

- its majority vote matched the reviewer on 15/24, and always answering "success" also scores 15/24;
- balanced accuracy was 0.61 (success recall 10/15, failure recall 5/9), against 0.50 for the baseline;
- the locked v5.1 scorer alone matched the reviewer on 23/24;
- shown the scorer's verdict, Nemotron matched 19/24, still below the scorer it was shown.

On A 4000 the blind label agreed with a scorer miss that the reviewer had already recorded. Shown the scorer's verdict, its answer flipped. A cited control step that exists in the timeline does not show that the step supports the label.

The usage cost was about $0.02, estimated from returned token counts and recorded prices; the Token Factory usage page confirmed it on 24 October. A held-out test with a revised prompt is drafted but not run.

## Repository layout

```
experiments/
  bowl_step5/    A-side instrumentation: policy server wrapper (run_policy.py), controlled-episode runners,
                 release gates v3/v4, rim support, stability, dataset converters and checks
  bowl_step6/    the adapter: local_ik.py, rigid_chain.py, paint_A.py (cross-painting), run_painted_dev.py
                 (frozen configuration), the shadow-test diagnosis and earlier bridge versions
  bowl_step7/    P1: collection, dataset conversion, fine-tuning (run_train_p1.*), checkpoint selection, pinning
  bowl_step8/    final test: run_step8.sh, step8_B_runner.py, abort classifier, canary comparison, STEP8_RUNNERS.md
  scoring_v5/    locked scorer (release_gate_v5_1.py, stages_r2.py, score_step8.py, score_select_step7.py),
                 validation sets, and the before, checkpoint-selection and after score outputs
  roger_bravo_one_shot_v1/      ABBA: the assisted-release supervisor that ran once, its launcher and 43 offline tests
  charlie_bravo_v1/             DROP supervisor, transport and synthetic offline tests; prepared, never run
  charlie_bravo_perception_v1/  RGB-D capture and perception design and offline tests; prepared, never run
  intent_demo_v1/               intent-checking and demo tooling; prepared, no robot demo run
evidence/abba_l4003/     ABBA's trace, score, API ledger and the 29 Nemotron requests and replies (see "ABBA")
frozen_B_adapter_v1r2/   freeze record: README, manifest.json, lock-relative.sha256 and 17 of the 19 frozen files
setup/           the install scripts used on the VM on 21 October, unchanged (see "Install")
prerequisites/   one step 5 record the step 7 and step 8 supervisors check (see "Run order")
nemotron/        text annotation and intent tooling; no credentials or raw API payloads
tools/           verify_locks.py and make_public_manifest.py (see "Verify this copy")
LICENSE  NOTICE  SCAN.md  SANITIZATION.md  SANITIZATION.json  PUBLIC_MANIFEST.sha256
CHARLIE_EXPORT_PLAN.md  EXPORT_PROVENANCE.json  OFFLINE_EXPORT_CHECKS.md
```

Every `*.sha256` lock is kept byte-identical to the one used during the experiment, and so is ABBA's `source-lock.json`; `evidence/abba_l4003/MANIFEST.sha256` is new, written for this export. Three of the files those locks pin ship sanitized: `experiments/scoring_v5/score_step8_mock/score_step8-mock.json`, `experiments/scoring_v5/validation-pose-variant.json` and ABBA's `preflight.py` (see "Verify this copy"). The never-run Charlie packages ship export-specific locks beside their original ones (see `CHARLIE_EXPORT_PLAN.md`). `experiments/scoring_v5/SCORING_V5.md` (line 178) states the significance threshold as "at least 6 discordant layouts, all in one direction"; it is locked and left as it was, and the statement under Honest limits is the corrected one. Markdown files inside `experiments/` and the freeze record cite a lab notebook (`EXPERIMENT_REPORT.md`, `STEP_REVIEW_LOG.md`) that is not part of this repository.

## Verify this copy

This repository is a sanitized copy of the experiment's files, not a byte-identical one. 21 files differ from their originals. In the redactions, local Mac paths and the cloud VM's instance ID become `<project-root>/` and `<vm-instance-id>`, and one reviewer nickname is reworded. The rest are labelled public-copy edits: rewritten package READMEs, a skipped image test, a licence header and two export locks. `SANITIZATION.md` lists each file with its original and public sha256, the changed lines and the reason, and `SANITIZATION.json` holds the same record for tools. Every other file is byte-identical to its original, and the private originals are kept unchanged.

From the repository root, with the Python 3 standard library:

```bash
shasum -a 256 -c PUBLIC_MANIFEST.sha256    # this copy is complete and unmodified: every line OK
python3 tools/verify_locks.py              # exit 0: no lock entry fails
```

`PUBLIC_MANIFEST.sha256` lists every shipped file; `python3 tools/make_public_manifest.py` rewrites it after a change. `tools/verify_locks.py` reads all 36 lock files (855 entries) and maps the sanitized files back to their originals. Each entry is OK (the hash matches) or SANITIZED (the file is a documented sanitized copy and the lock entry equals its recorded original sha256). It can also be NOT_SHIPPED: upstream code, bulky run outputs or held-back evidence, each with a reason in `SANITIZATION.json`. Anything else is FAIL. Result on 27 October: 553 OK, 5 SANITIZED, 297 NOT_SHIPPED, 0 FAIL.

## Reproduce

### Hardware and pins

| Item            | Value                                                                                            |
| --------------- | ------------------------------------------------------------------------------------------------ |
| Machine         | Nebius AI Cloud VM: 1× NVIDIA RTX PRO 6000 Blackwell (96 GB), 24 vCPUs, 218 GiB RAM, 256 GiB SSD |
| OS and driver   | Ubuntu 24.04 LTS NVIDIA GPU image; driver 580.173.02 (CUDA 13.0 compatibility)                   |
| GPU requirement | RT cores (Isaac Sim does not run on A100 or H100)                                                |
| RoboLab         | https://github.com/NVlabs/RoboLab at `b6cfcf6ccd2113c4cd1216d35b33024a7d5691da`                  |
| Isaac-GR00T     | https://github.com/NVIDIA/Isaac-GR00T at `dac54b313160c201c450f97df0fd74ebd379c892`              |
| Policy weights  | `nvidia/GR00T-N1.7-DROID` revision `05e7cc97e40dbd33b0890c35cc0214fcb0547ab5`                    |
| Backbone        | `nvidia/Cosmos-Reason2-2B` revision `9ce19a195e423419c349abfc86fd07178b230561`                   |
| Simulator env   | `RoboLab/.venv`: Python 3.11, Isaac Sim 5.1.0.0, Isaac Lab 2.3.2.post1, PyTorch 2.7.0+cu128      |
| Policy env      | `Isaac-GR00T/.venv`: Python 3.12, PyTorch 2.9.0+cu128                                            |
| Installer       | uv 0.13.4, each project from its committed lockfile; Git LFS for RoboLab's assets                |

RoboLab's Isaac Sim wheel reported three conflicts with its lockfile; the working overlay pinned filelock 3.13.1, fsspec 2024.6.1 and networkx 3.3. The policy server binds `127.0.0.1:15555` (port 5555 was taken by NVIDIA's `nv-hostengine` on the image).

### Install

The scripts in `setup/` are the ones used on the VM, unchanged. They install under `$HOME/robot-transfer/bowl`, which only matches the runners' hardcoded root for a user named `miodrag`; see "Root path" first. Isaac Sim and Isaac Lab come from NVIDIA's pip wheels through RoboLab's lockfile; there is no separate Isaac Sim install.

```bash
bash setup/bootstrap.sh                  # apt: ffmpeg, git-lfs, python3.12-venv; uv 0.13.4 in tools/uv;
                                         # shallow clones of Isaac-GR00T and RoboLab at the pinned commits
bash setup/setup_job.sh groot-env        # Isaac-GR00T: uv sync --locked --python 3.12, then uv pip check
bash setup/setup_job.sh robolab-env      # RoboLab: uv sync --locked --extra isaac51 --python 3.11, then uv pip check
bash setup/setup_job.sh robolab-assets   # RoboLab: git lfs pull and git lfs fsck
```

Three manual steps go with these; paths are relative to the root. The exact command lines for the first two were not saved, so the commands shown are equivalents:

- The first `groot-env` sync failed because uv reads metadata from Isaac-GR00T's bundled aarch64 torchcodec wheel, a Git LFS pointer in a fresh clone, even on x86 (it is not installed). Fetching that one object fixed it, for example `git -C Isaac-GR00T lfs pull --include "scripts/deployment/dgpu/wheels/torchcodec-0.8.0-cp312-cp312-linux_aarch64.whl"`, then rerun the job.
- After `robolab-env`, install the three pins in `setup/compatibility-overrides.txt` into RoboLab's environment, for example `tools/uv/bin/uv pip install --python RoboLab/.venv/bin/python -r setup/compatibility-overrides.txt`, and rerun `uv pip check`. Don't run a plain `uv sync` afterwards: it removes the `isaac51` extra and the overlay.
- Weights: `setup/download_models.py` downloads both pinned snapshots with `huggingface_hub` (Isaac-GR00T's environment has it) into the Hugging Face cache and writes `models/model-manifest.json` with every file's size and sha256. `nvidia/Cosmos-Reason2-2B` is gated on Hugging Face: accept its terms and log in first. Both models are under the NVIDIA Open Model License (see NOTICE).

The runners set `OMNI_KIT_ACCEPT_EULA=Y`, which accepts NVIDIA's Omniverse and Isaac Sim license terms without a prompt. Read the [Isaac Sim Additional Software and Materials License](https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/) before running them. The supervisors use GNU `sha256sum` and `nvidia-smi`, so they run on Linux with an NVIDIA GPU only.

### Root path

The runners hardcode `root=/home/miodrag/robot-transfer/bowl`. They are left byte-identical so their source locks still verify. Either create that directory (a symlink to your working tree is enough) or edit `root=` and accept that the locks will no longer match. The expected tree:

```
/home/miodrag/robot-transfer/bowl/
  RoboLab/               upstream clone at the pinned commit, with its .venv
  Isaac-GR00T/           upstream clone at the pinned commit, with its .venv
  experiments/           this repository's experiments/
  models/model-manifest.json   pinned snapshot paths with per-file size and sha256
                               (format: frozen_B_adapter_v1r2/files/models/model-manifest.json)
  runs/  datasets/       created by the runners
```

The adapter also reads four pinned inputs from under the root. Copy them from `frozen_B_adapter_v1r2/files/` to the same relative paths (`runs/step6_fixed_frames_cpu_v1/`, `runs/step6_A_tracking_v3/`), plus the A reference traces under `RoboLab/output/` for the development scenes. Two of the 19 frozen files are not in this repository:

- `RoboLab/policies/gr00t/client.py` is upstream RoboLab code, unmodified. It comes with the pinned clone; its sha256 `167a46fc…f41f1` is in `lock-relative.sha256`.
- `assets/report/evidence/2026-10-21/bowl-step6/kinematics_export_v1/joint_graphs.json` is a dump of body frames and joints extracted from RoboLab's robot assets (see NOTICE). Regenerate it with `experiments/bowl_step6/extract_arm_kinematics.py` under RoboLab's Python and compare with its sha256 in `lock-relative.sha256`. Byte-equality off the original VM has not been tested.

Check the freeze record:

```bash
cd frozen_B_adapter_v1r2/files && shasum -a 256 -c ../lock-relative.sha256   # 17 OK here; 19 OK once both files are added
```

### Run order (as run)

Every supervisor refuses to start unless its source lock verifies, both upstream repos sit at the pinned commits with clean trees, every pinned model file re-hashes, and the GPU is idle. The step 7 and step 8 supervisors also require the A runner to be byte-identical to the one hashed for the approved step 5 training-feasibility runs; copy `prerequisites/runs/step5_training_layout2002_v4/source-hashes.txt` to the same path under the root. Run from the root.

```bash
bash experiments/bowl_step8/run_step8.sh smoke                        # non-scoring runner check on training layout 2000; "before" requires it
bash experiments/bowl_step8/run_step8.sh before                       # A/P0 then B/P0 on 4000-4011
bash experiments/bowl_step7/run_collect_v1.sh first                   # A/P0 collection on 2004-2031 (2000-2003 reused from step 5)
bash experiments/bowl_step7/run_collect_v1.sh seed1                   # 2000-2031 again at policy seed 1
# convert reviewer-approved wins only: convert_step7.py candidates | export | build, then verify_step7_dataset.py
bash experiments/bowl_step7/run_train_p1.sh probe                     # 20-step probe
bash experiments/bowl_step7/run_train_p1.sh train                     # 2,000 steps, batch 32, lr 1e-4, checkpoint every 500
python3 experiments/bowl_step7/pin_step7_v1.py candidates models/step7_train_v1 models/P1_v1/candidates-sha256.json
bash experiments/bowl_step7/run_select_v1.sh models/P1_v1/candidates-sha256.json <sha256>   # A on 3000-3007
# score selection (raw, then with the reviewer's verdicts via --adjudicated), then pin the choice:
python3 experiments/bowl_step7/pin_step7_v1.py select models/P1_v1/candidates-sha256.json select-step7.json <cN> models/P1_v1/P1-manifest.json
bash experiments/bowl_step8/run_step8.sh after models/P1_v1/P1-manifest.json <sha256>       # canary, then A/P1 and B/P1
```

Checkpoint selection takes a reviewer's verdict file (scheme in `experiments/scoring_v5/score_select_step7.py`), and the adjudicated counts are binding for that choice. The test halves are different: the formal test reports the locked v5.1 counts with the independent reviewer's counts beside them, and the reviewer's counts never replace the headline. A B test episode has a budget of 9,000 controls (600 s of simulated time); A's is 750 controls (50 s).

### Score the before half (CPU only)

The scorer needs Python 3 and numpy (checked with Python 3.14.4 and numpy 2.4.4) plus the run evidence, which is not in this repository: https://github.com/MiodragMTasic/cross-embodiment-policy-transfer/releases/tag/evidence-v1. The scorer's inputs alone are about 265 MB; with videos the bundle is about 1.2 GB.

```bash
cd experiments/scoring_v5
PYTHONDONTWRITEBYTECODE=1 python3 -I score_step8.py --phase before --out /tmp/score_before \
  --stamp "$(date -u)" "$(cd /path/to/step8_before_v1 && pwd)"
```

Pass the evidence root as an **absolute** path. With a relative root, the scorer's temporary `tracking.npz` symlink for safety-abort episodes dangles and the run crashes. Re-running this on 24 October reproduced `score_step8-before.json` exactly, apart from the stamp. In the committed copy the local evidence prefix is replaced with `<project-root>/` (see SANITIZATION.md), so compare the scores, not the bytes.

For the after half, pass both evidence roots so P1 pairs with P0: `score_step8.py --phase after --out DIR --stamp ... <before root> <after root>`, and compare with `score_step8_after/score_step8-after.json`, whose evidence prefix is redacted the same way.

## Credits

- Cross-painting: L. Y. Chen, K. Hari, K. Dharmarajan, C. Xu, Q. Vuong and K. Goldberg, "Mirage: Cross-Embodiment Zero-Shot Policy Transfer with Cross-Painting", RSS 2024, arXiv:2402.19249.
- NVIDIA GR00T N1.7 DROID (Isaac-GR00T), whose vision-language backbone is NVIDIA Cosmos-Reason2-2B. NVIDIA post-trained this checkpoint on the DROID dataset (A. Khazatsky et al., "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset", RSS 2024, arXiv:2403.12945).
- NVIDIA RoboLab and its banana-in-bowl tasks and assets, including the YCB-Video banana and bowl models; NVIDIA Isaac Sim 5.1 and Isaac Lab 2.3.2.
- Nebius AI Cloud (one GPU VM for simulation, policy serving and fine-tuning) and Nebius Token Factory (`nvidia/Nemotron-3_5-Lightning` for the annotation add-on, `nvidia/Nemotron-3.5-Super-VL` for ABBA). Both models are NVIDIA's, under the OpenMDW-1.1 license per their model cards; they were used only as hosted APIs (see NOTICE).
- AI agents did most of the engineering under my direction: Claude (Anthropic) and Codex (OpenAI) from about 21 October; before that, Codex suggested the commands I ran by hand. Step reviews and outcome adjudication were by AI agents, not people: a Codex adversary agent until 22 October, then independent Claude agents, and Codex agents for the after-test review, the score replay and the training-data audit. Codex built and ran ABBA and checked its sampled frames; afterwards a Claude agent re-verified its evidence, re-score and ledger. A separate Claude analysis found the same training-data gap.

## License

Apache License 2.0 for the code and records written for this project; see `LICENSE`. The website in `docs/` is not under Apache: its design is adapted from a CC BY-SA 4.0 template, its simulation clips and posters are renders of NVIDIA RoboLab's CC BY-NC-SA 4.0 scene and are shared under CC BY-NC-SA 4.0, the notebook photographs are © Miodrag Tasic, all rights reserved, and the Seoul kickoff photograph is Bloom's, credited on the Why page (`docs/LICENSE.md`). That license does not cover the third-party software, models and assets the code runs against; each keeps its own terms. `NOTICE` lists them, what is deliberately not included, and the one file derived from RoboLab.
