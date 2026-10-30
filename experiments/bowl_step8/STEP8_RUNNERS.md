# Step 8 runners and supervisor

Built 2026-10-23 19:15 UTC. Revised 19:39 UTC for the review's CHANGES REQUIRED (B1 runner location, B2 safety aborts, N1–N3), and 19:47 UTC for the re-review (C1 step-0 aborts, the same-kind stop trigger, the A/B limit disclosure). Nothing has run. The governing documents are:

- the "Rule for step 8" in `assets/report/evidence/2026-10-23/bowl-step6/frozen_B_adapter_v1r2/README.md`;
- the reviewer's step 8 notes (STEP_REVIEW_LOG, 19:00 UTC entry);
- EXPERIMENT_REPORT §22.4 (reorder: the "before" half first).

## Files

The VM copies live in `~/robot-transfer/bowl/experiments/bowl_step8/`, except the B runner, which lives beside its frozen siblings in `experiments/bowl_step6/`. `sha256sum -c` passes 27/27 on the VM, and the local copies match.

| File                                  | sha256        | Role                                                                              |
| ------------------------------------- | ------------- | --------------------------------------------------------------------------------- |
| `bowl_step8/run_step8.sh`             | 0e6bbc53ee47… | Supervisor, one run per phase: `smoke`, `before` or `after`                       |
| `bowl_step6/step8_B_runner.py`        | 3487ea5749ea… | B runner: the frozen `run_painted_dev.py`, changed only inside the allowed ranges |
| `bowl_step8/check_step8_diff.py`      | a7b59b1f0c22… | Mechanical rule check, run by the supervisor at every launch                      |
| `bowl_step8/compare_canary.py`        | b3475044db08… | Field-by-field drift canary report for the after phase                            |
| `bowl_step8/classify_abort.py`        | 5c96bea489f7… | Sorts a B exception into safety_abort (task outcome) or crash                     |
| `bowl_step8/step8-source-lock.sha256` | b2816defe8bd… | 27 entries, listed below                                                          |

**Why the runner sits in `bowl_step6/`.** It imports `rigid_chain`, `local_ik` (:51/:52) and `paint_A` (:69) from its own directory, and :186 hashes those siblings plus `run_painted_dev.py` through `Path(__file__).with_name`. From `bowl_step8/` the imports fail. `bowl_step6/step8_B_runner.py` is a new file, byte-identical to the reviewed copy; no existing `bowl_step6` file changed (its frozen lock still passes 19/19). The import proof ran on the VM with the runner's interpreter (RoboLab venv, Python 3.11.17), from `bowl_step6/`, `PYTHONPATH=$root/RoboLab`, the runner's `sys.path` rebuilt step by step and resolved with `importlib.util.find_spec` only, so nothing was executed and no Isaac app started. `rigid_chain`, `local_ik` and `paint_A` resolve to `bowl_step6/`, `release_gate_v4` (with `release_gate_v3`, `stability`) and `rim_support` to `bowl_step5/`, each matching its lock hash, and all four :186 `with_name` targets exist. This is the same `sys.path` the frozen `run_painted_dev.py` ran under.

The first copy, `bowl_step8/step8_B_runner.py`, is still there, byte-identical, referenced by nothing and outside the lock. It cannot import its siblings, so it fails at once if run by hand. Deleting it needs the coordinator's say-so. `bowl_step8/` holds no module copies (no `rigid_chain`, `local_ik`, `paint_A` or others).

Note that the runner's own `checks.json` `source_hashes['run_painted_dev.py']` (frozen :184, outside the ranges) records the frozen template's hash, not the runner's. The runner's identity comes from the supervisor lock.

**What the lock covers:**

- the 5 step 8 files above;
- all 19 frozen B-adapter files, byte-identical to `frozen_B_adapter_v1r2/lock-relative.sha256`. `run_painted_dev.py` is included as the diff reference;
- the A runner `run_controlled_episode_v4_registered.py`;
- `audit_capture.py`;
- the approved step 5 `runs/step5_training_layout2002_v4/source-hashes.txt`, which is the A diff reference.

## B runner: what changed and where

The rule check output is `STEP8_DIFF_OK B 9 hunks, all inside the allowed ranges`, on the Mac and on the VM (`check_step8_diff.py B bowl_step6/run_painted_dev.py bowl_step6/step8_B_runner.py`). Every edit was applied by exact-line replacement, asserting the frozen text first.

| Frozen line  | Change                                                                                                                                                                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 25, 27, 28   | `--scene {layout, base}`. `--variant {lift}` only. Added `--layout-seed` and `--policy {P0,P1}`, both inserted after 28.                                                                                                                               |
| 31           | `assert POLICY_SEED==0`                                                                                                                                                                                                                                |
| 32–34        | Smoke case: `step8_B_smoke_l2000_s0_v1` on the `layout` branch with training layout 2000, P0, 300 controls, reading the A smoke's poses. Test case: `step8_B_{P0\|P1}_l{4000..4011}_s0_v1`, or the optional `step8_B_P0canary_l{L}_s0_v1`.                                                               |
| 98, 100, 101 | A `layout` branch, identical to the old `l2002` branch except `layout_seed=args.layout_seed` and `Aref = output/step8_A_P0_l{L}_s0_v1/BananaInBowlUniformInitPose10cmTask` (`step8_A_smoke_l2000_…` for the smoke case). Both B arms read A/P0's poses. The `Ametric` asserts on 105 are unchanged. |
| 164, 322     | Scope text, plus new keys `policy_label` and `layout_seed_argument` (164 also gets `frozen_record`). Existing keys are unchanged.                                                                                                                      |
| 315, 323     | Video label `STEP8 {policy}`, completion tag `B_STEP8_COMPLETE`                                                                                                                                                                                        |

**Tested negatively.** A rate-limit edit at frozen :281 and a line inserted near :119 both produce `STEP8_DIFF_FAIL`. The unchanged `B_PAINT_DEV_PROGRESS` tag (:317) and the dev-round module docstring (:1–9) stay as they are, because they lie outside the ranges.

## A runner: no derived copy

The step 5 v4 registered runner already takes the layout (`--seed`) and case (`--output-folder-name`) as arguments.

- Its `strict-metric.json` already carries `layout_seed`, `policy_seed` and `initial_poses_world` (banana, bowl and robot, 13 values each) in the step 5 schema, `command-linked-supported-release-v4`. This was checked on `step5_training_layout2002_v4`.
- So A runs that file itself, and its allowed range list is empty.
- `check_step8_diff.py A` requires it to be byte-identical to the hash recorded for the approved training-feasibility runs: `STEP8_DIFF_OK A`, a3191192… on both sides.

## Supervisor: `run_step8.sh` (run with `bash`)

**Copied blocks.** Checked by text extraction:

- From `run_paint_dev_lift1.sh`, byte-identical:
  - stop_server/finish/trap, lines 13–30;
  - the git pins, 34–37;
  - model-manifest hashing, 41–55;
  - the P0 load-check body, 79–87;
  - the GPU-release wait, 62–63;
  - server readiness and freshness, 69 and 71–76. Line 70's literal `seq 1 300` became `seq 1 "$attempts"`, because A and B share `start_server` (B passes 300, as frozen). The 19:15 text called 69–76 byte-identical, which was wrong;
  - the success read, 99. The failure read, 98, is replaced by the abort classification below.
- From `run_controlled_case_v4_registered.sh`, byte-identical:
  - the capture-integrity body, 71–85.
- The A episode command (64–68) has the same arguments and `timeout 600`. Only the variable names differ: `$s5`, `$layout`, and the case's run dir.

**Checks in every phase:**

- the lock, both git HEADs and clean tracked trees;
- `check_step8_diff.py` for both A and B;
- `isaacsim` 5.1.0.0 / `isaaclab` 2.3.2.post1, read from the RoboLab venv;
- the NVIDIA driver and CUDA versions recorded to `platform.txt`; currently 580.173.02 and 13.0;
- `git status --porcelain --untracked-files=all` empty on RoboLab and Isaac-GR00T, apart from `output/`;
- no ignored non-cache file under `RoboLab/robolab/tasks`, which is what `auto_register_kinova_envs` scans;
- all 32 pinned P0/Cosmos model files hashed.

**Phases:**

- **`smoke`.** Non-scoring.
  - A/P0 on training layout 2000: `step8_A_smoke_l2000_s0_v1`. Step 7 must exclude it from training data by name.
  - Then B/P0 on layout 2000 for 300 controls, reading the A smoke's `strict-metric.json`. This exercises the changed `layout` branch and the A-to-B pose hand-off end to end. The `base` branch is kept but unused.
  - Both must exit cleanly, which writes `STEP8_SMOKE_COMPLETE`.
- **`before`.** Requires the smoke supervisor's exit 0 and its marker. Per layout 4000→4011:
  - A/P0;
  - then the sha256 of A/P0's `strict-metric.json` is recorded to `A_P0_strict-metric.sha256`;
  - then B/P0.
  - If A/P0 yields no valid capture, B is skipped for that layout and the skip is logged.
- **`after <P1-manifest.json> <sha256> [--b-canary]`.** Requires all of:
  - the before phase's exit 0 and `STEP8_BEFORE_COMPLETE`;
  - `platform.txt` byte-equal to the before phase's (same driver and CUDA);
  - the P1 manifest's sha256 equal to the argument.

  Then the drift canary:
  - A/P0 is rerun on 4000 and 4001 as `step8_A_P0canary_l{L}_s0_v1`, compared field by field with the before-phase traces (`canary_A_l{L}.json`), including v4 outcome and initial poses.
  - `--b-canary` adds a B/P0 rerun on 4000.
  - The canary is recorded, not a gate. Repeated P0 requests aren't bit-reproducible (§20), and v5 for both runs is scored offline.

  Then per layout, in the before order:
  - A/P1. Its `initial_poses_world` must equal A/P0's to 1e-6, or the batch halts.
  - A/P0's `strict-metric.json` must still match the hash recorded in the before phase.
  - Then B/P1, reading those same A/P0 poses.

**P1 serving.**

- `run_policy.py` is unchanged and serves `POLICY_CHECKPOINT=<checkpoint_dir>` under the label "disposable copy".
- Before every P1 episode the supervisor checks all of the following:
  - every file in the sealed manifest is re-hashed;
  - the directory must hold exactly the listed files;
  - `model-loading.json`: label "disposable copy"; `effective_checkpoint` and the "disposable copy" load path equal to `checkpoint_dir`, exactly, with no trailing slash (the format was checked against `step5_reload_smoke_v1`); seed 0; same P0 snapshot name and Cosmos revision; loads `{disposable copy, Cosmos}`; and all four `loading_info` lists empty (missing, unexpected, mismatched, error_msgs).

**P1-manifest.json schema, which step 7 must produce:**

```
{"checkpoint_dir": "<absolute path, no trailing slash>",
 "files": [{"path": "<relative to checkpoint_dir>", "size": <bytes>, "sha256": "<hex>"}, ...]}
```

**Failures.**

- Any pin, hash, version, platform or pose check failing halts the phase. So does a server that won't start.
- **Stop trigger, declared before any run:** "Stop the phase and ask Miodrag after 3 consecutive B safety_aborts of the same kind within one phase." `record()` does this mechanically. It reads `kind` from each B safety abort's `abort-classification.json`, and on the third consecutive one of the same kind prints `HALT: 3 consecutive B safety_aborts of kind <kind> in this phase. Stop and ask Miodrag` and exits 1. Any other B episode (no abort, a crash, an abort of another kind) resets the run; A episodes don't. The halting episode's row is written first. Scoring is unchanged: every one of those aborts is still a task outcome for its completed layout.
- An infrastructure failure is recorded in `episodes.tsv` and the phase continues. It is any of: A exiting non-zero or failing capture integrity (`failure=capture`); B exiting non-zero without `failure.txt`; or B's `failure.txt` classified `crash`. The third one in a phase halts it.
- v4 outcomes print live: the `v4_success` column.
- **A halt in the before phase ends step 8 until Miodrag decides.** Nothing restarts it automatically. The case names are fixed in the runner (:34, the test-case assert) and the run dirs refuse to exist twice. So any rerun needs new case names, which means a runner edit at :34 and a fresh review of that edit before launch. The same holds for the after phase.

**Safety aborts are task outcomes (reviewer B2).** When B's `failure.txt` exists, `classify_abort.py runs/CASE STATUS` reads the last traceback in it. It writes `runs/CASE/abort-classification.json` (schema `step8-abort-classification-v1`, never overwritten) and the class lands in the `failure` column: `safety_abort` or `crash`.

- A `safety_abort` is the bridge stopping on what the policy commanded, through the frozen adapter, mid-episode. It is scored by v5.1 and stages_r2 on the partial trace that the runner's `finally` block saved: `gate_trace.jsonl`, plus `tracking.npz` with `complete=False`. It is included in ΔB, and it never counts toward the 3-failure halt.
- Everything else is a `crash`: any exception not matched exactly below, and any timeout (exit 124/137), even one after a matched abort (a hung close is infrastructure).
- The smoke phase still demands exit 0 and `failure=no`, so a safety abort fails the smoke.
- A gets no classifier. Its approved runner has no policy-driven assert: :181/:182 check the response's shape and finiteness, which is integrity, the same as B's :270/:271.

A match needs all of: the innermost frame's file, line, function and source text; the nearest runner frame calling it; the exception type; and the message. **Every kind also needs `gate_trace.jsonl` to hold at least one row (C1).** :297 appends one row per executed step, so with no row no command has reached the simulator yet, and the exit is a crash. At :237, :273 and :282 the row count equals the loop step, so this means step ≥ 1. A :298 soft-limit abort already has its row (step + 1), so a post-command abort at step 0 still counts. Lines are the derived runner's (frozen +2 after line 28). The step number inside an IK error message is not checked: the row count is the step-0 test, so don't read the message as the test.

| `kind`                           | Line (frozen)                       | Matched only when                                                         | Why it is the policy's doing                                                                    |
| -------------------------------- | ----------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `B_joint_beyond_soft_limit`      | :201 (199), in `state()`            | called from :298 (296), the post-step state                               | B's measured joints after a step, more than 0.03 rad past the soft limits the commands drove it to |
| `policy_A_target_outside_limits` | :273 (271)                          | message `P0 A joint target outside physical limits` (frozen text, also P1) | the policy's own A joint target                                                                  |
| `ik_action_A_to_B`               | :85 (83), `log_solve`               | called from :282 (280), message `(step,'action_A_to_B',reason)`           | B cannot reach the policy's A tool target: the adapter's action map                             |
| `ik_observation_B_to_A`          | :85 (83), `log_solve`               | called from :237 (235), message `(step,'observation_B_to_A',reason)` | A cannot represent where the commands took B: the adapter's observation map                 |
| `B_joint_beyond_physical_limit`  | `local_ik.py:24` `ValueError`       | called from :282 (280), `Seed outside physical limits`                    | B's measured joints past the physical limit, fed as the action IK seed: the soft-limit event in the window up to soft + 0.03 |

These stay crashes on purpose:

- **Before any command reached the simulator (no `gate_trace` row):** any of the five kinds at step 0. That covers the policy's first target out of limits, the first action IK, the physical-limit seed and the observation IK at step 0, and the soft-limit assert via :204 (startup) and :235 (step 0; at later steps :235 re-reads the state :298 already passed).
- **A bug in the frozen code:** the seed check via :237, since that seed is A's own previous IK solution.
- **Simulator or model integrity:** :197/:199/:200 (non-finite state, base moved, FK mismatch), :306 (non-finite objects or forces), and `local_ik`'s other `ValueError`s (invalid seed or limits, non-finite update, invalid final state).
- **Capture integrity:** :256–:258 (camera tracking) and :316 (video frame).
- **Server response integrity:** :270/:271.
- **Invariants guaranteed by the code:** :284/:285 (rate limit, command bounds).
- **Episode structure:** :293 (unexpected reset), :302/:303 (contact sensor wiring).
- **Startup and configuration:** :33–:158 and :219–:230.

The classifier was tested on 17 traceback/row cases × 2 exit statuses (34 cases), on the Mac and on the VM, with 0 wrong. The tracebacks were raised by Python 3.11 on the VM at the runner's real path and line numbers. The 12 originals at realistic row counts give `safety_abort` for the five signatures and `crash` for the rest: :237's seed check, :284, :293, :235, the same assert at the `bowl_step8` path, and the real `step6_B_P0_feedback_v1` IK failure. The step-0 variants (soft limit via :298 at 0 and 1 rows; policy target, action IK and physical-limit seed at 0 rows; observation IK at step 0) give `safety_abort` only for the soft limit with 1 row. Every timeout is a `crash`.

`record()` was tested with the real function, on the Mac and in the VM's bash 5.2:
- five safety aborts of alternating kinds leave the failure count at 0;
- the third infrastructure failure halts;
- 3 same-kind B safety aborts, with A episodes between them, halt;
- 2 of one kind and then 1 of another don't;
- a clean B or a B crash between same-kind aborts resets the run.

**For `score_step8.py`.** A B episode is scored if it has `checks.json` with `complete: true`, or `abort-classification.json` with `class: safety_abort`. `partial_trace` in that file gives `gate_trace_rows`, `last_gate_trace_step`, and whether `tracking.npz`, `failed_solve.json` and `checks.json` exist. An abort on the very first control leaves no trace files (`gate_trace_rows: 0`). How v5.1/stages_r2 score an empty trace is the scorer's to state.

**Disclosure: B checks limits that A doesn't.** A's approved runner has no target-limit check. It asserts only the action's shape and finiteness (`run_controlled_episode_v4_registered.py:181-182`) and hands the target to A's controller. So a policy target outside A's physical limits ends a B episode (:273, `policy_A_target_outside_limits`) but not an A episode. This doesn't touch the within-robot ΔB (B/P0 vs B/P1, same runner, same checks). It does mean an A-versus-B comparison of the same policy must name it. Abort counts are reported per kind, per B arm (P0, P1), from `abort-classification.json`.

## Partial runs (reviewer R4)

B/P1 runs the layouts in the same order as B/P0. If either phase stops early:

- ΔA uses only the layouts with both A/P0 and A/P1 complete;
- ΔB uses only the layouts with both B/P0 and B/P1 scored. A safety abort is scored, so it stays in ΔB; a crash is not.

Incomplete layouts are listed, never imputed. The after phase skips any layout without a recorded A/P0 pair.

## Step 7 converter (reviewer N2)

The training-data converter must not glob run directories. It takes an explicit list of case names: those in the two collection ledgers, plus the four named step 5 cases. It must assert that no name starts with `step8_`. That keeps the smoke case (`step8_A_smoke_l2000_s0_v1`, on training layout 2000) and every test episode out of training data.

## Planned wall time

Based on measured per-episode times: A about 205 s (step 5 ledger), B about 1934 s (lift-run launch log).

| Phase  | Episodes                                                   | Estimate                                               |
| ------ | ---------------------------------------------------------- | ------------------------------------------------------ |
| smoke  | A ×1, B ×1 (300 controls)                                  | about 7 min                                            |
| before | 12 × (A + B)                                               | 12 × 2139 s ≈ 7.1 h, plus about 2 min of model hashing |
| after  | 2 canary A + 12 × (A + B) + 24 P1 re-hashes (12–60 s each) | ≈ 7.3–7.6 h, plus 0.54 h with `--b-canary`             |

## Where the rule is not followed to the letter

These are for the reviewer.

1. **A has no derived runner.** No change is needed. The check enforces byte-identity with the approved step 5 hash instead of a range list.
2. **A episode command.** It is copied with variable renames, as noted above.
   - The A server gets `timeout -k 20 900` (the step 5 launcher had no `-k`) plus the freshness check. Both are stricter.
   - A's PYTHONPATH is set to exactly `$root/RoboLab`. The step 5 launcher appended any inherited value, which was empty.
3. **P0 load check.** It is invoked with `"$POLICY_SEED"` (always 0) where the template used `"$seed"`. The body is byte-identical.
4. **Lock size.** The lock carries all 19 frozen files, not 18: `run_painted_dev.py` is kept as the diff reference.
5. **New halt rules.** The third-failure halt and the after phase's dependence on a complete before phase are new. The dev launcher continued indefinitely.
6. **Clean-tree check.** The untracked check runs in every phase, not only the after phase. If it fails, any stray file has to be removed (with the reviewer's knowledge) before a launch.
7. **Runner location.** The B runner runs from `bowl_step6/` (new file, B1). The stale `bowl_step8/` copy is left in place, unreferenced, pending a deletion decision.
8. **Copied blocks, corrected.** Line 70 is parameterized, and line 98 is replaced by the classifier. Both are listed above.
9. **Abort classification and its stop trigger are new.** Safety aborts never count toward the 3-failure halt. Instead, 3 consecutive B aborts of the same kind halt the phase for Miodrag, declared above before any run.
10. **Drift canary.** Both `compare_canary.py` calls carry `|| true` (N3), so a comparison error never stops the after phase.
11. **Not checked here.** Nothing was executed against Isaac. Startup, registration of the `layout` branch with the 4000–4011 seeds, and the 300-control B path are first exercised by the `smoke` phase.
