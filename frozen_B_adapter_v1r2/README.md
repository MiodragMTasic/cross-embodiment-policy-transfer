# Robot B adapter: frozen v1, record revision 2

**This record supersedes `../frozen_B_adapter_v1/README.md`.** The 19 frozen files are byte-identical to v1 (same `lock-relative.sha256`, 19/19 OK), so the adapter itself has not changed. Revision 2 corrects the description after shadow-reviewer's step 6 verdict (CHANGES REQUIRED, R1–R5; STEP_REVIEW_LOG.md). The sealed v1 folder is kept as it was.

- **Frozen:** 23 October 2026 at 18:50 UTC, by Miodrag's decision (EXPERIMENT_REPORT §22).
- **What it is:** hand-written; it has no trained weights.
- **Where it applies:** step 8 runs B through exactly these files, settings and pins, for P0 and P1 alike. Step 7 is A-only and does not involve B.
- **Changing it:** any change needs a new version and the reviewer's OK, and is reported.

## What it is

The adapter makes a Franka-trained policy (GR00T N1.7 DROID, P0) drive a Kinova Gen3 with a Robotiq 2F-85 in simulation. The configuration is `run_painted_dev.py --arm painted --variant lift`. Each control works as follows.

**What the policy is shown.**

1. **Arm state.** B's measured joints give its canonical gripping-frame pose in the world. Local IK on a virtual Franka (A) solves for that pose, seeded from the previous solution. P0 receives that virtual A's joint angles and its 9-D end-effector pose.
2. **Gripper state.** B's measured `robotiq_85_left_knuckle_joint` angle / 0.8, clipped to [0, 1].
3. **Exterior camera: cross-painted.** B's visuals are hidden, and a Franka is rendered at the virtual-A joints with the current closure.
4. **Wrist camera: not painted.** It is remounted on B to match A's camera pose relative to the canonical gripping frame.

**What B is told.**

1. **Arm targets.** P0's A joint targets go through A's forward kinematics to the canonical world pose, then into B's root frame. Local IK, seeded from B's measured joints, gives B's 7 joint targets.
2. **Rate limit.** Each command moves at most 0.04 rad per control from the previous command (`run_painted_dev.py:281`).
3. **Safety abort.** The run aborts if a measured joint leaves its soft limits by more than 0.03 rad (`:199`).
4. **Timing.** Each P0 action is held for 12 controls at 15 Hz. The budget is 750 actions, which is 9000 controls or 600 s of simulation; this is A's 50 s budget. The open-loop horizon is 8, so P0 replans every 96 controls.
5. **Gripper.** P0's open/close command goes to the Robotiq. The native actuator gains are asserted unchanged (stiffness 100, damping 5, effort 50).
6. **Lift.** P0 must hold its close command for more than 24 controls, and the close must have started with the banana touching neither the bowl nor inside it. Then B's commanded tool target rises 15 mm in world +Z, ramped over 24 controls. It ramps back down when P0 opens.

**Fixed geometry.** The canonical gripping frame is the open-pad distal frame. Its origin is 0.149100 m along A's raw +X and 0.149346 m along B's raw +Z. The forward residual is at most 0.0626 mm (`frame_proposal.json`, `ik_results.json`).

## Simulator ground truth it uses (disclosed)

1. **Painting renders from simulator state, not inpainting.**
   - The simulator renders the scene with B's visuals hidden and a Franka model posed at the virtual-A joints.
   - So P0 also sees whatever B's body would occlude, using the renderer's exact knowledge of what is behind B. A real camera system would have to inpaint that region.
2. **Lift eligibility.** It uses banana–bowl contact and the geometric bowl predicate (RoboLab `object_in_container`), both from the previous control.
3. **Episode termination.** The episode ends on a v4-gate success, and that gate reads simulator object and contact state. This applies to A and B alike, and it ends the episode; it does not steer it.
4. **Wrist camera placement.** It is remounted to A's relative pose, a placement a real Kinova would need to have built.
5. **Everything else** that feeds control uses only B's own measured joints and gripper angle. The reviewer checked `run_painted_dev.py` and found no other object or contact state read into control.

## Tuning history (step 6, 21–23 October)

Every change made before the freeze, in order. The full records are in EXPERIMENT_REPORT §19–21 and BOWL_STEP6_PROGRESS_2026-10-21.md.

1. **Fixed frames and camera remount.** Derived on the CPU and checked: 32/32 forward solves, 32/32 reverse solves and 16/16 equivalence checks. Native B tracking passed (worst endpoint error 1.197 mm).
2. **First B/P0 run (v1).** The camera check failed at reset. The v2 run completed but the grasp failed: there was contact and no lift.
3. **12× action hold.** B's first gripper closing response is about 10× slower than A's. With the 12× hold, the recorded A motion replayed on B grasped and lifted the banana, but it slipped while closed. A live run grasped, lifted and released, but onto the table.
4. **Matched shadow test.** The pre-registered result was negative. The exploratory result pointed at the exterior image: P0 plans from seeing a Kinova instead of a Franka. The wrist camera and state mapping stayed at the noise floor.
5. **Exterior cross-painting.**
   - **Probe v1 collision:** the visual Franka copy's colliders pushed B out of its joint limits. The fix was to park the copy 20 m above the cell whenever physics runs.
   - **Painted check:** passed.
   - **Live comparison (3 matched pairs):** painted runs lifted and carried the banana in 3/3; unpainted runs lifted it in 0/3. All three painted runs slipped closed onto the bowl, and the v4 gate scored 0/6 deliberate releases.
6. **Lift of 15 mm.**
   - **The `lift_ok` eligibility rule** (no lift for a close that begins on or in the bowl) was added after the reviewer's B1 finding, before the run.
   - **Development round, one episode (layout l2002, seed 0):** P0 opened over the bowl and the banana dropped in and stayed. The reviewer passed it; the v4 gate failed it (centre 66 mm above the rim at the open).
7. **Grip-gain variant (×5).** Authorized and reviewed, but never run. It is not part of the frozen adapter.

## Step 6 outcome: useful B behaviour demonstrated, not established as a rate

- **Grasp, lift and carry to the bowl.** 4/4 painted episodes (3 live, 1 lift), on 2 in-sample scenes. Unpainted: 0/3.
- **Deliberate release.** 1/1 lift episode. Painted without the lift: 0/3.
  - That release is reviewer PASS and v4 FAIL. It counts as a win under Miodrag's rule that a drop counts.
- **Order of events, disclosed.**
  - Before the run, Miodrag said in chat that a placement into the bowl would count as success ("if it places it in the bowl, that is success").
  - The formal rule (a drop that lands and stays counts, to be automated as v5) was fixed at about 18:46 UTC, after the episode had been seen.
- **Open feasibility issue: release reliability on B.** Step 8 measures it on 12 unseen layouts. This satisfies the checklist's "or the feasibility issue reported" branch.

## Pins that are not files

Every step 8 supervisor must assert these, as `run_paint_dev_lift1.sh:34–37` and its model checks do.

| Pin                                                                                                          | Value                                                  | How it is checked                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| RoboLab (Kinova config and gripper gains, contact sensors, `object_in_container`, task registration, assets) | HEAD `b6cfcf6ccd2113c4cd1216d35b33024a7d5691da`        | `git rev-parse HEAD` plus `git status --porcelain --untracked-files=no` must be empty. Untracked files are not checked.                        |
| Isaac-GR00T                                                                                                  | HEAD `dac54b313160c201c450f97df0fd74ebd379c892`        | Same as RoboLab.                                                                                                                               |
| P0 and Cosmos model files                                                                                    | `models/model-manifest.json` (32 files)                | sha256 of every file before launch                                                                                                             |
| Isaac Sim / Isaac Lab                                                                                        | `isaacsim` 5.1.0.0, `isaaclab` 2.3.2.post1 (installed) | **Not asserted by the development launcher.** This corrects the reviewer's note. Step 8 supervisors add an `importlib.metadata` version check. |

## Frozen files and their roles

| Role                                                                  | Files                                                                                                                                                                                                                                                        |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Adapter** (changes what P0 sees or what B does)                     | `experiments/bowl_step6/run_painted_dev.py`, `paint_A.py`, `local_ik.py`, `rigid_chain.py`; `runs/step6_fixed_frames_cpu_v1/frame_proposal.json`, `ik_results.json`; `runs/step6_A_tracking_v3/inventory.json`; `.../kinematics_export_v1/joint_graphs.json` |
| **Scene inputs** (B's initial object poses and the policy-seed check) | the three `strict-metric.json` files. Their `initial_poses_world` is written into B's scene and asserted (`run_painted_dev.py:105, 129–131`). Step 8 replaces them with A's step 8 records under the rule below.                                             |
| **Policy serving and identity**                                       | `experiments/bowl_step5/run_policy.py` (the P0 server: seeding, checkpoint selection, load record); `models/model-manifest.json` (P0's pins); `RoboLab/policies/gr00t/client.py` (the upstream client, unmodified)                                           |
| **Measurement and termination**                                       | `experiments/bowl_step5/release_gate_v4.py`, `release_gate_v3.py`, `rim_support.py`, `stability.py`. The online v4 gate both scores and ends the episode on success.                                                                                         |
| **Development launcher** (template only; not run again)               | `experiments/bowl_step6/run_paint_dev_lift1.sh`. Its infrastructure checks are the template for step 8.                                                                                                                                                      |

## Rule for step 8

**Runner.** `step8_B_runner.py` is a copy of the frozen `run_painted_dev.py`.

- A script diffs the copy against the frozen file. Every hunk must fall inside these line ranges of the frozen file; anything outside them fails the check:

| Lines    | Allowed change                                                                                                                                                                                               |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 24–28    | The argument parser: `--scene` gains a `layout` choice plus a `--layout-seed` argument; `--variant` is restricted to `lift`; one `--policy {P0,P1}` label argument is added.                                 |
| 31       | The POLICY_SEED check becomes `== 0`.                                                                                                                                                                        |
| 32–34    | The case-name pattern (`step8_B_{P0\|P1}_l{seed}_s0_v1`) and an optional declared non-scoring smoke case.                                                                                                    |
| 98–105   | One `layout` branch, identical to the `l2002` branch except `layout_seed=args.layout_seed`, with `Aref` set to A/P0's step 8 output for that layout. The `Ametric` layout-seed and policy-seed asserts stay. |
| 164, 322 | The `scope` text; new keys may be added to these dictionaries, but existing keys and values stay.                                                                                                            |
| 315, 323 | The video label and the print tag (`DEV` → `STEP8`).                                                                                                                                                         |

- All 18 other frozen files must still match `lock-relative.sha256` on the VM before every episode.

**Layouts and env.** Layouts 4000–4011 on `BananaInBowlUniformInitPose10cmTask`:

- A uses its step 5 registration.
- B uses `BananaInBowlUniformInitPose10cmTaskKinovaJointPosition`, registered exactly as in the `l2002` branch.

**Initial poses and seed.**

- **Seed.** Every step 8 episode uses `POLICY_SEED=0`: A/P0, A/P1, B/P0 and B/P1.
- **Order.**
  1. A/P0 runs first on each layout.
  2. A/P1 runs next. Its `initial_poses_world` must equal A/P0's for the same layout to within 1e-6, or the batch halts.
  3. Both B conditions read A/P0's `strict-metric.json` for that layout. Its sha256 is recorded before the B episode, and B asserts the poses as the frozen code does.
  4. B/P0 and B/P1 alternate layout by layout, so a batch that stops early still leaves complete pairs.

**Stop rule and scoring.**

- v4 stop-on-success stays the episode stop rule, unchanged, for A and B.
- v5 is computed afterwards from each saved trace and is never wired into termination.
- Any episode where v4 passes and v5 fails goes to the reviewer.

**Serving P1.**

- `run_policy.py` stays byte-identical. It serves a non-P0 checkpoint only through `POLICY_CHECKPOINT` and labels it "disposable copy".
- P1's identity therefore comes from a pin: at the end of step 7, the selected checkpoint's files are hashed into a reviewed and sealed `P1-manifest.json`.
- Before every P1 episode, the supervisor checks all of the following against that pin:
  - it re-hashes every P1 file;
  - `model-loading.json` names that checkpoint path, with seed 0, empty missing and unexpected key lists, and the same Cosmos revision.
- Results say "P1" on the strength of the pin, not the server's label.

**Supervisor.** It is derived from `run_paint_dev_lift1.sh`, with these infrastructure checks kept byte-identical:

- the repository pins and clean-tree checks;
- P0/Cosmos hashing;
- `sha256sum -c` of the lock;
- the GPU-release wait;
- the timeouts and SIGTERM handling.

It may change only:

- the case list and order;
- the P1 pin and load assertions;
- the A-to-B pose hand-off;
- the added Isaac version check.

The reviewer approves the supervisor diff before step 8.
