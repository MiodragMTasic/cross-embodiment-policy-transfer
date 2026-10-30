# Step 7 collection driver (A only)

Built 23 October 2026, 19:10 UTC. **Nothing has run.** It follows STEP7_PLAN_2026-10-23.md §1 and the reviewer conditions in EXPERIMENT_REPORT §22.4.

**Before it launches, all of these must hold:**

- the reviewer's OK on this driver;
- Miodrag's explicit spend approval (S1);
- the v5.1 fixes (V1–V3) are in;
- the lead's ordering: step 8's "before" half runs first.

| File                                            | sha256                                                             |
| ----------------------------------------------- | ------------------------------------------------------------------ |
| `run_training_collection_v1.py` (batch driver)  | `f63501e8300f867ae4d12f0ff76b2f194e475a3cec510912921b1359d6a29a0f` |
| `run_collect_v1.sh` (supervisor)                | `a640a89276304dcad2ed15a3c32a4ea9a6abcebceeba1bfa8659592f8679ca49` |
| `step7-collect-source-lock.sha256` (12 entries) | `8036de60ba04fd61e907d0daf3668c173114184fe362747b2642dc92511a247d` |

All three are copied to the VM at `~/robot-transfer/bowl/experiments/bowl_step7/`, a new directory, and `sha256sum -c` there gives 12/12 OK.

## Commands (on the VM)

```
nohup bash ~/robot-transfer/bowl/experiments/bowl_step7/run_collect_v1.sh first >/dev/null 2>&1 &
```

- **Logs:** the supervisor writes to `runs/step7_collect_v1_first-supervisor/`, the ledger is `runs/training-collection-v1.jsonl`, and each episode goes to `runs/step7_collect_layout{seed}_s0_v1/`.
- **Seed-1 pass (D3):** `run_collect_v1.sh seed1`. It runs layouts 2000–2031 with policy seed 1 into a separate ledger.
  - **Allowed only by explicit decision**, after the first pass has been v5-scored and reviewed with fewer than 6 wins.
  - **Refuses to start** unless the first ledger ended with `batch_finished`.
  - **Never automatic and never repeated:** each ledger must not already exist.

## What runs

**The episode runner is unchanged.** `experiments/bowl_step5/run_controlled_case_v4_registered.sh` and `run_controlled_episode_v4_registered.py` run with the same task (`BananaInBowlUniformInitPose10cmTask`), P0 only, as follows:

- the runner unsets `POLICY_CHECKPOINT`;
- it asserts the P0 and Cosmos identity, with all four `loading_info` lists empty;
- it bounds the server at 900 s and the episode at 600 s;
- it audits the finished capture and both videos.

**The batch driver** is `run_training_feasibility_v4.py`, copied, with these changes:

- layouts `range(2004, 2032)` and policy seed 0, or the explicit `seed1` mode;
- case names `step7_collect_layout{seed}_s{policy}_v1`;
- a new ledger, with the never-repeat guard kept;
- a 60 s GPU-release wait before each case, as in step 6;
- a second source check against `step7-collect-source-lock.sha256`;
- the ledger key `passed_v4`, which names v4 explicitly.

`verify_sources()` is byte-identical. It checks the step 5 lock (32 files and both commits) and `git status --porcelain` including untracked files, before and after every case.

**The supervisor** carries over step 6's infrastructure checks unchanged (`run_paint_dev_lift1.sh`):

- a supervisor directory that must not already exist, and an idle GPU at start;
- the tee'd log, and an EXIT trap that writes the exit code and `gpu-after.txt`;
- `sha256sum -c` of the lock;
- the RoboLab and Isaac-GR00T HEAD pins, with clean tracked trees (`--untracked-files=no`);
- hashes of all 32 pinned P0 and Cosmos model files.

It adds an `importlib.metadata` check that `isaacsim` is 5.1.0.0 and `isaaclab` is 2.3.2.post1, and an overall batch timeout of 54,000 s.

**SIGTERM.** No new handler was added. Each case is bounded inside the unchanged runner, and the runner's EXIT trap stops its policy server. A signal to the supervisor does not interrupt a case mid-episode.

## Layout range guard (reviewer condition R3)

- **The driver asserts that every layout it runs is in 2000–2031.** It checks the whole list before starting, and each layout again inside the loop. A validation layout (3000–3007) or a test layout (4000–4023) can therefore never enter this ledger.
- **The converter must keep the same assert.** The existing step 5 converter (`convert_reviewed_training_dataset.py`) already asserts `2000 <= layout_seed <= 2031`, but it also hard-codes step 5 case names (`step5_training_layout{seed}_v4`). Step 7 needs a derived converter that:
  - accepts `step7_collect_layout{seed}_s{policy}_v1`;
  - keeps the 2000–2031 assert;
  - applies the truncation rule below.
    That converter is **not built yet** and needs its own review.
- **The validation driver (3000–3007), still to be built, must assert its own range** (3000–3007 only) in the same way, so that no training or test layout can enter selection.
- **This driver never touches 3000–3007 or 4000–4011.**

## S2: reusing layouts 2000–2003

**Result: MATCH, so 2000–2003 are reused and not recollected.** Checked at 19:08–19:10 UTC.

- **Identical records.** The four runs' `source-hashes.txt` files are identical to one another (sha256 `837329ae…`), as are their copied locks (`027a200b…`).
- **Recorded files unchanged.** All 32 files listed in those records still hash-match on the VM.
- **Full step 5 lock unchanged.** It includes the `.sh` launcher, which `source-hashes.txt` does not cover, and the RoboLab files. Every entry matches on the VM, and both HEADs match.
- **Clean trees.** `git status --porcelain`, including untracked files, is empty in both repositories.
- **Isaac versions** match: isaacsim 5.1.0.0 and isaaclab 2.3.2.post1.

**One difference, not a runner change.** `experiments/bowl_step5/` now holds three more `.py` files than during step 5: the dataset-proof tools `convert_reviewed_training_dataset.py`, `export_training_capture.py` and `verify_training_capture_all17.py`. New runs' `source-hashes.txt` will list them (31 `.py` files instead of 28). None is imported by the runner.

## Expected time

**28 episodes × about 205 s = about 1.6 h.** This is measured: the step 5 ledger shows 750-step runs of 203, 207 and 206 s, and the early-stopped case took 108 s. Output is about 3–7 GB (estimated).

## Scoring and demos, after the batch (nothing in the runner changes)

**v4 stop-on-success stays the episode stop rule.**

- A v4 pass ends the episode at its stable success.
- Every other episode runs its 750 steps, with P0 still acting.

**Every attempt is scored offline with v5.1** (after V1–V3) from the saved `strict-metric.json` trace:

- **Primary reading:** "first rest".
- **"At end"** is reported alongside it.

**Review set.** The reviewer checks every v5 pass, plus every v5 fail that ends supported by the bowl or had an eligible open (V2).

**Demo truncation.**

- A win that passes v5 but not v4 is cut at its v5 win control before conversion, so it ends where v4-stopped demos end.
- v4-stopped wins already end there.
- Winning episodes keep their earlier failed sub-attempts, such as a first drop that missed; this is disclosed.

Only wins enter training, and every attempt stays in the ledger with its denominator.

## Abort procedure (reviewer ruling (b), 23 October)

**Launch.** Launch under `setsid`, so that the batch is its own process group:

```
setsid nohup bash ~/robot-transfer/bowl/experiments/bowl_step7/run_collect_v1.sh first >/dev/null 2>&1 < /dev/null &
```

Record the group id with `ps -o pgid= -p <supervisor pid>`.

**Abort.**
1. Signal the whole group: `kill -TERM -<pgid>`. Never signal only the supervisor bash; that leaves the driver running headless, and its next print hits a dead tee pipe.
2. Wait 60 s. If anything in the group is still alive, `kill -KILL -<pgid>`.
3. Confirm that `nvidia-smi --query-compute-apps=pid --format=csv,noheader` is empty and that no Isaac or `run_policy` process remains (`pgrep -af 'isaac|run_policy|run_controlled'`).

**After an abort.**
- The interrupted case shows in the ledger as `started` with no `finished`. Python dies on SIGTERM without writing `halted_for_review`, which is accepted. That case is never rerun under the same name.
- Driver and CUDA versions are logged outside the locked supervisor, in the lead's launch record (`nvidia-smi --query-gpu=driver_version`, `nvidia-smi | grep 'CUDA Version'`).
