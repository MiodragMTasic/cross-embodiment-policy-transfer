# Step 7 dataset converter

Built 23 October 2026, 20:05 UTC; guards C-a to C-c added at 20:09 UTC after the reviewer's pass. It is offline and CPU-only, and it never overwrites anything. It turns reviewer-approved Robot A / P0 wins into one GR00T LeRobot-v2 dataset for the P1 fine-tune (STEP7_PLAN §1–3). Nothing has been trained, and no real dataset has been built yet: the only dataset so far is a scratch validation copy.

## Files

| File                          | Runs in                                      | Does                                                                                                                                            |
| ----------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `convert_step7.py candidates` | `RoboLab/.venv`                              | Scores the explicit case set with the locked v5.1. Writes the candidate list and an approved-wins template with every verdict set to `PENDING`. |
| `convert_step7.py export`     | `RoboLab/.venv` (h5py)                       | Writes one `.npz` per approved win, cut at the win. Also writes `export-manifest.json` with every exclusion and its reason.                     |
| `convert_step7.py build`      | `Isaac-GR00T/.venv`                          | Builds the LeRobot v2 dataset (parquet plus lossless RGB H264), then runs `gr00t.data.stats` with P0's modality contract.                       |
| `verify_step7_dataset.py`     | `Isaac-GR00T/.venv`, `CUDA_VISIBLE_DEVICES=` | Runs the CPU loader, processor and collator check (below).                                                                                      |
| `test_convert_step7.py`       | local python + numpy                         | Unit tests for the input guards and the cut mapping, including the real A 2001 and A 2002 traces.                                               |

On the VM, run every step from `~/robot-transfer/bowl` with `PYTHONDONTWRITEBYTECODE=1` and `CUDA_VISIBLE_DEVICES=` set. The scorer is imported from `experiments/scoring_v5/` and checked against the R1 hashes of its whole import chain: v5.1 5e900d6e, v5 7bba5a40, v4 1b249ae1, v3 8a5be7d4, stability 25971d4d, rim_support 97d9e833.

## Inputs: explicit only (reviewer N2, S2, D3, D4)

- **Case names** come from two places only:
  - the collection ledger `runs/training-collection-v1.jsonl`: layouts 2004–2031, seed 0, rows with stage `finished` and `infrastructure_ok`;
  - the optional D3 ledger `runs/training-collection-v1-seed1.jsonl`: layouts 2000–2031, seed 1.

  Added to these are the four named step 5 cases `step5_training_layout2000_v4` … `2003_v4`, reused under D4. Nothing is globbed.

- **Every name is checked:**
  - it never starts with `step8_`, so `step8_A_smoke_l2000_s0_v1` can't enter;
  - it matches `step7_collect_layout####_s[01]_v1` or a named step 5 case;
  - its layout is in 2000–2031;
  - the first ledger may not hold 2000–2003 or seed 1;
  - the ledger fields agree with the name;
  - each case has exactly one `started` row, followed by one terminal row.
- **Every converted case is checked:**
  - `source-hashes.txt` records the approved runner `run_controlled_episode_v4_registered.py` a3191192 (S2);
  - capture-integrity, the metric and the HDF5 attributes agree on layout, policy seed and task;
  - capture-integrity's `trace_rows` equals the metric's trace length (C-c);
  - the HDF5 holds exactly the 10 expected arrays, with every row executed and verified.
- **Approved wins** (`step7-approved-wins-v1`) list, per entry, the case, its win control, a verdict (`OK`, `REJECT` or `PENDING`) and the reviewer. The file must name the locked v5.1 hash.
  - Only `OK` entries are converted.
  - The locked v5.1 must reproduce each listed win control; any mismatch stops the export.
  - A case that isn't in the explicit set is an error.
  - **No selection by omission (C-a).** Export scores every case in the explicit set first. Every v5.1 pass must have a final verdict, `OK` or `REJECT`. A pass that is `PENDING` or missing from the file stops the export before anything is written, and the error lists those cases. This also fixes the D3 count: under 6 means under 6 `OK` among all passes, not among a chosen subset.
- **Whole batches only (C-b).** Export requires the first ledger to end with `batch_finished` (mode `first`), and the seed-1 ledger, if given, to end with `batch_finished` (mode `seed1`). `candidates` still reads a running ledger, so snapshots work.

## The cut

v4 trace rows are controls 1..n, and HDF5 row _i_ is trace row _i_. A win at control _S_ keeps rows 0..*S*−1, which is _S_ rows. Case 2002 is the reference: v4 stopped on its win, so it keeps 301 of 301.

A v5-only win (live v4 ran on to 750) loses everything after the win. The mapping is proven per episode at export:

- **Whole trace:**
  - the HDF5 gripper state equals the trace's `pre.closure` row for row: max error ≤ 2.6e-7 at shift 0, against ≥ 0.62 at shifts ±1 and ±2;
  - the executed gripper column equals the trace commands exactly.
- **At the cut row:**
  - `trace[row].step` equals the win control;
  - the gripper state matches the closure;
  - the command matches.

The npz keeps the trace step, command and closure for the kept rows, so the verifier can recheck the cut without the HDF5.

## Loader check (`verify_step7_dataset.py`)

Using the pinned `LeRobotEpisodeLoader`, P0's modality contract, and no model weights:

- **Counts:** the episode count and total frames match.
- **Per episode:**
  - the length equals the npz rows, `episodes.jsonl`, and the win control;
  - all 17 state dimensions and all 17 action dimensions equal the export, on every row, and are finite;
  - every RGB frame of both cameras decodes bit-exact;
  - the last row is the win-control row.
- **Processor and collator:** for each episode, 40-row samples at rows 0, the middle and *n*−40 go through the training processor and collator.
  - All 17 normalised action and state dimensions are recomputed independently against P0's own `statistics.json`, using the method of `bowl_step5/verify_training_capture_all17.py`.
  - The (40, 132) action padding, the mask and the batch are all finite.

## What is excluded, and why

- **Collection cases that are not finished.** Running, halted or interrupted cases are listed with their ledger stage.
- **v5.1 fails and unlisted passes.** These are excluded as "no approved win".
- **Approved entries with verdict REJECT or PENDING.**
- **All step 8, validation (3000–3007) and test (4000–4023) cases.** These can't be named.
- **The last 39 rows of each episode as anchors.** This is the same limit as in step 5: their actions still appear inside earlier 40-row windows.

## Not done here

- No reviewer approval file exists yet. The validation run used a fixture marking all 7 current v5.1 candidates `OK`, labelled "NOT a review". It lives in `/tmp/step7-convert-check.tkQc/` on the VM and is never for training.
- The real dataset waits for:
  - the collection to finish;
  - the reviewer's verdict on each v5.1 pass;
  - the decision on whether D3 is triggered (fewer than 6 OK wins).
- A failed export leaves its partial output directory behind. Rerun with a new `--out`; nothing is overwritten.
