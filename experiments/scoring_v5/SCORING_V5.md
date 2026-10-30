# v5 win check: offline scorer

Written 23 October 2026 for steps 7 and 8. It has not been reviewed yet.

## Rule

The rule is Miodrag's, from EXPERIMENT_REPORT §22. A win needs three things:

- the policy commands the gripper open while the banana is held and lifted;
- the banana lands on the original bowl (rim counts) directly from that release;
- it then settles there.

Both a set-down and a drop count. A slip from a closed gripper never counts.

## How it is used

v5 is an offline scorer only. It replays a saved trace after the episode has ended. v4 stop-on-success remains the episode stop rule for A and B, and the runners stay byte-identical (reviewer constraint).

## Changes from v4 (`release_gate_v5.py`, a subclass of `ReleaseGateV4`)

1. **The open edge ignores geometry.** It no longer requires the banana to be inside the bowl's geometric region (`near_bowl`). A drop from above therefore counts. v4 rejected exactly these opens as `ineligible_open_command`: A layouts 1000 retry 1, 1005 and 2001, and the B lift run.
2. **Landing.** After detachment, the banana's first support contact must be the bowl. A bowl contact is logged as `landed_on_bowl`. A table contact with no bowl contact invalidates the release (`landed_off_bowl`).
3. **Falling off.** After landing, any control with table contact and no bowl contact invalidates (`left_bowl`). This replaces v4's geometric escape.
4. **What is unchanged from v4, with the same thresholds:**
   - Arming: closed, fingertip contact, no table or bowl contact, AABB clearance ≥ 1 cm, held for 0.2 s.
   - Open eligibility: armed, contact, and closure > 0.2.
   - Release timeout: 0.5 s for A and 2.1 s for B, as in the B port of v4.
   - Measured opening: a closure drop of at least 0.05.
   - Regrasp and recontact revocation.
   - The 1 s stability window: bowl support and (upstream predicate or rim support), ≤ 0.03 m/s and ≤ 0.3 rad/s, 16 samples at 15 Hz.
5. **The `detached_near_bowl` event name** is inherited from v3. In v5 it only means "detached", because location is no longer checked there.

## Signals

Every signal comes from the saved per-control `pre` and `post` samples:

- `contact` (fingertips), `bowl_contact`, `table_contact` (|force| > 0.1 N), `closure`, `clearance_m`, `bowl_support`, `rim_support`;
- the P0 gripper command;
- the banana's linear and angular speed, and the upstream predicate.

These exist for A (strict-metric v2, v3 and v4) and for B (`gate_trace.jsonl`). A v2 and v3 captures predate the recorded support signals. Those signals are recomputed with `rim_support.support_signals` and the reviewed sensor order, exactly as `replay_supported_v4.py` does.

The reported position at the open is diagnostic only and is not part of the rule:

- `centre_above_bowl_top_m`: the banana AABB centre minus the top of the bowl AABB;
- `dxy_root_m`: the horizontal distance between the root origins.

## Two readings of "stays there"

The reviewer asked for both:

- **first rest:** the verdict is fixed when the first 1 s stable window completes, and nothing after it matters;
- **at end:** the first-rest pass, plus a supported placement (`bowl_support` and (upstream predicate or `rim_support`)) at the last recorded control.

**Caveat on the at-end reading.** Episodes stopped by v4 end at their v4 success, so for them "at end" is the moment of rest. v5-only wins keep running with P0 acting: up to 750 controls on A and 9,000 on B. The at-end reading is therefore not symmetric between the two.

**After the first rest, in the 7 passes:**

| Episode                      | After its first rest                                                                                            | End                   |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------- |
| B lift run                   | 3 fingertip-contact controls, 3 close commands, one `regrasp_after_release` (412.67 s); never lost bowl contact | In the bowl           |
| A base s1                    | 2 table-only controls at 15.6–15.67 s, the instant v4 logged `geometric_escape`                                 | Supported on the bowl |
| A 1000 retry 1, A base s0 v2 | A regrasp                                                                                                       | Supported on the bowl |
| A 1005, A 2001, A 2002       | Nothing                                                                                                         | Supported on the bowl |

## Replay fidelity

Fed to the original gates, the replay reproduces every recorded result: identical events and first-pass control, to 1e-6 s, up to the first pass. That holds on all 22 traces:

- against recorded v4 for the v4 runs and B;
- against the reviewed v4 retrospective for A base s0 v2, base s0 v3 and base s1 v3;
- against recorded v3 for the other v3 pilots.

Pre-action time is the previous control's recorded post time, as in the runners. B rows start at 2/15 s because of its initialization step.

## Validation against every adjudicated trace

`python3 validate_v5.py` writes `validation-v5.json`. Agreement means v5 matches the human verdict under both readings.

| Episode                                                       | Human | v4   | v5 first rest | v5 at end | Agree | Open, win or fail reason                                                                                                |
| ------------------------------------------------------------- | ----- | ---- | ------------- | --------- | ----- | ----------------------------------------------------------------------------------------------------------------------- |
| A base s0 v3                                                  | fail  | fail | fail          | fail      | yes   | no_open_while_held                                                                                                      |
| A base s1 v3                                                  | pass  | pass | pass          | pass      | yes   | open 140, win 169, 43 mm above rim, dxy 8 mm                                                                            |
| A layout 1000 retry1                                          | pass  | fail | pass          | pass      | yes   | open 128, win 152, 62 mm above rim, dxy 10 mm                                                                           |
| A layout 1001                                                 | fail  | fail | fail          | fail      | yes   | never_lifted                                                                                                            |
| A layout 1002                                                 | fail  | fail | fail          | fail      | yes   | never_lifted                                                                                                            |
| A layout 1003                                                 | fail  | fail | fail          | fail      | yes   | regrasp_after_release                                                                                                   |
| A layout 1004                                                 | fail  | fail | fail          | fail      | yes   | never_lifted                                                                                                            |
| A layout 1005                                                 | pass  | fail | pass          | pass      | yes   | open 128, win 153, 75 mm above rim, dxy 14 mm                                                                           |
| A layout 1006                                                 | fail  | fail | fail          | fail      | yes   | lost_contact_before_open                                                                                                |
| A layout 1007                                                 | fail  | fail | fail          | fail      | yes   | lost_contact_before_open (slip; ended on the bowl)                                                                      |
| A layout 2000                                                 | fail  | fail | fail          | fail      | yes   | never_lifted                                                                                                            |
| A layout 2001                                                 | pass  | fail | pass          | pass      | yes   | open 285, win 308, 66 mm above rim, dxy 61 mm (the first drop at 8.5 s hit the bowl and fell to the table: `left_bowl`) |
| A layout 2002                                                 | pass  | pass | pass          | pass      | yes   | open 280, win 301, 44 mm above rim, dxy 17 mm                                                                           |
| A layout 2003                                                 | fail  | fail | fail          | fail      | yes   | no_open_while_held                                                                                                      |
| B live l2002 s0 painted                                       | fail  | fail | fail          | fail      | yes   | lost_contact_before_open                                                                                                |
| B live l2002 s0 unpainted                                     | fail  | fail | fail          | fail      | yes   | never_lifted                                                                                                            |
| B live base s0 painted                                        | fail  | fail | fail          | fail      | yes   | lost_contact_before_open                                                                                                |
| B live base s0 unpainted                                      | fail  | fail | fail          | fail      | yes   | never_lifted                                                                                                            |
| B live base s1 painted                                        | fail  | fail | fail          | fail      | yes   | lost_contact_before_open                                                                                                |
| B live base s1 unpainted                                      | fail  | fail | fail          | fail      | yes   | never_lifted                                                                                                            |
| B dev lift l2002 s0                                           | pass  | fail | pass          | pass      | yes   | open 1525, win 1554, 66 mm above rim, dxy 13 mm                                                                         |
| A base s0 v2 (no explicit physical verdict; information only) | —     | pass | pass          | pass      | —     | open 126, win 162, 48 mm above rim, dxy 9 mm                                                                            |

**Result:** 21 of 21 agree under both readings. v4 agreed on 17 of 21; it missed the four drop-from-above wins.

**Not scorable.** These traces lack the gripper command and contact signals, so they were excluded:

- step 5 `base_s0_v1` (stable-upstream v1 schema);
- step 6 B diagnostics `P0_B_v2`, `P0_B_temporal_v1` (reviewed as a deliberate release onto the tabletop), `P0_B_feedback_v1` and `temporal_B_v1`.

## Limitations

- **Not independent.** The open-edge change was motivated by these same traces, so the 21/21 shows consistency, not held-out accuracy. All of them are development or pilot data.
- **`landed_off_bowl` has no real example.** It never fired on real data and is exercised only by `test_release_gate_v5.py`. `left_bowl` fired twice, in A 2001's first drop and after A base s1's rest.
- **Contact flicker.** Contacts are sampled per control (15 Hz) with a 0.1 N threshold, and no flicker debounce was added. A single table-only control after landing invalidates. A bounce off the table that lasts less than one control would go unseen.
- **No height cap.** The open has no height limit. Wins in the data opened 43–75 mm above the rim, and v4's region had a 0.15 m cap.
- **Release timeout differs by robot.** A uses 0.5 s and B 2.1 s, because of B's slower gripper, the same as the v4 B port.

## Files

- `release_gate_v5.py`: the gate plus the offline `score()`.
- `test_release_gate_v5.py`: 10 synthetic tests, all passing.
- `validate_v5.py`: the validation driver.
- `validation-v5.json`: the output, including each trace's sha256.
- `release_gate_v5.sha256`: the scorer hash.

## v5.1 (23 October 2026, 19:12 UTC; reviewer verdict on v5: changes required)

v5 is kept unchanged as history: `release_gate_v5.py`, sha256 `7bba5a40…`, `release_gate_v5.sha256` OK. v5.1 is the new file `release_gate_v5_1.py`, a subclass of `ReleaseGateV5`. It remains an offline scorer only, and v4 stop-on-success remains the stop rule.

**Reviewer decisions adopted:**
- the first stable rest is the primary reading, with the at-end reading reported alongside;
- training demos from v5-only wins are cut at the win control;
- there is no height cap.

### Changes

- **V1, no straddle.** The 1 s rest must be supported by the bowl alone: any table contact resets the window. The same test applies to the at-end placement.
- **V3, debounce.** `landed_off_bowl` and `left_bowl` need 3 consecutive posts (0.133 s first to last) with table contact and no bowl contact. Landing is judged on post samples only, because each pre sample repeats the previous post.
  - **What V3 lets through (N3):** a released banana may touch the table on up to 2 posts (≤ 0.133 s) before it reaches the bowl and still count as `landed_on_bowl`. Every v5.1 pass is human-checked.
- **V2, `review_flags()`.** A human must check every v5.1 pass, every case where v4 passes and v5.1 fails, and every v5.1 fail whose banana ends supported by the bowl alone or that had an eligible open.

### Effects on the 22 traces (`validate_v5_1.py` → `validation-v5_1.json`)

**V1.** No verdict and no win control changed (confirmed). It still agrees 21/21 with the human verdicts.

**V3.** It changes two event timelines and no verdict:
- **A base s1:** the post-rest blip at controls 235–236 (2 table-only posts) no longer fires `left_bowl`. The first event after the rest is now `regrasp_after_release` at 32.8 s.
- **A 2001:** the first drop's `left_bowl` moves from 8.93 s to 9.07 s, the third consecutive post.

**V2.** 13 of the 22 traces are flagged:
- 7 v5.1 passes (6 adjudicated and the information-only A base s0 v2);
- the 3 painted B slips (end on bowl);
- A 1003 (had an eligible open);
- A 1007 (ends on bowl);
- **A 2000 (ends on bowl).** This one was not in the expected list. The banana was never armed (no held lift that met v4's test), yet ended supported by the bowl alone. The human verdict is "no qualifying controlled release". Under the rule as written, it is correctly flagged.

No case had a v4 pass with a v5.1 fail.

## Step 8 stage metrics (`stages.py`)

The metrics compute identically on A strict-metric rows and on B gate_trace rows plus tracking.npz. Positions are post-control root origins.

**Attempt.** A maximal run of P0 gripper command > 0.5, the open edge that ends it, and what follows from that release. A stage counts only if every earlier stage happened first, in order, in the same attempt. The furthest stage is the maximum over attempts.

**The stages:**
- **M1 grasp:** the fingertip-contact flag on 3 consecutive closed-command controls. The flag is not the same sensor on both robots; see S3 in the R1 section.
- **Lift:** the banana root ≥ 2 cm above its position after control 1, with contact.
- **M3 over the bowl:** the banana root within **r = 0.081 m** horizontally of the bowl root, with contact. r is the bowl's outer half-width, 16.1 cm / 2, rounded up.
- **M4 open:** the edge that ends the attempt is a v5.1-eligible open.
- **Landed:** v5.1 `landed_on_bowl` from that release.
- **Win:** v5.1 first stable rest from that release.

**Measures:**
- **M2:** P0 time of the qualifying open, which is the control divided by 15 on A and by 180 on B. The win control and simulation seconds are also given. `score_step8.py` reports control/15 s on both robots instead of the recorded simulation seconds (S4).
- **M5:** `wilson()` gives a 95% interval per rate. `paired()` and `mcnemar_exact()` give the exact two-sided McNemar sign test on discordant layouts, for ΔA (P1 vs P0 on A) and ΔB.
- **Significance needs at least 6 discordant layouts, all in one direction.** p < 0.05 cannot be reached with fewer: 5–0 gives p = 0.0625 and 6–0 gives 0.03125. With 12 layouts, a real but smaller effect can come out non-significant.

**Stage table.** The controls are those of the furthest attempt.

| Episode | Human | v5.1 | Attempts | Furthest | Stage controls | M2: P0 open time / win control / sim s | Review flags |
|---|---|---|---|---|---|---|---|
| A base s0 v3 | fail | fail | 21 | over_bowl | grasp 187, lift 187, over 187 | — | — |
| A base s1 v3 | pass | pass | 11 | win | 83, 84, 106, open 140, landed 140, win 169 | 9.333 s / 169 / 11.267 s | pass |
| A 1000 retry1 | pass | pass | 6 | win | 69, 73, 91, 128, 129, 152 | 8.533 s / 152 / 10.133 s | pass |
| A 1001 | fail | fail | 14 | lift | 130, 132 | — | — |
| A 1002 | fail | fail | 3 | lift | 68, 69 | — | — |
| A 1003 | fail | fail | 22 | landed | 88, 90, 118, 144, 145 | — | eligible open |
| A 1004 | fail | fail | 0 | none | — | — | — |
| A 1005 | pass | pass | 2 | win | 83, 84, 104, 128, 129, 153 | 8.533 s / 153 / 10.2 s | pass |
| A 1006 | fail | fail | 9 | over_bowl | 67, 68, 82 | — | — |
| A 1007 | fail | fail | 11 | over_bowl | 106, 106, 139 | — | ends on bowl |
| A 2000 | fail | fail | 15 | over_bowl | 428, 433, 461 | — | ends on bowl |
| A 2001 | pass | pass | 7 | win | 208, 212, 264, 285, 285, 308 | 19.0 s / 308 / 20.533 s | pass |
| A 2002 | pass | pass | 2 | win | 217, 220, 235, 280, 280, 301 | 18.667 s / 301 / 20.067 s | pass |
| A 2003 | fail | fail | 13 | over_bowl | 684, 691, 715 | — | — |
| B live l2002 s0 painted | fail | fail | 14 | over_bowl | 1572, 1684, 1828 | — | ends on bowl |
| B live l2002 s0 unpainted | fail | fail | 1 | none | — | — | — |
| B live base s0 painted | fail | fail | 16 | over_bowl | 1248, 1252, 1444 | — | ends on bowl |
| B live base s0 unpainted | fail | fail | 1 | grasp | 2007 | — | — |
| B live base s1 painted | fail | fail | 13 | over_bowl | 652, 677, 868 | — | ends on bowl |
| B live base s1 unpainted | fail | fail | 1 | none | — | — | — |
| B dev lift l2002 s0 | pass | pass | 4 | win | 720, 772, 963, 1525, 1526, 1554 | 8.472 s / 1554 / 103.667 s | pass |
| A base s0 v2 (info) | — | pass | 13 | win | 76, 76, 95, 126, 126, 162 | 8.4 s / 162 / 10.8 s | pass |

**M5 illustration.** This uses the 3 sealed B live layouts, painted vs unpainted. It is not a step 8 result.

| Stage | Painted | Unpainted | Discordant | Exact McNemar p |
| --- | --- | --- | --- | --- |
| Lift | 3/3 [0.44, 1.00] | 0/3 [0.00, 0.56] | 3–0 | 0.25 |
| Grasp | 3/3 | 1/3 | 2–0 | 0.5 |

### Stage-metric caveats

- **Lift allows table contact.** Lift as specified does not exclude it. 10 of 18 first lift hits had table contact, meaning one end was raised. All but B live l2002 s0 painted reach a lift free of both table and bowl contact within the same attempt, within 25 controls. No lift hit had bowl contact.
- **Lift has no minimum duration.** A 1001 counts as lift on 8 rows, max 4.5 cm. Its human verdict is "no successful pickup observed".
- **The stages are descriptive and are not verdicts.** Win is the only scored outcome.

### v5.1 files (sha256)

| File | Contents | sha256 |
| --- | --- | --- |
| `release_gate_v5_1.py` | The gate, `replay()`, `score()` and `review_flags()` | `5e900d6e6f48a7aa1fed17cdaa253f86451e13dedd8e0e4059553c6d09692876` |
| `stages.py` | The stage metrics and M5 helpers | `f8c835b33240d1218df35e020d9840ec94fdb1edaef23c918061cb3e97ebd7a6` |
| `validate_v5_1.py` | The validation driver | `1934e60d00206b8f2034e2889660145acabac28813047ad63260ac9c3f22a960` |
| `test_release_gate_v5_1.py` | 26 tests: v5's 10, plus 16 run on v5.1, of which 3 are v5 cases that V1/V3 change, superseded | `629b684ea73a3acb41d06b767d38f18e47685d4f47c790cb3669ff46787e7cdc` |
| `validation-v5_1.json` | The validation output | `c57f5cf013de00e8796694173d0fe96783f24aefb0f6829f6349aa1e80a07c93` |
| `release_gate_v5_1.sha256` | The scorer and stages hashes | — |

`test_release_gate_v5_1.py` changed in the R1 package (N1, N2); its current hash is in the R1 table below.

## Stage metrics revision 2: stricter lift (23 October 2026, 19:15 UTC)

`stages.py` (sha256 `f8c835b33240d1218df35e020d9840ec94fdb1edaef23c918061cb3e97ebd7a6`) is kept unchanged as history. The new file is `stages_r2.py`.

**New lift rule:**
- the banana root is ≥ 2 cm above its position after control 1;
- it has fingertip contact, and **no table and no bowl contact**;
- this holds on **3 consecutive posts (0.133 s first to last)** on both robots. The `stages_r2.py` docstring says "0.2 s"; it is locked, so this line governs;
- all three samples are at or after the grasp control.

The stage control is the third of those samples. `over_bowl` and every later stage must come at or after it, in the same attempt.

Nothing else changes: grasp, `over_bowl` (r = 0.081 m), open, landed, win, M2 and M5. The v5.1 verdicts are untouched: all 22 match `validation-v5_1.json` in pass and win control.

### Effect (`validate_stages_r2.py` → `validation-stages-r2.json`)

**No furthest stage changed on any trace.** Stage controls changed on 18 of 22. In 15, the lift control moves later in the same attempt: by 2–9 controls, except B live base s0 painted at +61. In 3 (A base s0 v3, A 1001 and B live l2002 s0 painted), the best attempt moves to a later attempt.

| Episode | Furthest | Revision 1 controls | Revision 2 controls |
|---|---|---|---|
| A base s0 v3 | over_bowl | 187, 187, 187 | 720, 731, 731 (attempt moves; matches v4 `lift_armed` at 48.8 s) |
| A base s1 v3 | win | 83, 84, 106, 140, 140, 169 | 83, **88**, 106, 140, 140, 169 |
| A 1000 retry1 | win | 69, 73, 91, 128, 129, 152 | 69, **78**, 91, 128, 129, 152 |
| A 1001 | lift | 130, 132 | **329, 335** (attempt moves) |
| A 1002 | lift | 68, 69 | 68, **78** |
| A 1003 | landed | 88, 90, 118, 144, 145 | 88, **94**, 118, 144, 145 |
| A 1004 | none | — | — |
| A 1005 | win | 83, 84, 104, 128, 129, 153 | 83, **91**, 104, 128, 129, 153 |
| A 1006 | over_bowl | 67, 68, 82 | 67, **73**, 82 |
| A 1007 | over_bowl | 106, 106, 139 | 106, **114**, 139 |
| A 2000 | over_bowl | 428, 433, 461 | 428, **435**, 461 |
| A 2001 | win | 208, 212, 264, 285, 285, 308 | 208, **214**, 264, 285, 285, 308 |
| A 2002 | win | 217, 220, 235, 280, 280, 301 | 217, **225**, 235, 280, 280, 301 |
| A 2003 | over_bowl | 684, 691, 715 | 684, **695**, 715 |
| B live l2002 s0 painted | over_bowl | 1572, 1684, 1828 | **3879, 3987, 4227** (attempt moves) |
| B live l2002 s0 unpainted | none | — | — |
| B live base s0 painted | over_bowl | 1248, 1252, 1444 | 1248, **1313**, 1444 |
| B live base s0 unpainted | grasp | 2007 | 2007 |
| B live base s1 painted | over_bowl | 652, 677, 868 | 652, **679**, 868 |
| B live base s1 unpainted | none | — | — |
| B dev lift l2002 s0 | win | 720, 772, 963, 1525, 1526, 1554 | 720, **775**, 963, 1525, 1526, 1554 |
| A base s0 v2 (info) | win | 76, 76, 95, 126, 126, 162 | 76, **84**, 95, 126, 126, 162 |

### The two cases the reviewer named

**A 1001.** The human verdict is "no successful pickup observed".
- Revision 1's lift was a 2-sample blip at 132–133 in the attempt from control 130, and it no longer qualifies.
- A later attempt (grasp 329) has one contact-free lift run of **5 controls (0.33 s)**, the longest in the episode, with a maximum rise of 4.5 cm. So 1001 still reaches lift.
- Excluding it would need a longer hold, at least 6 samples (0.4 s), or a higher rise. Either is a further rule change, so the call is not made here.

**B live l2002 s0 painted.**
- Revision 1's attempt from control 1572 never had a contact-free lift sample around its lift control (1681–1687 all false), so it no longer counts.
- The furthest stage now comes from the attempt starting at 3877: lift 3987 on a 6-sample run, over the bowl at 4227.
- That attempt ends with `lost_contact_before_open` at 294.7 s (control ≈ 4421). That is the closed-gripper slip onto the bowl.
- The episode's longest contact-free lift run is 134 controls.
- **"Lifted and carried to the bowl" therefore still holds under the stricter rule.**

**M5 illustration**, the 3 sealed B layouts: unchanged. Grasp is 3/3 vs 1/3, lift 3/3 vs 0/3 (p = 0.25), and over_bowl 3/3 vs 0/3.

### Revision 2 files (sha256)

| File | sha256 |
| --- | --- |
| `stages_r2.py` | `4c08f687449e60ec5c3315247e69df349f93cce8ce1db6ef8d68221aadc46e56` |
| `validate_stages_r2.py` | `868a32fc9dfc74baadb7de596403d865851385c5a2c053f9c08adbf35b417389` |
| `validation-stages-r2.json` | `ea1e50025c2b30290b1f9e25ff290d3c431599990371ecbdd47932c1598f7fe6` |
| `stages_r2.sha256` | Holds the hashes of `stages_r2.py` and `release_gate_v5_1.py` |

History, unchanged: `stages.py` `f8c835b3…`, `release_gate_v5_1.py` `5e900d6e…` and `release_gate_v5.py` `7bba5a40…`.

## R1 lock package: `score_step8.py` (23 October 2026, 19:29 UTC)

The reviewer approved `release_gate_v5_1.py` (`5e900d6e…`) and `stages_r2.py` (`4c08f687…`) for locking. Neither file changed after approval; both hash checks pass. This package exists before the first step 8 test episode (R1).

### Test and doc follow-ups

- **N1.** `test_v3_counter_resets_on_invalidate`: a drop leaves the bowl after 3 table posts, the gate re-arms, and the new release's detaching post touches only the table, then the bowl. It must land. With the reset at `release_gate_v5_1.py:26` mutated to `pass` (scratch copy), this test fails and the other 23 that run still pass.
- **N2.** The three superseded v5 cases now call `self.skipTest('superseded by V1/V3')`. `test_release_gate_v5_1.py`: 27 run, 3 skipped, OK.
- **N3.** Added under V3 above.
- **S1.** The lift stage and the V3 debounce now read "3 consecutive posts (0.133 s first to last)". Values are unchanged.
- **S7.** `test_stages_r2.py`, on synthetic B-format traces: a table-contact post breaks the lift run (lift moves from control 6 to 9), and free posts before the grasp control don't count (lift at 8, not 6). A third test checks there is no lift without a grasp. Mutating out the table-contact term or the at-or-after-grasp start in a scratch copy of `stages_r2.py` fails the matching test.
- **N6.** This file's hash is in `r1-lock-package.sha256`, computed after this section was written.

### S3: the contact flag is not shared

A's `contact` comes from the 'gripper' sensor, which is one finger. B's flag is true when either of its two tip pads touches. Within-robot comparisons (ΔA: A/P1 vs A/P0; ΔB: B/P1 vs B/P0) use the same flag on both arms and are unaffected. Comparing stage rates across robots carries a B-lenient bias, because B can show contact where A's single finger would not. `score_step8.py` prints this caveat under any table that shows A and B stage rates together.

### What `score_step8.py` does

`python score_step8.py --phase {before,after,mock} --out DIR [--stamp "<date -u>"] ROOT [ROOT ...]`. It is offline only and reads, never writes, the run directories.

**Discovery.** Each root is searched for directories named as step 8 cases in `experiments/bowl_step8/STEP8_RUNNERS.md`: `step8_{A,B}_{P0,P1,P0canary}_l{4000..4011}_s0_v1`.
- **A:** the trace is `RoboLab/output/<case>/<task>/strict-metric.json`. The supervisor's `runs/<case>` shares the name and is collected with it.
- **B:** `runs/<case>/gate_trace.jsonl`, plus `tracking.npz` and `checks.json`.
- **Smoke cases are never scored.** Canary cases are scored per condition and kept out of the pairing.
- For `--phase after`, pass roots that hold both phases' runs. P1 pairs with the before phase's P0.

**Scored** means a complete run or a B safety abort (B2, below). A complete run has all of the following:
- a positive exit record: the case's `episodes.tsv` row has status 0 and failure `no`, or, where no row exists, every `exit-code.txt` is 0;
- no `failure.txt`;
- a trace;
- for B, `checks.json` with `complete` true, and `tracking.npz`.

Anything else, including every crash, is listed as unscored with its reason and is never imputed (R4).

**Per episode:**
- the v5.1 verdict: first rest (the headline) and at end;
- the win control, the qualifying open and the fail reason;
- M2;
- the furthest stage, its controls, and the attempt count;
- the review flags.

**How the per-episode fields are computed:**
- **Headline (S5).** The outcome is `score()['pass']`, never `stages()['win']`. If they disagree, in the pass itself or in the win control, the episode gets a "stages win … disagrees …" review flag. The batch carries on rather than asserting.
- **N4.** The live result comes from the run itself: B's `checks.json` `automatic_v4B_success`, and A's strict-metric `stable_success_s`. It is never a v4 replay of a trace that the runner cut at the v4 win. So "v4 pass, v5.1 fail" catches a V1 window that the truncation left open.
- **N5.** A failing episode whose last row has the upstream predicate true gets the flag "fail, upstream predicate true at end".
- **Times (S4).** Times are control/15 s on both robots. B records `simulation_s` = (control+1)/15 because of its reset step, so that field is not used. A's recorded value already equals control/15.
- **M2 (S6).** M2 is descriptive only, with no test. Per condition it is reported as the median and the list "among winners".

**Per condition:**
- v5.1 wins k/n with a Wilson 95% interval;
- the at-end wins;
- the live gate passes;
- the furthest-stage distribution;
- per-stage rates with Wilson intervals;
- M2 among winners;
- the incomplete cases.

**Paired, per robot (P1 vs P0):** only layouts scored in both conditions count, so a safety abort is paired and a crash is not. The output gives the discordant counts and the exact two-sided McNemar p for the win and for each stage, and lists the unscored layouts.

**Output:** `score_step8-<phase>.json`, which records the code hashes, notes and problems, and `score_step8-<phase>.md`.

### Mock runs (`validate_score_step8.py` → `validation-score-step8.json`, outputs in `score_step8_mock/`)

**Mock set** (`--phase mock`): the 6 sealed `paint_live_v1` B episodes, painted vs unpainted over 3 layouts, plus the step 5 A traces.

**Match against the earlier validation:** all 21 complete episodes match `validation-v5_1.json` and `validation-stages-r2.json`. That covers:
- pass, at end, win control and qualifying open;
- review flags, excluding the new N5 flag;
- furthest stage and stage controls;
- M2;
- S4 time against A's recorded seconds.

The M5 illustration (stage rates and paired McNemar) is reproduced exactly. `step5_controlled_layout1000_v3` has no trace and is listed incomplete; its retry is scored. B dev lift is not part of the mock set.

**Mock results:**

| Condition | v5.1 wins | Wilson 95% | Live gate passes | Furthest stage (none/grasp/lift/over/open/landed/win) |
| --- | --- | --- | --- | --- |
| A (step 5) | 6/15 | 0.20–0.64 | 1/15 (only 2002 ran v4 live; the pilots ran v3/v2) | 1/0/2/5/0/1/6 |
| B painted | 0/3 | 0.00–0.56 | 0/3 | 0/0/0/3/0/0/0 |
| B unpainted | 0/3 | 0.00–0.56 | 0/3 | 2/1/0/0/0/0/0 |

- **M2 among winners (A):** median 8.933 s, n = 6.
- **Paired, B painted vs unpainted:** win 0/0, p = 1. Grasp 2–0, p = 0.5; lift and over_bowl 3–0, p = 0.25.
- **N5 flags:** A 1007, A 2000, B base s1 painted, B l2002 painted.
- **S5 disagreements:** none.

**Synthetic after-phase tree,** built in a temp dir from sealed traces:
- **Discovery:** names under `runs/` and `RoboLab/output/` are found, and the smoke case is ignored.
- **Episode status:** an A/P1 with `episodes.tsv` status 1 and failure `capture` is unscored.
- **R4, A:** pairing uses `l4000` only. For B, see the B2 tests below.
- **Canary:** it is scored and not paired.
- **N4:** an A/P1 copy of 2002 cut at control 295 has a live v4 pass and a v5.1 fail (`not_stable_by_end`), and is flagged "v4 pass, v5.1 fail". A v4 replay of that truncated trace could not pass, because v4's pass came at 301.
- **S5:** with `stages()['win']` forced False, 2002 keeps its v5.1 pass and gets the disagreement flag.

### B2: policy-driven B aborts are task outcomes (23 October 2026, 19:47 UTC)

The reviewer ruled that a B safety abort is a task outcome, not a crash: it is scored, it is in ΔB, and it does not count toward the halt. The marker is the builder's `runs/<case>/abort-classification.json` (`step8-abort-classification-v1`, written by `bowl_step8/classify_abort.py`), and the `episodes.tsv` failure column reads `no`, `safety_abort` or `crash`.

**Scored as a safety abort** (revised 19:54 UTC for the reviewer's final round) when all of these hold:
- the marker has the right schema and case, and class `safety_abort`;
- its `failure_txt_sha256` matches `failure.txt`, hashed as the classifier hashes it;
- an `episodes.tsv` row exists (else `no_episode_row`), with failure `safety_abort` and the marker's exit status;
- **the locked classifier agrees.** `score_step8.py` imports `bowl_step8/classify_abort.py` (sha256 `5c96bea4…`, C1) and runs its `parse()` and `classify()` on `failure.txt`, with the marker's exit status (cross-checked with the row) and the gate rows on disk. If its class or kind differs from the marker's, the episode is unscored with reason `marker_disagrees_with_classifier`. Under C1, 0 gate rows is always a crash.

A marker of class `crash` (confirmed by the classifier), any mismatch, or a `failure.txt` without a marker leaves the episode unscored (R4). The classifier's own class, kind and gate-row count are recorded as `abort_classifier`.

**How an abort is scored:**
- **From the files on disk (P2).** The gate and stage rows come from `gate_trace.jsonl` and `tracking.npz`, not from the marker. The marker's `class` and `kind` are recorded next to the verdict (`abort_class`, `abort_kind`), and its `partial_trace` counts are kept; a disagreement with the files is a review flag.
- **Trace lengths (P1).**
  - A soft-limit abort in `state()` at runner :298 comes after the gate row is appended (:297) and before the tracking row is, so `gate_trace.jsonl` is one row longer than `tracking.npz`. Unaligned, `stages_r2` raises IndexError when touch is true on that last row (checked on a sealed trace cut at 1250 rows). The aborted control's gate row is dropped, so the scored rows end at the last control with positions. The output records `gate_trace_rows`, `tracking_rows`, `scored_rows` and `dropped_gate_rows`.
  - Aborts at :237, :273 and :282 come before the gate row, so the lengths are equal and nothing is dropped.
  - A :298 abort at step 0 leaves one gate row and no `tracking.npz`. After the drop, 0 rows remain.
  - So the gate rows must exceed the tracking rows by exactly 1 for `B_joint_beyond_soft_limit` and by 0 for every other kind, with matching steps. Any other shape is unscored (`trace shape: …`).
- **Empty or near-empty traces.** Under 3 scored rows, the episode is a fail with furthest stage none, labelled `empty_trace`. This is what v5.1 and `stages_r2` would give: grasp needs 3 rows and a win at least 16. It is never dropped.
- **Verdict.** A v5.1 first rest reached before the abort is a win. Otherwise `fail_reason` is `safety_abort:<kind>`, and the gate's own reason is kept as `gate_fail_reason`.
- **Live v4.** It is "not reached": the runner breaks on a v4 pass, so an aborted run never had one, and there is no complete `checks.json`. N4 is unchanged for complete episodes. A partial trace whose rows record a live v4 pass would be flagged.
- **Review flag.** Every scored safety abort carries the `safety_abort` review flag, including a win before the abort.

**Reporting.** Per condition, the JSON gives the safety-abort count, the count by kind, the wins among them, the empty traces, the dropped rows, the crash count, and the other unscored abort markers (for example a classifier disagreement). The markdown has a table of these per B arm.

**Disclosure.** A has no target-limit check, so the same out-of-limit target ends a B episode but not an A episode. Within-robot ΔB is unaffected. This line is printed under the B-arm abort table.

**Synthetic tests,** in the after-phase tree:

| Layout | B/P1 | Shape | Result |
| --- | --- | --- | --- |
| l4002 | soft-limit abort, base s0 painted cut at 1250 gate rows | one row longer | scored on 1249 rows, 1 dropped; fail, `safety_abort:B_joint_beyond_soft_limit`, furthest grasp |
| l4003 | a pre-C1 marker: IK action `safety_abort` on the first control | no trace files, 0 gate rows | unscored: `marker_disagrees_with_classifier` (the classifier says crash) |
| l4004 | crash marker, with a partial trace | — | unscored (R4) |
| l4005 | IK action abort, B dev lift cut at 1700 rows | equal lengths | scored on 1700 rows; the first rest at 1554 stands as a win |
| l4006 | soft-limit abort at step 0 | 1 gate row, no `tracking.npz` | scored; fail, furthest none, `empty_trace` |

Every B/P0 is a complete sealed run.
- **Locked classifier:** it gives each kept abort its kind with at least 1 gate row (l4002 and l4006 soft limit, l4005 IK action), and gives l4003 and l4004 crash. l4006 (step 0, 1 gate row, no tracking) stays a :298 soft-limit `safety_abort` under C1 and is scored as `empty_trace`.
- **ΔB pairs** l4000, l4002, l4005 and l4006. l4001 (no P1), l4003 (classifier disagreement) and l4004 (crash) are unscored.
- **Win discordance** is P1-only 1 (l4005), P0-only 0.
- **B/P1 aborts:** 3 safety aborts (`B_joint_beyond_soft_limit` 2, `ik_action_A_to_B` 1), 1 win, 1 empty trace and 2 dropped rows, plus 1 crash and 1 other unscored marker. The markdown table shows the same counts.
- **Class and kind:** every abort, and the crash, carries them next to its verdict, and every scored abort has the `safety_abort` flag.

**Markers that must stay unscored** (`test_score_step8.py`, a separate tree):
- a soft-limit abort with no `episodes.tsv` row → `no_episode_row`;
- an IK action abort whose gate trace is one row longer than tracking → `trace shape`, because only the soft limit may leave that row;
- an `episodes.tsv` exit status (137) that disagrees with the marker (1);
- a soft-limit traceback whose marker says `policy_A_target_outside_limits` → `marker_disagrees_with_classifier`.

### Tests

| File | Result |
| --- | --- |
| `test_release_gate_v5.py` | 10 OK |
| `test_release_gate_v5_1.py` | 27, 3 skipped, OK |
| `test_stages_r2.py` | 3 OK |
| `test_score_step8.py` | 4 OK (the mock reproduction, the synthetic after phase with the B2 cases, the S5 flag, and the markers that must stay unscored) |

### R1 files (sha256)

`r1-lock-package.sha256` holds the hashes of every file below and `SCORING_V5.md`. That covers the full import chain: `release_gate_v4.py` imports `release_gate_v3.py`, which imports `stability.py`; `score_step8.py` imports the locked `classify_abort.py` (R2). Check it with `shasum -a 256 -c r1-lock-package.sha256` from this directory.

| File | Role | sha256 |
| --- | --- | --- |
| `release_gate_v5_1.py` | v5.1 gate, locked | `5e900d6e6f48a7aa1fed17cdaa253f86451e13dedd8e0e4059553c6d09692876` |
| `stages_r2.py` | stage metrics r2, locked | `4c08f687449e60ec5c3315247e69df349f93cce8ce1db6ef8d68221aadc46e56` |
| `stages.py` | history; M5 helpers imported by stages_r2 | `f8c835b33240d1218df35e020d9840ec94fdb1edaef23c918061cb3e97ebd7a6` |
| `release_gate_v5.py` | history; base class | `7bba5a40da44adbbc8ad0b16fb038ab2fd74f65ac73fe5e9c4c0cf6e9335918e` |
| `../bowl_step5/release_gate_v4.py` | step 5 v4 gate; base class | `1b249ae1ac1913babb7c31325898bf01823409172cb0080ccb59430c218c842c` |
| `../bowl_step5/release_gate_v3.py` | step 5 v3 gate; imported by v4 | `8a5be7d433802a9eac89bb0ec5dbe1c1592306191a3140e8f447f6c9bea15b4a` |
| `../bowl_step5/stability.py` | StableWindow; imported by v3 | `25971d4dec1ae0562c418067aca7e205f0160d115a45d7cc8475a016cce7372f` |
| `../bowl_step5/rim_support.py` | step 5 support signals; A recompute | `97d9e833bb24f0b8fd97ae6ef4cf89c5928ed97d1ce387ddfc7efd731d7a2ea9` |
| `../bowl_step8/classify_abort.py` | locked abort classifier (C1); imported by score_step8 | `5c96bea489f72e8a0c0bacdf83ccb08fc3e9847656767689e4967e4993abfed9` |
| `score_step8.py` | step 8 offline scorer, with B2 aborts | `8406460be54e1481c517e259981b7e9e061b71a64345180a6978cb10356186e1` |
| `validate_score_step8.py` | mock reproduction and synthetic after phase | `8d6bb40947bad9d5aca22917af12dd22e706d0094f5a0c24e639005a145c6920` |
| `validation-score-step8.json` | its output | `1af953366362bb73e8d8d147bcfb7b3c7642e0864cf14751096fc32900efdc76` |
| `score_step8_mock/score_step8-mock.json` | mock scorer output | `6e4443faba0079aa0b37d1b2740d4e375822b722cd97f44966bd1ff9bb0cac31` |
| `score_step8_mock/score_step8-mock.md` | mock scorer table | `f472cdc0326bef86799480baa5898695d8b2315883d8581a05e0008abffcb533` |
| `test_release_gate_v5.py` | unchanged | `7bc346a227073d16a685a9dcfc89a8e4bd875d0b9275210b05a6f63feaf98ba1` |
| `test_release_gate_v5_1.py` | N1, N2 | `78e87997bc5acd215e22c14572d92cde6688fbc60a0cd5f34ea35e47d1260fae` |
| `test_stages_r2.py` | S7 | `c456fa875ed2deb0bf9ae964107b92d120d31fe5a64c7aa8b23eda48283a1287` |
| `test_score_step8.py` | score_step8 tests | `6d100cc260c60ec4f6951e70f62cac9d8d1182f0c9cf0851f5e02e560d859b30` |

`SCORING_V5.md` itself: see `r1-lock-package.sha256`.
