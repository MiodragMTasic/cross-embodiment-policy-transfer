# Step 8 offline score, phase `mock` (2026-10-23 19:54 UTC)

| Case | Live gate | Safety abort | v5.1 first rest | v5.1 at end | Win control (s = control/15) | M2: P0 time to open | Furthest | Stage controls | Review flags |
|---|---|---|---|---|---|---|---|---|---|
| step5_controlled_base_s0_v2 | fail (command-linked-release-v2) | — | pass | pass | 162 (10.8 s) | 8.4 s | win | grasp 76, lift 84, over_bowl 95, open 126, landed 126, win 162 | v5.1 pass |
| step5_controlled_base_s0_v3 | fail (command-linked-release-v3) | — | fail | fail | — | — | over_bowl | grasp 720, lift 731, over_bowl 731 | — |
| step5_controlled_base_s1_v3 | fail (command-linked-release-v3) | — | pass | pass | 169 (11.267 s) | 9.333 s | win | grasp 83, lift 88, over_bowl 106, open 140, landed 140, win 169 | v5.1 pass |
| step5_controlled_layout1000_v3 | unscored: no trace found | | | | | | | | |
| step5_controlled_layout1000_v3_retry1 | fail (command-linked-release-v3) | — | pass | pass | 152 (10.133 s) | 8.533 s | win | grasp 69, lift 78, over_bowl 91, open 128, landed 129, win 152 | v5.1 pass |
| step5_controlled_layout1001_v3 | fail (command-linked-release-v3) | — | fail | fail | — | — | lift | grasp 329, lift 335 | — |
| step5_controlled_layout1002_v3 | fail (command-linked-release-v3) | — | fail | fail | — | — | lift | grasp 68, lift 78 | — |
| step5_controlled_layout1003_v3 | fail (command-linked-release-v3) | — | fail | fail | — | — | landed | grasp 88, lift 94, over_bowl 118, open 144, landed 145 | fail, had an eligible open |
| step5_controlled_layout1004_v3 | fail (command-linked-release-v3) | — | fail | fail | — | — | none | — | — |
| step5_controlled_layout1005_v3 | fail (command-linked-release-v3) | — | pass | pass | 153 (10.2 s) | 8.533 s | win | grasp 83, lift 91, over_bowl 104, open 128, landed 129, win 153 | v5.1 pass |
| step5_controlled_layout1006_v3 | fail (command-linked-release-v3) | — | fail | fail | — | — | over_bowl | grasp 67, lift 73, over_bowl 82 | — |
| step5_controlled_layout1007_v3 | fail (command-linked-release-v3) | — | fail | fail | — | — | over_bowl | grasp 106, lift 114, over_bowl 139 | fail, ends on bowl; fail, upstream predicate true at end |
| step5_training_layout2000_v4 | fail | — | fail | fail | — | — | over_bowl | grasp 428, lift 435, over_bowl 461 | fail, ends on bowl; fail, upstream predicate true at end |
| step5_training_layout2001_v4 | fail | — | pass | pass | 308 (20.533 s) | 19.0 s | win | grasp 208, lift 214, over_bowl 264, open 285, landed 285, win 308 | v5.1 pass |
| step5_training_layout2002_v4 | pass | — | pass | pass | 301 (20.067 s) | 18.667 s | win | grasp 217, lift 225, over_bowl 235, open 280, landed 280, win 301 | v5.1 pass |
| step5_training_layout2003_v4 | fail | — | fail | fail | — | — | over_bowl | grasp 684, lift 695, over_bowl 715 | — |
| step6_B_paint_live_base_s0_painted_v1 | fail | — | fail | fail | — | — | over_bowl | grasp 1248, lift 1313, over_bowl 1444 | fail, ends on bowl |
| step6_B_paint_live_base_s1_painted_v1 | fail | — | fail | fail | — | — | over_bowl | grasp 652, lift 679, over_bowl 868 | fail, ends on bowl; fail, upstream predicate true at end |
| step6_B_paint_live_l2002_s0_painted_v1 | fail | — | fail | fail | — | — | over_bowl | grasp 3879, lift 3987, over_bowl 4227 | fail, ends on bowl; fail, upstream predicate true at end |
| step6_B_paint_live_base_s0_unpainted_v1 | fail | — | fail | fail | — | — | grasp | grasp 2007 | — |
| step6_B_paint_live_base_s1_unpainted_v1 | fail | — | fail | fail | — | — | none | — | — |
| step6_B_paint_live_l2002_s0_unpainted_v1 | fail | — | fail | fail | — | — | none | — | — |

| Condition | v5.1 wins | Wilson 95% | Live gate passes | Furthest stage (none/grasp/lift/over/open/landed/win) | M2 median among winners (descriptive) | Unscored |
|---|---|---|---|---|---|---|
| A/P0 | 6/15 | 0.20–0.64 | 1/15 | 1/0/2/5/0/1/6 | 8.933 s (n = 6) | step5_controlled_layout1000_v3 |
| B/painted | 0/3 | 0.00–0.56 | 0/3 | 0/0/0/3/0/0/0 | — | — |
| B/unpainted | 0/3 | 0.00–0.56 | 0/3 | 2/1/0/0/0/0/0 | — | — |

Caveat: The contact flag is not shared: A's 'gripper' sensor is one finger, B's flag is either of two tip pads. Within-robot comparisons are unaffected; A-vs-B stage rates carry a B-lenient bias.

| B arm | Safety aborts | By kind | v5.1 wins among them | Empty trace | Aborted control row dropped (:298) | Crashes (unscored) | Other unscored abort markers |
|---|---|---|---|---|---|---|---|
| B/painted | 0 | — | 0 | 0 | 0 | 0 | 0 |
| B/unpainted | 0 | — | 0 | 0 | 0 | 0 | 0 |

Note: A has no target-limit check, so the same out-of-limit target ends a B episode but not an A episode. Within-robot ΔB is unaffected.

| Paired | Scored layouts | Win discordant (first only / second only) | Exact McNemar p | Unscored layouts |
|---|---|---|---|---|
| B: painted vs unpainted | 3 | 0 / 0 | 1 | — |

Safety aborts are task outcomes (B2): scored on the partial trace and paired. Crashes are unscored (R4).

p < 0.05 needs at least 6 discordant layouts, all in one direction.
