# Step 8 offline score, phase `before` (Sat 24 Oct 2026 02:11:08 UTC)

| Case | Live gate | Safety abort | v5.1 first rest | v5.1 at end | Win control (s = control/15) | M2: P0 time to open | Furthest | Stage controls | Review flags |
|---|---|---|---|---|---|---|---|---|---|
| step8_A_P0_l4000_s0_v1 | fail | — | fail | fail | — | — | over_bowl | grasp 158, lift 171, over_bowl 187 | fail, ends on bowl; fail, upstream predicate true at end |
| step8_A_P0_l4001_s0_v1 | fail | — | fail | fail | — | — | over_bowl | grasp 83, lift 92, over_bowl 103 | fail, ends on bowl; fail, upstream predicate true at end |
| step8_A_P0_l4002_s0_v1 | fail | — | fail | fail | — | — | over_bowl | grasp 76, lift 89, over_bowl 96 | fail, ends on bowl; fail, upstream predicate true at end |
| step8_A_P0_l4003_s0_v1 | fail | — | fail | fail | — | — | over_bowl | grasp 83, lift 95, over_bowl 99 | — |
| step8_A_P0_l4004_s0_v1 | pass | — | pass | pass | 146 (9.733 s) | 8.533 s | win | grasp 75, lift 83, over_bowl 92, open 128, landed 128, win 146 | v5.1 pass |
| step8_A_P0_l4005_s0_v1 | fail | — | pass | pass | 158 (10.533 s) | 8.533 s | win | grasp 76, lift 82, over_bowl 94, open 128, landed 129, win 158 | v5.1 pass |
| step8_A_P0_l4006_s0_v1 | pass | — | pass | pass | 160 (10.667 s) | 9.467 s | win | grasp 84, lift 92, over_bowl 109, open 142, landed 142, win 160 | v5.1 pass |
| step8_A_P0_l4007_s0_v1 | pass | — | pass | pass | 149 (9.933 s) | 8.533 s | win | grasp 66, lift 72, over_bowl 97, open 128, landed 129, win 149 | v5.1 pass |
| step8_A_P0_l4008_s0_v1 | fail | — | pass | pass | 159 (10.6 s) | 8.533 s | win | grasp 75, lift 83, over_bowl 99, open 128, landed 129, win 159 | v5.1 pass |
| step8_A_P0_l4009_s0_v1 | pass | — | pass | pass | 151 (10.067 s) | 8.533 s | win | grasp 75, lift 82, over_bowl 89, open 128, landed 128, win 151 | v5.1 pass |
| step8_A_P0_l4010_s0_v1 | fail | — | pass | pass | 445 (29.667 s) | 28.067 s | win | grasp 347, lift 352, over_bowl 376, open 421, landed 422, win 445 | v5.1 pass |
| step8_A_P0_l4011_s0_v1 | fail | — | fail | fail | — | — | open | grasp 77, lift 86, over_bowl 111, open 175 | fail, ends on bowl; fail, had an eligible open; fail, upstream predicate true at end |
| step8_B_P0_l4000_s0_v1 | fail | — | pass | pass | 1278 (85.2 s) | 6.872 s | win | grasp 640, lift 711, over_bowl 903, open 1237, landed 1238, win 1278 | v5.1 pass |
| step8_B_P0_l4001_s0_v1 | fail | — | fail | fail | — | — | landed | grasp 2668, lift 2692, over_bowl 2692, open 2845, landed 2845 | fail, ends on bowl; fail, had an eligible open; fail, upstream predicate true at end |
| step8_B_P0_l4002_s0_v1 | fail | — | pass | pass | 1061 (70.733 s) | 5.739 s | win | grasp 637, lift 702, over_bowl 787, open 1033, landed 1034, win 1061 | v5.1 pass |
| step8_B_P0_l4003_s0_v1 | pass | — | pass | pass | 1172 (78.133 s) | 6.339 s | win | grasp 663, lift 699, over_bowl 835, open 1141, landed 1141, win 1172 | v5.1 pass |
| step8_B_P0_l4004_s0_v1 | not reached | ik_action_A_to_B at 7488 | fail | fail | — | — | over_bowl | grasp 4484, lift 4518, over_bowl 4803 | safety_abort |
| step8_B_P0_l4005_s0_v1 | not reached | ik_action_A_to_B at 4416 | fail | fail | — | — | over_bowl | grasp 1876, lift 1926, over_bowl 2186 | safety_abort |
| step8_B_P0_l4006_s0_v1 | not reached | B_joint_beyond_soft_limit at 5067 | pass | pass | 1553 (103.533 s) | 8.472 s | win | grasp 754, lift 808, over_bowl 963, open 1525, landed 1527, win 1553 | v5.1 pass; safety_abort |
| step8_B_P0_l4007_s0_v1 | not reached | ik_action_A_to_B at 2016 | fail | fail | — | — | over_bowl | grasp 640, lift 676, over_bowl 964 | safety_abort |
| step8_B_P0_l4008_s0_v1 | fail | — | pass | pass | 5478 (365.2 s) | 30.272 s | win | grasp 4140, lift 4287, over_bowl 4467, open 5449, landed 5450, win 5478 | v5.1 pass |
| step8_B_P0_l4009_s0_v1 | fail | — | fail | fail | — | — | over_bowl | grasp 637, lift 678, over_bowl 842 | — |
| step8_B_P0_l4010_s0_v1 | fail | — | pass | fail | 1745 (116.333 s) | 9.539 s | win | grasp 695, lift 795, over_bowl 963, open 1717, landed 1718, win 1745 | v5.1 pass |
| step8_B_P0_l4011_s0_v1 | fail | — | pass | pass | 1626 (108.4 s) | 8.872 s | win | grasp 640, lift 677, over_bowl 880, open 1597, landed 1597, win 1626 | v5.1 pass |

| Condition | v5.1 wins | Wilson 95% | Live gate passes | Furthest stage (none/grasp/lift/over/open/landed/win) | M2 median among winners (descriptive) | Unscored |
|---|---|---|---|---|---|---|
| A/P0 | 7/12 | 0.32–0.81 | 4/12 | 0/0/0/4/1/0/7 | 8.533 s (n = 7) | — |
| B/P0 | 7/12 | 0.32–0.81 | 1/12 | 0/0/0/4/0/1/7 | 8.472 s (n = 7) | — |

Caveat: The contact flag is not shared: A's 'gripper' sensor is one finger, B's flag is either of two tip pads. Within-robot comparisons are unaffected; A-vs-B stage rates carry a B-lenient bias.

| B arm | Safety aborts | By kind | v5.1 wins among them | Empty trace | Aborted control row dropped (:298) | Crashes (unscored) | Other unscored abort markers |
|---|---|---|---|---|---|---|---|
| B/P0 | 4 | B_joint_beyond_soft_limit 1, ik_action_A_to_B 3 | 1 | 0 | 1 | 0 | 0 |

Note: A has no target-limit check, so the same out-of-limit target ends a B episode but not an A episode. Within-robot ΔB is unaffected.
