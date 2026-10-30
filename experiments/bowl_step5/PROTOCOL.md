# Step 5 — development pilot and disposable training smoke

Frozen before the first strict-metric run on 2026-10-21. P0 remains the original pinned DROID checkpoint. All work is simulated. No adapter and no P1 are created in this step.

## Pilot and data separation

- Two base-scene development runs, layout seed 0, policy seeds 0 and 1.
- Eight development runs of `BananaInBowlUniformInitPose10cmTask`, layout seeds 1000–1007 and policy seed 0. This upstream variant moves **both banana and bowl** within ±10 cm in x/y and uses collision checks. It is not a bowl-only perturbation study. Preserve the actual initial poses, seed and class name even though the upstream task name aliases the base task.
- Reserve layout seeds 2000–2031 for A training collection, 3000–3007 for validation, and 4000–4023 for the later A/B comparison. Do not run the final test bank during development. Seeds and captured actual layouts identify cases; a seed alone does not prove exact replay across different robot implementations.
- This ten-run development pilot diagnoses feasibility. It does not establish a dependable success rate. Infrastructure failures are kept and reported separately from task failures, without silently dropping cases.

## Metrics and execution

Use unchanged P0, DROID A, cameras and current proprioception, instruction “Pick up the banana and place it in the bowl”, 40 predicted actions, eight executed before replanning, one environment, 15 control steps/second, and a 50-second simulation limit. Fresh policy process per case, with a separately recorded policy seed. No privileged object state is sent to the policy.

Before constructing the termination manager, replace the terminal-success term with a wrapper around the original predicate. The original predicate uses object centroid/open-top container bounds, bowl contact and gripper detachment. Keep its first-true time as a separate measurement. Do not call this an unchanged upstream final-score run: the rollout continues after that first instant.

The **stable-upstream-predicate** outcome requires that original predicate plus banana linear speed ≤0.03 m/s and angular speed ≤0.3 rad/s remain true for at least 1.0 elapsed simulation second, measured from the first qualifying post-step sample. At 15 Hz this normally needs 16 samples. Any false predicate, excess speed or nonfinite velocity resets the interval. The same policy continues controlling the robot throughout; do not freeze physics or insert a hold controller. A timeout after first raw success is still a strict failure. Record the trace and both times. This operational metric is not proof of complete geometric containment or indefinite stability.

Save video including the terminal frame, initial poses, environment config, timing, model load checks and every executed action. The metric uses simulator state only for evaluation. The native subtask score is auxiliary and must not be substituted for the strict result.

## Synchronized data and training smoke

Record pre-action resized RGB images and current model-format state on every control tick, before `env.step`, paired with the exact eight joint/gripper targets passed to the action manager. Record whether the action came from a new inference or a cached chunk. At cached ticks the observation is the current observable state, not a fresh model request.

Preserve the nine-dimensional EEF prediction for the same executed chunk row as an explicitly auxiliary model prediction. It is not an achieved EEF pose or an independently verified controller target. These self-generated labels may validate the pipeline on a disposable copy, but they are not corrective expert demonstrations and cannot alone justify an improvement claim for P1.

Convert to the format actually accepted by the pinned GR00T loader; verify a real loaded sample against the source capture, including channel order, current-state timing, relative-action processing, dimensions and checkpoint modality settings. Match the checkpoint's video horizon rather than blindly using the pretraining registry default. Run a few optimizer steps on a copy of P0, save, reload and verify inference/simulation. Preserve original P0 hashes. Label and exclude this candidate from P1 and all later result comparisons.

Step 5 requires independent adversary OK after evidence review before step 6 begins.

## First diagnostic and proposed amendment (before further pilot runs)

The first v1 case ended at 9.0 s under the stable-upstream-predicate criterion, but frame/event/action review showed the banana slipped from a closed gripper onto the rim. It is **not** a controlled grasp-and-open-release success. Its captured data is used only for a disposable technical pipeline test, not as successful expert data or P1 data. The remaining pilot is held while the user clarifies full-inside larger-bowl placement versus acceptable rim placement in the original scene. Any new geometry/criterion gets its own named variant and protocol; the v1 evidence stays unchanged.

## Decision received after v1

Miodrag selected the original bowl and accepts deliberate rim placement. The controlled-release pilot is specified separately in `PROTOCOL_V2.md`; old v1 evidence is unchanged. The larger-bowl alternative is not being implemented.
