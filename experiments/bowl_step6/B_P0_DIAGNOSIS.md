# B/P0 v2 failure diagnosis

22 October 2026. Step 6 remains incomplete. P0, physics, adapter settings and sealed recordings are unchanged.

The independent reviewer approved the v2 recording and CPU diagnosis. Completion of 750 steps proves integration, not task success: the banana stayed on the table.

## Findings

- The kinematic target-to-IK-goal error at the canonical gripper frame is at most 0.128 mm. This does not prove collision-free or dynamically feasible motion.
- The 0.04 rad/tick command limiter activated on 34/750 rows. It did not activate during rows 135–299, covering the brief closing attempts. It is not sufficient to explain all tracking error.
- Commanded-FK versus measured canonical position error: median 15.64 mm, maximum 114.92 mm. The maximum occurred late, after the first grasp attempts. During approach rows 50–134, median error was 7.75 mm and maximum 31.66 mm. These discrepancies combine dynamics, contacts and timing; they do not prove the gains are wrong.
- B received seven close-command bursts, each only 1–3 control ticks (0.067–0.2 seconds). Its measured normalized closure peaked at 0.131. No recorded row had both fingertip contacts above 0.1 N; no close-command row had either tip above that threshold. Contacts are sampled control-rate observations, not proof of absence at every physics substep.
- In successful A2002, the two close bursts lasted 67 and 65 ticks. Different layout and feedback prevent a causal comparison.
- Four selected existing original-scene A recordings, including the controlled-release failure, began with a sustained close burst of 56–73 ticks and lifted the banana. This narrows the issue beyond cherry-picking A2002, but remains a descriptive comparison rather than a matched experiment.
- Virtual-A joint disagreement has a larger local Jacobian-nullspace component than A2002's measured joint disagreement: medians 0.0226 versus 0.00466 rad. Only one B pair combined joint-vector error above 0.1 rad with pose error below 5 mm/0.02 rad. This is a local linear projection, not proof that redundancy handling caused the task failure.

## Data interpretation

The v2 inventory's prose mistakenly says “left inner knuckle.” The code and exact saved observations use `robotiq_85_left_knuckle_joint`, joint index 8, divided by 0.8 and clipped. This document corrects the description without altering sealed evidence.

The original result's raw-tool maximum error (95.69 mm) and this canonical-gripper maximum (114.92 mm) refer to different points on the same rigid tool. They are not contradictory.

In the CPU summary, maximum-row indices are relative to each statistic's input array. For an interval [s,e), add s; for previous-action comparisons, add 1 to obtain the observation row. All rows are zero-based.

## Next diagnostic and limits

First, preflight all 301 recorded A2002 commands through the existing FK/IK and rate limiter on CPU. An optimistic previous-command seed substitutes for unavailable measured B state; that limitation must be explicit. No simulation, model inference, fitting or collision claim follows.

If feasible, prepare a separately reviewed A replay control, followed only after its result review by B replay. Use the complete recorded action sequence at 15 Hz and the same recorded starting scene. A must sufficiently reproduce its reference before B replay can diagnose transfer. These runs would be prerecorded-command diagnostics, not evidence that B's closed-loop P0 succeeds.

A2002 has two initial-state entries because of resets. Its evaluator's `initial_poses_world`, captured immediately before the first executed action, matches the second entry. Replay setup must verify that correspondence rather than arbitrarily selecting a row.

No gripper latch, gain change, IK seed change, temporal scaling or automatic retry is currently approved. No dominant cause has been established.

The four selected A recordings do not exhaust earlier v1/v2 development attempts. The CPU preflight field `max_goal_change_rad` means goal minus previous limited command, not change between successive goals.

## Recorded-action controls and timing audit (October 22 evening)

A reproduced its 301-step reference exactly. B received all the same logical commands in the same scene, but never lifted the banana. The independent reviewer approved both evidence sets, not B task success. B's closure reached 1.000009 under two sustained close bursts (67 and 65 controls), yet there was only one sampled fingertip contact above 0.1 N and no bilateral contact. The first close-command response was 0.0417 normalized on B versus 0.416 on A in this task recording. Initial pre-step contact data may be stale and is excluded.

A separate CPU audit of three sealed stationary recordings found 95% normalized closure after 2 controls for A and 23 for both native and mapped B: 0.133 and 1.533 seconds at 15 Hz. This measures normalized joint response, not equal aperture or grasp force. The reviewer independently found the same timing difference during opening.

These measurements motivate a **single predetermined 12× uniform time-resampling diagnostic**, with factor ceil(23/2), rather than an outcome-selected speed sweep. Each complete original command is repeated 12 times. No open commands are dropped or latched. B's native physics, gains and limits stay unchanged. This gives 3,612 control ticks, 28,896 physics integration substeps and 240.8 simulated seconds; source logical time is 20.067 seconds. Only the episode timeout is extended to accommodate the diagnostic. Exact executable review remains required before execution.

Slowing every command also changes arm tracking, contact and dwell time. Even success would not isolate gripper speed as the sole cause, prove closed-loop P0 success or finish step 6. Later grasp attempts after object divergence are not independent trials.


### Temporal B replay result — independently reviewed

All 24 evidence files verified; 3,612 controls and decoded frames, 240.8 seconds at 15 Hz. Exact approved 12x action repetition, exit 0, saved GPU cleanup empty. Kinova B grasps and lifts: 222 bilateral contact samples; maximum banana center z 0.1335455 m, versus 0.0241033 m in ordinary B replay. It slips outside the bowl at about 76.6 seconds while commanded closed; first open command is 114.467 seconds. Parent and adversary independently reviewed video. This is diagnostic improvement, NOT task success or a live-policy success-rate estimate. Step 6 remains incomplete.

Next proposed direction: one closed-loop unchanged-P0 B test with explicitly declared temporal pacing and fresh B images/state at replans. Stop replay timing sweeps. Native B physics retained; no P1 training. Exact implementation needs adversary approval before execution. User reiterated Robot B must be a different robot, not an A variation.


### Live P0/B temporal result — independent Codex adversary review

Run step6_B_P0_temporal_v1 completed 2026-10-22 22:07:30 UTC. All 70 sealed evidence files verified. 38 fresh requests, 3,612 controls, 301 logical actions, exact response consumption; exit0 and GPU process cleanup empty. Independent request audit matched every recorded mapped state; both camera streams had 38 distinct frames.

Parent and adversary video review: Kinova B grasped and lifted the banana, then commanded opening outside the bowl at physical time 189.667 seconds. Bilateral contact spanned rows2212–2843 (632ticks); max banana center z0.075759m. The banana settled on the tabletop, not bowl/rim. This is deliberate release but misplaced, not task success. The prior recorded temporal run slipped while closed; this live run is a different diagnostic, not a controlled success-rate comparison. No training, adapter freeze or Step6 completion.

Possible next investigation: placement and coarse6.4s replanning; consuming fewer predictions per request while keeping12tick dwell is a hypothesis requiring a new reviewed protocol, not a proven fix. No new run started. User requested results only for now, and named the model setting for future adversary reviews [setting name removed in the public copy]. VM remains running; simulation/policy processes cleaned up.


### Faster-feedback failed-command inspection

Saved 1164controls/77.6s; attempted action1164 was rejected before execution. Native joint6 measured -2.0836856rad near its -2.0899998rad lower bound; local IK candidate landed on bound and stagnated after2iterations with1.148637mm/0.00791248rad residual, beyond0.1mm/0.001rad tolerances. Zero close commands and zero sampled fingertip contacts; banana maximumcenterz0.023477m. Parent video inspection confirms no pickup before stop.

The independent Codex adversary trace review agrees. Its bounded CPU cross-check with original seed/target/bounds improved residual to0.447mm/0.00277rad but still failed tolerances at the same bound. This does not prove global unreachability. Diagnosis: adapter local configuration/joint-limit problem before grasp; faster feedback was not demonstrated to improve task success. Recommend offline investigation of a nearby continuous feasible joint configuration before another simulation; do not loosen tolerances or blindly retry. No execution code changed, no new simulation. Partial evidence sealed under P0_B_feedback_v1/manifest.json.
