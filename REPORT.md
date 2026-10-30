# Cross-Embodiment Policy Transfer

**Can a policy trained for one robot perform the same task on another—and can further improvements transfer too?** A Franka-trained policy on a Kinova, through a fixed adapter, in simulation.

Miodrag Tasic · Nebius x NVIDIA Global AI Hackathon, Physical AI track · solo entry · simulation only · no institutional affiliation

## Overview

This is my solo learning project and a small exploratory simulation study. I investigated whether a policy trained for a Franka could complete a banana-and-bowl task on a Kinova through a fixed adapter, then whether further training on the Franka would improve both robots. The original policy completed 7 of 12 shared test layouts on each arm. Fine-tuning reduced those counts to 2 of 12 and 0 of 12. A later, separate Nemotron-assisted release passed the unchanged scorer in one selected Kinova run.

To make the comparison auditable, I froze the rules before testing and kept every failure. The win rule, scorer, adapter, training recipe and checkpoint rule were hash-locked before any test layout ran. The study covers one policy, two simulated robots, one task, twelve layouts, one fine-tuning run and one policy seed. It cannot establish general transfer reliability or real-hardware readiness. My use of engineering and review agents is documented in “How I worked”.

## The question

My hypothesis had two parts: first, the original policy could perform the task on Robot B through a fixed adapter without training on B; second, if further training improved the policy on A, that improvement could carry over to B through the same adapter. Fine-tuning is not guaranteed to improve performance, so the before/after test measures the change on each robot whatever its sign. Without an improvement on A, the second part cannot be established.

Robot A is a Franka Panda with a Robotiq 2F-85 gripper. Robot B is a Kinova Gen3 (7 joints) with a Robotiq 2F-85. The policy, P0, is NVIDIA's GR00T N1.7 DROID checkpoint, a vision-language-action model trained on DROID's Franka data. It sees an exterior camera, a wrist camera and the arm's state, and it is told "Pick up the banana and place it in the bowl."

The plan: measure P0 on both robots, fine-tune it on Franka data only to get P1, then measure P1 on both robots with the adapter unchanged. The pre-registered tests are within each robot: ΔA = A1 − A0 and ΔB = B1 − B0. A versus B is not the claim.

## Setup

Everything runs in NVIDIA Isaac Sim 5.1 and Isaac Lab 2.3.2 with RoboLab's banana-in-bowl task, on one Nebius AI Cloud VM with an RTX PRO 6000 Blackwell GPU. GR00T N1.7, an NVIDIA open model, runs there as a policy server from pinned weights (revision `05e7cc97`) with its Cosmos-Reason2-2B backbone, and the fine-tune ran on the same GPU. Separately, NVIDIA Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) on Nebius Token Factory turns plain requests into checkable task specs, kept apart from the test ("Nemotron on Token Factory" below). After the test, NVIDIA Nemotron 3.5 Super VL, also on Token Factory, checked camera images: saved frames first, then live in one assisted run, ABBA. Neither changes a locked score. The banana and bowl move between layouts.

Robot B is a different robot in the simulator, with its own kinematics, joint limits, gripper drive and physics. It is not a re-skinned Franka. I didn't have access to a physical robot arm, so there is no hardware result.

## Method: a fixed, rule-based adapter

The adapter has no trained weights. It was frozen on 23 October as 19 hash-locked files and never changed afterwards.

- **Arm mapping.** Inverse kinematics (IK) on a virtual Franka turns B's measured gripping-frame pose into the Franka joint state P0 expects. P0's Franka joint targets go back through forward kinematics to a world pose, and IK on the Kinova gives B's seven joint targets. A run aborts if a joint leaves its soft limits by more than 0.03 rad.
- **Timing.** In these simulator assets, A's gripper reaches 95% closure in 2 control steps and B's in 23. The adapter holds each P0 action for 12 steps so B's gripper can close, so a B episode runs 12× longer in simulated time (600 s against 50 s) for the same number of policy actions.
- **Lift.** Once P0 has held a close command for 24 steps, B's tool target rises 15 mm; it ramps back down when P0 opens. It applies only to a close that began with the banana away from the bowl, and that eligibility check reads privileged simulator state: banana–bowl contact and RoboLab's geometric bowl predicate, from the previous control step. A real robot would need its own way to know this.
- **Exterior cross-painting.** P0 sees a Franka where the Kinova is.

### Why painting

The first B runs failed even though P0 received an exact Franka-style arm state. A matched test held B's state fixed and swapped one input at a time (528 P0 queries, none executed). Its pre-registered measure, gripper tilt, came back negative, and its secondary window was inconclusive. An exploratory analysis written after that result found that B's wrist image and mapped state moved P0's planned target about as much as its own repeat noise (4–12 mm; the largest such shift, 14 mm, came from the wrist image), while B's exterior image alone moved it 19–116 mm, mostly sideways and away from the banana. Those images differ almost only in the robot pixels, which suggests P0 was reading the robot's appearance. That is an exploratory lead, not a proven or exclusive cause.

So I used the idea from Mirage (Chen et al., RSS 2024): show the policy the robot it was trained on. Before each P0 request the simulator hides B's visuals, poses a rendering copy of the Franka at the virtual-Franka joints, renders the exterior camera and restores the scene. That copy keeps its colliders, so it is parked outside the workspace, 20 m away, whenever physics advances. The wrist camera is never painted. This is ground-truth-assisted simulator rendering, not inpainting. A painted check passed before any live run: P0's target shift fell from 74.38 mm with B's real image to 7.11 mm (95% CI 6.09–8.96), against a 6.38 mm noise floor.

In three matched pairs, where only the exterior image differed, painted runs grasped, lifted and carried the banana to the bowl 3/3; unpainted runs lifted it 0/3. These are three development pairs: a consistent direction, not a measured effect size, and not proof that appearance explains every failure. None released deliberately: all three painted bananas slipped out of a closed gripper onto the bowl. One development run with the 15 mm lift then produced a deliberate release into the bowl (n = 1, layout 2002), and the adapter was frozen.

## Evaluation protocol

- **Splits.** Layouts 2000–2031 supply training demonstrations and 3000–3007 pick the P1 checkpoint, both Franka only. Layouts 4000–4011 are the test; 4012–4023 stay reserved and unseen. Adapter development used the base scene and layout 2002.
- **Win rule.** A commanded release, followed by a set-down or drop that settles on or in the original bowl; the rim counts. An uncommanded slip from a closed gripper is not a success. Scorer v5.1 requires the banana's first contact after leaving the fingers to be the bowl, and fixes the verdict when the first 1 s stable window completes ("first rest") with the bowl alone supporting it. Its parent, v5, agreed with 21 of 21 earlier reviewed verdicts; the older v4 gate agreed with 17.
- **Locks.** Scorer, stage measures, test runners, training recipe and checkpoint-selection rule were reviewed and hash-locked at 19:57 UTC on 23 October, before any test episode. Later changes are logged as deviations. One analysis came later: the exploratory pose-based variant of the scorer was declared 12 minutes after the before half started, before any outcome was inspected.
- **Order.** The "before" half (A/P0 and B/P0, alternating per layout, policy seed 0) ran first, so the adapter result exists even if training fails. B starts from A's recorded object poses.
- **Statistics.** Wins out of 12 with Wilson 95% intervals; ΔA and ΔB by exact McNemar test on paired layouts, each reported with its paired split and exact p. With 12 layouts, p < 0.05 needs a lopsided split of the discordant layouts: at least 6–0 (p = 0.031), and 8–1 also qualifies (p = 0.039). P0 won 7 of 12 on both robots, so P1 can gain at most 5 layouts on either, and even 5–0 gives p = 0.0625. No improvement can reach p < 0.05 in this design; a decline can (7–0 gives p = 0.016). That follows from the locked design and the before result; nothing was changed.
- **Review.** AI agents gated every step: a Codex adversary agent for steps 1–5 and the start of step 6, then, from 22 October, an independent AI reviewer (Claude, "shadow-reviewer"), which adjudicated earlier-phase scored outcomes from video and traces. That earlier reviewer did not see the scorer's numbers or reasoning, but knew which episodes the scorer had passed or flagged. The review log says which verdicts were checked on video and which on the trace alone. Formal adjudication was by AI agents. Codex reviewers returned for the after phase: sampled video frames plus event traces, not blinded and not continuous playback. Reviewer verdicts and the live v4 gate are reported beside the locked scorer, never pooled with it.

## Results

### Before training (P0 on test layouts 4000–4011)

| Condition                           | v5.1 first-rest wins (headline) | Wilson 95% | AI reviewer | Live v4 gate | Safety aborts |
| ----------------------------------- | ------------------------------- | ---------- | ----------- | ------------ | ------------- |
| A/P0, Franka                        | 7/12                            | 0.32–0.81  | 8/12        | 4/12         | —             |
| B/P0, Kinova via the frozen adapter | 7/12                            | 0.32–0.81  | 7/12        | 1/12         | 4             |

The reviewer confirmed all 14 scorer wins and found one miss (A 4000). The live v4 count is low because v4 rejects an open more than a short distance above the rim; most B wins released 2.4–4.0 cm above it, and A lost three live passes the same way.

On unseen layouts, the Franka-trained policy driving the Kinova through the frozen adapter matched the Franka's own count. The intervals are wide and contact sensing differs, so this is parity, not proof of equality. Equal counts are not equal layouts: both robots won 4006, 4008 and 4010; only A won 4004, 4005, 4007 and 4009; only B won 4000, 4002, 4003 and 4011.

### After training (P1, completed)

P1 is P0 fine-tuned on 27 reviewer-approved Franka wins (6,367 frames) from layouts 2000–2031, over two collection passes; the second pass's trigger was changed after the first pass's count was seen, a logged deviation that never touched the test layouts. Locked recipe: 2,000 steps, batch 32, learning rate 1e-4, a checkpoint every 500 steps, projector and diffusion head trained, backbone frozen. Training took about 23 minutes and loss fell from 0.087 to 0.008, which shows a close fit to 27 demonstrations, not generalisation. P1 is the checkpoint with the most reviewer-adjudicated wins out of 8 on 3000–3007, ties to the earlier one. The selection is final: adjudicated wins out of 8 were P0 6, c500 0, c1000 1, c1500 3 and c2000 3 (raw v5.1: 4, 0, 1, 3, 3), so P1 is checkpoint 1500, which ties c2000 and takes the tie as the earlier. No checkpoint beat P0 on validation, so it carries the locked label "P1, no A-validation improvement" (rule D5): the selected fine-tuned checkpoint, not an improved policy. Its validation count is selection-biased (the best of 4 on the same 8 layouts). The after half launched at 20:52 UTC on 24 October and completed. Its copied evidence was verified on 25 October UTC; the locked scorer reported no unscored episodes.

| Condition                         | v5.1 wins | Wilson 95% | AI reviewer | Live v4 gate | Safety aborts |
| --------------------------------- | --------- | ---------- | ----------- | ------------ | ------------- |
| A/P1, Franka                      | 2/12      | 0.05–0.45  | 3/12        | 1/12         | —             |
| B/P1, Kinova via the same adapter | 0/12      | 0.00–0.24  | 1/12        | 0/12         | 3             |

| Pre-registered test | Δ (wins) | Discordant layouts (P1 only / P0 only) | Exact McNemar p |
| ------------------- | -------- | -------------------------------------- | --------------- |
| ΔA = A1 − A0        | −5       | 0 / 5                                  | 0.0625          |
| ΔB = B1 − B0        | −7       | 0 / 7                                  | 0.015625        |

The after reviewers inspected sampled frames and event traces, not continuous playback, and were not blinded. They found A4007 (rest with velocity noise) and B4001 (command-only reclose without physical regrasp) to be physical placements missed by the locked scorer. These separate counts do not alter the headline or paired tests. Detailed evidence: the after-review verdicts in the [evidence release](https://github.com/MiodragMTasic/cross-embodiment-policy-transfer/releases/tag/evidence-v1).

Drift check, A/P0 rerun on 4000–4001 at the start of the after half: 0/2, matching the same two before-layout failure outcomes, but with different trajectories; this does not establish repeatability or rule out drift.

The selected fine-tuned checkpoint performed worse in these observed runs: A fell from 7/12 to 2/12 and B from 7/12 to 0/12 under the locked scorer. There was no improvement on A whose transfer to B could be assessed. These small, sequential before/after samples do not isolate fine-tuning as the only cause; the two canaries matched earlier failure outcomes but followed different trajectories. Reviewer outcomes are reported separately and do not replace the locked headline.

Pre-registered reading: ΔA and ΔB both positive is consistent with the improvement transferring under these conditions; ΔA positive with ΔB near zero or negative means it did not clearly transfer; no clear improvement on A means this run cannot assess transfer. With its own caveat: "Tiny samples or wide uncertainty: preliminary evidence, not a reliable general conclusion."

**Checks after the test.** None of these changes a locked score. A CPU replay of the locked scorer over all 48 saved before and after episodes reproduced every verdict and winning control, and the scorer's source hashes and every trace hash matched the score records, so the automated grading was the same before and after. That covers the automated scores, not the separate AI reviews.

An exploratory measure, defined after the test and applied the same way to all four groups, asks whether the banana ever rested on the bowl, rim included, with no table or fingertip contact, for at least one simulated second. Both B/P0 and B/P1 reached it in 8 of 12 runs; A fell from 12/12 to 8/12. It doesn't require a commanded release, so it is a secondary outcome, not the win rule, and it doesn't say where the banana ended up.

What was observed on the Franka, by a trace proxy (contact, clearance and closure, not measured grip force): on its first carry, P1 lost a lifted banana under a closed command in 6 of 12 test runs, against 3 of 12 for P0.

What explains it is a hypothesis, not a proven cause. An audit of the training data found a gap in how successful endings became training samples. Demonstrations were cut at the win, the loader only starts a sample where 39 more frames follow, and the robot executes the first 8 of each 40 predicted actions before replanning. In 25 of the 27 demonstrations, the successful opening never fell among those first eight targets, although it is present later in the 40-action targets. Ordinary opening commands were still common in the training samples; what was missing was the final, successful release. That is about how samples were drawn, not missing video. A sampling repair was worked out on CPU but neither trained nor shown to work, and no further retraining was run. Source and configuration checks found no demonstrated action-column permutation or one-row shift between observations and actions, but they didn't rule out every pipeline issue.

## Safety aborts on the Kinova

Four of B's twelve P0 test runs ended in a safety abort. All are scored on their partial traces; none was dropped. In three (4004, 4005, 4007) the local IK solver pinned wrist joint 6 at its lower bound, −2.09 rad, after the banana had been carried over the bowl. Other arm configurations were not tried. The fourth, 4006, left a soft joint limit at control step 5,067, long after its win at 1,553, and counts as a win. The P1 half had three IK safety aborts (4000, 4010, 4011), all retained as failures. The wrist limit is a lead for an adapter v2, which would be a separately labelled version tested on fresh layouts.

## Nemotron on Token Factory: from a plain request to a checkable task

I already knew LLMs are good at turning what a person wants into input a deterministic system can act on. I learned about MCP servers and deterministic systems at a LangChain4j workshop and used them in my own projects, Narra (travel.miodragdev.com) and Katchup (katchup.miodragdev.com), so this was something I was excited to explore here, just like the adapter: both were ideas I could build on from things I already knew.

Nemotron's job here is to turn a plain request into a structured task spec a program can check. NVIDIA Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) on Nebius Token Factory takes a request such as "put the fruit away" and returns the object and the target, named from the scene's own objects, where the object should end up, and the conditions a success has to meet, such as whether the gripper must be opened on purpose and whether the bowl's rim counts.

**The parsing ran; the robot run did not.** The spec also carries the one sentence the robot's policy should get and whether this robot should try at all, and a fixed local rule has to agree before anything moves. Each request was asked three times, and the robot would only get an answer given every time. With my first prompt, Nemotron declined "put the fruit away" as naming an object not in the scene, all three repeats agreeing; its repeats split on "put the banana beside the bowl, on its left"; and it refused "pick up the bowl and put it on the banana", as expected. So nothing qualified for the robot. A second prompt, written after seeing those answers and disclosed as such, gave the same answer on all three repeats of all three requests: the banana into the bowl, the banana left of the bowl (flagged as untested), and a refusal to move the bowl. The planned run of the unchanged policy (P0) on the Franka with those specs was never performed; the demo's code passed 87 offline tests, and that is all the checking it got.

An earlier version of the compile step, the intent contract, was tested on its own: 11 instructions, three runs each, from RoboLab's own wording and paraphrases to five hand-written adversarial ones. On the adversarial five, all 42 field checks matched a faithful reading: "the apple" came back as not in the scene, "don't let go of it" turned off the release requirement, and "the edge of the bowl" became the rim. It also exposed a real gap: it read "in the bowl" literally, so the rim didn't count in any of the 18 runs on the six ordinary instructions, while the project's win rule counts it.

**Is a bigger model better here? Not on this test.** I had a Claude agent run the same 11 instructions, five of them traps like "put the banana in the bowl but don't let go of it", three times each through four Nemotron models on Token Factory, with the rule for picking a winner written down before the first call. Lightning, Super (120B) and Ultra (550B) got every trap right and gave the same answer on every repeat; Nano ignored "don't let go" all three times. On a tie the cheapest wins, so Lightning stays: 33 calls for a fifth of a cent. Five traps can't tell the top three apart, so this doesn't show the bigger models are no better on harder requests.

| Model      | Traps (faithful field checks) | Same answer on repeats | Median latency | Cost (33 calls) |
| ---------- | ----------------------------- | ---------------------- | -------------- | --------------- |
| Nano       | 38/42                         | 8/11                   | 1.41 s         | $0.0024         |
| Lightning  | 42/42                         | 11/11                  | 0.87 s         | $0.0020         |
| Super 120B | 42/42                         | 11/11                  | 1.34 s         | $0.0085         |
| Ultra 550B | 42/42                         | 11/11                  | 1.25 s         | $0.0322         |

The vision model, Nemotron 3.5 Super VL, named exactly the banana and the bowl in the first simulator frame of all 12 Franka test runs, an easy test with only two objects on the table; it says nothing about judging when to let go. The pick rule was fixed before the first call; its file hash was logged just after the run, not before (a process slip).

I also tried using Nemotron to grade runs from their text logs; on 24 runs it did no better than always guessing "success", so the locked scorer and an independent reviewer do the grading. The 178 Nemotron calls before the comparison cost $0.02, per the Token Factory usage page.

### ABBA — Alpha Bravo Banana Away

After the test I proposed DROP: Nemotron 3.5 Super VL (`nvidia/Nemotron-3.5-Super-VL`) would watch P1 carry the banana on the Kinova and decide when to let go. Screened first on saved frames from the P1 Kinova runs, it passed a small set of held, approaching and already-resting cases after two failed prompt rounds, but near the rim it still said "ready to release" in 4 of 4 cases 5–8 cm off the bowl's centre and 2 of 4 at 8–12 cm (69 calls, $0.0522). That wasn't selective enough to trust with the gripper on its own.

On 26 October it ran once, as ABBA, from a new one-episode package adapted from the DROP code. Robot B ran P1 (checkpoint 1500) through the unchanged painted adapter on layout 4003, policy seed 0, for 3,600 controls (240 simulated seconds). While the gripper was closed, Nemotron looked at the Kinova's actual exterior camera, not the painted image P1 receives, and at the wrist camera. The simulation paused during each call, and P1 kept acting between checks. Call times below are the runner's ticks, counted from 0; the scorer counts controls from 1. Nemotron answered WAIT 27 times, every 12 ticks from the first gripper close at tick 672 to tick 984. At tick 996 it answered READY and the controller held the arm. At tick 1005 it confirmed READY from fresh views, and the open command executed at control 1006. The controller kept the gripper open for the remaining 2,595 controls. The 29 calls cost $0.0223 from returned token counts, at $0.30 per million input and $0.90 per million output tokens (prices checked 26 October), with 56.5 s of model wait in all.

The unchanged offline v5.1 scorer, the rule behind every locked comparison above, gives a PASS at control 1071, and the banana is still on the bowl at the end. At the open, the banana's root was 4.98 cm horizontally from the bowl centre and its centre 10.5 cm above the bowl top; those are evaluation-only measurements, never given to Nemotron. The runner's older live v4 checker reported no success: it ruled the open ineligible. v5.1 stays the primary rule because the before and after results were scored with it.

The earlier unassisted B/P1 run on layout 4003 reached bowl support through a slip and failed the controlled-release criterion (`experiments/scoring_v5/score_step8_after/score_step8-after.md`). In the separate ABBA run, Nemotron triggered a commanded release that passed the unchanged scorer. This demonstrates the assisted-release mechanism once; it does not establish a repeatable improvement or isolate Nemotron's contribution from the controller's hold and latched-open behavior.

- It is one exploratory assisted demonstration on one selected layout: not a success rate, not a controlled or paired causal comparison, and not a repair of training. The before/after results above stand.
- I picked 4003 because it was the only P1 Kinova layout where the banana stayed centred and held over the bowl for a long stretch in the after test. That earlier run is not a matched pair: the intervention changes everything after the hold.
- Nemotron's "central" judgement is not a calibrated measurement: the banana was about 5 cm off-centre.
- The hold and the latched open are controller behaviour around the model's decisions.
- After the forced open, P1's own gripper command also turned to open from control 1096, after 90 controls of the latched open (why it switched is not known), so the episode can't show what P1 alone would have done afterwards.
- One run doesn't resolve the near-rim READY answers from the saved-frame screen.

Before launch, the package had only Codex's own review and 43 CPU tests, with no independent sign-off or simulator validation. Claude's independent checks came after the run: Claude re-hashed all 298 copied evidence files against their manifests, re-ran the frozen scorer to an identical result, checked the 29-row cost ledger, matched the source lock to the run's permit and re-ran the 43 package tests; the last frame shows the banana in the bowl, leaning against the rim. The run's files keep their working names: case `roger_B_P1_l4003_assisted_v1`, banner "Roger Bravo, Banana away", video label "CHARLIE P1 ASSISTED". Evidence: `evidence/abba_l4003/` in this repository (score, gate trace, Nemotron ledger and tracking), and the full run with its videos in the [evidence release](https://github.com/MiodragMTasic/cross-embodiment-policy-transfer/releases/tag/evidence-v1). The RGB-D capture run and the intent-controlled robot demo never ran.

All 439 Nemotron calls, the comparison, the intent parsing, the saved-frame screening and ABBA included, came to $0.1441, priced from the token counts each response returned; the usage page confirmed only the first $0.02.

## Limits

- Simulation only.
- Painting uses simulator ground truth; a real camera needs mask-and-inpaint, as in Mirage. This is ground-truth-assisted simulator rendering; real-camera reconstruction was not tested.
- The 15 mm lift is part of the frozen adapter, and its eligibility check reads privileged simulator state (banana–bowl contact and the bowl's geometric predicate). B's wrist camera was also remounted to match A's relative pose. Both would need real sensing or real mounting on hardware.
- The lift was tuned on one development scene and tested once. The formal "a drop counts" rule was written after that run was seen; I had said beforehand that a placement into the bowl would count.
- Twelve layouts only resolve large effects. After a 7/12 baseline, no improvement can reach p < 0.05 (the best case, 5–0, gives p = 0.0625); only a decline can. A real but modest gain would look like noise.
- A's contact flag is one finger, B's either of two pads, so cross-robot stage rates lean toward B, and only B aborts on a target-limit check. Within each robot, P0 and P1 use the same measurement pipeline; if P1 behaves differently, the scorer's known blind spots could still bias the measured change.
- The banana's angular-speed noise at rest (0.25–0.29 rad/s) sits just under the 0.3 rad/s stillness limit, so v5.1 can miss or delay wins. P0 and P1 on one robot go through the same check, but different rests under P1 could still shift Δ. An exploratory pose-based variant, declared 12 minutes after the before half started and before any outcome was inspected, matched the headline on the before half.
- The reviewers are AI agents, not people. One checkpoint, one fine-tuning run, one policy seed: a case study of this run, not of a method. The training-data gap is an unproven explanation for P1's decline.

## How I worked

Up to about 21 October I worked by hand. I asked Codex for the command lines to type, the websites to go to and what to download, ran everything myself and took notes: SSH to the Nebius VM, Miniconda, Isaac Sim 5.1, Isaac Lab 2.3.2, a CUDA 12.8 build of PyTorch, and a pretrained Franka reaching policy played back and recorded. Alongside that I learned online, had ChatGPT quiz me and watched YouTube videos until I understood the networking and the Nebius setup behind each command, from SSH keys to security-group rules. The notebook on the project page shows what I covered. That first policy was far too simple for what I wanted to show. It moves the hand to a target pose with no object: it verified the stack, but it didn't show the pick-and-place task I wanted. On 21 October I changed the plan to GR00T N1.7 and the banana-in-bowl task.

My notes stop there, around 21 October. From then on, learning each step by hand and taking notes wasn't feasible in the time I had, so I directed AI agents, Claude (Anthropic) and Codex (OpenAI), including computer use, to build and run the rest while I watched: the code, the simulations, data collection, the fine-tune, the scoring and the drafts of this write-up. I made the calls on scope and success rules (the task, the win rule, the 12-layout test size, the order of the steps and the freeze), approved the spending and checked the results. AI agents reviewed every step: a Codex adversary agent until 22 October, then an independent AI reviewer (Claude), which adjudicated earlier outcomes. Codex reviewers adjudicated the after phase using non-blinded sampled frames and event traces; no human formal adjudication was performed. After the test, Codex agents replayed the scores and audited the training data, and a separate Claude analysis found the same data gap. Codex agents also wrote the DROP code and ran its saved-frame screening; Claude reviewers checked the code. On 26 October Codex agents built and ran ABBA under my approval for one attempt, and Claude checked its evidence afterwards.

## What I learned

My biggest lesson was how much research and planning goes into robot learning before you even start training—and how much there is to understand when an experiment doesn’t go as expected. Exploring all of this made me want to pursue a career in robotics and machine learning. I’d love the opportunity to keep working on this problem.

- **A first experiment can be too simple for the question.** My reaching-only pilot verified the stack, but it didn't show the pick-and-place task I wanted. Picking up an object made success and failure visible.
- **A vision policy can depend on what the robot looks like**, not only on where it is. Here that rests on one exploratory swap test and three development pairs.
- **Robots differ in timing, not only geometry.** A Franka grasp replayed on the Kinova failed until each action was held long enough for the gripper to close.
- **The win definition decided the results.** v3 and v4 rejected drops from a few centimetres above the rim, including the one Kinova release the reviewer passed. Writing the rule down and testing it against past verdicts mattered as much as the robot work.
- **An adversarial reviewer saves GPU hours.** Before the test it caught a runner that would have crashed on import and a rule that would have dropped Kinova safety stops from the score.
- **I didn't know what there was to learn until I started.** Page 1 of my notes is a short list of "What is …?" questions, and I understand those better now, but most of what I learned wasn't on it: setting up a Nebius VM (GPU, storage, security groups, CIDR, tenants and projects), SSH with ed25519 keys and trust on first use, Linux commands, Isaac Sim and Isaac Lab, what a neural network, a policy and an actor are, how PPO trains the actor and critic, joint stiffness, damping and effort limits, and how a fixed adapter differs from a learned one.

## Credits

- Cross-painting idea: L. Y. Chen, K. Hari, K. Dharmarajan, C. Xu, Q. Vuong and K. Goldberg, "Mirage: Cross-Embodiment Zero-Shot Policy Transfer with Cross-Painting", RSS 2024, arXiv:2402.19249. Mine is a simulator version using ground-truth rendering, not their inpainting pipeline, and none of Mirage's code or results are used.
- NVIDIA GR00T N1.7 DROID (`nvidia/GR00T-N1.7-DROID`) with NVIDIA Cosmos-Reason2-2B; Isaac-GR00T pinned at `dac54b31`. The policy's training data is the DROID dataset.
- NVIDIA RoboLab (pinned `b6cfcf6c`) and its banana-in-bowl tasks; asset licences as listed in RoboLab's third-party notices.
- NVIDIA Isaac Sim 5.1 and Isaac Lab 2.3.2.
- Nebius AI Cloud (GPU VM for simulation, serving and fine-tuning) and Nebius Token Factory (`nvidia/Nemotron-3_5-Lightning` for task specs, `nvidia/Nemotron-3.5-Super-VL` for the vision checks on saved frames and in the ABBA run).
- Built with Claude (Anthropic) and Codex (OpenAI) as engineering agents under my direction; before about 21 October, Codex only suggested the commands I ran by hand. Claude drafted this write-up from the project record. AI reviewers, not people, gated each step: a Codex adversary agent until 22 October, then independent Claude reviewers and, for the after phase, Codex reviewers using non-blinded sampled frames and event traces. Codex agents ran the post-test score replay and training-data audit, wrote the DROP code and its saved-frame screening, and built and ran ABBA; Claude checked the ABBA evidence.

Thank you to NVIDIA, Nebius, Tavily, and Bloom Korea for the opportunity to participate. [Bloom’s write-up of the Seoul event](https://blog.bloomworld.ai/en/physical-ai-is-this-years-christmas-gift-en/) records the kickoff that inspired this project. Tavily is acknowledged as an event supporter, not as a runtime integration in this project.
