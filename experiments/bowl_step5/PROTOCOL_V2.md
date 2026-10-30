# Command-linked release pilot v2 — original bowl

User decision, 21 October 2026: deliberately release the banana onto/into the original bowl; rim placement is acceptable. Original assets, policy P0, robot A and camera/control settings remain unchanged. This evaluator is a sampled command-linked release proxy, not proof of mental intent or causal force closure.

## Operational success

All signals below are evaluation-only. The policy sees only its original cameras, current robot state and instruction.

1. Arm a lifted-grasp history after at least 0.2 elapsed seconds of closed commands (>0.5), banana contact with the declared left inner finger, no banana contact with table/bowl, and banana OBB bottom at least 1 cm above the table OBB top. Contact uses the upstream absolute force-component threshold >0.1 N. One finger is a proxy and can have false negatives; it is not a verified two-finger force closure.
2. Preserve that history while contact continues, even when lowering to the bowl. Any loss of contact before an eligible open command invalidates it. A fresh regrasp and sustained lift can start another attempt; all events remain in the record.
3. On a close-to-open command edge (<=0.5), require contact in the immediately preceding pre-action sample, armed lift history, raw finger_joint/(pi/4) >0.2, and near-bowl geometry. Near-bowl means the original open-top-hull centroid predicate without contact/detachment requirements plus banana OBB centroid no more than 15 cm above the bowl OBB top. Check near-bowl again at first detachment.
4. Within 0.5 s of that eligible command, require detachment and measured normalized finger position <=0.2 with at least 0.05 reduction from the pre-opening value. A same-tick post-action detachment may count because the immediately preceding pre-action contact was recorded. Recontact, reclosure, timeout or nonfinite signals invalidate that release. Later opening after a closed slip cannot rescue it.
5. After verified opening/detachment, require the original bowl predicate, continued detachment/open fingers, banana linear speed <=0.03 m/s and angular speed <=0.3 rad/s continuously for one elapsed second. At15Hz this requires16 qualifying post-step samples. Predicate/velocity failure resets the stability interval; recontact/reclosure requires a fresh eligible release. Physics and P0 continue throughout.

The executed original predicate tests the object's hull-vertex centroid against the bowl convex hull with upper planes removed, then requires bowl contact and gripper detachment. It is not full-object containment; the wrapper's old AABB docstring is inaccurate. Rim placement is acceptable per user choice. Record the native predicate separately rather than substituting its score for v2 success.

## Pilot cases and gates

Use the previously declared two base cases (layout0, policy seeds0/1) and eight relocated development cases (layout1000–1007, policy seed0), all with fresh model processes, 40 predicted/8 executed rows,15Hz,50s simulation limit. The upstream10cm variant moves both banana and bowl; preserve this label and actual initial poses. The original v1 case stays a diagnostic, not a v2 result. Training2000–2031, validation3000–3007 and final4000–4023 scene banks remain unused until their assigned steps.

First approve implementation/synthetic tests, then run exactly one base case and review pre/post signals, raw force matrices, action rows, events and video before authorizing the other nine. Each process remains bounded (model900s,sim600s). No P0 modification, B work or P1 training. The already approved disposable smoke is separate and not successful training data. A usable successful/corrective demonstration route is still required for full step5.

Evidence includes synchronized pre-action images/state/action labels, per-step pre/post evaluator signals and actual poses, terminal-inclusive videos, hashes of scripts and both protocols, finite/infrastructure checks and preserved failure outcomes. The user decision changes semantic success, not recorded old results.
