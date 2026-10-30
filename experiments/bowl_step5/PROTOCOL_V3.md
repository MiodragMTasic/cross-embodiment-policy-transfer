# Command-linked release pilot v3 — development refinement

The user's task remains deliberate release onto/into the original bowl; rim placement is acceptable. Robot A, original P0, assets, cameras, observations, actions and speeds remain unchanged. No task actions are inserted by this evaluator.

## Why v3

The first v2 development case verified lift at 6.0667 s and an open-command-linked release at 8.4 s. Its permanent invalidation on an open-jaw touch at 8.5333 s, and its prohibition on later detached empty-jaw closure, were stricter than the chosen task. The reviewer independently confirmed this mismatch. The v2 outcome remains a failure under its recorded rule. Rescoring that observed trace under v3 is exploratory development analysis, not a new independent successful attempt. Do not pool v2 and v3 outcomes as one unchanged benchmark.

## Operational rule

Keep v2's sustained lift/contact, pre-action armed-contact opening edge, measured opening/detachment within 0.5 s and near-bowl checks unchanged. A slip before an eligible opening command still fails. The one-finger contact proxy and sampled nature retain their limitations.

After **verified release**, preserve that history through contact only if both the command and measured jaws are open (command <=0.5 and normalized finger position <=0.2). Any contact resets the settling interval. Contact combined with a closed command OR measured closure >0.2 revokes the release; check both pre-action and post-action signals. Geometric escape from the existing near-bowl region or nonfinite signals also revokes it. Reacquiring valid release history requires a fresh sustained lift and eligible opening. Before release verification, v2's reclosure/recontact rules still apply.

Detached fingers may close without revoking the historical release. A stable-placement pass requires the original native bowl predicate, no gripper contact, near-bowl geometry and the same linear <=0.03 m/s and angular <=0.3 rad/s thresholds for at least one elapsed second. Do not require detached fingers to remain open. The native predicate includes bowl contact and gripper detachment, not full-object containment. Table contact alone is not a veto because accepted rim placement can have mixed support. Geometric escape prevents a drop away from retaining release eligibility.

## Execution and splits

First review code, synthetic failure cases and replay of the preserved v2 trace. Then obtain adversary approval for exactly one fresh original-scene v3 case (layout0, policy0) before more GPU work. After its review, complete base policy1 and relocated development layouts1000–1007 with policy0; record actual poses because the upstream10cm variant moves both bowl and banana. Every trial has 50 simulated seconds maximum at15Hz, 40 predicted and8 executed action rows, fresh policy process and900s/600s server/simulator limits. Preserve all attempts, including infrastructure failures.

Training2000–2031, validation3000–3007 and final4000–4023 are separate and unused. Do not tune against final test results. v3 is a development-frozen primary measurement for this pilot, not a claim that published native scores reproduce this rule. Successful P0 recordings are filtered self-demonstrations; their auxiliary EEF predictions are not executed expert targets. Real training and subsequent robot stages still require the existing review gates.
