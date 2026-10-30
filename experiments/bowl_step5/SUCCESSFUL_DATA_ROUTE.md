# Proposed step-5 demonstration route: filtered P0 recordings

Status: methodological proposal; no new training-split simulation or P1 optimization is authorized by this file. First finish and review the frozen v3 development pilot, including its rim-placement proxy limitation.

## What changes from the earlier plan

The original plan requested expert/corrective A demonstrations. We have verified a disposable training/save/reload pipeline with a failed P0 rollout, but that is neither successful task data nor an expert. The next proposed route is success-filtered P0 self-demonstration/distillation. It may reinforce P0's successful behavior; it is not new expert supervision and does not guarantee improvement. Use A validation alone for model selection or data-source redesign. If a P1 candidate does not improve there, report that result and reconsider the data source before claiming transferred improvement. The final test is one locked assessment; any redesign after inspecting it requires a newly declared study and test bank.

The 17 supervised targets comprise seven actually executed joint targets, one actually executed gripper target, and nine auxiliary EEF values predicted by P0. The EEF values were not executed independently and are not measured achieved poses. This remains a mixed imitation/distillation objective, not independent expert control.

## Separate pilot, training, validation, final test

Pilot layouts1000–1007 and both base cases are evaluator/development evidence only; exclude them from fitting. The first proposed feasibility collection uses training layouts2000–2003 with policyseed0, four bounded attempts in predeclared order and unchanged P0. Keep every attempt and its actual initial poses, failures, infrastructure problems, and sampling denominator. No repeat-until-success deletion or tuning after individual cases. Additional reserved training layouts2004–2031 require a separately documented batch; validation3000–3007 and final4000–4023 stay unused.

Before collection, resolve and freeze a task criterion that accepts deliberate supported rim placement. Preserve v3 raw results; any revised scorer must have its own version, rationale, tests and adversary approval, with retrospective replay explicitly labeled as development analysis. Two-camera visual/contact adjudication complements the automated proxy and must not silently replace its scores.

## Feasibility gate, before real optimization

1. Independently verify at least one fresh successful training-layout rollout: held lift, commanded measured opening, detached placement supported by original bowl/rim, and sustained low velocity. Review both cameras and contact/state/action traces. No privileged object state enters P0.
2. Retain immutable raw capture and evaluator/model/source identities, layout/policy seed, actual poses, collection split and the inclusion decision. Record if the saved episode ended on the metric or reached50s.
3. Convert synchronized pre-action camera/state and actual executed action rows to the pinned loader format. Keep the auxiliary P0 EEF label disclosure in the dataset provenance.
4. Run the real loader, training processor and collator in CPU-only mode. Independently check all17 normalized action/state dimensions and40-row windows, specifically including grasp/lift/release, plus loss masks, padding, RGB fidelity before augmentation and finite augmented tensors. Preserve P0 normalization.
5. Obtain adversary approval of the complete evidence. One successful example proves feasibility, not dataset coverage or an improved policy. P1 fitting remains checklist step7, after B's fixed interface is developed and frozen in step6.
