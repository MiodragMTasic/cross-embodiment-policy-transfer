# Pre-registration: Nemotron failure labeller, prompt v2, on the step 8 "after" episodes

**Status: draft, awaiting Miodrag's yes. Nothing here has run. No call has been made.**

Drafted on 24 October 2026 by Claude, an AI agent working for me (Miodrag), after an external audit by Codex. Every claim below about the v1 run comes from `runs/20261024T182236Z/` (`report.md`, `analysis.json`).

## Why

In run 20261024T182236Z, prompt v1 read text timelines of the 24 step 8 "before" episodes. Its blind majority vote matched my reviewer on 15/24, the same as always answering "success" (15/24). Balanced accuracy was 0.61: success recall 10/15, failure recall 5/9. My locked v5.1 scorer alone matched the reviewer on 23/24.

Most blind false failures came from rim or end-state readings. The v1 prompt never said that the verdict is fixed at the first rest. It said the rim counts, but it did not say that a rim rest with no table contact is a rest on the bowl alone. Those two gaps are a plausible contributor, not a shown cause. v2 was written after seeing the v1 errors, so it cannot be tested on the before episodes. This test uses episodes that do not exist yet.

## Prompt v2 (frozen)

- **File:** `nemotron/prompts/failure_labeller_v2.txt`
- **sha256:** `85d94bea0cd7a8ae59f6cc099fd4c7672c472072123ddb01edbfb7398e64966b`
- **v1 for reference:** `failure_labeller.SYSTEM`, sha256 `9480f9ab87345350a440035a59495c17041f364a595e6daebb2fb1b3b6a9c1cf`, the value recorded in every v1 annotation.
- **Version string:** `failure-labeller-v2`.

At call time the system message is the file's text followed by the JSON schema, as in v1. Each annotation's `prompt_sha256` must equal the sha256 above. `test_analyse_run.TestPrereg` fails if the file changes or if any v1 line other than the declared ones is altered.

**The v1 → v2 change, in full:**

1. Two rules are added after v1's success definition:
   - **First-rest rule.** The outcome is decided at the banana's first 1 s rest after the release in which it is supported by the bowl alone; nothing after that rest changes it.
   - **Rim rule.** Rim support is bowl support. A rest with B or BR and no T counts. A rest with any T does not.
2. Two failure-mode definitions are reworded so they do not contradict the first-rest rule:
   - `ends_on_table` now refers to a rest on the table before any rest on the bowl alone, not to the end state;
   - `ends_on_rim` now means never having a 1 s rest on the bowl alone because the banana keeps touching the table.
3. Nothing else changes: the task line, the success definition, the other nine failure modes and the citation instruction are byte-identical to v1.

**Sources for the added rules** (locked definitions; the mapping onto the summary's T/B/R state codes is mine):

- `experiments/scoring_v5/SCORING_V5.md`, "Rule": "the banana lands on the original bowl (rim counts) directly from that release; it then settles there."
- "Two readings": "first rest: the verdict is fixed when the first 1 s stable window completes, and nothing after it matters".
- v5.1 V1: "The 1 s rest must be supported by the bowl alone: any table contact resets the window."

## Held-out set

- **Episodes:** the 24 step 8 "after" episodes, A/P1 and B/P1 on layouts 4000–4011 at policy seed 0 (`step8_{A,B}_P1_l{4000..4011}_s0_v1`). P1 = checkpoint 1500, labelled "P1, no A-validation improvement" (EXPERIMENT_REPORT §22.20). The after phase launched at 20:52 UTC on 24 October; none of these episodes has been summarised or shown to any model.
- **Excluded:** canary episodes (`P0canary`). Crashes are unscored under R4 and are excluded by name. Safety aborts are scored and stay in, as in the locked scorer.
- **Distribution shift, disclosed:** the policy is P1, not P0. A change in agreement against v1 mixes the prompt change with the episode change. This test measures v2 on held-out episodes; it does not isolate the prompt's effect (see the optional arm).

## Order of operations

1. **Now:** freeze this file and the prompt. Record the code hashes below.
2. **Before any call:**
   - the after half is complete and scored by the locked `score_step8.py --phase after`;
   - my reviewer's after-half adjudication is final.
     The reviewer never sees Nemotron output.
3. **Before any call, the labeller code is pointed at the after evidence:**
   - it reads the sealed after copy, with every file checked against the VM-computed sha256 list, and `score_step8-after.json`;
   - `reviewer_reference()` is transcribed from the after adjudication, each entry with a verbatim quote that a test checks against `STEP_REVIEW_LOG.md`;
   - v2 is loaded from the file above with a hash check.

   That change gets its own review, and its hashes go into an addendum at the end of this file. It may not change the prompt, the schema, the summary thresholds or functions, or the metric functions.

4. **The run:** 24 episodes × 3 repeats, blind arm only, 72 calls.
5. **Report:** all metrics below, whatever they show, then the claim rule.

**Code fixed now (sha256):**

| File                  | sha256                     | What must not change                                                                                                    |
| --------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `failure_labeller.py` | `cca41206df32414fd8c8e57d52186d74803733f7d770eaf4c7e602f0b3add4d3` | the summary thresholds and `summary()`, `events()`, `_items()`, `_state()`, `_abort_lines()`, `schema()`, `grounding()` |
| `analyse_run.py`      | `c1a7814fe3ddc0a55ceb5454367482e1f1978623e7462aeb2b6169ad55f4b078`      | `confusion()`, `majority_baseline()`, the majority vote (`run_all._majority`)                                           |
| `run_all.py`          | `e6555f71140b728b6bc39a45e6b75c3977e89773c4def74b73aa5a515dada441`          | request building: `seed()`, the enum-order shuffle, `max_tokens` 800, temperature 0                                     |
| `tf_client.py`        | `04688ce1ad186c03e3ec0c32edf4ccc0ce38075a0ee473278646b02c2c22ec54`        | `build_request()`, the $2.00 cap, the retry rule                                                                        |

## Fixed settings

These match v1 except the prompt:

- model `nvidia/Nemotron-3_5-Lightning` on Nebius Token Factory;
- `response_format` json_schema with strict true; temperature 0; `max_tokens` 800; `chat_template_kwargs.enable_thinking=false`;
- repeat 0 uses the schema's default enum order, and repeats 1 and 2 shuffle it with `seed(case, 'blind', k)`;
- one retry under `chat_json()`'s rule.

## Metrics, declared now

The reference is my independent reviewer's adjudicated outcome for each episode (success or failure). If the reviewer gives no verdict for an episode, that episode is excluded by name and n shrinks; the claim rule below still applies to the n that remains.

**Primary: the majority of 3 repeats per episode.** An outcome tie, an `uncertain` majority or no valid response counts as wrong. Reported:

1. **The confusion matrix** against the reviewer: success and failure rows; success, failure, and uncertain/split/none columns.
2. **Per-class recall:** reviewer successes judged success, and reviewer failures judged failure.
3. **Balanced accuracy:** the mean of the two recalls. If one class is absent from the reference, it is the recall of the class present.
4. **The always-majority baseline:** always answer the reviewer's most common class in this set (a tie goes to success). Its agreement count is reported; its balanced accuracy is 0.50 by construction.
5. **The locked v5.1 scorer's agreement with the reviewer** on the same episodes: confusion matrix, recall and balanced accuracy.

**Secondary, descriptive only:**

- per-call agreement and balanced accuracy;
- the split by robot;
- failure-mode agreement where the reviewer names a mechanism;
- repeat consistency;
- cited controls present in the summary. Presence is not semantic support.

`analyse_run.py` computes all of these from the saved outputs with no API call.

## Claim rule

With n = 24, one more correct episode moves accuracy by about 0.04, so a difference of an episode or two cannot be told apart from chance. The bar is set high for that reason.

- **Useful.** I will call Nemotron (with prompt v2, reading text timelines) useful as an outcome judge on these episodes only if **both** hold:
  - balanced accuracy is **≥ 0.80**;
  - majority-vote agreement **beats the always-majority baseline**, with strictly more episodes correct.
- **Not demonstrated.** If either fails, the result is reported as "usefulness not demonstrated", with the numbers.

Even a pass is a preliminary result on 24 episodes from one policy, not a general claim. Its labels stay annotations: never merged into the locked v5.1 counts or the reviewer's verdicts, and never used to change a score. The v5.1 agreement is printed next to it. Whichever way it falls, it is not re-run with a new prompt and reported as the same test.

## Cost

- **Expected:** about $0.01 for 72 calls. v1's blind arm cost $0.0087 for 72 calls, an estimate from returned token counts and recorded prices, and v2 adds about 190 prompt tokens per call.
- **Pre-run estimate:** the run script's conservative estimate (characters/3) is about $0.016 likely and $0.023 worst case.
- **Cap:** the ledger stands at $0.019773 of the $2.00 cap.
- **Verification:** none of these figures is a verified billing deduction.

## Optional arm, not part of this registration unless approved separately

Run v1 on the same 24 after episodes as well: 72 more calls, about $0.01. That would make a paired, versioned comparison on the same held-out episodes, and it is the only way to test whether the first-rest and rim rules themselves reduce errors. If approved, it is added here before any call, with its own claim rule: the v2 − v1 difference in correct episodes, paired by episode, reported as descriptive with an exact two-sided McNemar on the discordant episodes.

## Approval

- [ ] Miodrag: yes to the 72-call v2 run as registered (about $0.01).
- [ ] Miodrag: yes or no to the optional v1 arm (about $0.01 more).

## Addendum (filled in before the first call)

_Empty._
