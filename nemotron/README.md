# Nemotron annotation and intent tooling

This directory exports the authored tooling for text-timeline labels and structured task specifications using Nebius Token Factory. Historical annotation results are reported in the top-level README; labels do not replace the locked scorer. This code does not itself demonstrate robot control. The separate Charlie packages contain the DROP supervisor, prepared and never run; the vision-based release supervisor that did run once, ABBA, is in `experiments/roger_bravo_one_shot_v1/` and calls Nemotron 3.5 Super VL with its own transport, not this tooling.

`tf_client.py` handles API transport and a spending ledger, `failure_labeller.py` labels episode timelines, `intent_contract.py` parses instructions, and `analyse_run.py` compares annotations. Prompts and synthetic tests are included. Credentials, raw model payloads and spend ledgers are excluded. Supply your own credentials through the documented environment/key-file mechanism; never commit them. Recorded price constants are historical estimates, not a current price guarantee.

## Offline tests

From repository root, with Python dependencies installed:

```sh
PYTHONPATH=nemotron python -m unittest test_tf_client test_analyse_run.TestMetrics -v
```

This tests transport guards and metric calculations without API calls. Optional SDK-dependent checks skip if the OpenAI-compatible client SDK is absent. Other tests require the full sealed episode evidence, pinned RoboLab task sources, original reviewer records or original saved API runs; they are not a self-contained exported test suite. See `common.py` for expected relative inputs and `OFFLINE_EXPORT_CHECKS.md` for exact tested scope.

Do not infer live-model quality from offline tests. Commands that request a smoke test or real model inference make paid external calls; no such calls were made during export verification. Historical held-out preregistration is included as a plan, not a completed result.
