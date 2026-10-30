> **Public copy, 26 October 2026.** This package ran once, as ABBA (Alpha Bravo Banana Away): case `roger_B_P1_l4003_assisted_v1`. The README below was written before that run and is kept as written. For the result, caveats, cost and the re-score command, see "ABBA" in `../../README.md`; the small evidence set is in `../../evidence/abba_l4003/`. In this copy `preflight.py` and `permit.example.json` carry `<vm-instance-id>` in place of the cloud VM's instance ID, so `preflight.py` differs from its `source-lock.json` entry, which pins the original. `tools/verify_locks.py` at the repository root reports that entry as SANITIZED and the other 10 as OK (see `../../SANITIZATION.md`).

# Roger Bravo, Banana away — one exploratory assisted attempt

Prepared 26 October at Miodrag's explicit request. **Local code is ready to attempt; live simulator behavior and success are unproven.** No VM, SSH, GPU or API was used in preparation.

This is a new, isolated vision-only test. It deliberately does not wait for the older RGB-D capture/perception gate. Those packages and their NO-GO decisions remain unchanged. Saved-frame near-rim errors are known; this test measures whether the imperfect vision supervisor can cause a useful release, not whether it is reliably safe or improves transfer in general.

## Exactly one episode

- B / P1 checkpoint1500 / layout4003 / seed0 / existing painted fixed adapter.
- 3,600 controls (240 simulated seconds) maximum, with existing native/task/joint-safety stops. Known selected case, not a new unbiased evaluation. No baseline, reruns or retraining.
- Actual exterior and wrist RGB go to Nemotron-3.5-Super-VL. P1 still receives its existing painted observation. No depth or object coordinates enter Nemotron's decision.
- Check roughly every 12 controls (0.8 simulated seconds) while P1 commands closed. Physics pauses during each bounded API call.
- READY holds the last executed arm target. A second decision sees three NEW post-hold frames; only confirmed readiness plus visually still motion opens the gripper.
- Gripper remains open thereafter; arm holds for90controls, then P1 arm motion resumes with the adapter lift ramp returning to zero. Three failed confirmations disable interventions and the episode continues/counts. Three transport/schema failures or an allowance/deadline stop terminate and retain failure evidence.
- `Roger Bravo, Banana away` is logged after the open action reaches env.step. Video labels say release commanded, outcome pending. This is not a success announcement; score the saved executed-command trace afterward.

## Local checks

43 CPU tests pass: fresh confirmation, persistent opening, held target, expiry, bad replies, transport timeouts/accounting, key isolation, source checks, launch no-op, supervisor failure cleanup and copy manifests. Launch sequencing uses fake processes, not Isaac. All Python parses. See LOCAL_REVIEW.md and tests/.

## Launch-day blockers (not more development experiments)

1. Working SSH to the stopped VM's current address; confirm no other task is running.
2. Fresh balance/rate/start time and a console-stop timer owned by Miodrag. Default one-hour whole-session estimate is $1.80 compute at historical rate plus $0.50 reserve: needs at least $2.30. Current balance is unknown. Timing/model startup may consume the opportunity; no guaranteed completion.
3. Fresh model/price check and Token Factory key supplied privately in the runner environment; never put credentials in permit or commands/history.
4. Fill a launch permit with current facts and actual approvals, including the guest-halt backstop. Example remains intentionally non-launchable. No spending authorization is inferred from this README.

Use LAUNCH_DAY.md. No second capture-only episode is required by THIS exploratory protocol. One run can end with no intervention, early/wrong release, task failure, or infrastructure failure; all must be reported.
