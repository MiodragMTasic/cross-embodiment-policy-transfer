# One-attempt launch checklist

Keep VM off until networking, balance, key availability and operator readiness are established. Do not guess current balance, IP, rates or approval fields.

1. Upload this entire package to `/home/miodrag/robot-transfer/bowl/experiments/roger_bravo_one_shot_v1/`. Preserve frozen dependencies. Copy permit.example.json to a separate launch permit and fill current values. Pin source-lock.json's SHA from LOCAL_REVIEW.md. Keep the exact checkpoint manifest pin and case id.
2. Confirm the served model and prices before GPU loading. Use current UTC date in prices_verified_date; input/output prices are dollars per million tokens, not copied estimates. API cap $1 is separate from cloud balance; reservations are an estimate, not a provider-enforced ceiling.
3. Supply NEBIUS_API_KEY privately via the existing secret-loading mechanism or a hidden terminal prompt. Never write it into the permit, repository, argv, screenshots or handoff. The policy-server child does not receive this key; only the vision runner does.
4. Start a console-stop timer tied to actual VM start. Whole-session seconds include startup/checks/copy, not just runner duration. Guest shutdown is a backstop and does not establish that Nebius billing stopped.
5. On VM, from package folder, run the read-only preflight:

   `python3 launch.py /absolute/path/to/launch-permit.json`

   This does not load models, schedule shutdown or call Nemotron. Fix reported blockers before --execute. It needs at least50minutes left before integrity checks and45after. It checks source/model/layout pins, dependencies, free space, idle GPU and no existing output.
6. Execute ONCE inside an existing persistent terminal/tmux session:

   `python3 -u launch.py /absolute/path/to/launch-permit.json --execute`

   Do not background it without a watcher/operator. No automatic reruns. The runner has a40minute timeout and stops at the five-minute copy reserve. Preserve a failed attempt too.
7. Copy both `runs/roger_B_P1_l4003_assisted_v1/` and `runs/roger_B_P1_l4003_assisted_v1-supervisor/` to local review evidence. Use verify_copy.py to check each copy's manifest, including video. Inspect exit-code, failure files and gpu-after.txt. If copy is incomplete, do not call it verified.
8. Stop the VM through Nebius console and verify Stopped by the budget deadline. Retain disk; do not delete checkpoints. If connection/copy fails, preserve remote disk and stop by deadline rather than allowing indefinite billing.
9. Score executed `gate_trace.jsonl` with the locked offline scorer. Report API decisions, confirmation, release marker, actual landing and failure separately. `ATTEMPT_COMPLETE` means lifecycle completion only. No results go into the original 48-run comparison.
