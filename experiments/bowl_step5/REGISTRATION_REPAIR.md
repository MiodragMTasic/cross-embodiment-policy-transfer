# v3 relocated-task registration repair

The first relocated launch `step5_controlled_layout1000_v3` failed before environment creation or actions. The source resolver defaults to benchmark/ and did not find the variant in randomize_initial_pose/. The simulator process returned0 despite that startup traceback; the batch integrity check caught the missing metric/capture and stopped. Keep this attempt as an infrastructure failure, outside completed task-trial outcomes.

Separate registered launcher/wrapper copies now specify the variant subdirectory. After registration, they verify the exact intended environment class/file and use its upstream task-name alias only for the runner selection. The original requested class remains in capture/metric metadata. Runtime assertions verify both randomized objects, the ±0.1m xy range, collision check and50s duration. This changes task resolution only; v3 scoring, P0, observation/action interface, geometry and control timing are unchanged. Original v3 files and failed-launch logs are preserved.

The retry uses unique case `step5_controlled_layout1000_v3_retry1`; remaining layouts1001–1007 retain their original planned names. A separate r1 dispatcher ledger and source lock preserve the failed first batch. Retry only after adversary review.
