# mtp-lookup handoff (open-work 27)

State: code done and committed (server + EngineAdapter + tests). Box runs under ~/b70/logs/mtp-lookup/:
- gpu_test_run4.log: mtp_lookup_gpu_test with the f32 judges and bf16top references (run1-3 before them).
- ab1/ (ab.sh, arm_replay.py): the speed A/B, arms lk (--spec auto) / nolk (B70_SPEC_LOOKUP=0), 3 rounds;
  tables: `python3 tools/box_validate/serve_client.py table --dir ~/b70/logs/mtp-lookup/ab1/{replay,sets} --ref nolk --ratio lk/nolk`.
Left: read both, fill open-work item 27, report.
