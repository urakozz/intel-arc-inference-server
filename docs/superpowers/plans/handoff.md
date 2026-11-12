# Handoff: Ornith MoE prefill (plan 2026-11-12), 2026-11-12

The plan is done (Tasks 1-8, record in `docs/probes/probe-moe-prefill-2026-11-12.md`, "Task 8"). Two things are left:

- The W4A8 serving check (open-work item 16): `.pmp/t5r_restart.sh` makes the f32 references for W4A8's 76 serve
  contexts on the CPU (~10 h from 2026-11-12 06:33, log `~/b70/logs/ornith-moe-prefill/t5r2.out`), then judges
  them (`t5c/serve_judge2.log`). Record the result in item 16 as a later commit.
- Not deleted (Task 8 Step 3): Ornith's old grouped-GEMM binaries and `pf_moe.cl`'s sort / dequant entries, which
  only `pf_moe_test` still builds.
