# lookup-wide-verify handoff (open-work 28)

Branch `lookup-wide-verify` (worktree `.claude/worktrees/lookup-wide-verify`), off main 061a7012. Record:
`docs/probes/probe-lookup-wide-verify-2026-11-14.md`. Box tree `b70/trees/lookup-wide-verify` (built), logs
`~/b70/logs/item28/probe/`.

## State
- Step 1 done (CPU): replay recording 23.3 % of ids in lookup runs >= 8, edit / copy 74 %, the real opencode session
  10.7 % (14.4 % over both recordings) - the stop rule fails on the real session. Tool: `tools/spec/lookup_runs.py`.
- Step 2 done (probe): a 16-row verify derived at 1.51 plain steps at depth 4096, 1.75 at 32768 (bar 2.2).
- Step 3 NOT started: stopped for the operator (the real-session result, and the GDN base below).

## Step 3 build plan (when it goes)
1. GDN rows without a stored state per row: item 25's DEFERRED gdn_step (branch `verify-cost-m8`, PR = 8 rows) to
   PR = 16. Today's slots form at 16 rows needs 12 more GDN states (+1.8 GB VRAM). Merging verify-cost-m8 into this
   branch conflicts in `src/runtime/capture.cc` only; wait for item 25 to land on main and rebase.
2. Conv ring 16 -> 32 deep everywhere it is Qwen3.8's / Ornith's (gdn_step needs RING >= M + 3; a rejected row 13-15
   would overwrite the window the next step reads): `PersistentDims::kConvRing`, every `RING 16` in gdn_step.cl
   (already `#ifndef` in the default branch), pf_gdn_conv.cl / pf_gdn_front.cl, qwen4exp's mirror, tp_gather. Bitwise
   by construction; snapshots grow by the ring (Qwen3.8 +7.9 MB).
3. One 16-row list (not 9..15): `DecodeScratchDims::kM` 8 -> 16 (79 uses), hh rows, the M = 16 builds of prep_res_fold,
   prep_norm_finish(_strided), prep_silu_mul(_s1), prep_gated_head, embed_gather, argmax_stage1, attn_prep(_s1),
   attn_v2 (its reduce), gemv_bf16 a||b, the head's KV fill; gemm_sm rows are already built at 16 (kSmAll);
   attn_verify_dpas at 16 with KS 1 DSPLIT 128 (the probe's `K32_Pg_S1_D128_Q1_G256_P1`, N1 / N2 pass). bf16 KV first;
   int8 KV keeps K <= 3 / 7 until its KV8 builds exist.
4. Engine: verify(15) on the wide list (lookup only; drafts padded to 15 when the match is shorter - padding is
   rejected like any wrong draft). EngineAdapter::lookup_run: k up to 15 when the policy asks.
5. Policy (server::LookupInDFlash, no flag): first lookup verify after ENTER at today's K; when it keeps every draft
   and the source continues >= 8 more ids, the next verify is wide; stay wide while a wide verify keeps >= 8.
6. Check (once, batched): edit / copy + opencode-session-int8 replay with item 27's harness
   (`~/b70/logs/mtp-lookup/ab.sh` pattern, arm_replay.py), `mtp_lookup_gpu_test`, the verify tests at 16,
   `dflash_gpu_test`.

## Notes
- `src/kernels/gdn_step.cl`: `RING` made overridable (no existing build changes).
- At 32768 the in-list verify at M = 5 measured 1.658 (M = 6: 1.184) - an outlier for item 25 to look at.
