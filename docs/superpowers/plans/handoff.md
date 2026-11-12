# Handoff: Ornith MoE prefill (plan 2026-11-12), stopped 2026-11-12

Branch worktree-agent-a4e422addb46c41c6 on main 7cfbf058. Tasks 1-7 done; Task 8's switch-over (W4A8 is l0-int8's
only expert path, no env variables, the near-miss gate rule) is committed.

Stopped on: `prefill_gate_ornith_l0-int8_test` and `_c16_test` fail. Both misses are under 8 bf16 ulps but lose the
rule's cosine half:

| gate | row | f32 margin | engine cos vs f32 | bf16 twin cos vs f32 |
|---|---|---|---|---|
| l0-int8 (chunk 2048) | cjk 15 | 3.0 ulp | 0.999465 | 0.999467 |
| l0-int8 c16 | prose 18 | 7.2 ulp | 0.995337 | 0.997036 (the twin also picks 32) |

Every other gate in the run passes (pf_moe_test, l0 gates, splits, ornith_prefill/decode, Flash-Next synthetic);
Ornith l0 and Flash-Next synthetic routes are bitwise main's. Needs an operator ruling before the gate can land.

Record (card 0; card 1 held by another agent; the CPU f32-refs job running; median of 3, --max-len auto):
4k l0 540.5 -> 506.9 ms, l0-int8 750.2 -> 480.9 ms; 16k l0 2517 -> 2409 ms, l0-int8 3365 -> 2321 ms.
Logs: ~/b70/logs/ornith-moe-prefill/t9 and t10. Not yet written: the probe-doc record, open-work, README, BENCHMARKS.
Still running: t5r_restart.sh (f32 refs for the W4A8 serving check, ~10 h, log t5r2.out).
