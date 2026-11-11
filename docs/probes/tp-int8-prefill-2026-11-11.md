# TP int8 prefill validation, 2026-11-11

Branch `tp-int8-prefill`, rebased onto main `02ec4323`. Implementation and validation
fixes through `cdf28a36`; subgroup lane tuning in `538d97e3`. No push.

## Rotation choice

The selected path zero-extends local down K=8704 to 9216 and uses 1024-wide
Hadamard blocks throughout. Both inputs and weights are padded inside their
kernels; checkpoint addressing remains raw K. It preserves TP's wavefront,
requires no extra crossing, and the memory plan includes padded int8 scratch.

The 512-wide alternative was measured and rejected by f32 quality below. Full
17408-wide rotation before splitting would mix channels across the 8704 boundary
and require activation and weight data from the other card.
Per-card int4 linear shapes (K × N): QkvZ 5120 × 8192, Qkv 5120 × 7168,
gate/up 5120 × 17408, out/o 3072 × 5120, down 8704 × 5120. Every N is whole
1024-column slabs; down alone needs a smaller rotation block. Decode and bf16 TP
prefill retain their execution paths.

## Initial timings

Idle box, both GPUs held by `tools/gpu_lock.sh box`, CPU-model lock held to keep
reference jobs out. Three runs of each backend at each length, interleaved (the
second round reverses order), chunk cap 2048, int8 head, bf16 KV, max_len 32768,
tg 32. The prefill timer excludes loading and scale preparation.

| Prompt tokens | TP bf16, t/s | TP int8, t/s | Ratio |
|---|---:|---:|---:|
| 4096 | 2939.97 | 3274.29 | 1.114 |
| 16384 | 2684.03 | 3003.44 | 1.119 |

These are `cdf28a36`'s medians, before lane tuning. They are below the brief's
~3700 / ~3400 targets. Its 512-wide weight rotation used eight active lanes with
64 values each. The revised variant uses all sixteen lanes with 32 values each,
retaining the exact ascending FWHT arithmetic. `pf_int8_test` passes with zero
quantisation/scale mismatches at both layouts; throughput remeasurement follows
quality validation, after the CPU reference work finishes.

## Validation evidence

Initial logs: `~/b70/logs/tp-int8-prefill/validation-324952f5/`.
Completion logs: `~/b70/logs/tp-int8-prefill/completion-cdf28a36/`.

- Targeted and production builds passed on the box. The fresh targeted build
  exposed missing short-chunk bf16 kernel dependencies; the shared test list now
  names the variants the runtime actually selects.
- `pipeline_args_test`, `pf_int8_test`, `linear_i8_test`, `prefill_int8_test` and
  `tp_prefill_test` passed. The tuned rotation also passes `pf_int8_test`.
- Initial `tp_prefill_int8_test`: 31 of 40 contexts had identical ids; nine await
  their f32 references. The test teacher-forces one card's bf16 continuation on TP
  and requires every differing id to pass the f32 judge at that same context.
- The nine references are being generated under `/tmp/b70-cpu-model.lock`.
  An inherited scratch symlink was inaccessible inside the oracle container;
  its link was moved to `~/b70/trash/2026-11-11/` and a local scratch directory
  now holds this branch's reference work.
- The TP int8 prefill + int8 KV CLI run completed and printed the existing known
  fail warning (`kv8-warning.log`). No attempt was made to fix that combination.
- Server runs use `serve_run.sh`, max_len 32768, prefix cache off, spec off,
  temperature zero, 256 output tokens, three golden and 36 tool-call requests.
  The bf16 arm answered all 39; the int8 arm and f32 comparisons are in progress.
- A separate, deliberately corrupted cjk judge is queued as the negative control.
  It never overwrites a real reference. Real-reference quality gates and final
  speed measurements remain pending.

## The 512-wide path failed quality

The nine f32 references completed in 722 seconds (8505 trie positions, peak RSS
36 GiB). The real-reference gate failed. Examples: long row 62, 22.96 bf16 ulp;
t1_define-openai row 41, 19.58 ulp; t2_explain-loader row 32, 30.67 ulp and f32
rank 12. These exceed even the engine's 8-ulp near-miss bar. The isolated bad-judge
negative control failed as intended (`negative-control-status`: PASS).

The remaining reference work for the obsolete server streams was stopped through
`tools/box_stop.sh tree tp-int8-prefill`; its scope is processes working in this
branch's remote tree and oracle containers bound to it. GPU and CPU locks were
verified free afterwards. Completed real references were preserved, and partial
work was moved to `~/b70/trash/2026-11-11/tp-int8-prefill-512-reference-work`.

The replacement zero-extends each card's local down input and weight to K=9216
before the normal 1024-block rotation and int8 GEMM. Raw input/weight addressing
still uses K=8704. No extra activation crossing or persistent weight copy is
needed. The scratch planner now includes the padded width. This is an
orthonormal rotation of the zero-extended dot product; it requires its own gate.

Padding tests first failed against the old kernel (required workgroup mismatch)
and old scratch plan (8704 versus 9216). The replacement build and kernel,
linear and memory-plan tests pass. Model gates are running under
`~/b70/logs/tp-int8-prefill/padded-1024/`; quality and speed are not established.

The initial gate used a one-ulp cutoff. It is now aligned to the requested
engine rule (`docs/14-golden-gate.md`, `spec_judge.h`): differing ids also pass
when at most `golden::kNearMissUlp` = 8 below the f32 top. The rejected 512 path
still fails this rule. The padded path's known differences are 1.38–2.55 ulp;
five additional references are running detached before the final verdict.
An optional one-card diagnostic keeps the same teacher-forced contexts and
separates native int8 behaviour from TP differences.
