# Open questions

Ranked by how much the answer would change the design. Nothing here is settled -
do not let an estimate from these docs become an assumption in a spec.

## 1. What is `W`, the bytes read per decode token? - **resolved for phase 1**

**15.52 GB per token** for `Vishva007/Qwen3.8-27B`, measured 2026-08-22 from
the safetensors headers with the index as the manifest (doc 03): 12.163 GB
int4 `qweight`, 0.760 GB f16 scales, 2.543 GB bf16 `lm_head`, 0.052 GB bf16
small tensors. `qzeros` (0.190) and `g_idx` (0.012) are dropped at load; the
embedding is gathered; the vision tower is skipped. The checkpoint has no
duplicated tensor names, so header arithmetic is exact here - the
dedup-by-name rule stays in the loader for checkpoints that do.

Reproduce: `python3 tools/probe/checkpoint_bytes.py <snapshot>`.

Consequence: vLLM's 31.50 t/s is **81% MBU**. The host-overhead thesis is
nearly exhausted on this model (doc 05); it remains the main lever for the MoE
phases, where `W` is still an estimate. Confirm resident bytes on first load.

## 2. The phase-1 model does not currently load in vLLM - **resolved by re-scoping (#14)**

Phase 1 is now the 27B, which loads and has a baseline. The loader
requirements at the end of this entry still stand unchanged - the 27B ships
the identical `model.language_model.*` / `model.visual.*` / top-level
`lm_head.weight` layout. Kept for the record:

`letechlead/Ornith-1.5-9B-INT4-W4A16-AutoRound` declares
`architectures: ["Qwen3_5ForConditionalGeneration"]` and ships tensors named
`model.language_model.*` (2177), `model.visual.*` (549), `lm_head.weight` (1).

vLLM (`v0.27.2rc1.dev342+g0a5a55136`, image `p314-t214-vxkp0`) instantiates
`Qwen3_5Model`, whose parameters are `layers.*`, and fails:

```
ValueError: There is no module or parameter named 'language_model' in Qwen3_5Model
```

Fails identically with and without `--language-model-only`.

Two consequences:

- **The "beat vLLM" baseline for phase 1 has no number yet.** Either fix the vLLM
  side, or pick a phase-1 checkpoint that loads, or accept the 27B numbers as the
  reference and re-scope phase 1.
- **Our loader must handle this layout natively**: strip the
  `model.language_model.` prefix, skip `model.visual.*` entirely, and treat
  `lm_head.weight` as top-level. Design for it rather than requiring
  re-exported checkpoints.

## 3. Is GDN or GEMM the larger share of decode time?

48 of 64 layers are linear attention, and GDN is the one kernel with no
`sycl-tla` starting point. If GDN dominates, it should be the first kernel
written and the first optimised. If GEMM dominates despite GDN being 48/64
layers, the decode GEMV carries more of the outcome than the layer count
suggests.

Measure on the current stack with a profiler before choosing what to build first.

**Estimate (2026-08-22):** GDN state is 2 MB per layer (doc 03), ~100 MB per
token across 24 layers - ~2% of `W`. GDN should be a kernel-count problem, not
a bandwidth one, and replay removes most of the kernel-count cost. Expect GEMM
to dominate. If a vLLM profile says otherwise, the Triton launch path is the
first suspect.

## 4. What is vLLM's MBU on the phase-1 model? - **resolved: 81%**

31.50 t/s × 15.52 GB = 489 GB/s of 600. The 50-63% figure was MoE-only. The
projected headroom for phase 1 is therefore ×1.23 to the bf16-`lm_head`
roofline and ~×1.35 with `lm_head` at int4 (doc 05) - not 1.4-1.5× from host
work alone. The thesis survives; its weight shifts to the MoE phases and to
phase 2.

## 5. What is the per-kernel fixed cost inside a replayed command list?

Measured 2026-08-22 by `tools/probe/probe_replay`: one in-order regular list
of N launches, closed once, replayed 1000 times behind a fence (median, after
20 warm-ups).

| kernel | N | us/replay | us/kernel |
|---|---|---|---|
| (empty list) | 0 | 6.4 | - |
| noop | 1 | 9.9 | 9.89 |
| noop | 250 | 134.6 | 0.54 |
| noop | 700 | 360.7 | 0.52 |
| ctrl_read | 1 | 5.9 | 5.90 |
| ctrl_read | 250 | 161.4 | 0.65 |
| ctrl_read | 700 | 438.6 | 0.63 |

Per-kernel cost is 0.52 µs (`noop`) and 0.63 µs (`ctrl_read`, which reads one
dword of a shared-memory control block - the shape of every decode kernel's
first instruction) → **< 1 µs, so fusion is not on the phase-1 critical path**
(spec 1 §4.1 rule: ≥ 3 µs yes, < 1 µs no) and the unfused ~645-kernel list
(spec 1 §9.1) ships first at ~0.4 ms of a ~26 ms step.

The empty submit + fence round trip is **6.4 µs** - the floor no fusion
removes, paid once per token. The N = 1 rows are that floor plus one kernel,
not a per-kernel number.

## 6. How much accuracy does quantising `lm_head` cost?

On the 27B it is 2.54 GB → 0.66 GB per token at int4 (×1.14 on the ceiling)
or 1.27 GB at int8 (×1.09) - doc 05. `lm_head` is known to be
quantisation-sensitive and the checkpoint author chose not to (`lm_head:
false` in `quantization_config`). Needs a perplexity or `lm_eval` comparison
against the unquantised baseline before adoption. In phase 2 the same question
applies to the 0.85 GB bf16 MTP head, and `lm_head` is read twice per step.

## 7. Can 24 MB of L2 be exploited deliberately?

Unusually large. A 4096×4096 int4 tile is 8 MB. Whether multi-layer weight
residency is achievable, or whether the replacement policy defeats it, is
unknown. Measure before designing around it.

## 8. Is `MainloopIntelW8A8` worth it for prefill?

int8 XMX has roughly 2× bf16 throughput and prefill *is* compute-bound. But it
requires int8 activations, i.e. dynamic activation quantisation and its accuracy
cost. Out of scope for phase 1; revisit once decode is winning.

**Largely superseded by #9.** W4A8 gets the same int8 systolic path while keeping
weights at 4 bits - same compute win, half the memory traffic, and no need to
re-quantise the checkpoint. Answer #9 first; if it holds, W8A8 is uninteresting.

## 9. Is native mixed s8 × s4 DPAS reachable, and what does it cost?

`oneDNN/src/gpu/intel/gemm/jit/pd.cpp:772` documents a **native int8 × int4 DPAS**
on all pre-Xe3p hardware - Battlemage included - gated only on the int4 matrix
having no zero points. Our checkpoints are all symmetric, so the gate opens.

If real, a **W4A8 prefill** path gets int8 systolic throughput *and* zero
dequantisation, which neither vLLM nor OpenVINO uses (both run W4A16). That would
beat both rather than matching one.

**Partly answered (2026-08-22):** `sycl-tla` *does* declare the atoms -
`include/cute/arch/mma_xe.hpp:287-297`, `XE_DPAS_TT(d, s8, s4, d)` plus every
u/s permutation, compiled out only for CRI. What it lacks is a mainloop that
feeds them: `xe_mma_mixed_input.hpp` widens int4 to 16-bit. So the question is
no longer "is it reachable" but "write a CuTe mainloop on an existing atom, and
does it win".

Still unknown:

- Whether the s8×s4 DPAS runs at the int8 rate on a B70 (it might be
  microcoded to the same cost as s8×s8 plus a widen - measure with a single
  atom in a loop before writing a mainloop).
- What dynamic int8 activation quantisation costs in output quality.
- Whether the win survives the quantise-activations step, which is itself a pass
  over the activations.

Cheapest probe: benchmark oneDNN directly at prefill shapes, W4A16 vs W4A8, before
building anything.

## 10. Does MXFP4 need original kernel work on Battlemage?

`sycl-tla`'s block-scaled examples are `50_xe35_*` / `51_xe35_*` - Xe3.5, not
BMG. Phase 3 may need to write or adapt block-scale handling rather than lift it.
Confirm by attempting to instantiate the block-scaled mainloop for a BMG target
before committing to the phase-3 schedule.

## 11. Which tokenizer dependency, and what does it cost goal 3?

Not mentioned in the first draft of these docs at all. Every request passes
through BPE encode; every token through detokenise. Options and trade-offs in
[11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md). The
decision is between parity (HF `tokenizers` via FFI - exact, but a Rust
toolchain in the build) and ownership (a hand-written byte-level BPE - no
dependency, parity must be proven on a corpus). Resolve before the server
milestone, not before the decode core.

## 12. What does a fixed-grid attention kernel cost at short context?

Under replay the attention grid is sized for `max_model_len` and idle
work-groups exit early (doc 04, "Attention under replay"). Measure the step
time at `seq_len = 64` vs `seq_len = 4096` with the same list. If the
difference exceeds ~2% of a step, capture context-bucketed lists instead.

## 13. Can a `sycl-tla` kernel be appended to a raw L0 command list at all?

Doc 04 recommends keeping `sycl-tla` out of the decode list, which sidesteps
this. But phase 3's grouped GEMM may want a CuTe kernel at decode. Probe:
take `00_bmg_gemm`, get its `ze_kernel_handle_t` via
`sycl::get_native<backend::ext_oneapi_level_zero>` on a named kernel bundle,
append it to a regular command list, replay twice, diff. Cheap, and it settles
whether the two toolchains can ever share a list.

## 14. Which checkpoint is phase 1? - **resolved: `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`**

No working 9B exists on the box (2026-08-22). Phases 1 and 2 share the 27B;
its architecture, byte accounting and baseline are in doc 03 and
BENCHMARKS.md. Open sub-question: the implied MTP acceptance rate on this
model is unknown - vLLM's 42.56 / 45.23 at 1 / 2 drafts bounds it from below
but does not separate acceptance from step cost. Log it from vLLM
(`--speculative-config` metrics) before sizing the phase-2 ceiling.
