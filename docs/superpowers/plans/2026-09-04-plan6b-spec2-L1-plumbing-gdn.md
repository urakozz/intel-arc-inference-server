# Spec 2 - Stage 1 / L1: plumbing (S4) + chunked GDN (S2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Engine::prefill()` exist and pass spec-2 §6's correctness bars on
widened GEMV and a temporary `C ≤ 64` decode attention - the buffer split, the
runtime-`M` kernel set, the WY-representation chunked gated delta rule
(`gdn_chunk`), the `Control` handoff, `b70-decode --pp`, and the three prefill
gates. Nothing about speed: L1 exists to put the hardest correctness problem
(48 layers of differently-rounded recurrence) on the table first. Yield: prefill
becomes *possible*; throughput is whatever widened GEMV gives
(**estimated 5-10× the 29.5 ms/id ingest**, spec §5 L1).

**Architecture:** `DecodeBuffers` splits into `PersistentBuffers` (shared state)
+ `DecodeScratch` (kM = 8) + `PrefillScratch` (kC = 4096), with `DecodeBuffers`
surviving as a **view** so `capture.cc`'s 129 `res_norm` sites and every other
binding are untouched and the 774/19 decode invariants hold byte for byte. A
second, *dynamic* execution path - `runtime::prefill::Context`, an in-order
asynchronous L0 immediate command list with runtime kernel arguments - runs a
new `pf_*` kernel family under `src/kernels/prefill/` whose `M` is a kernel
argument, not a `-D`. `gdn_chunk` is ours by default (nine kernels in three `.cl`
files, the FLA algorithm op for op); one task short-circuits to Intel's CuTe
kernel if plan 6a's P5 rules it fits. The 16 full-attention layers run a new
runtime-shape `pf_attn_prep` plus decode's **unmodified** `attn_decode` /
`attn_reduce` at a new `M = 64` binary, which is why L1's attention chunks are
capped at 64 and why plan 6d retires that route.

**Tech Stack:** OpenCL C 3.0, ocloc AOT (`bmg-g31`), pure Level Zero (no SYCL in
this plan - plan 6a owns `Context`'s SYCL half), C++17 `-Wall -Wextra -Werror`,
ctest on the box (box, `tools/box.sh`, JOBS 44), the reference
container (read-only) for the long-prompt oracle.

**Spec:** `docs/superpowers/specs/2026-09-04-spec2-prefill-design.md` - §3.4
(GDN chunk), §3.5 (everything else widened), §3.6 (execution model + buffers),
§5 L1 (this plan's scope), §6 (the bars), §8 (constraints, copied verbatim
below). **Interface contract:**
`.superpowers/sdd/2026-09-04-plan6-spec2-prefill/interfaces.md` - binding; the
four changes this plan needs are listed under "Interface change requests" and
must be ruled before Task 6 starts.

**Explorer evidence:** `explorer-1-engine-readiness.md` (what exists, with
file:line), `explorer-5-vllm-prefill-baseline.md` §2 (the FLA chunked
delta-rule vLLM runs).

---

## Global Constraints

Spec §8, verbatim:

- Box workflow `tools/box.sh`; JOBS 44; the **system** oneAPI toolchain
  (`/opt/intel/oneapi/compiler/2026.1/bin/icpx` via full path or
  `setvars.sh`, `docs/10-the-box.md:42`) - the container's SYCL is the
  operator's and stays read-only; `sycl-tla` built once as a static
  dependency, version pinned in the build.
- Probes run on card 1 (`ZE_AFFINITY_MASK=1`) while a server holds card 0;
  record rows only on a provably idle box (zero DRM holders).
- Every number labelled measured vs derived vs estimated/external, with
  grade and conditions; no two values for one quantity without a
  reconciliation sentence.
- Nothing pushed; tags local; work on the branch the operator names.
- Design must not preclude prefix caching: chunk boundaries at multiples
  of 1024 align with the queued block-snapshot scheme
  (`docs/04-architecture.md`, "Follow-on"); no work on it here.

Spec §6 item 6, verbatim:

- `-Wall -Wextra -Werror`; `-cl-denorms-are-zero` forbidden; correctly-
  rounded div/sqrt default; the launch/module-count invariants of the
  decode list unchanged (774 / 19).

And carried from the house rules (plan 5's Global Constraints, still binding):

- **The Mac never compiles.** Every build, every ctest, every probe goes
  through `tools/box.sh` with `JOBS=44`. GPU work runs with
  `ZE_AFFINITY_MASK=1` when a container holds card 0; nothing in this plan
  touches docker, installs anything in the operator's container, or kills a
  process.
- **Decode's 774/19 invariants and its gate rows are unchanged.** Task 1
  re-runs `replay_determinism_test` **and** both golden gates after the buffer
  split and pastes the output; every later task that touches
  `src/runtime/buffers.{h,cc}`, `src/runtime/capture.cc` or
  `src/kernels/*.cl` (as opposed to `src/kernels/prefill/*.cl`) does the same.
  A determined-row flip is STOPPED and surfaced.
- Full suite green before any commit touching `src/`.
- Determinism: no fp atomics anywhere in the prefill path, fixed grids and
  fixed reduction orders, byte-identical replays; per-op bf16 RNE at the same
  points decode rounds (the P1-P13 list in Task 6 Step 1).
- Every commit message ends with
  `Claude-Session: `.
- Checkpoints and goldens: RTN at
  `/home/user/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64` with goldens in
  `oracle-out-rtn/`; Vishva snapshot
  `2a9077667e28aa53e61d91bdee5d7962e8674668` with goldens in `oracle-out/`;
  the new long prompt's goldens in `oracle-out-long/`.

---

## Rulings this plan makes (read before any task)

These are decisions the spec and interfaces.md leave open. They are made here,
once, so no task re-derives them.

- **R1 - the prefill path runs every int4 GEMV at `S = 1`.** Split-K exists to
  buy hardware threads at `M = 1`; at `M = C` the `M` tile axis already
  saturates the grid, and an `[S][C][N]` fp32 partial rectangle at `S = 8` and
  `N = 34816` would be **4.56 TB** (derived: 8·4096·34816·4). One `partials`
  buffer of `[C][34816]` fp32 = **570,425,344 B** therefore covers every int4
  GEMV on the prefill path, exactly as decode's max-S×max-N rectangle covers
  its own. Consequence: every consumer (`pf_res_fold`, `pf_silu_mul`,
  `pf_gated_head`, `pf_attn_prep`, `gdn_chunk`) folds **one** slice.
- **R2 - the producer's row stride always equals the consuming GEMV's `K`.**
  `pf_norm_finish` writes `x` at stride 5120 and the next GEMV reads `K` 5120;
  `pf_silu_mul` writes `x` at stride 17408 and `down` reads `K` 17408;
  `pf_gated_head` and `attn_reduce` write `mixer_out` at stride 6144 and
  `out_proj`/`o_proj` read `K` 6144. This holds by construction at `M = 1`
  (decode) and is the invariant that makes it hold at `M = C`. Any new
  producer/consumer pair states which stride it is on.
- **R3 - `gdn_chunk` owns the whole GDN mixer**, from the fp32 `qkv‖z` partials
  to the bf16 `out_proj` input: its last launch is `pf_gated_head`, so its `y`
  is `[C][6144]` bf16 (interfaces.md's type). `pf_gated_head` is still an
  S4-owned kernel (Task 5); `gdn_chunk` calls it.
- **R4 - the bf16 rounding of the `qkv‖z` linear happens *inside*
  `gdn_chunk`**, at the same single `rne_bf16` `gdn_step.cl:253` and
  `prep.cl:369` apply, so `gdn_chunk` takes **fp32** `[C][16384]` partials, not
  a materialised bf16 copy. This removes a 134 MB buffer and a 400 MB/layer
  pass and - more importantly - keeps the rounding point bit-identical to
  decode's, which is what spec §6.3's self-consistency bar rests on.
  (Interface change request #1.)
- **R5 - L1's attention route is `pf_attn_prep` + decode's *unmodified*
  `attn_decode` / `attn_reduce` at a new `M = 64` binary.** `pf_attn_prep`
  writes decode's exact `attn_q` fp32 `[·][24][256]`, `attn_gate` fp32
  `[·][24][256]` and `kv_k`/`kv_v` rows; the two heavy kernels are the existing
  source, recompiled at `M = 64` only. `attn_part` is
  `[24][MAXLEN/64][64][258]` fp32 = **405,798,912 B** at `max_len` 16384, which
  is the whole reason the attention sub-chunk is capped at 64. **This is a
  per-`M` compiled binary and therefore a deliberate, recorded exception to
  interfaces.md's "no per-M compiled variants on the prefill path" rule,
  scoped to the L1 stand-in plan 6d retires.** (Interface change request #4.)
  L1 does **not** implement `runtime::prefill::attn.h`; plan 6d does. L1's two
  functions live in `src/runtime/prefill/attn_l1.h` and are named
  `attn_prep_l1` / `attn_l1` so that 6d's landing is one `if` in `engine.cc`.
- **R6 - prefill writes no per-layer residual tap.** The tap is
  `bf16[64][M=1][5120]` and its consumer is `golden_gate_test`'s
  ingest-by-decode walk. `Engine::prefill()` ignores `tap_`;
  `prefill_gate_test` grades logits and tokens, which is what spec §6.1 asks
  for. `Engine::debug_resid()` keeps meaning what it means for decode.
- **R7 - `PrefillScratch` is allocated lazily, on the first `prefill()` call.**
  A decode-only `Engine` therefore has byte-identical device residency to
  today's, which is what makes spec §6.5 ("decode is untouched") checkable
  rather than argued.
- **R8 - fp32 throughout the chunk algebra, deviating from FLA in two named
  places.** FLA rounds the triangular inverse (`solve_tril(..., output_dtype =
  k.dtype)`) and `v_new` (`b_v.to(k.dtype)`) to bf16. Spec §3.4 says "fp32
  throughout" and the reference this plan is graded against is the fp32
  recurrent `gdn_ref.h`, so both stay fp32 here. Recorded in docs/12 as a
  deliberate deviation, with the two FLA line references.

### Interface change requests (to the controller, before Task 6)

1. `gdn.h`: `gdn_chunk`'s `const uint16_t* x_qkvz` → `const float*
   qkvz_partials /* fp32 [C][16384], S = 1 - decode's partials contract;
   rounded to bf16 inside, at the point gdn_step.cl:253 rounds */`. Reason: R4.
2. `gdn.h`: `const uint16_t* a_b /* [C][96] */` → `const float* ab_out /*
   [C][128] fp32 - decode's ab_out layout: a at [0,48), b at [48,96),
   zero-padded to 128 */`. Reason: `pf_ab_proj` is a straight batching of
   `gemv_bf16`, which writes fp32, and `gdn_step.cl:289-290` is where the bf16
   rounding belongs. Bar §6.3 needs that rounding point unmoved.
3. CLI: add `--pp-chunk C` (default `PrefillScratch::kC`). Spec §6.2 gates the
   multi-chunk case **at `C = 1024`**, which is otherwise unreachable from the
   binary. Also fix the stdout row: the existing `tg` row stays byte-identical
   and `--pp` adds a **second** row,
   `| b70-decode <sha> pp | <ids> | <chunk> | <ms> | <t/s> |`, so nothing that
   parses docs/BENCHMARKS' decode rows changes.
4. Layout conventions: add the sentence "the L1 attention stand-in
   (`attn_decode`/`attn_reduce` at `M = 64`) is an explicit exception to the
   runtime-`M` rule and is retired by plan 6d." Reason: R5.

If the controller declines #1 or #2, Task 6 adds a `pf_lin_bf16` rounding
kernel and a `qkvz_b` bf16 `[C][16384]` = 134,217,728 B buffer, and Task 1's
byte table grows by that much - the arithmetic is in Task 1 Step 2 so the
alternative is priced, not guessed.

---

### Task 1: The buffer split - `PersistentBuffers` / `DecodeScratch` / `PrefillScratch`, decode re-proved

**Files:**
- Modify: `src/runtime/buffers.h`, `src/runtime/buffers.cc`,
  `src/runtime/engine.h`, `src/runtime/engine.cc`
- Test: `tests/runtime/buffers_test.cc` (the two pinned totals keep their
  values; a third total is added with its arithmetic)
- Untouched, deliberately: `src/runtime/capture.cc` (129 `res_norm` sites + 17
  buffer names), `src/cli/b70_decode.cc`,
  `tests/runtime/replay_determinism_test.cc`

**Interfaces:**
- Consumes: `l0::Mem` / `l0::Context` / `l0::CmdList` (`src/l0/`),
  `model::Qwen35`, `runtime::Control`.
- Produces, for every later task in this plan and for plans 6a/6c/6d:

```cpp
// src/runtime/buffers.h
namespace runtime {

// Everything that survives a token boundary and is SHARED between the decode
// list and the prefill path. Layouts are decode's, unchanged.
struct PersistentBuffers {
  static constexpr uint32_t kConvRing = 16;          // was DecodeBuffers::kConvRing
  PersistentBuffers(l0::Context& ctx, uint32_t max_len);
  l0::Mem control;      // shared, sizeof(Control)
  l0::Mem gdn_state;    // fp32 [48][48][128][128]
  l0::Mem conv_ring;    // bf16 [48][kConvRing][10240]
  l0::Mem kv_k, kv_v;   // bf16 [16][max_len][4][256] each
  uint32_t max_len;
  size_t bytes() const;
  void zero(l0::CmdList& imm);   // the five fills Engine::reset() does
};

// Decode's per-step scratch, sized by kM. Field names and sizes unchanged.
struct DecodeScratch {
  static constexpr uint32_t kM = 8;
  static constexpr uint32_t kAttnBlock = 64;
  static constexpr uint32_t kNormGroups = 20;
  DecodeScratch(l0::Context& ctx, uint32_t max_len);
  l0::Mem resid, x, partials, ab_out, norm_sumsq, gdn_o, attn_q, attn_gate,
          attn_part, attn_out, logits, argmax_part;
  size_t bytes() const;
};

// The prefill path's per-chunk scratch. kC is the widest chunk; kAttnC is the
// L1 attention sub-chunk (attn_part is linear in it - R5).
struct PrefillScratch {
  static constexpr uint32_t kC = 4096;
  static constexpr uint32_t kAttnC = 64;
  static constexpr uint32_t kGdnChunk = 64;          // the FLA intra-chunk size
  static constexpr uint32_t kNormGroups = DecodeScratch::kNormGroups;
  static constexpr uint32_t kAttnBlock = DecodeScratch::kAttnBlock;
  PrefillScratch(l0::Context& ctx, uint32_t max_len);
  l0::Mem ids;          // uint32 [kC], Host  - pf_embed_gather's input
  l0::Mem resid;        // bf16 [kC][5120]
  l0::Mem x;            // bf16 [kC][17408]
  l0::Mem partials;     // fp32 [kC][34816]   (R1: one S=1 rectangle)
  l0::Mem ab_out;       // fp32 [kC][128]
  l0::Mem norm_sumsq;   // fp32 [kNormGroups][kC]
  l0::Mem gdn_o;        // fp32 [kC][48][128]
  l0::Mem mixer_out;    // bf16 [kC][6144]    (out_proj / o_proj input, R2)
  l0::Mem attn_q;       // fp32 [kAttnC][24][256]
  l0::Mem attn_gate;    // fp32 [kAttnC][24][256]
  l0::Mem attn_part;    // fp32 [24][max_len/kAttnBlock][kAttnC][258]
  l0::Mem logits;       // fp32 [1][248320]   (last position only)
  l0::Mem argmax_part;  // fp32 [1][243][2]
  l0::Mem dequant;      // bf16 [5120][34816] = 356,515,840 B - plan 6c's, unused in L1
  // gdn_chunk's own scratch (Tasks 6-8)
  l0::Mem gdn_xb;       // bf16 [kC][10240]  conv+SiLU, then l2normed in place
  l0::Mem gdn_seed;     // bf16 [3][10240]   the ring's last 3 positions
  l0::Mem gdn_g;        // fp32 [kC][48]     the intra-chunk cumulative gate
  l0::Mem gdn_beta;     // fp32 [kC][48]
  l0::Mem gdn_A;        // fp32 [kC/kGdnChunk][48][64][64]  A, then T in place
  l0::Mem gdn_A2;       // fp32 [kC/kGdnChunk][48][64][64]
  l0::Mem gdn_w, gdn_u; // bf16 [kC][48][128] each
  uint32_t max_len;
  size_t bytes() const;
};

// The VIEW. Every public name capture.cc uses, with the same types as before
// except that the l0::Mem members are now references into the two groups.
// The (ctx, max_len) constructor still OWNS both groups, so every existing
// construction site compiles and behaves unchanged.
struct DecodeBuffers {
 private:
  std::unique_ptr<PersistentBuffers> own_p_;   // declared first: the reference
  std::unique_ptr<DecodeScratch> own_s_;       // members below bind to these
 public:
  static constexpr uint32_t kM = DecodeScratch::kM;
  static constexpr uint32_t kConvRing = PersistentBuffers::kConvRing;
  static constexpr uint32_t kAttnBlock = DecodeScratch::kAttnBlock;
  static constexpr uint32_t kNormGroups = DecodeScratch::kNormGroups;
  DecodeBuffers(l0::Context& ctx, uint32_t max_len);       // owning
  DecodeBuffers(PersistentBuffers& p, DecodeScratch& s);   // view (Engine's)
  l0::Mem &control, &gdn_state, &conv_ring, &kv_k, &kv_v;
  l0::Mem &resid, &x, &partials, &ab_out, &norm_sumsq, &gdn_o, &attn_q,
          &attn_gate, &attn_part, &attn_out, &logits, &argmax_part;
  uint32_t max_len;
  size_t persistent_bytes() const;   // unchanged value
  size_t scratch_bytes() const;      // unchanged value
};
}  // namespace runtime
```

- [ ] **Step 1: Failing test first - the third total, with its arithmetic.**
  Append to `tests/runtime/buffers_test.cc`, above the existing asserts, the
  block below. It fails to compile (no `PrefillScratch`), which is the failing
  state.

```cpp
  // Prefill scratch @ kC = 4096, kAttnC = 64, max_len = 16384. Every line is
  // derived from model::Qwen35 and the two chunk constants; the totals are
  // pinned so a size change arrives here as a diff with an explanation.
  //   ids          4096 x 4 B                                    =        16,384
  //   resid        4096 x 5120 x 2 B                             =    41,943,040
  //   x            4096 x 17408 x 2 B  (largest GEMV K: down)    =   142,606,336
  //   partials     4096 x 34816 x 4 B  (S=1 max-N: gate||up)     =   570,425,344
  //   ab_out       4096 x 128 x 4 B                              =     2,097,152
  //   norm_sumsq   20 x 4096 x 4 B                               =       327,680
  //   gdn_o        4096 x 48 x 128 x 4 B                         =   100,663,296
  //   mixer_out    4096 x 6144 x 2 B                             =    50,331,648
  //   attn_q       64 x 24 x 256 x 4 B                           =     1,572,864
  //   attn_gate    same                                          =     1,572,864
  //   attn_part    24 x (16384/64 = 256) x 64 x 258 x 4 B        =   405,798,912
  //   logits       1 x 248320 x 4 B                              =       993,280
  //   argmax_part  1 x 243 x 2 x 4 B                             =         1,944
  //   dequant      5120 x 34816 x 2 B  (plan 6c's; unused in L1) =   356,515,840
  //   gdn_xb       4096 x 10240 x 2 B                            =    83,886,080
  //   gdn_seed     3 x 10240 x 2 B                               =        61,440
  //   gdn_g        4096 x 48 x 4 B                               =       786,432
  //   gdn_beta     same                                          =       786,432
  //   gdn_A        64 x 48 x 64 x 64 x 4 B                       =    50,331,648
  //   gdn_A2       same                                          =    50,331,648
  //   gdn_w        4096 x 48 x 128 x 2 B                         =    50,331,648
  //   gdn_u        same                                          =    50,331,648
  //                                                       total  = 1,961,713,560
  runtime::PrefillScratch pf(ctx, 16384);
  std::printf("prefill scratch %zu B (%.3f GB)\n", pf.bytes(), pf.bytes() / 1e9);
  CHECK_EQ(pf.bytes(), size_t{1961713560});
  CHECK_EQ(runtime::PrefillScratch::kC, 4096u);
  CHECK_EQ(runtime::PrefillScratch::kAttnC, 64u);
  CHECK_EQ(runtime::PrefillScratch::kGdnChunk, 64u);
  // The split must not move decode's numbers by a byte.
  runtime::PersistentBuffers p(ctx, 16384);
  runtime::DecodeScratch s(ctx, 16384);
  CHECK_EQ(p.bytes(), size_t{1240465536});
  CHECK_EQ(s.bytes(), size_t{68652864});
  runtime::DecodeBuffers view(p, s);
  CHECK_EQ(view.persistent_bytes(), p.bytes());
  CHECK_EQ(view.scratch_bytes(), s.bytes());
  CHECK_EQ(view.control.ptr(), p.control.ptr());
  CHECK_EQ(view.attn_part.ptr(), s.attn_part.ptr());
  CHECK_EQ(view.max_len, 16384u);
```

- [ ] **Step 2: Run it, see it fail.**
  `tools/box.sh test buffers_test` - expect a compile error naming
  `runtime::PrefillScratch`. Paste it.
- [ ] **Step 3: Split `buffers.h`/`buffers.cc`.** Move the five persistent
  allocations into `PersistentBuffers` and the twelve scratch allocations into
  `DecodeScratch` verbatim - same `l0::MemKind`, same size expressions, same
  order (the order matters only for the zero-fill loop, which moves to
  `PersistentBuffers::zero`). Keep `partials_bytes()` and `attn_blocks()` in
  the anonymous namespace, retargeted at `DecodeScratch::kM`. `DecodeBuffers`'
  owning constructor is
  `own_p_(new PersistentBuffers(ctx, max_len)), own_s_(new DecodeScratch(ctx, max_len)),
   control(own_p_->control), …, max_len(max_len)`; the view constructor binds
  the same references to `p`/`s` with both unique_ptrs null. Move the
  constructor's `l0::CmdList imm = l0::CmdList::immediate(ctx); … fill(…)`
  block into `PersistentBuffers`' constructor unchanged, so a freshly
  constructed group is still zeroed before the constructor returns.
- [ ] **Step 4: Add `PrefillScratch`** with exactly the fields and sizes in
  Step 1's table. `ids` is `l0::MemKind::Host` (the host writes `C` ids per
  chunk and the device reads them); everything else is `Device`. Nothing is
  zero-filled: no prefill kernel reads scratch it has not first written, and
  Task 12's determinism test is the standing proof of that - the same rule and
  the same reason `Engine::reset()` gives for decode.
- [ ] **Step 5: Rewire `Engine`** to own the groups and hold the view:
  members become `PersistentBuffers persist_; DecodeScratch decode_scratch_;
  DecodeBuffers buffers_; std::unique_ptr<PrefillScratch> pf_;` in that
  declaration order, with `buffers_(persist_, decode_scratch_)`.
  `Engine::reset()` becomes `persist_.zero(imm_);` - same five fills, same
  order, so the bytes it writes are identical. `Engine::buffers()` keeps
  returning `DecodeBuffers&`. `pf_` stays null (R7).
- [ ] **Step 6: Run the unit tests.**
  `tools/box.sh test 'buffers_test|kernel_table_test|qwen35_test'` - green.
- [ ] **Step 7: Re-prove decode, and paste the evidence.** The three that pin
  the invariants, then both gates:

```bash
tools/box.sh test 'replay_determinism_test|engine_smoke_test|profile_capture_test'
tools/box.sh run 'ZE_AFFINITY_MASK=1 ./build/tests/golden_gate_test \
  "$PWD/oracle-out" "$PWD/tests/golden/prompts" \
  "$HOME/.cache/huggingface/hub/models--Vishva007--Qwen3.8-27B-W4A16-AutoRound-GPTQ/snapshots/2a9077667e28aa53e61d91bdee5d7962e8674668"'
tools/box.sh run 'ZE_AFFINITY_MASK=1 ./build/tests/golden_gate_test \
  "$PWD/oracle-out-rtn" "$PWD/tests/golden/prompts" \
  "$HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64"'
```

  Required: `kernel_count == 774`, `modules.size() == 19`, all three
  replay_determinism runs bitwise identical, and both gates' `TOTAL:` lines
  showing every determined row exact. Paste the four `TOTAL:` / `OK` lines into
  the commit message body.
- [ ] **Step 8: Run the full suite.** `tools/box.sh test` - every test green
  (checkpoint-labelled ones included; the box has the model).
- [ ] **Step 9: Commit** -
  `refactor(runtime): split DecodeBuffers into persistent/decode/prefill groups`

### Task 2: `runtime::prefill::Context` - the L0 dynamic-dispatch side

**Files:**
- Create: `src/runtime/prefill/context.h`, `src/runtime/prefill/context.cc`,
  `src/runtime/prefill/CMakeLists.txt`, `tests/prefill/context_test.cc`
- Modify: `src/l0/cmdlist.h`, `src/l0/cmdlist.cc` (the async immediate list and
  the host synchronise), `src/runtime/CMakeLists.txt` (add the new sources to
  `b70_runtime`), `tests/CMakeLists.txt`, `CMakeLists.txt` (nothing - no new
  subdirectory; the prefill host code joins `b70_runtime`)

**Interfaces:**
- Consumes: `l0::Context`, `l0::Module`, `l0::Kernel`, `l0::CmdList`
  (`src/l0/`), `kernels::path()` (`src/kernels/kernels.h`).
- Produces, for every later task and for plans 6a/6c/6d:

```cpp
// src/runtime/prefill/context.h
namespace runtime::prefill {

struct KernelArg { const void* ptr; size_t size; };   // zeKernelSetArgumentValue form
// Convenience makers, so a call site reads like the kernel's signature.
inline KernelArg arg_ptr(const void* p) { return {&p, sizeof(void*)}; }  // see note
template <class T> KernelArg arg_val(const T& v) { return {&v, sizeof(T)}; }

// One in-order execution context for a whole prefill step. THIS PLAN owns the
// Level Zero half: an ASYNCHRONOUS in-order immediate command list, kernel
// arguments set at launch time (not at capture time), and wait() draining it.
// Plan 6a's T0 adds the SYCL half - the sycl::queue built from the same
// ze_context/ze_device and the sycl() accessor - and owns the constructor's
// final shape. This plan does NOT declare or define sycl(); it declares the
// constructor and wait() with the signatures below and 6a extends them.
class Context {
 public:
  Context(l0::Context& ctx);
  void wait();                                        // the L0 list drained
  // Load (or reuse) a variant's device binary and return a Kernel bound to
  // `entry`. Cached by variant name for the Context's lifetime, exactly as
  // runtime::Capture::kernel() caches per CapturedStep.
  l0::Kernel& kernel(const std::string& variant, const char* entry, uint32_t wg);
  // Append a launch with the arguments resolved NOW. `args` are set in order
  // starting at index 0.
  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy, uint32_t gz,
              std::initializer_list<KernelArg> args);
  l0::Context& l0() const { return ctx_; }
  size_t launches() const { return launches_; }       // for the anatomy report
 private:
  l0::Context& ctx_;
  l0::CmdList list_;                                  // immediate, async, in-order
  std::map<std::string, std::unique_ptr<l0::Module>> modules_;
  std::map<std::string, std::unique_ptr<l0::Kernel>> kernels_;
  size_t launches_ = 0;
};
}  // namespace runtime::prefill
```

```cpp
// src/l0/cmdlist.h additions
  static CmdList immediate_async(Context& ctx, uint32_t ordinal = 0);
  void host_sync(uint64_t timeout_ns = UINT64_MAX);   // zeCommandListHostSynchronize
```

- [ ] **Step 1: Discover whether the driver has in-order immediate lists.**
  On the box:

```bash
tools/box.sh run 'grep -n "ZE_COMMAND_QUEUE_FLAG_IN_ORDER\|zeCommandListHostSynchronize" \
  /usr/include/level_zero/ze_api.h | head'
tools/box.sh run 'pkg-config --modversion level-zero'
```

  Paste both. If `ZE_COMMAND_QUEUE_FLAG_IN_ORDER` is present, Step 4 uses it
  and the list is implicitly ordered. If it is absent, Step 4 instead chains
  every launch through a signal event and a single wait-event
  (`zeCommandListAppendLaunchKernel(list, k, &gc, sig, 1, &prev)`), which is
  the ordering interfaces.md's "ordering by events" sentence describes; the
  fallback costs one event per launch from a pool of
  `ProfileEvents::kProfileCapacity`-style capacity and the same test in Step 2
  proves it either way. Record which branch was taken in `context.cc`'s header
  comment with the grep output quoted.
- [ ] **Step 2: Failing test first.** `tests/prefill/context_test.cc`, no
  checkpoint and no model - a GPU and the `noop`/`bw_sum` binaries only, so it
  belongs in the plain suite like `launch_test`:
  - open `pf_probe_chain.cl` (created in Step 3) and launch
    `pf_chain_step(buf, n)` **1024 times** in a row on one `Context`, each
    launch incrementing every element of a device `uint[4096]` buffer by its
    own launch index passed as a runtime argument;
  - `wait()`, copy back, require every element equals `Σ_{i<1024} i = 523776`
    - which can only hold if (a) arguments are resolved per launch and not
    frozen at the first append, and (b) the launches ran in order with no
    overlap on the buffer;
  - run the same 1024 launches a second time on the same `Context` and require
    the *doubled* value, proving `kernel()`'s cache is reusable;
  - launch with a deliberately wrong argument count and require a throw naming
    the kernel.
- [ ] **Step 3: Run it, see it fail** (`tools/box.sh test context_test`):
  no such target. Paste.
- [ ] **Step 4: Implement `l0::CmdList::immediate_async` and `host_sync`**
  exactly as `immediate()` is written (`src/l0/cmdlist.cc:7-16`), with
  `qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS` and, per Step 1,
  `qd.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER`. `host_sync` is one
  `ZE_CHECK(zeCommandListHostSynchronize(l_, timeout_ns));`. Add a
  `require(immediate_, …)`-style throw in `host_sync` if the list is regular,
  so misuse is loud.
- [ ] **Step 5: Implement `Context`.** `kernel()` mirrors
  `runtime::Capture::kernel()` (`capture.cc:246-267`) - `modules_.find`, then
  `emplace(variant, make_unique<l0::Module>(ctx_, kernels::path(variant)))`,
  then a `Kernel` cached under `variant + "/" + entry` with `group_size(wg)`
  set once. `launch()` sets each argument with
  `zeKernelSetArgumentValue` through `l0::Kernel::arg`/`arg_ptr` in list order
  and then `list_.launch(k, gx, gy, gz)`; `++launches_`. `wait()` is
  `list_.host_sync()`.
  **The pointer-argument trap, stated in the header:** `zeKernelSetArgumentValue`
  copies `size` bytes from `ptr`, so a pointer argument must pass the *address
  of the pointer variable*. `arg_ptr(const void*)` above takes its parameter by
  value and would return a dangling `&p`; write it instead as a small struct
  the caller keeps alive, or take the pointer by reference. Implement it as

```cpp
// Holds the pointer VALUE, so &value_ stays valid for the whole call.
struct PtrArg {
  const void* value_;
  explicit PtrArg(const void* p) : value_(p) {}
  operator KernelArg() const { return KernelArg{&value_, sizeof(const void*)}; }
};
```
  and let call sites write `cx.launch(k, gx, gy, 1, {PtrArg(a), PtrArg(b), arg_val(M)})`
  - every temporary lives to the end of the full-expression, which is after
  `launch` returns. Put that sentence in the header; it is the one way this
  API can be silently wrong.
- [ ] **Step 6: Add `src/kernels/prefill/pf_probe_chain.cl`** - a two-line
  kernel (`buf[i] += step;`) with `M`-free signature
  `(__global uint* buf, uint n, uint step)` and a
  `add_ocloc_kernel(pf_probe_chain …)` row in a new
  `src/kernels/prefill/CMakeLists.txt` included from
  `src/kernels/CMakeLists.txt` via `add_subdirectory(prefill)`.
- [ ] **Step 7: Wire the build.** `src/runtime/CMakeLists.txt`:
  `add_library(b70_runtime STATIC buffers.cc capture.cc engine.cc
  prefill/context.cc)` - no new library, because the prefill host code is the
  engine's and links exactly what the engine links.
  `tests/CMakeLists.txt`: `context_test` with `b70_target_kernel_dir`,
  `add_dependencies(context_test kernel_pf_probe_chain)`, no label.
- [ ] **Step 8: Run it, see it pass.** `tools/box.sh test context_test`, then
  `tools/box.sh test` (full suite, so the `l0::CmdList` change is covered by
  `copy_test`/`launch_test`/`arg_capture_test` too).
- [ ] **Step 9: Commit** -
  `feat(runtime): runtime::prefill::Context - in-order async immediate list with runtime args`

### Task 3: Runtime-`M` kernels, part 1 - `pf_embed_gather`, `pf_res_fold` / `pf_norm_finish`, `pf_silu_mul`

**Files:**
- Create: `src/kernels/prefill/pf_embed.cl`, `src/kernels/prefill/pf_prep.cl`
  (three entry points in one file - `prep.cl`'s pattern),
  `src/kernels/prefill/pf_kernels.h` (variant names, `kernels::` namespace),
  `tests/prefill/pf_ref.h` (the runtime-`M`, `S = 1` CPU references),
  `tests/prefill/pf_prep_test.cc`
- Modify: `src/kernels/prefill/CMakeLists.txt`, `src/kernels/CMakeLists.txt`
  (two additive test-only decode variants - see Step 1), `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `prep.cl`'s rounding discipline and reduction trees
  (`src/kernels/prep.cl:1-36`, `:209-300`, `:302-335`), `prep_ref.h`'s verbatim
  restatement of them, `embed_gather.cl:45-68`.
- Produces:

```cpp
// src/kernels/prefill/pf_kernels.h   (namespace kernels)
// Every prefill variant name. `M` is NEVER in a name (interfaces.md); K, S_PREV,
// the norm's group counts and the int4 layout still are, because they are
// strides and grids the binary bakes.
inline std::string pf_embed_gather_variant();                       // "pf_embed_gather"
inline std::string pf_res_fold_variant(unsigned K, unsigned SP, unsigned G);
inline std::string pf_norm_finish_variant(unsigned K, unsigned G, unsigned W);
inline std::string pf_silu_mul_variant();                           // "pf_silu_mul"
```

Kernel signatures (the contract Task 10 binds and Task 12 tests):

```c
// pf_embed.cl - grid (1, M), WG 256. Reads ids from a BUFFER, not from Control:
// Control::cur_token[8] holds eight ids and a chunk holds up to 4096.
__kernel void pf_embed_gather(__global const uint* restrict ids,
                              __global const ushort* restrict embed,
                              __global ushort* restrict resid, uint m_count);

// pf_prep.cl - S_PREV in {0,1} (R1), FOLD_G/NORM_G/NORM_WGS as decode's 20.
// Grid (FOLD_G, M) / (NORM_WGS, M), WG 256. `m_count` bounds nothing in the
// grid (grid y IS M) and is passed only so the partials/resid/sumsq strides
// can be computed: sumsq[g*m_count + m].
__kernel void pf_res_fold(__global const float* restrict partials,
                          __global ushort* restrict resid,
                          __global float* restrict sumsq, uint m_count);
__kernel void pf_norm_finish(__global const float* restrict sumsq,
                             __global const ushort* restrict resid,
                             __global const float* restrict norm_w,
                             __global ushort* restrict x_out, uint m_count);
// grid (ceil(17408/4096) = 5, M), WG 256. SILU_S is 1 here (R1).
__kernel void pf_silu_mul(__global const float* restrict partials,
                          __global ushort* restrict x_out, uint m_count);
```

- [ ] **Step 1: Failing test first.** `tests/prefill/pf_prep_test.cc`, three
  comparisons per kernel:
  1. **M = 1, bit-identical to the decode kernel.** `pf_embed_gather` vs
     `embed_gather_M1` (same id, same table); `pf_res_fold` at `SP = 0` vs
     `prep_res_fold_M1_K5120_SP0_G20`; `pf_res_fold` at `SP = 1` vs a new
     **test-only** `prep_res_fold_M1_K5120_SP1_G20`; `pf_norm_finish` vs
     `prep_norm_finish_M1_K5120_G20_W20`; `pf_silu_mul` vs `prep_silu_mul_M1`
     driven with an 8-slice `partials` whose slices 1..7 are **`+0.0f`** -
     `Σ_s` then reduces to slice 0 exactly. Assert in the test that no input
     value is `-0.0f` before doing this (`+0.0f + -0.0f == +0.0f` would flip a
     sign of zero and therefore a bf16 word), and say so in a comment.
     Bars: `memcmp` on `resid`, `x_out`, `sumsq` - **bit-identical, no
     tolerance**.
  2. **M = 64 vs `pf_ref.h`.** New CPU references mirroring `prep_ref.h`'s
     chains verbatim with runtime `M` and `S = 1`. `resid`, `sumsq` and
     `x_out` bit-identical for `pf_res_fold`/`pf_norm_finish` and
     `pf_embed_gather` (no `exp`, no `sqrt` beyond the correctly-rounded pair);
     `pf_silu_mul` to the established prep ulp bar (its `exp` carries OpenCL's
     3 ulp - `prep_test`'s existing bar, quoted by value).
  3. **Row independence.** Run at `M = 64` with row 17 zeroed, then at `M = 1`
     on row 17's data alone, and require row 17's output identical - the
     property that makes a chunk's rows not leak into one another.
- [ ] **Step 2: Run it, see it fail.** `tools/box.sh test pf_prep_test`. Paste.
- [ ] **Step 3: Write `pf_embed.cl`.** Transcribe `embed_gather.cl:45-68` with
  three changes and nothing else: the id comes from `ids[m]` instead of
  `ctrl[CTRL_CUR + m]`; the `#if M > 8` guard is deleted (there is no
  `cur_token` limit any more); the out-of-range id path writes nothing and
  **cannot** report through `Control::debug_flag` (no ctrl argument) so it
  returns silently and the HOST validates ids before upload - state that in the
  header and add the host-side bound check in Task 10 Step 4.
- [ ] **Step 4: Write `pf_prep.cl`.** Copy `prep.cl:209-300` and `:302-335`
  verbatim, then make exactly these edits:
  - delete `#ifndef M / #define M 1`; add `uint m_count` as the last kernel
    parameter of all three entries;
  - replace every `M` in an index expression with `m_count`: `partials[((size_t)s
    * m_count + m) * K + k]`, `sumsq[(size_t)g * m_count + m]`;
  - `#define SILU_S 1` (R1) with the `#error` guard rewritten to say that the
    prefill path is `S = 1` by ruling R1 and that `pf_gemv_int4_M` is what
    keeps it true;
  - `m = get_group_id(1)` stays; the grid's y extent **is** `M`, so no mask is
    needed and none is added (state it: `ceil(M/1)` is the tile rule with
    tile 1).
  Keep both bf16 RNE steps, the `fma` square-accumulate, `1.0f / sqrt(x)`, the
  `red[WG]` pairwise tree with a barrier after every step, and the
  ascending-`g` fold in stage B - the per-element chain and both tree orders
  are the numerics contract, not an implementation detail.
- [ ] **Step 5: Add the AOT rows.** In `src/kernels/prefill/CMakeLists.txt`:

```cmake
set(PF_PREP_CL ${CMAKE_CURRENT_SOURCE_DIR}/pf_prep.cl)
function(add_pf_res_fold K SP G)
  add_ocloc_kernel(pf_res_fold_K${K}_SP${SP}_G${G} SOURCE ${PF_PREP_CL}
                   DEFINES K=${K} S_PREV=${SP} FOLD_G=${G})
endfunction()
add_pf_res_fold(5120 0 20)   # layer 0's leading norm: nothing to fold
add_pf_res_fold(5120 1 20)   # every other norm (R1: S = 1)
add_ocloc_kernel(pf_norm_finish_K5120_G20_W20 SOURCE ${PF_PREP_CL}
                 DEFINES K=5120 NORM_G=20 NORM_WGS=20)
add_ocloc_kernel(pf_silu_mul SOURCE ${PF_PREP_CL})
add_ocloc_kernel(pf_embed_gather SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_embed.cl
                 DEFINES HIDDEN=5120 VOCAB=248320)
```

  and in `src/kernels/CMakeLists.txt`, one additive test-only decode variant
  (it is never bound by `capture.cc`, so the module count stays 19):

```cmake
add_prep_res_fold(1 5120 1 20)   # test-only: pf_prep_test's M=1 bit-identity control
```

- [ ] **Step 6: Run it, see it pass.** `tools/box.sh test pf_prep_test`, then
  `tools/box.sh test 'prep_test|embed_gather_test|kernel_table_test'`.
- [ ] **Step 7: Re-prove decode** (`src/kernels/CMakeLists.txt` changed):
  `tools/box.sh test 'replay_determinism_test|profile_capture_test'`, paste
  the `774` / `19` lines.
- [ ] **Step 8: Commit** -
  `feat(kernels): pf_embed_gather / pf_res_fold / pf_norm_finish / pf_silu_mul - runtime M`

### Task 4: Runtime-`M` kernels, part 2 - `pf_gemv_int4_M` and `pf_ab_proj`

*`pf_gemv_int4_M` exists only until plan 6c lands `gemm_bf16` + `dequant_to_bf16`
behind spec §3.2's interface. Its file header says so in the first sentence, and
Task 14's docs/12 section labels it **L1 temporary - retired by plan 6c (L2)**.*

**Files:**
- Create: `src/kernels/prefill/pf_gemv.cl`, `src/kernels/prefill/pf_gemv_bf16.cl`,
  `tests/prefill/pf_gemv_test.cc`
- Modify: `src/kernels/prefill/pf_kernels.h`,
  `src/kernels/prefill/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `gemv.cl:78-130` (the dequant contract, the `dot8` orders, both
  layouts), `gemv_bf16.cl:87-137` (the tile layout and the `KSPLIT` SLM tree),
  `kernels::gemv_bf16_tiling` (`kernels.h:66-69`), `tests/kernels/gemv_ref.h`,
  `common::Int4Gptq` (`src/common/`).
- Produces:

```cpp
// src/kernels/prefill/pf_kernels.h
// K, N and LAYOUT stay in the name (they are strides and the grid); S is
// gone - the prefill path is S = 1 by ruling R1 - and M is a kernel argument.
inline std::string pf_gemv_variant(unsigned K, unsigned N, unsigned L);
// The a||b projection, mirroring gemv_bf16's {COLS_PER_WG 16, KSPLIT 16}
// tiling so that at M = 1 it is BIT-IDENTICAL to the binary capture.cc binds.
inline std::string pf_ab_proj_variant();                            // "pf_ab_proj"
```

```c
// pf_gemv.cl - grid (N/64, ceil(M/MT)), WG 64 = 4 subgroups of 16.
// MT = 8 is the register tile: `float acc[MT]`, the same register array
// gemv.cl's `acc[M]` was designed for at M = 8.
__kernel void pf_gemv_int4_M(__global const uint* restrict w,
                             __global const half* restrict scales,
                             __global const ushort* restrict x,
                             __global float* restrict out, uint m_count);
// pf_gemv_bf16.cl - grid (N/COLS_PER_WG, ceil(M/MT)), WG COLS_PER_WG*KSPLIT.
__kernel void pf_ab_proj(__global const ushort* restrict w,
                         __global const ushort* restrict x,
                         __global float* restrict out, uint m_count);
```

- [ ] **Step 1: Failing test first.** `tests/prefill/pf_gemv_test.cc`:
  1. **M = 1, bit-identical to a decode `S = 1` variant, all five production
     shapes.** The probe matrix in `src/kernels/CMakeLists.txt:29-37` already
     builds `gemv_M1_K{6144_N5120,5120_N14336,5120_N16384,5120_N34816,17408_N5120}_S1_L{0,1}`,
     and at `S = 1` none of them receives Task 4's `extra_defs` (the tuned
     cells are keyed on `S EQUAL {4,2,8}` / layout, `CMakeLists.txt:12-22`) -
     so those binaries are the **plain** baseline and `pf_gemv` must be built
     plain too (no `GEMV_BLOCK2D`, no `GEMV_DEQ_SHIFT`). Assert `memcmp` on the
     full `[1][N]` fp32 row: **bit-identical, no tolerance**, for
     `{K,N,L}` ∈ {`6144,5120,0`; `5120,14336,0`; `5120,34816,0`;
     `17408,5120,0`; `5120,16384,1`}.
     `pf_ab_proj` at `M = 1` vs `gemv_bf16_M1_K5120_N128_C16_S16` - the binary
     `capture.cc:436-452` actually binds - also `memcmp`-exact.
  2. **M = 64 vs `gemv_ref.h`.** `gemv_ref(w, x_bf16, …)` per row, driven at
     `M = 64` with independent random rows; bar: the established
     `tol_for(ref)` from `gemv_harness.h` (quote the expression, do not invent
     a number). Plus a **row-independence** case: `M = 64` with row 40's
     activations zeroed must give an exactly-zero output row 40.
  3. **The ragged tile.** `M = 65` (one full `MT` tile plus one live lane out
     of eight) must give the same rows 0..63 as the `M = 64` run,
     bit-identically, and a correct row 64 - this is the masking test, and it
     is the one that catches an unmasked `acc[]` writeback.
  4. **All sixteen nibbles.** Drive one shape with a weight tile containing
     every `q ∈ [0,16)` at least once, as `gemv_test` does, so the dequant
     `(q − 8)·scale` contract is exercised end to end at runtime `M`.
- [ ] **Step 2: Run it, see it fail.** `tools/box.sh test pf_gemv_test`. Paste.
- [ ] **Step 3: Write `pf_gemv.cl`.** Copy `gemv.cl` whole, then:
  - delete `#ifndef M / #define M 1`; `#define MT 8` and `#define S 1`;
    keep `LAYOUT`, `K`, `N` as `-D`;
  - the grid's dim 1 was the split-K slice `s` and is now the `M` tile:
    `const uint mt = get_group_id(1); const uint m0 = mt * MT;`
    and `g0 = 0, g1 = G` (S = 1, so every work-group walks the whole `K`);
  - `float acc[MT]` and `float gacc[MT]` unchanged in shape; every `m` loop
    becomes `for (int i = 0; i < MT; ++i)` with
    `const uint m = m0 + (uint)i; const bool live = m < m_count;` and the
    activation load guarded - read row `m0` when `!live` (never out of
    bounds, and the dead lanes' arithmetic is discarded):
    `const uint mr = live ? m : m0;`
    `ushort8 xv = vload8(0, x + (size_t)mr * K + g * GROUP + j * 8);`
  - the writeback is the only place the mask must bite:
    `for (int i = 0; i < MT; ++i) { const uint m = m0 + i; if (m < m_count) out[(size_t)m * N + n] = acc[i]; }`
    - note `out[(size_t)m * N + n]`, i.e. `gemv.cl:129`'s
    `out[((size_t)s*M + m)*N + n]` at `s = 0`, which is why R1 makes this legal;
  - **the arithmetic order is untouched**: `g` ascending, `j` ascending within
    the group, `dot8`'s eight nibble terms in the same order, `acc[i] +=
    gacc[i] * scale` once per group. That is what buys Step 1's `memcmp`.
- [ ] **Step 4: Write `pf_gemv_bf16.cl`.** Copy `gemv_bf16.cl` whole, then:
  - `#define MT 8`; `COLS_PER_WG 16` and `KSPLIT 16` as the built defaults
    (the a‖b tiling, `kernels.h:44-56`);
  - `__local float red[SG_PER_WG][MT][SG]` - 16·8·16·4 = **8192 B**, stated in
    the comment beside it (the decode build's note says 1024 B at `M = 1`);
  - the `mt`/`m0`/`live`/`mr` treatment of Step 3, applied to the `acc[MT]`
    loop, the `red[sg][i][lane] = acc[i]` store, the pairwise tree (which runs
    over all `MT` slots - the dead slots carry garbage that is never read
    back) and the masked final store;
  - **the tree order is untouched**: `stride = KSPLIT/2 … 1`, `q < stride`,
    barrier after every step (`gemv_bf16.cl:129-133`). That is what buys the
    `memcmp` against `_C16_S16`.
- [ ] **Step 5: Add the AOT rows** to `src/kernels/prefill/CMakeLists.txt`:

```cmake
set(PF_GEMV_CL ${CMAKE_CURRENT_SOURCE_DIR}/pf_gemv.cl)
function(add_pf_gemv K N L)
  add_ocloc_kernel(pf_gemv_K${K}_N${N}_L${L} SOURCE ${PF_GEMV_CL}
                   DEFINES K=${K} N=${N} LAYOUT=${L})
endfunction()
add_pf_gemv(5120 16384 1)   # qkv||z
add_pf_gemv(6144 5120  0)   # out_proj and o_proj (same shape)
add_pf_gemv(5120 34816 0)   # gate||up
add_pf_gemv(17408 5120 0)   # down
add_pf_gemv(5120 14336 0)   # qkv
add_ocloc_kernel(pf_ab_proj SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gemv_bf16.cl
                 DEFINES K=5120 N=128 COLS_PER_WG=16 KSPLIT=16)
```

  Note what is **not** here: `lm_head`. It runs on the chunk's last position
  only, at `M = 1`, through the existing `gemv_M1_K5120_N248320_S1_L1` /
  `gemv_bf16_M1_K5120_N248320` binaries - the same checkpoint-dependent choice
  `capture.cc:625-631` makes, read off the loaded weight's `kind`.
- [ ] **Step 6: Run it, see it pass.** `tools/box.sh test pf_gemv_test`, then
  `tools/box.sh test 'gemv_test|gemv_bf16_test|kernel_table_test'`.
- [ ] **Step 7: Commit** -
  `feat(kernels): pf_gemv_int4_M + pf_ab_proj - runtime M, S=1 (L1 temporary, retired by plan 6c)`

### Task 5: `pf_gated_head`, `pf_attn_prep`, and the `M = 64` attention route

**Files:**
- Create: `src/kernels/prefill/pf_gated_head.cl`,
  `src/kernels/prefill/pf_attn_prep.cl`, `src/runtime/prefill/attn_l1.h`,
  `src/runtime/prefill/attn_l1.cc`, `tests/prefill/pf_attn_test.cc`
- Modify: `src/kernels/prefill/pf_kernels.h`,
  `src/kernels/prefill/CMakeLists.txt`, `src/kernels/CMakeLists.txt` (the
  additive `M = 64` attention rows), `src/runtime/CMakeLists.txt`,
  `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `attn.cl:307-393` (`qkv_sum`, the norm tree, partial RoPE, the KV
  writeback), `attn.cl:407+` / `:565+` (`attn_decode` / `attn_reduce`, used
  unmodified), `prep.cl:355-385` (`prep_gated_head`), `tests/kernels/attn_ref.h`
  (`attn_ref::prep`, `::decode`, `::reduce`, all already `M`-parametric),
  `tests/kernels/prep_ref.h::gated_head`.
- Produces:

```cpp
// src/runtime/prefill/attn_l1.h
namespace runtime::prefill {
// ---------------------------------------------------------------------------
// L1 ONLY - the temporary prefill attention (spec §5 L1, ruling R5).
// `attn_part` is fp32 [24][max_len/64][kAttnC][258] = 405,798,912 B at max_len
// 16384 and is LINEAR IN THE SUB-CHUNK, which is why C_sub <= kAttnC = 64.
// Plan 6d replaces BOTH functions with interfaces.md attn.h's
// attn_prep_chunk / attn_chunk and deletes attn_part with them; engine.cc has
// exactly one call site for each (Task 10 Step 5 names the lines).
//
// Both read Control::pos and Control::n_active - they are decode's kernels and
// decode's stand-in - so the caller sets pos = the sub-chunk's first absolute
// position and n_active = C_sub, and calls Context::wait() before writing them
// (the immediate list is asynchronous; Task 2's header states the rule).
// ---------------------------------------------------------------------------
void attn_prep_l1(Context& cx, uint32_t C_sub,
                  const float* qkv_partials /* fp32 [C_sub][14336], S=1 */,
                  const float* fa_small, const float* rope,
                  void* ctrl, float* attn_q /* [C_sub][24][256] */,
                  float* attn_gate /* [C_sub][24][256] */,
                  uint16_t* kv_k, uint16_t* kv_v);
void attn_l1(Context& cx, uint32_t C_sub, uint32_t max_len, void* ctrl,
             const float* attn_q, const uint16_t* kv_k, const uint16_t* kv_v,
             const float* attn_gate, float* attn_part,
             uint16_t* out /* bf16 [C_sub][6144], pre o_proj */);
}  // namespace runtime::prefill
```

```c
// pf_gated_head.cl - grid (48 v-heads, M), WG 128. Qwen3_5RMSNormGated over one
// GDN v-head, S = 1 (R1). Reads z from the fp32 qkv||z partials at column
// Z_OFF + h*128 + i, exactly as prep.cl:367-369 does, so at M = 1 it is
// bit-identical to prep_gated_head_M1.
__kernel void pf_gated_head(__global const float* restrict qkvz_partials,
                            __global const float* restrict gdn_o,
                            __global const ushort* restrict gated_w,
                            __global ushort* restrict x_out, uint m_count);

// pf_attn_prep.cl - grid (24 q-heads + 4 kv-heads, C_sub), WG 256. NO compiled
// M at all: every output stride it writes is M-independent
// (attn_q/attn_gate are [.][24*256] = stride 6144; kv is indexed by pos+m; the
// partials row base is m*QKV_N because QKV_S is 1), so `n_act` comes from
// Control and the grid's y extent is the sub-chunk.
__kernel void pf_attn_prep(__global const uint* restrict ctrl,
                           __global const float* restrict qkv_partials,
                           __global const float* restrict fa_small,
                           __global const float* restrict rope,
                           __global float* restrict attn_q,
                           __global float* restrict attn_gate,
                           __global ushort* restrict kv_k,
                           __global ushort* restrict kv_v);
```

- [ ] **Step 1: Failing test first.** `tests/prefill/pf_attn_test.cc`:
  1. **`pf_gated_head` at M = 1, bit-identical to `prep_gated_head_M1`** -
     same `partials`, same `gdn_o`, same bf16 `gated_w`; `memcmp` on the
     `[1][6144]` bf16 output.
  2. **`pf_gated_head` at M = 64 vs `prep_ref::gated_head`** run per row -
     bit-identical through `t_b` and to the established prep ulp bar on the
     final `silu`-carrying product (`prep_test`'s bar, quoted).
  3. **`pf_attn_prep` at n_act = 1, bit-identical to `attn_prep_M1`** - driven
     with an `S = 2` partials buffer whose slice 1 is `+0.0f` so
     `attn.cl:307-311`'s `qkv_sum` reduces to slice 0 exactly (same `-0.0`
     caveat and same assertion as Task 3 Step 1). `memcmp` on `attn_q`,
     `attn_gate`, and the four written `kv_k`/`kv_v` rows - the KV cache rows
     are the load-bearing output (decode continues from them) and they are
     held **bit-exact**, which `attn.cl:190-196` says is achievable because
     only `exp` carries slack and there is none here.
  4. **`pf_attn_prep` at n_act = 64 vs `attn_ref::prep(pos, 64, 64, …)`** -
     bit-exact on all four outputs, at `pos = 0` and at `pos = 4033` (so the
     RoPE table is read at a high index and the causal window straddles a
     block edge).
  5. **The full L1 route at C_sub = 64 vs `attn_ref`.** Synthetic cache filled
     to depth 4096 (`attn_test`'s pattern), then `attn_prep_l1` +
     `attn_l1` and a comparison of `out` against
     `attn_ref::prep` → `::decode` → `::reduce` at `M = 64`, `max_len = 4096`:
     `attn_part` bit-exact where `attn_ref` is bit-exact and `out` to
     `attn_test`'s existing `exp`-slack bar (quote it).
  6. **`C_sub > kAttnC` throws** from `attn_l1`, naming `attn_part` and plan 6d.
- [ ] **Step 2: Run it, see it fail.** `tools/box.sh test pf_attn_test`. Paste.
- [ ] **Step 3: Write `pf_gated_head.cl`** - `prep.cl:355-385` verbatim with
  `#define GATED_S 1`, `M` deleted, `uint m_count` appended, and
  `((size_t)s * m_count + m)` in the `z` load. Keep all five RNE points
  (`o_b`, `z_b`, `n_b`, `t_b`, the final product), the 128-lane variance tree,
  `1.0f / sqrt(var + 1e-6f)`, and the `silu` factor **last**.
- [ ] **Step 4: Write `pf_attn_prep.cl`** - `attn.cl:307-393` verbatim with
  `#define QKV_S 1`, `M` deleted from the source (its only uses were the
  `n_act` clamp and `qkv_sum`'s stride, and both go away at `QKV_S = 1`), the
  `if (m >= n_act) return;` guard kept (the grid's y extent may exceed
  `n_act` when the caller rounds up - it does not here, but the guard is the
  kernel's own contract and stays), and `ATTN_BLOCK`/`MAXLEN` **not** required
  (`attn_prep` expands neither; drop the `#error` guards it only had because
  it shared a file with the other two entries).
- [ ] **Step 5: Add the AOT rows.** `src/kernels/prefill/CMakeLists.txt`:

```cmake
add_ocloc_kernel(pf_gated_head SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gated_head.cl)
add_ocloc_kernel(pf_attn_prep SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_attn_prep.cl
                 DEFINES ${CTRL_DEFINES} QNORM_OFF=0 KNORM_OFF=256)
```

  `CTRL_DEFINES` is set in `src/kernels/CMakeLists.txt:142`; the prefill
  subdirectory is added **after** that line so the variable is in scope
  (`add_subdirectory` inherits the parent scope). `src/kernels/CMakeLists.txt`
  gains the two additive `M = 64` rows - additive, so decode's `M = 1`
  binaries' ocloc command lines are unchanged:

```cmake
# Spec 2 L1's temporary prefill-attention route (plan 6b ruling R5). These are
# decode's UNMODIFIED attn_decode/attn_reduce at the L1 attention sub-chunk
# width; `attn_part` is [24][MAXLEN/BLOCK][M][258], so M is in the binary. Plan
# 6d retires both rows with the buffer. capture.cc binds neither, so the decode
# list's 19 modules are unchanged.
add_attn_decode(64 16384)
add_attn_reduce(64 16384)
add_attn_decode(64 4096)    # pf_attn_test's buffers
add_attn_reduce(64 4096)
```

- [ ] **Step 6: Write `attn_l1.cc`.** `attn_prep_l1` is one launch:
  `cx.kernel("pf_attn_prep", "pf_attn_prep", 256)`, grid
  `(Qwen35::kFaQHeads + Qwen35::kFaKvHeads, C_sub, 1)`, eight `PtrArg`s in
  `attn.cl:328-333`'s order. `attn_l1` is two:
  `kernels::attn_decode_variant(PrefillScratch::kAttnC, max_len, PrefillScratch::kAttnBlock)`
  at grid `(4, max_len / kAttnBlock, 1)` and
  `attn_reduce_variant(...)` at grid `(24, C_sub, 1)`, with the
  `require(C_sub <= PrefillScratch::kAttnC, …)` throw at the top naming
  `attn_part` and plan 6d.
- [ ] **Step 7: Run it, see it pass.** `tools/box.sh test pf_attn_test`, then
  `tools/box.sh test 'attn_test|prep_test'`.
- [ ] **Step 8: Re-prove decode** (`src/kernels/CMakeLists.txt` changed):
  `tools/box.sh test 'replay_determinism_test|profile_capture_test'` - `774` /
  `19` pasted.
- [ ] **Step 9: Commit** -
  `feat(kernels): pf_gated_head + pf_attn_prep + the M=64 L1 attention route`

---

## `gdn_chunk` - the algorithm, written out once (Tasks 6-8 implement this)

The WY-representation chunked gated delta rule, transcribed from the FLA
reference vLLM runs (`~/PycharmProjects/vllm/vllm/third_party/flash_linear_attention/ops/`:
`chunk.py:23-82` is the driver, `cumsum.py:27-71`, `chunk_scaled_dot_kkt.py:46-112`,
`solve_tril.py:38-100`, `wy_fast.py:30-115`, `chunk_delta_h.py:88-317`,
`chunk_o.py:84-137` are the six stages). Intra-chunk size **64**
(`utils.py:31` `FLA_CHUNK_SIZE = 64`; Intel's CuTe kernel agrees -
`vllm-xpu-kernels/csrc/xpu/gdn_attn/gdn_attn_utils.h:8` `chunk_size_xe2 = 64`).

For one v-head `h`, one 64-chunk covering positions `p_0 … p_{L-1}` (`L ≤ 64`),
with the state `S ∈ R^{128×128}` k-major (`S[k][x]` - decode's `gdn_state`
layout, `gdn_step.cl:22`) **as it stands at the chunk's first position**:

```
g[i]      = negA[h] · softplus(f32(rne(a[p_i])) + dt_bias[h])            fp32, <= 0
beta[i]   = 1 / (1 + exp(-f32(rne(b[p_i]))))                             fp32
gc[i]     = SUM_{j<=i} g[j]                                              fp32, j ascending
gl        = gc[L-1]
q[i][k]   = f32(qf_b[p_i][k]) * Q_SCALE      (bf16 in gdn_xb; Q_SCALE = 1/sqrt(128))
k[i][k]   = f32(kf_b[p_i][k])                (bf16 in gdn_xb)
v[i][x]   = f32(xb_v[p_i][x])                (bf16 in gdn_xb, conv+SiLU output)

A[i][j]   = beta[i] * (SUM_k k[i][k]*k[j][k]) * exp(gc[i] - gc[j])   for i > j, else 0
T         = (I - A)^-1                                               unit lower triangular
vb[j][x]  = rne( v[j][x] * beta[j] )                          Q1  bf16 (FLA wy_fast:88)
kb[j][k]  = rne( k[j][k] * beta[j] * exp(gc[j]) )             Q2  bf16 (FLA wy_fast:110)
u[i][x]   = rne( SUM_{j<=i} T[i][j] * f32(vb[j][x]) )         Q3  bf16 (FLA wy_fast:89)
w[i][k]   = rne( SUM_{j<=i} T[i][j] * f32(kb[j][k]) )         Q4  bf16 (FLA wy_fast:112)
A2[i][j]  = (SUM_k q[i][k]*k[j][k]) * exp(gc[i] - gc[j])             for j <= i, else 0

# the sequential step. S is the chunk-start state for BOTH lines below.
vn[i][x]  = f32(u[i][x]) - SUM_k f32(w[i][k]) * S[k][x]                       fp32
o[i][x]   = ( SUM_k q[i][k] * S[k][x] ) * exp(gc[i])  +  SUM_{j<=i} A2[i][j] * vn[j][x]
vs[i][x]  = vn[i][x] * exp(gl - gc[i])
S[k][x]  <- S[k][x] * exp(gl)  +  SUM_i k[i][k] * vs[i][x]                    i ascending
```

`rne(·)` is `common::f32_to_bf16` / the kernels' `rne_bf16` - the same
add-and-shift every `.cl` in the tree uses. **Q1-Q4 are the four bf16 rounding
points the chunked form adds and for which decode has no twin**; every other
rounding in the block is one of P1-P8 (Task 6 Step 1's list, read out of
`gdn_step.cl`). Q1-Q4 are FLA's, kept because they are the reference vLLM runs;
they are what Task 7's `pf_gdn_wu` bar is written against.

Three properties are load-bearing and each has a task that pins it:

- **`A2`'s mask includes the diagonal** (`chunk_o.py:123`, `>=`) and **`A`'s
  does not** (`chunk_scaled_dot_kkt.py:108`, `>`). Getting that pair backwards
  is an off-by-one in the recurrence that a 64-position test catches and a
  1-position test does not - Task 8 Step 1 case 3 is that test.
- **`o` reads the chunk-start `S`, not the updated one** (`chunk_o.py` is a
  separate kernel over `chunk_delta_h`'s per-chunk `h` snapshots). The fused
  scan of Task 8 computes `vn` and `o` before touching `S`, which is what
  removes FLA's `[C/64][48][128][128]` fp32 `h` buffer (**201,326,592 B at
  C = 4096, derived**) entirely.
- **`Q_SCALE` is folded into `q` at read**, not applied as FLA's trailing
  `* scale` (`chunk_o.py:137`). That is algebraically the same for both terms
  of `o` and it is what makes our `q` decode's `qf_s`
  (`gdn_step.cl:312` - rounded to bf16, then widened and scaled in fp32).

Two deliberate deviations from FLA, both recorded (ruling R8): its `solve_tril`
rounds `T` to bf16 (`chunk.py:47-49`, `output_dtype=k.dtype`) and its
`chunk_delta_h` rounds `v_new` to bf16 before the `S` update
(`chunk_delta_h.py:274`). Both stay **fp32** here, per spec §3.4's "fp32
throughout" and because the reference this is graded against is the fp32
recurrent `gdn_ref.h`.

### Task 6: `gdn_chunk` part A - conv seed/writeback, the batched conv1d, l2norm, gate + cumsum

**Files:**
- Create: `src/kernels/prefill/pf_gdn_conv.cl` (four entry points:
  `pf_gdn_seed`, `pf_gdn_conv`, `pf_gdn_l2norm`, `pf_gdn_gate`),
  `tests/prefill/gdn_chunk_ref.h` (the host chunked reference, fp32),
  `tests/prefill/gdn_conv_test.cc`
- Modify: `src/kernels/prefill/pf_kernels.h`,
  `src/kernels/prefill/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `gdn_step.cl:223-266` (the conv prologue and the ring), `:73-93`
  (the ring-ownership argument this task replaces), `:164-178` (the
  `M + 3 <= RING` constraint this task replaces), `:286-315` (the head scalars
  and the l2norm), `tests/kernels/gdn_ref.h:115-182`, `cumsum.py:27-71`.
- Produces:

```cpp
// src/kernels/prefill/pf_kernels.h
inline std::string pf_gdn_conv_variant();   // "pf_gdn_conv" - all four entries
```

```c
// pf_gdn_conv.cl, all four M-free. CONV_ROWS 10240, CONV_TAPS 4, RING 16,
// HEADS 48, DIM 128, QKVZ_N 16384, Q_OFF 0, K_OFF 2048, V_OFF 4096 - the same
// literals gdn_step.cl defines, from model::Qwen35.

// (1) Lift the ring's three older slots into a flat seed. Grid (40), WG 256:
// one work-item per channel. Positions below zero contribute 0 -- the
// reference's conv state is zero-initialised (gdn_step.cl:243).
//   seed[t][ch] = pos-3+t < 0 ? 0 : conv_ring[((pos-3+t) % RING)][ch]   t = 0,1,2
__kernel void pf_gdn_seed(__global const ushort* restrict conv_ring,
                          __global ushort* restrict seed, uint pos);

// (2) The batched depthwise 4-tap causal conv1d + SiLU over the whole chunk,
// AND the ring writeback for the chunk's last min(C,3) positions. One
// work-item owns one channel for every position, so the window's newer slots
// are its own registers and the only ring traffic is (1)'s seed and this
// kernel's <=3 stores. Grid (40), WG 256.
//
// THIS IS WHAT REPLACES THE `M + 3 <= RING` ARGUMENT (gdn_step.cl:176-178):
// the seed was lifted by a PRIOR launch, so no work-group reads a slot any
// work-group is writing, whatever C is. The list is in-order; that is the
// whole proof, and it is why the ring is not made deeper.
__kernel void pf_gdn_conv(__global const float* restrict qkvz_partials,
                          __global const ushort* restrict seed,
                          __global const float* restrict conv_w,
                          __global ushort* restrict xb,
                          __global ushort* restrict conv_ring,
                          uint pos, uint c_count);

// (3) l2norm q and k IN PLACE in `xb`, per (position, k-head). Grid (32, C),
// WG 128: group x in [0,16) is the q of k-head x, [16,32) the k of k-head x-16.
__kernel void pf_gdn_l2norm(__global ushort* restrict xb, uint c_count);

// (4) Head scalars and the intra-chunk inclusive cumulative gate. Grid
// (48 heads, ceil(C/64)), WG 64 = one lane per position of the 64-chunk.
//   g_out[i][h]    = the CUMULATIVE gc[i] (fp32)
//   beta_out[i][h] = beta[i] (fp32)
__kernel void pf_gdn_gate(__global const float* restrict ab_out,
                          __global const float* restrict gdn_small,
                          __global float* restrict g_out,
                          __global float* restrict beta_out, uint c_count);
```

- [ ] **Step 1: Read the rounding points and write them down.** Read
  `src/kernels/gdn_step.cl` lines **96-128** (the discipline preamble),
  **223-266** (the conv), **286-315** (the scalars and l2norm) and
  `src/kernels/prep.cl` lines **337-385** (`prep_gated_head`). Mirror the
  rounding at exactly these points, and put this list verbatim in
  `pf_gdn_conv.cl`'s header:
  - **P1** `gdn_step.cl:253` - `raw_b = rne_bf16(qkvz_partials[(0*M+m)*16384 + ch])`,
    the qkv linear's single bf16 output; **this is what the ring stores** (the
    reference's conv state holds the *input* sequence, not the convolved one).
  - **P2** `gdn_step.cl:256-260` - the conv accumulates fp32 over widened bf16
    inputs and fp32 weights, taps **ascending** (t = 0 oldest, t = 3 the
    current position), with **explicit `fma`**.
  - **P3** `gdn_step.cl:261` - `xb = rne_bf16(silu_f32(acc))`, the activation's
    bf16 output; `silu_f32(x) = x / (1 + exp(-x))`, plain `exp`, never
    `native_exp`.
  - **P4** `gdn_step.cl:296-308` - the l2norm sums squares of the widened bf16
    in fp32, **one term per lane so a plain multiply and no `fma`**, then the
    128-wide pairwise tree `stride = 64, 32, …, 1` with `r[i] += r[i+stride]`
    and a barrier after every step.
  - **P5** `gdn_step.cl:309-310` - `inv = 1.0f / sqrt(sum + 1e-6f)`, **never
    `rsqrt`**.
  - **P6** `gdn_step.cl:312` - `qf = bf16f(rne_bf16(bf16f(xb_q) * inv_q)) * Q_SCALE`:
    the normalised value is rounded to bf16 and the `1/sqrt(128)` scale is
    applied **after** the round, in fp32. `pf_gdn_l2norm` therefore stores the
    **rounded, unscaled** word and every consumer multiplies by `Q_SCALE` in
    fp32 at read.
  - **P7** `gdn_step.cl:314` - `kf = bf16f(rne_bf16(bf16f(xb_k) * inv_k))`,
    not scaled.
  - **P8** `gdn_step.cl:289-291` - `a_b = rne_bf16(ab_out[m*128 + h])`,
    `b_b = rne_bf16(ab_out[m*128 + 48 + h])`, then
    `g = negA * softplus_f32(bf16f(a_b) + dt_bias)` and
    `beta = 1/(1 + exp(-bf16f(b_b)))` in fp32.
    `softplus_f32(x) = x > 20.0f ? x : log1p(exp(x))` - torch's threshold,
    spelled identically.
  - **P9** `gdn_step.cl:317-356` - **the recurrence rounds nothing** (Task 8).
  - **P10** `gdn_step.cl:368-370` - `gdn_o` is written fp32.
  - **P11-P13** `prep.cl:365-384` - `prep_gated_head`'s five RNE points, which
    `pf_gated_head` already mirrors (Task 5 Step 3).
- [ ] **Step 2: Failing test first.** `tests/prefill/gdn_conv_test.cc`, one
  synthetic layer, no checkpoint:
  1. **The conv, bit-exact against `gdn_ref`, over 4096 positions in one
     chunk.** Fill `qkvz_partials` fp32 `[4096][16384]` with a seeded PRNG,
     zero the ring, run `pf_gdn_seed` + `pf_gdn_conv` once at `pos = 0`,
     `c_count = 4096`; then run `gdn_ref::step(p, 1, 1, …)` 4096 times against
     a host copy of the ring collecting its `xb` per position. Bars, all
     **`memcmp`-exact, no tolerance** (the chain is fp32 `fma` over bf16 words
     - the only transcendental is `silu`'s `exp`, which is identical between
     two runs of *the same device*, and the host bar is therefore stated
     against `gdn_ref` as **≤ 2 ulp on `xb`** and **bit-exact on the ring**,
     because the ring stores `raw_b`, which no `exp` touches):
     - `conv_ring`'s three live slots after the chunk: **bit-exact** vs the
       recurrent walk's final ring;
     - `xb`: ≤ 2 ulp of `gdn_ref`'s per-position `xb` (state the count of
       differing words and the max ulp; require max ≤ 2).
  2. **Multi-chunk seeding.** Same data run as **four** chunks of 1024
     (`pos = 0, 1024, 2048, 3072`), each a seed + conv pair: `xb` and the ring
     must be **bit-identical to the single-chunk run**. This is the test that
     proves the explicit seed/writeback replaces the `M + 3 <= RING` argument.
  3. **The ragged tail.** `C = 1` and `C = 2` chunks (so `min(C,3) < 3` slots
     are written) followed by a `C = 64` chunk: `xb` bit-identical to the
     single-chunk run over the same 67 positions.
  4. **`pf_gdn_l2norm`.** Run it on `xb` and compare `qf`/`kf` against
     `gdn_ref`'s (which computes them per position): **bit-exact** - the tree
     is the same 128-lane shape and `1.0f / sqrt` is correctly rounded.
     Additionally require that the stored q word is the **unscaled** one
     (P6): `f32(stored) * Q_SCALE == gdn_ref's qf[i]` exactly.
  5. **`pf_gdn_gate`.** Compare `beta` **bit-exact** against `gdn_ref`'s and
     `gc[i]` against a host ascending fp32 prefix sum of `gdn_ref`'s `g` -
     bit-exact within a 64-chunk, because both are the same ascending fp32
     chain and `exp`/`log1p` are evaluated once per position on the same
     device in both paths. Bar: `memcmp` on `beta`; `memcmp` on `gc` **if the
     host `exp`/`log1p` agree**, otherwise ≤ 3 ulp with the count printed -
     read the measured result and pin whichever it is, with the number.
  6. **Chunk-boundary reset.** `gc` must restart at every multiple of 64
     (`gc[64] == g[64]`, not `gc[63] + g[64]`), and the test asserts it
     directly - a running cumsum across the whole chunk is the single most
     likely transcription error in this task.
- [ ] **Step 3: Run it, see it fail.** `tools/box.sh test gdn_conv_test`. Paste.
- [ ] **Step 4: Write `pf_gdn_conv.cl`.** Structure per entry point:
  - `pf_gdn_seed`: `for (ch = gid; ch < CONV_ROWS; ch += stride)`, three
    guarded loads at `(pos-3+t)`, `seed[t*CONV_ROWS + ch] = …`.
  - `pf_gdn_conv`: per channel, `win0..2` from `seed`, `win3` from
    `rne_bf16(qkvz_partials[(size_t)m * QKVZ_N + ch])` (P1), the four
    ascending `fma` taps (P2), `xb[(size_t)m * CONV_ROWS + ch] =
    rne_bf16(silu_f32(acc))` (P3), then the shift. Ring writeback: after the
    loop, `for (t = 0; t < 3; ++t) { const int m = (int)c_count - 3 + t;
    if (m >= 0) conv_ring[((size_t)((pos + (uint)m) % RING)) * CONV_ROWS + ch]
    = rne_bf16(qkvz_partials[(size_t)m * QKVZ_N + ch]); }` - recomputing the
    RNE rather than caching three registers, because it is three loads out of
    `c_count` and the code stays one expression.
    **No `own_ring` predicate**: every channel is owned by exactly one
    work-item here (the grid is over channels, not over heads), so
    `gdn_step.cl:73-93`'s four-fold redundancy and its ownership rule are gone
    - say so in the header, with the line reference.
  - `pf_gdn_l2norm`: `wg = get_group_id(0)`, `m = get_group_id(1)`,
    `i = get_local_id(0)`; `is_q = wg < 16`, `kh = is_q ? wg : wg - 16`;
    base `= (is_q ? Q_OFF : K_OFF) + kh * DIM`; read, square (plain multiply,
    P4), tree, `inv` (P5), write `rne_bf16(bf16f(word) * inv)` back to the same
    slot (P6/P7). A barrier before the writeback so no lane overwrites a word
    another lane has yet to read - it does not, each lane owns its own `i`, but
    the barrier after the tree is already there and the note says why none is
    added.
  - `pf_gdn_gate`: `h = get_group_id(0)`, `chunk = get_group_id(1)`,
    `i = get_local_id(0)` (WG 64); `m = chunk * 64 + i`; `if (m < c_count)`
    compute `g`/`beta` per P8 into `__local float gs[64]`; barrier; **lane 0
    walks `gs` ascending accumulating a single fp32 running sum** and writes
    `g_out[(size_t)m * HEADS + h]` - a sequential scan, not a Hillis-Steele
    tree, because 64 elements are nothing and a sequential ascending sum is
    exactly `tl.cumsum`'s order (`cumsum.py:67`) and exactly what the host
    reference will do.
- [ ] **Step 5: Write `tests/prefill/gdn_chunk_ref.h`** - the host fp32
  chunked reference for the *whole* algorithm block above, in the same six
  stages, mirroring every rounding point P1-P8 and Q1-Q4 (the `vb`/`kb`/`u`/`w`
  bf16 rounds named in the algorithm section). Tasks 7 and 8 grade against it;
  this task uses only its conv/l2norm/gate stages. Its header repeats the
  algorithm block verbatim, the way `gdn_ref.h` repeats `gdn_step.cl`'s
  ordering statements, and says the two files must be edited together.
- [ ] **Step 6: AOT row** -
  `add_ocloc_kernel(pf_gdn_conv SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_conv.cl
   DEFINES NEGA_OFF=40960 DTBIAS_OFF=41008)` (the same offsets-in-floats
  `src/kernels/CMakeLists.txt:153` passes, owned by `loader/small_layout.h`).
- [ ] **Step 7: Run it, see it pass.** `tools/box.sh test gdn_conv_test`, and
  paste the measured ulp counts from cases 1, 4 and 5 into the commit body -
  they are the first numbers of this stream's numerics band.
- [ ] **Step 8: Commit** -
  `feat(kernels): pf_gdn conv/l2norm/gate - batched conv1d with explicit ring seed+writeback`

### Task 7: `gdn_chunk` part B - `A`, the unit-lower-triangular solve, `W`/`U`, `A2`

**Files:**
- Create: `src/kernels/prefill/pf_gdn_wy.cl` (four entry points:
  `pf_gdn_A`, `pf_gdn_solve`, `pf_gdn_wu`, `pf_gdn_A2`),
  `tests/prefill/gdn_wy_test.cc`
- Modify: `src/kernels/prefill/pf_kernels.h`,
  `src/kernels/prefill/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: the algorithm block above; `chunk_scaled_dot_kkt.py:82-112`,
  `solve_tril.py:64-100`, `wy_fast.py:69-115`, `chunk_o.py:90-125`;
  `tests/prefill/gdn_chunk_ref.h` (Task 6 Step 5).
- Produces:

```cpp
inline std::string pf_gdn_wy_variant();   // "pf_gdn_wy" - all four entries
```

```c
// pf_gdn_wy.cl. CT 64 is the intra-chunk size (PrefillScratch::kGdnChunk).
// Every kernel's grid is (48 heads, nchunks = ceil(C/64)); `c_count` is the
// chunk's total position count so the tail chunk's live length is
// L = min(CT, c_count - chunk*CT) and every loop is bounded by it.

// A[i][j] = beta[i] * (SUM_k k[i][k]*k[j][k]) * exp(gc[i]-gc[j]) for i > j.
// WG 256; stages k[CT][DIM] bf16 (16 KB), gc[CT] and beta[CT] fp32 in SLM,
// then work-item `lid` walks pairs p = lid, lid+256, ... over CT*CT.
__kernel void pf_gdn_A(__global const ushort* restrict xb,
                       __global const float* restrict g_cum,
                       __global const float* restrict beta,
                       __global float* restrict A, uint c_count);

// T = (I - A)^-1, IN PLACE over A. WG 64 = one lane per column j. Forward
// substitution, i ascending:  T[i][j] = A[i][j] + SUM_{j<l<i} A[i][l]*T[l][j].
// A[i][*] is read at step i, before row i is overwritten; T[l][*] for l < i is
// already written. `l` ascending, explicit fma.
__kernel void pf_gdn_solve(__global float* restrict A, uint c_count);

// vb/kb (Q1/Q2), then u = T*vb and w = T*kb (Q3/Q4), stored bf16.
// WG 256; stages T[CT][CT] fp32 (16 KB) + vb[CT][DIM] and kb[CT][DIM] bf16
// (16 KB each) = 48 KB SLM of the 128 KB a work-group may have.
__kernel void pf_gdn_wu(__global const ushort* restrict xb,
                        __global const float* restrict T,
                        __global const float* restrict g_cum,
                        __global const float* restrict beta,
                        __global ushort* restrict w,
                        __global ushort* restrict u, uint c_count);

// A2[i][j] = (SUM_k q[i][k]*k[j][k]) * exp(gc[i]-gc[j]) for j <= i.
// NOTE THE DIAGONAL: `>=`, where pf_gdn_A uses `>`.
__kernel void pf_gdn_A2(__global const ushort* restrict xb,
                        __global const float* restrict g_cum,
                        __global float* restrict A2, uint c_count);
```

- [ ] **Step 1: Failing test first.** `tests/prefill/gdn_wy_test.cc`, one
  synthetic head set, `C = 256` (four chunks) and `C = 4096` (64 chunks):
  1. **`pf_gdn_A` vs `gdn_chunk_ref`** - fp32, bar ≤ 4 ulp relative on every
     non-zero entry (`exp` carries 3 ulp; the 128-term dot is the same
     ascending `fma` chain on both sides), and **exactly 0.0f** on every
     `i <= j` entry. Print the count of non-zeros and require it equals
     `nchunks * 48 * L*(L-1)/2`.
  2. **`pf_gdn_solve` - the algebraic identity, not a reference.** Read back
     `T`, multiply `(I - A) * T` on the host in fp64 and require
     `max |result - I| <= 1e-9` (fp64 host arithmetic over fp32 inputs, so the
     bar tests the kernel, not the host). Additionally require `T[i][i] ==
     1.0f` exactly and `T[i][j] == 0.0f` exactly for `j > i`. This is a much
     stronger bar than comparing to a host substitution and it is the reason
     to state it this way.
  3. **`pf_gdn_wu` vs `gdn_chunk_ref`** - `w` and `u` are bf16; bar
     **bit-identical**, because every step is fp32 `fma` in a stated order
     with one RNE at the end and no transcendental except the `exp(gc[j])`
     factor in `kb`, which is evaluated once per (j, head) on the device in
     both paths. If it is not bit-identical, print the differing-word count
     and the max ulp, pin ≤ 2 ulp, and record why in the file header - do not
     widen silently.
  4. **`pf_gdn_A2` vs `gdn_chunk_ref`** - bar as case 1, and **the diagonal
     is non-zero**: assert `A2[i][i] != 0.0f` for at least one `i` per head,
     which is the direct test for the `>=` / `>` pair.
  5. **`L < 64` tail.** `C = 100` (chunks of 64 and 36): every entry with
     `i >= L` or `j >= L` in the tail chunk must be **exactly 0.0f** in `A`,
     `T` (except the identity diagonal, which the kernel must NOT write past
     `L`) and `A2`, and `w`/`u` rows `>= L` untouched. State which convention
     the kernel takes for `T`'s dead diagonal and assert it.
- [ ] **Step 2: Run it, see it fail.** `tools/box.sh test gdn_wy_test`. Paste.
- [ ] **Step 3: Write `pf_gdn_A` and `pf_gdn_A2`** in one file, sharing an
  SLM-staging prologue (`k[CT][DIM]` bf16 from `xb + K_OFF + kh*DIM`, `gc`,
  `beta`) and differing in three lines: the left operand (`k` vs
  `q = f32(word) * Q_SCALE`), the `beta[i]` factor (present vs absent) and the
  mask (`i > j` vs `j <= i`). Write the two entry points side by side in the
  file with a comment on the mask pair, so a reader checks them against each
  other in one screen.
- [ ] **Step 4: Write `pf_gdn_solve`.** Stage `A[CT][CT]` into
  `__local float As[CT][CT]` (16 KB). Then:

```c
  for (uint i = 0; i < L; ++i) {
    float acc = 0.0f;
    if (j < i) {
      acc = As[i][j];
      for (uint l = j + 1; l < i; ++l) acc = fma(As[i][l], As[l][j], acc);
    } else if (j == i) {
      acc = 1.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);   // every lane has READ row i's A values
    if (j <= i) As[i][j] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);   // row i is T before step i+1 reads it
  }
```

  The two barriers are the whole correctness argument for the in-place solve
  and the comment says exactly that: the first separates "all lanes have read
  `As[i][*]` as `A`" from "row `i` becomes `T`", the second separates that
  write from step `i+1`'s reads. Rows `>= L` are left as `pf_gdn_A` wrote them
  (zero) and lanes `j > i` write nothing.
- [ ] **Step 5: Write `pf_gdn_wu`.** Prologue: each work-item computes some of
  `vb[j][x] = rne(f32(v)*beta[j])` and `kb[j][k] = rne(f32(k)*beta[j]*exp(gc[j]))`
  into SLM (Q1/Q2 - round **once**, at the end of each expression, as
  `wy_fast.py:88` and `:110` do with `.to(dtype)`), barrier, then
  `for (p = lid; p < L*DIM; p += 256) { i = p/DIM; x = p%DIM;
   float au = 0, aw = 0; for (j = 0; j <= i; ++j) { au = fma(Ts[i][j], f32(vbs[j][x]), au);
   aw = fma(Ts[i][j], f32(kbs[j][x]), aw); } u[…] = rne(au); w[…] = rne(aw); }`
  - `j` ascending, one RNE each (Q3/Q4). `u` and `w` are `[C][48][128]` bf16,
  indexed `((size_t)m * HEADS + h) * DIM + x`.
- [ ] **Step 6: AOT row** -
  `add_ocloc_kernel(pf_gdn_wy SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_gdn_wy.cl)`.
- [ ] **Step 7: Run it, see it pass.** `tools/box.sh test gdn_wy_test`; paste
  the four bars' measured values (max ulp / max `|(I-A)T - I|` / differing-word
  counts) into the commit body.
- [ ] **Step 8: Commit** -
  `feat(kernels): pf_gdn_wy - A, unit-lower-triangular solve, W/U, A2`

### Task 8: `gdn_chunk` part C - the sequential state scan, the host wrapper, the gate test

**Files:**
- Create: `src/kernels/prefill/pf_gdn_scan.cl`,
  `src/runtime/prefill/gdn.h`, `src/runtime/prefill/gdn.cc`,
  `tests/prefill/gdn_chunk_test.cc`
- Modify: `src/kernels/prefill/pf_kernels.h`,
  `src/kernels/prefill/CMakeLists.txt`, `src/runtime/CMakeLists.txt`,
  `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `gdn_step.cl:26-70` (the tile mapping and both reduction trees -
  this kernel reuses them exactly), `:269-381` (the register-resident state
  tile and its writeback), Tasks 6 and 7's kernels,
  `tests/prefill/gdn_chunk_ref.h`, `tests/kernels/gdn_ref.h`.
- Produces - **the S2 interface, which is what interfaces.md `gdn.h` declares
  (with change requests #1 and #2 applied):**

```cpp
// src/runtime/prefill/gdn.h
namespace runtime::prefill {
// The WY-representation chunked gated delta rule for ONE GDN layer over C new
// positions, intra-chunk 64. Reads and writes the SAME gdn_state and conv_ring
// buffers gdn_step uses, in place, so decode continues from position pos+C with
// no translation.
//
//   qkvz_partials  fp32 [C][16384]   the layer's qkv||z GEMV output, S = 1.
//                                    Rounded to bf16 inside, at the single
//                                    point gdn_step.cl:253 rounds (R4).
//   ab_out         fp32 [C][128]     decode's ab_out layout: a at [0,48),
//                                    b at [48,96), zero-padded to 128 (R4).
//   gdn_state      fp32 [48][128][128]  THIS layer's slice, k-major
//   conv_ring      bf16 [16][10240]     THIS layer's slice, slot-major
//   y              bf16 [C][6144]    the out_proj input (R3: gdn_chunk's last
//                                    launch is pf_gated_head)
// `s` supplies the nine kernels' scratch and `small` the layer's GDN block
// base (conv weights at 0, negA at NEGA_OFF, dt_bias at DTBIAS_OFF, the gated
// norm at loader::kGdnOffGatedNorm).
void gdn_chunk(Context& cx, PrefillScratch& s, uint32_t pos, uint32_t C,
               const float* qkvz_partials, const float* ab_out,
               const void* small, float* gdn_state, uint16_t* conv_ring,
               uint16_t* y);
}  // namespace runtime::prefill
```

```c
// pf_gdn_scan.cl - grid (48 heads, 4 state-column chunks), WG 256 = 16
// subgroups of 16 lanes. THE TILE MAPPING IS gdn_step.cl's, unchanged
// (gdn_step.cl:26-49): work-group (h, c) owns state columns [32c, 32c+32) of
// head h, work-item (sgid, lane) owns k-rows 8*sgid..+7 and columns
// 32c+lane and 32c+lane+16 -- 16 fp32 of state in registers. BOTH reduction
// trees are gdn_step.cl:52-65's: band sgid accumulates 8 terms in ascending kk
// with fma, then the 16 band partials collapse with stride = 8,4,2,1 and a
// barrier after every step. Choosing the same tile and the same trees is
// deliberate: it is the closest the chunked form can sit to the recurrent one,
// and spec 2 section 6.3's band is measured against exactly that choice.
//
// SLM: red[16][32] fp32 (2 KB) + vn[64][32] fp32 (8 KB) + A2s[64][64] fp32
// (16 KB) + gcv[64]/expg[64] fp32 (512 B) = ~27 KB.
__kernel void pf_gdn_scan(__global const ushort* restrict xb,
                          __global const ushort* restrict w,
                          __global const ushort* restrict u,
                          __global const float* restrict A2,
                          __global const float* restrict g_cum,
                          __global float* restrict state,
                          __global float* restrict gdn_o, uint c_count);
```

- [ ] **Step 1: Failing test first.** `tests/prefill/gdn_chunk_test.cc`, one
  synthetic GDN layer, no checkpoint, `TIMEOUT 1800`:
  1. **vs the CPU fp32 recurrent reference over 4096 positions - the
     diagnostic band.** Zero `gdn_state` and `conv_ring`; seeded-random
     `qkvz_partials` fp32 `[4096][16384]`, `ab_out` fp32 `[4096][128]`,
     `gdn_small`. Run `gdn_chunk(pos=0, C=4096)` once; separately run
     `gdn_ref::step(p, 1, 1, …)` 4096 times on the host over host copies.
     Report and **record in the commit body**: `gdn_state` max and mean
     relative difference, `gdn_o` max/mean rel-diff, and the `y` (post-gated-
     head) max/mean rel-diff. **This is a diagnostic, not a gate** (spec §6.3:
     tokens gate, tensors diagnose). The test *fails* only on a non-finite
     value or a max rel-diff above **1e-2**, a loose tripwire whose only job is
     to catch a transcription error rather than to grade rounding - say so at
     the assert.
  2. **vs decode's `gdn_step` run 4096 times - the self-consistency oracle.**
     The same inputs driven through the **device** `gdn_step_M1` +
     `prep_gated_head_M1` pair, 4096 launches, with `Control::pos` advanced by
     the host between them. Report the same three bands. Same tripwire. This
     is the comparison that isolates chunked-vs-recurrent rounding from
     host-vs-device rounding, and the two bands printed side by side are what
     the docs quote.
  3. **`C = 1` collapses to the recurrence.** Run `gdn_chunk` with `C = 1`,
     4096 times in sequence, against the same `gdn_step` walk. At `C = 1` the
     chunked algebra degenerates (`A` is empty, `T = I`, `u = vb`, `w = kb`,
     `A2` is the single diagonal entry), so the two paths differ only by
     association order on a single rank-1 update: require `gdn_state` max
     rel-diff **≤ 1e-5** and print it. A failure here is a bug in the
     degenerate path, and this is the only case that finds it.
  4. **Multi-chunk equals single-chunk.** `C = 4096` in one call vs four calls
     of 1024 vs 64 calls of 64: `gdn_state` and `conv_ring` **bit-identical**
     across all three (the chunk boundary must carry the state and the ring
     exactly, and 1024 is the width spec §6.2's gate uses).
  5. **The 64-chunk boundary inside a call.** `C = 100` (chunks of 64 and 36)
     vs two calls of 64 and 36: bit-identical state.
  6. **Determinism.** The `C = 4096` call twice from a zeroed state:
     `gdn_state`, `conv_ring`, `gdn_o` and `y` all **bit-identical**. No fp
     atomics exist in any of the nine kernels; this is the standing proof.
- [ ] **Step 2: Run it, see it fail.** `tools/box.sh test gdn_chunk_test`.
  Paste.
- [ ] **Step 3: Write `pf_gdn_scan.cl`.** Per work-group `(h, c)`: load the
  8×2 state tile into `float S[8][2]` (`gdn_step.cl:273-278` verbatim), then
  `for (uint t = 0; t * CT < c_count; ++t)`:
  1. `L = min(CT, c_count - t*CT)`; stage `A2s[L][L]` and `gcv[L]` into SLM,
     compute `expg[i] = exp(gcv[i])`, `gl = gcv[L-1]`, barrier.
  2. `for (i = 0; i < L; ++i)`: `vn[i][·]` - band partial
     `p0 = SUM_{kk} f32(w[i][8*sgid+kk]) * S[kk][0]` in ascending `kk` with
     `fma`, the stride-8/4/2/1 tree through `red`, then
     `vn[i][lane]   = f32(u[i][32c+lane])    - red_result0;`
     `vn[i][lane+16] = f32(u[i][32c+lane+16]) - red_result1;`
     (two columns per work-item, so two trees per `i` - run them the way
     `gdn_step.cl:323-339` runs `kv_red`'s pair, in one tree over a
     `[16][32]` array).
  3. `for (i = 0; i < L; ++i)`: the state term of `o` -
     `SUM_{kk} q[i][8*sgid+kk] * S[kk][·]` with
     `q = f32(xb[…Q_OFF…]) * Q_SCALE`, the same tree, `* expg[i]`; then the
     intra-chunk term, entirely local: `for (j = 0; j <= i; ++j) acc =
     fma(A2s[i][j], vn[j][col], acc);` `j` ascending, `fma`. Write
     `gdn_o[((size_t)(t*CT+i) * HEADS + h) * DIM + c*CHUNK_V + col]` fp32
     (P10). **`S` has not been touched yet** - that is the property the
     algorithm block calls out.
  4. `const float dl = exp(gl);` `for (kk) { S[kk][0] *= dl; S[kk][1] *= dl; }`
     then the batched update: `for (i = 0; i < L; ++i) { const float sc =
     exp(gl - gcv[i]); const float d0 = vn[i][lane] * sc, d1 = …;
     for (kk) { const float kfv = f32(xb[(t*CT+i)*CONV_ROWS + K_OFF + kh*DIM +
     8*sgid+kk]); S[kk][0] = fma(kfv, d0, S[kk][0]); S[kk][1] = fma(kfv, d1,
     S[kk][1]); } }` - `i` ascending, `kk` ascending, `fma`. **This is the one
     reassociation relative to decode** (decode interleaves the decay and the
     update per position, `gdn_step.cl:318-349`); the header says so and names
     it as the source of the band Step 1 records.
  5. Barrier at the end of the chunk iteration (the next chunk reuses `red`,
     `vn` and `A2s`) - `gdn_step.cl:372`'s reason, restated.
  After the chunk loop, write the tile back once (`gdn_step.cl:378-381`).
- [ ] **Step 4: Write `gdn.cc`.** Nine launches in this order, all on
  `cx`, all with `PtrArg`/`arg_val`:

```
  1 pf_gdn_seed   (conv_ring, s.gdn_seed, pos)            grid (40, 1, 1)   WG 256
  2 pf_gdn_conv   (qkvz_partials, s.gdn_seed, small,
                   s.gdn_xb, conv_ring, pos, C)           grid (40, 1, 1)   WG 256
  3 pf_gdn_l2norm (s.gdn_xb, C)                           grid (32, C, 1)   WG 128
  4 pf_gdn_gate   (ab_out, small, s.gdn_g, s.gdn_beta, C)  grid (48, nch, 1) WG 64
  5 pf_gdn_A      (s.gdn_xb, s.gdn_g, s.gdn_beta,
                   s.gdn_A, C)                            grid (48, nch, 1) WG 256
  6 pf_gdn_solve  (s.gdn_A, C)                            grid (48, nch, 1) WG 64
  7 pf_gdn_wu     (s.gdn_xb, s.gdn_A, s.gdn_g, s.gdn_beta,
                   s.gdn_w, s.gdn_u, C)                   grid (48, nch, 1) WG 256
  8 pf_gdn_A2     (s.gdn_xb, s.gdn_g, s.gdn_A2, C)        grid (48, nch, 1) WG 256
  9 pf_gdn_scan   (s.gdn_xb, s.gdn_w, s.gdn_u, s.gdn_A2,
                   s.gdn_g, gdn_state, s.gdn_o, C)        grid (48, 4, 1)   WG 256
 10 pf_gated_head (qkvz_partials, s.gdn_o,
                   small + kGdnOffGatedNorm, y, C)        grid (48, C, 1)   WG 128
```

  with `nch = (C + kGdnChunk - 1) / kGdnChunk` and a
  `require(C <= PrefillScratch::kC)` throw at the top. **Ten launches**, not
  nine - `pf_gated_head` is the tenth per ruling R3, and Task 14's launch
  arithmetic uses 10.
  Note for plan 6c: launches 5, 7, 8 and the two contractions inside 9 are
  64×128×128-shaped per head and are plain vector code here; 6c **may** route
  5/7/8 through `gemm_bf16` (spec §3.4's "goes through the §3.2 interface where
  the shape suits DPAS, plain vector code where it does not - the solve").
  Launch 6 is the solve and stays vector code. Say this in `gdn.cc`'s header.
- [ ] **Step 5: Run it, see it pass.** `tools/box.sh test gdn_chunk_test`.
- [ ] **Step 6: Record the band.** Paste all three comparisons' max/mean
  rel-diffs (cases 1, 2, 3) into the commit body, each labelled **measured**
  with the box conditions. These are the numbers spec §6.3 asks to be recorded
  and the numbers docs/12 quotes; every later stage re-runs this test and any
  growth is a finding.
- [ ] **Step 7: Commit** -
  `feat(runtime): gdn_chunk - the WY chunked gated delta rule, ours (state band recorded)`

### Task 9: If plan 6a's P5 ruled Intel's CuTe kernel fits - wrap it instead

*Conditional. Read plan 6a's P5 report first. If P5 ruled **no fit** (or if it
has not run), this task is **skipped by ruling** and Tasks 6-8 stand. If P5
ruled **fit**, this task replaces `gdn_chunk`'s body and Tasks 6-8 are not
started - their `pf_gdn_*` kernels are never written, and Task 8's
`tests/prefill/gdn_chunk_test.cc` is kept verbatim, because it grades the
interface and not the implementation.*

**Files:**
- Create: `src/sycl/gdn_chunk_xe2.cc` (the wrapper translation unit),
  `src/kernels/prefill/pf_gdn_relayout.cl` (the state transpose and the a/b
  transpose), `tests/prefill/gdn_relayout_test.cc`
- Modify: `src/runtime/prefill/gdn.cc` (the body only - the header and the
  signature do not change), `src/kernels/prefill/CMakeLists.txt`,
  `CMakeLists.txt` / `src/sycl/CMakeLists.txt` (plan 6a owns the SYCL build
  rules; this task adds one source to them)

**Interfaces:**
- Consumes: plan 6a's SYCL build (`icpx -fsycl-targets=spir64_gen -device
  bmg-g31`, spec §3.2), plan 6a's `Context::sycl()`, and
  `~/PycharmProjects/vllm-xpu-kernels/csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp:1506-1525`:

```cpp
void gdn::chunk_gated_delta_rule_impl_xe2(
    sycl::queue& queue, torch::Tensor& core_attn_out,
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& b, const torch::Tensor& a,
    const torch::Tensor& A_log, const torch::Tensor& dt_bias,
    torch::Tensor& ssm_state,            // [cache_batch][num_v_heads][head_v_dim][head_k_dim]
    const torch::Tensor& query_start_loc, const torch::Tensor& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const int num_prefills, const int num_decodes, const int* token_indx);
```

- Produces: the same `runtime::prefill::gdn_chunk` signature Task 8 declares.
  **The interface does not move**; only its body does.

- [ ] **Step 1: Read P5's report and record the ruling** in `gdn.cc`'s header:
  which of P5's three questions passed (it builds; the state/conv layout
  matches or is cheaply adaptable; its fp32 numerics keep §6's gates), with
  P5's measured state max-rel-diff after 4096 positions and its µs per layer
  per chunk. If any of the three failed, write "skipped by ruling" here and
  stop.
- [ ] **Step 2: Price the torch dependency, which is the adoption's real
  cost.** The header includes `<torch/all.h>` and its entry point takes
  `torch::Tensor`. This engine links no libtorch and never will. Two ways, and
  the task picks by measurement, not preference:
  - **(a) de-torch the header** - copy it into `src/sycl/` and replace every
    `torch::Tensor` with a `{void* data; int64_t sizes[4]; int64_t strides[4];}`
    descriptor plus the `TORCH_CHECK`s as `throw`s. Count the `torch::`
    occurrences first: `grep -c "torch::\|TORCH_CHECK\|at::k" <header>` and
    paste it. If it is under ~80 this is a day's mechanical edit.
  - **(b) link libtorch** - record the size and the version coupling
    (`python3 -c "import torch, os; print(torch.__version__,
    os.path.dirname(torch.__file__))"` inside the reference container) and
    reject it if it pulls the container's SYCL runtime, which spec §8 keeps
    read-only.
  Write the choice and the number down before editing anything.
- [ ] **Step 3: Write `pf_gdn_relayout.cl`** - two entry points,
  `pf_gdn_state_to_vk` and `pf_gdn_state_to_kv`, transposing one layer's
  `[48][128][128]` fp32 state between our **k-major** `S[k][x]`
  (`gdn_step.cl:22`) and Intel's **v-major** `ssm_state[...][head_v_dim][head_k_dim]`
  (the signature above). Grid (48, 128), WG 128; 150,994,944 B / 48 =
  3,145,728 B per layer read + written = **6.3 MB per call, ~10.7 µs at the
  measured 590 GB/s (derived)**. Test: round-trip identity, bit-exact, in
  `tests/prefill/gdn_relayout_test.cc`.
- [ ] **Step 4: Write the wrapper.** `gdn_chunk`'s body becomes:
  1. launches 1-4 of Task 8's list stay ours (`pf_gdn_seed`, `pf_gdn_conv`,
     `pf_gdn_l2norm` - Intel's kernel takes q/k/v post-conv and post-l2norm,
     which P5 must confirm; if it takes them pre-l2norm, drop launch 3 and say
     so);
  2. `pf_gdn_state_to_vk(gdn_state → s.gdn_state_vk)`;
  3. `cx.wait()`, then `gdn::chunk_gated_delta_rule_impl_xe2(cx.sycl(), …)`
     with `query_start_loc = {0, C}`, `cache_indices = {0}`,
     `has_initial_state = {true}`, `num_prefills = 1`, `num_decodes = 0`,
     `token_indx = nullptr` - the solo-request, whole-cache configuration
     spec §3.3 uses for FMHA, applied here;
  4. `pf_gdn_state_to_kv(s.gdn_state_vk → gdn_state)`;
  5. `pf_gated_head` as launch 10, unchanged.
  `a` and `b` come from `ab_out` columns `[0,48)` and `[48,96)`; note that
  Intel's `a`/`b` are `[num_v_heads][total_seqlen]` - **transposed** relative
  to our `[C][128]` - so a fourth relayout kernel (`pf_gdn_ab_transpose`,
  4096×48 fp32 = 786,432 B, negligible) is needed and is part of this step.
  `A_log` must be fp32 and `dt_bias` must match the output dtype (bf16) - our
  loader bakes `negA = -exp(A_log)` (`model::SmallBake::NegExpFp32`), so the
  wrapper must **undo** that (`A_log = log(-negA)`) into a small fp32 `[48]`
  device buffer once per layer at load time, or Intel's `act_softplus` path
  computes a different `g`. Record this as the sharpest edge of the adoption.
- [ ] **Step 5: Add `PrefillScratch::gdn_state_vk`** fp32 `[48][128][128]` =
  **3,145,728 B** and `gdn_ab_t` fp32 `[2][48][kC]` = **1,572,864 B**, update
  `tests/runtime/buffers_test.cc`'s pinned total by exactly those two lines
  (new total 1,961,713,560 + 4,718,592 = **1,966,432,152**), and delete the
  eight `pf_gdn_*` scratch lines Tasks 6-8 would have needed
  (`gdn_xb` stays; `gdn_seed` stays; `gdn_g`, `gdn_beta`, `gdn_A`, `gdn_A2`,
  `gdn_w`, `gdn_u` go - 786,432 + 786,432 + 4 × 50,331,648 =
  **−202,899,456 B**, new total **1,763,532,696**).
  Write the arithmetic in the test comment; do not carry both tables.
- [ ] **Step 6: Run Task 8's test verbatim.** `tools/box.sh test
  gdn_chunk_test` - every one of its six cases must pass against Intel's
  kernel, including case 4's bit-identical multi-chunk carry and case 6's
  determinism (spec §6.4 forbids any split-K atomic; grep the header for
  `atomic` and paste the result).
- [ ] **Step 7: Record the band** exactly as Task 8 Step 6 does, and add a
  reconciliation sentence against P5's own figure - two values for one
  quantity are not allowed to stand unexplained (spec §8).
- [ ] **Step 8: Commit** -
  `feat(runtime): gdn_chunk via Intel's Xe2 CuTe kernel (P5 ruled fit) - band recorded`

### Task 10: `Engine::prefill()` - the chunk walk and the `Control` handoff

**Files:**
- Create: `src/runtime/prefill/step.h`, `src/runtime/prefill/step.cc` (the
  per-chunk walk, so `engine.cc` stays the loop it is)
- Modify: `src/runtime/engine.h`, `src/runtime/engine.cc`,
  `src/runtime/CMakeLists.txt`
- Test: `tests/prefill/prefill_smoke_test.cc` (created here; the three gates
  are Task 12)

**Interfaces:**
- Consumes: `runtime::prefill::Context` (Task 2), every `pf_*` variant
  (Tasks 3-5), `runtime::prefill::gdn_chunk` (Task 8 or 9),
  `runtime::prefill::attn_prep_l1` / `attn_l1` (Task 5),
  `PrefillScratch` (Task 1), `runtime::Control` (`src/runtime/control.h`),
  `loader::LoadedModel`, `model::Qwen35::layers()`,
  `kernels::gemv_variant` / `gemv_bf16_variant` / `argmax_stage1_variant` /
  `argmax_stage2_variant` (for the `lm_head` + sampler tail).
- Produces:

```cpp
// src/runtime/engine.h additions
  // Prefill `ids` in chunks of at most `chunk` positions starting at the
  // current pos; on return pos == old pos + ids.size(), the persistent state
  // holds those positions, and control_->cur_token[0] is the argmax of the
  // LAST position's logits (the first generated id), exactly as
  // ingest-by-decode would leave it. Default chunk = PrefillScratch::kC.
  //
  // Throws if ids is empty, if chunk > PrefillScratch::kC, if any id >=
  // Qwen35::kVocab (pf_embed_gather has no Control::debug_flag channel - Task
  // 3 Step 3 - so the host is the only bound), or if pos + ids.size() >
  // max_len. Allocates PrefillScratch and the prefill Context on first call
  // (ruling R7): a decode-only Engine's device residency is unchanged.
  void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);
  // Non-null only after the first prefill(); for tests and the CLI's report.
  const PrefillScratch* prefill_scratch() const { return pf_.get(); }
  size_t prefill_launches() const;    // Context::launches(), 0 before the first call
```

```cpp
// src/runtime/prefill/step.h
namespace runtime::prefill {
// One chunk of at most PrefillScratch::kC positions, at absolute position
// `pos`. The caller has already uploaded the chunk's ids into `s.ids` and set
// Control::{pos, n_active}; this function appends every launch of the chunk to
// `cx` and returns without waiting.
void step_chunk(Context& cx, PrefillScratch& s, const loader::LoadedModel& m,
                uint32_t max_len, void* ctrl, uint32_t pos, uint32_t C);
// The chunk tail that only the LAST chunk runs: the final norm on the last
// position, lm_head at M = 1, and the two argmax stages -- the Control handoff.
void step_head(Context& cx, PrefillScratch& s, const loader::LoadedModel& m,
               void* ctrl, uint32_t last_row);
}  // namespace runtime::prefill
```

- [ ] **Step 1: Failing test first.** `tests/prefill/prefill_smoke_test.cc`,
  label `checkpoint`, `TIMEOUT 1800`, argv[1] = the snapshot:
  1. `Engine eng(ctx, model, 16384)`; `CHECK(eng.prefill_scratch() == nullptr)`
     and `CHECK_EQ(eng.buffers().persistent_bytes(), 1240465536)` - ruling R7's
     property, asserted.
  2. `eng.reset(); eng.prefill(first 64 ids of kBenchPrompt cycled);`
     then `CHECK_EQ(eng.pos(), 64u)`, `CHECK(eng.prefill_scratch() != nullptr)`,
     `CHECK(eng.prefill_launches() == 1158)` - the launch count derived in
     Task 14 Step 2 for `C = 64` (`48*20 + 16*(9 + 3*ceil(64/64)) + 6`), pinned
     here so a walk change is a diff with arithmetic. Print it if it differs.
  3. `Control::cur_token[0] < Qwen35::kVocabUsed` and `out_token[0] ==
     cur_token[0]` - the same handoff invariant `argmax_stage2` leaves for
     decode (`argmax.cl:166-169`).
  4. `eng.generate(8)` runs and returns eight ids all `< kVocabUsed` - the
     decode list continuing from prefill's state, which is the whole point.
  5. Multi-chunk: `eng.reset(); eng.prefill(200 ids, /*chunk=*/64)` →
     `pos == 200`, and no throw at the ragged last chunk of 8.
  6. Rejections: empty `ids` throws; `chunk = 5000` throws naming
     `PrefillScratch::kC`; an id `== Qwen35::kVocab` throws naming the
     vocabulary; `prefill` of `max_len + 1` ids throws naming `max_len`.
- [ ] **Step 2: Run it, see it fail.** `tools/box.sh test prefill_smoke_test`.
  Paste.
- [ ] **Step 3: Write `step_chunk`.** The walk, per layer, in `capture.cc`'s
  order (`capture.cc:472-591`) with R1/R2/R3/R5 applied. Helpers first:

```cpp
namespace {
using model::LinearId; using model::Qwen35;
constexpr uint32_t kWgGemv = 64, kWgEmbed = 256, kWgFold = 256, kWgNorm = 256,
                   kWgSilu = 256, kWgGated = 128, kWgAttn = 256, kWgArgmax = 256;
uint8_t* at(const l0::Mem& m, size_t off) {
  return static_cast<uint8_t*>(m.ptr()) + off;
}
// Every int4 GEMV on the prefill path: S = 1 (R1), grid (N/64, ceil(M/8)).
void pf_gemv(Context& cx, const loader::DeviceWeight& w, const void* x,
             void* out, uint32_t M) {
  const model::GemvShape& sh = w.shape;
  l0::Kernel& k = cx.kernel(kernels::pf_gemv_variant(sh.K, sh.N, sh.layout),
                            "pf_gemv_int4_M", kWgGemv);
  cx.launch(k, sh.N / 64, (M + 7) / 8, 1,
            {PtrArg(w.mem.ptr()),
             PtrArg(sh.layout == 0 ? w.scales->ptr() : w.mem.ptr()),
             PtrArg(x), PtrArg(out), arg_val(M)});
}
// The res_norm pair, at `M` rows starting from the given row bases. `s_prev`
// is 0 for layer 0's leading norm and 1 everywhere else (R1).
void pf_res_norm(Context& cx, PrefillScratch& s, uint32_t s_prev,
                 const void* norm_w, const void* partials, void* resid,
                 void* x, uint32_t M) {
  const uint32_t g = PrefillScratch::kNormGroups;
  l0::Kernel& a = cx.kernel(kernels::pf_res_fold_variant(Qwen35::kHidden, s_prev, g),
                            "pf_res_fold", kWgFold);
  cx.launch(a, g, M, 1, {PtrArg(partials), PtrArg(resid),
                         PtrArg(s.norm_sumsq.ptr()), arg_val(M)});
  l0::Kernel& b = cx.kernel(kernels::pf_norm_finish_variant(Qwen35::kHidden, g, g),
                            "pf_norm_finish", kWgNorm);
  cx.launch(b, g, M, 1, {PtrArg(s.norm_sumsq.ptr()), PtrArg(resid),
                         PtrArg(norm_w), PtrArg(x), arg_val(M)});
}
}  // namespace
```

  Then the body, with the strides spelled at every hand-off (R2):

```cpp
// The four persistent l0::Mem& are passed in rather than reached through an
// Engine, so this function is unit-testable and holds no engine reference.
// step.h's declaration carries the same eleven parameters.
void step_chunk(Context& cx, PrefillScratch& s, const loader::LoadedModel& m,
                uint32_t max_len, void* ctrl, uint32_t pos, uint32_t C,
                l0::Mem& gdn_state_mem, l0::Mem& conv_ring_mem,
                l0::Mem& kv_k_mem, l0::Mem& kv_v_mem) {
  // embed: ids -> resid (bf16 [C][5120])
  {
    l0::Kernel& k = cx.kernel(kernels::pf_embed_gather_variant(),
                              "pf_embed_gather", kWgEmbed);
    cx.launch(k, 1, C, 1, {PtrArg(s.ids.ptr()), PtrArg(m.embed.ptr()),
                           PtrArg(s.resid.ptr()), arg_val(C)});
  }
  uint32_t gdn = 0, fa = 0;
  const size_t kv_stride =
      size_t(max_len) * Qwen35::kFaKvHeads * Qwen35::kFaHeadDim * 2;
  const size_t gdn_state_stride =
      size_t(Qwen35::kGdnVHeads) * Qwen35::kGdnHeadDim * Qwen35::kGdnHeadDim * 4;
  const size_t conv_ring_stride =
      size_t(PersistentBuffers::kConvRing) * 10240 * 2;
  for (const model::LayerDesc& L : Qwen35::layers()) {
    const uint32_t l = L.index;
    pf_res_norm(cx, s, l == 0 ? 0u : 1u,
                at(m.layer_small[l].norms, loader::kNormsOffInput),
                s.partials.ptr(), s.resid.ptr(), s.x.ptr(), C);   // x stride 5120
    if (L.kind == model::LayerKind::GDN) {
      pf_gemv(cx, m.linears.at({l, LinearId::QkvZ}), s.x.ptr(),
              s.partials.ptr(), C);                              // [C][16384] fp32
      {   // a||b, bf16 GEMV -> fp32 [C][128]
        const loader::DeviceWeight& w = m.linears.at({l, LinearId::AB});
        l0::Kernel& k = cx.kernel(kernels::pf_ab_proj_variant(), "pf_ab_proj",
                                  16 * 16);
        cx.launch(k, w.shape.N / 16, (C + 7) / 8, 1,
                  {PtrArg(w.mem.ptr()), PtrArg(s.x.ptr()),
                   PtrArg(s.ab_out.ptr()), arg_val(C)});
      }
      gdn_chunk(cx, s, pos, C, s.partials.as<float>(), s.ab_out.as<float>(),
                m.layer_small[l].gdn.ptr(),
                reinterpret_cast<float*>(at(gdn_state_mem, size_t(gdn) * gdn_state_stride)),
                reinterpret_cast<uint16_t*>(at(conv_ring_mem, size_t(gdn) * conv_ring_stride)),
                s.mixer_out.as<uint16_t>());                     // y stride 6144
      ++gdn;
      pf_gemv(cx, m.linears.at({l, LinearId::OutProj}), s.mixer_out.ptr(),
              s.partials.ptr(), C);                              // K 6144 == stride
    } else {
      pf_gemv(cx, m.linears.at({l, LinearId::Qkv}), s.x.ptr(),
              s.partials.ptr(), C);                              // [C][14336] fp32
      void* kk = at(kv_k_mem, size_t(fa) * kv_stride);
      void* vv = at(kv_v_mem, size_t(fa) * kv_stride);
      // R5: attn_part is linear in the sub-chunk, so <= kAttnC per pass.
      for (uint32_t sub = 0; sub < C; sub += PrefillScratch::kAttnC) {
        const uint32_t Cs = std::min(PrefillScratch::kAttnC, C - sub);
        cx.wait();                                               // Control is host-written
        Control* c = static_cast<Control*>(ctrl);
        c->pos = pos + sub;
        c->n_active = Cs;
        attn_prep_l1(cx, Cs,
                     reinterpret_cast<const float*>(at(s.partials, size_t(sub) * 14336 * 4)),
                     m.layer_small[l].gdn.as<float>(), m.rope.as<float>(), ctrl,
                     s.attn_q.as<float>(), s.attn_gate.as<float>(),
                     static_cast<uint16_t*>(kk), static_cast<uint16_t*>(vv));
        attn_l1(cx, Cs, max_len, ctrl, s.attn_q.as<float>(),
                static_cast<const uint16_t*>(kk), static_cast<const uint16_t*>(vv),
                s.attn_gate.as<float>(), s.attn_part.as<float>(),
                reinterpret_cast<uint16_t*>(at(s.mixer_out, size_t(sub) * 6144 * 2)));
      }
      ++fa;
      pf_gemv(cx, m.linears.at({l, LinearId::OProj}), s.mixer_out.ptr(),
              s.partials.ptr(), C);
    }
    // The MLP half, identical in both layer kinds.
    pf_res_norm(cx, s, 1u, at(m.layer_small[l].norms, loader::kNormsOffPost),
                s.partials.ptr(), s.resid.ptr(), s.x.ptr(), C);   // x stride 5120
    pf_gemv(cx, m.linears.at({l, LinearId::GateUp}), s.x.ptr(),
            s.partials.ptr(), C);                                // [C][34816] fp32
    {
      l0::Kernel& k = cx.kernel(kernels::pf_silu_mul_variant(), "pf_silu_mul", kWgSilu);
      cx.launch(k, (Qwen35::kIntermediate + 4095) / 4096, C, 1,
                {PtrArg(s.partials.ptr()), PtrArg(s.x.ptr()), arg_val(C)});
    }                                                            // x stride 17408
    pf_gemv(cx, m.linears.at({l, LinearId::Down}), s.x.ptr(), s.partials.ptr(), C);
  }
}
```

  The four `*_mem` parameters are `PersistentBuffers`' own members, passed by
  reference; `step.h`'s declaration in the Interfaces block above is updated to
  the same eleven parameters when this step is written. The per-layer slice
  strides (`kv_stride`, `gdn_state_stride`, `conv_ring_stride`) are re-derived
  here from `model::Qwen35` and `PersistentBuffers::kConvRing` rather than
  shared with `buffers.cc` - the same arrangement `capture.cc:105-116` uses,
  and for the same reason: a divergence is then a throw from the `require`s
  Step 5 adds, not a wrong KV slot at position 3000. Add those three `require`s
  (stride × layer count == allocation size) at the top of `step_chunk`,
  mirroring `capture.cc:198-201`.
- [ ] **Step 4: Write `step_head`** - the final norm at `M = 1` on the last
  row, `lm_head` at `M = 1`, and the two argmax stages. Row offsets, not a full
  pass, because only the last position's logits are wanted:

```cpp
void step_head(Context& cx, PrefillScratch& s, const loader::LoadedModel& m,
               void* ctrl, uint32_t last_row) {
  // The final norm, one row. sumsq[g*1 + 0] == sumsq[g], which is what the
  // M = 1 pair reads (Task 3's index rewrite).
  pf_res_norm(cx, s, 1u, m.final_norm.ptr(),
              at(s.partials, size_t(last_row) * Qwen35::kHidden * 4),
              at(s.resid, size_t(last_row) * Qwen35::kHidden * 2),
              at(s.x, size_t(last_row) * Qwen35::kHidden * 2), 1u);
  // lm_head: the EXISTING M = 1 decode binary, chosen off the loaded weight's
  // kind exactly as capture.cc:625-631 chooses it, writing straight into
  // `logits` (gemv.cl's out[(s*M+m)*N+n] at S = 1, M = 1 IS the [1][N] row
  // argmax_stage1 reads).
  const loader::DeviceWeight& lm = m.linears.at({loader::kTopLevel, LinearId::LmHead});
  const void* xrow = at(s.x, size_t(last_row) * Qwen35::kHidden * 2);
  if (lm.kind == model::WeightKind::Int4) {
    l0::Kernel& k = cx.kernel(kernels::gemv_variant(1, lm.shape.K, lm.shape.N,
                                                    lm.shape.S, lm.shape.layout),
                              "gemv", kWgGemv);
    cx.launch(k, lm.shape.N / 64, lm.shape.S, 1,
              {PtrArg(lm.mem.ptr()),
               PtrArg(lm.shape.layout == 0 ? lm.scales->ptr() : lm.mem.ptr()),
               PtrArg(xrow), PtrArg(s.logits.ptr())});
  } else {
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(lm.shape.N);
    l0::Kernel& k = cx.kernel(kernels::gemv_bf16_variant(1, lm.shape.K, lm.shape.N, t),
                              "gemv_bf16", t.cols * t.ksplit);
    cx.launch(k, lm.shape.N / t.cols, 1, 1,
              {PtrArg(lm.mem.ptr()), PtrArg(xrow), PtrArg(s.logits.ptr())});
  }
  // The sampler, byte for byte decode's: argmax_stage1_M1 then argmax_stage2,
  // which is the ONLY writer of pos and cur_token and therefore the last
  // launch of the step (argmax.cl:117-170). The caller has set
  // pos = base + L - 1 and n_active = 1, so stage 2 advances pos to base + L
  // and puts the first generated id in cur_token[0] -- indistinguishable from
  // what the last ingest-by-decode replay would have left (spec section 3.5).
  {
    l0::Kernel& k = cx.kernel(kernels::argmax_stage1_variant(1), "argmax_stage1", kWgArgmax);
    cx.launch(k, (Qwen35::kVocab + 1023) / 1024, 1, 1,
              {PtrArg(s.logits.ptr()), PtrArg(s.argmax_part.ptr())});
  }
  {
    l0::Kernel& k = cx.kernel(kernels::argmax_stage2_variant(), "argmax_stage2", kWgArgmax);
    cx.launch(k, 1, 1, 1, {PtrArg(ctrl), PtrArg(s.argmax_part.ptr())});
  }
}
```

- [ ] **Step 5: Write `Engine::prefill()`.** The loop, the validation, and the
  two `Control` writes - and **the `cx_->wait()` before every host write to
  Control**, which is Task 2's stated rule:

```cpp
void Engine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk) {
  if (ids.empty())
    throw std::runtime_error("runtime::Engine::prefill: no ids");
  if (chunk == 0) chunk = PrefillScratch::kC;
  if (chunk > PrefillScratch::kC)
    throw std::runtime_error("runtime::Engine::prefill: chunk " + std::to_string(chunk) +
                             " exceeds PrefillScratch::kC " +
                             std::to_string(PrefillScratch::kC));
  for (uint32_t id : ids)
    if (id >= model::Qwen35::kVocab)
      throw std::runtime_error("runtime::Engine::prefill: id " + std::to_string(id) +
                               " is outside the vocabulary (" +
                               std::to_string(model::Qwen35::kVocab) +
                               " rows) -- pf_embed_gather has no debug_flag channel, so"
                               " the host is the only bound");
  const uint32_t base = control_->pos;
  if (size_t(base) + ids.size() > size_t(buffers_.max_len))
    throw std::runtime_error(/* same message shape as replay()'s */);
  if (!pf_) {                                     // ruling R7
    pf_.reset(new PrefillScratch(ctx_, buffers_.max_len));
    cx_.reset(new prefill::Context(ctx_));
  }
  for (size_t off = 0; off < ids.size(); off += chunk) {
    const uint32_t C = uint32_t(std::min<size_t>(chunk, ids.size() - off));
    cx_->wait();                                  // before touching ids / Control
    std::memcpy(pf_->ids.ptr(), ids.data() + off, size_t(C) * 4);   // Host memory
    control_->pos = base + uint32_t(off);
    control_->n_active = C;
    prefill::step_chunk(*cx_, *pf_, model_, buffers_.max_len, control_,
                        base + uint32_t(off), C, persist_.gdn_state,
                        persist_.conv_ring, persist_.kv_k, persist_.kv_v);
    cx_->wait();                                  // the chunk's state has landed
  }
  // The tail: pos = base + L - 1 and n_active = 1 make argmax_stage2 leave
  // pos = base + L and cur_token[0] = the first generated id.
  const uint32_t last = uint32_t((ids.size() - 1) % chunk);
  control_->pos = base + uint32_t(ids.size()) - 1;
  control_->n_active = 1;
  prefill::step_head(*cx_, *pf_, model_, control_, last);
  cx_->wait();
}
```

  `Engine`'s new members: `std::unique_ptr<PrefillScratch> pf_;
  std::unique_ptr<prefill::Context> cx_;`, declared after `buffers_`.
  `prefill_launches()` returns `cx_ ? cx_->launches() : 0`.
- [ ] **Step 6: Run it, see it pass.**
  `tools/box.sh test prefill_smoke_test` - and if the launch count in Step 1
  case 2 differs from 1158, print both, re-derive the arithmetic from the walk
  and fix **the test's derivation comment**, not the number.
- [ ] **Step 7: Re-prove decode.** `tools/box.sh test
  'replay_determinism_test|engine_smoke_test|profile_capture_test'` plus the
  two golden-gate invocations from Task 1 Step 7 - `engine.h`/`engine.cc`
  changed, so the whole decode surface is re-checked. Paste.
- [ ] **Step 8: Commit** -
  `feat(runtime): Engine::prefill - chunked prefill and the Control handoff`

### Task 11: CLI - `--pp N` and `--pp-chunk C`

**Files:**
- Modify: `src/cli/b70_decode.cc` (usage, the parse loop, the mode checks, the
  bench path), `tests/CMakeLists.txt` (four new `b70_cli_reject` cases)

**Interfaces:**
- Consumes: `Engine::prefill` (Task 10), `kBenchPrompt`
  (`b70_decode.cc:57-61`), the existing `--bench` report shape
  (`b70_decode.cc:729-776`).
- Produces - the CLI contract, per interfaces.md with change request #3:

```
b70-decode <snapshot-or-repo> --bench [--pp N [--pp-chunk C]] [--depth D] [--tg T]

  --pp N          prefill N ids of the baked bench prompt (cycled) through
                  Engine::prefill instead of ingesting them one replay at a
                  time, and report DEVICE-SIDE pp. --bench only; mutually
                  exclusive with --depth (the prefilled ids ARE the depth).
  --pp-chunk C    positions per prefill chunk (default PrefillScratch::kC =
                  4096). --pp only. The spec section 6.2 multi-chunk gate runs at 1024.

stderr:  pp: N ids in X ms (Y t/s) -- device-side, loader excluded, first
         prefill launch to the first generated id in cur_token; chunk C
stdout:  the existing tg row, byte-identical, PLUS
         | b70-decode <sha> pp | <N> | <C> | <X> | <Y> |
```

- [ ] **Step 1: Failing test first** - four `b70_cli_reject` cases in
  `tests/CMakeLists.txt`, in the existing block's style (each asserts **exit
  1**, that the rejection landed **before** `l0::Context`, and the message
  substring, because "exit 1 before the device" is also what an unknown flag
  does):

```cmake
b70_cli_reject(cli_reject_pp_without_bench ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --pp 64
  EXPECT "--pp belongs to --bench")
b70_cli_reject(cli_reject_pp_and_depth ${B70_NO_MODEL} --bench --pp 64 --depth 64
  EXPECT "--pp and --depth are exclusive")
b70_cli_reject(cli_reject_pp_zero ${B70_NO_MODEL} --bench --pp 0
  EXPECT "--pp 0 would prefill nothing")
b70_cli_reject(cli_reject_pp_chunk_without_pp ${B70_NO_MODEL} --bench --pp-chunk 1024
  EXPECT "--pp-chunk belongs to --pp")
```

- [ ] **Step 2: Run them, see them fail.**
  `tools/box.sh test 'cli_reject_pp'` - all four fail (the flags are unknown,
  so the message substring does not match, which is exactly what the `EXPECT`
  argument exists to catch). Paste.
- [ ] **Step 3: Extend the parse loop and the mode checks.** Add `--pp` and
  `--pp-chunk` with `have_pp`/`have_pp_chunk` flags beside the existing
  `have_depth`/`have_tg`/`have_steps`/`have_repeats` set, and the four
  rejections above, placed with the others (`b70_decode.cc:634-662`) so they
  land before `l0::Context`. Extend `usage()` with the two lines. `--pp` sets
  the synthetic id count in place of `--depth`:
  `if (have_pp) { ids.resize(pp); for (i) ids[i] = kBenchPrompt[i % kBenchPromptLen]; }`
  and the `--depth + --tg <= --max-len` bound becomes `pp + tg <= max_len`
  when `--pp` is given.
- [ ] **Step 4: Extend the bench path.** Replace the `eng.ingest(ids)` call
  with a branch; the measured window is **the whole prefill call**, which ends
  with `argmax_stage2` having written `cur_token[0]` - that is precisely
  "first prefill launch to the first generated id" and it needs no extra
  instrumentation because `prefill()` returns only after `cx_->wait()`:

```cpp
  const auto t0 = std::chrono::steady_clock::now();
  if (have_pp) eng.prefill(ids, pp_chunk); else eng.ingest(ids);
  const double ingest_ms = /* unchanged */;
  if (have_pp) {
    const double tps = ids.empty() ? 0.0 : double(ids.size()) * 1000.0 / ingest_ms;
    std::fprintf(stderr,
                 "pp: %zu ids in %.1f ms (%.2f t/s) -- device-side, loader excluded,"
                 " first prefill launch to the first generated id in cur_token;"
                 " chunk %u, %zu launches\n",
                 ids.size(), ingest_ms, tps, pp_chunk, eng.prefill_launches());
  } else {
    /* the existing `ingest:` line, unchanged */
  }
```

  and, after the existing stdout `tg` row (which does **not** change),

```cpp
  if (have_pp)
    std::printf("| b70-decode %s pp | %zu | %u | %.1f | %.2f |\n", sha, ids.size(),
                pp_chunk, ingest_ms, double(ids.size()) * 1000.0 / ingest_ms);
```

- [ ] **Step 5: Run the rejections, see them pass.**
  `tools/box.sh test 'cli_reject'` - the whole block, so the four new cases
  and the fifteen existing ones.
- [ ] **Step 6: Drive it once end to end, small.** On the box, idle card:

```bash
tools/box.sh run 'ZE_AFFINITY_MASK=1 B70_GIT_SHA=$(git -C . rev-parse --short HEAD 2>/dev/null || echo dev) \
  ./build/src/cli/b70-decode "$HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64" \
  --bench --pp 256 --pp-chunk 64 --tg 8'
```

  Paste both lines of output. This is **not** a record row (spec §8: record
  rows need a provably idle box and medians of three) and the commit body says
  so; it is the first device-side `pp` number this project has and it is
  labelled **measured, one run, conditions stated**.
- [ ] **Step 7: Commit** -
  `feat(cli): b70-decode --pp / --pp-chunk - device-side prefill measurement`

### Task 12: The three prefill gates - golden, self-consistency, determinism

**Files:**
- Create: `tests/golden/golden_common.h` (extracted, see Step 1),
  `tests/prefill/prefill_gate_test.cc`,
  `tests/prefill/prefill_consistency_test.cc`,
  `tests/prefill/prefill_determinism_test.cc`
- Modify: `tests/golden/golden_gate_test.cc` (include the extracted header;
  **behaviour byte-identical**), `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `Engine::prefill` / `ingest` / `generate` / `reset`,
  `PersistentBuffers`, `runtime::Control`, the golden safetensors sets, and -
  reused verbatim, not reimplemented - `golden_gate_test.cc`'s tie-aware gate
  machinery: `GoldenDecision` (`:267-274`), `golden_decision()` (`:288-299`),
  `argmax_masked()` / `argmax_full()` (`:242-249`), `print_top5()`
  (`:301-312`), `Golden` (`:114-192`), `read_ids()` (`:314-326`),
  `exists()` (`:327`), `Metric` / `compare*` (`:195-237`).
- Produces: the three gate binaries, registered with the existing
  `B70_TEST_SNAPSHOT` convention and a new `B70_RTN_SNAPSHOT` beside it.

- [ ] **Step 1: Extract the golden machinery - a pure move, proven by a
  re-run.** Cut `tests/golden/golden_gate_test.cc:85-345`'s reusable half into
  `tests/golden/golden_common.h` under `namespace golden` - `Golden`,
  `Metric`, `compare`, `compare_bf16`, `compare_f32`, `argmax_masked`,
  `argmax_full`, `GoldenDecision`, `golden_decision`, `print_top5`,
  `read_ids`, `exists`, `kBar` - **carrying every comment verbatim**,
  especially `golden_decision`'s ruling paragraph (`:251-266`, `:276-287`) and
  `kBar`'s (`:94-107`): those comments are the ruling's record and must not be
  paraphrased in a move. `golden_gate_test.cc` then includes it and keeps
  `Verdict` and `main` unchanged.
  **Acceptance for this step alone:** re-run both golden gates (Task 1 Step
  7's two commands) and diff the output against the run pasted in Task 1's
  commit - the same determined/tie counts, the same cosines, the same order.
  A pure move that changes a number is not a pure move.
- [ ] **Step 2: Failing test first - `prefill_gate_test.cc`** (spec §6.1).
  Argv: `[1]` golden dir, `[2]` prompt dir, `[3]` snapshot. Structure it as
  `golden_gate_test`'s section 4 with two changes and nothing else:
  - ingestion is **one `prefill(ids)` call** instead of `T` × `ingest({id})`;
  - there is no per-layer tap (ruling R6), so sections 2 and 3 of
    `golden_gate_test` (the tap tables and the `gdn_state` cosines) are
    replaced by **one** `gdn_state` comparison after the prefill, per GDN
    layer, printed with `**LOW**` under `kBar` and **not gating** - the same
    2026-08-25 ruling `kBar`'s comment records.
  Then the gate itself, unchanged in semantics: 32 greedy ids as 32 ×
  `generate(1)` reading `buffers().logits` before each step; `golden_decision`
  per decision row; determined rows element-exact; undetermined rows
  set-membership; **teacher-forced after the first divergence of either kind,
  with `forced` set BEFORE the advance** (the ordering `golden_gate_test.cc:668-684`
  measured and pinned - copy that comment too). Prompts: `prose`, `code`,
  `cjk`. Print the census and the same summary table.
- [ ] **Step 3: Failing test first - `prefill_consistency_test.cc`**
  (spec §6.3). Per prompt, on one loaded model:
  1. `eng.reset(); eng.ingest(ids);` snapshot all five persistent buffers and
     `generate(64)` → `ids_decode`;
  2. `eng.reset(); eng.prefill(ids);` snapshot again and `generate(64)` →
     `ids_prefill`;
  3. **GATE:** `ids_prefill == ids_decode`, all 64, element-exact. On a
     mismatch, print the first differing position, both ids, and the logit-row
     cosine at that row (read `buffers().logits` before the diverging step in
     both walks - run the two walks interleaved one `generate(1)` at a time so
     that row is available), then fail.
  4. **DIAGNOSTICS, printed always, gating nothing** (spec §6.3: "tokens gate,
     tensors diagnose"): max and mean relative difference of `gdn_state`
     (per GDN layer and overall), `conv_ring`, the chunk's `kv_k`/`kv_v` rows
     `[0, T)`, and the last hidden (`buffers().resid` after the fence in the
     decode walk vs `PrefillScratch::resid`'s last row). Print them as a table
     so the band is greppable, and print the same table for `chunk = 64` and
     `chunk = 1024` runs so a chunk-width dependence is visible.
  5. Run every case on both `chunk = PrefillScratch::kC` (one chunk; these
     prompts are 38-61 ids) and `chunk = 16` (three or four chunks), so the
     multi-chunk carry is exercised on a prompt with a golden set.
  **This is the bar most at risk** (spec §11: "chunked-GDN rounding breaks
  token exactness on long prompts") and the risk is stated, not hidden: the
  chunked recurrence is algebraically equal and differently rounded, so 64
  identical greedy tokens is a real bar and not a formality. If it fails after
  the arithmetic has been checked against Task 8's bands, **stop and write the
  priced record** - first divergence position, the logit cosine there, the
  state band, and which of the ten launches the band localises to - and take
  it to the operator as a ruling request. Do not widen the bar in this plan.
- [ ] **Step 4: Failing test first - `prefill_determinism_test.cc`**
  (spec §6.4). `reset()`, `prefill(ids)`, snapshot the five persistent buffers
  and `cur_token`; `reset()`, `prefill(ids)` again, snapshot again; require
  **bitwise identical** on all five plus the token - `same_bytes()` from
  `replay_determinism_test.cc:99-111`, copied with its diff-reporting body.
  Run it at `chunk = 4096`, `1024` and `16` (three chunk widths, same result
  required at each width - not across widths, which Task 8 case 4 covers for
  GDN alone). Then, deliberately, the **run-C property**: a third `prefill`
  after a `reset()` but with `PrefillScratch` **not** re-zeroed must give the
  same bytes, which is the standing proof that no prefill kernel reads scratch
  it has not first written. State that and do not zero the scratch.
- [ ] **Step 5: Run all three, see them fail.**
  `tools/box.sh test 'prefill_gate|prefill_consistency|prefill_determinism'`.
  Paste.
- [ ] **Step 6: Register them** in `tests/CMakeLists.txt`, beside the
  `B70_TEST_SNAPSHOT` block (`tests/CMakeLists.txt:173-197`) whose comment
  explains why every checkpoint test passes its snapshot explicitly rather
  than depending on `refs/main`:

```cmake
# The int4-lm_head checkpoint (spec 1.6 section 5.1, tools/quantize_qwen38_rtn.sh).
# It never went through `hf download`, so it is a plain path and not a cache
# lookup; the same override rule as B70_TEST_SNAPSHOT applies.
set(B70_RTN_SNAPSHOT "$ENV{HOME}/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64"
  CACHE PATH "The int4-lm_head checkpoint the RTN-labelled gates load")

set(B70_PREFILL_KERNELS
  kernel_pf_embed_gather kernel_pf_res_fold_K5120_SP0_G20
  kernel_pf_res_fold_K5120_SP1_G20 kernel_pf_norm_finish_K5120_G20_W20
  kernel_pf_silu_mul kernel_pf_gated_head kernel_pf_attn_prep
  kernel_pf_gemv_K5120_N16384_L1 kernel_pf_gemv_K6144_N5120_L0
  kernel_pf_gemv_K5120_N34816_L0 kernel_pf_gemv_K17408_N5120_L0
  kernel_pf_gemv_K5120_N14336_L0 kernel_pf_ab_proj
  kernel_pf_gdn_conv kernel_pf_gdn_wy kernel_pf_gdn_scan
  kernel_attn_decode_M64_L16384_B${ATTN_BLOCK}
  kernel_attn_reduce_M64_L16384_B${ATTN_BLOCK}
  # the M = 1 lm_head + sampler tail is decode's, so the decode list's set too
  ${B70_DECODE_LIST_KERNELS})

foreach(t prefill_gate prefill_consistency prefill_determinism)
  add_executable(${t}_test prefill/${t}_test.cc)
  target_include_directories(${t}_test PRIVATE ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
  target_link_libraries(${t}_test PRIVATE b70_runtime b70_loader b70_l0 b70_model)
  add_dependencies(${t}_test ${B70_PREFILL_KERNELS})
endforeach()

# Vishva (bf16 lm_head) and RTN (int4 lm_head) -- one golden set per checkpoint
# (tools/oracle/README.md, "One golden set per checkpoint"), so two tests and
# never one that takes whichever directory is lying around.
add_test(NAME prefill_gate_test COMMAND prefill_gate_test
  ${CMAKE_SOURCE_DIR}/oracle-out ${CMAKE_SOURCE_DIR}/tests/golden/prompts
  ${B70_TEST_SNAPSHOT})
add_test(NAME prefill_gate_rtn_test COMMAND prefill_gate_test
  ${CMAKE_SOURCE_DIR}/oracle-out-rtn ${CMAKE_SOURCE_DIR}/tests/golden/prompts
  ${B70_RTN_SNAPSHOT})
add_test(NAME prefill_consistency_test COMMAND prefill_consistency_test
  ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT})
add_test(NAME prefill_determinism_test COMMAND prefill_determinism_test
  ${CMAKE_SOURCE_DIR}/tests/golden/prompts ${B70_TEST_SNAPSHOT})
set_tests_properties(prefill_gate_test prefill_gate_rtn_test PROPERTIES
  LABELS "checkpoint;golden;prefill" SKIP_RETURN_CODE 77 TIMEOUT 3600)
set_tests_properties(prefill_consistency_test prefill_determinism_test PROPERTIES
  LABELS "checkpoint;prefill" TIMEOUT 3600)
```

- [ ] **Step 7: Run them, see them pass.**
  `tools/box.sh test prefill` (the label picks all five), then
  `tools/box.sh test` for the whole suite.
- [ ] **Step 8: Record the bands.** Paste into the commit body: both
  `prefill_gate` runs' `TOTAL:` lines, the consistency test's diagnostic table
  at all three chunk widths, and the determinism test's `OK` line. These are
  spec §6.1/§6.3/§6.4's evidence.
- [ ] **Step 9: Commit** -
  `test(prefill): the three spec-2 section 6 gates -- golden, self-consistency, determinism`

### Task 13: The long-prompt oracle (`oracle-out-long`) and the multi-chunk gate

*Spec §6.2. One CPU oracle run on the box, in the reference container. The
runtime is **estimated** below and the run's own `/usr/bin/time -v` is the
measurement; budget two hours wall and run it detached.*

**Files:**
- Create: `tools/oracle/make_long_prompt.sh`,
  `tests/golden/prompts/long.txt` (generated by that script and committed),
  `tests/golden/prompts/long.ids` (produced on the box and committed)
- Modify: `tools/oracle/golden.sh` (a `PROMPTS` env override so the long
  prompt is one invocation, not a copy of the loop),
  `tools/oracle/README.md` (the new set, its commands, its measured numbers),
  `tests/prefill/prefill_gate_test.cc` (accept a prompt name list),
  `tests/CMakeLists.txt` (the multi-chunk gate registration)

**Interfaces:**
- Consumes: `tools/oracle/{tokenize,dump}.py`, `run_in_container.sh`
  (`ORACLE_SNAP`, `ORACLE_THREADS`), `golden.sh`'s `OUT_DIR`/`IDS_DIR`
  contract, `dump.py --max-prompt` (default 64, `dump.py:223`).
- Produces: `oracle-out-long/long.golden.safetensors` on the box (**not
  committed** - `.gitignore` already covers `oracle-out-*/`), a committed
  `long.ids` of **≥ 2048** ids, and a `prefill_gate_long_test` registered at
  `C = 1024`.

- [ ] **Step 1: Read how the existing goldens were made, then write the long
  prompt's generator.** `tools/oracle/README.md`, "Running it" and "One golden
  set per checkpoint"; `tools/oracle/golden.sh` in full. The three committed
  prompts are 42 / 61 / 38 ids, and `dump.py --max-prompt` caps at 64 by
  default. Do **not** invent 2048 tokens of prose: build the long prompt
  deterministically from the committed texts, so it is reproducible and
  reviewable.

```bash
#!/usr/bin/env bash
# tools/oracle/make_long_prompt.sh -- build tests/golden/prompts/long.txt from
# the three committed prompt texts, deterministically. Spec 2 section 6.2 needs one
# prompt of >= 2048 ids so that a C = 1024 gate crosses a chunk boundary in
# attention-over-cache AND in the GDN state carry. The three short prompts are
# 42 / 61 / 38 ids = 141; twenty repetitions of the trio, each block preceded
# by a numbered heading, is ~2900 ids (estimated: 20 x 141 plus 20 x ~4 heading
# ids), comfortably over the bar with margin for a tokenizer that merges
# differently across a heading. Step 3 measures the real count.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=tests/golden/prompts/long.txt
: > "$out"
for k in $(seq 1 20); do
  printf '## Section %d\n\n' "$k" >> "$out"
  for p in prose code cjk; do
    cat "tests/golden/prompts/$p.txt" >> "$out"
    printf '\n\n' >> "$out"
  done
done
wc -c "$out"
```

- [ ] **Step 2: Generate and commit `long.txt`.** Run the script on the Mac
  (it is `cat` and `printf`, no model), paste its `wc -c`, and commit the
  generated file with the script - the recipe under version control, the way
  `golden.sh` is committed rather than retyped.
- [ ] **Step 3: Tokenize on the box and commit `long.ids`.** Seconds, tokenizer
  only:

```bash
tools/box.sh sync
ssh user@box
cd ~/b70-inference-server
tools/oracle/run_in_container.sh 'python3 tools/oracle/tokenize.py "$SNAP" \
  encode tests/golden/prompts/long.txt > /ws/tests/golden/prompts/long.ids'
wc -w tests/golden/prompts/long.ids     # the id count -- must be >= 2048
```

  Paste the count. If it is under 2048, raise the repetition count in the
  script, regenerate, re-tokenize - do not hand-edit `long.ids`. Then pull it
  back and commit it: `tools/box.sh pull tests/golden/prompts/long.ids`.
  Both checkpoints ship the identical tokenizer (README, "One golden set per
  checkpoint"), so one `.ids` file serves both sets.
- [ ] **Step 4: Pre-register the run's cost, then run it.** Write these down
  **before** starting, so the measurement has something to be compared against:
  - **load + dequant: 118 s (measured**, README's table, three prompts within
    0.2 s of each other**)** - fixed, independent of `T`.
  - **prefill forward: estimated 900 s.** Derived from the measured 14.8 s at
    42 ids by scaling the per-position term linearly in `T` (the MLP is 70% of
    the FLOPs and is linear) and adding 10% for the quadratic attention term
    (attention is ~3% of prefill FLOPs at depth 4096, spec §3.3): 14.8 ×
    (2400/42) × 1.1 ≈ 930 s. The fixed part of the 14.8 s is unknown, so this
    is an upper-ish bound.
  - **32 greedy: estimated 250 s**, from the measured ~5.8 s/token at 42 ids
    plus the deeper KV walk.
  - **total: estimated 20-40 min for ONE prompt**, plus ~18 s of container
    start (measured). **Reconciliation with spec §6.2**, which said
    "hours-class - estimated from the 18-minute 32-id runs": that estimate
    scaled the whole 16m48s three-prompt run by prompt length; scaling the
    three terms separately (fixed load, linear prefill, 32 decodes) gives the
    20-40 min above. Both are estimates and neither is measured yet; **the
    run's own `/usr/bin/time -v` is the number that goes in the record**, and
    whichever it lands on, the reconciliation sentence stays.
  - **peak RSS: estimated ~70 GiB** of the box's 121 GB. Measured 61.4 GiB at
    42 ids; the additions at `T ≈ 2400` are the 192 hook activations
    (192 × 2400 × 5120 × 2 B = **4.72 GB derived**), `logits`
    ((2400+32) × 248320 × 4 B = **2.42 GB derived**) and the eager attention
    scores (24 × 2400² × 4 B = **553 MB derived**, transient). The box must be
    otherwise idle for this one; **never give the container a memory limit**
    (README's last line).
  - **output file: estimated ~7.3 GB** (the three terms above plus 151 MB of
    GDN states and 4 MB of conv states). Check free space first:
    `df -h ~/b70-inference-server`.

```bash
# on the box, from the repo root, detached so the run outlives the ssh
mkdir -p oracle-out-long
OUT_DIR=oracle-out-long PROMPTS=long \
ORACLE_SNAP=$HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64 \
ORACLE_THREADS=28 \
  setsid nohup tools/oracle/golden.sh > oracle-out-long/golden.log 2>&1 </dev/null &
tail -f oracle-out-long/golden.log
OUT_DIR=oracle-out-long PROMPTS=long \
ORACLE_SNAP=$HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64 \
  tools/oracle/check.sh
```

  `golden.sh` needs one edit for this: `for p in ${PROMPTS:-prose code cjk}; do`
  in place of the hardcoded triple, and `--max-prompt 4096` added to the
  `dump.py` line (its default 64 would abort at id 65). Both are one-line
  changes; make them in this step and say in the header comment that
  `--max-prompt` is now the ceiling `PrefillScratch::kC` sets, not 64.
  **The RTN checkpoint is the one to run** - it is the gate checkpoint
  (spec §7 records on RTN) and one CPU oracle run is what §6.2 budgets. If the
  operator wants the Vishva long set too, it is a second run of the same
  command with `ORACLE_SNAP` unset and `OUT_DIR=oracle-out-long-vishva`; price
  it in the memo, do not run it here.
- [ ] **Step 5: Record the measured numbers** in `tools/oracle/README.md`,
  in the same table shape as the existing three rows (ids, wall, load+dequant,
  prefill, 32 greedy, peak RSS, output bytes, manifest count), plus the
  `check.sh` block quoted verbatim and the decoded continuation. Add a
  `oracle-out-long/` row to the "One golden set per checkpoint" table with the
  RTN snapshot path. Every number labelled **measured** with the box's
  conditions; the four estimates from Step 4 kept beside them with the
  reconciliation sentence.
- [ ] **Step 6: Register the multi-chunk gate.** `prefill_gate_test.cc` takes
  a fourth argv: a comma-separated prompt-name list (default
  `prose,code,cjk`), and a fifth: the chunk width (default 0 = `kC`). Then:

```cmake
# Spec 2 section 6.2: the multi-chunk golden gate. One oracle prompt of >= 2048 ids
# (oracle-out-long, RTN checkpoint) gated at C = 1024, so every chunk boundary
# is crossed in attention-over-cache and in the GDN state carry.
#
# **EXPECTED-LIMITED UNDER L1.** The L1 attention route caps a sub-chunk at
# PrefillScratch::kAttnC = 64 (ruling R5), so at C = 1024 an FA layer runs 16
# sub-chunk passes; that is CORRECT but it is also 3 x 16 launches per FA layer
# and the test is slow. What may still fail here and not in the short gates is
# the depth: 2400 positions of chunked GDN carry against 141. The gate is
# REGISTERED HERE and, if it is short under L1, its failure is recorded and
# taken to the operator rather than papered over -- plan 6d's real attention and
# plan 6c's GEMM are what this row is finally graded under (spec section 5: "this gate
# may only pass fully after plan 6d").
add_test(NAME prefill_gate_long_test COMMAND prefill_gate_test
  ${CMAKE_SOURCE_DIR}/oracle-out-long ${CMAKE_SOURCE_DIR}/tests/golden/prompts
  ${B70_RTN_SNAPSHOT} long 1024)
set_tests_properties(prefill_gate_long_test PROPERTIES
  LABELS "checkpoint;golden;prefill;long" SKIP_RETURN_CODE 77 TIMEOUT 7200)
```

- [ ] **Step 7: Run it.** `tools/box.sh test prefill_gate_long_test`. Paste
  the `TOTAL:` line. On a pass, the multi-chunk bar is met at L1 and that is a
  result worth stating plainly. On a short row: paste the first divergence, the
  logit cosine there, and the `gdn_state` band from
  `prefill_consistency_test` at `chunk = 1024`, mark the row
  **expected-limited under L1's `C ≤ 64` attention**, and stop at the record.
- [ ] **Step 8: Commit** -
  `test(prefill): the >= 2048-id long-prompt oracle and the C=1024 multi-chunk gate`

### Task 14: Docs - mechanisms, the prefill step's anatomy, the execution model

**Files:**
- Modify: `docs/12-kernels.md` (a new prefill section), `docs/15-step-anatomy.md`
  (a new prefill-anatomy section), `docs/04-architecture.md` ("Execution model"
  and "Kernel inventory for phase 1"), `docs/07-open-questions.md` (the items
  this plan closes and the one it opens)

**Interfaces:**
- Consumes: every measured band and count this plan produced (Tasks 6-8's
  numerics bands, Task 10's launch count, Task 11's device-side `pp`, Task 13's
  oracle numbers).
- Produces: the mechanism record spec §9 requires in the same commit as the
  code - which for this plan means **this task's edits are split across the
  earlier commits wherever a kernel lands**, and this task is the assembly
  and the cross-check. Land the per-kernel paragraphs with their kernels;
  land the anatomy and the architecture edits here.

- [ ] **Step 1: `docs/12-kernels.md` - one new top-level section**,
  `## Prefill kernels (spec 2, Stage 1 L1)`, placed after `## attn` and before
  the closing material, with a subsection per kernel in the file's established
  shape (what it computes, work assignment and why, rounding discipline,
  what the test asserts, what was rejected, measured):
  - `### pf_gemv_int4_M` - **labelled "L1 temporary, retired by plan 6c (L2)"
    in its first sentence.** The `MT = 8` register tile, the `ceil(M/8)` grid,
    ruling R1 (why `S = 1`, with the 4.56 TB arithmetic), the masked writeback,
    and the measured `memcmp` identity against the five `S = 1` decode
    binaries.
  - `### pf_ab_proj` - the `{16, 16}` tiling inherited from `gemv_bf16`'s
    lever L2 and why the tree order is copied rather than simplified (bit
    identity against the binary `capture.cc` binds, which is what keeps
    `ab_out` - and therefore `g` and `beta` - unmoved from decode's).
  - `### pf_res_fold / pf_norm_finish / pf_silu_mul / pf_gated_head /
    pf_embed_gather` - the runtime-`M` rewrite, the `sumsq[g*M + m]` stride,
    `pf_embed_gather`'s ids buffer and the loss of the `debug_flag` channel
    (with the host-side bound that replaces it), and the `+0.0f`-padding trick
    the M = 1 identity tests use, including the `-0.0` caveat.
  - `### pf_attn_prep and the M = 64 route` - ruling R5 in full: what is new
    (one kernel), what is reused unmodified (two binaries), the
    `attn_part` = 405,798,912 B arithmetic, the `C ≤ 64` limit, and the
    sentence that plan 6d retires all three.
  - `### gdn_chunk - the WY-representation chunked gated delta rule` - the
    algorithm block from this plan copied verbatim (it is the mechanism), the
    six FLA stage references, the ten launches with their grids, the tile
    mapping and both reduction trees inherited from `gdn_step` **and why that
    inheritance is the point**, the explicit seed/writeback that replaces
    `gdn_step.cl:176-178`'s `M + 3 ≤ RING`, ruling R8's two deviations from
    FLA with their line references, the one reassociation relative to decode
    (the batched `S` update), and **the measured band** from Task 8 Step 6 -
    all three comparisons, labelled measured, with box conditions.
- [ ] **Step 2: `docs/15-step-anatomy.md` - a new section**,
  `## Spec 2 L1 - the prefill step's anatomy (skeleton)`. It carries **no
  per-kernel timings**: L1 is a correctness stage and the profiler is not
  wired to the dynamic path (say so, and say that pricing it is L2's, when the
  GEMM swap makes the numbers mean something). What it does carry:
  - **the launch arithmetic, derived from Task 10's walk**, so the anatomy has
    a structure to hang timings on later:
    - per GDN layer: 2 (norm) + 1 (qkv‖z) + 1 (a‖b) + 10 (`gdn_chunk`) +
      1 (out_proj) + 5 (MLP: 2 norm + gate‖up + silu + down) = **20**
    - per FA layer: 2 + 1 (qkv) + 3·⌈C/64⌉ (attention) + 1 (o_proj) + 5 =
      **9 + 3·⌈C/64⌉**
    - boundary: 1 (embed) + 2 (final norm) + 1 (lm_head) + 2 (argmax) = **6**
    - totals: **C = 64 → 48·20 + 16·12 + 6 = 1158**;
      **C = 1024 → 960 + 16·57 + 6 = 1878**;
      **C = 4096 → 960 + 16·201 + 6 = 4182**. Note what the table shows: the
      FA family's launch count is the only term that grows with `C`, and it
      grows because of ruling R5 alone - at plan 6d's real attention the
      `3·⌈C/64⌉` collapses and `C = 4096` becomes 960 + 16·9 + 6 = **1110**,
      i.e. **fewer launches than a 64-position chunk pays today**. That is the
      single clearest statement of what L1's temporary attention costs.
  - the per-chunk byte table from Task 1 Step 1, and the device total against
    the card's 32,656 MB: weights (13.673 GB RTN / 15.540 GB Vishva,
    both **measured**, `LoadReport::read_per_token`'s two values) + persistent
    1.240 GB + decode scratch 0.069 GB + prefill scratch 1.962 GB =
    **16.94 GB / 18.81 GB (derived)** - spec §3.6's "fits" claim, now with the
    prefill scratch actually enumerated.
  - the device-side `pp` from Task 11 Step 6, labelled **measured, one run,
    not a record row**, beside the 121 s / 29.5 ms-per-id ingest baseline
    (bench log `2a7df0b`, RTN) and vLLM's **1973 t/s pp4096** with both labels
    attached (device-side vs HTTP-inclusive, chunk width, checkpoint), exactly
    as spec §2 requires the row to be quoted.
- [ ] **Step 3: `docs/04-architecture.md`.** "Execution model" gains a
  subsection: decode is unchanged (one captured list, replayed); **prefill is
  dynamic dispatch on one in-order asynchronous L0 immediate command list**
  with arguments resolved per launch, `Context::wait()` as the only
  synchronisation, and the host writing `Control::{pos, n_active}` only
  between chunks and between attention sub-chunks - with the rule that a host
  write to `Control` is always preceded by a `wait()`, and why (the list is
  asynchronous). Note that the SYCL half of `Context` is plan 6a's and that
  L1 uses none of it. "Kernel inventory for phase 1" gains the `pf_*` rows and
  the two `M = 64` attention rows, each marked with its retiring plan (6c for
  `pf_gemv_int4_M`, 6d for the attention three). Update the paragraph at
  `:150-178` that says the prefill path "is not static-linkable until its
  kernels are also prebuilt with `ocloc`" - they now are, for L1, and the
  remaining non-AOT term is plan 6a's SYCL.
- [ ] **Step 4: `docs/07-open-questions.md`.** Close: "does the `M`
  dimension reach a prefill width" (it does not - R1's arithmetic and
  explorer-1 §3 are the answer, and the prefill path is a second kernel family
  instead). Open, with a price: **the profiler does not see the dynamic path**
  - `--profile` builds two `CapturedStep`s and the prefill walk is not one, so
  per-kernel attribution of a prefill step needs either per-launch events on
  the immediate list (cheap: `Context` already has a launch counter; add an
  optional `ProfileEvents`-shaped pool) or a capture of the walk (impossible
  while arguments are per-launch). Recommend the former, price it at one
  task, and leave it to L2 where the numbers matter.
- [ ] **Step 5: Cross-check the docs against the code.** Grep for every number
  written in Steps 1-4 and confirm each is either in a test's pinned assert or
  in a pasted measurement in this plan's commit bodies. No number appears in
  docs that does not appear in a test or a log.
- [ ] **Step 6: Commit** -
  `docs(kernels,anatomy,architecture): the prefill path -- mechanisms, anatomy skeleton, execution model`

---

## Plan self-review (2026-09-04, at authoring)

**Spec coverage.**

| spec | where |
|---|---|
| §3.4 GDN chunked delta rule (ours; batched conv1d with explicit seed/writeback; the intra-chunk block; the state scan; "FLOPs negligible, this is a correctness problem") | the algorithm block + Tasks 6, 7, 8; the CuTe first-candidate branch is Task 9, gated on plan 6a's P5 |
| §3.5 everything else widened (RMSNorm, SiLU·mul, a/b, embed gather; `lm_head` on the last position only through the existing S = 1 route; the sampler writing `out_token`/`cur_token` and advancing `pos` exactly as `argmax_stage2` does; the handoff indistinguishable from ingest-by-decode) | Tasks 3, 4, 5 (kernels); Task 10 Step 4 (`step_head`: the existing `lm_head` binary, `argmax_stage1_M1` + `argmax_stage2`, `pos = base + L - 1` / `n_active = 1`); Task 12 Step 3 (the bar that proves "indistinguishable") |
| §3.6 execution model (dynamic dispatch, one in-order context, ordering by events or a queue wait; chunk width the widest that fits, 4096 default, 2048 fallback, multi-chunk from the start; the buffer split into a persistent group and two scratch groups; two `CapturedStep`s over one buffer set is already proven) | Task 2 (`Context`, with Step 1's discovery of the in-order flag and the event-chain fallback); Task 1 (`kC = 4096`, the byte table, the split, `DecodeBuffers` as a view); Task 10 (the chunk loop; `--pp-chunk` in Task 11 makes 2048 and 1024 reachable) |
| §5 L1 (buffer split, runtime-`M` variants, batched conv1d + chunked GDN, widened `attn_prep`, the temporary `attn_decode` route at `C ≤ 64`, the `Control` handoff, `Engine::prefill()`, `b70-decode --pp`; gate = §6 on the short goldens; nothing about speed) | Tasks 1-12 in order; Task 11 Step 6 is the only speed number and it is labelled not-a-record |
| §6.1 golden gate after prefill, tie-aware, unchanged semantics, both checkpoints | Task 12 Steps 1, 2, 6 (`prefill_gate_test` + `prefill_gate_rtn_test`, reusing `golden_decision()` by extraction, not reimplementation) |
| §6.2 multi-chunk golden gate, a new ≥ 2048-id oracle prompt in `oracle-out-long`, gated at `C = 1024` | Task 13 (the generator, the tokenize step, the pre-registered cost, the run, the registration, and the expected-limited marking §5 anticipates) |
| §6.3 self-consistency control: 64 generated tokens identical; state diagnostics recorded as max/mean rel-diff; diagnostic not gate | Task 12 Step 3, at three chunk widths, with the failure path (priced record + ruling request, no bar widening) written out |
| §6.4 determinism: twice from reset, bitwise-identical state and tokens; no fp atomics; no split-K atomics | Task 12 Step 4 (three chunk widths + the run-C scratch property); Task 8 case 6; Task 9 Step 6 greps Intel's header for `atomic` |
| §6.5 decode untouched: gate rows re-measured at the final sha | Global Constraints + Task 1 Step 7 + Tasks 3, 5, 10 Step 7 (every task that touches decode's files re-runs it). **The final-sha re-measurement of 32.22 / 29.33 t/s is plan 6e's gate task, not this plan's** - flagged here so it is not lost |
| §6.6 flags, forbidden options, 774/19 | Global Constraints, verbatim; every kernel goes through `add_ocloc_kernel`, which fatals on `-cl-denorms-are-zero` and passes the correctly-rounded pair by default |

**Placeholder scan.** No TBD, no TODO, no "similar to". Task 9 is conditional
but concrete (the entry-point signature quoted from the header, the two
relayouts sized, the `negA` → `A_log` inversion named, the torch dependency
priced by a counted grep before any edit, the byte arithmetic for both
outcomes). `<sha>`, `<ids>`, `<C>`, `<X>`, `<Y>` in commit messages and output
lines are measurement outputs. Task 6 Step 2 case 5 and Task 7 Step 1 case 3
say "pin whichever it is, with the number" - that is a measurement to be taken,
with the bar and the fallback both stated, not an unresolved decision. Task 13
Step 3 has a loop-back if the id count lands under 2048.

**Type consistency vs interfaces.md.** Checked name by name:
- `runtime::prefill::Context`, `KernelArg` - Task 2, matching interfaces.md's
  fields; `Context`'s constructor and `sycl()` are **consumed from plan 6a's
  T0 and not redefined here** (Task 2's header comment says so explicitly, and
  this plan declares only the `l0::Context&` constructor and `wait()`).
- `gemm_bf16`, `dequant_to_bf16`, `GemmDims`, `loader::Linear` - plan 6a/6c's;
  untouched here. `PrefillScratch::dequant` is allocated at interfaces.md's
  exact size (356,515,840 B) and left unused, so 6c's landing moves no bytes.
- `gdn_chunk` - Task 8, with change requests #1 (`qkvz_partials` fp32) and #2
  (`ab_out` fp32 [C][128]) applied; `layer` is dropped from the signature in
  favour of the caller passing the layer's own `gdn_state`/`conv_ring` slices
  and `small` base, which is how `capture.cc` binds them
  (`capture.cc:492-493`) - **that is a third deviation from interfaces.md's
  declaration and it is in the change-request list's spirit; if the controller
  prefers `layer`, `gdn.cc` derives the three pointers itself from the
  `LoadedModel` and the signature keeps `layer`.** Flagged rather than assumed.
- `attn_prep_chunk` / `attn_chunk` - **not implemented here**; L1's stand-ins
  are `attn_prep_l1` / `attn_l1` in `runtime::prefill` (ruling R5), so plan 6d
  implements interfaces.md's declarations without a collision.
- `Engine::prefill(const std::vector<uint32_t>&, uint32_t chunk = 0)` -
  Task 10, exactly interfaces.md's signature and semantics, including "default
  chunk = PrefillScratch::kC" (interfaces.md writes `PrefillBuffers::kC`; the
  struct it declares two lines earlier is `PrefillScratch`, so `PrefillScratch`
  is used and the typo is reported).
- `PersistentBuffers` / `DecodeScratch` / `PrefillScratch` /
  `DecodeBuffers`-as-a-view with `kC = 4096` - Task 1, matching interfaces.md's
  refactor block; the field list is longer than interfaces.md's parenthetical
  because `gdn_chunk`'s nine kernels need scratch, and every added line carries
  its byte arithmetic.
- Test names - `tests/prefill/gdn_chunk_test.cc`, `prefill_gate_test.cc`,
  `prefill_consistency_test.cc`, `prefill_determinism_test.cc`: all
  interfaces.md's. `gemm_test.cc` / `dequant_test.cc` / `attn_chunk_test.cc`
  are 6a/6c/6d's and are not created here. The extra files
  (`context_test.cc`, `pf_prep_test.cc`, `pf_gemv_test.cc`, `pf_attn_test.cc`,
  `gdn_conv_test.cc`, `gdn_wy_test.cc`, `prefill_smoke_test.cc`) are this
  plan's own unit cycles and collide with nothing.
- Numbers quoted the interfaces.md way: vLLM **1973 t/s pp4096**
  (HTTP-inclusive); today **121 s / 29.5 ms per id** at `2a7df0b` on RTN; the
  decode rows to re-verify **32.22 / 29.33**; the dequant scratch
  **356,515,840 B**; the KV / `gdn_state` / `conv_ring` layouts unchanged.

**Known tension, ruled here.** Ruling R5 knowingly compiles a per-`M`
attention binary on a path whose interface contract forbids per-`M` binaries.
The alternative - writing a runtime-`M` transcription of `attn_decode` and
`attn_reduce` for a route plan 6d deletes - is more new code, more numerics
surface, and a second place for the online-softmax merge order to drift from
`attn_ref.h`. The exception is recorded in the interface (change request #4),
in `attn_l1.h`'s header, in `src/kernels/CMakeLists.txt`'s comment beside the
four rows, and in docs/12, each naming plan 6d as the retirement. `capture.cc`
binds none of the four binaries, so the decode list's 19 modules are unchanged
and the invariant that actually matters is untouched.
