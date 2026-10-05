#include "runtime/prefill/gdn.h"

#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernels/prefill/pf_kernels.h"
#include "loader/small_layout.h"
#include "model/qwen35.h"
#include "runtime/prefill/profile.h"

// The ten launches of one GDN layer's chunk, in order. Every one takes `C` as a
// RUNTIME argument, so one binary set serves every chunk width `--pp-chunk` can
// ask for (interfaces.md, "Layout conventions").
//
// **Ordering is the in-order immediate list and nothing else.** Launch 2 reads
// the ring slots launch 1 lifted; launch 3 rewrites `xb` in place after launch
// 2 wrote it; launch 6 overwrites `A` with `T` and launch 7 reads that `T`;
// launch 9 reads launches 7 and 8's outputs and is the only writer of
// `gdn_state`. No events, no fences: `src/sycl/context.cc` creates the list
// with ZE_COMMAND_QUEUE_FLAG_IN_ORDER and `tests/prefill/context_test.cc`'s
// 1024-launch chain is the empirical check on that guarantee.
//
// **Note for plan 6c (L2).** Launches 5, 7, 8 and the two contractions inside 9
// are 64x128x128-shaped per head and are plain vector code here. 6c MAY route
// 5, 7 and 8 through `gemm_bf16` - spec §3.4's "goes through the §3.2 interface
// where the shape suits DPAS, plain vector code where it does not". **Launch 6,
// the triangular solve, stays vector code**: it is inherently sequential in `i`
// and has no DPAS shape. Any such reroute changes the reduction order and
// therefore the band `tests/prefill/gdn_chunk_test.cc` records, so it is a
// re-measure, not a refactor.
namespace runtime::prefill {
namespace {
// Spec 15b: the conv channels and head counts are the descriptor's
// (PrefillScratch::desc()): conv rows 10240 on Qwen3.8 (40 work-groups of 256),
// 16 k-heads, 48 v-heads.
constexpr uint32_t kConvWg = 256;                      // pf_gdn_conv's channels per work-group
constexpr uint32_t kStateColChunks = 4;                // gdn_step.cl's grid.y

// Ruling A27: `pf_gdn_conv`'s position range is blocked, and this is the only
// place the block count is decided. The kernel derives its own block WIDTH from
// `get_num_groups(1)` rather than sharing this literal, so the two cannot
// disagree about coverage; what this number buys is the grid. 128 positions per
// block puts C = 2048 at 16 blocks = 640 work-groups = 10,240 threads (5 waves
// of the 2048-slot machine, ~640 KB of loads in flight against the ~295 KB
// Little's law asks for at 590 GB/s) while the three-position halo each block
// re-reads stays at 2.3% of the block's work.
constexpr uint32_t kConvBlock = 128;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::prefill::gdn_chunk: " + what);
}

}  // namespace

// This intentionally resolves once per process.  Captured replay may retain
// the first kernel selection, so changing getenv between engine calls is not a
// supported configuration operation; compare selectors in separate processes.
static const char* gdn_scan_entry() {
  static const char* const entry = [] {
    // **The default is the SPLIT-BF16 scan since 2026-09-23**, promoted on the
    // evidence recorded in docs/prefill-gdn-scan-split-fix-2026-09-23.md and in
    // the parity program's §12: `prefill_gate_l0_test` 93/93 with prose
    // 0.999912369, code 0.999712545 and cjk 0.999901750 all over the > 0.999
    // bar, determinism / consistency / replay green, the vector entry unchanged
    // at 0.999903435 / 0.999189068 / 0.999903208, and every one of those runs
    // carrying its own dispatch proof ("gdn scan entry LAUNCHED: ...") read back
    // from the pointer `gdn_chunk` handed the launch.
    //
    // This is the ONE default in the prefill walk whose arithmetic is not
    // bitwise equal to what it replaced. `vn` inside the A2 term is still a
    // single BF16 operand -- measured harmless on three prompts of one
    // checkpoint, not proven in general. `vector` keeps the old kernel.
    //
    // The history below is kept because it is the reason this default carries a
    // dispatch proof at all. The split entry was promoted once before, on
    // 2026-09-22, and reverted the same hour because it FAILED
    // `prefill_gate_l0_test`: 92/93 determined rows,
    // the code prompt's L60 `gdn_state` cosine 0.996344994 against a > 0.999 bar
    // (`$HOME/split-default-suite.log`, reproduced to nine digits on device 0 in
    // `$HOME/split-dev0.log`). The cause was found on 2026-09-23 and fixed --
    // `A2` was the last single-BF16 DPAS operand and it feeds `o` directly, so
    // it now carries hi/lo limbs like `S` and `D` (pf_gdn_scan.cl entry (3), and
    // docs/prefill-gdn-scan-split-fix-2026-09-23.md for the 2x2 that attributed
    // it). The fixed entry passes the gate at 93/93 with every cosine over the
    // bar, and it is 2.33x faster than the vector scan on the `gdn_scan` profile
    // row -- but it is still approximate arithmetic behind an opt-in selector,
    // and promoting it is a separate decision with its own evidence. The next
    // person to try it re-runs the gates AND proves the dispatch (see gdn.h).
    //
    // **One claim in the reverted note was wrong and is corrected here.** It
    // also blamed "`gdn_chunk_test`'s pre-registered band ... rel L2 2.7395e-03,
    // max rel 2.9650e-02". Those two numbers are `attn_chunk_test`'s standing
    // band MISS against the L1-engine pre-registration, not `gdn_chunk_test`'s
    // and not the scan's: the identical pair is printed by
    // `$HOME/spec21-suite.log` (2026-09-18) and `$HOME/suite-igc2415.log`
    // (2026-09-19), both under the vector default, and attention runs no GDN
    // kernel. `gdn_chunk_test` prints no rel L2 at all.
    const char* const v = std::getenv("B70_PREFILL_GDN_SCAN");
    if (!v || !*v || std::string(v) == "dpas_split") return "pf_gdn_scan_dpas_split";
    if (std::string(v) == "vector") return "pf_gdn_scan";
    throw std::runtime_error("runtime::prefill::gdn_chunk: B70_PREFILL_GDN_SCAN must be "
                             "unset, 'vector', or 'dpas_split' (got '" +
                             std::string(v) + "')");
  }();
  return entry;
}

// The entry the last scan launch was built with; see gdn.h for why it exists.
// Written by `gdn_chunk` below, read by the gate/determinism/consistency/replay
// tests so a selector run carries its own dispatch evidence.
static const char* g_scan_launched = nullptr;

const char* gdn_scan_entry_name() { return gdn_scan_entry(); }
const char* gdn_scan_launched_entry() { return g_scan_launched; }

// Stage S4's selector, resolved exactly as `gdn_scan_entry()` is and for the
// same reason: a recorded command list retains the kernel handle it was built
// with, so changing the environment between engine calls is not a supported
// configuration operation. Compare selectors in separate processes.
//
// Both entries are **bit-identical by construction** - the register entry keeps
// `A` immutable in SLM and gives `T` its own array, which deletes the two
// per-row barriers without moving a single rounding point. Identical bytes are
// why the dispatch test in `tests/prefill/gdn_chunk_test.cc` observes
// `KernelCache::kernels()` rather than the output.
static const char* gdn_solve_entry() {
  static const char* const entry = [] {
    const char* const v = std::getenv("B70_PREFILL_GDN_SOLVE");
    if (!v || !*v || std::string(v) == "vector") return "pf_gdn_solve";
    if (std::string(v) == "register") return "pf_gdn_solve_register";
    throw std::runtime_error("runtime::prefill::gdn_chunk: B70_PREFILL_GDN_SOLVE must be "
                             "unset, 'vector', or 'register' (got '" +
                             std::string(v) + "')");
  }();
  return entry;
}

void gdn_chunk(Context& cx, KernelCache& kc, PrefillScratch& s, uint32_t pos, uint32_t C,
               const float* qkvz_partials, const float* ab_out, float* gdn_state,
               uint16_t* conv_ring, const void* small, uint16_t* y) {
  require(C > 0, "C must be > 0");
  require(C <= PrefillScratch::kC,
          "C = " + std::to_string(C) + " exceeds PrefillScratch::kC = " +
              std::to_string(PrefillScratch::kC) + " - the scratch is sized for kC");
  // Validate/cache the process selector before reading or launching any GDN
  // buffer; an unsupported value therefore cannot mutate state or append work.
  const char* const scan_entry = gdn_scan_entry();
  const char* const solve_entry = gdn_solve_entry();

  const uint32_t nch = (C + PrefillScratch::kGdnChunk - 1) / PrefillScratch::kGdnChunk;
  const model::ModelDesc& d = s.desc();
  const uint32_t heads = d.gdn_v_heads;                 // 48 on Qwen3.8
  const uint32_t k_heads = d.gdn_k_heads;               // 16
  require(d.gdn_conv_dim() % kConvWg == 0, "the conv channels are not whole work-groups");
  const uint32_t conv_groups = d.gdn_conv_dim() / kConvWg;   // 40 on Qwen3.8
  const void* gated_w =
      static_cast<const uint8_t*>(small) + d.small_layout().gdn_off_gated_norm;

  void* p_xb = s.gdn_xb.ptr();
  void* p_seed = s.gdn_seed.ptr();
  void* p_g = s.gdn_g.ptr();
  void* p_beta = s.gdn_beta.ptr();
  void* p_A = s.gdn_A.ptr();
  void* p_A2 = s.gdn_A2.ptr();
  void* p_w = s.gdn_w.ptr();
  void* p_u = s.gdn_u.ptr();
  void* p_o = s.gdn_o.ptr();

  const std::string conv = kernels::pf_gdn_conv_variant(k_heads, heads);
  const std::string wy = kernels::pf_gdn_wy_variant(k_heads, heads);
  const std::string scan = kernels::pf_gdn_scan_variant(k_heads, heads);

  //  1 - lift the ring's three older slots (pos-3..pos-1) into a flat seed.
  cx.launch(kc(conv, "pf_gdn_seed"), conv_groups, 1, 1,
            {PtrArg(conv_ring), PtrArg(p_seed), arg_val(pos)});
  profile_wait(cx, Phase::kGdnSeed);
  //  2 - the batched conv1d + SiLU over the chunk, and the ring writeback.
  //      Grid.y is A27's position blocking; grid.x and the work-group are
  //      unchanged.
  const uint32_t conv_blocks = (C + kConvBlock - 1) / kConvBlock;
  cx.launch(kc(conv, "pf_gdn_conv"), conv_groups, conv_blocks, 1,
            {PtrArg(qkvz_partials), PtrArg(p_seed), PtrArg(small), PtrArg(p_xb),
             PtrArg(conv_ring), arg_val(pos), arg_val(C)});
  profile_wait(cx, Phase::kGdnConv);
  //  3 - l2norm q and k in place.
  cx.launch(kc(conv, "pf_gdn_l2norm"), 2 * k_heads, C, 1, {PtrArg(p_xb), arg_val(C)});
  profile_wait(cx, Phase::kGdnL2);
  //  4 - head scalars and the intra-chunk cumulative gate.
  cx.launch(kc(conv, "pf_gdn_gate"), heads, nch, 1,
            {PtrArg(ab_out), PtrArg(small), PtrArg(p_g), PtrArg(p_beta), arg_val(C)});
  profile_wait(cx, Phase::kGdnGate);
  //  5 - A = beta_i (k_i . k_j) exp(gc_i - gc_j), i > j. Grid.z is A28's
  //      quadrant split, as step 8's.
  cx.launch(kc(wy, "pf_gdn_A"), heads, nch, 4,
            {PtrArg(p_xb), PtrArg(p_g), PtrArg(p_beta), PtrArg(p_A), arg_val(C)});
  profile_wait(cx, Phase::kGdnA);
  //  6 - T = (I + A)^-1, IN PLACE over A. **`(I + A)`, not `(I - A)`**: FLA
  //      negates A at staging (solve_tril.py:82), which `pf_gdn_wy.cl`'s entry
  //      (3) derives at length and `gdn_wy_test`'s case 2 checks as an
  //      identity. This comment said `(I - A)` until stage S4.
  //      `B70_PREFILL_GDN_SOLVE=register` routes this one launch to the
  //      separate-A/T entry; the two are bit-identical and the default is
  //      `pf_gdn_solve`.
  cx.launch(kc(wy, solve_entry), heads, nch, 1, {PtrArg(p_A), arg_val(C)});
  profile_wait(cx, Phase::kGdnSolve);
  //  7 - vb/kb, then u = T vb and w = T kb. Grid.z is A27's column split: a
  //      work-group owns 64 of the 128 output columns, which is what lets the
  //      operands be staged fp32 in the same SLM budget class.
  cx.launch(kc(wy, "pf_gdn_wu"), heads, nch, 2,
            {PtrArg(p_xb), PtrArg(p_A), PtrArg(p_g), PtrArg(p_beta), PtrArg(p_w), PtrArg(p_u),
             arg_val(C)});
  profile_wait(cx, Phase::kGdnWu);
  //  8 - A2 = (q_i . k_j) exp(gc_i - gc_j), j <= i. NOTE the diagonal.
  //      Grid.z is A28's quadrant split: a work-group owns a 32 x 32 block of
  //      the 64 x 64 tile, which is what lets both operands be staged fp32 in
  //      32,768 B = four resident work-groups per Xe-core.
  cx.launch(kc(wy, "pf_gdn_A2"), heads, nch, 4,
            {PtrArg(p_xb), PtrArg(p_g), PtrArg(p_A2), arg_val(C)});
  profile_wait(cx, Phase::kGdnA2);
  //  9 - the sequential chunk-to-chunk state scan; the only writer of gdn_state.
  cx.launch(kc(scan, scan_entry), heads, kStateColChunks, 1,
            {PtrArg(p_xb), PtrArg(p_w), PtrArg(p_u), PtrArg(p_A2), PtrArg(p_g),
             PtrArg(gdn_state), PtrArg(p_o), arg_val(C)});
  g_scan_launched = scan_entry;          // the dispatch proof, after the append
  profile_wait(cx, Phase::kGdnScan);
  // 10 - the gated head. Ruling R3: it belongs to the mixer, not to the caller.
  cx.launch(kc.get(kernels::pf_gated_head_variant(k_heads, heads), "pf_gated_head"), heads, C, 1,
            {PtrArg(qkvz_partials), PtrArg(p_o), PtrArg(gated_w), PtrArg(y), arg_val(C)});
  profile_wait(cx, Phase::kGdnHead);
}

}  // namespace runtime::prefill
