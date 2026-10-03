// The decode step's allocations are baked into a command list that is
// recorded once and replayed for every token, so their sizes are a contract,
// not an implementation detail: a buffer that is one element short is a
// silent out-of-bounds write on token 2, not a crash at construction. This
// test pins both byte totals to numbers computed by hand from the model
// constants (below), so any change to a size shows up here as a diff with an
// arithmetic explanation rather than as a wrong logit somewhere.
//
// Persistent state @ max_len = 16384 (48 GDN layers, 16 FA layers):
//   control      128 B (two cache lines)                          =           128
//   gdn_state    48 x 48 v-heads x 128 k x 128 v x 4 B (fp32)     =   150,994,944
//   conv_ring    48 x 16 slots x 10240 channels x 2 B (bf16)      =    15,728,640
//   kv_k         16 x 16384 pos x 4 kv-heads x 256 x 2 B (bf16)   =   536,870,912
//   kv_v         same                                             =   536,870,912
//                                                           total = 1,240,465,536  (1240.47 MB)
//
// Per-step scratch, M = 8:
//   resid        8 x 5120 x 2 B                                   =        81,920
//   x            8 x 17408 x 2 B                                  =       278,528
//   partials      8 (S_max) x 8 x 34816 (2 x intermediate) x 4 B  =     8,912,896
//   ab_out       8 x 128 (a||b padded N) x 4 B                    =         4,096
//   norm_sumsq   20 (kNormGroups) x 8 x 4 B                       =           640
//   gdn_o        8 x 48 x 128 x 4 B                               =       196,608
//   attn_q       8 x 24 q-heads x 256 x 4 B                       =       196,608
//   attn_gate    same                                             =       196,608
//   attn_part    24 x (16384/64 = 256 blocks) x 8 x 258 x 4 B     =    50,724,864
//   attn_out     8 x 24 x 256 x 2 B                               =        98,304
//   logits       8 x 248320 x 4 B                                 =     7,946,240
//   argmax_part  8 x ceil(248320/1024) = 243 x 2 x 4 B            =        15,552
//                                                           total =    68,652,864  (68.65 MB)
//
// `attn_part` grew 4x with spec 1.5's lever L5: `DecodeBuffers::kAttnBlock`
// went 256 -> 64 KV positions per `attn_decode` work-group, which shortens each
// work-group's serial walk of a block (the critical path docs/15 §2 measured)
// and quadruples the block count this buffer is strided by. It is the ONE home
// for that number on the host - `attn_part`'s size, `attn_decode`'s grid in
// capture.cc, and the `_B64` half of both compiled variant names all read it,
// so a host that disagrees with the kernel's `ATTN_BLOCK` names a binary that
// does not exist and throws at capture instead of striding `attn_part` wrongly.
//
// Task 4's lower split widths save **8,912,896 B** in `partials`, taking the
// whole per-step scratch from 77.57 MB to 68.65 MB. `attn_part` is now the
// largest line in this table, and it is worth stating what it is NOT: it is not
// persistent (1240.47 MB of KV and GDN state dwarf it) and it is not close to
// any limit on a device holding a 15.5 GB checkpoint. What it buys is
// measured - the attn family 6.058 -> 3.839 ms/token, docs/15 §L5 - and it is
// the reason the sweep stopped at 64: `kAttnBlock` 32 would take this buffer to
// 101 MB for 0.070 ms/token that the step's own total does not show.
//
// `norm_sumsq` is the twelfth field and the only one added since plan 3: spec
// 1.5's lever L1 split `prep_res_norm` into `prep_res_fold`, which writes one
// fp32 sum-of-squares per (work-group, token), and `prep_norm_finish`, which
// folds the 20 of them into the rms. `DecodeBuffers::kNormGroups` = 20 is the
// one home for that count - it is also stage A's grid and half of both
// compiled variant names - and 20 x 8 x 4 = 640 B moves the scratch total from
// plan 3's 39,521,472 to 39,522,112. That is +0.0016%, which is why the
// "39.52 MB" summary is unchanged and the byte figure is not.
//
// NOTE: plan 3's prose says scratch is "≈ 44 MB"; the eleven sizes it gave per
// field add to 39,521,472 B, and its own itemisation (17.8 + 12.7 + 7.9 =
// 38.4 for the three big ones, 1.1 for the other eight) agrees with 39.52.
// The "44" is a slip in the summary line, not a missing buffer. Every
// per-field MB figure in the plan matches the table above exactly - the plan
// simply predates `norm_sumsq`.
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include "check.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "model/model_desc.h"
#include "runtime/buffers.h"
#include "runtime/control.h"

// Prefill scratch @ kC = 2048, max_len = 16384 - ruling A13 (2048, not 4096;
// every figure RECOMPUTED from model::Qwen35, not scaled) and ruling A14 (the
// M = 64 attention route is retired, so plan 6b's attn_q / attn_gate /
// attn_part are gone and plan 6d-composed's S/P scratch is here instead, sized
// once so L3 does not resize this struct when it lands). The derivation and
// the three cross-checks are docs/prefill-l1-preregistration-2026-09-05.md §1.
//
// Spec 2.1 §3.3 (this file's S1): the bf16 dequant scratch (356,515,840 B,
// [5120][34816]) is no longer a fixed member of this table -- it is
// `dequant_buffer()`, allocated lazily on first use, same as the new
// `slab_buffer()` (17408 x 1024 x 2 B = 35,651,584 B, the L0 backend's
// expansion). `bytes()` below is therefore 356,515,840 B smaller than plan
// 6c/6d's total; `lazy_bytes()` reports whichever of the two a session has
// actually built.
//
//   ids          2048 x 4 B                                    =         8,192
//   resid        2048 x 5120 x 2 B                             =    20,971,520
//   x            2048 x 17408 x 2 B  (largest GEMV K: down)     =    71,303,168
//   partials     2048 x 34816 x 4 B  (S=1 max-N: gate||up, R1)  =   285,212,672
//   ab_out       2048 x 128 x 4 B                               =     1,048,576
//   norm_sumsq   20 x 2048 x 4 B                                =       163,840
//   gdn_o        2048 x 48 x 128 x 4 B                          =    50,331,648
//   mixer_out    2048 x 6144 x 2 B   (out_proj/o_proj input, R2) =   25,165,824
//   logits       1 x 248320 x 4 B    (last position only)        =       993,280
//   argmax_part  1 x 243 x 2 x 4 B                              =         1,944
//   gdn_xb       2048 x 10240 x 2 B                             =    41,943,040
//   gdn_seed     3 x 10240 x 2 B                                =        61,440
//   gdn_g        2048 x 48 x 4 B                                =       393,216
//   gdn_beta     same                                           =       393,216
//   gdn_A        32 x 48 x 64 x 64 x 4 B                        =    25,165,824
//   gdn_A2       same                                           =    25,165,824
//   gdn_w        2048 x 48 x 128 x 2 B                          =    25,165,824
//   gdn_u        same                                           =    25,165,824
//   pf_q         2048 x 24 x 256 x 2 B  (bf16 - ruling A9)      =    25,165,824
//   pf_attn      same                                           =    25,165,824
//   pf_o         24 x 2048 x 256 x 4 B                          =    50,331,648
//   pf_rowsum    24 x 2048 x 4 B                                =       196,608
//                                                        total =   699,514,776
//
// Spec 6 (plan 6b): `pf_s` / `pf_p` are LAZY, `pf_s_buffer()` / `pf_p_buffer()`, built on
// the first composed `attn_chunk`; the default flash path never builds them. They were
//   pf_s         6 x 2048 x max_len x 4 B (kSHeads = one GQA grp) = 805,306,368 @16384
//   pf_p         6 x 2048 x max_len x 2 B                       =   402,653,184 @16384
// and the old total 1,907,474,328 - 1,207,959,552 = 699,514,776. Nothing left in `bytes()`
// depends on max_len, so the same 699,514,776 holds at 131072, where the two would have
// been 9,663,676,416 B (derived: 6 x 2048 x 131072 x 6).
//
// Cross-checks, each of which fails loudly if a row above is wrong:
//   * the 6b-owned rows alone (everything but the six pf_* ones, and now also
//     without the lazy dequant scratch) = 598,654,872;
//   * the six pf_* rows = 1,308,819,456, plan 6d-composed's "total added" line
//     exactly (pf_kt excluded: transB measured native and bitwise identical,
//     commit 792e1dd, so the transpose fallback is not built); the four still
//     eager (pf_q, pf_attn, pf_o, pf_rowsum) = 100,859,904 since spec 6;
//   * 598,654,872 + 100,859,904 = 699,514,776.
static void check_prefill_scratch(l0::Context& ctx) {
  // Spec 6: at max_len 131072 no score scratch exists until a composed call asks for it.
  {
    runtime::PrefillScratch big(ctx, 131072, model::qwen38());
    std::printf("prefill scratch @131072: %zu B, lazy %zu B, pf_s %zu B\n", big.bytes(),
                big.lazy_bytes(), big.pf_s_bytes());
    CHECK_EQ(big.bytes(), size_t{699514776});
    CHECK_EQ(big.lazy_bytes(), size_t{0});
    CHECK_EQ(big.pf_s_bytes(), size_t{0});
  }
  runtime::PrefillScratch pf(ctx, 16384, model::qwen38());
  std::printf("prefill scratch %zu B (%.3f GB)\n", pf.bytes(), pf.bytes() / 1e9);
  CHECK_EQ(pf.bytes(), size_t{699514776});
  CHECK_EQ(pf.lazy_bytes(), size_t{0});
  CHECK_EQ(pf.pf_s_bytes(), size_t{0});
  // The composed path's two, on first use: 6 x 2048 x 16384 x 4 B and x 2 B.
  CHECK_EQ(pf.pf_s_buffer().size(), size_t{805306368});
  CHECK_EQ(pf.pf_s_bytes(), size_t{805306368});
  CHECK_EQ(pf.lazy_bytes(), size_t{805306368});
  CHECK_EQ(pf.pf_p_buffer().size(), size_t{402653184});
  CHECK_EQ(&pf.pf_s_buffer(), &pf.pf_s_buffer());   // built once
  CHECK_EQ(pf.lazy_bytes(), size_t{805306368 + 402653184});
  CHECK_EQ(pf.bytes(), size_t{699514776});          // lazy buffers are not in bytes()
  CHECK_EQ(pf.dequant_buffer().size(), size_t{356515840});
  // 17408 x 1024 x 2 B (derived: qwen38().intermediate * 1024 * kBf16, the formula in PrefillScratch::slab_buffer()).
  CHECK_EQ(pf.slab_buffer().size(), size_t{35651584});
  CHECK_EQ(pf.lazy_bytes(), size_t{805306368 + 402653184 + 356515840 + 35651584});
  CHECK_EQ(runtime::PrefillScratch::kC, 2048u);
  CHECK_EQ(runtime::PrefillScratch::kGdnChunk, 64u);
  CHECK_EQ(runtime::PrefillScratch::kSHeads, 6u);
  CHECK_EQ(pf.max_len, 16384u);
  // The six composed-attention rows, as one group, are 6d's own figure.
  CHECK_EQ(pf.pf_q.size() + pf.pf_attn.size() + pf.pf_s_buffer().size() +
               pf.pf_p_buffer().size() + pf.pf_o.size() + pf.pf_rowsum.size(),
           size_t{1308819456});
  // `ids` is the one host-resident field: the host writes C ids per chunk.
  CHECK(pf.ids.kind() == l0::MemKind::Host);
  CHECK(pf.resid.kind() == l0::MemKind::Device);
}

int main() {
  CHECK_EQ(sizeof(runtime::Control), size_t{128});

  l0::Context ctx(0);
  std::printf("device: %s\n", ctx.name().c_str());

  // The split must not move decode's numbers by a byte, and the view must
  // alias the groups rather than copy them.
  {
    runtime::PersistentBuffers p(ctx, 16384, model::qwen38());
    runtime::DecodeScratch s(ctx, 16384, model::qwen38());
    CHECK_EQ(p.bytes(), size_t{1240465536});
    CHECK_EQ(s.bytes(), size_t{68652864});
    runtime::DecodeBuffers view(p, s);
    CHECK_EQ(view.persistent_bytes(), p.bytes());
    CHECK_EQ(view.scratch_bytes(), s.bytes());
    CHECK_EQ(view.control.ptr(), p.control.ptr());
    CHECK_EQ(view.gdn_state.ptr(), p.gdn_state.ptr());
    CHECK_EQ(view.kv_v.ptr(), p.kv_v.ptr());
    CHECK_EQ(view.attn_part.ptr(), s.attn_part.ptr());
    CHECK_EQ(view.argmax_part.ptr(), s.argmax_part.ptr());
    CHECK_EQ(view.max_len, 16384u);
    check_prefill_scratch(ctx);
  }

  // Spec 6 (plan 6b Task 3): 128k context. The KV cache is bf16
  // 16 FA layers x 131072 x 4 kv heads x 256 x 2 B, x 2 for K and V = 8,589,934,592 B, and
  // attn_part is 24 x (131072 / 64 = 2048 blocks) x 8 x 258 x 4 B = 405,798,912 B.
  {
    runtime::PersistentBuffers p(ctx, 131072, model::qwen38());
    runtime::DecodeScratch s(ctx, 131072, model::qwen38());
    std::printf("max_len 131072: kv %zu B, attn_part %zu B, persistent %zu B, scratch %zu B\n",
                p.kv_k.size() + p.kv_v.size(), s.attn_part.size(), p.bytes(), s.bytes());
    CHECK_EQ(p.kv_k.size() + p.kv_v.size(), size_t{16} * 131072 * 4 * 256 * 2 * 2);
    CHECK_EQ(p.kv_k.size() + p.kv_v.size(), size_t{8589934592});
    CHECK_EQ(s.attn_part.size(), size_t{24} * 2048 * 8 * 258 * 4);
    CHECK_EQ(s.attn_part.size(), size_t{405798912});
    // Spec 10 (plan 10b): v2's partials are [24][kAttnV2Blocks = 32][M][258] fp32 at
    // any max_len - 792,576 B per row, 3,170,304 B at M = 4 - inside v1's allocation.
    CHECK_EQ(runtime::DecodeScratch::kAttnV2Blocks, 32u);
    CHECK(size_t{24} * runtime::DecodeScratch::kAttnV2Blocks * runtime::DecodeScratch::kM * 258 * 4 <=
          s.attn_part.size());
    runtime::DecodeBuffers view(p, s);
    CHECK_EQ(view.max_len, 131072u);
  }

  runtime::DecodeBuffers b(ctx, 16384, model::qwen38());
  std::printf("persistent %zu B, scratch %zu B\n", b.persistent_bytes(), b.scratch_bytes());

  CHECK_EQ(b.max_len, 16384u);
  CHECK(size_t{24} * runtime::DecodeScratch::kAttnV2Blocks * runtime::DecodeScratch::kM * 258 * 4 <=
        b.attn_part.size());
  CHECK_EQ(b.persistent_bytes(), size_t{1240465536});
  CHECK_EQ(b.scratch_bytes(), size_t{68652864});

  // The control block is shared memory: the host reads and writes it directly,
  // and the constructor left it zeroed - including the padding, which the
  // kernels index as a flat uint array.
  const uint32_t* words = b.control.as<uint32_t>();
  for (size_t i = 0; i < sizeof(runtime::Control) / sizeof(uint32_t); ++i) CHECK_EQ(words[i], 0u);

  runtime::Control* c = b.control.as<runtime::Control>();
  c->pos = 1234;
  c->n_active = 1;
  c->cur_token[0] = 151643;
  c->cur_token[7] = 9;
  c->out_token[0] = 42;
  c->out_token[7] = 7;
  c->debug_flag = 5;
  const runtime::Control* rc = b.control.as<runtime::Control>();
  CHECK_EQ(rc->pos, 1234u);
  CHECK_EQ(rc->n_active, 1u);
  CHECK_EQ(rc->cur_token[0], 151643u);
  CHECK_EQ(rc->cur_token[7], 9u);
  CHECK_EQ(rc->out_token[0], 42u);
  CHECK_EQ(rc->out_token[7], 7u);
  CHECK_EQ(rc->debug_flag, 5u);

  *c = runtime::Control{};  // re-zero, the state the engine starts a run in
  for (size_t i = 0; i < sizeof(runtime::Control) / sizeof(uint32_t); ++i) CHECK_EQ(words[i], 0u);

  // The persistent device state is zero-filled at construction too: a decode
  // run starts from an empty GDN state, an empty conv ring and an empty KV
  // cache, and nothing else ever writes those before token 0 does.
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const size_t probe = 4096;
  l0::Mem host(ctx, l0::MemKind::Host, probe);
  const l0::Mem* state[] = {&b.gdn_state, &b.conv_ring, &b.kv_k, &b.kv_v};
  for (const l0::Mem* m : state) {
    for (uint32_t half = 0; half < 2; ++half) {
      // Head and tail of each buffer: a fill that stopped early shows up here.
      const size_t off = half == 0 ? 0 : m->size() - probe;
      imm.copy(host.ptr(), static_cast<const uint8_t*>(m->ptr()) + off, probe);
      const uint32_t* w = host.as<uint32_t>();
      for (size_t i = 0; i < probe / sizeof(uint32_t); ++i) CHECK_EQ(w[i], 0u);
    }
  }

  // Spec 8 (plan 8b): the MTP head's buffers, constructed only when the model
  // carries the head (Engine), so none of the sizes above moved.
  //   hctl       128 B (a second Control)                          =           128
  //   gdn_spec   3 slots x 48 x 48 x 128 x 128 x 4 B               =   452,984,832
  //   kv_k/kv_v  2 x 16384 x 4 x 256 x 2 B                         =    67,108,864
  //   hh         (8 + 1) x 5120 x 2 B                              =        92,160
  //   dh         5120 x 2 B                                        =        10,240
  //   logits     3 x 248320 x 4 B                                  =     2,979,840
  //                                                          total =   523,176,064
  {
    runtime::MtpBuffers mb(ctx, 16384, model::qwen38());
    CHECK_EQ(mb.gdn_spec.size(), size_t{3} * 150994944);
    CHECK_EQ(mb.kv_k.size(), size_t{33554432});
    CHECK_EQ(mb.kv_v.size(), size_t{33554432});
    CHECK_EQ(mb.hh.size(), size_t{92160});
    CHECK_EQ(mb.logits.size(), size_t{2979840});
    CHECK_EQ(mb.bytes(), size_t{523176064});
    const uint32_t* hw = mb.hctl.as<uint32_t>();
    for (size_t i = 0; i < sizeof(runtime::Control) / sizeof(uint32_t); ++i) CHECK_EQ(hw[i], 0u);
    CHECK_EQ(runtime::MtpBuffers::kSlots, 4u);
    // The live-slot index sits in what was Control's padding: the persistent sizes
    // above are what they were.
    CHECK_EQ(offsetof(runtime::Control, gdn_live), size_t{76});
  }

  std::puts("buffers_test OK");
  return 0;
}
