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
#include <cstdio>
#include "check.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "runtime/buffers.h"
#include "runtime/control.h"

int main() {
  CHECK_EQ(sizeof(runtime::Control), size_t{128});

  l0::Context ctx(0);
  std::printf("device: %s\n", ctx.name().c_str());
  runtime::DecodeBuffers b(ctx, 16384);
  std::printf("persistent %zu B, scratch %zu B\n", b.persistent_bytes(), b.scratch_bytes());

  CHECK_EQ(b.max_len, 16384u);
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

  std::puts("buffers_test OK");
  return 0;
}
