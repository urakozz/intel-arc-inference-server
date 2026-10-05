// Spec 6 §10 (max_len auto): the memory planner, on the host. No device, no
// checkpoint: the buffer sizes are runtime/buffer_sizes.h's functions - the ones the
// allocations are made from - and the weights and the device total come in as
// numbers. tests/runtime/memory_plan_box_test.cc holds the plan against the card.
//
// Three things are pinned here:
//   1. the sizes reproduce buffers_test's hand-computed table byte for byte (the
//      device test pins the same numbers against real allocations);
//   2. the plan reproduces spec 6's measured 131072 line (§8.4) wherever the formula
//      covers it - KV, decode state, prefill scratch and int8 exactly, the model as
//      the measured load total with its RoPE table re-sized;
//   3. max_len_that_fits's contract: a multiple of 256, at most the cap, fits with the
//      reserve, and one more quantum does not.
// It also prints the derived auto lengths the README quotes.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "check.h"
#include "model/model_desc.h"
#include "runtime/buffer_sizes.h"
#include "runtime/memory_plan.h"

namespace {
using runtime::PrefillBackend;

// The measured Qwen3.8 load (docs/13-loader.md's printout, bf16 lm_head, max_len
// 16384): total 18,086,971,392 B of which the RoPE table is 4,194,304 (16384 x 256).
constexpr size_t kQwenBf16Weights = 18086971392ull - 4194304ull;            // 18,082,777,088
// The int8 head (spec 9) in place of the bf16 one: 5120 x 248320 int8 + 248320 fp32
// scales (loader::lm_head_int8_bytes) for 2,542,796,800 B of bf16. Derived.
constexpr size_t kLmBf16 = 2542796800ull, kLmInt8 = 1271398400ull + 993280ull;
constexpr size_t kQwenInt8Weights = kQwenBf16Weights - kLmBf16 + kLmInt8;  // 16,812,371,968
// Agnes with the int8 head, derived from its safetensors headers (load_agnes_test pins
// the first two): int4 14,816,378,880 + scales 926,023,680 + a||b bf16 54 x 983,040 +
// a||b pad 54 x 327,680 + small (54 x 205,440 + 18 x 43,008 + 20,480 final norm) +
// embed 2,542,796,800 + the int8 head.
constexpr size_t kAgnesInt8Weights = 14816378880ull + 926023680ull + 54ull * 983040 +
                                     54ull * 327680 + (54ull * 205440 + 18ull * 43008 + 20480) +
                                     2542796800ull + kLmInt8;               // 19,640,258,304
// The MTP head's device bytes (loader.cc load_mtp): 424,673,280 bf16 weight elements
// (fc, qkv, o, gate||up, down at mtp_intermediate 17408) + five fp32 norms + the FA
// block. Derived; Qwen3.8 and Agnes alike.
constexpr size_t kMtpWeights = 424673280ull * 2 + 5ull * 5120 * 4 + 2048;  // 849,451,008
// The device total memory_line() printed in spec 6 §8.4: 32.530 GB (rounded there,
// so +-0.5 MB here).
constexpr size_t kDevice = 32530000000ull;
constexpr size_t kReserve = size_t(runtime::kDefaultReserveGb * 1e9);
constexpr uint32_t kTrained = 262144;   // max_position_embeddings, Qwen3.8 and Agnes

void check_sizes_qwen38() {
  const model::ModelDesc& q = model::qwen38();
  // buffers_test's tables at max_len 16384, every field.
  const runtime::PersistentSizes p = runtime::PersistentDims::sizes(16384, q);
  CHECK_EQ(p.control, size_t{128});
  CHECK_EQ(p.gdn_state, size_t{150994944});
  CHECK_EQ(p.conv_ring, size_t{15728640});
  CHECK_EQ(p.kv_k, size_t{536870912});
  CHECK_EQ(p.kv_v, size_t{536870912});
  CHECK_EQ(p.total(), size_t{1240465536});
  // v1's attn_part strides every 64-position block of the cache (B70_DECODE_ATTN=v1).
  const runtime::DecodeScratchSizes d =
      runtime::DecodeScratchDims::sizes(16384, q, runtime::DecodeAttn::V1);
  CHECK_EQ(d.attn_part, size_t{50724864});
  CHECK_EQ(d.partials, size_t{8912896});
  CHECK_EQ(d.logits, size_t{7946240});
  CHECK_EQ(d.total(), size_t{68652864});
  // 128k (spec 6): KV 8,589,934,592, attn_part 24 x 2048 x 8 x 258 x 4.
  CHECK_EQ(runtime::PersistentDims::sizes(131072, q).kv_k * 2, size_t{8589934592ull});
  CHECK_EQ(runtime::DecodeScratchDims::sizes(131072, q, runtime::DecodeAttn::V1).attn_part,
           size_t{405798912});
  // v2 (the default): [24][32][8][258] fp32 whatever max_len is, 6,340,608 B.
  for (uint32_t L : {4096u, 16384u, 131072u, 262144u})
    CHECK_EQ(runtime::DecodeScratchDims::sizes(L, q, runtime::DecodeAttn::V2).attn_part,
             size_t{24} * 32 * 8 * 258 * 4);
  CHECK_EQ(runtime::DecodeScratchDims::sizes(16384, q, runtime::DecodeAttn::V2).total(),
           size_t{68652864 - 50724864 + 6340608});
  // The default follows B70_DECODE_ATTN, as capture does.
  unsetenv("B70_DECODE_ATTN");
  CHECK_EQ(runtime::DecodeScratchDims::sizes(131072, q).attn_part, size_t{6340608});
  const runtime::PrefillScratchSizes s = runtime::PrefillScratchDims::sizes(16384, q);
  CHECK_EQ(s.eager(), size_t{699514776});
  CHECK_EQ(runtime::PrefillScratchDims::sizes(131072, q).eager(), size_t{699514776});
  CHECK_EQ(s.dequant, size_t{356515840});
  CHECK_EQ(s.slab, size_t{35651584});
  CHECK_EQ(s.pf_s, size_t{805306368});
  CHECK_EQ(s.pf_p, size_t{402653184});
  CHECK_EQ(s.pf_q + s.pf_attn + s.pf_s + s.pf_p + s.pf_o + s.pf_rowsum, size_t{1308819456});
  const runtime::MtpSizes m = runtime::MtpDims::sizes(16384, q);
  CHECK_EQ(m.gdn_spec, size_t{3} * 150994944);
  CHECK_EQ(m.kv_k, size_t{33554432});
  CHECK_EQ(m.total(), size_t{523176064});
  // (kC + 1) x 5120 x 2 B.
  CHECK_EQ(runtime::mtp_prefill_hidden_bytes(q), size_t{2049} * 5120 * 2);
  // Int8State: xq 2048 x 17408 + xs 2048 x 4 + w8 (17408 / 4) x 1024 x 4, and the
  // column scales: 2 x 4 B x the int4 N of every layer - 48 GDN layers x (16384 +
  // 5120 + 34816 + 5120) + 16 FA layers x (14336 + 5120 + 34816 + 5120).
  CHECK_EQ(runtime::int8_scratch_sizes(17408).total(), size_t{35651584 + 8192 + 17825792});
  CHECK_EQ(runtime::int8_scale_bytes(q), size_t{8} * (48 * 61440 + 16 * 59392));
  // = 84,680,704 B: spec 6 §8.4's measured "int8 0.085 GB".
  CHECK_EQ(runtime::int8_scratch_sizes(17408).total() + runtime::int8_scale_bytes(q),
           size_t{84680704});
}

void check_sizes_agnes() {
  // Agnes (spec 14): 54 GDN + 18 FA layers, the MLP folded to 19456; every width
  // else is Qwen3.8's.
  const model::ModelDesc& a = model::agnes();
  const runtime::PersistentSizes p = runtime::PersistentDims::sizes(16384, a);
  CHECK_EQ(p.gdn_state, size_t{54} * 48 * 128 * 128 * 4);         // 169,869,312
  CHECK_EQ(p.conv_ring, size_t{54} * 16 * 10240 * 2);              //  17,694,720
  CHECK_EQ(p.kv_k, size_t{18} * 16384 * 4 * 256 * 2);              // 603,979,776
  // 72 KiB of KV per position (spec 14 §3.3).
  CHECK_EQ(runtime::PersistentDims::sizes(1, a).kv_k * 2, size_t{72} * 1024);
  const runtime::DecodeScratchSizes d = runtime::DecodeScratchDims::sizes(16384, a);
  CHECK_EQ(d.x, size_t{8} * 19456 * 2);
  CHECK_EQ(d.partials, size_t{8} * 8 * 38912 * 4);                 // gate||up N = 2 x 19456
  const runtime::PrefillScratchSizes s = runtime::PrefillScratchDims::sizes(16384, a);
  CHECK_EQ(s.x, size_t{2048} * 19456 * 2);
  CHECK_EQ(s.partials, size_t{2048} * 38912 * 4);
  CHECK_EQ(s.slab, size_t{19456} * 1024 * 2);
  CHECK_EQ(runtime::MtpDims::sizes(16384, a).gdn_spec, size_t{3} * 169869312);
  CHECK_EQ(runtime::int8_scratch_sizes(19456).xq, size_t{2048} * 19456);
  CHECK_EQ(runtime::int8_scale_bytes(a),
           size_t{8} * (54 * (16384 + 5120 + 38912 + 5120) + 18 * (14336 + 5120 + 38912 + 5120)));
}

// Spec 6 §8.4, measured on the card at 131072 (bf16 head, l0-int8, no MTP, before the
// first prefill): "memory: model 18.116 GB, kv 8.590 GB, decode state 0.590 GB,
// prefill scratch 0.700 GB, int8 0.085 GB, total 28.081 GB of 32.530 GB".
void check_spec6_line() {
  // Measured before spec 10, so with v1's attn_part: the plan reproduces it under v1.
  setenv("B70_DECODE_ATTN", "v1", 1);
  const runtime::MemoryPlan p = runtime::plan(model::qwen38(), 131072, false, kQwenBf16Weights);
  unsetenv("B70_DECODE_ATTN");
  // With v2 (the default) the same engine plans 0.399 GB less decode state.
  CHECK_EQ(runtime::plan(model::qwen38(), 131072, false, kQwenBf16Weights).decode_state,
           size_t{590450624} - 405798912 + 6340608);
  CHECK_EQ(p.rope, size_t{131072} * 256);
  CHECK_EQ(p.model, size_t{18116331520ull});   // the 16384 load re-tabled to 131072
  CHECK_EQ(p.kv, size_t{8589934592ull});
  // gdn_state + conv_ring + control + DecodeScratch with attn_part at 2048 blocks.
  CHECK_EQ(p.decode_state, size_t{150994944 + 15728640 + 128} + (68652864 - 50724864 + 405798912));
  CHECK_EQ(p.decode_state, size_t{590450624});
  CHECK_EQ(p.prefill_scratch, size_t{699514776});
  CHECK_EQ(p.prefill_lazy, size_t{0});         // flash attention, no MTP: nothing lazy
  CHECK_EQ(p.int8, size_t{84680704});
  const std::string line = runtime::format_memory("memory", p, kDevice);
  std::printf("%s\n", line.c_str());
  CHECK(line ==
        "memory: model 18.116 GB, kv 8.590 GB, decode state 0.590 GB, prefill scratch 0.700 GB,"
        " int8 0.085 GB, total 28.081 GB of 32.530 GB");
}

void check_paths() {
  const model::ModelDesc& q = model::qwen38();
  const runtime::PrefillScratchSizes s = runtime::PrefillScratchDims::sizes(65536, q);
  const runtime::MemoryPlan base = runtime::plan(q, 65536, false, kQwenInt8Weights);
  // MTP: the head's buffers and prefill hidden rows in decode state, and the L0 slab
  // (step_mtp_kv's bf16 linears) on l0-int8. The head's weights are the caller's.
  const runtime::MemoryPlan mtp = runtime::plan(q, 65536, true, kQwenInt8Weights + kMtpWeights);
  CHECK_EQ(mtp.mtp_buffers, runtime::MtpDims::sizes(65536, q).total());
  CHECK_EQ(mtp.mtp_hidden, runtime::mtp_prefill_hidden_bytes(q));
  CHECK_EQ(mtp.decode_state, base.decode_state + mtp.mtp_buffers + mtp.mtp_hidden);
  CHECK_EQ(mtp.prefill_lazy, s.slab);
  CHECK_EQ(mtp.model, base.model + kMtpWeights);
  // l0: the slab always, no Int8State. sycl-tla: dequant + the composed S / P.
  runtime::PrefillPath l0{true, PrefillBackend::L0, false};
  const runtime::MemoryPlan pl0 = runtime::plan(q, 65536, false, kQwenInt8Weights, l0);
  CHECK_EQ(pl0.prefill_lazy, s.slab);
  CHECK_EQ(pl0.int8, size_t{0});
  runtime::PrefillPath sycl{true, PrefillBackend::SyclTla, false};
  const runtime::MemoryPlan psy = runtime::plan(q, 65536, false, kQwenInt8Weights, sycl);
  CHECK_EQ(psy.prefill_lazy, s.dequant + s.pf_s + s.pf_p);
  // Composed attention on l0-int8: 6 x 2048 x (4 + 2) B = 73,728 B per position.
  runtime::PrefillPath comp{true, PrefillBackend::L0Int8, true};
  const runtime::MemoryPlan pc = runtime::plan(q, 65536, false, kQwenInt8Weights, comp);
  CHECK_EQ(pc.prefill_lazy, size_t{65536} * 73728);
  // Decode only (ruling R7): no prefill scratch, no int8, no MTP hidden rows.
  runtime::PrefillPath none{false, PrefillBackend::L0Int8, false};
  const runtime::MemoryPlan pd = runtime::plan(q, 65536, true, kQwenInt8Weights + kMtpWeights, none);
  CHECK_EQ(pd.prefill_scratch, size_t{0});
  CHECK_EQ(pd.int8, size_t{0});
  CHECK_EQ(pd.mtp_hidden, size_t{0});
}

// The contract, checked from the outside with plan() itself.
uint32_t check_fit(const model::ModelDesc& d, bool mtp, size_t weights, const char* what,
                   const runtime::PrefillPath& path = {}) {
  const uint32_t L = runtime::max_len_that_fits(d, mtp, weights, kDevice, kReserve, kTrained, path);
  CHECK(L != 0);
  CHECK_EQ(L % runtime::kMaxLenQuantum, 0u);
  CHECK(L <= kTrained);
  const runtime::MemoryPlan p = runtime::plan(d, L, mtp, weights, path);
  CHECK(p.total() + kReserve <= kDevice);
  if (L < kTrained)
    CHECK(runtime::plan(d, L + runtime::kMaxLenQuantum, mtp, weights, path).total() + kReserve >
          kDevice);
  std::printf("auto %-30s -> max_len %6u  (%s)\n", what, L,
              runtime::describe(p, kDevice, kReserve).c_str());
  return L;
}

void check_max_len_that_fits() {
  const model::ModelDesc& q = model::qwen38();
  const model::ModelDesc& a = model::agnes();
  const uint32_t q8 = check_fit(q, false, kQwenInt8Weights, "qwen3.8 int8 head");
  const uint32_t q8m = check_fit(q, true, kQwenInt8Weights + kMtpWeights, "qwen3.8 int8 head + MTP");
  const uint32_t qb = check_fit(q, false, kQwenBf16Weights, "qwen3.8 bf16 head");
  const uint32_t a8 = check_fit(a, false, kAgnesInt8Weights, "agnes int8 head");
  const uint32_t a8m = check_fit(a, true, kAgnesInt8Weights + kMtpWeights, "agnes int8 head + MTP");
  const uint32_t qc = check_fit(q, false, kQwenInt8Weights, "qwen3.8 int8, composed attn",
                                runtime::PrefillPath{true, PrefillBackend::L0Int8, true});
  CHECK(q8m < q8 && qb < q8 && a8 < q8 && a8m < a8 && qc < q8);
  // Every one of them is above 128k but below the trained 262144 on one card.
  CHECK(q8 > 131072 && q8 < kTrained);
  CHECK(a8 > 131072 && a8m < a8);

  // A huge card: the trained context caps it, rounded down to the quantum.
  CHECK_EQ(runtime::max_len_that_fits(q, false, kQwenInt8Weights, size_t{1} << 40, kReserve, kTrained),
           kTrained);
  CHECK_EQ(runtime::max_len_that_fits(q, false, kQwenInt8Weights, size_t{1} << 40, kReserve, 40000),
           39936u);
  // Weights alone over the budget: 0, not some tiny context.
  CHECK_EQ(runtime::max_len_that_fits(q, false, size_t{31} * 1000000000, kDevice, kReserve, kTrained),
           0u);
  // The reserve is honoured: a bigger reserve, a shorter context.
  CHECK(runtime::max_len_that_fits(q, false, kQwenInt8Weights, kDevice, size_t{3} * 1000000000,
                                   kTrained) < q8);
  // A cap below one quantum is a caller error.
  bool threw = false;
  try {
    runtime::max_len_that_fits(q, false, kQwenInt8Weights, kDevice, kReserve, 100);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  // A trained context under 4096 still gets planned (the floor is min(4096, cap)).
  CHECK_EQ(runtime::max_len_that_fits(q, false, kQwenInt8Weights, kDevice, kReserve, 2048), 2048u);
}

// Spec 8 §11 (`--draft-vocab`): the compact draft head and its id table are the loader's,
// outside LoadReport::total() (so outside the weights the CLI passes), and MtpBuffers
// gains the compact logits. The plan's `draft_vocab` term is |V'| x 5120 x {1 B + a 4 B
// row scale (int8), 2 B (bf16)} + 4 B ids per row; decode state grows by 4 B per row;
// auto gives the context what is left.
void check_draft_vocab() {
  const model::ModelDesc& q = model::qwen38();
  const size_t weights = kQwenInt8Weights + kMtpWeights;
  const runtime::MemoryPlan off = runtime::plan(q, 65536, true, weights);
  CHECK_EQ(off.draft_vocab, size_t{0});
  for (uint32_t rows : {32768u, 65536u, 131072u}) {
    for (bool int8 : {true, false}) {
      const runtime::MemoryPlan p = runtime::plan(q, 65536, true, weights, {}, {rows, int8});
      const size_t want = size_t(rows) * 5120 * (int8 ? 1 : 2) + (int8 ? size_t(rows) * 4 : 0) +
                          size_t(rows) * 4;
      CHECK_EQ(p.draft_vocab, want);
      CHECK_EQ(p.mtp_buffers, off.mtp_buffers + size_t(rows) * 4);   // dv_logits fp32 [|V'|]
      CHECK_EQ(runtime::MtpDims::sizes(65536, q, rows).dv_logits, size_t(rows) * 4);
      CHECK_EQ(p.total(), off.total() + want + size_t(rows) * 4);
      CHECK_EQ(p.model, off.model);   // not in the weights
    }
  }
  // 128k int8: 671,088,640 + 524,288 + 524,288; bf16: 1,342,177,280 + 524,288.
  CHECK_EQ(runtime::plan(q, 4096, true, weights, {}, {131072, true}).draft_vocab, size_t{672137216});
  CHECK_EQ(runtime::plan(q, 4096, true, weights, {}, {131072, false}).draft_vocab, size_t{1342701568});
  // The memory line names the term only when it is there.
  const std::string line = runtime::format_memory(
      "memory", runtime::plan(q, 65536, true, weights, {}, {131072, true}), kDevice);
  CHECK(line.find(" draft vocab 0.672 GB, total ") != std::string::npos);
  CHECK(runtime::format_memory("memory", off, kDevice).find("draft vocab") == std::string::npos);
  // Auto under a draft vocabulary: shorter than without, and the plan at the chosen
  // length (the draft-vocab term included) fits with the reserve.
  const uint32_t base = runtime::max_len_that_fits(q, true, weights, kDevice, kReserve, kTrained);
  const uint32_t qb = runtime::max_len_that_fits(q, true, kQwenBf16Weights + kMtpWeights, kDevice,
                                                 kReserve, kTrained);
  for (bool int8 : {true, false}) {
    const size_t w = int8 ? weights : kQwenBf16Weights + kMtpWeights;
    const runtime::DraftVocabPlan dv{131072, int8};
    const uint32_t L = runtime::max_len_that_fits(q, true, w, kDevice, kReserve, kTrained, {}, dv);
    CHECK(L != 0 && L < (int8 ? base : qb));
    CHECK(runtime::plan(q, L, true, w, {}, dv).total() + kReserve <= kDevice);
    CHECK(runtime::plan(q, L + runtime::kMaxLenQuantum, true, w, {}, dv).total() + kReserve > kDevice);
    std::printf("auto qwen3.8 %s head + MTP + --draft-vocab 128k -> max_len %6u (without: %u)\n",
                int8 ? "int8" : "bf16", L, int8 ? base : qb);
  }
  // A draft vocabulary without the MTP head is a caller error.
  bool threw = false;
  try {
    runtime::plan(q, 4096, false, weights, {}, {32768, true});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}
}  // namespace

int main() {
  check_sizes_qwen38();
  check_sizes_agnes();
  check_spec6_line();
  check_paths();
  check_max_len_that_fits();
  check_draft_vocab();
  std::puts("memory_plan_test OK");
  return 0;
}
