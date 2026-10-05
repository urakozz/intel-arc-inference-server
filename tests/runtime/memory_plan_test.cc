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

// Spec 15c: Ornith 1.5 35B-A3B with the int8 head, derived from the descriptor (no int4
// checkpoint exists yet): 30 GDN layers x (qkv||z 2048 x 12288 int4 + scales 13,369,344,
// out_proj 4096 x 2048 4,456,448, a||b bf16 128 x 2048 524,288 with its padding, norms
// 16,384, GDN block 131,584) + 10 FA layers x (q||k||v 2048 x 9216 10,027,008, o_proj
// 4,456,448, norms 16,384, FA block 2,048) + 40 MoE layers x 430,604,288
// (loader::moe_layer_bytes) + embed 248320 x 2048 x 2 + the int8 head 2048 x 248320 +
// 248320 x 4 + the final norm 8,192.
constexpr size_t kOrnithInt8Weights = 30ull * 18498048 + 10ull * 14501888 + 40ull * 430604288 +
                                      1017118720ull + 509552640ull + 8192;   // 19,450,811,392

void check_sizes_ornith() {
  const model::ModelDesc& o = model::ornith();
  const runtime::PersistentSizes p = runtime::PersistentDims::sizes(16384, o);
  CHECK_EQ(p.gdn_state, size_t{30} * 32 * 128 * 128 * 4);          //  62,914,560
  CHECK_EQ(p.conv_ring, size_t{30} * 16 * 8192 * 2);               //   7,864,320
  // 20 KiB of KV per position (spec 15 §2): 10 FA layers x 2 kv-heads x 256 x bf16, K and V.
  CHECK_EQ(runtime::PersistentDims::sizes(1, o).kv_k * 2, size_t{20} * 1024);
  // The MoE scratch (runtime::moe_scratch_layout): router logits and route rows per layer
  // slot (40 + the MTP head's), the slots' SiLU x up rows once.
  const runtime::MoeScratchLayout ml = runtime::moe_scratch_layout(o);
  CHECK_EQ(ml.layer_slots, uint32_t(41));
  CHECK_EQ(ml.logits_layer, size_t{8} * 272 * 4);                  // [kM][router_n] fp32
  CHECK_EQ(ml.route_layer, size_t{8} * 32 * 4);                     // [kM][32] u32
  CHECK_EQ(ml.route_off, size_t{41} * 8704);
  CHECK_EQ(ml.h_off, size_t{41} * (8704 + 1024));
  CHECK_EQ(ml.h, size_t{8} * 9 * 512 * 2);                          // [kM][slots][512] bf16
  CHECK_EQ(ml.total, size_t{472576});
  CHECK_EQ(ml.route_at(40), ml.route_off + 40 * size_t{1024});
  const runtime::DecodeScratchSizes d = runtime::DecodeScratchDims::sizes(16384, o);
  CHECK_EQ(d.moe, size_t{472576});
  CHECK_EQ(d.x, size_t{8} * 4096 * 2);                              // 2 x hidden > 512
  CHECK_EQ(d.attn_part, size_t{16} * 32 * 8 * 258 * 4);             // v2 at 16 q-heads
  // Dense models carry no MoE scratch: their sizes are what they were.
  CHECK_EQ(runtime::DecodeScratchDims::sizes(16384, model::qwen38()).moe, size_t{0});
  CHECK_EQ(runtime::moe_scratch_layout(model::agnes()).total, size_t{0});
  // The plan (decode only: Ornith's prefill is spec 15d) carries both MoE terms, and the
  // full trained context fits the card with the default reserve (spec 15 §2: "262k
  // context fits", derived).
  runtime::PrefillPath decode_only;
  decode_only.prefill = false;
  const runtime::MemoryPlan pl = runtime::plan(o, 262144, false, kOrnithInt8Weights, decode_only);
  CHECK_EQ(pl.moe_weights, size_t{17224171520ull});
  CHECK_EQ(pl.moe_scratch, size_t{472576});
  CHECK_EQ(pl.kv, size_t{262144} * 20 * 1024);
  CHECK_EQ(pl.prefill_scratch, size_t{0});
  CHECK_EQ(runtime::max_len_that_fits(o, false, kOrnithInt8Weights, kDevice, kReserve, kTrained,
                                      decode_only),
           kTrained);
  std::printf("ornith (int8 head, decode only): %s\n",
              runtime::describe(pl, kDevice, kReserve).c_str());

  // Spec 15d: Ornith's prefill. The MoE prefill scratch (runtime::moe_prefill_layout) at
  // kC 2048: tmax = (2048 x 8 + 256 x 31) / 32 + 2048 / 32 = 760 + 64 = 824 tiles, 26,368
  // rows; regions 256-aligned.
  const runtime::MoePrefillLayout ml2 = runtime::moe_prefill_layout(o);
  CHECK_EQ(ml2.tmax, uint32_t(824));
  CHECK_EQ(runtime::moe_prefill_tiles(o, 2048), uint32_t(824));
  CHECK_EQ(runtime::moe_prefill_tiles(o, 1), uint32_t(248 + 1));   // (8 + 7936) / 32 + 1
  CHECK_EQ(ml2.rows, uint32_t(26368));
  CHECK_EQ(ml2.route_layer, size_t{2048} * 32 * 4);
  CHECK_EQ(ml2.route_off, size_t{2048} * 272 * 4);                // after the logits
  // The weight batch: the bf16 gate||up of ceil(257 / 2) = 129 blocks (2048 x 1024 x 2 B
  // each) outgrows the int8 gate||up and the bf16 down of all 257 (2 MiB each).
  CHECK_EQ(ml2.w, size_t{129} * 2048 * 1024 * 2);                  // 541,065,216
  CHECK_EQ(ml2.total, size_t{2228224} + 10485760 + 1280 + 6656 + 105472 + 65536 + 105472 +
                          27000832 + 108003328 + 541065216);       // 689,067,776
  CHECK_EQ(runtime::moe_prefill_batch_blocks(o, runtime::MoeWeightForm::GateUpInt8), uint32_t(257));
  CHECK_EQ(runtime::moe_prefill_batch_blocks(o, runtime::MoeWeightForm::DownBf16), uint32_t(257));
  CHECK_EQ(runtime::moe_prefill_batch_blocks(o, runtime::MoeWeightForm::GateUpBf16), uint32_t(129));
  const runtime::PrefillScratchSizes ps = runtime::PrefillScratchDims::sizes(16384, o);
  CHECK_EQ(ps.moe, ml2.total);
  CHECK_EQ(ps.slab, size_t{4096} * 1024 * 2);                       // out_proj / o_proj's K
  CHECK_EQ(ps.partials, size_t{2048} * 12288 * 4);                  // qkv||z's N
  // Dense models carry none, and their slab / int8 K are what they were.
  CHECK_EQ(runtime::moe_prefill_layout(model::qwen38()).total, size_t{0});
  CHECK_EQ(runtime::PrefillScratchDims::sizes(16384, model::qwen38()).moe, size_t{0});
  CHECK_EQ(runtime::prefill_int8_max_k(model::qwen38()), uint32_t(17408));
  CHECK_EQ(runtime::prefill_int8_max_k(model::agnes()), uint32_t(19456));
  CHECK_EQ(runtime::prefill_int8_max_k(o), uint32_t(4096));
  // h8's column scales: the four mixer linears (the shared expert's GateUp / Down rows are
  // inside the expert blocks, not linears) + every layer's 257 x 1024-column gate||up array.
  CHECK_EQ(runtime::int8_scale_bytes(o),
           size_t{8} * (30 * (12288 + 2048) + 10 * (9216 + 2048)) + size_t{40} * 8 * 257 * 1024);
  for (PrefillBackend b : {PrefillBackend::L0Int8, PrefillBackend::L0}) {
    runtime::PrefillPath path;
    path.backend = b;
    const runtime::MemoryPlan pp = runtime::plan(o, 262144, false, kOrnithInt8Weights, path);
    CHECK_EQ(pp.moe_prefill, ml2.total);
    CHECK_EQ(pp.int8, b == PrefillBackend::L0Int8
                          ? runtime::int8_scratch_sizes(4096).total() + runtime::int8_scale_bytes(o)
                          : size_t{0});
    // The full trained context still fits with the prefill scratch (derived).
    CHECK_EQ(runtime::max_len_that_fits(o, false, kOrnithInt8Weights, kDevice, kReserve, kTrained,
                                        path),
             kTrained);
    std::printf("ornith (int8 head, %s prefill): %s\n", runtime::prefill_backend_name(b),
                runtime::describe(pp, kDevice, kReserve).c_str());
  }
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
// Spec 12b (`--kv-cache int8`): the KV term follows the cache's form. One position of one
// FA layer is 4 kv-heads x (256 int8 + one fp16 scale) = 1032 B for K and as much for V,
// against 2048 + 2048 at bf16: 32 KiB + 256 B per position on Qwen3.8 (16 layers), 36 KiB
// + 288 B on Agnes (18). The scales follow every layer's rows in the same allocation.
void check_kv_int8() {
  using runtime::KvCache;
  const model::ModelDesc& q = model::qwen38();
  const model::ModelDesc& a = model::agnes();
  // bf16 is the allocation it always was, whatever B70_KV_CACHE says when it is explicit.
  CHECK_EQ(runtime::PersistentDims::sizes(16384, q, KvCache::Bf16).kv_k, size_t{536870912});
  const runtime::PersistentSizes p8 = runtime::PersistentDims::sizes(16384, q, KvCache::Int8);
  CHECK_EQ(p8.kv_k, size_t{16} * 16384 * 4 * (256 + 2));   // 270,532,608
  CHECK_EQ(p8.kv_v, p8.kv_k);
  CHECK_EQ(p8.total() - p8.kv_k - p8.kv_v,
           runtime::PersistentDims::sizes(16384, q, KvCache::Bf16).total() - 2 * size_t{536870912});
  CHECK_EQ(runtime::PersistentDims::sizes(1, q, KvCache::Int8).kv_k * 2, size_t{32} * 1024 + 256);
  CHECK_EQ(runtime::PersistentDims::sizes(1, a, KvCache::Int8).kv_k * 2, size_t{36} * 1024 + 288);
  // 262144 (the trained context) at int8: 8,657,043,456 B - 0.8 % over bf16 at 131072.
  CHECK_EQ(runtime::PersistentDims::sizes(262144, q, KvCache::Int8).kv_k * 2, size_t{8657043456ull});
  // The MTP head's one layer in the same form.
  CHECK_EQ(runtime::MtpDims::sizes(16384, q, 0, KvCache::Int8).kv_k, size_t{16384} * 1032);
  CHECK_EQ(runtime::MtpDims::sizes(16384, q, 0, KvCache::Int8).total(),
           runtime::MtpDims::sizes(16384, q, 0, KvCache::Bf16).total() - 2 * size_t{16384} * (2048 - 1032));

  // The layout: rows where they always were, then every layer's scales.
  const runtime::KvLayout l8 = runtime::kv_layout(16384, q, q.fa_layers, KvCache::Int8);
  const runtime::KvLayout l16 = runtime::kv_layout(16384, q, q.fa_layers, KvCache::Bf16);
  CHECK_EQ(l16.bytes(), size_t{536870912});
  CHECK_EQ(l16.rows_offset(3), size_t{3} * 16384 * 2048);
  CHECK_EQ(l16.scale_row_bytes(), size_t{0});
  CHECK_EQ(l8.row_bytes(), size_t{1024});
  CHECK_EQ(l8.scale_row_bytes(), size_t{8});
  CHECK_EQ(l8.pos_bytes(), size_t{1032});
  CHECK_EQ(l8.rows_offset(15) + l8.layer_rows(), l8.scales_offset(0));
  CHECK_EQ(l8.scales_offset(0), size_t{16} * 16384 * 1024);
  CHECK_EQ(l8.scales_offset(15) + l8.layer_scales(), l8.bytes());
  CHECK_EQ(l8.bytes(), p8.kv_k);
  {
    char k[1], v[1];
    const runtime::KvLayer L = l8.layer(k, v, 2);
    CHECK(L.int8());
    CHECK_EQ(static_cast<char*>(L.k) - k, std::ptrdiff_t(2 * 16384 * 1024));
    CHECK_EQ(static_cast<char*>(L.vs) - v, std::ptrdiff_t(16 * 16384 * 1024 + 2 * 16384 * 8));
    CHECK(!l16.layer(k, v, 2).int8());
  }

  // The plan's kv term and the MTP head's buffers follow the form; nothing else moves.
  const runtime::MemoryPlan b16 = runtime::plan(q, 131072, true, kQwenInt8Weights + kMtpWeights, {}, {},
                                                KvCache::Bf16);
  const runtime::MemoryPlan b8 = runtime::plan(q, 131072, true, kQwenInt8Weights + kMtpWeights, {}, {},
                                               KvCache::Int8);
  CHECK_EQ(b16.kv, size_t{8589934592ull});
  CHECK_EQ(b8.kv, size_t{131072} * 33024);
  CHECK_EQ(b16.mtp_buffers - b8.mtp_buffers, size_t{2} * 131072 * (2048 - 1032));
  CHECK_EQ(b16.model, b8.model);
  CHECK_EQ(b16.prefill_scratch, b8.prefill_scratch);
  CHECK_EQ(b16.int8, b8.int8);
  CHECK(runtime::describe(b8, kDevice, kReserve).find(", int8 KV cache)") != std::string::npos);
  CHECK(runtime::describe(b16, kDevice, kReserve).find("int8 KV") == std::string::npos);

  // B70_KV_CACHE sets the default (as B70_DECODE_ATTN does for the decode pair).
  setenv("B70_KV_CACHE", "int8", 1);
  CHECK(runtime::default_kv_cache() == KvCache::Int8);
  CHECK_EQ(runtime::PersistentDims::sizes(16384, q).kv_k, p8.kv_k);
  CHECK_EQ(runtime::plan(q, 131072, true, kQwenInt8Weights + kMtpWeights).kv, b8.kv);
  setenv("B70_KV_CACHE", "bf16", 1);
  CHECK(runtime::default_kv_cache() == KvCache::Bf16);
  unsetenv("B70_KV_CACHE");
  CHECK(runtime::default_kv_cache() == KvCache::Bf16);
  CHECK(runtime::parse_kv_cache("int8") == KvCache::Int8);
  bool threw = false;
  try {
    runtime::parse_kv_cache("fp8");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);

  // --max-len auto with the int8 cache: the derived lengths the README quotes. Qwen3.8
  // reaches its trained 262144 on one card with or without MTP; each is at least the bf16
  // length, and the contract (fits, the next quantum does not) holds as for bf16.
  struct Row {
    const model::ModelDesc* d;
    bool mtp;
    size_t w;
    const char* what;
  };
  const Row rows[] = {
      {&q, false, kQwenInt8Weights, "qwen3.8 int8 head"},
      {&q, true, kQwenInt8Weights + kMtpWeights, "qwen3.8 int8 head + MTP"},
      {&q, false, kQwenBf16Weights, "qwen3.8 bf16 head"},
      {&q, true, kQwenBf16Weights + kMtpWeights, "qwen3.8 bf16 head + MTP"},
      {&a, false, kAgnesInt8Weights, "agnes int8 head"},
      {&a, true, kAgnesInt8Weights + kMtpWeights, "agnes int8 head + MTP"},
  };
  for (const Row& r : rows) {
    const uint32_t l16 = runtime::max_len_that_fits(*r.d, r.mtp, r.w, kDevice, kReserve, kTrained, {},
                                                    {}, KvCache::Bf16);
    const uint32_t l8 = runtime::max_len_that_fits(*r.d, r.mtp, r.w, kDevice, kReserve, kTrained, {},
                                                   {}, KvCache::Int8);
    CHECK(l8 >= l16 && l8 % runtime::kMaxLenQuantum == 0 && l8 <= kTrained);
    const runtime::MemoryPlan p = runtime::plan(*r.d, l8, r.mtp, r.w, {}, {}, KvCache::Int8);
    CHECK(p.total() + kReserve <= kDevice);
    if (l8 < kTrained)
      CHECK(runtime::plan(*r.d, l8 + runtime::kMaxLenQuantum, r.mtp, r.w, {}, {}, KvCache::Int8)
                    .total() + kReserve > kDevice);
    std::printf("auto --kv-cache int8 %-26s -> max_len %6u (bf16 KV: %6u)  (%s)\n", r.what, l8,
                l16, runtime::describe(p, kDevice, kReserve).c_str());
    if (r.d == &q) CHECK_EQ(l8, kTrained);
  }
}
}  // namespace

int main() {
  check_sizes_qwen38();
  check_sizes_agnes();
  check_sizes_ornith();
  check_spec6_line();
  check_paths();
  check_max_len_that_fits();
  check_draft_vocab();
  check_kv_int8();
  std::puts("memory_plan_test OK");
  return 0;
}
