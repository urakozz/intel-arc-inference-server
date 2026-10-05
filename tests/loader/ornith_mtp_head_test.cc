// Spec 15e: Ornith's MoE MTP head on the host - no device, no checkpoint.
//
//   1. loader::rtn_int4_g64 (loader/rtn.h): the GPTQ v1 nibble and scale layout, GPTQ's
//      symmetric formula on the stored f16 scale, the dequant error inside half a step,
//      an all-zero group, a tiny-scale group (subnormal f16 kept);
//   2. the head's MoE layer through loader::repack_moe_layer with `rtn_bf16_experts` on a
//      synthetic checkpoint in the published head's naming (`mtp.layers.0.mlp.experts.E.
//      {gate,up,down}_proj.weight` bf16, the router and the shared gate bf16) at a small
//      MoE shape: every expert block (shared last) holds exactly the RTN of ITS source
//      linears, gate||up interleaved in 16-column blocks; the bytes and counts the loader
//      reports; a head whose first experts ship int4 repacks those as shipped and
//      quantises only the rest; without the flag a bf16 expert is refused, as the main
//      layers' are;
//   3. the figures: Ornith's head is 785 bf16 tensors / 1,689,281,536 B in the checkpoint
//      (ModelDesc::mtp_checkpoint_*), 501,990,464 B on the card (loader::mtp_head_bytes),
//      Qwen3.8's dense head unchanged at 849,451,008 B;
//   4. the memory planner with the head: its buffers (MtpDims) at Ornith's 30 GDN layers
//      and one 2-kv-head layer, the prefill hidden rows and the L0 slab, and the full
//      262144 context still fitting the card with the int8 head on both L0 prefill paths.
//
// usage: ornith_mtp_head_test
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/kv8.h"
#include "loader/moe.h"
#include "loader/moe_layout.h"
#include "loader/rtn.h"
#include "loader/safetensors.h"
#include "model/model_desc.h"
#include "runtime/buffer_sizes.h"
#include "runtime/memory_plan.h"

namespace {

namespace fs = std::filesystem;

// --- a minimal safetensors writer (ornith_repack_test's) -----------------------------
struct Tensor {
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> bytes;
};

void write_checkpoint(const fs::path& dir, const std::map<std::string, Tensor>& ts) {
  fs::create_directories(dir);
  std::string hdr = "{";
  uint64_t off = 0;
  bool first = true;
  for (const auto& [name, t] : ts) {
    hdr += (first ? "" : ",");
    first = false;
    hdr += "\"" + name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < t.shape.size(); ++i) hdr += (i ? "," : "") + std::to_string(t.shape[i]);
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t.bytes.size()) +
           "]}";
    off += t.bytes.size();
    off = (off + 7) / 8 * 8;
  }
  hdr += "}";
  while ((8 + hdr.size()) % 8 != 0) hdr += ' ';
  std::ofstream f(dir / "model-00001-of-00001.safetensors", std::ios::binary);
  const uint64_t hlen = hdr.size();
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f.write(hdr.data(), std::streamsize(hdr.size()));
  uint64_t pos = 0;
  for (const auto& [name, t] : ts) {
    (void)name;
    f.write(reinterpret_cast<const char*>(t.bytes.data()), std::streamsize(t.bytes.size()));
    pos += t.bytes.size();
    const uint64_t pad = (pos + 7) / 8 * 8 - pos;
    const char zeros[8] = {};
    f.write(zeros, std::streamsize(pad));
    pos += pad;
  }
  std::ofstream ix(dir / "model.safetensors.index.json");
  ix << "{\"metadata\":{},\"weight_map\":{";
  first = true;
  for (const auto& [name, t] : ts) {
    (void)t;
    ix << (first ? "" : ",") << "\"" << name << "\":\"model-00001-of-00001.safetensors\"";
    first = false;
  }
  ix << "}}";
}

void remove_fixture(const fs::path& dir) {
  fs::remove(dir / "model-00001-of-00001.safetensors");
  fs::remove(dir / "model.safetensors.index.json");
  fs::remove(dir);
}

template <class T>
std::vector<uint8_t> as_bytes(const std::vector<T>& v) {
  std::vector<uint8_t> b(v.size() * sizeof(T));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

// --- 1. RTN ------------------------------------------------------------------------
// The formula, element by element, as rtn.h states it (an independent loop order).
uint32_t rtn_q(float v, float s) {
  if (!(s > 0.f)) return 8;
  float q = std::rint(v / s) + 8.0f;
  return uint32_t(std::fmin(std::fmax(q, 0.0f), 15.0f));
}
uint32_t nibble(const std::vector<uint32_t>& qw, uint32_t N, uint32_t k, uint32_t n) {
  return (qw[size_t(k / 8) * N + n] >> (4 * (k % 8))) & 15u;
}

void check_rtn() {
  const uint32_t K = 192, N = 5;   // three groups, five columns
  std::vector<uint16_t> w(size_t(N) * K);
  std::mt19937 rng(15);
  std::normal_distribution<float> nd(0.f, 0.02f);
  for (uint16_t& x : w) x = common::f32_to_bf16(nd(rng));
  for (uint32_t k = 64; k < 128; ++k) w[size_t(2) * K + k] = 0;   // col 2, group 1: all zero
  for (uint32_t k = 128; k < 192; ++k)                             // col 4, group 2: tiny
    w[size_t(4) * K + k] = common::f32_to_bf16((k % 2 ? 1.f : -1.f) * 3e-7f);
  std::vector<uint32_t> qw(size_t(K / 8) * N);
  std::vector<uint16_t> sc(size_t(K / 64) * N);
  loader::rtn_int4_g64(w.data(), K, N, qw.data(), sc.data());
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t g = 0; g < K / 64; ++g) {
      float amax = 0.f;
      for (uint32_t k = g * 64; k < g * 64 + 64; ++k)
        amax = std::fmax(amax, std::fabs(common::bf16_to_f32(w[size_t(n) * K + k])));
      const uint16_t s16 = common::kv8::f32_to_f16_rne(2.0f * amax / 15.0f);
      CHECK_EQ(sc[size_t(g) * N + n], s16);
      const float s = common::f16_to_f32(s16);
      for (uint32_t k = g * 64; k < g * 64 + 64; ++k) {
        const float v = common::bf16_to_f32(w[size_t(n) * K + k]);
        const uint32_t q = nibble(qw, N, k, n);
        CHECK_EQ(q, rtn_q(v, s));
        // The dequant the kernels compute is within half a step (the ends of the range
        // clip by about half a step too: |v| <= 7.5 s up to the f16 rounding of s).
        CHECK(std::fabs(s * (float(q) - 8.0f) - v) <= 0.51f * s + 1e-12f);
      }
    }
  CHECK_EQ(sc[size_t(1) * N + 2], uint16_t(0));                     // the zero group: s = 0,
  for (uint32_t k = 64; k < 128; ++k) CHECK_EQ(nibble(qw, N, k, 2), 8u);   // q = 8
  CHECK(sc[size_t(2) * N + 4] != 0 && sc[size_t(2) * N + 4] < 0x0400);     // a subnormal scale
  std::printf("rtn: GPTQ sym int4 g64, %u x %u, half-step bound, zero and subnormal groups\n", K,
              N);
}

// --- 2. the head's MoE layer -------------------------------------------------------
model::ModelDesc small_moe() {   // ornith_repack_test's small shape
  model::ModelDesc d = model::ornith();
  d.hidden = 128;
  d.moe = {16, 4, 64, 64, true};
  d.intermediate = 64;
  return d;
}

const std::string kL = "mtp.layers.0.";

struct Bf16Linear {   // [N][K] row-major
  uint32_t K = 0, N = 0;
  std::vector<uint16_t> w;
};
Bf16Linear random_bf16(uint32_t K, uint32_t N, std::mt19937& rng) {
  Bf16Linear l;
  l.K = K;
  l.N = N;
  l.w.resize(size_t(N) * K);
  std::normal_distribution<float> nd(0.f, 0.05f);
  for (uint16_t& x : l.w) x = common::f32_to_bf16(nd(rng));
  return l;
}

struct Head {   // the source linears of the head's MoE layer, per block (shared last)
  std::vector<Bf16Linear> g, u, dn;
};

// The published head's names: bf16 per-expert linears. Experts below `int4_below` ship
// their three linears int4 instead (GPTQ-packed RTN of the same weights, qzeros 0x77...).
Head make_head(const model::ModelDesc& d, const fs::path& dir, uint32_t int4_below) {
  const uint32_t E = d.moe.experts, H = d.hidden, I = d.moe.expert_intermediate;
  std::mt19937 rng(20261005);
  std::map<std::string, Tensor> ts;
  std::vector<uint16_t> router(size_t(E) * H), gate(H);
  std::normal_distribution<float> nd(0.f, 0.3f);
  for (uint16_t& x : router) x = common::f32_to_bf16(nd(rng));
  for (uint16_t& x : gate) x = common::f32_to_bf16(nd(rng));
  ts[kL + "mlp.gate.weight"] = {"BF16", {E, H}, as_bytes(router)};
  ts[kL + "mlp.shared_expert_gate.weight"] = {"BF16", {1, H}, as_bytes(gate)};
  Head h;
  auto add = [&](const std::string& p, const Bf16Linear& l, bool int4) {
    if (!int4) {
      ts[p + ".weight"] = {"BF16", {l.N, l.K}, as_bytes(l.w)};
      return;
    }
    std::vector<uint32_t> q(size_t(l.K / 8) * l.N);
    std::vector<uint16_t> s(size_t(l.K / 64) * l.N);
    loader::rtn_int4_g64(l.w.data(), l.K, l.N, q.data(), s.data());
    ts[p + ".qweight"] = {"I32", {l.K / 8, l.N}, as_bytes(q)};
    ts[p + ".scales"] = {"F16", {l.K / 64, l.N}, as_bytes(s)};
    std::vector<uint32_t> qz(size_t(l.K / 64) * (l.N / 8), 0x77777777u);
    ts[p + ".qzeros"] = {"I32", {l.K / 64, l.N / 8}, as_bytes(qz)};
  };
  for (uint32_t b = 0; b <= E; ++b) {
    h.g.push_back(random_bf16(H, I, rng));
    h.u.push_back(random_bf16(H, I, rng));
    h.dn.push_back(random_bf16(I, H, rng));
    const std::string p = b == E ? kL + "mlp.shared_expert." : kL + "mlp.experts." + std::to_string(b) + ".";
    const bool int4 = b < int4_below;
    add(p + "gate_proj", h.g[b], int4);
    add(p + "up_proj", h.u[b], int4);
    add(p + "down_proj", h.dn[b], int4);
  }
  write_checkpoint(dir, ts);
  return h;
}

// The loader's binding for the head (loader.cc load_mtp): the `mtp.layers.0.` prefix.
loader::MoeSource head_source(const loader::SafetensorsSet& set, std::vector<std::string>* asked) {
  loader::MoeSource src;
  src.has = [&set](const std::string& part) {
    return set.tensors().count(kL + part + ".qweight") != 0 ||
           set.tensors().count(kL + part + ".weight") != 0;
  };
  src.linear = [&set, asked](const std::string& part) {
    if (asked) asked->push_back(part);
    return loader::LinearSrc::classify(set, kL + part);
  };
  return src;
}

struct Nib {
  uint32_t q;
  uint16_t scale;
};
Nib layout1_at(const uint32_t* blk, uint32_t K, uint32_t n, uint32_t k) {
  const uint32_t* tile = blk + (size_t(n / 16) * (K / 64) + k / 64) * loader::kInt4TileU32;
  const uint32_t word = tile[((k % 64) / 8) * 16 + n % 16];
  const uint32_t sw = tile[128 + (n % 16) / 2];
  return {(word >> (4 * (k % 8))) & 15u, uint16_t(n % 2 == 0 ? sw & 0xFFFFu : sw >> 16)};
}

// Every block of `host` against the RTN of its own source linears.
void check_blocks(const model::ModelDesc& d, const loader::MoeHost& host, const Head& h) {
  const uint32_t E = d.moe.experts, H = d.hidden, I = d.moe.expert_intermediate;
  const loader::MoeLayerBytes b = loader::moe_layer_bytes(d);
  auto rtn = [](const Bf16Linear& l, std::vector<uint32_t>& q, std::vector<uint16_t>& s) {
    q.resize(size_t(l.K / 8) * l.N);
    s.resize(size_t(l.K / 64) * l.N);
    loader::rtn_int4_g64(l.w.data(), l.K, l.N, q.data(), s.data());
  };
  std::vector<uint32_t> gq, uq, dq;
  std::vector<uint16_t> gs, us, ds;
  for (uint32_t blk = 0; blk <= E; ++blk) {
    rtn(h.g[blk], gq, gs);
    rtn(h.u[blk], uq, us);
    rtn(h.dn[blk], dq, ds);
    const uint32_t* gu = host.gate_up.data() + blk * (b.gate_up_block / 4);
    for (uint32_t c = 0; c < 2 * I; ++c) {
      const bool is_up = (c % 32) >= 16;
      const uint32_t col = (c / 32) * 16 + c % 16;
      const std::vector<uint32_t>& q = is_up ? uq : gq;
      const std::vector<uint16_t>& s = is_up ? us : gs;
      for (uint32_t k = 0; k < H; ++k) {
        const Nib got = layout1_at(gu, H, c, k);
        if (got.q != nibble(q, I, k, col) || got.scale != s[size_t(k / 64) * I + col]) {
          std::fprintf(stderr, "head gate||up block %u column %u k %u mismatch\n", blk, c, k);
          std::exit(1);
        }
      }
    }
    const uint32_t* dn = host.down.data() + blk * (b.down_block / 4);
    for (uint32_t n = 0; n < H; ++n)
      for (uint32_t k = 0; k < I; ++k) {
        const Nib got = layout1_at(dn, I, n, k);
        if (got.q != nibble(dq, H, k, n) || got.scale != ds[size_t(k / 64) * H + n]) {
          std::fprintf(stderr, "head down block %u column %u k %u mismatch\n", blk, n, k);
          std::exit(1);
        }
      }
  }
}

template <class F>
bool throws_with(F&& f, const std::string& needle) {
  try {
    f();
  } catch (const std::runtime_error& e) {
    if (std::string(e.what()).find(needle) != std::string::npos) return true;
    std::fprintf(stderr, "threw, but without '%s': %s\n", needle.c_str(), e.what());
    return false;
  }
  return false;
}

void check_head_repack() {
  const model::ModelDesc d = small_moe();
  const uint32_t E = d.moe.experts, H = d.hidden, I = d.moe.expert_intermediate;
  const fs::path root = fs::temp_directory_path() / ("ornith_mtp_head_test_" + std::to_string(getpid()));
  // (a) the published form: every head tensor bf16.
  {
    const fs::path dir = root / "bf16";
    const Head h = make_head(d, dir, 0);
    loader::SafetensorsSet set(dir.string() + "/");
    std::vector<std::string> asked;
    loader::MoeHost host;
    loader::repack_moe_layer(d, head_source(set, &asked), host, "the MTP head",
                             /*rtn_bf16_experts=*/true);
    CHECK_EQ(asked.size(), size_t(2 + 3 * (E + 1)));   // each linear classified ONCE
    CHECK_EQ(host.rtn_linears, size_t(3 * (E + 1)));
    CHECK_EQ(host.int4_src_bytes, size_t(0));
    CHECK_EQ(host.bf16_src_bytes, size_t(E + 1) * H * 2 + size_t(E + 1) * 3 * H * I * 2);
    check_blocks(d, host, h);
    // Without the flag (the main layers' rule) a bf16 expert is refused by name.
    loader::MoeHost refused;
    CHECK(throws_with([&] { loader::repack_moe_layer(d, head_source(set, nullptr), refused, "layer 0"); },
                      "mlp.experts.0.gate_proj' is bf16"));
    remove_fixture(dir);
  }
  // (b) a head whose experts 0..5 ship int4 (the RTN of the same weights, GPTQ-packed):
  // those are repacked as shipped - the same bytes - and only the rest quantised here.
  {
    const fs::path dir = root / "mixed";
    const Head h = make_head(d, dir, 6);
    loader::SafetensorsSet set(dir.string() + "/");
    loader::MoeHost host;
    loader::repack_moe_layer(d, head_source(set, nullptr), host, "the MTP head", true);
    CHECK_EQ(host.rtn_linears, size_t(3 * (E + 1 - 6)));
    const size_t lin_i4 = size_t(H / 8) * I * 4 + size_t(H / 64) * I * 2;   // one K x N int4
    CHECK_EQ(host.int4_src_bytes, size_t(6) * 3 * lin_i4);
    check_blocks(d, host, h);
    remove_fixture(dir);
  }
  fs::remove(root);
  std::printf("head repack: %u experts + shared, bf16 -> RTN int4 per linear, int4 as shipped\n", E);
}

// --- 3 + 4. the figures and the planner ---------------------------------------------
// memory_plan_test's Ornith weights with the int8 head (derived from the descriptor).
constexpr size_t kOrnithInt8Weights = 30ull * 18498048 + 10ull * 14501888 + 40ull * 430604288 +
                                      1017118720ull + 509552640ull + 8192;   // 19,450,811,392
constexpr size_t kDevice = 32530000000ull;
constexpr size_t kReserve = size_t(runtime::kDefaultReserveGb * 1e9);

void check_figures() {
  const model::ModelDesc& o = model::ornith();
  CHECK(o.mtp_head_moe());
  CHECK_EQ(o.mtp_checkpoint_tensors(), size_t(785));
  CHECK_EQ(o.mtp_checkpoint_bytes(), size_t(1689281536));
  CHECK_EQ(loader::mtp_head_bytes(o),
           size_t(16777216) + 37748736 + 16777216 + 40960 + 2048 + 430604288);   // 501,990,464
  CHECK_EQ(loader::mtp_head_bytes(model::qwen38()), size_t(849451008));
  CHECK_EQ(loader::mtp_head_bytes(model::agnes()), size_t(849451008));
  CHECK_EQ(model::qwen38().mtp_checkpoint_tensors(), size_t(15));

  // The head's buffers at Ornith's shapes (MtpDims::sizes), 262144 positions, bf16 KV.
  const runtime::MtpSizes m = runtime::MtpDims::sizes(262144, o, 0, runtime::KvCache::Bf16);
  CHECK_EQ(m.gdn_spec, size_t{3} * 30 * 32 * 128 * 128 * 4);   // 3 slots x 62,914,560
  CHECK_EQ(m.kv_k, size_t{262144} * 2 * 256 * 2);              // one layer, 2 kv-heads
  CHECK_EQ(m.hh, size_t{9} * 2048 * 2);
  CHECK_EQ(m.dh, size_t{2048} * 2);
  CHECK_EQ(runtime::mtp_prefill_hidden_bytes(o), size_t{2049} * 2048 * 2);
  const size_t weights = kOrnithInt8Weights + loader::mtp_head_bytes(o);
  for (runtime::PrefillBackend b : {runtime::PrefillBackend::L0Int8, runtime::PrefillBackend::L0}) {
    runtime::PrefillPath path;
    path.backend = b;
    const runtime::MemoryPlan off = runtime::plan(o, 262144, false, kOrnithInt8Weights, path, {},
                                                  runtime::KvCache::Bf16);
    const runtime::MemoryPlan on = runtime::plan(o, 262144, true, weights, path, {},
                                                 runtime::KvCache::Bf16);
    CHECK_EQ(on.mtp_buffers, m.total());
    CHECK_EQ(on.mtp_hidden, runtime::mtp_prefill_hidden_bytes(o));
    CHECK_EQ(on.decode_state, off.decode_state + m.total() + on.mtp_hidden);
    CHECK_EQ(on.model, off.model + loader::mtp_head_bytes(o));
    // step_mtp_kv's bf16 linears walk the L0 slab: on l0-int8 it is built for MTP alone.
    const runtime::PrefillScratchSizes ps = runtime::PrefillScratchDims::sizes(262144, o);
    CHECK_EQ(on.prefill_lazy, ps.slab);
    CHECK(ps.slab >= size_t{4096} * 1024 * 2);   // fc's K 4096 fits the slab (step_mtp_kv)
    CHECK_EQ(runtime::max_len_that_fits(o, true, weights, kDevice, kReserve, 262144, path, {},
                                        runtime::KvCache::Bf16),
             uint32_t(262144));
    std::printf("ornith + MTP (int8 head, %s prefill): %s\n", runtime::prefill_backend_name(b),
                runtime::describe(on, kDevice, kReserve).c_str());
  }
}

}  // namespace

int main() {
  check_rtn();
  check_head_repack();
  check_figures();
  std::puts("ornith_mtp_head_test OK");
  return 0;
}
