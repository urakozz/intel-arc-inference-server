// Spec 25b Task 4: the Bolt loader's host half over a synthetic real-width checkpoint (tools/quantize/bolt/make_synth.py, N
// blocks, either spelling) - NO Level Zero.
//   argv[1] = the synthetic's directory (SKIP 77 when absent: the box's $DATA/oracle-out-bolt-synth/<name>)
//   argv[2] = (optional) the OTHER spelling of the same weights (the g64 form made with `--expand-from`): every block's
//             repacked bytes must equal argv[1]'s, byte for byte
// Checks: config.json held to model::bolt() and its descriptor at the checkpoint's block count; the quantisation config and
// invariants (signed F16 scales counted); every name accounted for both ways; every block repacked; the repacked words equal
// the shards' (sampled against the raw tensors, independent index arithmetic); the small block's A = -exp(A_log), conv taps,
// router tiles; the vision tensors skipped by prefix and counted; 0 unconsumed.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/exp_torch.h"
#include "common/json.h"
#include "loader/bolt_loader.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/bolt.h"

namespace {

bool exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}

common::json::Value read_json(const std::string& path) {
  std::ifstream f(path);
  CHECK(f.good());
  std::stringstream s;
  s << f.rdbuf();
  return common::json::parse(s.str());
}

// The (k, n) weight's nibble of a layout-1 buffer.
int tile_q(const uint32_t* w, uint32_t K, uint32_t k, uint32_t n) {
  const uint32_t G = K / 64, nt = n / 16, g = k / 64, kk = k % 64;
  return int((w[(size_t(nt) * G + g) * 136 + (kk / 8) * 16 + n % 16] >> (4 * (kk % 8))) & 0xFu);
}
uint16_t tile_s(const uint32_t* w, uint32_t K, uint32_t k, uint32_t n) {
  const uint32_t G = K / 64, nt = n / 16, g = k / 64;
  const uint32_t word = w[(size_t(nt) * G + g) * 136 + 128 + (n % 16) / 2];
  return uint16_t((n % 2) ? word >> 16 : word & 0xFFFFu);
}

// Raw access to a linear's tensors as the shards hold them (no classify): qweight [K/8][N] and scales [K/g][N].
struct Raw {
  const uint32_t* qw;
  const uint16_t* sc;
  uint32_t K, N, group;
};
Raw raw(const loader::SafetensorsSet& set, const std::string& base) {
  const auto& q = set.tensors().at(base + ".qweight");
  const auto& s = set.tensors().at(base + ".scales");
  CHECK(q.dtype == "I32" && s.dtype == "F16");
  Raw r;
  r.qw = reinterpret_cast<const uint32_t*>(set.data(q));
  r.sc = reinterpret_cast<const uint16_t*>(set.data(s));
  r.K = uint32_t(q.shape[0]) * 8;
  r.N = uint32_t(q.shape[1]);
  r.group = r.K / uint32_t(s.shape[0]);
  return r;
}
int raw_q(const Raw& r, uint32_t k, uint32_t n) { return int((r.qw[size_t(k / 8) * r.N + n] >> (4 * (k % 8))) & 0xFu); }
uint16_t raw_s(const Raw& r, uint32_t k, uint32_t n) { return r.sc[size_t(k / r.group) * r.N + n]; }

void same_layer(const loader::BoltHostLayer& a, const loader::BoltHostLayer& b, uint32_t l) {
  auto eq = [&](const char* what, const void* x, size_t xn, const void* y, size_t yn) {
    if (xn != yn || std::memcmp(x, y, xn) != 0) {
      std::fprintf(stderr, "block %u: %s differs between the two spellings\n", l, what);
      std::exit(1);
    }
  };
#define SAME_VEC(f) eq(#f, a.f.data(), a.f.size() * sizeof(a.f[0]), b.f.data(), b.f.size() * sizeof(b.f[0]))
  SAME_VEC(in_proj.words);
  SAME_VEC(in_proj.scales);
  SAME_VEC(out_proj.words);
  SAME_VEC(out_proj.scales);
  SAME_VEC(qkv.words);
  SAME_VEC(qkv.scales);
  SAME_VEC(o_proj.words);
  SAME_VEC(o_proj.scales);
  SAME_VEC(fc1_shared_up.words);
  SAME_VEC(fc1_shared_up.scales);
  SAME_VEC(fc2.words);
  SAME_VEC(fc2.scales);
  SAME_VEC(shared_down.words);
  SAME_VEC(shared_down.scales);
  SAME_VEC(experts);
  SAME_VEC(router);
  SAME_VEC(small);
#undef SAME_VEC
}

}  // namespace

int main(int argc, char** argv) {
  CHECK(argc >= 2);
  const std::string dir = std::string(argv[1]) + "/";
  if (!exists(dir + "config.json")) {
    std::printf("SKIP: no synthetic checkpoint at %s (tools/quantize/bolt/make_synth.py)\n", argv[1]);
    return 77;
  }
  const common::json::Value cfg = read_json(dir + "config.json");
  CHECK(model::is_bolt(cfg));
  model::check_bolt_config(cfg);
  const model::BoltDesc d = model::bolt_desc(cfg);
  const model::BoltDesc& real = model::bolt();
  CHECK(d.layers >= 8 && d.layers <= 88);
  CHECK(std::equal(d.block.begin(), d.block.end(), real.block.begin()) &&
        std::equal(d.experts.begin(), d.experts.end(), real.experts.begin()) &&
        std::equal(d.inter.begin(), d.inter.end(), real.inter.begin()));   // the first N blocks of the real model

  const loader::QuantConfig qc = loader::QuantConfig::parse(cfg);
  CHECK(qc.group_size == 128 || qc.group_size == 64);
  loader::SafetensorsSet set(dir);
  const loader::QuantScan scan = loader::assert_quant_invariants(set);
  loader::check_quant_scan(qc, scan);
  if (qc.group_size == 128) {
    CHECK_EQ(qc.dynamic_rule_count, size_t(44));
    CHECK_EQ(qc.dynamic_restating, size_t(1));
    CHECK(scan.g_idx_tensors > 0);
  } else {
    CHECK_EQ(scan.g_idx_tensors, size_t(0));
  }
  const std::string trunk = loader::bolt_trunk_prefix(set);
  CHECK(trunk == "language_model.model.");
  loader::BoltCheckpoint ck(d, set, trunk);
  ck.check_names();   // every name accounted for, both ways
  const size_t vision = ck.skip_vision();
  CHECK(vision >= 3);
  std::printf("bolt_synth_host_test: %s: %u blocks, group %u, %zu tensors, %zu g_idx, %zu subnormal f16 scales, %zu vision "
              "tensors skipped\n", argv[1], d.layers, qc.group_size, set.tensors().size(), scan.g_idx_tensors,
              scan.subnormal_scales, vision);

  std::unique_ptr<loader::SafetensorsSet> set2;
  std::unique_ptr<loader::BoltCheckpoint> ck2;
  if (argc > 2) {
    const std::string dir2 = std::string(argv[2]) + "/";
    CHECK(exists(dir2 + "config.json"));
    const model::BoltDesc d2 = model::bolt_desc(read_json(dir2 + "config.json"));
    CHECK(d2.block == d.block && d2.experts == d.experts && d2.inter == d.inter);
    set2 = std::make_unique<loader::SafetensorsSet>(dir2);
    loader::assert_quant_invariants(*set2);
    ck2 = std::make_unique<loader::BoltCheckpoint>(d2, *set2, loader::bolt_trunk_prefix(*set2));
    ck2->check_names();
    ck2->skip_vision();
  }

  std::mt19937 rng(7);
  size_t sampled = 0;
  loader::BoltHostLayer host, host2;
  for (uint32_t l = 0; l < d.layers; ++l) {
    ck.repack_layer(l, host);
    if (ck2) {
      ck2->repack_layer(l, host2);
      same_layer(host, host2, l);   // both spellings: the same bytes
    }
    const std::string m = trunk + "layers." + std::to_string(l) + ".mixer.";
    const loader::bolt::SmallBlock sb = loader::bolt::small_block(d, l);
    CHECK_EQ(host.small.size() * 4, sb.bytes);
    // the block norm, widened bf16
    {
      const auto& t = set.tensors().at(trunk + "layers." + std::to_string(l) + ".norm.weight");
      const uint16_t* p = reinterpret_cast<const uint16_t*>(set.data(t));
      for (uint32_t i = 0; i < d.hidden; i += 97) CHECK_EQ(host.small[sb.norm / 4 + i], common::bf16_to_f32(p[i]));
    }
    // sampled words of the repacked linears against the raw shard tensors (independent index arithmetic)
    auto sample_fused = [&](const loader::BoltHostLinear& h, const model::BoltLinear& ln) {
      uint32_t off = 0;
      for (size_t p = 0; p < ln.parts.size(); ++p) {
        const Raw r = raw(set, m + ln.parts[p]);
        CHECK(r.K == ln.shape.K && r.N == ln.part_n[p]);
        for (int s = 0; s < 400; ++s) {
          const uint32_t k = rng() % r.K, n = rng() % r.N, fn = off + n;
          if (ln.shape.layout == 1) {
            CHECK_EQ(tile_q(h.words.data(), r.K, k, fn), raw_q(r, k, n));
            CHECK_EQ(tile_s(h.words.data(), r.K, k, fn), raw_s(r, k, n));
          } else {   // layout 0: GPTQ-native words + separate g64 scales
            CHECK_EQ(int((h.words[size_t(k / 8) * ln.shape.N + fn] >> (4 * (k % 8))) & 0xFu), raw_q(r, k, n));
            CHECK_EQ(h.scales[size_t(k / 64) * ln.shape.N + fn], raw_s(r, k, n));
          }
          ++sampled;
        }
        off += r.N;
      }
    };
    for (const model::BoltLinear& ln : d.linears(l)) {
      const loader::BoltHostLinear* h = nullptr;
      switch (ln.id) {
        case model::BoltLinearId::InProj: h = &host.in_proj; break;
        case model::BoltLinearId::OutProj: h = &host.out_proj; break;
        case model::BoltLinearId::Qkv: h = &host.qkv; break;
        case model::BoltLinearId::OProj: h = &host.o_proj; break;
        case model::BoltLinearId::Fc1SharedUp: h = &host.fc1_shared_up; break;
        case model::BoltLinearId::Fc2: h = &host.fc2; break;
        case model::BoltLinearId::SharedDown: h = &host.shared_down; break;
        default: break;
      }
      if (h) {
        CHECK_EQ(h->words.size() * 4 + h->scales.size() * 2, loader::bolt::int4_bytes(ln.shape));
        sample_fused(*h, ln);
      }
    }
    if (d.is_mamba(l)) {
      // A = -exp(A_log) (the F32 tensor holds bf16 values), D, dt_bias verbatim; conv taps oldest first; gated norm widened
      const auto& al = set.tensors().at(m + "A_log");
      CHECK(al.dtype == "F32");
      const float* a_log = reinterpret_cast<const float*>(set.data(al));
      const float* D = reinterpret_cast<const float*>(set.data(set.tensors().at(m + "D")));
      const float* dtb = reinterpret_cast<const float*>(set.data(set.tensors().at(m + "dt_bias")));
      for (uint32_t h = 0; h < d.m_heads; ++h) {
        CHECK_EQ(host.small[sb.A / 4 + h], -common::exp_torch(a_log[h]));
        CHECK(host.small[sb.A / 4 + h] < 0.0f);
        CHECK_EQ(host.small[sb.D / 4 + h], D[h]);
        CHECK_EQ(host.small[sb.dt_bias / 4 + h], dtb[h]);
      }
      const uint16_t* cw = reinterpret_cast<const uint16_t*>(set.data(set.tensors().at(m + "conv1d.weight")));
      for (uint32_t c = 0; c < d.m_conv_dim; c += 61)
        for (uint32_t j = 0; j < d.m_conv; ++j)   // taps ascending: j = 0 is the OLDEST position
          CHECK_EQ(host.small[sb.conv_w / 4 + size_t(c) * d.m_conv + j], common::bf16_to_f32(cw[size_t(c) * d.m_conv + j]));
      const uint16_t* cb = reinterpret_cast<const uint16_t*>(set.data(set.tensors().at(m + "conv1d.bias")));
      for (uint32_t c = 0; c < d.m_conv_dim; c += 61) CHECK_EQ(host.small[sb.conv_b / 4 + c], common::bf16_to_f32(cb[c]));
      const uint16_t* gn = reinterpret_cast<const uint16_t*>(set.data(set.tensors().at(m + "norm.weight")));
      for (uint32_t i = 0; i < d.m_inner; i += 131) CHECK_EQ(host.small[sb.gnorm / 4 + i], common::bf16_to_f32(gn[i]));
    }
    if (d.is_moe(l)) {
      const uint32_t E = d.experts[l], I = d.inter[l];
      CHECK_EQ(host.experts.size() * 4, loader::bolt::layer_expert_bytes(d.latent, E, I));
      CHECK_EQ(host.router.size() * 2, loader::bolt::router_bytes(d.hidden, E));
      // the router tiles: (k, n) of the bf16 [E][hidden] tensor
      const uint16_t* gw = reinterpret_cast<const uint16_t*>(set.data(set.tensors().at(m + "gate.weight")));
      for (int s = 0; s < 400; ++s) {
        const uint32_t n = rng() % E, k = rng() % d.hidden;
        CHECK_EQ(host.router[loader::bolt::embed_row_index(d.hidden, k, n)], gw[size_t(n) * d.hidden + k]);
      }
      const float* eb = reinterpret_cast<const float*>(set.data(set.tensors().at(m + "gate.e_score_correction_bias")));
      for (uint32_t e = 0; e < E; ++e) CHECK_EQ(host.small[sb.e_bias / 4 + e], eb[e]);
      // sampled experts: up and down at the formula's offsets, against the raw tensors
      for (int s = 0; s < 6; ++s) {
        const uint32_t e = (s == 0) ? 0 : (s == 1) ? E - 1 : rng() % E;
        const Raw U = raw(set, m + "experts." + std::to_string(e) + ".up_proj"),
                  Dn = raw(set, m + "experts." + std::to_string(e) + ".down_proj");
        CHECK(U.K == d.latent && U.N == I && Dn.K == I && Dn.N == d.latent);
        const uint32_t* up = host.experts.data() + loader::bolt::expert_up_offset(d.latent, I, e) / 4;
        const uint32_t* dn = host.experts.data() + loader::bolt::expert_dn_offset(d.latent, I, e) / 4;
        for (int t = 0; t < 300; ++t) {
          uint32_t k = rng() % U.K, n = rng() % U.N;
          CHECK_EQ(tile_q(up, U.K, k, n), raw_q(U, k, n));
          CHECK_EQ(tile_s(up, U.K, k, n), raw_s(U, k, n));
          k = rng() % Dn.K;
          n = rng() % Dn.N;
          CHECK_EQ(tile_q(dn, Dn.K, k, n), raw_q(Dn, k, n));
          CHECK_EQ(tile_s(dn, Dn.K, k, n), raw_s(Dn, k, n));
          sampled += 2;
        }
      }
    }
  }
  // the embedding, the untied head and norm_f are read once
  const uint16_t* emb = ck.embed();
  const uint16_t* head = ck.lm_head();
  CHECK(emb != head);
  CHECK_EQ(ck.final_norm().size(), size_t(d.hidden));
  std::string names;
  CHECK_EQ(ck.unconsumed(&names), size_t(0));
  if (ck2) {
    ck2->embed();
    ck2->lm_head();
    ck2->final_norm();
    CHECK_EQ(ck2->unconsumed(&names), size_t(0));
  }
  CHECK_EQ(ck.group_mask(), qc.group_size == 128 ? 2u : 1u);
  std::printf("bolt_synth_host_test OK: %u blocks repacked, %zu sampled words equal the shards', 0 unconsumed%s\n", d.layers,
              sampled, ck2 ? ", both spellings give byte-identical blocks" : "");
  return 0;
}
