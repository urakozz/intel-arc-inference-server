// Spec 23b Task 4: the loader's host half over a checkpoint on disk - make_synth.py's synthetic one (Review Focus 5:
// the Python writer and the C++ reader held to each other on real files) or the real int4 checkpoint (either
// Vishva007 spelling) - no device. Host only; label checkpoint (the files live on the box / the Mac by hand).
//   muse_synth_host_test <snapshot dir | repo id> [layers]
// config.json -> check_muse_config (the served constructor) at the checkpoint's own layer count; the quantisation
// config; the names both ways (muse_expected_names of the spelling found); the data asserts; every layer (or the
// first `layers`) repacked, its bytes = loader::muse_layer_bytes and Review Focus 1's columns read back against the
// shards; the head, the embedding and the final norm; 0 unconsumed; the plan of that descriptor on one 32.53 GB card.
// Exit 77 (SKIP) when the directory has no config.json.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "loader/muse_layout.h"
#include "loader/muse_repack.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"
#include "runtime/muse/muse_sizes.h"

int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  std::string dir = argc > 1 ? argv[1] : "oracle-out-muse-synth/ckpt4";
  const uint32_t layers = argc > 2 ? uint32_t(std::stoul(argv[2])) : 0;
  if (!fs::exists(fs::path(dir) / "config.json")) {   // a repo id: the HF cache's snapshot
    try {
      dir = loader::resolve_snapshot(dir);
    } catch (const std::exception&) {
    }
  }
  if (!fs::exists(fs::path(dir) / "config.json")) {
    std::printf("SKIP: no checkpoint at %s (tools/box_validate/muse_oracle.sh <data> synth-ckpt; box: r43.synth)\n",
                dir.c_str());
    return 77;
  }
  const auto t0 = std::chrono::steady_clock::now();
  const loader::MuseCheckpoint ck(dir, layers);
  const model::MuseGlimmerDesc& d = ck.desc();
  const double t_open = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  %s: %zu tensors, %u of %u layers, %s spelling, %zu text + %zu skipped by --layers + %zu vision "
              "(skipped by name), %zu subnormal f16 scales; names and data asserts %.1f s\n",
              dir.c_str(), ck.tensors(), d.layers, ck.checkpoint_layers(), loader::muse_quant_form_name(ck.form()),
              ck.text_tensors(), ck.skipped_layer_tensors(), ck.vision_skipped(), ck.subnormal_scales(), t_open);
  CHECK_EQ(ck.text_tensors() + ck.skipped_layer_tensors() + ck.vision_skipped(), ck.tensors());
  {
    model::MuseGlimmerDesc full = d;
    full.layers = ck.checkpoint_layers();
    full.full = model::muse_glimmer().full;
    full.full.resize(full.layers);
    CHECK_EQ(ck.text_tensors() + ck.skipped_layer_tensors(), loader::muse_expected_names(full, ck.form()).size());
  }

  const loader::SafetensorsSet set(ck.snapshot());
  loader::MuseHostLayer h;
  const loader::MuseLayerBytes want = loader::muse_layer_bytes(d);
  for (uint32_t l = 0; l < d.layers; ++l) {
    ck.layer(l, h);
    CHECK_EQ(h.bytes(), want.total());
    // Review Focus 1 at the fused GEMVs' part boundaries: the device-form words / scales are the source's.
    const loader::MuseHostLinear* hl[] = {&h.qgkv, &h.oproj, &h.gate_up, &h.down};
    const std::vector<model::MuseLinearId> ids = d.layer_linears();
    for (size_t i = 0; i < ids.size(); ++i) {
      const model::MuseLinear ln = d.linear(ids[i]);
      uint32_t at = 0;
      for (size_t p = 0; p < ln.parts.size(); ++p) {
        const std::string base = model::MuseGlimmerDesc::layer_prefix(l) + ln.parts[p];
        const uint32_t* qw = reinterpret_cast<const uint32_t*>(set.data(set.tensors().at(base + ".qweight")));
        const uint16_t* sc = reinterpret_cast<const uint16_t*>(set.data(set.tensors().at(base + ".scales")));
        const uint32_t Np = ln.part_n[p], N = ln.shape.N, K = ln.shape.K;
        for (uint32_t j : {0u, Np - 1}) {
          for (uint32_t r = 0; r < K / 8; ++r) CHECK_EQ(hl[i]->words[size_t(r) * N + at + j], qw[size_t(r) * Np + j]);
          for (uint32_t g = 0; g < K / 64; ++g) CHECK_EQ(hl[i]->scales[size_t(g) * N + at + j], sc[size_t(g) * Np + j]);
        }
        at += Np;
      }
    }
    const uint16_t* w = reinterpret_cast<const uint16_t*>(
        set.data(set.tensors().at(model::MuseGlimmerDesc::layer_prefix(l) + "post_feedforward_layernorm.weight")));
    for (uint32_t k = 0; k < d.hidden; ++k) {
      const float f = common::bf16_to_f32(w[k]);
      CHECK(std::memcmp(&f, &h.post_ff[k], 4) == 0);
    }
  }
  (void)ck.embed();
  (void)ck.lm_head();
  CHECK_EQ(ck.final_norm().size(), size_t(d.hidden));
  const std::vector<std::string> left = ck.unconsumed();
  for (size_t i = 0; i < left.size() && i < 5; ++i) std::fprintf(stderr, "unconsumed: %s\n", left[i].c_str());
  CHECK(left.empty());
  const runtime::muse::MusePlan plan = runtime::muse::plan_muse(d, 1, 0, true, {}, {32530000000ull});
  runtime::muse::require_fits(plan);
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s\n", plan.describe().c_str());
  std::printf("muse_synth_host_test OK: %u layers repacked (%zu B a layer = muse_layer_bytes), 0 unconsumed, %.1f s\n",
              d.layers, want.total(), secs);
  return 0;
}
