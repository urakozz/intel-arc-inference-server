#pragma once
// Spec 24c: an LFM2.5-24B-A2B engine on one or two cards for the card tests (lfm2moe_decode_test,
// lfm2moe_golden_test, lfm2moe_pp_test) - the CLI's flow (cli/lfm2moe_decode.h) without its argument parsing: the
// descriptor from the checkpoint, the placement, the load straight onto the devices (in_proj in the conv-tile
// layout exactly under the arm conv), the engine.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "l0/context.h"
#include "loader/lfm2moe_loader.h"
#include "loader/snapshot.h"
#include "runtime/lfm2moe/lfm2moe_engine.h"
#include "runtime/lfm2moe/lfm2moe_sizes.h"

namespace lfm2moe_rig {

struct Options {
  uint32_t max_len = 16384;
  bool int8_head = true;   // the CLI's default (spec 9's int8 head, built from the embedding)
  uint32_t devices = 1;
  uint32_t split = 0;      // 0 with two devices: runtime::lfm2moe::pp_split
  runtime::PpHandoff handoff = runtime::kDefaultPpHandoff;
  bool debug_tap = false;
  bool profile = false;
  uint32_t timeout_ms = 30000;
  runtime::lfm2moe::LfmAttn attn = runtime::lfm2moe::LfmAttn::Flash;
  runtime::lfm2moe::LfmFuse fuse{};
};

struct Rig {
  std::unique_ptr<l0::Context> c0, c1;
  std::vector<l0::Context*> ctx;
  std::unique_ptr<runtime::lfm2moe::Lfm2MoeEngine> eng;
};

// The snapshot `arg` resolves to when it is an LFM2.5-24B-A2B checkpoint, else "" (the caller SKIPs, 77).
inline std::string lfm2moe_snapshot(const std::string& arg) {
  try {
    const std::string s = loader::resolve_snapshot(arg);
    return loader::is_lfm2moe_checkpoint(s) ? s : "";
  } catch (const std::exception&) {
    return "";
  }
}

// Contexts are opened once per process and reused (the caller keeps the Rig alive across engines).
inline void open(Rig& r, uint32_t devices) {
  if (!r.c0) r.c0 = std::make_unique<l0::Context>(0u);
  if (devices == 2 && !r.c1) r.c1 = std::make_unique<l0::Context>(*r.c0, 1u);
  r.ctx = {r.c0.get()};
  if (devices == 2) r.ctx.push_back(r.c1.get());
}

inline void build(Rig& r, const std::string& snap, const Options& o) {
  r.eng.reset();
  open(r, o.devices);
  const model::Lfm2MoeDesc d = loader::lfm2moe_checkpoint_desc(snap);
  model::LfmPlacement p = model::LfmPlacement::one(d);
  if (o.devices == 2) p = model::LfmPlacement::two(d, o.split ? o.split : runtime::lfm2moe::pp_split(d, o.int8_head));
  loader::LfmLoadOptions lo;
  lo.int8_head = o.int8_head;
  lo.conv_tiles = o.fuse.conv;
  lo.max_len = o.max_len;
  loader::LfmLoaded m = loader::load_lfm2moe(r.ctx, snap, lo, p);
  runtime::PipelineOptions po;
  po.handoff = o.handoff;
  po.timeout_ms = o.timeout_ms;
  r.eng = std::make_unique<runtime::lfm2moe::Lfm2MoeEngine>(r.ctx, std::move(m), o.max_len, o.debug_tap, po, o.profile,
                                                            o.attn, o.fuse);
}

// The ids of a whitespace-separated file, each < vocab.
inline std::vector<uint32_t> read_ids(const std::string& p, uint32_t vocab) {
  std::vector<uint32_t> ids;
  FILE* f = std::fopen(p.c_str(), "r");
  if (!f) return ids;
  long long v;
  while (std::fscanf(f, "%lld", &v) == 1)
    if (v >= 0 && v < (long long)vocab) ids.push_back(uint32_t(v));
  std::fclose(f);
  return ids;
}

// The plan fits one B70 (32.530 GB as Level Zero reports it) with the default reserve.
inline bool fits_one_card(const model::Lfm2MoeDesc& d, bool int8_head, uint32_t max_len) {
  runtime::lfm2moe::Plan p = runtime::lfm2moe::plan(d, model::LfmPlacement::one(d), 0, max_len, int8_head);
  return p.total() <= size_t(32530000000ull);
}

}  // namespace lfm2moe_rig
