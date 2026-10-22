#pragma once
// Spec 25c: a Nemotron 3.5 Bolt engine on one or two cards for the card tests (bolt_decode_test, bolt_golden_test) - the
// CLI's flow (cli/bolt_decode.h) without its argument parsing: the descriptor from the checkpoint (--layers N truncates),
// the placement, the load straight onto the devices, the engine.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "l0/context.h"
#include "loader/bolt_loader.h"
#include "loader/snapshot.h"
#include "runtime/bolt/bolt_engine.h"
#include "runtime/bolt/bolt_sizes.h"

namespace bolt_rig {

struct Options {
  uint32_t max_len = 16384;
  bool int8_head = true;   // the CLI's default (Decision 18)
  uint32_t devices = 1;
  uint32_t split = 0;      // 0 with two devices: runtime::bolt::pp_split
  uint32_t layers = 0;     // --layers N (0: the checkpoint's)
  runtime::PpHandoff handoff = runtime::kDefaultPpHandoff;
  bool debug_tap = false;
  bool profile = false;
  uint32_t timeout_ms = 60000;
  runtime::bolt::BoltAttn attn = runtime::bolt::BoltAttn::Flash;
  runtime::bolt::BoltFuse fuse{};
  runtime::bolt::BoltState state = runtime::bolt::BoltState::Fp32;
};

struct Rig {
  std::unique_ptr<l0::Context> c0, c1;
  std::vector<l0::Context*> ctx;
  std::unique_ptr<runtime::bolt::BoltEngine> eng;
};

// The snapshot `arg` resolves to when it is a Bolt checkpoint, else "" (the caller SKIPs, 77).
inline std::string bolt_snapshot(const std::string& arg) {
  try {
    const std::string s = loader::resolve_snapshot(arg);
    return loader::is_bolt_checkpoint(s) ? s : "";
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

inline model::BoltPlacement placement(const model::BoltDesc& d, const Options& o) {
  if (o.devices == 2) return model::BoltPlacement::two(d, o.split ? o.split : runtime::bolt::pp_split(d, o.int8_head));
  return model::BoltPlacement::one(d);
}

inline void build(Rig& r, const std::string& snap, const Options& o) {
  r.eng.reset();
  open(r, o.devices);
  const model::BoltDesc d = loader::bolt_checkpoint_desc(snap, o.layers);
  loader::BoltLoadOptions lo;
  lo.int8_head = o.int8_head;
  lo.max_len = o.max_len;
  lo.layers = o.layers;
  loader::BoltLoaded m = loader::load_bolt(r.ctx, snap, lo, placement(d, o));
  runtime::PipelineOptions po;
  po.handoff = o.handoff;
  po.timeout_ms = o.timeout_ms;
  r.eng = std::make_unique<runtime::bolt::BoltEngine>(r.ctx, std::move(m), o.max_len, o.debug_tap, po, o.profile, o.attn,
                                                      o.fuse, o.state);
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

// The plan (with the decode scratch, and with `prefill` spec 25d's prefill scratch and link) fits the placement's cards (a
// B70: 32.530 GB as Level Zero reports it).
inline bool fits(const model::BoltDesc& d, const Options& o, bool prefill = false) {
  try {
    const model::BoltPlacement p = placement(d, o);
    runtime::bolt::BoltReserve r;
    for (uint32_t dev = 0; dev < p.devices; ++dev) {
      r.decode_scratch = std::max(r.decode_scratch,
                                  runtime::bolt::decode_scratch_bytes(d, p, dev, o.max_len, o.attn, o.state));
      if (prefill) r.prefill_scratch = std::max(r.prefill_scratch, runtime::bolt::prefill_scratch_bytes(d, p, dev));
    }
    r.bf16_state = o.state == runtime::bolt::BoltState::Bf16;
    runtime::bolt::plan(d, o.max_len, std::vector<size_t>(p.devices, size_t(32530000000ull)), r,
                        p.devices == 2 ? std::optional<uint32_t>(p.split) : std::nullopt,
                        size_t(runtime::kDefaultReserveGb * 1e9), o.int8_head);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace bolt_rig
