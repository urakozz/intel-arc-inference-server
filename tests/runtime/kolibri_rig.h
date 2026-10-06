#pragma once
// Spec 20c: a Kolibri-1 engine on one or two cards for the card tests (kolibri_decode_test,
// kolibri_golden_test, kolibri_partial_test, kolibri_pp_test) - the CLI's flow (cli/kolibri_decode.h)
// without its argument parsing: the descriptor from the checkpoint, the placement, the load straight
// onto the devices, the engine.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "l0/context.h"
#include "loader/kolibri1_loader.h"
#include "loader/snapshot.h"
#include "runtime/kolibri/kolibri_engine.h"
#include "runtime/kolibri/kolibri_sizes.h"

namespace kolibri_rig {

struct Options {
  uint32_t max_len = 16384;
  bool int8_head = false;
  uint32_t devices = 1;
  uint32_t split = 0;   // 0 with two devices: runtime::kolibri::pp_split at max_len
  runtime::PpHandoff handoff = runtime::kDefaultPpHandoff;
  uint32_t layers = 0;  // development mode
  bool debug_tap = false;
};

struct Rig {
  std::unique_ptr<l0::Context> c0, c1;
  std::vector<l0::Context*> ctx;
  std::unique_ptr<runtime::kolibri::KolibriEngine> eng;
};

// The snapshot `arg` resolves to when it is a Kolibri-1 checkpoint, else "" (the caller SKIPs, 77).
inline std::string kolibri_snapshot(const std::string& arg) {
  try {
    const std::string s = loader::resolve_snapshot(arg);
    return loader::is_kolibri1_checkpoint(s) ? s : "";
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
  const model::Kolibri1Desc d = loader::kolibri1_checkpoint_desc(snap, o.layers);
  model::KolPlacement p = model::KolPlacement::one(d);
  if (o.devices == 2)
    p = model::KolPlacement::two(d, o.split ? o.split : runtime::kolibri::pp_split(d, o.max_len, o.int8_head, o.handoff).split);
  loader::KolLoadedModel m = loader::load_kolibri1(r.ctx, snap, o.max_len, p,
                                                   o.int8_head ? loader::LmHeadForm::Int8 : loader::LmHeadForm::Checkpoint,
                                                   o.layers);
  runtime::PipelineOptions po;
  po.handoff = o.handoff;
  r.eng = std::make_unique<runtime::kolibri::KolibriEngine>(r.ctx, std::move(m), o.max_len, o.debug_tap, po);
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

}  // namespace kolibri_rig
