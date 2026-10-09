#pragma once
// Spec 21c: a Qwen3.8-Flash-Next engine on one or two cards for the card tests (qwen4exp_decode_test,
// qwen4exp_golden_test, qwen4exp_partial_test, qwen4exp_pp_test) - the CLI's flow (cli/qwen4exp_decode.h) without
// its argument parsing: the descriptor from the checkpoint, the placement, the load straight onto the devices
// (the PLE table pinned in host USM), the engine. kolibri_rig.h's arrangement.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "l0/context.h"
#include "loader/qwen4exp_loader.h"
#include "loader/qwen4exp_ple.h"
#include "loader/snapshot.h"
#include "runtime/qwen4exp/qwen4exp_engine.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace qwen4exp_rig {

struct Options {
  uint32_t max_len = 16384;
  bool int8_head = false;
  uint32_t devices = 1;
  uint32_t split = 0;   // 0 with two devices: runtime::qwen4exp::pp_split at max_len
  runtime::PpHandoff handoff = runtime::kDefaultPpHandoff;
  uint32_t layers = 0;  // development mode (0: the checkpoint's)
  bool debug_tap = false;
  uint32_t timeout_ms = 60000;
  std::string ple_dir;  // "" = loader::q4_ple_dir(snapshot)
};

struct Rig {
  std::unique_ptr<l0::Context> c0, c1;
  std::vector<l0::Context*> ctx;
  std::unique_ptr<runtime::qwen4exp::Qwen4ExpEngine> eng;
};

// The snapshot `arg` resolves to when it is a qwen4_exp checkpoint WITH its PLE int8 file, else "" (the caller
// SKIPs, 77) - `why` says what was missing.
inline std::string qwen4exp_snapshot(const std::string& arg, std::string* why = nullptr) {
  try {
    const std::string s = loader::resolve_snapshot(arg);
    if (!loader::is_qwen4exp_checkpoint(s)) {
      if (why) *why = s + " is not model_type qwen4_exp";
      return "";
    }
    loader::q4_ple_dir(s);   // throws naming the converter's command when the PLE file is absent
    return s;
  } catch (const std::exception& e) {
    if (why) *why = e.what();
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

inline model::Q4Placement placement(const model::Qwen4ExpDesc& d, const Options& o) {
  if (o.devices == 1) return model::Q4Placement::one(d);
  return model::Q4Placement::two(d, o.split ? o.split : runtime::qwen4exp::pp_split(d, o.max_len, o.int8_head, false).split);
}

inline void build(Rig& r, const std::string& snap, const Options& o) {
  r.eng.reset();
  open(r, o.devices);
  const model::Qwen4ExpDesc d = loader::qwen4exp_checkpoint_desc(snap, o.layers);
  const model::Q4Placement p = placement(d, o);
  loader::Q4LoadedModel m = loader::load_qwen4exp(r.ctx, snap, o.max_len, p,
                                                  o.int8_head ? loader::LmHeadForm::Int8 : loader::LmHeadForm::Checkpoint,
                                                  o.layers, false, o.ple_dir);
  runtime::PipelineOptions po;
  po.handoff = o.handoff;
  po.timeout_ms = o.timeout_ms;
  r.eng = std::make_unique<runtime::qwen4exp::Qwen4ExpEngine>(r.ctx, std::move(m), o.max_len, o.debug_tap, po);
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

// Every state a replay leaves that the next one reads or a gate compares - per layer, in layer order, whichever
// device holds it: the route rows, the selections and their diagnostics, every layer's persistent bytes (KV, the
// compressed keys and tail rings, the GDN state and conv rings, the PLE rings), the PLE ids.
struct State {
  std::vector<float> logits;
  std::vector<uint32_t> routes;
  std::vector<std::vector<uint32_t>> selections;   // per QSA layer
  std::vector<float> diag;
  std::vector<std::vector<uint8_t>> layers;
  std::vector<uint64_t> ple_ids;
  bool operator==(const State& o) const {
    return logits == o.logits && routes == o.routes && selections == o.selections && diag == o.diag &&
           layers == o.layers && ple_ids == o.ple_ids;
  }
};
inline State read_state(runtime::qwen4exp::Qwen4ExpEngine& e) {
  const model::Qwen4ExpDesc& d = e.model().desc;
  State s;
  s.logits = e.read_logits();
  s.routes = e.read_routes();
  for (uint32_t l = 0; l < d.layers; ++l)
    if (d.is_qsa(l)) s.selections.push_back(e.read_selection(l));
  s.diag = e.read_selection_diag();
  for (uint32_t l = 0; l < d.layers; ++l) s.layers.push_back(e.read_layer_state(l));
  s.ple_ids = e.read_ple_ids();
  return s;
}
// The first part of two States that differs, or "" when none does.
inline std::string first_difference(const State& a, const State& b) {
  if (a.logits != b.logits) return "logits";
  if (a.routes != b.routes) return "route rows";
  for (size_t i = 0; i < a.selections.size() && i < b.selections.size(); ++i)
    if (a.selections[i] != b.selections[i]) return "the selection of QSA layer #" + std::to_string(i);
  if (a.diag != b.diag) return "the selection diagnostics";
  for (size_t l = 0; l < a.layers.size() && l < b.layers.size(); ++l)
    if (a.layers[l] != b.layers[l]) return "layer " + std::to_string(l) + "'s persistent state";
  if (a.ple_ids != b.ple_ids) return "the PLE ids";
  return a == b ? "" : "the shapes";
}

}  // namespace qwen4exp_rig
