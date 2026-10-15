#pragma once
// Spec 23c: a Muse-Glimmer engine on one or two cards for the card tests (muse_decode_test, muse_golden_test,
// muse_partial_test, muse_pp_test) - the CLI's flow (cli/muse_decode.h) without its argument parsing. MuseEngine
// plans and loads itself; the rig opens the contexts once per process and reuses them.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "l0/context.h"
#include "loader/muse_loader.h"
#include "loader/snapshot.h"
#include "runtime/muse/muse_engine.h"
#include "runtime/muse/muse_sizes.h"

namespace muse_rig {

struct Rig {
  std::unique_ptr<l0::Context> c0, c1;
  std::vector<l0::Context*> ctx;
  std::unique_ptr<runtime::muse::MuseEngine> eng;
};

// The snapshot `arg` resolves to when it is a Muse-Glimmer checkpoint, else "" (the caller SKIPs, 77).
inline std::string muse_snapshot(const std::string& arg) {
  try {
    const std::string s = loader::resolve_muse_snapshot(arg);   // either Vishva007 spelling
    return loader::is_muse_checkpoint(s) ? s : "";
  } catch (const std::exception&) {
    return "";
  }
}

inline void open(Rig& r, uint32_t devices) {
  if (!r.c0) r.c0 = std::make_unique<l0::Context>(0u);
  if (devices == 2 && !r.c1) r.c1 = std::make_unique<l0::Context>(*r.c0, 1u);
  r.ctx = {r.c0.get()};
  if (devices == 2) r.ctx.push_back(r.c1.get());
}

inline void build(Rig& r, const std::string& snap, uint32_t devices, const runtime::muse::MuseOptions& o) {
  r.eng.reset();
  open(r, devices);
  r.eng = std::make_unique<runtime::muse::MuseEngine>(r.ctx, snap, o);
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

// The smallest multiple of 256 holding `n` positions, at least 4096 (the planner's grid).
inline uint32_t len_for(size_t n) {
  const uint32_t q = uint32_t((n + 255) / 256 * 256);
  return q < 4096 ? 4096 : q;
}

}  // namespace muse_rig
