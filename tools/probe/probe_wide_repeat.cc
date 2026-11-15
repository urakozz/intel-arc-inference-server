// Open-work 28: is the one-card 16-row verify list deterministic? One engine (MTP head, int8 head, bf16 KV), a prompt
// prefilled, its state saved; then `runs` times: the state restored, the pending id and 15 drafts (the prompt's own
// ids after a copy point, so the rows are a plausible copy), verify(15) - and every run's verify ids, the hh rows the
// list wrote (the rows' post-final-norm hidden) and the head's K rows of the 16 positions compared bitwise with the
// first run's. Differences name the rows.
//   probe_wide_repeat <snapshot> <prompt.ids> [runs = 20] [max_len = 16384]   (cwd = the source tree)
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/control.h"
#include "runtime/engine.h"

namespace {
std::vector<uint32_t> read_ids(const std::string& p) {
  std::vector<uint32_t> v;
  FILE* f = std::fopen(p.c_str(), "r");
  if (!f) return v;
  unsigned x;
  while (std::fscanf(f, "%u", &x) == 1) v.push_back(x);
  std::fclose(f);
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <snapshot> <prompt.ids> [runs] [max_len]\n", argv[0]);
    return 2;
  }
  const std::vector<uint32_t> ids = read_ids(argv[2]);
  const uint32_t runs = argc > 3 ? std::stoul(argv[3]) : 20;
  const uint32_t max_len = argc > 4 ? std::stoul(argv[4]) : 16384;
  if (ids.size() < 64 || ids.size() + 64 > max_len) return 2;
  l0::Context ctx(0);
  runtime::Engine e(ctx, loader::load(ctx, argv[1], max_len, /*mtp=*/true, loader::LmHeadForm::Int8), max_len);
  if (!e.wide_built()) {
    std::fprintf(stderr, "probe_wide_repeat: the engine has no wide verify list\n");
    return 2;
  }
  e.prefill(ids);
  const uint32_t pos = e.pos();
  const uint32_t pending = e.control()->cur_token[0];
  l0::Mem saved(ctx, l0::MemKind::Host, e.state_bytes());
  e.save_state(saved.ptr());
  // Drafts: the 15 prompt ids after the earliest occurrence of the pending id (a copy), else the prompt's first 15.
  std::vector<uint32_t> d(ids.begin(), ids.begin() + 15);
  for (size_t i = 0; i + 16 < ids.size(); ++i)
    if (ids[i] == pending) {
      d.assign(ids.begin() + i + 1, ids.begin() + i + 16);
      break;
    }
  const runtime::MtpBuffers& mb = *e.mtp_buffers();
  const size_t hid = size_t(e.model().desc->hidden) * 2, rb = mb.kv_lay.row_bytes();
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  std::vector<uint8_t> hh0, k0;
  std::vector<uint32_t> v0;
  uint32_t bad_runs = 0;
  for (uint32_t r = 0; r < runs; ++r) {
    e.load_state(saved.ptr(), pos);
    e.control()->cur_token[0] = pending;
    for (uint32_t i = 0; i < 15; ++i) e.control()->wide_cur[1 + i] = d[i];
    e.verify(runtime::Engine::kWideK);
    std::vector<uint32_t> v(e.verify_ids(), e.verify_ids() + 16);
    std::vector<uint8_t> hh(17 * hid), k(16 * rb);
    imm.copy(hh.data(), mb.hh.ptr(), hh.size());
    imm.copy(k.data(), static_cast<const uint8_t*>(mb.kv_k.ptr()) + size_t(pos - 1) * rb, k.size());
    if (r == 0) {
      hh0 = hh;
      k0 = k;
      v0 = v;
      uint32_t kept = 0;
      while (kept < 15 && v[kept] == d[kept]) ++kept;
      std::printf("WIDE_REPEAT pos %u pending %u: run 0 keeps %u of 15 drafts\n", pos, pending, kept);
      continue;
    }
    std::string rows_h, rows_k;
    for (uint32_t m = 1; m <= 16; ++m)
      if (std::memcmp(hh.data() + m * hid, hh0.data() + m * hid, hid)) rows_h += " " + std::to_string(m - 1);
    for (uint32_t m = 0; m < 16; ++m)
      if (std::memcmp(k.data() + m * rb, k0.data() + m * rb, rb)) rows_k += " " + std::to_string(m);
    const bool same = rows_h.empty() && rows_k.empty() && v == v0;
    bad_runs += !same;
    if (!same)
      std::printf("WIDE_REPEAT run %u: hidden rows differ:%s | head K rows differ:%s | ids %s\n", r,
                  rows_h.empty() ? " none" : rows_h.c_str(), rows_k.empty() ? " none" : rows_k.c_str(),
                  v == v0 ? "same" : "DIFFER");
  }
  std::printf("WIDE_REPEAT %s: %u of %u runs differ from run 0\n", bad_runs ? "NONDETERMINISTIC" : "bitwise",
              bad_runs, runs - 1);
  return bad_runs ? 1 : 0;
}
