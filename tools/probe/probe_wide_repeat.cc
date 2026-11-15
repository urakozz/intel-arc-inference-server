// Open-work 28: is the one-card 16-row verify list deterministic? One engine (MTP head, int8 head, bf16 KV), a prompt
// prefilled, its state saved; then `runs` times: the state restored, the pending id and 15 drafts (the prompt's own
// ids after a copy point, so the rows are a plausible copy), verify(15) - and every run's verify ids, the hh rows the
// list wrote (the rows' post-final-norm hidden) and the head's K rows of the 16 positions compared bitwise with the
// first run's. Differences name the rows.
//   probe_wide_repeat <snapshot> <prompt.ids> [runs = 20] [max_len = 16384] [attn = rule|v2] [flush]   (cwd = the source tree)
// attn v2: the lists recaptured with the verify attention forced to v2 (CaptureOptions::force_attn) - which kernel
// a difference comes from.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
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
  if (argc > 5 && std::string(argv[5]) == "v2") {
    runtime::CaptureOptions o;
    o.force_attn = runtime::VerifyAttnForce::V2;
    e.recapture_verify(o);
    std::printf("WIDE_REPEAT: verify attention forced to v2\n");
  }
  e.prefill(ids);
  const uint32_t pos = e.pos();
  // `flip <t>` (7th, 8th arguments): which 16-row verifies emit output t differently from one-row decode - the plain
  // ids first, then for every start s = t - 15 .. t the wide verify from one-row decode's state at s, its row t - s.
  if (argc > 8 && std::string(argv[7]) == "flip") {
    const uint32_t t = std::stoul(argv[8]);
    l0::Mem st(ctx, l0::MemKind::Host, e.state_bytes());
    e.save_state(st.ptr());
    const uint32_t first = e.control()->cur_token[0];
    std::vector<uint32_t> out{first};
    for (uint32_t i = 0; i < t + 16; ++i) {
      e.verify(0);
      const uint32_t id = e.verify_ids()[0];
      e.commit(0, id);
      out.push_back(id);
    }
    for (uint32_t s0 = t >= 15 ? t - 15 : 0; s0 <= t; ++s0) {
      e.load_state(st.ptr(), pos);
      e.control()->cur_token[0] = first;
      for (uint32_t i = 0; i < s0; ++i) {
        e.verify(0);
        e.commit(0, out[i + 1]);
      }
      for (uint32_t i = 0; i < 15; ++i) e.control()->wide_cur[1 + i] = out[s0 + 1 + i];
      e.verify(runtime::Engine::kWideK);
      const uint32_t r = t - s0;
      std::printf("FLIP start %u row %u: wide %u, one-row %u%s\n", s0, r, e.verify_ids()[r], out[t + 1],
                  e.verify_ids()[r] == out[t + 1] ? "" : "  <<< differs");
    }
    return 0;
  }
  const uint32_t pending = e.control()->cur_token[0];
  l0::Mem saved(ctx, l0::MemKind::Host, e.state_bytes());
  e.save_state(saved.ptr());
  l0::Mem saved2(ctx, l0::MemKind::Host, e.state_bytes());
  // Drafts: the 15 prompt ids after the earliest occurrence of the pending id (a copy), else the prompt's first 15.
  std::vector<uint32_t> d(ids.begin(), ids.begin() + 15);
  for (size_t i = 0; i + 16 < ids.size(); ++i)
    if (ids[i] == pending) {
      d.assign(ids.begin() + i + 1, ids.begin() + i + 16);
      break;
    }
  const runtime::MtpBuffers& mb = *e.mtp_buffers();
  // The plain reference: 16 R one-row steps (verify(0) + commit) from the same state - their ids become the drafts of R
  // consecutive wide verifies (every one kept, each committed at 15), their hidden rows and the head's K rows of
  // positions pos - 1 .. pos + 16 R - 2 the comparison. `flush` (6th argument) flushes the deferred GDN rows between the
  // wide verifies (save_state does), to tell the replay of 16 rows from the rest.
  {
    const uint32_t R = 3;
    const bool flush = argc > 6 && std::string(argv[6]) == "flush";
    const size_t hid1 = size_t(e.model().desc->hidden) * 2, rb1 = mb.kv_lay.row_bytes();
    l0::CmdList im = l0::CmdList::immediate(ctx);
    e.load_state(saved.ptr(), pos);
    e.control()->cur_token[0] = pending;
    std::vector<uint8_t> ph(16 * R * hid1), pk(16 * R * rb1), wh(16 * R * hid1), wk(16 * R * rb1), row(17 * hid1);
    std::vector<uint32_t> pids;
    for (uint32_t i = 0; i < 16 * R; ++i) {
      e.verify(0);
      const uint32_t id = e.verify_ids()[0];
      im.copy(ph.data() + i * hid1, static_cast<const uint8_t*>(mb.hh.ptr()) + hid1, hid1);
      pids.push_back(id);
      e.commit(0, id);
    }
    im.copy(pk.data(), static_cast<const uint8_t*>(mb.kv_k.ptr()) + size_t(pos - 1) * rb1, pk.size());
    // The main cache's K and V rows of the 16 R positions, every FA layer.
    runtime::DecodeBuffers& db = e.buffers();
    const runtime::KvLayout& kl = db.kv_lay;
    const uint32_t fl = kl.layers;
    auto main_kv = [&](std::vector<uint8_t>& out) {
      out.resize(size_t(fl) * 2 * 16 * R * kl.row_bytes());
      for (uint32_t L = 0; L < fl; ++L)
        for (uint32_t kv = 0; kv < 2; ++kv)
          im.copy(out.data() + (size_t(L) * 2 + kv) * 16 * R * kl.row_bytes(),
                  static_cast<const uint8_t*>((kv ? db.kv_v : db.kv_k).ptr()) + kl.rows_offset(L) + size_t(pos) * kl.row_bytes(),
                  16 * R * kl.row_bytes());
    };
    std::vector<uint8_t> pkv, wkv;
    main_kv(pkv);
    // The same ids through the 4-row lists (verify(3) + commit(3), 4 ids a step): MTP's rows.
    std::vector<uint8_t> qh(16 * R * hid1);
    e.load_state(saved.ptr(), pos);
    e.control()->cur_token[0] = pending;
    for (uint32_t i = 0; i < 16 * R; i += 4) {
      for (uint32_t t = 0; t < 3; ++t) e.control()->cur_token[1 + t] = pids[i + t];
      e.verify(3);
      im.copy(qh.data() + i * hid1, static_cast<const uint8_t*>(mb.hh.ptr()) + hid1, 4 * hid1);
      e.commit(3, pids[i + 3]);
    }
    e.load_state(saved.ptr(), pos);
    e.control()->cur_token[0] = pending;
    uint32_t kept_all = 0;
    for (uint32_t v = 0; v < R; ++v) {
      if (v > 0 && flush) e.save_state(saved2.ptr());
      for (uint32_t i = 0; i < 15; ++i) e.control()->wide_cur[1 + i] = pids[16 * v + i];
      e.verify(runtime::Engine::kWideK);
      uint32_t kept = 0;
      while (kept < 15 && e.verify_ids()[kept] == pids[16 * v + kept]) ++kept;
      kept_all += kept;
      im.copy(row.data(), mb.hh.ptr(), row.size());
      std::memcpy(wh.data() + size_t(16 * v) * hid1, row.data() + hid1, 16 * hid1);
      e.commit(15, pids[16 * v + 15]);
    }
    im.copy(wk.data(), static_cast<const uint8_t*>(mb.kv_k.ptr()) + size_t(pos - 1) * rb1, wk.size());
    main_kv(wkv);
    auto cosine = [](const uint8_t* x, const uint8_t* y, size_t bytes) {
      double dot = 0, na = 0, nb = 0;
      for (size_t i = 0; i < bytes; i += 2) {
        uint16_t u, w;
        std::memcpy(&u, x + i, 2);
        std::memcpy(&w, y + i, 2);
        uint32_t a = uint32_t(u) << 16, c = uint32_t(w) << 16;
        float fa, fc;
        std::memcpy(&fa, &a, 4);
        std::memcpy(&fc, &c, 4);
        dot += double(fa) * fc;
        na += double(fa) * fa;
        nb += double(fc) * fc;
      }
      return dot / std::sqrt(std::max(na * nb, 1e-300));
    };
    std::printf("WIDE_VS_PLAIN %u wide verifies%s keep %u of %u plain ids\n", R, flush ? " (GDN flushed between)" : "",
                kept_all, 15 * R);
    for (uint32_t m = 0; m < 16 * R; ++m) {
      double worst = 1;
      uint32_t wl = 0, wkvi = 0;
      for (uint32_t L = 0; L < fl; ++L)
        for (uint32_t kv = 0; kv < 2; ++kv) {
          const size_t o = ((size_t(L) * 2 + kv) * 16 * R + m) * kl.row_bytes();
          const double cs = cosine(wkv.data() + o, pkv.data() + o, kl.row_bytes());
          if (cs < worst) { worst = cs; wl = L; wkvi = kv; }
        }
      if (worst < 0.999)
        std::printf("WIDE_VS_PLAIN verify %u row %2u: main %s row (position %u) worst FA layer %u cos %.6f  <<<\n", m / 16,
                    m % 16, wkvi ? "V" : "K", pos + m, wl, worst);
    }
    for (uint32_t m = 0; m < 16 * R; ++m) {
      const double ch = cosine(wh.data() + m * hid1, ph.data() + m * hid1, hid1);
      const double c4 = cosine(wh.data() + m * hid1, qh.data() + m * hid1, hid1);
      const double c41 = cosine(qh.data() + m * hid1, ph.data() + m * hid1, hid1);
      if (c4 < 0.999 || c41 < 0.999)
        std::printf("WIDE_VS_PLAIN verify %u row %2u: hidden cos wide-vs-4row %.6f, 4row-vs-1row %.6f  <<<\n", m / 16,
                    m % 16, c4, c41);
      const double ck = cosine(wk.data() + m * rb1, pk.data() + m * rb1, rb1);
      std::printf("WIDE_VS_PLAIN verify %u row %2u: hidden cos %.6f | head K row (position %u) cos %.6f%s\n", m / 16,
                  m % 16, ch, pos - 1 + m, ck, ch < 0.999 || ck < 0.999 ? "  <<<" : "");
    }
  }
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
