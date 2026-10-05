// Spec 7 C1 (plan 7b): Engine::save_state / load_state / save_kv / load_kv restore a
// session bitwise, and (Task 2) the block hook fires at every block end and at the
// prompt end, with block-aligned chunks.
//   A: snapshot at 4395 (4096 + 299), overwrite with a 3000-id other prompt (a different
//      pos % 16, stale KV above), restore, prefill id 4395, 64 greedy ids == straight run;
//      device KV [0, 4395) after the restore == the saved copy.
//   B: the same at 4096 (a block end).
//   A': A and B over a 6000-id other prompt: the restore goes below the current pos
//      with stale KV above it.
//   C: save_kv(4096, 4395) / load_kv round-trip into a zeroed cache, nothing written
//      outside the range; an empty range copies nothing.
//   D: hook set, prefill 5000 ids from 0 -> (2048,true), (4096,true), (5000,false).
//   E: hook set, prefill 4000 ids from 300 -> chunks 1748, 2048, 204; calls (2048,true),
//      (4096,true), (4300,false). A hook that throws at 2048 leaves pos == 2048.
//   F: the hook saves the state and the new KV range at every call; restoring the 4096
//      snapshot + KV [0,4096) and prefilling [4096,5000) gives the hooked run's first
//      id and 64 greedy ids, bitwise.
// Spec 12b: every case runs in the engine's KV form (B70_KV_CACHE=int8: snapshot_kv8_test).
// The host layout is save_kv's - per tensor (K, V) every layer's rows, then at int8 every
// layer's fp16 scales - so the runs below are KvLayout's, not 2 x fa_layers equal runs.
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "runtime/control.h"
#include "runtime/engine.h"

namespace {
constexpr uint32_t kMaxLen = 16384;

std::vector<uint32_t> repeat_to(const std::vector<uint32_t>& src, size_t n) {
  std::vector<uint32_t> out;
  while (out.size() < n) out.insert(out.end(), src.begin(), src.end());
  out.resize(n);
  return out;
}

std::vector<uint32_t> head(const std::vector<uint32_t>& v, size_t n) {
  return std::vector<uint32_t>(v.begin(), v.begin() + n);
}

bool same(const l0::Mem& a, const l0::Mem& b, size_t n) {
  return std::memcmp(a.ptr(), b.ptr(), n) == 0;
}

// The bytes per position of each run of save_kv's host layout, in order (no MTP head here):
// K rows of every layer, [K scales of every layer], V rows, [V scales]. bf16: 2 x fa_layers
// runs of 2048 B on Qwen3.8; int8: 4 x fa_layers runs, 1024 B and 8 B.
std::vector<size_t> run_units(runtime::Engine& e) {
  const runtime::KvLayout& L = e.buffers().kv_lay;
  std::vector<size_t> u;
  for (int kv = 0; kv < 2; ++kv) {
    for (uint32_t l = 0; l < L.layers; ++l) u.push_back(L.row_bytes());
    if (L.scale_row_bytes())
      for (uint32_t l = 0; l < L.layers; ++l) u.push_back(L.scale_row_bytes());
  }
  return u;
}

// C1 at restore point p over `ids` (size > p); `other` is the overwriting prompt.
void check_restore(l0::Context& ctx, runtime::Engine& e, const std::vector<uint32_t>& ids,
                   uint32_t p, const std::vector<uint32_t>& other) {
  l0::Mem st(ctx, l0::MemKind::Host, e.state_bytes());
  l0::Mem kv(ctx, l0::MemKind::Host, e.kv_bytes(p));
  l0::Mem kv2(ctx, l0::MemKind::Host, e.kv_bytes(p));
  const std::vector<uint32_t> next = {ids[p]};

  e.reset();
  e.prefill(head(ids, p));
  CHECK_EQ(e.pos(), p);
  e.save_state(st.ptr());
  e.save_kv(0, p, kv.ptr());
  e.prefill(next);
  const auto x = e.generate(64);

  e.reset();
  e.prefill(other);
  CHECK_EQ(e.pos(), uint32_t(other.size()));
  e.load_state(st.ptr(), p);
  e.load_kv(0, p, kv.ptr());
  CHECK_EQ(e.pos(), p);
  e.save_kv(0, p, kv2.ptr());
  CHECK(same(kv, kv2, e.kv_bytes(p)));
  e.prefill(next);
  const auto y = e.generate(64);
  size_t diff = 0;
  for (size_t i = 0; i < x.size(); ++i) diff += x[i] != y[i];
  std::printf("restore at %u over a state at %zu: KV [0,%u) exact, %zu/64 ids differ\n", p,
              other.size(), p, diff);
  std::fflush(stdout);
  CHECK(x == y);
}
}  // namespace

int main(int argc, char** argv) {
  const std::string checkpoint =
      argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  l0::Context ctx(0);
  runtime::Engine e(ctx, loader::load(ctx, checkpoint, kMaxLen), kMaxLen);
  e.prepare_prefill();
  const auto ids = golden::read_ids("tests/golden/prompts/long32k.ids");
  CHECK(ids.size() > 5000);
  const auto other = repeat_to(golden::read_ids("tests/golden/prompts/code.ids"), 3000);
  CHECK_EQ(e.state_bytes(), size_t(166723584));
  // 2048 positions x K and V x 16 FA layers x 4 kv heads x (256 x 2 B) = 134,217,728 at bf16;
  // x (256 + 2) B = 67,633,152 at int8 (spec 12b: the rows and their fp16 scales).
  CHECK_EQ(e.kv_bytes(2048), e.kv_cache() == runtime::KvCache::Int8
                                 ? size_t(2048) * 2 * 16 * 4 * (256 + 2)
                                 : size_t(134217728));

  // A and B.
  check_restore(ctx, e, ids, 4395, other);
  std::puts("case A OK");
  check_restore(ctx, e, ids, 4096, other);
  std::puts("case B OK");
  // A': restore BELOW the current pos, stale KV of another prompt above the restore
  // point (plan 7b review focus 1): 6000 other ids, restore at 4395 and at 4096.
  const auto longer = repeat_to(golden::read_ids("tests/golden/prompts/code.ids"), 6000);
  check_restore(ctx, e, ids, 4395, longer);
  check_restore(ctx, e, ids, 4096, longer);
  std::puts("case A' OK");

  // C: a range whose ends are not 2048-aligned at the top, into a zeroed cache.
  {
    e.reset();
    e.prefill(head(ids, 4395));
    const size_t n = e.kv_bytes(4395 - 4096);
    l0::Mem h1(ctx, l0::MemKind::Host, n), h2(ctx, l0::MemKind::Host, n);
    e.save_kv(4096, 4395, h1.ptr());
    bool nonzero = false;
    for (size_t i = 0; i < n && !nonzero; ++i) nonzero = h1.as<uint8_t>()[i] != 0;
    CHECK(nonzero);
    e.reset();
    e.load_kv(4096, 4395, h1.ptr());
    e.save_kv(4096, 4395, h2.ptr());
    CHECK(same(h1, h2, n));
    // Nothing outside [4096, 4395) was written: a whole-cache readback is zero there -
    // rows, and at int8 (spec 12b) the scales too.
    const auto& b = e.buffers();
    const runtime::KvLayout& L = b.kv_lay;
    std::vector<uint8_t> full(b.kv_k.size());
    l0::CmdList copy = l0::CmdList::immediate(ctx);
    for (const l0::Mem* m : {&b.kv_k, &b.kv_v}) {
      copy.copy(full.data(), m->ptr(), full.size());
      size_t outside = 0;
      for (uint32_t l = 0; l < L.layers; ++l)
        for (size_t p = 0; p < kMaxLen; ++p) {
          if (p >= 4096 && p < 4395) continue;
          const uint8_t* q = full.data() + L.rows_offset(l) + p * L.row_bytes();
          for (size_t i = 0; i < L.row_bytes(); ++i) outside += q[i] != 0;
          const uint8_t* s8 = full.data() + L.scales_offset(l) + p * L.scale_row_bytes();
          for (size_t i = 0; i < L.scale_row_bytes(); ++i) outside += s8[i] != 0;
        }
      CHECK_EQ(outside, size_t(0));
    }
    // Empty ranges: nothing is copied, a null host pointer is never touched.
    e.save_kv(100, 100, nullptr);
    e.load_kv(100, 100, nullptr);
    CHECK_EQ(e.kv_bytes(0), size_t(0));
    std::puts("case C OK");
  }

  using Calls = std::vector<std::pair<uint32_t, bool>>;
  Calls calls;
  const auto record = [&](uint32_t end, bool block) { calls.emplace_back(end, block); };
  // D
  e.set_block_hook(record);
  e.reset();
  e.prefill(head(ids, 5000));
  CHECK(calls == (Calls{{2048, true}, {4096, true}, {5000, false}}));
  CHECK_EQ(e.pos(), 5000u);
  std::puts("case D OK");
  // E
  calls.clear();
  e.reset();
  e.set_block_hook({});
  e.prefill(head(ids, 300));
  e.set_block_hook(record);
  e.prefill(std::vector<uint32_t>(ids.begin() + 300, ids.begin() + 4300));
  CHECK(calls == (Calls{{2048, true}, {4096, true}, {4300, false}}));
  CHECK_EQ(e.pos(), 4300u);
  e.reset();
  e.set_block_hook([&](uint32_t end, bool) {
    if (end == 2048) throw std::runtime_error("hook");
  });
  bool threw = false;
  try {
    e.prefill(head(ids, 5000));
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
  CHECK_EQ(e.pos(), 2048u);
  std::puts("case E OK");
  // F
  {
    std::vector<std::unique_ptr<l0::Mem>> states;
    std::vector<uint32_t> ends;
    // The KV of the whole run, gathered range by range into one [0, 5000) host layout:
    // save_kv's runs (run_units: 2 x fa_layers at bf16 - 32 on Qwen3.8, 36 on Agnes - and
    // twice that at int8) of 5000 positions each.
    const std::vector<size_t> units = run_units(e);
    std::vector<uint8_t> kv_all(e.kv_bytes(5000));
    uint32_t prev = 0;
    // Copy runs of [p0, p1) from a host block of `n` positions at `src` into `dst` of `m`
    // positions starting at q0 (both in save_kv's layout).
    const auto move_runs = [&](uint8_t* dst, size_t m, size_t q0, const uint8_t* src, size_t n) {
      size_t so = 0, d0 = 0;
      for (size_t u : units) {
        std::memcpy(dst + d0 + q0 * u, src + so, n * u);
        so += n * u;
        d0 += m * u;
      }
    };
    e.set_block_hook([&](uint32_t end, bool) {
      states.emplace_back(new l0::Mem(ctx, l0::MemKind::Host, e.state_bytes()));
      e.save_state(states.back()->ptr());
      ends.push_back(end);
      l0::Mem part(ctx, l0::MemKind::Host, e.kv_bytes(end - prev));
      e.save_kv(prev, end, part.ptr());
      move_runs(kv_all.data(), 5000, prev, part.as<uint8_t>(), end - prev);
      prev = end;
    });
    e.reset();
    e.prefill(head(ids, 5000));
    e.set_block_hook({});
    const uint32_t first = e.buffers().control.as<runtime::Control>()->cur_token[0];
    const auto x = e.generate(64);
    CHECK((ends == std::vector<uint32_t>{2048, 4096, 5000}));
    // Restore the 4096 snapshot and KV [0, 4096) over another prompt's state: the first
    // 4096 positions of every run of kv_all.
    l0::Mem kv(ctx, l0::MemKind::Host, e.kv_bytes(4096));
    {
      size_t so = 0, d0 = 0;
      for (size_t u : units) {
        std::memcpy(kv.as<uint8_t>() + d0, kv_all.data() + so, 4096 * u);
        so += 5000 * u;
        d0 += 4096 * u;
      }
    }
    e.reset();
    e.prefill(other);
    e.load_state(states[1]->ptr(), 4096);
    e.load_kv(0, 4096, kv.ptr());
    e.prefill(std::vector<uint32_t>(ids.begin() + 4096, ids.begin() + 5000));
    const uint32_t first2 = e.buffers().control.as<runtime::Control>()->cur_token[0];
    const auto y = e.generate(64);
    std::printf("F: first id %u vs %u, 64 greedy ids %s\n", first, first2,
                x == y ? "identical" : "differ");
    CHECK_EQ(first, first2);
    CHECK(x == y);
    std::puts("case F OK");
  }
  std::puts("snapshot_test OK");
}
