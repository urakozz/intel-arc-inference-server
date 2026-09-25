// flash_long_test - spec 6 K3, long context end to end (plan 6b Tasks 4 and 5).
//
//   flash_long_test <snapshot> <long32k.ids>
//       K3a: max_len 32768 (a compiled decode variant; the prompt plus 64 fills it exactly,
//       and pad256 of the prefill depth, 32704, is 32768, so the composed path fits). Prefill the prompt minus its last 64 ids in FLASH mode
//       (pf_flash_attn), generate 64 greedy tokens; then the same in a fresh engine in
//       COMPOSED mode (B70_PREFILL_ATTN=composed's path, selected here through
//       set_attn_mode_for_test). Bars: the same 64 greedy tokens, and the prefill's
//       last-row logits cosine >= 0.9999. Prints the cosine and the first divergence.
//   flash_long_test <snapshot> --128k <ids>
//       K3b: max_len 131072, flash mode (not registered; run by hand). The prompt (131000
//       ids, tools/probe/mk_long_ids.py) prefilled in two fresh engines: last-row logits
//       bitwise equal and finite. Then prefill replay (set_prefill_replay(true)) twice:
//       both bitwise equal to the immediate run.
//
// There is no CPU oracle at these depths; the composed path is the reference at 32k
// (it still fits there) and determinism / replay carry the 128k gate.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/prefill/attn.h"

namespace {
using runtime::prefill::AttnMode;

struct Run {
  std::vector<float> logits;     // the prefill's last-row logits
  uint32_t first = 0;            // cur_token after prefill
  std::vector<uint32_t> gen;     // greedy tokens after the first (may be empty)
  double prefill_ms = 0;
};

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

Run prefill_once(runtime::Engine& e, l0::Context& ctx, const std::vector<uint32_t>& ids,
                 uint32_t n_gen) {
  Run r;
  e.reset();
  const double t0 = now_ms();
  e.prefill(ids);
  r.prefill_ms = now_ms() - t0;
  const l0::Mem& lg = e.prefill_scratch()->logits;
  r.logits.resize(lg.size() / sizeof(float));
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(r.logits.data(), lg.ptr(), lg.size());
  r.first = e.buffers().control.as<runtime::Control>()->cur_token[0];
  if (n_gen) r.gen = e.generate(n_gen);
  return r;
}

size_t nonfinite(const std::vector<float>& v) {
  size_t n = 0;
  for (float x : v) n += !std::isfinite(x);
  return n;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += double(a[i]) * b[i];
    na += double(a[i]) * a[i];
    nb += double(b[i]) * b[i];
  }
  return dot / std::sqrt(na * nb);
}

int k3a(const std::string& snap, const std::string& ids_path) {
  const std::vector<uint32_t> all = golden::read_ids(ids_path);
  CHECK(all.size() > 64);
  const std::vector<uint32_t> prompt(all.begin(), all.end() - 64);
  const uint32_t max_len = 32768;
  CHECK(prompt.size() + 64 <= max_len);
  std::printf("K3a: %zu prompt ids (%s minus its last 64), 64 greedy tokens, max_len %u\n",
              prompt.size(), ids_path.c_str(), max_len);
  l0::Context ctx(0);
  Run runs[2];
  const AttnMode modes[2] = {AttnMode::Flash, AttnMode::Composed};
  for (int i = 0; i < 2; ++i) {
    runtime::prefill::set_attn_mode_for_test(modes[i]);
    runtime::Engine e(ctx, loader::load(ctx, snap, max_len), max_len);
    // generate(64) returns the prefill's own argmax (cur_token) first, then 63 more.
    runs[i] = prefill_once(e, ctx, prompt, 64);
    CHECK_EQ(runs[i].gen.size(), size_t(64));
    CHECK_EQ(runs[i].gen[0], runs[i].first);
    std::printf("  %-8s (%s): prefill %.1f ms, logits non-finite %zu, first token %u\n",
                runtime::prefill::attn_mode_name(modes[i]),
                runtime::prefill_backend_name(e.prefill_backend()), runs[i].prefill_ms,
                nonfinite(runs[i].logits), runs[i].first);
    std::fflush(stdout);
  }
  const double cs = cosine(runs[0].logits, runs[1].logits);
  size_t div = runs[0].gen.size();
  for (size_t i = 0; i < runs[0].gen.size(); ++i)
    if (runs[0].gen[i] != runs[1].gen[i]) { div = i; break; }
  std::printf("  last-row logits cosine flash vs composed: %.9f (bar 0.9999)\n", cs);
  if (div == runs[0].gen.size())
    std::printf("  greedy tokens: all 64 equal (first divergence: none)\n");
  else
    std::printf("  greedy tokens: FIRST DIVERGENCE at %zu: flash %u, composed %u\n", div,
                runs[0].gen[div], runs[1].gen[div]);
  std::printf("  flash tokens:");
  for (uint32_t t : runs[0].gen) std::printf(" %u", t);
  std::printf("\n");
  CHECK_EQ(nonfinite(runs[0].logits), size_t(0));
  CHECK_EQ(nonfinite(runs[1].logits), size_t(0));
  CHECK(cs >= 0.9999);
  CHECK_EQ(div, runs[0].gen.size());
  std::puts("flash_long_test OK -- K3a: flash == composed at 32k (64 greedy tokens)");
  return 0;
}

int k3b(const std::string& snap, const std::string& ids_path) {
  const std::vector<uint32_t> prompt = golden::read_ids(ids_path);
  const uint32_t max_len = 131072;
  CHECK(prompt.size() < max_len);
  runtime::prefill::set_attn_mode_for_test(AttnMode::Flash);
  std::printf("K3b: %zu prompt ids (%s), max_len %u, flash\n", prompt.size(), ids_path.c_str(),
              max_len);
  l0::Context ctx(0);
  Run a, b, r1, r2;
  {
    runtime::Engine e(ctx, loader::load(ctx, snap, max_len), max_len);
    e.prepare_prefill();
    std::printf("  %s\n", e.memory_line().c_str());
    e.set_prefill_replay(false);
    a = prefill_once(e, ctx, prompt, 0);
    std::printf("  engine 1 immediate: prefill %.1f ms (%.2f t/s), first token %u, logits"
                " non-finite %zu\n", a.prefill_ms, prompt.size() * 1000.0 / a.prefill_ms,
                a.first, nonfinite(a.logits));
    std::fflush(stdout);
  }
  {
    runtime::Engine e(ctx, loader::load(ctx, snap, max_len), max_len);
    e.set_prefill_replay(false);
    b = prefill_once(e, ctx, prompt, 0);
    std::printf("  engine 2 immediate: prefill %.1f ms, first token %u, bitwise == engine 1:"
                " %s\n", b.prefill_ms, b.first, a.logits == b.logits ? "yes" : "NO");
    std::fflush(stdout);
    e.set_prefill_replay(true);
    r1 = prefill_once(e, ctx, prompt, 0);
    r2 = prefill_once(e, ctx, prompt, 0);
    std::printf("  engine 2 replay x2: %.1f / %.1f ms, bitwise == immediate: %s / %s\n",
                r1.prefill_ms, r2.prefill_ms, r1.logits == a.logits ? "yes" : "NO",
                r2.logits == a.logits ? "yes" : "NO");
  }
  CHECK_EQ(nonfinite(a.logits), size_t(0));
  CHECK(a.logits == b.logits && a.first == b.first);
  CHECK(r1.logits == a.logits && r1.first == a.first);
  CHECK(r2.logits == a.logits && r2.first == a.first);
  std::puts("flash_long_test OK -- K3b: 128k determinism and replay, bitwise");
  return 0;
}

// Diagnostic (not a gate): flash vs composed, and yardsticks for what a harmless
// perturbation costs at the same depth. One engine at max_len 32768, modes switched in
// place (prefill_smoke_test's Review Focus pins that as bitwise equal to fresh engines).
int sweep(const std::string& snap, const std::string& ids_path) {
  const std::vector<uint32_t> all = golden::read_ids(ids_path);
  const uint32_t max_len = 32768;
  l0::Context ctx(0);
  runtime::Engine e(ctx, loader::load(ctx, snap, max_len), max_len);
  struct Cfg { AttnMode m; runtime::PrefillBackend b; uint32_t chunk; };
  auto run = [&](const Cfg& c, uint32_t n) {
    runtime::prefill::set_attn_mode_for_test(c.m);
    e.set_prefill_backend(c.b);
    e.reset();
    const std::vector<uint32_t> ids(all.begin(), all.begin() + n);
    e.prefill(ids, c.chunk);
    std::vector<float> lg(e.prefill_scratch()->logits.size() / 4);
    l0::CmdList imm = l0::CmdList::immediate(ctx);
    imm.copy(lg.data(), e.prefill_scratch()->logits.ptr(), lg.size() * 4);
    return lg;
  };
  auto argmax = [](const std::vector<float>& v) {
    return size_t(std::max_element(v.begin(), v.end()) - v.begin());
  };
  auto cmp = [&](const char* what, uint32_t n, const Cfg& a, const Cfg& b) {
    const auto la = run(a, n), lb = run(b, n);
    std::printf("  n %5u  %-48s cos %.9f  argmax %zu / %zu\n", n, what, cosine(la, lb),
                argmax(la), argmax(lb));
    std::fflush(stdout);
  };
  using B = runtime::PrefillBackend;
  const Cfg fi{AttnMode::Flash, B::L0Int8, 2048}, ci{AttnMode::Composed, B::L0Int8, 2048};
  const Cfg fl{AttnMode::Flash, B::L0, 2048}, cl{AttnMode::Composed, B::L0, 2048};
  for (uint32_t n : {2048u, 4096u, 8192u, 16384u, 32704u}) {
    cmp("flash vs composed, l0-int8", n, fi, ci);
    cmp("flash vs composed, l0 (bf16 linears)", n, fl, cl);
  }
  const uint32_t n = 32704;
  cmp("YARDSTICK composed l0-int8, chunk 2048 vs 1024", n, ci, Cfg{AttnMode::Composed, B::L0Int8, 1024});
  cmp("YARDSTICK composed l0, chunk 2048 vs 1024", n, cl, Cfg{AttnMode::Composed, B::L0, 1024});
  cmp("YARDSTICK flash l0, chunk 2048 vs 1024", n, fl, Cfg{AttnMode::Flash, B::L0, 1024});
  cmp("YARDSTICK composed, l0 vs l0-int8", n, cl, ci);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: flash_long_test <snapshot> <long32k.ids> | <snapshot> --128k <ids>\n");
    return 2;
  }
  std::string snap;
  try {
    snap = loader::resolve_snapshot(argv[1]);
  } catch (const std::exception& e) {
    std::printf("SKIP: the checkpoint is not here (%s)\n", e.what());
    return 77;
  }
  if (std::string(argv[2]) == "--sweep" && argc > 3) return sweep(snap, argv[3]);
  if (std::string(argv[2]) == "--128k") {
    if (argc < 4) return 2;
    return k3b(snap, argv[3]);
  }
  return k3a(snap, argv[2]);
}
