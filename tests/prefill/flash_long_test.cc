// flash_long_test - spec 6 K3, long context end to end (plan 6b Tasks 4 and 5).
//
//   flash_long_test <snapshot> <long32k.ids> [<oracle last_logits.safetensors>]
//       K3a, as redefined by the operator's 2026-09-26 ruling: at max_len 32768, on l0-int8
//       and on l0, prefill each prefix length the oracle file holds (4096, 8192, 16384,
//       32704; tools/oracle/last_logits.py, the CPU oracle chunked through one cache) in
//       FLASH and in COMPOSED mode, and take the last-row logits cosine against the oracle.
//       Bar, per backend: mean(flash) >= mean(composed) - 1e-6, i.e. flash is no further from
//       the oracle than the composed reference. Every logit finite. Printed as INFORMATION,
//       no longer a bar: the flash-vs-composed cosine at 32704 and the first differing step
//       of 64 greedy tokens (the original K3a; measured 0.999662 on l0-int8, split at step
//       10 -- the bf16 rounding of unnormalised exp(s - m) against normalised P, spec 6
//       §3.1, amplified by the model, the near-tie class of spec 5 §8).
//       WITHOUT the oracle file (the registered form): the 32704-id prefill in both modes on
//       both backends, every logit finite, the flash-vs-composed figures printed. The 32k
//       CPU oracle is not affordable on the box: tools/oracle/last_logits.py measured 3799 s
//       to reach 4096 ids and 11359 s to reach 8192 (2026-09-26), so 32704 is a day-class
//       run; the oracle bar is registered as flash_vs_oracle_test on the golden set instead.
//   flash_long_test <snapshot> --128k <ids>
//       K3b: max_len 131072, flash mode (not registered; run by hand). The prompt (131000
//       ids, tools/probe/mk_long_ids.py) prefilled in two fresh engines: last-row logits
//       bitwise equal and finite. Then prefill replay (set_prefill_replay(true)) twice:
//       both bitwise equal to the immediate run.
//
// The 32k CPU oracle is last-row logits only (tools/oracle/last_logits.py); at 128k there
// is none, and determinism, replay and passkey retrieval carry the gate.
//   flash_long_test <snapshot> --sweep <ids>   diagnostic: flash vs composed by depth.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
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

int k3a(const std::string& snap, const std::string& ids_path, const std::string& oracle_path) {
  // With no oracle file the 32k run checks what it can without one (every logit finite)
  // and records flash-vs-composed as information; the oracle bar is then carried by
  // flash_vs_oracle_test on the golden set (see tests/CMakeLists.txt).
  bool have_oracle = false;
  if (!oracle_path.empty()) {
    std::FILE* f = std::fopen(oracle_path.c_str(), "rb");
    if (!f) {
      std::fprintf(stderr, "flash_long_test: cannot open oracle %s\n", oracle_path.c_str());
      return 2;
    }
    std::fclose(f);
    have_oracle = true;
  }
  std::unique_ptr<golden::Golden> oracle_file;
  size_t npts = 1, vocab = 0;
  const int32_t* at = nullptr;
  const float* orow = nullptr;
  static const int32_t kDeepest[1] = {32704};
  if (have_oracle) {
    oracle_file = std::make_unique<golden::Golden>(oracle_path);
    const golden::Golden& oracle = *oracle_file;
    npts = oracle.dim("at", 1, 0);
    vocab = oracle.dim("logits", 2, 1);
    CHECK_EQ(oracle.dim("logits", 2, 0), npts);
    at = oracle.i32("at", npts);
    orow = oracle.f32("logits", npts * vocab);
  } else {
    at = kDeepest;
  }
  const std::vector<uint32_t> all = golden::read_ids(ids_path);
  const uint32_t max_len = 32768;
  CHECK(size_t(at[npts - 1]) + 64 <= max_len && size_t(at[npts - 1]) <= all.size());
  if (have_oracle)
    std::printf("K3a (2026-09-26 ruling): last-row logits cosine against the CPU oracle (%s) at"
                " %zu prefix lengths of %s; bar per backend: mean(flash) >= mean(composed) - 1e-6\n",
                oracle_path.c_str(), npts, ids_path.c_str());
  else
    std::printf("K3a at 32k without an oracle file: %d ids of %s, flash and composed on l0-int8 and"
                " l0; bar: every logit finite. flash vs composed is INFORMATION (see header)\n",
                at[0], ids_path.c_str());
  l0::Context ctx(0);
  // One engine; modes and backends switched in place (prefill_smoke_test pins a switched
  // run bitwise equal to a fresh engine's).
  runtime::Engine e(ctx, loader::load(ctx, snap, max_len), max_len);
  using B = runtime::PrefillBackend;
  bool ok = true;
  for (B b : {B::L0Int8, B::L0}) {
    e.set_prefill_backend(b);
    double mean[2] = {0, 0};
    Run last[2];
    const AttnMode modes[2] = {AttnMode::Flash, AttnMode::Composed};
    for (int k = 0; k < 2; ++k) {
      runtime::prefill::set_attn_mode_for_test(modes[k]);
      for (size_t i = 0; i < npts; ++i) {
        const std::vector<uint32_t> ids(all.begin(), all.begin() + at[i]);
        const bool deepest = i + 1 == npts;
        Run r = prefill_once(e, ctx, ids, deepest ? 64 : 0);
        CHECK_EQ(nonfinite(r.logits), size_t(0));
        const size_t am =
            size_t(std::max_element(r.logits.begin(), r.logits.end()) - r.logits.begin());
        if (have_oracle) {
          CHECK_EQ(r.logits.size(), vocab);
          const std::vector<float> o(orow + i * vocab, orow + (i + 1) * vocab);
          const double cs = cosine(r.logits, o);
          mean[k] += cs / double(npts);
          std::printf("  %-8s %-8s n %5d: cos vs oracle %.9f, argmax %zu (oracle %zu), prefill %.1f ms\n",
                      runtime::prefill_backend_name(b), runtime::prefill::attn_mode_name(modes[k]),
                      at[i], cs, am, size_t(std::max_element(o.begin(), o.end()) - o.begin()),
                      r.prefill_ms);
        } else {
          std::printf("  %-8s %-8s n %5d: logits finite, argmax %zu, prefill %.1f ms\n",
                      runtime::prefill_backend_name(b), runtime::prefill::attn_mode_name(modes[k]),
                      at[i], am, r.prefill_ms);
        }
        std::fflush(stdout);
        if (deepest) last[k] = std::move(r);
      }
    }
    const bool pass = !have_oracle || mean[0] >= mean[1] - 1e-6;
    if (have_oracle) std::printf("  %s: mean cos vs oracle flash %.9f, composed %.9f (flash - composed %+.3e) -- %s\n",
                runtime::prefill_backend_name(b), mean[0], mean[1], mean[0] - mean[1],
                pass ? "PASS" : "FAIL");
    // Information, not a bar (dropped by the ruling): flash against composed directly.
    size_t div = last[0].gen.size();
    for (size_t i = 0; i < last[0].gen.size(); ++i)
      if (last[0].gen[i] != last[1].gen[i]) { div = i; break; }
    std::printf("  %s INFO at n %d: flash vs composed logits cos %.9f; 64 greedy tokens %s",
                runtime::prefill_backend_name(b), at[npts - 1], cosine(last[0].logits, last[1].logits),
                div == last[0].gen.size() ? "all equal\n" : "first differ at step ");
    if (div != last[0].gen.size())
      std::printf("%zu (flash %u, composed %u)\n", div, last[0].gen[div], last[1].gen[div]);
    std::fflush(stdout);
    ok &= pass;
  }
  runtime::prefill::set_attn_mode_for_test(AttnMode::Flash);
  CHECK(ok);
  std::puts(have_oracle
                ? "flash_long_test OK -- K3a: flash no further from the oracle than composed, l0 and l0-int8"
                : "flash_long_test OK -- 32k: flash and composed finite on l0 and l0-int8 (oracle bar: flash_vs_oracle_test)");
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
  return k3a(snap, argv[2], argc > 3 ? argv[3] : "");
}
