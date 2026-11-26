// Plan 41f Task 4: Inkling 2 Mini's prefix-cache snapshots on the card, through the server's own path
// (server::PrefixSession over cli::inkling2::Inkling2EngineAdapterT over runtime::inkling2::Engine):
//   inkling2_snapshot_test <snapshot> [kv8] [pp]
// For the plain decode and the drafted one (DSpark, K 7, greedy), each against the same request run cold (reset, one
// prefill) on the same engine, bitwise:
//   - a follow-up turn (the first turn's prompt + its output + new ids) restores the first turn's request-end snapshot;
//   - a prompt that keeps the first 4096 ids and changes the rest restores the mid-prompt snapshot at 4096;
//   - the same prompt again restores the whole of it (PERF-92: the head image, nothing prefilled);
//   - --prefix-split-last's form: the last prompt id through one decode replay; the same prompt again restores its
//     len - 1 snapshot and equals it;
//   - a reset after a long session (25,000 ids), then a short prompt, equals that prompt right after the engine's start.
// The int8 cache (`kv8`) runs the same with its own store key; `pp` runs it all on the two-card engine (both cards;
// its mid-prompt snapshots from the shadows while the next chunks run). No snapshot: SKIP (77).
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "check.h"
#include "cli/inkling2_serve.h"
#include "l0/context.h"
#include "runtime/inkling2/inkling2_pp_engine.h"
#include "server/prefix_cache.h"

namespace {

using Kind = server::PrefixCache::Plan::Kind;

struct MallocAlloc : server::HostAlloc {
  void* alloc(size_t bytes) override { return std::malloc(bytes); }
  void free(void* p, size_t) override { std::free(p); }
};

// A prompt of text-like ids: a 211-id phrase with drift, so the drafter and lookup accept some drafts.
std::vector<uint32_t> prompt_ids(uint32_t n, uint32_t seed) {
  std::vector<uint32_t> ids(n);
  for (uint32_t i = 0; i < n; ++i) ids[i] = 1000 + ((i % 211) * 7919u + (i / 997) * 31u + seed * 104729u) % 150000u;
  return ids;
}

struct Turn {
  std::vector<uint32_t> out;
  Kind kind;
  uint32_t restart;
};

template <class Adapter>
Turn request(Adapter& a, server::PrefixSession& session, const std::vector<uint32_t>& prompt, uint32_t n, bool drafted,
             bool split_last = false) {
  const server::PrefixSession::Report r = session.begin(prompt, split_last);
  server::Sampling s;
  std::vector<uint32_t> out;
  while (out.size() < n) {
    const uint32_t before = a.pos();
    std::vector<uint32_t> run = drafted ? a.step_many(s, 7) : std::vector<uint32_t>{a.step(s)};
    if (out.size() + run.size() > n) {
      run.resize(n - out.size());
      a.truncate_to(before + uint32_t(run.size()));
    }
    for (uint32_t id : run) {
      out.push_back(id);
      session.fed(id);
    }
  }
  (void)session.end();
  return {out, r.kind, r.restart};
}

template <class Adapter>
int check_mode(Adapter& a, bool drafted, const char* label) {
  MallocAlloc alloc;
  server::PrefixSession cached(a, size_t(8) << 30, &alloc);
  server::PrefixSession uncached(a, 0, nullptr);
  // The cold run of the same request on the same engine; the cached session no longer holds what the engine holds.
  const auto cold_run = [&](const std::vector<uint32_t>& prompt, uint32_t n, bool d) {
    Turn t = request(a, uncached, prompt, n, d);
    cached.fail();
    return t;
  };
  int bad = 0;
  const auto expect = [&](const char* what, const Turn& got, const Turn& want, Kind kind) {
    const bool ok = got.out == want.out && got.kind == kind;
    std::printf("%s %s: %s at %u, %zu ids %s\n", label, what, server::plan_kind_name(got.kind), got.restart,
                got.out.size(), ok ? "identical to the cold run" : "DIFFER");
    if (!ok) ++bad;
  };
  constexpr uint32_t n = 48;
  const std::vector<uint32_t> first = prompt_ids(5000, 1);
  const Turn t1 = request(a, cached, first, n, drafted);
  expect("first turn (cold, stored)", t1, cold_run(first, n, drafted), Kind::Cold);

  std::vector<uint32_t> second = first;
  second.insert(second.end(), t1.out.begin(), t1.out.end());
  const std::vector<uint32_t> more = prompt_ids(700, 2);
  second.insert(second.end(), more.begin(), more.end());
  expect("follow-up turn", request(a, cached, second, n, drafted), cold_run(second, n, drafted), Kind::Restore);

  std::vector<uint32_t> fork(first.begin(), first.begin() + 4096);
  const std::vector<uint32_t> tail = prompt_ids(1500, 3);
  fork.insert(fork.end(), tail.begin(), tail.end());
  const Turn tf = request(a, cached, fork, n, drafted);
  expect("fork after 4096", tf, cold_run(fork, n, drafted), Kind::Restore);
  if (tf.restart != 4096) {
    std::printf("%s fork after 4096: restarted at %u, not 4096\n", label, tf.restart);
    ++bad;
  }
  const Turn again = request(a, cached, first, n, drafted);
  expect("the same prompt again (head image)", again, t1, Kind::Restore);
  if (again.restart != first.size()) {
    std::printf("%s same prompt: restarted at %u, not its whole %zu\n", label, again.restart, first.size());
    ++bad;
  }
  // --prefix-split-last: the prompt's last id replayed; its next turn restores the len - 1 snapshot.
  const std::vector<uint32_t> third = prompt_ids(3000, 4);
  const Turn s1 = request(a, cached, third, n, drafted, true);
  if (s1.kind != Kind::Cold) ++bad;
  cached.fail();   // the next begin restores from the store, not from what the engine holds
  expect("split-last again", request(a, cached, third, n, drafted, true), s1, Kind::Restore);
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || !std::filesystem::exists(std::string(argv[1]) + "/config.json")) {
    std::printf("SKIP: no Inkling 2 Mini snapshot\n");
    return 77;
  }
  bool kv8 = false, pp = false;
  std::string dflash;   // dflash=DIR (plan 41k): the DFlash 2 drafter in DSpark's place, one card
  for (int i = 2; i < argc; ++i) {
    kv8 = kv8 || std::string(argv[i]) == "kv8";
    pp = pp || std::string(argv[i]) == "pp";
    if (std::string_view(argv[i]).starts_with("dflash=")) dflash = argv[i] + 7;
  }
  CHECK(dflash.empty() || !pp);
  if (pp && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: pp needs two cards\n");
    return 77;
  }
  const loader::Inkling2Checkpoint checkpoint(argv[1], dflash);
  CHECK(checkpoint.has_drafter());
  l0::Context ctx(0);
  const auto run = [&](auto& engine) {
    using Engine = std::remove_reference_t<decltype(engine)>;
    cli::inkling2::Inkling2EngineAdapterT<Engine> a(engine, model::Inkling2Desc::vocab_used,
                                                    runtime::inkling2::kPrefillRows, 7, false);
    std::printf("state %zu B, kv %zu B a position, kv form %016llx\n", a.state_bytes(), a.kv_bytes(1),
                (unsigned long long)a.kv_form());

    // The short prompt right after the engine's start, for the reset check.
    server::PrefixSession cold(a, 0, nullptr);
    const std::vector<uint32_t> shortp = prompt_ids(600, 9);
    const Turn fresh = request(a, cold, shortp, 32, false);

    int bad = check_mode(a, false, "plain") + check_mode(a, true, "drafted");

    (void)request(a, cold, prompt_ids(25000, 7), 32, true);
    const Turn after = request(a, cold, shortp, 32, false);
    std::printf("reset after a 25,000-id session, then 600 ids: %s\n", after.out == fresh.out ? "identical" : "DIFFER");
    if (after.out != fresh.out) ++bad;
    CHECK_EQ(bad, 0);
  };
  if (pp) {
    l0::Context card1(ctx, 1);
    runtime::inkling2::PpEngine engine(ctx, card1, checkpoint, 32768, true, runtime::inkling2::kPrefillRows, kv8);
    run(engine);
  } else {
    runtime::inkling2::Engine engine(ctx, checkpoint, 32768, true, runtime::inkling2::kPrefillRows, kv8);
    run(engine);
  }
  std::printf("inkling2_snapshot_test OK (%s cache%s)\n", kv8 ? "int8" : "bf16", pp ? ", pp" : "");
  return 0;
}
