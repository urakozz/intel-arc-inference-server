#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "check.h"
#include "golden/forced_judge.h"
#include "runtime/inkling2/inkling2_tp_engine.h"
#include "runtime/inkling2_prefill_test_access.h"

namespace {

std::vector<uint32_t> read_prompt(const char* path) {
  std::ifstream file(path);
  std::vector<uint32_t> prompt;
  for (uint32_t id; file >> id;) prompt.push_back(id);
  CHECK(file.eof() && !prompt.empty());
  return prompt;
}

template <class E>
uint32_t ingest(E& engine, std::span<const uint32_t> prompt) {
  uint32_t next = 0;
  for (size_t pos = 0; pos < prompt.size(); pos += runtime::inkling2::kRows) {
    const size_t n = std::min<size_t>(runtime::inkling2::kRows, prompt.size() - pos);
    next = engine.verify(prompt.subspan(pos, n)).back();
    engine.commit(uint32_t(n));
  }
  return next;
}

// The three-call round (propose, verify, commit), step_drafted's form before its one submission.
template <class E>
uint32_t step_split(E& engine, uint32_t anchor, uint32_t k, std::vector<uint32_t>& output,
                    uint64_t* drafts = nullptr) {
  std::array<uint32_t, runtime::inkling2::kRows> ids{};
  ids[0] = anchor;
  const auto proposed = engine.propose(anchor, k);
  if (drafts)   // FNV-1a over every proposed id: two builds' drafts compared byte for byte
    for (uint32_t id : proposed) *drafts = (*drafts ^ id) * 1099511628211ull;
  std::copy(proposed.begin(), proposed.end(), ids.begin() + 1);
  const auto predicted = engine.verify({ids.data(), k + 1});
  uint32_t accepted = 1;
  while (accepted <= k && ids[accepted] == predicted[accepted - 1]) ++accepted;
  const uint32_t next = predicted[accepted - 1];
  output.insert(output.end(), ids.begin(), ids.begin() + accepted);
  engine.commit(accepted);
  return next;
}

// One drafted round of 7 drafts both ways from the same state: which persistent buffers differ afterwards.
void compare_round(runtime::inkling2::Engine& engine, const char* path) {
  using Access = runtime::inkling2::PrefillTestAccess;
  const auto prompt = read_prompt(path);
  std::array<std::vector<std::vector<std::byte>>, 2> after;
  std::array<std::vector<uint32_t>, 2> out;
  for (int way = 0; way < 2; ++way) {
    engine.reset();
    const uint32_t anchor = ingest(engine, prompt);
    out[way].clear();
    if (way == 0) engine.step_drafted(anchor, 7, out[way]);
    else step_split(engine, anchor, 7, out[way]);
    auto list = l0::CmdList::immediate(*Access::context(engine));
    for (const auto* mem : Access::state(engine)) {
      auto& bytes = after[way].emplace_back(mem ? mem->size() : 0);
      if (mem) list.copy(bytes.data(), mem->ptr(), bytes.size());
    }
  }
  constexpr const char* names[] = {"full_k", "full_v", "ring_k", "ring_v", "conv_k", "conv_v", "conv_attn",
                                   "conv_mlp", "draft_full_k", "draft_full_v", "draft_ring_k", "draft_ring_v"};
  std::printf("Inkling round compare: one submission kept %zu ids, three calls %zu;", out[0].size(), out[1].size());
  for (size_t i = 0; i < after[0].size(); ++i)
    if (after[0][i] != after[1][i]) {
      size_t first = 0;
      while (after[0][i][first] == after[1][i][first]) ++first;
      std::printf(" %s differs from byte %zu", names[i], first);
    }
  std::printf("\n");
}

template <class E>
void measure_acceptance(E& engine, const char* path, uint32_t eos, bool split = false) {
  constexpr uint32_t limit = 128;
  const auto prompt = read_prompt(path);
  CHECK(prompt.size() + limit <= engine.max_len());
  engine.reset();
  uint32_t next = ingest(engine, prompt);
  uint32_t proposed = 0, accepted = 0, generated = 0;
  std::array<uint32_t, 7> tested_at{}, accepted_at{};
  std::vector<uint32_t> output;
  uint32_t rounds = 0;
  uint64_t drafts = 14695981039346656037ull;   // the wall time a round: the host side of the three calls against one submission
  const auto start = std::chrono::steady_clock::now();
  while (output.size() < limit) {
    ++rounds;
    const auto before = uint32_t(output.size());
    const auto count = std::min(7u, limit - before - 1);
    next = split ? step_split(engine, next, count, output, &drafts) : engine.step_drafted(next, count, output);
    const auto end = std::find(output.begin() + before, output.end(), eos);
    generated = uint32_t(end - output.begin()) + (end != output.end());
    const uint32_t kept = generated - before - 1;
    proposed += count;
    accepted += kept;
    for (uint32_t i = 0; i < std::min(kept + 1, count); ++i) {
      ++tested_at[i];
      accepted_at[i] += i < kept;
    }
    if (end != output.end()) break;
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  const auto name = std::filesystem::path(path).filename().string();
  std::printf("Inkling %s %s%s accepted %u/%u drafts (%.1f%%) over %u output tokens through EOS or the sample limit;"
              " by position (accepted/tested):", engine.dflash() ? "DFlash 2" : "DSpark", name.c_str(), split ? " (three calls)" : "", accepted, proposed,
              proposed ? 100.0 * accepted / proposed : 0.0, generated);
  for (uint32_t i = 0; i < 7 && tested_at[i]; ++i) std::printf(" %u/%u", accepted_at[i], tested_at[i]);
  std::printf("; %.3f ms a round over %u", ms / rounds, rounds);
  if (split) std::printf("; drafts %016llx", static_cast<unsigned long long>(drafts));
  std::printf("\n");
}

// Plain, drafted (one submission) and three-call drafted rounds from the same prompt pass: the same ids. The rounds'
// draft counts cycle through 0..7 with plain steps between rounds (the server's mix), so each width, rejected first
// drafts and every order of plain and drafted steps are met. A verify row is not bitwise the one-row step, so with an
// f32 golden (`gold`, for this prompt) a drafted id that leaves plain's passes when the reference row accepts both
// (the one-ulp near-tie rule) - judged while plain has kept the reference's ids, the row's context.
int same_ids(const loader::Inkling2Checkpoint& checkpoint, std::vector<uint32_t> prompt, uint32_t n, bool kv8,
             const golden::Golden* gold) {
  using D = model::Inkling2Desc;
  const auto length = uint32_t(prompt.size());
  l0::Context ctx(0);
  runtime::inkling2::Engine engine(ctx, checkpoint, length + n + runtime::inkling2::kRows, true,
                                   runtime::inkling2::kPrefillRows, kv8);
  const uint32_t chunk = runtime::inkling2::kPrefillRows;
  std::array<std::vector<uint32_t>, 3> out;
  for (uint32_t way = 0; way < 3; ++way) {
    engine.reset();
    uint32_t next = engine.prefill(prompt, chunk), round = 0;
    while (out[way].size() < n) {
      const uint32_t left = n - uint32_t(out[way].size());
      constexpr uint32_t ks[] = {0, 1, 0, 0, 3, 0, 5, 7, 2, 0, 4, 6};
      const uint32_t k = std::min(ks[round++ % std::size(ks)], left - 1);
      if (way == 0 || k == 0) next = engine.step_drafted(next, way == 0 ? 0 : k, out[way]);
      else if (way == 1) next = engine.step_drafted(next, k, out[way]);
      else next = step_split(engine, next, k, out[way]);
    }
    out[way].resize(n);
  }
  // The reference's tokens and the row of output index i, when a golden is given.
  const int32_t* tokens = gold ? gold->i32("tokens", gold->dim("tokens", 1, 0)) : nullptr;
  const auto rows = gold ? uint32_t(gold->dim("logits", 2, 0)) : 0u;
  const float* logits = gold ? gold->f32("logits", size_t(rows) * D::vocab_used) : nullptr;
  const uint32_t row0 = length + n - rows;   // the logits' first row: the prompt's last at 0 rows of output
  uint32_t kept = 0;   // plain keeps the reference's ids for its first `kept`
  while (gold && kept < n && out[0][kept] == uint32_t(tokens[kept])) ++kept;
  constexpr const char* names[] = {"plain", "drafted", "three-call drafted"};
  bool ok = true;
  for (uint32_t way = 1; way < 3; ++way) {
    uint32_t first = 0;
    while (first < n && out[way][first] == out[0][first]) ++first;
    if (first == n) continue;
    std::printf("Inkling %s ids differ from plain at %u of %u: %u, plain %u", names[way], first, n, out[way][first],
                out[0][first]);
    if (!gold) {
      std::printf("\n");
      ok = false;
    } else if (first > kept) {
      std::printf(" (after plain left the reference at %u: not judged)\n", kept);
    } else {
      const float* row = logits + size_t(length - 1 + first - row0) * D::vocab_used;
      const bool a = golden::row_accepts(row, out[way][first], D::vocab_used, D::vocab_used);
      const bool p = golden::row_accepts(row, out[0][first], D::vocab_used, D::vocab_used);
      std::printf(": the f32 row accepts it %s, plain's %s\n", a ? "yes" : "NO", p ? "yes" : "NO");
      if (!a) golden::judge_explain(*gold, length - 1 + first - row0, out[way][first], D::vocab_used, D::vocab_used,
                                    names[way]);
      ok = ok && a;
    }
  }
  CHECK(ok);
  std::printf("Inkling %u ids after a %u-id prompt (%s cache): plain, drafted and three-call drafted %s\n", n, length,
              kv8 ? "int8" : "bf16", gold ? "equal or near-ties the f32 reference accepts" : "equal");
  return 0;
}

template <class E>
int judged(E& engine, const loader::Inkling2Checkpoint& checkpoint, int argc, char** argv, bool draft,
           std::span<const uint32_t> prompt, const golden::Golden& gold, const golden::Golden* twin);
}  // namespace

int main(int argc, char** argv) {
  // Any mode: an argument dflash=DIR (plan 41k) loads the DFlash 2 drafter from DIR in DSpark's place.
  std::string dflash;
  for (int i = 1; i < argc; ++i)
    if (std::string_view(argv[i]).starts_with("dflash=")) {
      dflash = argv[i] + 7;
      std::copy(argv + i + 1, argv + argc, argv + i);
      --argc;
      break;
    }
  // inkling2_decode_test SNAPSHOT accept PROMPT...: only the drafted acceptance of each prompt (128 ids, 7 drafts).
  if (argc >= 4 && std::string_view(argv[2]) == "accept") {
    const loader::Inkling2Checkpoint checkpoint(argv[1], dflash);
    l0::Context ctx(0);
    runtime::inkling2::Engine engine(ctx, checkpoint, 8192, true);
    compare_round(engine, argv[3]);
    for (int i = 3; i < argc; ++i) {
      measure_acceptance(engine, argv[i], checkpoint.desc().eos);
      measure_acceptance(engine, argv[i], checkpoint.desc().eos, true);
    }
    return 0;
  }
  // inkling2_decode_test SNAPSHOT same PROMPT LENGTH N [kv8]: plain and drafted ids equal after the first LENGTH ids;
  // SNAPSHOT same-golden GOLDEN [kv8]: after the golden's prompt, its ids' count, judged by it where they differ.
  if (argc >= 4 && (std::string_view(argv[2]) == "same" || std::string_view(argv[2]) == "same-golden")) {
    const bool by_golden = std::string_view(argv[2]) == "same-golden";
    const char* path = argv[3];
    if (!std::filesystem::exists(path)) {
      std::printf("SKIP: %s not found\n", path);
      return 77;
    }
    const loader::Inkling2Checkpoint checkpoint(argv[1], dflash);
    const bool kv8 = std::string_view(argv[argc - 1]) == "kv8";
    if (by_golden) {
      const golden::Golden gold(path);
      const auto count = uint32_t(gold.dim("prompt_ids", 1, 0));
      const auto* ids = gold.i32("prompt_ids", count);
      return same_ids(checkpoint, {ids, ids + count}, uint32_t(gold.dim("tokens", 1, 0)), kv8, &gold);
    }
    CHECK(argc >= 6);
    auto prompt = read_prompt(path);
    const auto length = uint32_t(std::stoul(argv[4]));
    CHECK(prompt.size() >= length);
    prompt.resize(length);
    return same_ids(checkpoint, prompt, uint32_t(std::stoul(argv[5])), kv8, nullptr);
  }
  // Last arguments `kv8` (the int8 cache, judged by its own f32 reference, GOLDEN <set>.fp32-kv8: the plain ids by the
  // near-miss rule against its bf16 twin <set>.bf16-kv8) and `tp` (the same judgements under `--parallel tp`, cards 0
  // and 1; skipped, 77, with one), in either order.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  bool tp = false, kv8 = false;
  for (; argc >= 5; --argc) {
    const std::string_view last = argv[argc - 1];
    if (last == "tp") tp = true;
    else if (last == "kv8") kv8 = true;
    else break;
  }
  CHECK(argc == 4 || (argc >= 5 && std::string_view(argv[4]) == "draft"));
  const bool draft = argc >= 5;
  std::string twin_path;
  if (const std::string g = argv[3]; kv8 && g.find(".fp32-kv8.") != std::string::npos)
    twin_path = g.substr(0, g.find(".fp32-kv8.")) + ".bf16-kv8." + g.substr(g.find(".fp32-kv8.") + 10);
  for (const std::string& path : {std::string(argv[2]), std::string(argv[3]), twin_path})
    if (!path.empty() && !std::filesystem::exists(path)) {
      std::printf("SKIP: %s not found\n", path.c_str());
      return 77;
    }
  const auto prompt = read_prompt(argv[2]);
  const golden::Golden gold(argv[3]);
  std::optional<golden::Golden> twin;
  if (!twin_path.empty()) twin.emplace(twin_path);
  const auto max_len = std::max(4096u, uint32_t(prompt.size() + gold.dim("tokens", 1, 0)) + runtime::inkling2::kRows);
  const loader::Inkling2Checkpoint checkpoint(argv[1], dflash);
  l0::Context ctx(0);
  if (!tp) {
    runtime::inkling2::Engine engine(ctx, checkpoint, max_len, draft, 0, kv8);
    return judged(engine, checkpoint, argc, argv, draft, prompt, gold, twin ? &*twin : nullptr);
  }
  if (l0::Context::gpu_count() < 2) {
    std::printf("SKIP: --parallel tp needs two cards\n");
    return 77;
  }
  l0::Context ctx1(ctx, 1);
  runtime::inkling2::TpEngine engine(ctx, ctx1, checkpoint, max_len, draft, 0, kv8);
  std::printf("Inkling --parallel tp (cards 0 and 1)\n");
  return judged(engine, checkpoint, argc, argv, draft, prompt, gold, twin ? &*twin : nullptr);
}

namespace {
template <class E>
int judged(E& engine, const loader::Inkling2Checkpoint& checkpoint, int argc, char** argv, bool draft,
           std::span<const uint32_t> prompt, const golden::Golden& gold, const golden::Golden* twin) {
  const auto gen = gold.dim("tokens", 1, 0);
  const auto rows = gold.dim("logits", 2, 0);
  using D = model::Inkling2Desc;
  // The logits cover the whole stream or its tail from the prompt's last row (the judge sets' prompt-tail form).
  CHECK(rows > gen && rows <= prompt.size() + gen);
  const auto row0 = prompt.size() + gen - rows;
  CHECK(!twin || twin->dim("logits", 2, 0) == rows);
  const auto* logits = gold.f32("logits", rows * D::vocab_used);
  const auto* tokens = gold.i32("tokens", gen);
  uint32_t ties = 0;   // decisions where the engine's id is not the reference's but the f32 row accepts it
  const auto judge = [&](uint32_t index, uint32_t id) {
    const auto row = uint32_t(prompt.size() - 1 + index - row0);
    const bool ok = golden::row_accepts(logits + size_t(row) * D::vocab_used, id, D::vocab_used, D::vocab_used);
    if (!ok) golden::judge_explain(gold, row, id, D::vocab_used, D::vocab_used, "engine");
    ties += ok && id != uint32_t(tokens[index]);
    return ok;
  };
  // A drafted decision the judge rejects still passes with a twin by the near-miss rule's row test: the twin's own
  // argmax rejected there too, or the id within kNearMissUlp of the f32 top (the cosine half is the plain test's).
  uint32_t near_misses = 0;
  const auto decide = [&](uint32_t index, uint32_t id) {
    if (judge(index, id)) return true;
    if (!twin) return false;
    const auto row = size_t(prompt.size() - 1 + index - row0) * D::vocab_used;
    const golden::JudgedRow jr = golden::judge_row(logits + row, D::vocab_used, D::vocab_used);
    const golden::GoldenDecision td =
        golden::golden_decision(twin->f32("logits", rows * D::vocab_used) + row, D::vocab_used, D::vocab_used);
    bool twin_ok = false;
    for (uint32_t t : td.set) twin_ok = twin_ok || golden::row_accepts(logits + row, t, D::vocab_used, D::vocab_used);
    const double below = (double(logits[row + jr.top]) - double(logits[row + id])) / jr.ulp;
    std::printf("    near-miss rule: %.2f ulp below the f32 top (bar %.0f), the bf16 twin's id %s\n", below,
                golden::kNearMissUlp, twin_ok ? "accepted" : "rejected too");
    ++near_misses;
    return !twin_ok || below <= golden::kNearMissUlp;
  };
  if (draft) {
    const auto count = uint32_t(gen);
    uint32_t next = ingest(engine, prompt), steps = 0;
    std::vector<uint32_t> actual;
    while (actual.size() < count) {
      next = engine.step_drafted(next, std::min(7u, count - uint32_t(actual.size()) - 1), actual);
      ++steps;
    }
    CHECK(steps < count);
    uint32_t greedy_judged = 0;
    for (uint32_t i = 0; i < count; ++i) {
      CHECK(decide(i, actual[i]));
      ++greedy_judged;
      if (actual[i] != uint32_t(tokens[i]) || actual[i] == checkpoint.desc().eos) break;
    }

    // At a near tie the two valid answers can lead to different contexts.
    // Force the golden tokens through the draft verifier to judge the rest.
    engine.reset();
    {   // the reset clears every row the drafted rounds and plain steps wrote (each card's under tp)
      using Access = runtime::inkling2::PrefillTestAccess;
      std::vector<runtime::inkling2::Engine*> cards;
      if constexpr (std::is_same_v<E, runtime::inkling2::TpEngine>) cards = {&engine.card(0), &engine.card(1)};
      else cards = {&engine};
      for (auto* card : cards) {
        auto list = l0::CmdList::immediate(*Access::context(*card));
        for (const auto* mem : Access::state(*card)) {
          if (!mem) continue;
          std::vector<std::byte> bytes(mem->size());
          list.copy(bytes.data(), mem->ptr(), bytes.size());
          CHECK(std::all_of(bytes.begin(), bytes.end(), [](std::byte b) { return b == std::byte{0}; }));
        }
      }
    }
    next = ingest(engine, prompt);
    uint32_t graded = 0;
    while (graded < count) {
      CHECK(decide(graded, next));
      const auto anchor = uint32_t(tokens[graded]);
      if (graded + 1 == count || anchor == checkpoint.desc().eos) {
        ++graded;
        break;
      }
      const auto k = std::min(7u, count - graded - 1);
      std::array<uint32_t, runtime::inkling2::kRows> input{};
      input[0] = anchor;
      const auto draft_ids = engine.propose(anchor, k);
      std::copy(draft_ids.begin(), draft_ids.end(), input.begin() + 1);
      const auto predictions = engine.verify({input.data(), k + 1});
      uint32_t accepted = 1;
      while (accepted <= k && input[accepted] == uint32_t(tokens[graded + accepted]) &&
             input[accepted] != checkpoint.desc().eos) {
        CHECK(decide(graded + accepted, predictions[accepted - 1]));
        ++accepted;
      }
      next = predictions[accepted - 1];
      engine.commit(accepted);
      graded += accepted;
    }
    std::printf("Inkling %s generated %u ids in %u verifier steps, %u greedy and %u forced decisions judged, %u"
                " f32 near-ties, %u near-misses\n", engine.dflash() ? "DFlash 2" : "DSpark", count, steps, greedy_judged, graded, ties, near_misses);
    for (int i = 5; i < argc; ++i) measure_acceptance(engine, argv[i], checkpoint.desc().eos);
    return 0;
  }
  uint32_t misses = 0, graded = 0;
  golden::CosRows drift;
  golden::NearMiss near;
  const float* twin_logits = twin ? twin->f32("logits", rows * D::vocab_used) : nullptr;
  std::vector<float> actual(D::vocab_used);
  for (size_t p = 0; p + 1 < prompt.size();) {
    const auto count = uint32_t(std::min<size_t>(8, prompt.size() - 1 - p));
    engine.verify({prompt.data() + p, count});
    engine.commit(count);
    p += count;
  }
  uint32_t input = prompt.back();
  for (uint32_t i = 0; i < gen; ++i) {
    const auto index = uint32_t(prompt.size() - 1 + i);
    const auto* row = logits + size_t(index - row0) * D::vocab_used;
    const auto next = engine.verify({&input, 1})[0];
    engine.read_logits(0, actual);
    drift.c.push_back(golden::row_cos(row, actual.data(), D::vocab_used));
    if (!judge(i, next)) ++misses;
    if (twin) near.add(index - row0, row, next, actual.data(), twin_logits + size_t(index - row0) * D::vocab_used,
                       D::vocab_used, D::vocab_used, "engine", true);
    ++graded;
    if (uint32_t(tokens[i]) == checkpoint.desc().eos) break;
    engine.commit(1);
    input = uint32_t(tokens[i]);
  }
  std::printf("Inkling CPU golden judged %u/%u decisions, %u f32 near-ties, logit cosine min/median %s\n",
              graded - misses, graded, ties, drift.str().c_str());
  if (twin) CHECK(near.verdict("Inkling int8 cache", "engine"));
  else CHECK_EQ(misses, 0u);
  return 0;
}
}  // namespace
