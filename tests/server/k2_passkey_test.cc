// K2-Horizon past 2048 positions, end to end on the card: text -> tok::Tokenizer (the engine's,
// with the checkpoint's own tokenizer.json) -> K2Engine -> 8 greedy ids -> text, which must
// hold the pass key. Box only (labels `checkpoint;k2`), SKIP (77) without the int4 checkpoint.
//
// Why it exists (2026-11-01): the K2 passkey probe "failed at every length from 2049 ids on".
// The engine was not at fault: the int4 repo's tokenizer.json ships truncation at 2048 ids,
// which both the probe's prompt generator and the engine's tokenizer applied, so every prompt
// lost its tail - the question - and the model continued the filler. This test runs the
// tokenizer the server runs and crosses 2048 positions on both paths:
//   - prefill: ~2100 ids (one chunk + a short second one) and ~8000 ids (four chunks), the
//     needle at 5 %, 50 % and 95 % of the filler;
//   - decode only (one replay per id, K2Engine::ingest): ~2100 ids, needle at 95 %.
// The prompt is tools/probe/passkey.py's (no chat template, BOS prepended by the tokenizer).
// argv: [1] snapshot. The KV form is B70_KV_CACHE's, the attention B70_K2_ATTN's.
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "check.h"
#include "l0/context.h"
#include "loader/k2_loader.h"
#include "loader/snapshot.h"
#include "runtime/k2/k2_engine.h"
#include "tokenizer/tokenizer.h"

namespace {
using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 8192, kGen = 8;
const std::string kFiller =
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ";
const std::string kNeedle = "The pass key is 71432. Remember it. 71432 is the pass key. ";
const std::string kQuestion = "What is the pass key? The pass key is";

std::string repeat(const std::string& s, size_t n) {
  std::string out;
  out.reserve(s.size() * n);
  for (size_t i = 0; i < n; ++i) out += s;
  return out;
}

// passkey.py's arithmetic: as many filler repetitions as fit `target` ids, the needle after
// round(place x reps) of them, the question last.
std::string prompt(const tok::Tokenizer& t, uint32_t target, double place) {
  const double per = double(t.encode(repeat(kFiller, 100), false).size()) / 100.0;
  const size_t reps = size_t((double(target) - double(t.encode(kNeedle, false).size()) -
                              double(t.encode(kQuestion, false).size())) / per);
  const size_t before = size_t(place * double(reps) + 0.5);
  return repeat(kFiller, before) + kNeedle + repeat(kFiller, reps - before) + kQuestion;
}

bool run(runtime::k2::K2Engine& eng, const tok::Tokenizer& t, uint32_t target, double place, bool decode_only) {
  const std::string text = prompt(t, target, place);
  const Ids ids = t.encode(text, true);   // BOS 0 + the text
  eng.reset();
  if (decode_only)
    eng.ingest(ids);
  else
    eng.prefill(ids);
  const Ids out = eng.generate(kGen);
  const std::string answer = t.decode(out, false);
  const bool ok = answer.find("71432") != std::string::npos && ids.size() > 2048;
  std::printf("  %-7s %5zu ids, needle at %4.0f %%: \"%s\"%s\n", decode_only ? "decode" : "prefill", ids.size(),
              place * 100.0, answer.c_str(), ok ? "" : "  ** FAIL **");
  return ok;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string snap = argc > 1 ? argv[1] : "";
  std::string dir;
  try {
    dir = loader::resolve_snapshot(snap);
  } catch (const std::exception& e) {
    std::printf("SKIP: no K2-Horizon checkpoint at '%s' (%s)\n", snap.c_str(), e.what());
    return 77;
  }
  if (!dir.empty() && dir.back() != '/') dir += '/';
  const tok::Tokenizer t(dir + "tokenizer.json");
  l0::Context ctx(0);
  runtime::k2::K2Engine eng(ctx, loader::load_k2(ctx, snap, kMaxLen), kMaxLen);
  eng.prepare_prefill();
  std::printf("k2_passkey_test: %s KV, attention %s, max_len %u\n", runtime::kv_cache_name(eng.kv_cache()),
              runtime::k2::k2_attn_name(runtime::k2::k2_attn()), kMaxLen);
  bool all = true;
  for (const uint32_t target : {2100u, 8000u})
    for (const double place : {0.05, 0.5, 0.95}) all = run(eng, t, target, place, false) && all;
  all = run(eng, t, 2100, 0.95, true) && all;
  if (!all) {
    std::fprintf(stderr, "k2_passkey_test FAILED\n");
    return 1;
  }
  std::printf("k2_passkey_test OK: the pass key retrieved past 2048 positions (prefill at ~2100 / ~8000 ids, "
              "decode-only at ~2100)\n");
  return 0;
}
