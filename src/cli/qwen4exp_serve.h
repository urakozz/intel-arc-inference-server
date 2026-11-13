#pragma once
// Spec 21e Task 4: b70-serve's Qwen3.8-Flash-Next path (model_type qwen4_exp) - what the dispatch reads and refuses
// before the device; the body (settle, load, engine, adapter, options, listen) is b70_serve.cc's serve_qwen4exp, beside
// serve_kolibri, over these. The request path is the Qwen family's (cli/qwen4exp_chat.h: ChatFormat Qwen, XML tool
// calls, reasoning on "<think>\n", EOS from generation_config.json, decision 11's greedy-unless-asked); the engine is
// runtime::qwen4exp::Qwen4ExpEngine behind spec 16d's cli::pp::PipelineEngineAdapterT (src/cli/pipeline_serve_adapter.h:
// EngineAdapter's logic over the engine's host calls - no third copy of the speculative loop), on one card or two.
//
//   --layers N|auto        REQUIRED until spec 22 (b70-decode's development mode): without it the whole model is
//                          refused naming its bytes and spec 22 (the expert-offload tier)
//   the PLE int8 file      <snapshot>-ple-int8/, beside the checkpoint (no flag)
//   the tokenizer          the snapshot's tokenizer.json, held to the original's (Qwen/Qwen3.8-Flash-Next): 21a found the
//                          original's split regex adds \p{M}, and Intel's checkpoint ships Qwen3.8's file - a snapshot
//                          tokenizer.json without \p{M} in its pre-tokenizer is refused (check_tokenizer)
//   --parallel auto|1|pp, --pipeline-split auto|N, --max-len N|auto   as b70-decode
//   the lm_head            int8 (spec 9), as every served model
//   --spec mtp|lookup      the MTP head (spec 21e: draft / verify / commit through the adapter, a fixed K = --spec-max
//                          (1..3, default 3); lookup's drafts through the same verify lists); the planner carries the
//                          head's weights and buffers; --spec auto is off for this model
//   the prefix cache       the existing flags (--prefix-cache-gb, --prefix-split-last): kv_form 0 (bf16); the
//                          snapshots are runtime/qwen4exp/qwen4exp_sizes.h's (GDN, PLE and indexer state; KV and
//                          compressed-key blocks; with the head its tail, R, KV and keys)
// Refused before the device, each by name (the adaptive K waits for spec 22's full-model cost numbers: cli::resolve_spec
// gives this family a fixed K), --draft-vocab (spec 8 §11's compact head is not built for this head), --kv-cache int8 (decision 8),
// --prefill-backend l0-int8 (b70-decode's message), --pipeline-split on one card, the whole
// model without --layers.
#include <cstdint>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "cli/qwen4exp_chat.h"
#include "cli/qwen4exp_decode.h"

namespace cli::qwen4exp {

struct ServeArgs {
  std::string snapshot_dir;
  DecodeArgs args;                       // the planner's and the refusals' inputs (prefill on, lm_head int8)
  uint32_t mtp_k = 0;                    // --spec mtp's fixed K (0: no head)
  bool spec_lookup = false, draft_vocab = false;
  bool prefix_auto = true;
  uint32_t prefix_cache_gb = 0;
  bool loads_head() const { return mtp_k > 0 || spec_lookup; }
};

// Every refusal that needs no device and no tensor, in this order (tests/CMakeLists.txt's cli_reject_serve_qwen4exp_*).
inline void check_serve(const ServeArgs& s) {
  if (s.draft_vocab)
    throw std::runtime_error("--draft-vocab is not built for Qwen3.8-Flash-Next's MTP head (spec 8 §11's compact head is "
                             "Qwen3.8's): drop --draft-vocab");
  if (s.args.kv8)
    throw std::runtime_error("--kv-cache int8: Qwen3.8-Flash-Next's KV is bf16 (spec 21 decision 8: int8 later - kv8.cl "
                             "hard-codes 24 q / 4 kv heads and this model has 2)");
  if (s.mtp_k > runtime::qwen4exp::kMaxDraft)
    throw std::runtime_error("--spec-max " + std::to_string(s.mtp_k) + ": Qwen3.8-Flash-Next verifies at M <= 4 (K 1..3)");
  DecodeArgs a = s.args;   // b70-decode's refusals: the prefill backends, the --parallel sub-flags, --layers
  a.mtp = false;
  a.kv8 = false;
  a.profile = false;
  check_args(a, "b70-serve");
}

// True when a JSON value (recursively) holds a string containing \p{M} - the original's split regex (21a's finding 3).
inline bool holds_mark_class(const nlohmann::json& v) {
  if (v.is_string()) return v.get<std::string>().find("\\p{M}") != std::string::npos;
  if (v.is_object() || v.is_array())
    for (const auto& e : v)
      if (holds_mark_class(e)) return true;
  return false;
}

// The served tokenizer.json must be Qwen3.8-Flash-Next's own (its pre-tokenizer's split regex keeps combining marks
// with their letters, \p{M}); Qwen3.8's file (Intel's checkpoint ships it) encodes Devanagari, Thai and Arabic text
// differently (tests/tokenizer/qwen4exp_tokenizer.json, cases_qwen38_differs). Throws naming the fix.
inline void check_tokenizer(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open '" + path + "' (Qwen3.8-Flash-Next's tokenizer.json, in the snapshot)");
  const nlohmann::json j = nlohmann::json::parse(f, nullptr, /*allow_exceptions=*/false);
  if (!j.is_object() || !j.contains("pre_tokenizer"))
    throw std::runtime_error("'" + path + "' is not a tokenizer.json with a pre_tokenizer");
  if (!holds_mark_class(j.at("pre_tokenizer")))
    throw std::runtime_error("'" + path + "' splits without \\p{M}: it is Qwen3.8's tokenizer.json (Intel's "
                             "Qwen3.8-Flash-Next-W4A16-AutoRound ships it), not Qwen3.8-Flash-Next's - the original's "
                             "split regex keeps combining marks with their letters (spec 21 §12, finding 3). Serve a snapshot "
                             "that ships Qwen/Qwen3.8-Flash-Next's tokenizer.json");
}

}  // namespace cli::qwen4exp
