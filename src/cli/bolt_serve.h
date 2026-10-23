#pragma once
// Spec 25e: Nemotron 3.5 Bolt behind b70-serve - runtime::bolt::BoltEngine as a server::EngineIface (LFM2's Lfm2MoeEngineAdapterT
// shape, src/cli/lfm2moe_serve.h, for a hybrid Mamba / attention / MoE model). b70-serve instantiates it over BoltEngine (two
// cards by default, Decision 4: the engine holds the placement); the template exists so the host test
// (tests/runtime/bolt_snapshot_test.cc) runs the same adapter over a device-free engine with Bolt's real state and KV layouts.
// This header names no Level Zero type.
//
// What the adapter forwards, and what Bolt does not have:
//   - prefill / step / reset / pos / max_len / ingest: BoltEngine's (25d's chunked prefill on l0; greedy = the device argmax;
//     sampled = the logits row read back and filtered as every model's - server::filter_probs at the tokenizer's id count,
//     temperature, top-k (none by default), top-p -, the chosen id written by set_token on every device);
//   - every row of the head is a token (131072: tokenizer.json defines all of them), so no id is ever masked on the release;
//     `vocab_used` guards a tokenizer that defines fewer (a greedy argmax on a row without a token takes the masked row's argmax);
//   - the repetition penalty: a SAMPLED request with Sampling::repetition_penalty != 1 has the logits row penalised over the
//     request's distinct prompt and output ids (server::apply_repetition_penalty, HF's rule) before the filters; a greedy request
//     keeps the device argmax and ignores it (LFM2's rule). NVIDIA's defaults carry none;
//   - a SEEDED sampled request is reproducible: the RNG is re-seeded from the request's seed at its first step (the first step
//     after a prefill, an ingest, a restore or a reset - every request begins with one of them); an unseeded request re-seeds from
//     the random device, so it never continues an earlier seeded request's stream;
//   - the next id is chosen at the START of step(), from the logits row of the previous replay (or of the prefill's head);
//   - spec 7's prefix cache (spec 25 §6.3): per Mamba block the SSM state and the three raw-xBC conv columns before the position
//     as the state, the 8 attention blocks' KV as the blocks (runtime/bolt/bolt_sizes.h's state_runs / kv_runs), the block hook on
//     BoltEngine::prefill (chunks end at block ends), ingest for --prefix-split-last; the store keyed by Bolt's own root
//     (kBoltStoreKey, the bf16-state arm another one: its snapshots are half the size), so no entry another model's engine made
//     - or another state precision's - can ever hit;
//   - spec 26c: speculative decoding with `spec_k` > 0 - --spec dflash (LiLiCorr: step_many) or --spec lookup (step_drafts) over
//     BoltEngine's verify lists (the adapter section below says how greedy and sampled acceptance run); without it every path
//     is 25e's (mtp_k() and verify_k() 0). No MTP head: b70-serve refuses --mtp and --spec mtp by name.
//
// Also here, host-side and device-free: the generation defaults (Decision 12) and the flags b70-serve refuses for this model.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "runtime/dflash/dflash_control.h"   // spec 26c: LiliDraftOut (a plain struct)
#include "server/deps.h"
#include "server/openai.h"
#include "server/spec_accept.h"

namespace cli::bolt {

// The prefix cache's root for Bolt (PrefixCache's kv_form): "NEMOBOLT". Qwen3.8 / Agnes / Ornith keep 0 / 1, the other models
// their own keys: no chain crosses models; the entries' bytes are Bolt's SSM states, conv columns and attention KV (bf16 only),
// meaningless to any other engine. The bf16-state arm (B70_BOLT_SSM=bf16) flips the low bit: its states are half the size.
inline constexpr uint64_t kBoltStoreKey = 0x4E454D4F424F4C54ull;
inline constexpr uint64_t store_key(bool bf16_state) { return kBoltStoreKey ^ (bf16_state ? 1ull : 0ull); }

// Bolt's chat template (chat_template.jinja, 12,514 B; byte-identical in NVIDIA's and SergiioB's repos) - what the startup line
// compares the template it reads against. Not a refusal: --chat-template FILE and a newer snapshot template are the operator's.
inline constexpr const char* kTemplateSha256 = "6521afc75afb0661e3558e7e6dee5f2d15de8be1c4dece8171935e778ea283f4";

// ---- generation defaults (Decision 12) ---------------------------------------------------------------------------------------
// NVIDIA's published values (their generation_config.json, the card's "Recommended Sampling: temperature 1.0, top_p 0.95"):
// sampled, temperature 1.0, top-p 0.95, EOS [2, 11] (`</s>` and `<|im_end|>`). SergiioB's export ships a generation_config.json
// with EOS 11 only and no sampling keys - it gives the published set, not "greedy, EOS 11". The card names no top-k and vLLM
// (the reference server) applies none, so an absent top_k is NONE here (0), not Hugging Face's 50 that server::generation_sampling
// assumes for a field a file leaves out.
inline constexpr uint32_t kEosEnd = 11, kEosSentence = 2;

struct Generation {
  server::Sampling sampling;       // what a request that names no temperature / top_p / top_k samples with
  std::vector<uint32_t> eos;       // sorted, distinct
  std::vector<std::string> stops;  // generation_config.json's stop_strings (none in either published file)
  std::string source;              // "published defaults" | "<path> (the snapshot's file carries sampling keys)" | "--generation-config <path>"
};

inline bool has_sampling_keys(const nlohmann::json& g) {
  if (!g.is_object()) return false;
  for (const char* k : {"do_sample", "temperature", "top_p", "top_k", "repetition_penalty"})
    if (g.contains(k)) return true;
  return false;
}

inline Generation published_generation() {
  Generation out;
  out.sampling = server::generation_sampling(nlohmann::json{{"do_sample", true}, {"temperature", 1.0}, {"top_p", 0.95}});
  out.sampling.top_k = 0;
  out.eos = {kEosSentence, kEosEnd};
  out.source = "published defaults (NVIDIA's generation_config.json: sampled, T 1.0, top-p 0.95, EOS [2, 11])";
  return out;
}

// `snapshot` is the snapshot's generation_config.json (null when it has none), `explicit_file` the parsed --generation-config FILE
// (null when not given; `explicit_path` names it for the startup line). Rules:
//   - an explicit file is the operator's word: its sampling fields (a field it leaves out takes the published value) and its
//     eos_token_id (absent: the published [2, 11]) are used as they are;
//   - otherwise the snapshot's file is used only when it carries sampling keys (NVIDIA's own does: do_sample, temperature,
//     top_p, eos [2, 11]); a file without them (SergiioB's: eos 11, nothing else) gives the published defaults, and its
//     eos_token_id is ADDED to the published set (an export listing another end id keeps it).
// Throws std::invalid_argument naming the key of a malformed field.
inline Generation resolve_generation(const nlohmann::json* snapshot, const nlohmann::json* explicit_file,
                                     const std::string& explicit_path = "", const std::string& snapshot_path = "") {
  Generation out = published_generation();
  const auto apply_sampling = [&](const nlohmann::json& g) {
    // The file's keys OVERLAY the published values (review I2): {"temperature": 0.6} keeps do_sample true and top_p 0.95; a
    // field neither names is not HF's default (top-k 50, greedy) but the published one.
    nlohmann::json merged = {{"do_sample", true}, {"temperature", 1.0}, {"top_p", 0.95}};
    if (!g.is_object()) throw std::invalid_argument("generation_config.json is not an object");
    for (auto it = g.begin(); it != g.end(); ++it) merged[it.key()] = it.value();
    const server::Sampling s = server::generation_sampling(merged);
    out.sampling = s;
    if (!g.contains("top_k")) out.sampling.top_k = 0;
  };
  const auto eos_of = [](const nlohmann::json& g) {
    return g.is_object() && g.contains("eos_token_id") ? server::generation_eos_ids(g) : std::vector<uint32_t>{};
  };
  std::set<uint32_t> eos(out.eos.begin(), out.eos.end());
  if (explicit_file != nullptr) {
    apply_sampling(*explicit_file);
    const std::vector<uint32_t> e = eos_of(*explicit_file);
    if (!e.empty()) eos = std::set<uint32_t>(e.begin(), e.end());
    out.stops = server::generation_stop_strings(*explicit_file);
    out.source = "--generation-config " + explicit_path;
  } else if (snapshot != nullptr) {
    const std::vector<uint32_t> e = eos_of(*snapshot);
    eos.insert(e.begin(), e.end());
    out.stops = server::generation_stop_strings(*snapshot);
    if (has_sampling_keys(*snapshot)) {
      apply_sampling(*snapshot);
      out.source = snapshot_path + " (the snapshot's file carries sampling keys)";
    } else {
      out.source += "; the snapshot's generation_config.json names no sampling key" +
                    std::string(e.empty() ? "" : " (its eos_token_id is added)");
    }
  }
  out.eos.assign(eos.begin(), eos.end());
  return out;
}

// ---- what b70-serve refuses before the device -----------------------------------------------------------------------------------
// What b70-serve parsed that Bolt has no use for (the rest - --mtp, --spec, --kv-cache int8, the prefill backends, --device with
// --pp 2, a split without --pp 2 - is cli/bolt_decode.h's check_args, shared with b70-decode), each refusal by name
// (tests/server/bolt_request_test.cc, and cli_reject_serve_bolt_* through the binary on the box).
// Spec 26c: --spec dflash / lookup and their flags (--spec-max, --spec-cost, --draft-model, --draft-format, --draft-q,
// --spec-min-match, --spec-history) are cli/bolt_spec.h's (check_spec); what stays refused here is an MTP head's.
struct ServeFlags {
  bool mtp = false;              // --mtp N | auto
  std::string spec;              // --spec's value ("" when not given): off | mtp | lookup | dflash
  bool proposer_tuning = false;  // --mtp-max, --mtp-cost (an MTP head's tuning)
  bool lookup_tuning = false;    // --spec-min-match, --spec-history (prompt lookup's)
  bool draft_vocab = false;      // --draft-vocab, --draft-vocab-ids
  bool qwen4exp_flags = false;   // --layers, --ple-dir, --tokenizer
};

inline void check_serve_flags(const ServeFlags& f) {
  if (f.mtp || f.spec == "mtp")
    throw std::runtime_error("Nemotron 3.5 Bolt has no MTP head (spec 25 §0: num_nextn_predict_layers 0): drop --mtp / --spec mtp");
  if (f.proposer_tuning)
    throw std::runtime_error("--mtp-max and --mtp-cost belong to an MTP head; Nemotron 3.5 Bolt has none (its proposers are "
                             "--spec dflash / lookup, tuned by --spec-max and --spec-cost - spec 26)");
  if (f.lookup_tuning && f.spec != "lookup")
    throw std::runtime_error("--spec-min-match and --spec-history tune prompt lookup: they need --spec lookup");
  if (f.draft_vocab)
    throw std::runtime_error("--draft-vocab and --draft-vocab-ids narrow an MTP head's or a DFlash drafter's vocabulary; Nemotron 3.5 "
                             "Bolt has neither (LiLiCorr reads its whole lm_head, spec 26)");
  if (f.qwen4exp_flags)
    throw std::runtime_error("--layers, --ple-dir and --tokenizer belong to Qwen3.8-Flash-Next (model_type qwen4_exp, spec 21e); "
                             "this checkpoint is Nemotron 3.5 Bolt (b70-decode --layers is its development mode)");
}

// The served context is capped at the trained 262144 (Decision 14) - refused before the device (the planner refuses it too, after
// the cards are open).
inline void check_serve_max_len(bool is_auto, uint32_t value, uint32_t trained_max_len) {
  if (!is_auto && value > trained_max_len)
    throw std::runtime_error("--max-len " + std::to_string(value) + " exceeds Nemotron 3.5 Bolt's trained context " +
                             std::to_string(trained_max_len) + " (max_position_embeddings; spec 25 Decision 14)");
}

// ---- the top-p filter without a top-k (review I1) -------------------------------------------------------------------------------
// With no top-k (Bolt's default) server::filter_probs partial_sorts the WHOLE 131072-row vocabulary for every sampled id: measured
// 17.2 ms on the Mac (k 50: 0.15 ms) against a ~16.5 ms decode step. This is the same filter without the full sort: the softmax
// is normalised once over every row (one pass, no sort; rows more than 40 nats under the top are not added: < 6e-13 of the sum), then the head is sorted in growing prefixes (64, 256, 1024 ...) until the
// cumulative mass reaches top_p. Same ids in the same descending order and the same kept count as filter_probs on distinct
// logits (its probabilities differ only in the last bits: the normaliser is summed in id order, not sorted order); among EQUAL
// logits the order is ascending id (filter_probs' partial_sort leaves it unspecified). It is Bolt's alone - the shared
// filter_probs and every other model's sampling are untouched. Only for top_k 0 and no softcap / multiplier; else filter_probs.
inline server::FilteredProbs filter_probs_top_p(const float* logits, uint32_t vocab_used, const server::Sampling& s) {
  if (s.top_k != 0 || s.softcap > 0.0f || s.logit_mult != 1.0f) return server::filter_probs(logits, vocab_used, s);
  const float temperature = s.temperature > 0.0f ? s.temperature : 1.0f;
  float top = logits[0];
  for (uint32_t i = 1; i < vocab_used; ++i) top = std::max(top, logits[i]);
  double sum = 0.0;
  for (uint32_t i = 0; i < vocab_used; ++i) {
    const double x = static_cast<double>((logits[i] - top) / temperature);
    if (x > -40.0) sum += std::exp(x);   // a row 40 nats under the top adds < 4.3e-18: dropped (all of them < 6e-13 of the sum)
  }
  const double top_p = std::clamp(static_cast<double>(s.top_p), 0.0, 1.0);
  std::vector<uint32_t> idx(vocab_used);
  for (uint32_t i = 0; i < vocab_used; ++i) idx[i] = i;
  const auto before = [logits](uint32_t a, uint32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
  server::FilteredProbs f;
  size_t sorted = 0;
  for (size_t n = 64;; n *= 4) {
    n = std::min<size_t>(n, vocab_used);
    std::partial_sort(idx.begin() + sorted, idx.begin() + n, idx.end(), before);   // [0, sorted) are already the head
    sorted = n;
    double cumulative = 0.0;
    uint32_t kept = 0;
    f.probs.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      f.probs[i] = std::exp(static_cast<double>((logits[idx[i]] - top) / temperature)) / sum;   // (kept rows are never pruned out of the sum above: the nucleus holds the mass)
      cumulative += f.probs[i];
      if (cumulative >= top_p) {
        kept = uint32_t(i + 1);
        break;
      }
    }
    if (kept == 0 && n == vocab_used) kept = uint32_t(n);
    if (kept != 0) {
      f.probs.resize(kept);
      f.ids.assign(idx.begin(), idx.begin() + kept);
      break;
    }
  }
  f.mass = 0.0;
  for (double p : f.probs) f.mass += p;
  return f;
}

// ---- spec 26c: the drafter as the adapter sees it ---------------------------------------------------------------------------
// b70-serve binds a runtime::dflash::LiliDrafter beside the engine (cli/bolt_spec.h); the host test a fake. One block at the
// engine's pos (the anchor its pending id) through the device's GREEDY walk; the readback carries every slot's 8 candidates,
// their log-probs and the whole lattice [7][8 previous][8 next], so a sampled request walks on the host (below).
struct BoltDraftHooks {
  std::function<const runtime::dflash::LiliDraftOut&(uint32_t k)> draft;
  bool on() const { return static_cast<bool>(draft); }
};
// The sampled draft distribution (spec 26 Decision 6, PROVISIONAL until r63.accept): `lattice` - q_s = softmax(realized / T)
// over slot s's 8 candidates, realized = lattice[s][previous] (vLLM's) - or `lattice+lp` - softmax((realized + lp_s) / T), the
// alternative 26a records. Either is lossless through accept_sparse; only acceptance moves.
enum class BoltDraftQ { Lattice, LatticeLp };
inline bool parse_bolt_draft_q(const std::string& v, BoltDraftQ& out) {
  if (v == "lattice") out = BoltDraftQ::Lattice;
  else if (v == "lattice+lp") out = BoltDraftQ::LatticeLp;
  else return false;
  return true;
}
inline const char* bolt_draft_q_name(BoltDraftQ q) { return q == BoltDraftQ::LatticeLp ? "lattice+lp" : "lattice"; }

// ---- the adapter ------------------------------------------------------------------------------------------------------------
// Spec 26c: speculative decoding (`spec_k` > 0) - `--spec dflash` (the LiLiCorr drafter: mtp_k() = --spec-max; step_many()
// drafts a block through BoltDraftHooks, verifies at M = K + 1 and accepts on the host) or `--spec lookup` (verify_k():
// step_drafts() over the server's prompt-lookup drafts). Greedy: the device's walk, then the longest prefix the target's argmax
// agrees with (D2: the output is plain decode's). Sampled: the drafts are DRAWN HERE - the walk over the readback's lattice
// (slot s's scores = lattice[s][previous], + lp under `lattice+lp`; q_s = server::dflash_q at the request's T), the request's
// RNG - then server::accept_sparse_rows against the target's filtered p (filter_probs_top_p: the plain path's filter) - lossless
// whatever q is (D4); lookup's point mass through accept_point_mass_rows. RNG order: the k draft draws, then the acceptance's
// uniforms, then its one correction / bonus draw. The commit is deferred to the next call that touches the engine (truncate_to
// may commit fewer), as Muse's; the pending id it leaves was CHOSEN by the acceptance (`decided_`), so the next step does not
// draw it again. A request with a repetition penalty (sampled) decodes plainly: its p depends on ids a run has not committed.
// Without spec_k every path is 25e's, byte for byte.
// An engine with BoltEngine's speculative calls (verify / commit / ...): runtime::bolt::BoltEngine and the spec tests' fakes;
// 25e's fakes have none, and the adapter steps them plainly.
template <class E, class = void>
struct HasVerify : std::false_type {};
template <class E>
struct HasVerify<E, std::void_t<decltype(std::declval<E&>().verify(static_cast<const uint32_t*>(nullptr), 0u)),
                                decltype(std::declval<E&>().commit(0u, 0u))>> : std::true_type {};

template <class Engine>
struct BoltEngineAdapterT : server::EngineIface {
  static constexpr bool kSpec = HasVerify<Engine>::value;
  // `vocab_used`: the tokenizer's id count (131072); ids at or above it are never emitted.
  BoltEngineAdapterT(Engine& engine, uint32_t vocab_used, uint32_t spec_k = 0, bool lookup = false, BoltDraftHooks drafter = {},
                     BoltDraftQ q = BoltDraftQ::Lattice)
      : eng(engine), vocab_used_(vocab_used), host_logits_(engine.vocab()), k_(spec_k), lookup_(lookup),
        df_(std::move(drafter)), q_(q) {
    if (vocab_used_ == 0 || vocab_used_ > engine.vocab())
      throw std::runtime_error("BoltEngineAdapter: vocab_used " + std::to_string(vocab_used_) +
                               " is outside Nemotron 3.5 Bolt's vocabulary (" + std::to_string(engine.vocab()) + " rows)");
    if (k_ > kMaxK) throw std::runtime_error("BoltEngineAdapter: K " + std::to_string(k_) + " exceeds 7 (LiLiCorr's block - 1)");
    if (lookup_ && df_.on()) throw std::runtime_error("BoltEngineAdapter: --spec dflash and --spec lookup: pick one");
    if (k_ > 0 && !lookup_ && !df_.on())
      throw std::runtime_error("BoltEngineAdapter: K > 0 without a proposer (--spec dflash or --spec lookup)");
    if (k_ > 0 && !kSpec) throw std::runtime_error("BoltEngineAdapter: K > 0 over an engine without verify lists");
    if (k_ > 0) spec_logits_.resize(size_t(kMaxK + 1) * engine.vocab());
  }

  static constexpr uint32_t kMaxK = 7;   // LiLiCorr's block of 8 - 1 (runtime::bolt::kVerifyRows - 1)

  void reset() override {
    pending_ = false;   // the session is discarded; nothing to commit
    decided_ = false;
    eng.reset();
    new_request_ = true;
  }

  // EngineAdapter's measurement line: the previous request's generation, printed when the next request's prefill starts.
  void prefill(const std::vector<uint32_t>& ids) override {
    report();
    flush_commit();
    eng.prefill(ids);
    new_request_ = true;
    decided_ = false;
    gen_start_ = std::chrono::steady_clock::now();
  }

  uint32_t step(const server::Sampling& sampling) override {
    flush_commit();
    // The pending id is the argmax of the last replay (or of the prefill's head row). A sampled request replaces it with a draw
    // from that same row; a greedy one keeps it unless it is a row the tokenizer does not define. Spec 26c: an id an acceptance
    // chose (decided_) is kept as it is.
    const bool penalised = sampling.repetition_penalty != 1.0f && sampling.seen != nullptr;
    begin_request(sampling);
    if (!decided_) {
      if (!sampling.greedy) choose_on_host(sampling, penalised);
      else mask_pending();
    }
    decided_ = false;
    const uint32_t id = eng.generate(1)[0];
    ++gen_tokens_;
    return id;
  }

  uint32_t max_len() override { return eng.max_len(); }
  uint32_t pos() override { return pending_ ? pending_pos_ + pending_j_ + 1 : eng.pos(); }

  // --- spec 26c: speculative decoding ----------------------------------------------------------------------------------------
  uint32_t mtp_k() override { return df_.on() ? k_ : 0; }
  uint32_t verify_k() override { return lookup_ ? k_ : 0; }

  // One LiLiCorr iteration at pos = n with x_n pending: the drafter's block (greedy: the device's walk; sampled: the host's walk
  // over its lattice), verify at M = K + 1, accept on the host; returns x_n, d_1 .. d_j. The commit is deferred. (An engine
  // without BoltEngine's verify / commit - 25e's host fakes - steps plainly: no spec_k is accepted for it.)
  std::vector<uint32_t> step_many(const server::Sampling& sampling, uint32_t k_req) override {
    if constexpr (kSpec) {
      return step_many_spec(sampling, k_req);
    } else {
      (void)k_req;
      return {step(sampling)};
    }
  }
  std::vector<uint32_t> step_drafts(const server::Sampling& sampling, const Proposer& propose) override {
    if constexpr (kSpec) {
      return step_drafts_spec(sampling, propose);
    } else {
      (void)propose;
      return {step(sampling)};
    }
  }

  // Keep x_n, d_1 .. d_m of the last iteration (pos = n + m + 1), pending d_{m+1} (or the run's next id when m = j).
  void truncate_to(uint32_t p) override {
    if (!pending_) {
      if (p == eng.pos()) return;
      throw std::logic_error("BoltEngineAdapter::truncate_to: no speculative run to shorten");
    }
    if (p < pending_pos_ + 1 || p > pending_pos_ + pending_j_ + 1)
      throw std::logic_error("BoltEngineAdapter::truncate_to: pos " + std::to_string(p) + " is outside the last run [" +
                             std::to_string(pending_pos_ + 1) + ", " + std::to_string(pending_pos_ + pending_j_ + 1) + "]");
    const uint32_t m = p - pending_pos_ - 1;
    gen_tokens_ -= pending_j_ - m;
    pending_ = false;
    if constexpr (kSpec) eng.commit(m, m < pending_j_ ? pending_drafts_[m] : pending_next_);
  }

  void flush_commit() {
    if (!pending_) return;
    pending_ = false;
    if constexpr (kSpec) eng.commit(pending_j_, pending_next_);
  }

  uint64_t drafter_key() override {
    if constexpr (kSpec) return eng.drafter_key();
    else return 0;
  }
  uint64_t spec_iterations() const { return iters_; }   // tests
  uint64_t spec_accepted() const { return accepted_; }
  uint64_t spec_drafted() const { return drafted_; }

 private:
  std::vector<uint32_t> step_many_spec(const server::Sampling& sampling, uint32_t k_req) {
    flush_commit();
    const bool penalised = sampling.repetition_penalty != 1.0f && sampling.seen != nullptr;
    uint32_t k = std::min({k_req, k_, eng.max_verify_k()});
    if (!df_.on() || vocab_used_ < eng.vocab() || (penalised && !sampling.greedy)) k = 0;   // plain (masked / penalised) steps
    ++iters_at_k_[k];
    if (k == 0) return {step(sampling)};
    settle_pending(sampling);
    const uint32_t n = eng.pos(), x = eng.pending();
    const runtime::dflash::LiliDraftOut& o = df_.draft(k);
    std::vector<uint32_t> drafts(k);
    server::AcceptResult a{};
    if (sampling.greedy) {
      for (uint32_t j = 0; j < k; ++j) drafts[j] = o.ids[j];
      eng.verify(drafts.data(), k);
      a = server::accept_greedy(eng.verify_ids(), drafts.data(), k);
    } else {
      const float T = server::dflash_temperature(sampling);
      server::SparseQ q[kMaxK];
      uint32_t prev = 0;
      for (uint32_t j = 0; j < k; ++j) {   // the walk: slot j's scores are the lattice row of slot j - 1's pick
        float v[runtime::dflash::LiliDraftOut::kTopK];
        for (uint32_t c = 0; c < runtime::dflash::LiliDraftOut::kTopK; ++c)
          v[c] = o.lattice[j][prev][c] + (q_ == BoltDraftQ::LatticeLp ? o.lp[j][c] : 0.0f);
        q[j] = server::dflash_q(o.cand[j], v, runtime::dflash::LiliDraftOut::kTopK, T);
        prev = draw_index(q[j]);
        drafts[j] = o.cand[j][prev];
      }
      eng.verify(drafts.data(), k);
      eng.read_verify_logits(spec_logits_.data(), k + 1);
      a = server::accept_sparse_rows(spec_logits_.data(), q, drafts.data(), k, eng.vocab(), row_filter(sampling), rng_);
    }
    return keep(n, x, drafts, k, a);
  }

  // One lookup iteration (spec 19e's contract): propose(x_n) -> d_1 .. d_k, verify, accept (greedy: argmax; sampled: the point
  // mass, lossless).
  std::vector<uint32_t> step_drafts_spec(const server::Sampling& sampling, const Proposer& propose) {
    flush_commit();
    const bool penalised = sampling.repetition_penalty != 1.0f && sampling.seen != nullptr;
    begin_request(sampling);
    if (!lookup_ || vocab_used_ < eng.vocab() || (penalised && !sampling.greedy)) {
      ++iters_at_k_[0];
      return {step(sampling)};
    }
    settle_pending(sampling);
    const uint32_t x = eng.pending();
    const std::vector<uint32_t> d = propose(x);
    const uint32_t k = std::min<uint32_t>({uint32_t(d.size()), k_, eng.max_verify_k()});
    ++iters_at_k_[k];
    if (k == 0) return {step(sampling)};
    const uint32_t n = eng.pos();
    std::vector<uint32_t> drafts(d.begin(), d.begin() + k);
    eng.verify(drafts.data(), k);
    server::AcceptResult a{};
    if (sampling.greedy) {
      a = server::accept_greedy(eng.verify_ids(), drafts.data(), k);
    } else {
      eng.read_verify_logits(spec_logits_.data(), k + 1);
      a = server::accept_point_mass_rows(spec_logits_.data(), drafts.data(), k, eng.vocab(), row_filter(sampling), rng_);
    }
    return keep(n, x, drafts, k, a);
  }

 public:
  // --- spec 7 (plan 25e Review Focus 4) -----------------------------------------------------------------------------------
  uint32_t block() override { return Engine::kBlock; }
  size_t state_bytes() override { return eng.state_bytes(); }
  size_t kv_bytes(uint32_t n) override { return eng.kv_bytes(n); }
  uint64_t kv_form() override { return store_key(eng.bf16_state()); }
  // Spec 26c: a deferred commit is made before a save (the request-end snapshot follows a run the server kept whole); a restore
  // discards it, as reset() does.
  void save_state(void* host) override {
    flush_commit();
    eng.save_state(host);
  }
  void load_state(const void* host, uint32_t p) override {
    pending_ = false;
    decided_ = false;
    eng.load_state(host, p);
    new_request_ = true;
  }
  void save_kv(uint32_t b, uint32_t e, void* host) override {
    flush_commit();
    eng.save_kv(b, e, host);
  }
  void load_kv(uint32_t b, uint32_t e, const void* host) override { eng.load_kv(b, e, host); }
  void set_block_hook(BlockHook hook) override { eng.set_block_hook(std::move(hook)); }
  void ingest(const std::vector<uint32_t>& ids) override {
    flush_commit();
    eng.ingest(ids);
    new_request_ = true;
    decided_ = false;
  }

  // The pending id for a sampled request, chosen from the logits row the engine left: penalised first (HF's order: the repetition
  // penalty is a logits processor, temperature / top-k / top-p the warpers after it), then drawn.
  void choose_on_host(const server::Sampling& sampling, bool penalised) {
    eng.read_logits_into(host_logits_.data());
    if (penalised)
      server::apply_repetition_penalty(host_logits_.data(), vocab_used_, sampling.seen->distinct(), sampling.repetition_penalty);
    eng.set_token(server::draw(filter_probs_top_p(host_logits_.data(), vocab_used_, sampling), rng_));
    ++sampled_;
  }

  // Greedy: a device argmax on an id the tokenizer does not define becomes the masked row's argmax (the lowest id among equal
  // maxima, the device argmax's tie rule).
  void mask_pending() {
    if (eng.pending() < vocab_used_) return;
    eng.read_logits_into(host_logits_.data());
    const float* row = host_logits_.data();
    eng.set_token(uint32_t(std::max_element(row, row + vocab_used_) - row));
    ++masked_;
  }

  void report() {
    if (k_ > 0 && iters_ > 0) {   // spec 26c: the previous request's speculative line (the mean accepted length is D5's)
      std::fprintf(stderr, "%s: K %u, %llu iterations, %llu/%llu drafts accepted (%.3f), %.3f tokens a verify, iterations at K "
                   "0..7:", lookup_ ? "lookup" : "dflash", k_, (unsigned long long)iters_, (unsigned long long)accepted_,
                   (unsigned long long)drafted_, drafted_ ? double(accepted_) / double(drafted_) : 0.0,
                   1.0 + double(accepted_) / double(iters_));
      for (uint32_t k = 0; k <= kMaxK; ++k) std::fprintf(stderr, "%s%llu", k ? "/" : " ", (unsigned long long)iters_at_k_[k]);
      std::fprintf(stderr, "\n");
    }
    iters_ = accepted_ = drafted_ = 0;
    for (auto& c : iters_at_k_) c = 0;
    if (gen_tokens_ == 0) return;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gen_start_).count();
    std::fprintf(stderr, "gen: %u tokens, %.3f ms, %.4f ms/token\n", gen_tokens_, ms, ms / gen_tokens_);
    std::fflush(stderr);
    gen_tokens_ = 0;
  }

  uint64_t masked() const { return masked_; }     // greedy ids moved off an undefined row (tests)
  uint64_t sampled() const { return sampled_; }   // rows drawn on the host (tests)

  Engine& eng;

 private:
  // A request's first step (after a prefill, an ingest, a restore or a reset) seeds the RNG: from its seed, or the random device.
  void begin_request(const server::Sampling& sampling) {
    if (new_request_) {
      rng_.seed(sampling.has_seed ? sampling.seed : uint64_t(std::random_device{}()));
      new_request_ = false;
    }
  }
  // Spec 26c: the pending id before a speculative iteration - step()'s choice (sampled: drawn from the row the engine left;
  // greedy: masked), unless an acceptance chose it.
  void settle_pending(const server::Sampling& sampling) {
    begin_request(sampling);
    if (!decided_) {
      if (!sampling.greedy) choose_on_host(sampling, false);
      else mask_pending();
    }
    decided_ = true;
  }
  server::RowFilter row_filter(const server::Sampling& sampling) const {
    const uint32_t used = vocab_used_;
    return [used, sampling](const float* row) { return filter_probs_top_p(row, used, sampling); };
  }
  // One candidate index from a sparse q (in order; the last positive one on a rounding shortfall).
  uint32_t draw_index(const server::SparseQ& q) {
    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
    double acc = 0.0;
    uint32_t last = 0;
    for (uint32_t c = 0; c < q.n; ++c) {
      if (!(q.probs[c] > 0.0)) continue;
      last = c;
      acc += q.probs[c];
      if (u < acc) return c;
    }
    return last;
  }
  // The iteration's bookkeeping: the deferred commit and the ids returned.
  std::vector<uint32_t> keep(uint32_t n, uint32_t x, const std::vector<uint32_t>& drafts, uint32_t k,
                             const server::AcceptResult& a) {
    pending_ = true;
    decided_ = true;
    pending_pos_ = n;
    pending_j_ = a.accepted;
    pending_next_ = a.next_token;
    pending_drafts_ = drafts;
    ++iters_;
    drafted_ += k;
    accepted_ += a.accepted;
    std::vector<uint32_t> out{x};
    out.insert(out.end(), drafts.begin(), drafts.begin() + a.accepted);
    gen_tokens_ += static_cast<uint32_t>(out.size());
    return out;
  }

  uint32_t vocab_used_;
  std::vector<float> host_logits_;
  std::mt19937_64 rng_;
  bool new_request_ = true;   // the next step() is a request's first
  uint32_t gen_tokens_ = 0;
  uint64_t masked_ = 0, sampled_ = 0;
  std::chrono::steady_clock::time_point gen_start_{};
  // Spec 26c.
  uint32_t k_ = 0;
  bool lookup_ = false;
  BoltDraftHooks df_;
  BoltDraftQ q_ = BoltDraftQ::Lattice;
  std::vector<float> spec_logits_;   // [K + 1][vocab]: the verify rows' raw logits (sampled acceptance)
  bool pending_ = false;             // a run's commit is deferred
  bool decided_ = false;             // the pending id was chosen by an acceptance / step(): never drawn again
  uint32_t pending_pos_ = 0, pending_j_ = 0, pending_next_ = 0;
  std::vector<uint32_t> pending_drafts_;
  uint64_t iters_ = 0, drafted_ = 0, accepted_ = 0;
  uint64_t iters_at_k_[kMaxK + 1] = {};
};

}  // namespace cli::bolt
