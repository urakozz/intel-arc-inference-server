#pragma once
// Spec 21-QFN-v1c: b70-decode's Qwen3.8-Flash-Next path - the CLI dispatches on config.json's model_type
// (loader::is_qwen4exp_checkpoint: qwen4_exp / qwen4_exp_text) and runs runtime::qwen4exp::Qwen4ExpEngine on one or
// two cards (cli/kolibri_decode.h's shape). `--ids` and `--bench` exactly as the other paths print them (ids on
// stdout, one per line; the bench row on stdout); the prompt goes through the decode list one replay per id, or
// (spec 21-QFN-v1d, --prefill / --prefill-length) through Qwen4ExpEngine::prefill in chunks.
//
//   --layers N|auto    the development mode (spec 21-QFN-v1 §6) - load only layers [0, N) of the checkpoint, then the final
//                      mixer and the head; `auto` is the planner's largest N that fits
//                      (runtime::qwen4exp::layers_that_fit). Without it the whole model loads with the expert tier
//                      (plan 21-QFN-v2a Task 8: settle) - 512 experts x 48 layers do not fit two B70s, so each card keeps
//                      the most-routed experts its VRAM holds and the kernels read the rest from pinned host memory
//   the PLE int8 file  <snapshot>-ple-int8/, beside the checkpoint (loader::q4_ple_dir; no flag)
//   --parallel auto|1|pp|tp   auto (the default): pp on two visible cards for the whole model, else one card (the
//                      --layers development mode: cli::qwen4exp_auto_devices); 1; pp (spec 16b's pieces: the
//                      materialised H crosses); tp (plan 21v2b):
//                      tensor parallel - each card holds half of every layer (runtime::qwen4exp::Qwen4ExpTpEngine), the
//                      two all-reduces a layer over P2P;
//   --pipeline-split auto|N   16b's switch and parser; auto is
//                      runtime::qwen4exp::pp_split (pp_balance over the per-layer bytes at the session's max_len)
//   --lm-head bf16|int8 (bf16 by default, as b70-decode's other paths), --max-len N|auto (<= the trained 262144)
//   --bench [--depth N] [--tg N]; B70_Q4_ATTN=flash|eager picks the decode attention (and the prefill's)
// Spec 21-QFN-v1d: the prompt through Qwen4ExpEngine::prefill in chunks:
//   --prefill          (--ids) the prompt prefilled, then --n greedy ids
//   --prefill-length N (--bench) depth N prefilled and timed (the pp row), then --tg decoded
//   --prefill-chunk C  the chunk width, <= 4096 (the argument parser's cap; the default: the planner's 2048 or 4096,
//                      runtime::qwen4exp::pf_chunk_choice; a C past 2048 plans the 4096-row scratch)
//   --prefill-backend l0  the only backend (the default here)
// A prefilling run plans the prefill scratch (and, on two cards, the prefill hand-off) into --layers auto, --max-len
// auto and the fit.
// Refused before the device, each by name: --prefill-backend l0-int8 (spec 5's h8 linears rotate in whole 1024-k
// Hadamard blocks; the hidden is 2560), --spec mtp
// (b70-decode runs plain; the head - spec 21-QFN-v1e - is b70-serve's and qwen4exp_mtp_test's), --kv-cache int8 (decision 8:
// bf16 first; kv8.cl hard-codes 24 q / 4 kv heads) and --profile (Task 7 times it with --bench).
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <print>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cli/max_len.h"
#include "cli/pipeline_args.h"
#include "l0/context.h"
#include "loader/lm_head_int8.h"
#include "loader/qwen4exp_layout.h"
#include "loader/qwen4exp_loader.h"
#include "loader/snapshot.h"
#include "model/qwen4exp.h"
#include "runtime/vram_bars.h"
#include "runtime/prefill_backend.h"
#include "runtime/qwen4exp/qwen4exp_engine.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"
#include "runtime/qwen4exp/qwen4exp_tp_engine.h"

namespace cli::qwen4exp {

// The bench prompt, cycled to --depth: the first 42 ids of tests/golden/prompts/q4exp_short.ids (prose.txt through
// the original's tokenizer - 21-QFN-v1a: Qwen3.8's committed ids re-encode identically).
inline constexpr uint32_t kQwen4ExpBenchPrompt[] = {
    760,   72103, 506,   37119, 557,  11012, 3213, 310,   6512, 279,   61789, 272,  1072, 2272,
    279,   197616, 2271, 13,    469,  68042, 29123, 7247, 383,  279,   1387,  12615, 1345, 279,
    49813, 78911, 1141,  20459, 13,   3113,  7840, 279,  2981, 1000,  381,   16850, 1495, 13};
inline constexpr size_t kQwen4ExpBenchPromptLen = sizeof(kQwen4ExpBenchPrompt) / sizeof(kQwen4ExpBenchPrompt[0]);


// True when `path` resolves to a snapshot whose config.json says model_type qwen4_exp (or qwen4_exp_text). A path
// that does not resolve is not this family: the caller's own flow then reports it.
inline bool is_qwen4exp(const std::string& path) {
  try {
    return loader::is_qwen4exp_checkpoint(loader::resolve_snapshot(path));
  } catch (const std::exception&) {
    return false;
  }
}

struct DecodeArgs {
  std::string path, ids_path;
  uint32_t n = 0, depth = 4096, tg = 256;
  bool bench = false;
  MaxLenArg max_len{false, 16384};
  size_t reserve = 0;
  size_t card_bytes = 0;        // the smallest visible card's memory (cli::hardware_gate): the fit checks before a device opens
  uint32_t cards = 1;           // the visible cards (cli::hardware_gate): --parallel auto's choice (devices_of)
  size_t prefix_bytes = 0;     // b70-serve's explicit --prefix-cache-gb: pinned host memory settle keeps for it (auto
                                // sizes itself after the load from what is left)
  loader::LmHeadForm lm_head = loader::LmHeadForm::Checkpoint;
  uint32_t layers = 0;          // --layers N (0: not given)
  bool layers_auto = false;     // --layers auto
  PipelineArgs pipe;            // devices as parsed (1 when --parallel was not given)
  bool pp_given = false;
  // Spec 21-QFN-v1d: --prefill (--ids), --prefill-length N (--bench at depth N: the pp row), --prefill-chunk C (0 = the planner's),
  // --prefill-backend as parsed (l0 only).
  bool prefill = false, pp = false, pp_backend_given = false;
  uint32_t pp_chunk = 0;
  runtime::PrefillBackend pp_backend = runtime::PrefillBackend::L0;
  // What else was asked for, for the refusals.
  bool mtp = false, kv8 = false, profile = false;
  // Spec 21-QFN-v1e: b70-serve's --spec mtp / --spec loads the MTP head - the planner (settle) then carries its weights, its
  // buffers and the verify rows (runtime::qwen4exp::plan(..., mtp)). b70-decode never sets it.
  bool load_mtp = false;
  bool int8_head() const { return lm_head == loader::LmHeadForm::Int8; }
  bool prefills() const { return prefill || pp; }
};

inline uint32_t devices_of(const DecodeArgs& a) {
  if (a.pipe.tp()) return 2u;
  return a.pp_given ? a.pipe.devices : qwen4exp_auto_devices(a.cards, a.layers == 0 && !a.layers_auto);
}

// Every refusal that needs no device and no tensor - in this order, each by name (tests/CMakeLists.txt's
// cli_reject_qwen4exp_*).
inline void check_args(const DecodeArgs& a) {
  if (a.pp_backend_given && a.pp_backend == runtime::PrefillBackend::L0Int8)
    throw std::runtime_error("Qwen3.8-Flash-Next prefills on the l0 backend only, not l0-int8: its h8 linears "
                             "rotate in whole 1024-k Hadamard blocks and this model's hidden is 2560 - "
                             "--prefill-backend l0, or omit it");
  if (a.mtp)
    throw std::runtime_error("b70-decode runs Qwen3.8-Flash-Next plain: its MTP head is served by b70-serve "
                             "--spec mtp and measured by qwen4exp_mtp_test - drop --spec mtp");
  if (a.kv8)
    throw std::runtime_error("--kv-cache int8: Qwen3.8-Flash-Next's KV is bf16 (int8 later - kv8.cl "
                             "hard-codes 24 q / 4 kv heads and this model has 2)");
  if (a.profile)
    throw std::runtime_error("--profile is not built for Qwen3.8-Flash-Next (--bench times it)");
  if (devices_of(a) == 1 && a.pipe.have_split)
    throw std::runtime_error("--pipeline-split belongs to --parallel pp (the layer device 1 starts at)");
  if (a.pipe.tp() && !a.int8_head())
    throw std::runtime_error("--parallel tp splits the int8 lm_head by vocabulary between the cards: add --lm-head int8");
}

inline std::vector<uint32_t> read_ids(const std::string& path, uint32_t vocab) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open --ids file '" + path + "'");
  std::vector<uint32_t> ids;
  std::string w;
  while (f >> w) {
    if (w.empty() || w.find_first_not_of("0123456789") != std::string::npos || w.size() > 10)
      throw std::runtime_error("--ids file '" + path + "' is not whitespace-separated ids");
    const unsigned long long v = std::stoull(w);
    if (v >= vocab)
      throw std::runtime_error("--ids file '" + path + "': id " + w + " is outside Qwen3.8-Flash-Next's vocabulary (" +
                               std::to_string(vocab) + " rows)");
    ids.push_back(uint32_t(v));
  }
  if (ids.empty()) throw std::runtime_error("--ids file '" + path + "' is empty");
  return ids;
}

struct Settled {
  model::Qwen4ExpDesc desc;   // the truncated descriptor
  model::Q4Placement placement;
  uint32_t max_len = 0;
  std::optional<runtime::qwen4exp::ExpertTier> tier;   // the whole model's (load_qwen4exp's tier); with --layers N
                                                       // none: every expert resident (or B70_Q4_RESIDENT's, the loader)
  std::optional<runtime::qwen4exp::ExpertTier> tier1;  // --parallel tp: card 1's (`tier` is card 0's)
  uint32_t pf_rows = runtime::qwen4exp::kPfC;          // the prefill scratch's chunk rows (runtime::qwen4exp::pf_chunk_choice)
};

// The whole model's --max-len auto (spec 21-QFN-v2 §9): each GB of context is a GB of experts out of VRAM, so the largest
// of 32768 / 65536 / 131072 / 262144 whose projected decode stays within 5 % of 32768's (runtime::qwen4exp::
// tier_decode_ms, derived).
inline constexpr uint32_t kWholeMinLen = 32768;
using runtime::qwen4exp::tier_decode_ms;

// The prefill chunk's line: what the 4096-row chunk's scratch keeps from the experts and the projected decode either way.
inline void print_pf_rows(const runtime::qwen4exp::PfChunkChoice& c) {
  size_t held = 0, grow = 0;
  for (size_t i = 0; i < c.held.size(); ++i) {
    held = std::max(held, c.held[i]);
    grow = std::max(grow, c.grow[i]);
  }
  std::print(stderr, "prefill chunk: {:d} rows (the 4096-row chunk's scratch takes {:.3f} GB a card from the experts, "
             "{:.3f} GB of it while decoding - the rest is lent back between prompts; projected decode {:.2f} -> {:.2f} ms "
             "a token, {} {:.0f}%)\n", c.rows, grow / 1e9, held / 1e9, c.ms_narrow, c.ms_wide,
             c.rows == runtime::qwen4exp::kPfCWide ? "within" : "over", 100 * runtime::qwen4exp::kPfWideDecodeBar);
}

// The layer count, placement and max_len for this run, from the planner (before the load: each layer goes straight
// onto its device). `full` is the checkpoint's descriptor (all its layers). Prints the choice and the plan.
// Without --layers the whole model loads with the expert tier: each card's VRAM left after the rest of the plan holds
// the most-routed experts (the routing profile beside the checkpoint, else id order), the rest pinned host memory.
// B70_Q4_RESIDENT (a test placement, runtime::qwen4exp::fill_tier) forces the residents at any layer count.
// Host memory (plan 21-QFN-v2a Task 8): the PLE table, the experts' host mirrors and an explicit prefix cache are pinned
// beside the cards - they must fit MemAvailable less 16 GiB (loader::check_q4_host_fit) before anything is pinned;
// the whole model's --max-len auto stops at the first length whose mirrors would not.
inline Settled settle(const model::Qwen4ExpDesc& full, const DecodeArgs& a,
                      const std::array<size_t, runtime::kPpDevices>& dev, const std::string& snapshot) {
  namespace rq = runtime::qwen4exp;
  const bool int8 = a.int8_head(), pf = a.prefills();   // spec 21-QFN-v1d: the prefill scratch planned
  const bool hd = a.load_mtp;                            // spec 21-QFN-v1e: the MTP head planned
  const uint32_t devs = devices_of(a), cap = full.trained_max_len;
  const size_t ple = loader::Q4PleHost(loader::q4_ple_dir(loader::resolve_snapshot(snapshot))).bytes();
  const size_t avail = loader::q4_mem_available();
  // `note`: what the experts' figure leaves out (the --layers path: a B70_Q4_RESIDENT tier is the loader's).
  const auto host_line = [&](size_t mirror, const char* note = "") {
    loader::check_q4_host_fit(ple, mirror, a.prefix_bytes, avail);
    std::print(stderr, "host: pinned {:.3f} GB (the PLE table {:.3f} + experts {:.3f}{} + the prefix cache {:.3f}) of "
               "MemAvailable {:.3f} GB, 16 GiB kept\n", (ple + mirror + a.prefix_bytes) / 1e9, ple / 1e9, mirror / 1e9,
               note, a.prefix_bytes / 1e9, avail / 1e9);
  };
  // The startup chart: one VRAM block per card from the plan, then the host memory pinned (zero parts left out).
  const auto bars = [&](const std::vector<rq::DevicePlan>& pl, size_t experts) {
    for (const rq::DevicePlan& dp : pl) {
      const runtime::VramCard c{runtime::vram_parts(dp, dp.mtp ? "MTP head" : "decode state", "drafter"), a.reserve,
                                dev[dp.device]};
      std::print(stderr, "{}", runtime::vram_block(c, dp.device, uint32_t(pl.size()), runtime::vram_utf8()));
    }
    std::print(stderr, "{}", runtime::vram_host_line({{"experts", experts}, {"PLE table", ple}}));
  };
  Settled s;
  if (a.pipe.tp()) {   // plan 21v2b: every layer's half on each card, the embedding on card 0 (card 1 reads it)
    if (a.layers_auto) throw std::runtime_error("--layers auto is not planned under --parallel tp: --layers N");
    if (a.layers > full.layers)
      throw std::runtime_error("--layers " + std::to_string(a.layers) + " exceeds the checkpoint's " +
                               std::to_string(full.layers));
    s.desc = a.layers ? rq::truncated(full, a.layers) : full;
    const model::Qwen4ExpDesc& d = s.desc;
    const bool whole = d.layers == full.layers;
    s.placement = model::Q4Placement::tp2(d, model::Q4TpEnds::Split);
    const model::Qwen4ExpDesc card = model::q4_tp_card(d, 0);
    const char* forced = std::getenv("B70_Q4_RESIDENT");
    std::string prof_path;
    const std::vector<uint32_t> prof = whole ? loader::load_q4_profile(snapshot, full, &prof_path) : std::vector<uint32_t>{};
    const auto budget = [&](uint32_t len, uint32_t rows = rq::kPfC) {
      return rq::tp_expert_budget(d, s.placement, len, int8, hd, pf, dev, a.reserve, rows);
    };
    const auto tier = [&](const rq::ExpertBudget& b, uint32_t r) {
      return rq::tp_fill_tier(d, b, r, prof.empty() ? nullptr : &prof, forced ? forced : "");
    };
    if (!a.max_len.is_auto) {
      s.max_len = a.max_len.value;
      if (s.max_len > cap)
        throw std::runtime_error("--max-len " + std::to_string(s.max_len) + " exceeds Qwen3.8-Flash-Next's trained "
                                 "context " + std::to_string(cap));
    } else if (whole) {   // the pp rule: the longest whose projected decode stays within 5 % of 32768's
      const rq::ExpertBudget b0 = budget(kWholeMinLen);
      const double t0 = tier_decode_ms(b0, tier(b0, 0), prof);
      s.max_len = kWholeMinLen;
      std::string tried;
      for (uint32_t len = kWholeMinLen; len <= cap; len *= 2) {
        const rq::ExpertBudget b = budget(len);
        if (b.cache[0] == 0 || b.cache[1] == 0) break;
        const rq::ExpertTier ta = tier(b, 0), tb = tier(b, 1);
        if (!loader::q4_host_fits(ple + ta.host_bytes(card) + tb.host_bytes(card) + a.prefix_bytes, avail)) {
          tried += std::format("{}{:d}: host memory", tried.empty() ? "" : ", ", len);
          break;
        }
        const double ms = tier_decode_ms(b, ta, prof);
        tried += std::format("{}{:d}: {:.2f} ms", tried.empty() ? "" : ", ", len, ms);
        if (ms > 1.05 * t0) break;
        s.max_len = len;
      }
      std::print(stderr, "max_len: auto -> {:d} (the longest whose projected decode stays within 5% of {:d}'s; derived "
                 "ms a token: {})\n", s.max_len, kWholeMinLen, tried.c_str());
    } else {
      s.max_len = runtime::kMinAutoMaxLen;
      std::print(stderr, "max_len: auto -> {:d} (--layers N under --parallel tp)\n", s.max_len);
    }
    if (pf && (whole || forced)) {   // the 4096-row chunk when decode keeps its residents (pf_chunk_choice)
      const rq::PfChunkChoice c = rq::pf_chunk_choice(d, budget(s.max_len), budget(s.max_len, rq::kPfCWide),
                                                      prof.empty() ? nullptr : &prof, true);
      s.pf_rows = c.rows;
      print_pf_rows(c);
    } else if (pf && rq::fits(rq::tp_plan(d, s.placement, s.max_len, int8, hd, false, pf, rq::kPfCWide), dev, a.reserve)) {
      s.pf_rows = rq::kPfCWide;   // every expert resident either way: the wide chunk when its plan fits
    }
    if (a.pp_chunk > rq::kPfC) s.pf_rows = rq::kPfCWide;   // an asked chunk past 2048 rows is planned
    const rq::ExpertBudget b = budget(s.max_len, s.pf_rows);
    if (whole || forced) {
      s.tier = tier(b, 0);
      s.tier1 = tier(b, 1);
    }
    const std::vector<rq::DevicePlan> pl = rq::tp_plan(d, s.placement, s.max_len, int8, hd, false, pf, s.pf_rows);
    if (!s.tier) rq::require_fits(pl, dev, a.reserve);   // every expert resident: the whole plan must fit
    const size_t mirror = s.tier ? s.tier->host_bytes(card) + s.tier1->host_bytes(card) : 0;
    std::print(stderr, "layers: {} {:d}, each card half of every layer (--parallel tp); max_len {:d}\n",
               whole ? "all" : "the first", d.layers, s.max_len);
    if (s.tier)
      std::print(stderr, "experts: {}{}; resident a card: f {:.4f} / {:.4f} (cache {:.3f} / {:.3f} GB)\n",
                 s.tier->source.c_str(), prof.empty() ? "" : (" (" + prof_path + ")").c_str(), rq::tp_card_f(b, 0),
                 rq::tp_card_f(b, 1), b.cache[0] / 1e9, b.cache[1] / 1e9);
    host_line(mirror);
    for (const rq::DevicePlan& dp : pl)
      std::print(stderr, "card {:d}: weights {:.3f} GB with every expert half resident, kv {:.3f}, state + scratch + crossings "
                 "{:.3f}, prefill {:.3f}\n", dp.device, dp.weights / 1e9, dp.kv / 1e9, dp.decode_state / 1e9,
                 dp.prefill_scratch / 1e9);
    return s;
  }
  if (a.layers == 0 && !a.layers_auto) {   // the whole model
    if (!a.pp_given && devs == 2)
      std::print(stderr, "parallel: auto -> pp (the whole model does not fit one card: its layers split over two)\n");
    const char* forced = std::getenv("B70_Q4_RESIDENT");
    s.desc = full;
    const model::Qwen4ExpDesc& d = s.desc;
    std::string prof_path;
    const std::vector<uint32_t> prof = loader::load_q4_profile(snapshot, full, &prof_path);
    const auto place = [&](uint32_t len) {
      return devs == 1 ? model::Q4Placement::one(d)
                       : model::Q4Placement::two(d, a.pipe.split_auto ? rq::pp_split(d, len, int8, hd, pf).split : a.pipe.split);
    };
    const auto tier_at = [&](uint32_t len, rq::ExpertBudget* out = nullptr, uint32_t rows = rq::kPfC) {
      const rq::ExpertBudget b = rq::expert_budget(d, place(len), len, int8, hd, pf, dev, a.reserve, rows);
      if (out) *out = b;
      return rq::fill_tier(d, b, prof.empty() ? nullptr : &prof, forced ? forced : "");
    };
    if (a.max_len.is_auto) {
      rq::ExpertBudget b;
      const double t0 = tier_decode_ms(b, tier_at(kWholeMinLen, &b), prof);
      s.max_len = kWholeMinLen;
      std::string tried;
      for (uint32_t len = kWholeMinLen; len <= cap; len *= 2) {
        const rq::ExpertTier t = tier_at(len, &b);
        std::vector<rq::DevicePlan> pl = rq::plan(d, place(len), len, int8, hd, false, pf);
        rq::apply_tier(pl, d, t);
        if (!rq::fits(pl, dev, a.reserve)) break;
        if (!loader::q4_host_fits(ple + t.host_bytes(d) + a.prefix_bytes, avail)) {
          tried += std::format("{}{:d}: host memory", tried.empty() ? "" : ", ", len);
          break;
        }
        const double ms = tier_decode_ms(b, t, prof);
        tried += std::format("{}{:d}: {:.2f} ms", tried.empty() ? "" : ", ", len, ms);
        if (ms > 1.05 * t0) break;
        s.max_len = len;
      }
      std::print(stderr, "max_len: auto -> {:d} (the longest whose projected decode stays within 5% of {:d}'s; derived "
                 "ms a token: {})\n", s.max_len, kWholeMinLen, tried.c_str());
    } else {
      s.max_len = a.max_len.value;
      if (s.max_len > cap)
        throw std::runtime_error("--max-len " + std::to_string(s.max_len) + " exceeds Qwen3.8-Flash-Next's trained "
                                 "context " + std::to_string(cap));
    }
    s.placement = place(s.max_len);
    std::print(stderr, "layers: all {:d} (the routed experts that do not fit the card{} in pinned host memory); max_len {:d}\n",
               d.layers, devs == 2 ? "s" : "", s.max_len);
    if (pf) {   // the 4096-row chunk when decode keeps its residents (pf_chunk_choice), on the split chosen above
      const auto at = [&](uint32_t rows) {
        return rq::expert_budget(d, s.placement, s.max_len, int8, hd, pf, dev, a.reserve, rows);
      };
      const rq::PfChunkChoice c = rq::pf_chunk_choice(d, at(rq::kPfC), at(rq::kPfCWide), prof.empty() ? nullptr : &prof,
                                                      false);
      s.pf_rows = c.rows;
      print_pf_rows(c);
    }
    if (a.pp_chunk > rq::kPfC) s.pf_rows = rq::kPfCWide;   // an asked chunk past 2048 rows is planned
    s.tier = tier_at(s.max_len, nullptr, s.pf_rows);
    std::print(stderr, "experts: {}{}\n", s.tier->source.c_str(),
               prof.empty() ? "" : (" (" + prof_path + ")").c_str());
    std::vector<rq::DevicePlan> pl = rq::plan(d, s.placement, s.max_len, int8, hd, false, pf, s.pf_rows);
    rq::apply_tier(pl, d, *s.tier);
    rq::require_fits(pl, dev, a.reserve);
    host_line(s.tier->host_bytes(d));
    std::print(stderr, "{}\n", rq::describe(pl, s.placement, s.max_len, dev, a.reserve).c_str());
    bars(pl, s.tier->host_bytes(d));
    return s;
  }
  uint32_t layers = a.layers;
  if (a.layers_auto) {
    const uint32_t len = a.max_len.is_auto ? runtime::kMinAutoMaxLen : a.max_len.value;
    layers = rq::layers_that_fit(full, devs, len, int8, hd, std::min(dev[0], devs == 2 ? dev[1] : dev[0]), a.reserve, pf);
    if (layers == 0)
      throw std::runtime_error("--layers auto: not even " + std::to_string(full.ple_layer + 1) + " layers fit " +
                               std::to_string(devs) + " card(s) at max_len " + std::to_string(len));
    std::print(stderr, "layers: auto -> {:d} of {:d} (the largest N whose truncated model fits {:d} card(s) at max_len {:d} "
               "with a {:.3f} GB reserve)\n", layers, full.layers, devs, len, a.reserve / 1e9);
  }
  if (layers > full.layers)
    throw std::runtime_error("--layers " + std::to_string(layers) + " exceeds the checkpoint's " + std::to_string(full.layers));
  s.desc = rq::truncated(full, layers);
  const model::Qwen4ExpDesc& d = s.desc;
  model::Q4Placement p = devs == 1 ? model::Q4Placement::one(d) : model::Q4Placement::two(d, a.pipe.split ? a.pipe.split : 1);
  if (devs == 2 && d.layers < 2) throw std::runtime_error("--parallel pp needs at least two layers to split (--layers " +
                                                          std::to_string(d.layers) + ")");
  if (devs == 2 && !a.pipe.split_auto && (a.pipe.split < 1 || a.pipe.split >= d.layers))
    throw std::runtime_error("--pipeline-split " + std::to_string(a.pipe.split) + " leaves a device no layer: it is the "
                             "first layer of device 1, in [1, " + std::to_string(d.layers - 1) + "]");
  uint32_t len = a.max_len.value;
  if (a.max_len.is_auto) {
    if (devs == 2 && a.pipe.split_auto)
      p = model::Q4Placement::two(d, rq::pp_split(d, runtime::kMinAutoMaxLen, int8, hd, pf).split);
    len = rq::max_len_that_fits(d, p, int8, hd, dev, a.reserve, cap, pf);
    if (len == 0)
      throw std::runtime_error("--max-len auto: not even " + std::to_string(runtime::kMinAutoMaxLen) + " positions fit - " +
                               rq::describe(rq::plan(d, p, runtime::kMinAutoMaxLen, int8, hd, false, pf), p,
                                            runtime::kMinAutoMaxLen, dev, a.reserve));
    std::print(stderr, "max_len: auto -> {:d} (the largest multiple of {:d} that fits every card with a {:.3f} GB "
               "reserve; trained context {:d})\n", len, runtime::kMaxLenQuantum, a.reserve / 1e9, cap);
  } else {
    if (len > cap)
      throw std::runtime_error("--max-len " + std::to_string(len) + " exceeds Qwen3.8-Flash-Next's trained context " +
                               std::to_string(cap));
    if (devs == 2 && a.pipe.split_auto) p = model::Q4Placement::two(d, rq::pp_split(d, len, int8, hd, pf).split);
    std::print(stderr, "max_len: {:d} (--max-len)\n", len);
  }
  // every expert resident (B70_Q4_RESIDENT aside): the 4096-row chunk when its plan fits
  if ((pf && rq::fits(rq::plan(d, p, len, int8, hd, false, pf, rq::kPfCWide), dev, a.reserve)) || a.pp_chunk > rq::kPfC)
    s.pf_rows = rq::kPfCWide;
  const std::vector<rq::DevicePlan> pl = rq::plan(d, p, len, int8, hd, false, pf, s.pf_rows);
  rq::require_fits(pl, dev, a.reserve);
  const char* forced_l = std::getenv("B70_Q4_RESIDENT");
  host_line(0, forced_l && *forced_l ? " (B70_Q4_RESIDENT's mirrors are checked at load)" : "");
  if (devs == 2)
    std::print(stderr, "split: {}{:d} (layers [0, {:d}) on device 0, [{:d}, {:d}) and the head on device 1)\n",
               a.pipe.split_auto ? "auto -> " : "", p.split, p.split, p.split, d.layers);
  std::print(stderr, "{}\n", rq::describe(pl, p, len, dev, a.reserve).c_str());
  bars(pl, 0);
  s.placement = p;
  s.max_len = len;
  return s;
}

// The loader prints its report to stdout (its convention); `Redirect` is the caller's StdoutToStderr so the report
// joins the diagnostics on stderr.
template <class Redirect>
int run_decode_tp(const DecodeArgs& a);

template <class Redirect>
int run_decode(const DecodeArgs& a) {
  check_args(a);   // before any tensor is read
  if (a.pipe.tp()) return run_decode_tp<Redirect>(a);
  const uint32_t devs = devices_of(a);
  const model::Qwen4ExpDesc full = loader::qwen4exp_checkpoint_desc(a.path, 0);
  std::vector<uint32_t> ids;
  if (!a.bench) {
    ids = read_ids(a.ids_path, full.vocab);
  } else {
    ids.resize(a.depth);
    for (uint32_t i = 0; i < a.depth; ++i) ids[i] = kQwen4ExpBenchPrompt[i % kQwen4ExpBenchPromptLen];
  }
  const uint32_t after = a.bench ? a.tg : a.n;
  const size_t need = ids.size() + after;
  if (!a.max_len.is_auto && need > a.max_len.value)
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) + " generated ids exceeds --max-len " +
                             std::to_string(a.max_len.value));
  if (devs == 2) require_two_devices(l0::Context::gpu_count());
  std::unique_ptr<l0::Context> c0 = std::make_unique<l0::Context>(0u);
  std::unique_ptr<l0::Context> c1 = devs == 2 ? std::make_unique<l0::Context>(*c0, 1u) : nullptr;
  std::vector<l0::Context*> ctx = {c0.get()};
  if (c1) ctx.push_back(c1.get());
  std::array<size_t, runtime::kPpDevices> dev{c0->memory_bytes(), c1 ? c1->memory_bytes() : c0->memory_bytes()};
  if (devs == 1)
    std::print(stderr, "device: {} ({:d} EUs)\n", c0->name().c_str(), c0->eu_count());
  else
    std::print(stderr, "devices: {} + {} (--parallel pp)\n", c0->name().c_str(), c1->name().c_str());
  if (devs == 2 && !c0->can_access_peer(*c1))
    throw std::runtime_error("--parallel pp: device 0 cannot access device 1's memory (zeDeviceCanAccessPeer is false), and "
                             "the hand-off writes it (docs/10-the-box.md)");
  const Settled s = settle(full, a, dev, a.path);
  if (need > s.max_len)
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) + " generated ids exceeds max_len " +
                             std::to_string(s.max_len));
  loader::Q4LoadedModel model = [&] {
    Redirect redirect;
    return loader::load_qwen4exp(ctx, a.path, s.max_len, s.placement, a.lm_head,
                                 s.desc.layers == full.layers ? 0 : s.desc.layers, false, s.tier ? &*s.tier : nullptr,
                                 nullptr, s.pf_rows);
  }();
  if (model.desc.layers < model.checkpoint_layers)
    std::print(stderr, "layers: {:d} of the checkpoint's {:d} (--layers: development mode; the final mixer reads layer "
               "{:d}'s H)\n", model.desc.layers, model.checkpoint_layers, model.desc.layers - 1);
  runtime::PipelineOptions opt;
  runtime::qwen4exp::Qwen4ExpEngine eng(ctx, std::move(model), s.max_len, /*debug_tap=*/false, opt);
  std::print(stderr, "engine: Qwen3.8-Flash-Next, {:d} device(s){}, {:d} launches a token ({} attention, B70_Q4_ATTN), "
             "max_len {:d}; the PLE table's {:d} host pages read back by the device; {:d} expert block words read "
             "back through the tables ({:.1f} ms)\n", eng.devices(),
             devs == 2 ? (std::string(", split ") + std::to_string(eng.split())).c_str() : "",
             eng.launches(), runtime::qwen4exp::q4_attn_name(eng.attention()), eng.max_len(), eng.ple_pages_checked(),
             eng.expert_words_checked(), eng.model().expert_check_ms);
  // Spec 21-QFN-v1d: the prefill setup (every device's scratch, Context, the prefill hand-off, the binaries' check) happens
  // here, outside the timed window, which is then the whole prefill() call: first launch to the first generated id in
  // cur_token (argmax_stage2 is its last launch).
  const uint32_t chunk = a.pp_chunk ? a.pp_chunk : s.pf_rows;
  size_t pp_launches0 = 0;
  if (a.prefills()) {
    eng.prepare_prefill();
    pp_launches0 = eng.prefill_launches();
    std::print(stderr, "prefill: l0 backend, chunk {:d}, attention {} (B70_Q4_ATTN), {:d} launches for a chunk of {:d} at "
               "pos 0 ({}; they depend on its dense / sparse rows) + {:d} for the head{}\n", chunk,
               runtime::qwen4exp::q4_attn_name(eng.attention()),
               runtime::qwen4exp::prefill_chunk_launches(eng.model().desc, eng.model().placement, 0, chunk), chunk,
               devs == 2 ? "both cards" : "one card", runtime::qwen4exp::kPrefillHeadLaunches,
               devs == 2 ? "; each chunk crosses by copy, card 0 running the next chunk while card 1 finishes this one" : "");
  }
  std::print(stderr, "{}\n", eng.memory_line().c_str());
  const auto t0 = std::chrono::steady_clock::now();
  if (a.prefills())
    eng.prefill(ids, chunk);
  else
    eng.ingest(ids);
  const double ingest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (a.prefills())
    std::print(stderr, "{}: {:d} ids in {:.1f} ms ({:.2f} t/s) - device-side, loader excluded, first prefill launch to the "
               "first generated id; chunk {:d}, {:d} L0 launches, pos {:d}\n", a.pp ? "pp" : "prefill", ids.size(), ingest_ms,
               ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0, chunk,
               eng.prefill_launches() - pp_launches0, eng.pos());
  else
    std::print(stderr, "ingest: {:d} ids in {:.1f} ms ({:.2f} ms/token, one decode replay per id), pos {:d}\n", ids.size(),
               ingest_ms, ingest_ms / double(ids.size()), eng.pos());
  const uint32_t n = a.bench ? a.tg : a.n;
  eng.generate(n, [&](uint32_t id) {
    if (a.bench) return;
    std::print("{:d}\n", id);
    std::fflush(stdout);
  });
  std::print(stderr, "generate: {:d} ids, {:.2f} t/s, {:.2f} ms/token ({:.1f}% of it inside the fence)\n", n,
             eng.last_tok_per_s(), n ? eng.last_gen_ms() / n : 0.0,
             eng.last_gen_ms() > 0.0 ? 100.0 * eng.last_fence_ms() / eng.last_gen_ms() : 0.0);
  if (!a.bench) return 0;
  // Probes only: B70_Q4_PROFILE=R prints each launch's mean device time over R profiled replays after the bench.
  if (const char* pr = std::getenv("B70_Q4_PROFILE"); pr != nullptr && *pr != '\0') {
    const std::vector<double> us = eng.profile_decode(uint32_t(std::stoul(pr)));
    double sum = 0.0;
    for (size_t i = 0; i < us.size(); ++i) {
      std::print(stderr, "prof {:d} {} {:.3f}\n", i, eng.step(0).labels[i], us[i]);
      sum += us[i];
    }
    std::print(stderr, "prof sum {:.3f} us over {:d} launches\n", sum, us.size());
  }
  const double ms_per_token = eng.last_gen_ms() / double(a.tg);
  const double gb = double(eng.model().read_per_token) / 1e9;
  std::print(stderr, "  MBU: {:.2f} t/s x {:.3f} GB = {:.0f} GB/s (W = this load's derived read/token: both HCs, the "
             "mixers, the top-10 + shared experts, the routers, the head; the 16 PLE rows from host memory apart); {:d} "
             "launches/token\n", eng.last_tok_per_s(), gb, eng.last_tok_per_s() * gb, eng.launches());
  const char* sha = std::getenv("B70_GIT_SHA");
  if (sha == nullptr || *sha == '\0') sha = "unknown";
  std::print("| b70-decode {} qwen4exp L{:d}{}{} {}{} | {:d} | {:d} | {:.2f} | {:.2f} |\n", sha, eng.model().desc.layers,
             a.int8_head() ? " int8-head" : "",
             eng.model().desc.forms.dense == model::Q4Form::Bf16 ? " bf16-dense" : "",
             runtime::qwen4exp::q4_attn_name(eng.attention()),
             devs == 2 ? (std::string(" pipeline2-copy-s") + std::to_string(eng.split())).c_str() : "",
             a.depth, a.tg, eng.last_tok_per_s(), ms_per_token);
  // Spec 21-QFN-v1d: the pp row, a second line as the Kolibri / Qwen paths print it (depth, chunk, ms, t/s).
  if (a.pp)
    std::print("| b70-decode {} qwen4exp L{:d}{}{} {}{} l0 pp | {:d} | {:d} | {:.1f} | {:.2f} |\n", sha, eng.model().desc.layers,
               a.int8_head() ? " int8-head" : "", eng.model().desc.forms.dense == model::Q4Form::Bf16 ? " bf16-dense" : "",
               runtime::qwen4exp::q4_attn_name(eng.attention()),
               devs == 2 ? (std::string(" pipeline2-copy-s") + std::to_string(eng.split())).c_str() : "", a.depth, chunk,
               ingest_ms, ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0);
  return 0;
}


// Plan 21v2b: b70-decode --parallel tp - run_decode's flow over runtime::qwen4exp::Qwen4ExpTpEngine (both cards in one
// context with peer access, the tp placement and each card's tier from settle, the load, the prompt fed or prefilled,
// greedy decode, the bench rows).
template <class Redirect>
int run_decode_tp(const DecodeArgs& a) {
  const model::Qwen4ExpDesc full = loader::qwen4exp_checkpoint_desc(a.path, 0);
  std::vector<uint32_t> ids;
  if (!a.bench) {
    ids = read_ids(a.ids_path, full.vocab);
  } else {
    ids.resize(a.depth);
    for (uint32_t i = 0; i < a.depth; ++i) ids[i] = kQwen4ExpBenchPrompt[i % kQwen4ExpBenchPromptLen];
  }
  const uint32_t after = a.bench ? a.tg : a.n;
  const size_t need = ids.size() + after;
  require_two_devices_for_tp(l0::Context::gpu_count());
  std::unique_ptr<l0::Context> c0 = std::make_unique<l0::Context>(0u);
  std::unique_ptr<l0::Context> c1 = std::make_unique<l0::Context>(*c0, 1u);
  std::vector<l0::Context*> ctx = {c0.get(), c1.get()};
  const std::array<size_t, runtime::kPpDevices> dev{c0->memory_bytes(), c1->memory_bytes()};
  std::print(stderr, "devices: {} + {} (--parallel tp)\n", c0->name().c_str(), c1->name().c_str());
  const Settled s = settle(full, a, dev, a.path);
  if (need > s.max_len)
    throw std::runtime_error("the prompt / --depth plus " + std::to_string(after) + " generated ids exceeds max_len " +
                             std::to_string(s.max_len));
  loader::Q4LoadedModel model = [&] {
    Redirect redirect;
    return loader::load_qwen4exp(ctx, a.path, s.max_len, s.placement, a.lm_head,
                                 s.desc.layers == full.layers ? 0 : s.desc.layers, false, s.tier ? &*s.tier : nullptr,
                                 s.tier1 ? &*s.tier1 : nullptr, s.pf_rows);
  }();
  runtime::qwen4exp::Qwen4ExpTpEngine eng(*c0, *c1, std::move(model), s.max_len);
  std::print(stderr, "engine: Qwen3.8-Flash-Next, --parallel tp, {:d} launches a token a card, max_len {:d}; {:d} expert "
             "block words read back through the tables\n", eng.launches(0), eng.max_len(), eng.expert_words_checked());
  const uint32_t chunk = a.pp_chunk ? a.pp_chunk : s.pf_rows;
  size_t pp_launches0 = 0;
  if (a.prefills()) {
    eng.prepare_prefill();
    pp_launches0 = eng.prefill_launches();
    std::print(stderr, "prefill: l0 backend, chunk {:d}, each chunk on both cards with two crossings a layer + {:d} "
               "launches a card for the head\n", chunk, runtime::qwen4exp::kPrefillHeadLaunches);
  }
  const auto t0 = std::chrono::steady_clock::now();
  if (a.prefills())
    eng.prefill(ids, chunk);
  else
    eng.ingest(ids);
  const double ingest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (a.prefills())
    std::print(stderr, "{}: {:d} ids in {:.1f} ms ({:.2f} t/s) - first prefill launch to the first generated id; chunk "
               "{:d}, {:d} L0 launches, pos {:d}\n", a.pp ? "pp" : "prefill", ids.size(), ingest_ms,
               ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0, chunk,
               eng.prefill_launches() - pp_launches0, eng.pos());
  else
    std::print(stderr, "ingest: {:d} ids in {:.1f} ms ({:.2f} ms/token, one decode replay per id), pos {:d}\n",
               ids.size(), ingest_ms, ingest_ms / double(ids.size()), eng.pos());
  const uint32_t n = a.bench ? a.tg : a.n;
  eng.generate(n, [&](uint32_t id) {
    if (a.bench) return;
    std::print("{:d}\n", id);
    std::fflush(stdout);
  });
  std::print(stderr, "generate: {:d} ids, {:.2f} t/s, {:.2f} ms/token\n", n, eng.last_tok_per_s(),
             n ? eng.last_gen_ms() / n : 0.0);
  if (!a.bench) return 0;
  const char* sha = std::getenv("B70_GIT_SHA");
  if (sha == nullptr || *sha == '\0') sha = "unknown";
  std::print("| b70-decode {} qwen4exp L{:d}{} flash tp2 | {:d} | {:d} | {:.2f} | {:.2f} |\n", sha,
             eng.model().desc.layers, a.int8_head() ? " int8-head" : "", a.depth, a.tg, eng.last_tok_per_s(),
             eng.last_gen_ms() / double(a.tg));
  if (a.pp)
    std::print("| b70-decode {} qwen4exp L{:d}{} flash tp2 l0 pp | {:d} | {:d} | {:.1f} | {:.2f} |\n", sha,
               eng.model().desc.layers, a.int8_head() ? " int8-head" : "", a.depth, chunk, ingest_ms,
               ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0);
  return 0;
}

}  // namespace cli::qwen4exp
