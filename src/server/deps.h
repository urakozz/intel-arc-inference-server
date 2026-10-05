#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace server {

struct StreamerIface {
  virtual ~StreamerIface() = default;
  virtual std::string push(uint32_t id) = 0;
  virtual std::string flush() = 0;
};

struct TokIface {
  virtual ~TokIface() = default;
  virtual std::vector<uint32_t> encode(std::string_view text) = 0;
  virtual std::string decode(const std::vector<uint32_t>& ids) = 0;
  virtual std::unique_ptr<StreamerIface> streamer() = 0;
  virtual uint32_t vocab_used() = 0;
};

struct TemplateIface {
  virtual ~TemplateIface() = default;
  virtual std::string render(const nlohmann::json& messages, const nlohmann::json& tools,
                             bool enable_thinking) = 0;
  // Spec 18d: with the request's chat_template_kwargs as template variables beside
  // enable_thinking (K2-Horizon's tool_call_format, reasoning_effort; server/chat_format.h says
  // which models get them). The default ignores them.
  virtual std::string render_with_kwargs(const nlohmann::json& messages, const nlohmann::json& tools,
                                         bool enable_thinking, const nlohmann::json& kwargs) {
    (void)kwargs;
    return render(messages, tools, enable_thinking);
  }
};

struct Sampling {
  bool greedy = true;
  float temperature = 1.0f;
  uint32_t top_k = 20;
  float top_p = 0.95f;
  uint64_t seed = 0;
  bool has_seed = false;
};

struct EngineIface {
  virtual ~EngineIface() = default;
  virtual void reset() = 0;
  virtual void prefill(const std::vector<uint32_t>& ids) = 0;
  virtual uint32_t step(const Sampling& s) = 0;
  virtual uint32_t max_len() = 0;
  virtual uint32_t pos() = 0;

  // Spec 7 (plan 7c): the prefix cache's engine calls, runtime::Engine's (plan 7b) behind
  // EngineAdapter. Defaults throw, so an engine without them runs with the cache off only.
  using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
  virtual uint32_t block() { return 2048; }                  // runtime::Engine::kBlock
  virtual size_t state_bytes() { return unsupported<size_t>(); }
  virtual size_t kv_bytes(uint32_t) { return unsupported<size_t>(); }
  // Spec 12b: the KV cache's form (0 = bf16, 1 = int8), which keys the prefix cache.
  virtual uint64_t kv_form() { return 0; }
  virtual void save_state(void*) { unsupported<int>(); }
  virtual void load_state(const void*, uint32_t) { unsupported<int>(); }
  virtual void save_kv(uint32_t, uint32_t, void*) { unsupported<int>(); }
  virtual void load_kv(uint32_t, uint32_t, const void*) { unsupported<int>(); }
  virtual void set_block_hook(BlockHook) { unsupported<int>(); }
  // runtime::Engine::ingest: ids through the decode replay, one each; the prefix cache
  // feeds the last prompt id this way so that the prompt-end snapshot sits at len - 1.
  virtual void ingest(const std::vector<uint32_t>&) { unsupported<int>(); }

  // Spec 8 §3.6 (plan 8c): speculative decoding. mtp_k() == 0 (the default) means the
  // server calls step() only, exactly as before. Otherwise step_many(s, k) runs one
  // draft/verify/accept iteration with k drafts (k <= mtp_k(); spec 8 §10: `--mtp auto`
  // chooses k per iteration, k = 0 is one plain step) and returns 1 .. k + 1 ids in
  // order, the first being what step() would have returned; pos() then counts all of
  // them. When the server keeps fewer (a stop, EOS or max_tokens inside the run),
  // truncate_to(pos) rewinds the engine to the last kept id, so the session holds exactly
  // the prompt and the ids the server consumed - as a run of step() calls would.
  virtual uint32_t mtp_k() { return 0; }
  virtual std::vector<uint32_t> step_many(const Sampling& s, uint32_t k) {
    (void)k;
    return {step(s)};
  }
  virtual void truncate_to(uint32_t pos) { (void)pos; }

  // Spec 19e (`--spec lookup`, spec 19 §3 C): verify drafts proposed OUTSIDE the engine
  // (prompt lookup). verify_k() == 0 (the default) means the engine cannot, and the server
  // never calls step_drafts(). Otherwise step_drafts(s, propose) runs one iteration:
  //   1. it calls propose(x) exactly once, with x the pending id (what step() would return
  //      now); the proposer answers d_1 .. d_k (k <= verify_k(), possibly 0) continuing x;
  //   2. it verifies them at M = k + 1 rows (the engine may verify fewer, near max_len; k = 0
  //      is one plain step) and keeps the longest acceptable prefix d_1 .. d_j - greedy: the
  //      target's argmax equals the draft; sampled: server::accept_point_mass (the proposal is
  //      a point mass, lossless);
  //   3. it returns x, d_1 .. d_j (1 .. k + 1 ids); pos() counts all of them, and
  //      truncate_to() shortens the run exactly as after step_many().
  using Proposer = std::function<std::vector<uint32_t>(uint32_t pending)>;
  virtual uint32_t verify_k() { return 0; }
  virtual std::vector<uint32_t> step_drafts(const Sampling& s, const Proposer& propose) {
    (void)propose;
    return {step(s)};
  }

 private:
  template <class T>
  static T unsupported() {
    throw std::logic_error("EngineIface: this engine does not support the prefix cache");
  }
};

struct Deps {
  TokIface& tok;
  TemplateIface& tmpl;
  EngineIface& engine;
};

}  // namespace server
