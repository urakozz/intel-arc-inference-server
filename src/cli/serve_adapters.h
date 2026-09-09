#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "runtime/engine.h"
#include "server/deps.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/streamer.h"
#include "tokenizer/tokenizer.h"

struct TokAdapter : server::TokIface {
  explicit TokAdapter(const std::string& json) : t(json) {}

  std::vector<uint32_t> encode(std::string_view text) override { return t.encode(text, false); }
  std::string decode(const std::vector<uint32_t>& ids) override { return t.decode(ids, false); }

  struct S : server::StreamerIface {
    explicit S(const tok::Tokenizer& tokenizer) : st(tokenizer) {}

    std::string push(uint32_t id) override { return st.push(id); }
    std::string flush() override { return st.flush(); }

    tok::Streamer st;
  };

  std::unique_ptr<server::StreamerIface> streamer() override { return std::make_unique<S>(t); }
  uint32_t vocab_used() override { return t.vocab_size(); }

  tok::Tokenizer t;
};

struct TemplateAdapter : server::TemplateIface {
  explicit TemplateAdapter(const std::string& dir) : tmpl(dir) {}

  std::string render(const nlohmann::json& messages, const nlohmann::json& tools,
                     bool enable_thinking) override {
    return tmpl.render(messages, tools, enable_thinking);
  }

  chat::Template tmpl;
};

struct EngineAdapter : server::EngineIface {
  explicit EngineAdapter(runtime::Engine& engine, uint32_t vocab) : eng(engine), vocab_used(vocab) {}

  void reset() override { eng.reset(); }

  void prefill(const std::vector<uint32_t>& ids) override {
#if B70_HAVE_PREFILL
    eng.prefill(ids);
#else
    eng.ingest(ids);
#endif
  }

  uint32_t step(const server::Sampling& sampling) override {
    const uint32_t id = eng.generate(1)[0];
    if (!sampling.greedy) sample_into_control(sampling);
    return id;
  }

  uint32_t max_len() override { return eng.max_len(); }
  uint32_t pos() override { return eng.pos(); }

  void sample_into_control(const server::Sampling&) {}

  runtime::Engine& eng;
  uint32_t vocab_used;
};
