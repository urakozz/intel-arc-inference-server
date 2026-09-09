#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "server/deps.h"

struct MockTok : server::TokIface {
  std::vector<std::string> vocab;
  std::map<std::string, uint32_t> index;

  uint32_t id_of(const std::string& word) {
    if (word == "<|im_end|>") return 248046;
    const auto it = index.find(word);
    if (it != index.end()) return it->second;
    const uint32_t id = 1000 + static_cast<uint32_t>(vocab.size());
    vocab.push_back(word);
    index[word] = id;
    return id;
  }

  std::vector<uint32_t> encode(std::string_view text) override {
    std::vector<uint32_t> out;
    std::string word;
    for (char c : text) {
      if (c == ' ' || c == '\n') {
        if (!word.empty()) out.push_back(id_of(word));
        word.clear();
        if (c == '\n') out.push_back(id_of("\n"));
      } else {
        word += c;
      }
    }
    if (!word.empty()) out.push_back(id_of(word));
    return out;
  }

  std::string piece(uint32_t id) {
    if (id == 248046) return "<|im_end|>";
    const std::string& word = vocab.at(id - 1000);
    return word == "\n" ? "\n" : word + " ";
  }

  std::string decode(const std::vector<uint32_t>& ids) override {
    std::string text;
    for (uint32_t id : ids) text += piece(id);
    return text;
  }

  struct Streamer : server::StreamerIface {
    explicit Streamer(MockTok& tokenizer) : tokenizer(tokenizer) {}
    std::string push(uint32_t id) override { return tokenizer.piece(id); }
    std::string flush() override { return ""; }
    MockTok& tokenizer;
  };

  std::unique_ptr<server::StreamerIface> streamer() override {
    return std::make_unique<Streamer>(*this);
  }

  uint32_t vocab_used() override { return 248077; }
};

struct MockTemplate : server::TemplateIface {
  std::string render(const nlohmann::json& messages, const nlohmann::json&, bool think) override {
    std::string text;
    for (const auto& message : messages) {
      text += message.at("role").get<std::string>() + ": " +
              message.at("content").get<std::string>() + "\n";
    }
    return text + (think ? "assistant(think): " : "assistant: ");
  }
};

struct MockEngine : server::EngineIface {
  std::vector<uint32_t> script;
  size_t at = 0;
  int step_ms = 0;
  std::vector<uint32_t> last_prompt;
  std::vector<std::chrono::steady_clock::time_point> step_times;
  std::atomic<int> active{0};
  std::atomic<int> max_active{0};

  void reset() override { at = 0; }
  void prefill(const std::vector<uint32_t>& ids) override { last_prompt = ids; }

  uint32_t step(const server::Sampling&) override {
    const int now_active = ++active;
    max_active = std::max(max_active.load(), now_active);
    if (step_ms != 0) std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
    step_times.push_back(std::chrono::steady_clock::now());
    const uint32_t id = at < script.size() ? script[at++] : 248046;
    --active;
    return id;
  }

  uint32_t max_len() override { return 1024; }
  uint32_t pos() override { return static_cast<uint32_t>(last_prompt.size() + at); }
};
