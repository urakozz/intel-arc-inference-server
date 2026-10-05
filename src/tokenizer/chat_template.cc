#include "tokenizer/chat_template.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <minja/chat-template.hpp>

namespace chat {
namespace {

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("chat::Template: cannot read " + path);
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

std::string token_string(const nlohmann::json& config, const char* key) {
  if (!config.contains(key) || config[key].is_null()) return "";
  const auto& value = config[key];
  return value.is_string() ? value.get<std::string>() : value.value("content", "");
}

uint32_t rotate_right(uint32_t value, uint32_t shift) {
  return (value >> shift) | (value << (32U - shift));
}

std::string sha256(const std::string& input) {
  static constexpr std::array<uint32_t, 64> k = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
      0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
      0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
      0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
      0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
      0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
  std::array<uint32_t, 8> state = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::vector<uint8_t> bytes(input.begin(), input.end());
  const uint64_t bits = static_cast<uint64_t>(bytes.size()) * 8U;
  bytes.push_back(0x80U);
  while ((bytes.size() % 64U) != 56U) bytes.push_back(0);
  for (int shift = 56; shift >= 0; shift -= 8) {
    bytes.push_back(static_cast<uint8_t>(bits >> static_cast<uint32_t>(shift)));
  }

  for (size_t offset = 0; offset < bytes.size(); offset += 64U) {
    std::array<uint32_t, 64> words{};
    for (size_t i = 0; i < 16; ++i) {
      const size_t p = offset + i * 4U;
      words[i] = (static_cast<uint32_t>(bytes[p]) << 24U) |
                 (static_cast<uint32_t>(bytes[p + 1]) << 16U) |
                 (static_cast<uint32_t>(bytes[p + 2]) << 8U) |
                 static_cast<uint32_t>(bytes[p + 3]);
    }
    for (size_t i = 16; i < words.size(); ++i) {
      const uint32_t s0 = rotate_right(words[i - 15], 7U) ^ rotate_right(words[i - 15], 18U) ^
                          (words[i - 15] >> 3U);
      const uint32_t s1 = rotate_right(words[i - 2], 17U) ^ rotate_right(words[i - 2], 19U) ^
                          (words[i - 2] >> 10U);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (size_t i = 0; i < words.size(); ++i) {
      const uint32_t s1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
      const uint32_t choose = (e & f) ^ ((~e) & g);
      const uint32_t temp1 = h + s1 + choose + k[i] + words[i];
      const uint32_t s0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(64);
  for (uint32_t word : state) {
    for (int shift = 28; shift >= 0; shift -= 4) {
      result.push_back(hex[(word >> static_cast<uint32_t>(shift)) & 0x0fU]);
    }
  }
  return result;
}

}  // namespace

struct Template::Impl {
  std::unique_ptr<minja::chat_template> tmpl;
  std::string bos;
  std::string eos;
};

Template::Template(const std::string& dir) : impl_(std::make_unique<Impl>()) {
  const nlohmann::json config = nlohmann::json::parse(slurp(dir + "/tokenizer_config.json"));
  impl_->bos = token_string(config, "bos_token");
  impl_->eos = token_string(config, "eos_token");
  const std::string template_source = slurp(dir + "/chat_template.jinja");
  // This checkpoint's template uses Jinja's unsupported `is undefined` test.
  // The source SHA keeps the equivalent minja fallback model-specific.
  static constexpr const char* kFallbackTemplateSha256 =
      "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041";
  // Spec 15e: Ornith 1.5's chat_template.jinja (ornith-ai/Ornith-1.5-35B-A3B, Qwen3.5's
  // template: no reasoning-effort block, every assistant turn keeps its <think> block) has
  // one `is undefined` test, in render_content's null branch. minja knows `defined` (as
  // "not null") but not `undefined`, so that test is spelled `is not defined` - the same
  // predicate under minja - for this source only, keyed like the fallback above.
  static constexpr const char* kOrnithTemplateSha256 =
      "182e77dd83bd8e9ca818b240b82e28f243762cd5dda32e6eef327df7b1cd107e";
  const std::string sha = sha256(template_source);
  std::string source = sha == kFallbackTemplateSha256 ? slurp(B70_CHAT_TEMPLATE_FALLBACK_PATH)
                                                      : template_source;
  if (sha == kOrnithTemplateSha256) {
    static constexpr const char* kFrom = " is undefined";
    static constexpr const char* kTo = " is not defined";
    for (size_t at = source.find(kFrom); at != std::string::npos; at = source.find(kFrom, at))
      source.replace(at, std::string(kFrom).size(), kTo);
  }
  impl_->tmpl = std::make_unique<minja::chat_template>(source, impl_->bos, impl_->eos);
}

Template::~Template() = default;

std::string Template::render(const nlohmann::json& messages, const nlohmann::json& tools,
                             bool enable_thinking) const {
  minja::chat_template_inputs inputs;
  inputs.messages = messages;
  inputs.tools = tools.is_null() ? nlohmann::ordered_json() : nlohmann::ordered_json(tools);
  inputs.add_generation_prompt = true;
  inputs.extra_context = nlohmann::ordered_json{{"enable_thinking", enable_thinking}};
  minja::chat_template_options options;
  return impl_->tmpl->apply(inputs, options);
}

const std::string& Template::eos_token() const { return impl_->eos; }

}  // namespace chat
