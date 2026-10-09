// Spec 21e Task 1: Qwen3.8-Flash-Next's tokenizer and chat template through the engine's (the Rust `tokenizers`
// crate behind tok::Tokenizer, minja behind chat::Template) against Hugging Face's, from
// tools/tokenizer/dump_qwen4exp.py's facts:
//   qwen4exp_tokenizer_test <tests/tokenizer> <tokenizer.json>
// The tokenizer.json's directory is the checkpoint's (the original's small files: tokenizer_config.json,
// chat_template.jinja, generation_config.json - 21b's synthetic checkpoints copy them unchanged; nothing of the
// release is vendored, its licence text unread). The environment's B70_Q4EXP_TOKENIZER_JSON replaces the second
// argument. No tokenizer.json: SKIP (77).
//
// 21a found the original's tokenizer.json NOT Qwen3.8's (same vocabulary, merges and added tokens; the
// pre-tokenizer's split regex adds \p{M}, so combining marks stay with their letters): the server serves the
// original's file (Intel's checkpoint ships Qwen3.8's), and this holds the engine's encode / decode to it:
//   - the file's sha256 (the original's 0997f410..., not Qwen3.8's 06b95093...), the id count 248077, every added
//     token's id both ways (248044..248076), generation_config.json's EOS [248046, 248044];
//   - a 24-text set (prose, code, digits, emoji, CJK, and the scripts whose marks the \p{M} split keeps -
//     Devanagari, Thai, Arabic harakat, Hebrew points, Zalgo, combining marks NFC cannot compose): encode without
//     and with special tokens (identical: no BOS), both decodes, the streamer's pieces concatenating to the decode;
//     the facts record which texts Qwen3.8's file encodes differently (the gate tells the files apart);
//   - tests/tokenizer/corpus.txt's 10240 lines: FNV-1a digests of the ids and the decodes, in 10 chunks;
//   - the template: chat_template.jinja's sha256 is c3cf9e34... (Qwen3.8's / Agnes's - chat::Template's fallback
//     for that hash applies, src/tokenizer/chat_template.cc), two of agnes_template_cases.json's cases render byte
//     for byte as agnes_template_<name>.txt through the snapshot's own file, and three renders' ids equal
//     transformers' apply_chat_template(tokenize = True) (transformers 5.19.0 - 5.15.0 builds this checkpoint's
//     tokenizer from Qwen2's default regex, not the file's: measured, dump_qwen4exp.py's note).
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "check.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/streamer.h"
#include "tokenizer/tokenizer.h"

namespace {

using json = nlohmann::json;

bool exists(const std::string& path) { return std::ifstream(path).good(); }

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

std::vector<uint32_t> ids_of(const json& array) { return array.get<std::vector<uint32_t>>(); }

uint64_t fnv1a64(const uint8_t* data, size_t n, uint64_t h) {
  for (size_t i = 0; i < n; ++i) h = (h ^ data[i]) * 0x100000001b3ull;
  return h;
}

std::string hex64(uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
  return buf;
}

// FIPS 180-4 SHA-256 (the file hashes the facts name; chat_template.cc keeps its own private copy).
std::string sha256(const std::string& msg) {
  static const uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
      0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
      0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
      0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
      0xc67178f2};
  std::array<uint32_t, 8> h = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                               0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::string m = msg;
  const uint64_t bits = uint64_t(msg.size()) * 8;
  m.push_back(char(0x80));
  while (m.size() % 64 != 56) m.push_back('\0');
  for (int s = 56; s >= 0; s -= 8) m.push_back(char((bits >> s) & 0xff));
  const auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  for (size_t at = 0; at < m.size(); at += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = uint32_t(uint8_t(m[at + 4 * i])) << 24 | uint32_t(uint8_t(m[at + 4 * i + 1])) << 16 |
             uint32_t(uint8_t(m[at + 4 * i + 2])) << 8 | uint32_t(uint8_t(m[at + 4 * i + 3]));
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
  }
  char out[65];
  for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
  return out;
}

std::string streamed(const tok::Tokenizer& t, const std::vector<uint32_t>& ids) {
  tok::Streamer s(t);
  std::string out;
  for (uint32_t id : ids) out += s.push(id);
  return out + s.flush();
}

int check_cases(const tok::Tokenizer& t, const json& facts) {
  int bad = 0;
  for (const json& c : facts.at("cases")) {
    const std::string text = c.at("text");
    const std::vector<uint32_t> want = ids_of(c.at("ids"));
    const std::vector<uint32_t> got = t.encode(text, false);
    const bool enc = got == want;
    const bool enc_special = t.encode(text, true) == ids_of(c.at("ids_special"));
    const bool no_bos = ids_of(c.at("ids_special")) == want;   // Qwen's post-processor adds nothing
    const std::string decoded = t.decode(want, false);
    const bool dec = decoded == c.at("decoded").get<std::string>();
    const bool dec_skip = t.decode(want, true) == c.at("decoded_skip").get<std::string>();
    const bool stream = streamed(t, want) == decoded;
    if (!(enc && enc_special && no_bos && dec && dec_skip && stream)) {
      std::fprintf(stderr, "case %s: encode %d, encode+special %d, no BOS %d, decode %d, decode-skip %d, stream %d\n",
                   json(text).dump().substr(0, 80).c_str(), enc, enc_special, no_bos, dec, dec_skip, stream);
      ++bad;
    }
  }
  std::printf("text set: %zu cases, %d mismatches (Qwen3.8's file encodes %zu of them differently)\n",
              facts.at("cases").size(), bad, facts.at("cases_qwen38_differs").size());
  return bad;
}

std::vector<std::string> corpus_lines(const std::string& path) {
  const std::string all = slurp(path);
  std::vector<std::string> out;
  size_t b = 0;
  for (size_t i = 0; i < all.size(); ++i) {
    if (all[i] == '\n') {
      out.emplace_back(all, b, i - b);
      b = i + 1;
    }
  }
  if (b < all.size()) out.emplace_back(all, b);
  return out;
}

int check_corpus(const tok::Tokenizer& t, const json& facts, const std::string& dir) {
  const std::vector<std::string> lines = corpus_lines(dir + "/corpus.txt");
  int bad = 0;
  size_t total = 0;
  for (const json& chunk : facts.at("corpus_chunks")) {
    const size_t first = chunk.at("first"), n = chunk.at("lines");
    CHECK(first + n <= lines.size());
    uint64_t he = 0xcbf29ce484222325ull, hd = 0xcbf29ce484222325ull;
    size_t ids = 0;
    for (size_t i = first; i < first + n; ++i) {
      const std::vector<uint32_t> got = t.encode(lines[i], false);
      ids += got.size();
      std::vector<uint8_t> le;
      for (uint32_t id : got)
        for (int s = 0; s < 32; s += 8) le.push_back(static_cast<uint8_t>(id >> s));
      for (int k = 0; k < 4; ++k) le.push_back(0xFF);
      he = fnv1a64(le.data(), le.size(), he);
      std::string text = t.decode(got, false);
      text.push_back('\0');
      hd = fnv1a64(reinterpret_cast<const uint8_t*>(text.data()), text.size(), hd);
    }
    total += ids;
    const bool ok = ids == chunk.at("ids").get<size_t>() &&
                    hex64(he) == chunk.at("encode_fnv1a64").get<std::string>() &&
                    hex64(hd) == chunk.at("decode_fnv1a64").get<std::string>();
    if (!ok) {
      std::fprintf(stderr, "corpus lines %zu..%zu: ids %zu (want %zu), encode %s (want %s), decode %s (want %s)\n",
                   first, first + n - 1, ids, chunk.at("ids").get<size_t>(), hex64(he).c_str(),
                   chunk.at("encode_fnv1a64").get<std::string>().c_str(), hex64(hd).c_str(),
                   chunk.at("decode_fnv1a64").get<std::string>().c_str());
      ++bad;
    }
  }
  std::printf("corpus: %zu lines, %zu ids, %d chunk mismatches\n", lines.size(), total, bad);
  return bad;
}

const json& find_case(const json& cases, const std::string& name) {
  for (const json& c : cases)
    if (c.at("name") == name) return c;
  std::fprintf(stderr, "no case '%s' in agnes_template_cases.json\n", name.c_str());
  std::exit(1);
}

// The template: its hash, two cases' text through the snapshot's own file, three renders' ids.
int check_template(const tok::Tokenizer& t, const json& facts, const std::string& dir, const std::string& snapshot) {
  const std::string source = slurp(snapshot + "/chat_template.jinja");
  CHECK_EQ(sha256(source), facts.at("chat_template_sha256").get<std::string>());
  CHECK_EQ(sha256(source), std::string("c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"));
  std::printf("template: chat_template.jinja sha256 c3cf9e34... (Qwen3.8's / Agnes's: chat::Template's fallback)\n");
  const json cases = json::parse(slurp(dir + "/agnes_template_cases.json"));
  const chat::Template tmpl(snapshot);
  CHECK_EQ(tmpl.eos_token(), std::string("<|im_end|>"));
  int bad = 0;
  for (const char* name : {"thinking", "tool_call_history"}) {
    const json& c = find_case(cases, name);
    const std::string got = tmpl.render(c.at("messages"), c.at("tools"), c.at("think").get<bool>());
    const std::string want = slurp(dir + "/agnes_template_" + name + ".txt");
    if (got != want) {
      std::fprintf(stderr, "render %s: %zu bytes, not agnes_template_%s.txt's %zu\n", name, got.size(), name, want.size());
      ++bad;
    } else {
      std::printf("render %s: %zu bytes, identical to agnes_template_%s.txt\n", name, got.size(), name);
    }
  }
  for (const json& r : facts.at("renders")) {
    const std::string name = r.at("case");
    const json& c = find_case(cases, name);
    const std::string prompt = tmpl.render(c.at("messages"), c.at("tools"), c.at("think").get<bool>());
    const std::vector<uint32_t> got = t.encode(prompt, false);
    const std::vector<uint32_t> want = ids_of(r.at("ids"));
    const bool adds_nothing = t.encode(prompt, true) == got;
    if (got != want || !adds_nothing || !r.at("text_is_agnes_render").get<bool>()) {
      std::fprintf(stderr, "render %s: %zu ids (want %zu), equal %d, add_special adds nothing %d\n", name.c_str(),
                   got.size(), want.size(), got == want, adds_nothing);
      ++bad;
    } else {
      std::printf("render %s: %zu ids, equal to apply_chat_template's\n", name.c_str(), got.size());
    }
  }
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: qwen4exp_tokenizer_test <tests/tokenizer> <tokenizer.json>\n");
    return 2;
  }
  const std::string dir = argv[1];
  const char* env = std::getenv("B70_Q4EXP_TOKENIZER_JSON");
  const std::string path = env != nullptr && *env != '\0' ? env : argv[2];
  if (!exists(path)) {
    std::printf("qwen4exp_tokenizer_test: SKIP (no %s; set B70_Q4EXP_TOKENIZER_JSON)\n", path.c_str());
    return 77;
  }
  const size_t slash = path.find_last_of('/');
  const std::string snapshot = slash == std::string::npos ? std::string(".") : path.substr(0, slash);
  for (const char* f : {"/tokenizer_config.json", "/chat_template.jinja", "/generation_config.json"})
    if (!exists(snapshot + f)) {
      std::printf("qwen4exp_tokenizer_test: SKIP (no %s%s beside the tokenizer.json)\n", snapshot.c_str(), f);
      return 77;
    }
  const json facts = json::parse(slurp(dir + "/qwen4exp_tokenizer.json"));
  // The original's file, not Qwen3.8's (21a: the \p{M} split).
  const std::string file_sha = sha256(slurp(path));
  if (file_sha != facts.at("tokenizer_json_sha256").get<std::string>()) {
    std::fprintf(stderr, "%s has sha256 %s, not the original's %s%s\n", path.c_str(), file_sha.c_str(),
                 facts.at("tokenizer_json_sha256").get<std::string>().c_str(),
                 file_sha == facts.value("qwen38_tokenizer_json_sha256", std::string())
                     ? " - it is Qwen3.8's (Intel's checkpoint ships it): serve the original's tokenizer.json"
                     : "");
    return 1;
  }
  const tok::Tokenizer t(path);
  CHECK_EQ(t.vocab_size(), facts.at("vocab_size_with_added").get<uint32_t>());
  CHECK_EQ(t.vocab_size(), facts.at("len_tok").get<uint32_t>());
  CHECK_EQ(t.vocab_size(), 248077U);
  CHECK(facts.at("bos_id").is_null());
  CHECK_EQ(facts.at("tag_ids").size(), size_t(248077 - 248044));
  for (const auto& [tag, id] : facts.at("tag_ids").items()) {
    const auto got = t.token_to_id(tag);
    CHECK(got.has_value());
    CHECK_EQ(*got, id.get<uint32_t>());
    CHECK_EQ(t.id_to_token(id.get<uint32_t>()), tag);
  }
  const json gen = json::parse(slurp(snapshot + "/generation_config.json"));
  CHECK_EQ(gen.at("eos_token_id"), json({248046, 248044}));
  CHECK_EQ(facts.at("generation_eos"), json({248046, 248044}));
  CHECK_EQ(t.decode({248046}, false), std::string("<|im_end|>"));
  CHECK_EQ(t.decode({248044}, false), std::string("<|endoftext|>"));

  int bad = check_cases(t, facts);
  bad += check_corpus(t, facts, dir);
  bad += check_template(t, facts, dir, snapshot);
  CHECK_EQ(bad, 0);
  std::printf("qwen4exp_tokenizer_test OK (%s)\n", path.c_str());
  return 0;
}
