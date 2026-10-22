// Spec 25e Task 1: Nemotron 3.5 Bolt's tokenizer through the engine's (the Rust `tokenizers` crate behind tok::Tokenizer,
// 0.22.2) against transformers 5.19.0's (tokenizers 0.23.2), from tools/tokenizer/dump_bolt.py's facts:
//   bolt_tokenizer_test <tests/tokenizer> <snapshot dir>
// <snapshot dir> (tokenizer.json + tokenizer_config.json + chat_template.jinja + generation_config.json; B70_BOLT_SNAPSHOT
// replaces it) is read at run time, never copied into the repository (Decision 20: generated fixtures are the default).
// Absent: SKIP (77). Checked:
//   - the id count (131072 = len(tok), every row of the head a token) and the added tokens 0-21 and 990 by name, id and
//     `special` flag, both ways - <think> 12, </think> 13, <tool_call> 14, </tool_call> 15, <tool_response> 16 and
//     </tool_response> 17 are NOT special (a decode that skips special tokens keeps them: the output parser reads them from
//     the text), <|im_start|> 10, <|im_end|> 11, <s> 1 and </s> 2 are;
//   - EVERY id's single-id decode and the ids that string encodes back to: FNV-1a digests over all 131072 and the count of
//     exact round trips (the byte-level partial tokens decode to U+FFFD, so the round trip is not the identity everywhere);
//   - 30 texts (code, CJK, digits, emoji, combining marks, every tag inline, Qwen XML calls, empty): encode without and with
//     special tokens - which adds NOTHING (no BOS anywhere) -, decode with and without skipping, the streamer's pieces
//     concatenating to the decode;
//   - tests/tokenizer/corpus.txt's 10240 lines: FNV-1a digests of the ids and the decodes, in 10 chunks;
//   - every template case (bolt_templates.json, rendered by chat::Template): the engine's render + encode equal HF's ids (a
//     tokenizer(text) and an apply_chat_template(tokenize = True) that the dump asserted equal), BOS never, and adding
//     special tokens adds nothing.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
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

void put_le32(std::vector<uint8_t>& out, uint32_t v) {
  for (int s = 0; s < 32; s += 8) out.push_back(static_cast<uint8_t>(v >> s));
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
    const bool enc = t.encode(text, false) == want;
    const bool enc_special = t.encode(text, true) == ids_of(c.at("ids_special")) && t.encode(text, true) == want;
    const std::string decoded = t.decode(want, false);
    const bool dec = decoded == c.at("decoded").get<std::string>();
    const bool dec_skip = t.decode(want, true) == c.at("decoded_skip").get<std::string>();
    const bool stream = streamed(t, want) == decoded;
    if (!(enc && enc_special && dec && dec_skip && stream)) {
      std::fprintf(stderr, "case %s: encode %d, encode+special %d, decode %d, decode-skip %d, stream %d\n",
                   json(text).dump().substr(0, 80).c_str(), enc, enc_special, dec, dec_skip, stream);
      ++bad;
    }
  }
  std::printf("text set: %zu cases, %d mismatches\n", facts.at("cases").size(), bad);
  return bad;
}

// Every single id: its decode and what that string encodes back to.
int check_all_ids(const tok::Tokenizer& t, const json& facts) {
  const json& want = facts.at("all_ids");
  uint64_t hd = 0xcbf29ce484222325ull, he = 0xcbf29ce484222325ull;
  size_t exact = 0;
  const uint32_t n = t.vocab_size();
  for (uint32_t i = 0; i < n; ++i) {
    const std::string s = t.decode({i}, false);
    const std::vector<uint32_t> back = t.encode(s, false);
    exact += back.size() == 1 && back[0] == i ? 1 : 0;
    std::vector<uint8_t> d;
    put_le32(d, i);
    d.insert(d.end(), s.begin(), s.end());
    d.push_back(0);
    hd = fnv1a64(d.data(), d.size(), hd);
    std::vector<uint8_t> e;
    for (uint32_t id : back) put_le32(e, id);
    for (int k = 0; k < 4; ++k) e.push_back(0xFF);
    he = fnv1a64(e.data(), e.size(), he);
  }
  const bool ok = n == want.at("count").get<uint32_t>() && exact == want.at("exact_round_trips").get<size_t>() &&
                  hex64(hd) == want.at("decode_fnv1a64").get<std::string>() &&
                  hex64(he) == want.at("reencode_fnv1a64").get<std::string>();
  std::printf("single ids: %u decoded and re-encoded, %zu exact round trips (HF: %zu), decode digest %s (HF %s), re-encode "
              "digest %s (HF %s)\n", n, exact, want.at("exact_round_trips").get<size_t>(), hex64(hd).c_str(),
              want.at("decode_fnv1a64").get<std::string>().c_str(), hex64(he).c_str(),
              want.at("reencode_fnv1a64").get<std::string>().c_str());
  if (!ok) std::fprintf(stderr, "the single-id round trips differ from HF's\n");
  return ok ? 0 : 1;
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
      for (uint32_t id : got) put_le32(le, id);
      for (int k = 0; k < 4; ++k) le.push_back(0xFF);
      he = fnv1a64(le.data(), le.size(), he);
      std::string text = t.decode(got, false);
      text.push_back('\0');
      hd = fnv1a64(reinterpret_cast<const uint8_t*>(text.data()), text.size(), hd);
    }
    total += ids;
    const bool ok = ids == chunk.at("ids").get<size_t>() && hex64(he) == chunk.at("encode_fnv1a64").get<std::string>() &&
                    hex64(hd) == chunk.at("decode_fnv1a64").get<std::string>();
    if (!ok) {
      std::fprintf(stderr, "corpus lines %zu..%zu: ids %zu (want %zu), encode %s (want %s), decode %s (want %s)\n", first,
                   first + n - 1, ids, chunk.at("ids").get<size_t>(), hex64(he).c_str(),
                   chunk.at("encode_fnv1a64").get<std::string>().c_str(), hex64(hd).c_str(),
                   chunk.at("decode_fnv1a64").get<std::string>().c_str());
      ++bad;
    }
  }
  std::printf("corpus: %zu lines, %zu ids, %d chunk mismatches\n", lines.size(), total, bad);
  return bad;
}

int check_renders(const tok::Tokenizer& t, const json& facts, const std::string& dir, const std::string& snapshot) {
  const json spec = json::parse(slurp(dir + "/bolt_templates.json"));
  const chat::Template tmpl(snapshot);
  std::map<std::string, const json*> by_name;
  for (const json& c : spec.at("cases")) by_name[c.at("name")] = &c;
  int bad = 0;
  size_t n = 0, ids_total = 0;
  for (const json& r : facts.at("renders")) {
    const std::string name = r.at("case");
    const auto it = by_name.find(name);
    CHECK(it != by_name.end());
    const json& c = *it->second;
    json kw = c.at("kwargs");
    kw["add_generation_prompt"] = c.at("add_generation_prompt").get<bool>();
    const bool thinking = !c.at("kwargs").contains("enable_thinking") || c.at("kwargs").at("enable_thinking").get<bool>();
    const std::string prompt = tmpl.render(c.at("messages"), c.at("tools"), thinking, kw);
    const std::vector<uint32_t> got = t.encode(prompt, false);
    const std::vector<uint32_t> want = ids_of(r.at("ids"));
    const bool special_adds_nothing = t.encode(prompt, true) == got;
    bool has_bos = false;
    for (uint32_t id : got) has_bos = has_bos || id == 1;
    if (got != want || !special_adds_nothing || has_bos) {
      std::fprintf(stderr, "render %s: %zu ids (want %zu), equal %d, add_special adds nothing %d, BOS present %d\n",
                   name.c_str(), got.size(), want.size(), got == want, special_adds_nothing, has_bos);
      ++bad;
    } else {
      ++n;
      ids_total += got.size();
    }
  }
  std::printf("renders: %zu cases, %zu equal to HF's ids (%zu ids), no BOS, %d mismatches\n", facts.at("renders").size(), n,
              ids_total, bad);
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: bolt_tokenizer_test <tests/tokenizer> <snapshot dir>\n");
    return 2;
  }
  const std::string dir = argv[1];
  const char* env_snap = std::getenv("B70_BOLT_SNAPSHOT");
  const std::string snapshot = env_snap != nullptr && *env_snap != '\0' ? env_snap : argv[2];
  const std::string path = snapshot + "/tokenizer.json";
  if (!exists(path) || !exists(snapshot + "/chat_template.jinja")) {
    std::printf("bolt_tokenizer_test: SKIP (no %s or no chat_template.jinja beside it; set B70_BOLT_SNAPSHOT)\n", path.c_str());
    return 77;
  }
  const json facts = json::parse(slurp(dir + "/bolt_tokenizer.json"));
  CHECK_EQ(chat::sha256_hex(slurp(path)), facts.at("tokenizer_json_sha256").get<std::string>());
  CHECK_EQ(facts.at("tokenizer_json_sha256").get<std::string>().substr(0, 8), std::string("d0432ad8"));
  const tok::Tokenizer t(path);

  // The id count: 131072, every row of the head a token.
  CHECK_EQ(t.vocab_size(), facts.at("vocab_size_with_added").get<uint32_t>());
  CHECK_EQ(t.vocab_size(), facts.at("len_tok").get<uint32_t>());
  CHECK_EQ(t.vocab_size(), 131072u);
  CHECK_EQ(facts.at("bos_id").get<uint32_t>(), 1u);
  CHECK_EQ(facts.at("eos_id").get<uint32_t>(), 11u);
  CHECK_EQ(facts.at("pad_id").get<uint32_t>(), 11u);
  CHECK_EQ(facts.at("added_token_count").get<uint32_t>(), 1000u);
  CHECK_EQ(facts.at("named").size(), 23u);
  for (const auto& [tag, v] : facts.at("named").items()) {
    const auto got = t.token_to_id(tag);
    CHECK(got.has_value());
    CHECK_EQ(*got, v.at("id").get<uint32_t>());
    CHECK_EQ(t.id_to_token(v.at("id").get<uint32_t>()), tag);
  }
  // The ids the server and the parser lean on, and which of them are special (decode skip_special_tokens).
  const json& named = facts.at("named");
  CHECK_EQ(named.at("<s>").at("id").get<uint32_t>(), 1u);
  CHECK_EQ(named.at("</s>").at("id").get<uint32_t>(), 2u);
  CHECK_EQ(named.at("<|im_start|>").at("id").get<uint32_t>(), 10u);
  CHECK_EQ(named.at("<|im_end|>").at("id").get<uint32_t>(), 11u);
  CHECK_EQ(named.at("<think>").at("id").get<uint32_t>(), 12u);
  CHECK_EQ(named.at("</think>").at("id").get<uint32_t>(), 13u);
  CHECK_EQ(named.at("<tool_call>").at("id").get<uint32_t>(), 14u);
  CHECK_EQ(named.at("</tool_call>").at("id").get<uint32_t>(), 15u);
  CHECK_EQ(named.at("<tool_response>").at("id").get<uint32_t>(), 16u);
  CHECK_EQ(named.at("</tool_response>").at("id").get<uint32_t>(), 17u);
  CHECK_EQ(named.at("<image>").at("id").get<uint32_t>(), 18u);
  CHECK_EQ(named.at("<SPECIAL_990>").at("id").get<uint32_t>(), 990u);
  CHECK(named.at("<|im_end|>").at("special").get<bool>());
  CHECK(named.at("<|im_start|>").at("special").get<bool>());
  CHECK(named.at("</s>").at("special").get<bool>());
  for (const char* tag : {"<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"})
    CHECK(!named.at(tag).at("special").get<bool>());
  CHECK_EQ(t.decode({12, 13, 14, 15, 16, 17}, true), std::string("<think></think><tool_call></tool_call><tool_response></tool_response>"));
  CHECK_EQ(t.decode({11}, true), std::string(""));
  CHECK_EQ(t.decode({11}, false), std::string("<|im_end|>"));
  const json gen = json::parse(slurp(snapshot + "/generation_config.json"));
  CHECK_EQ(gen.at("eos_token_id"), facts.at("generation_eos"));

  int bad = check_cases(t, facts);
  bad += check_all_ids(t, facts);
  bad += check_corpus(t, facts, dir);
  bad += check_renders(t, facts, dir, snapshot);
  CHECK_EQ(bad, 0);
  std::printf("bolt_tokenizer_test OK (%s)\n", path.c_str());
  return 0;
}
