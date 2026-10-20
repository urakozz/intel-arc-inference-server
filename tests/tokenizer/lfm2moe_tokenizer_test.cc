// Spec 24e (plan 24e Task 1): LFM2.5-24B-A2B's tokenizer through the engine's (the Rust `tokenizers` crate behind
// tok::Tokenizer, 0.22.2) against transformers 5.19.0's (tokenizers 0.23.2), from tools/tokenizer/dump_lfm2moe.py's
// facts:
//   lfm2moe_tokenizer_test <tests/tokenizer> <snapshot dir> <tokenizer.json>
// <snapshot dir> (tokenizer_config.json + chat_template.jinja + generation_config.json; B70_LFM_SNAPSHOT_DIR replaces
// it) is used for the template renders; B70_LFM_TOKENIZER_JSON replaces the third argument. Either absent: SKIP (77).
// Checked:
//   - the id count (125017 = len(tok), ids 0..125016 contiguous) and that the head's remaining rows (125017..127999)
//     have no token - what the server's sampler masks (vocab_used);
//   - all 124 added tokens by name, id and `special` flag, both ways - <|tool_call_start|> 124905 and
//     <|tool_call_end|> 124906 are NOT special (a decode that skips special tokens keeps them), <|im_end|> 124900 is;
//   - 21 texts (code, CJK, digits, emoji, contractions, the tags inline, pythonic calls between the non-special
//     tool-call tags, empty): encode without and with special tokens, decode with and without skipping special tokens
//     (the call tags survive the skip, <|im_end|> does not), and the streamer's pieces concatenating to the decode;
//   - tests/tokenizer/corpus.txt's 10240 lines: FNV-1a digests of the ids and the decodes, in 10 chunks;
//   - every template case and the agentic session: the engine's render + encode equal HF's ids (a tokenizer(text)
//     and an apply_chat_template(tokenize = True) that the dump asserted equal), with BOS exactly once, and adding
//     special tokens adds nothing;
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
    const bool enc_special = t.encode(text, true) == ids_of(c.at("ids_special"));
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
      for (uint32_t id : got) {
        for (int s = 0; s < 32; s += 8) le.push_back(static_cast<uint8_t>(id >> s));
      }
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

int check_renders(const tok::Tokenizer& t, const json& facts, const std::string& dir, const std::string& prompts,
                  const std::string& snapshot) {
  const json spec = json::parse(slurp(dir + "/lfm2moe_template_cases.json"));
  const chat::Template tmpl(snapshot);
  json cases = spec.at("cases");
  const json agentic = json::parse(slurp(prompts + "/lfm2moe_agentic.json"));
  cases.push_back({{"name", "agentic"}, {"messages", agentic.at("messages")}, {"tools", agentic.at("tools")},
                   {"kwargs", json::object()}, {"inline_tools", true}});
  int bad = 0;
  size_t n = 0;
  const uint32_t bos = *t.token_to_id("<|startoftext|>");
  for (const json& c : cases) {
    const std::string name = c.at("name");
    json tools = c.at("tools");
    if (tools.is_array() && !c.contains("inline_tools")) {
      json named = json::array();
      for (const json& tn : tools) named.push_back(spec.at("tools").at(tn.get<std::string>()));
      tools = named;
    }
    const std::string prompt = tmpl.render(c.at("messages"), tools, true, c.at("kwargs"));
    const std::vector<uint32_t> got = t.encode(prompt, false);
    const std::vector<uint32_t> want = ids_of(facts.at("template_ids").at(name));
    const bool special_adds_nothing = t.encode(prompt, true) == got;
    size_t boses = 0;
    for (uint32_t id : got) boses += id == bos ? 1 : 0;
    if (got != want || !special_adds_nothing || boses != 1 || got.front() != bos) {
      std::fprintf(stderr, "render %s: %zu ids (want %zu), equal %d, add_special adds nothing %d, BOS x%zu\n",
                   name.c_str(), got.size(), want.size(), got == want, special_adds_nothing, boses);
      ++bad;
    } else {
      ++n;
    }
  }
  std::printf("renders: %zu cases, %zu equal to HF's ids, %d mismatches\n", cases.size(), n, bad);
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: lfm2moe_tokenizer_test <tests/tokenizer> <tests/golden/prompts> <snapshot dir> "
                         "<tokenizer.json>\n");
    return 2;
  }
  const std::string dir = argv[1], prompts = argv[2];
  const char* env_snap = std::getenv("B70_LFM_SNAPSHOT_DIR");
  const std::string snapshot = env_snap != nullptr && *env_snap != '\0' ? env_snap : argv[3];
  const char* env = std::getenv("B70_LFM_TOKENIZER_JSON");
  const std::string path = env != nullptr && *env != '\0' ? env : argv[4];
  if (!exists(path) || !exists(snapshot + "/chat_template.jinja")) {
    std::printf("lfm2moe_tokenizer_test: SKIP (no %s or no %s/chat_template.jinja; set B70_LFM_TOKENIZER_JSON / "
                "B70_LFM_SNAPSHOT_DIR)\n", path.c_str(), snapshot.c_str());
    return 77;
  }
  const json facts = json::parse(slurp(dir + "/lfm2moe_tokenizer.json"));
  CHECK_EQ(chat::sha256_hex(slurp(path)), facts.at("tokenizer_json_sha256").get<std::string>());
  const tok::Tokenizer t(path);

  // The id count: 125017, ids 0..125016; the head's other rows have no token.
  CHECK_EQ(t.vocab_size(), facts.at("vocab_size_with_added").get<uint32_t>());
  CHECK_EQ(t.vocab_size(), facts.at("len_tok").get<uint32_t>());
  CHECK_EQ(t.vocab_size(), 125017u);
  CHECK_EQ(facts.at("undefined_from").get<uint32_t>(), t.vocab_size());
  CHECK_EQ(facts.at("head_rows").get<uint32_t>(), 128000u);
  CHECK_EQ(facts.at("bos_id").get<uint32_t>(), 124894u);
  CHECK_EQ(facts.at("eos_id").get<uint32_t>(), 124900u);
  CHECK_EQ(facts.at("pad_id").get<uint32_t>(), 124893u);
  CHECK_EQ(facts.at("added_tokens").size(), 124u);
  for (const auto& [tag, v] : facts.at("added_tokens").items()) {
    const auto got = t.token_to_id(tag);
    CHECK(got.has_value());
    CHECK_EQ(*got, v.at("id").get<uint32_t>());
    CHECK_EQ(t.id_to_token(v.at("id").get<uint32_t>()), tag);
  }
  // The ids the server and the parser lean on, and which of them are special (decode skip_special_tokens).
  const json& added = facts.at("added_tokens");
  CHECK_EQ(added.at("<|startoftext|>").at("id").get<uint32_t>(), 124894u);
  CHECK_EQ(added.at("<|im_end|>").at("id").get<uint32_t>(), 124900u);
  CHECK_EQ(added.at("<think>").at("id").get<uint32_t>(), 124901u);
  CHECK_EQ(added.at("</think>").at("id").get<uint32_t>(), 124902u);
  CHECK_EQ(added.at("<|tool_call_start|>").at("id").get<uint32_t>(), 124905u);
  CHECK_EQ(added.at("<|tool_call_end|>").at("id").get<uint32_t>(), 124906u);
  CHECK(added.at("<|im_end|>").at("special").get<bool>());
  CHECK(!added.at("<|tool_call_start|>").at("special").get<bool>());
  CHECK(!added.at("<|tool_call_end|>").at("special").get<bool>());
  CHECK_EQ(t.decode({124905, 124906}, true), std::string("<|tool_call_start|><|tool_call_end|>"));
  CHECK_EQ(t.decode({124900}, true), std::string(""));
  CHECK_EQ(t.decode({124900}, false), std::string("<|im_end|>"));
  const json gen = json::parse(slurp(snapshot + "/generation_config.json"));
  CHECK_EQ(gen.at("eos_token_id"), json({124900}));

  int bad = check_cases(t, facts);
  bad += check_corpus(t, facts, dir);
  bad += check_renders(t, facts, dir, prompts, snapshot);
  CHECK_EQ(bad, 0);
  std::printf("lfm2moe_tokenizer_test OK (%s)\n", path.c_str());
  return 0;
}
