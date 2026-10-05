// Spec 18d: K2-Horizon's tokenizer through the engine's (the Rust `tokenizers` crate behind
// tok::Tokenizer) against Hugging Face's, from tools/tokenizer/dump_k2.py's facts:
//   k2_tokenizer_test <tests/tokenizer> <k2-snapshot-dir>
// tokenizer.json is <k2-snapshot-dir>/tokenizer.json unless B70_K2_TOKENIZER_JSON names it;
// absent, the test SKIPs (77). Checked:
//   - the vocabulary size with the added tokens, and every chat / tool tag's id both ways;
//   - a varied text set (prose, code, CJK, Cyrillic, Arabic, emoji, NFD input, zero-width
//     characters, digit runs, the special tokens inline, long runs, empty): encode without and
//     with special tokens (BOS 0 prepended by the post-processor, as HF does), decode with
//     and without them, and the streamer's pieces concatenating to the decode;
//   - tests/tokenizer/corpus.txt's 10240 lines: FNV-1a digests of the ids and the decodes,
//     in 10 chunks (so a mismatch names its chunk);
//   - three chat renders: the engine's renderer + encode(add_special = false) equal HF's
//     apply_chat_template(tokenize = True) ids - one BOS (the template writes it), never two;
//   - generation_config.json's EOS ids [1, 250019] (both stop generation, Review Focus 5).
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
    const std::vector<uint32_t> got = t.encode(text, false);
    const bool enc = got == want;
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

int check_renders(const tok::Tokenizer& t, const json& facts, const std::string& dir,
                  const std::string& snapshot) {
  if (!exists(snapshot + "/chat_template.jinja")) {
    std::printf("renders: SKIPPED (no chat_template.jinja in %s)\n", snapshot.c_str());
    return 0;
  }
  const json spec = json::parse(slurp(dir + "/k2_template_cases.json"));
  const chat::Template tmpl(snapshot);
  int bad = 0;
  for (const json& r : facts.at("renders")) {
    const std::string name = r.at("case");
    const json* found = nullptr;
    for (const json& c : spec.at("cases")) {
      if (c.at("name") == name) found = &c;
    }
    CHECK(found != nullptr);
    json tools = found->at("tools");
    if (tools.is_array()) {
      json named = json::array();
      for (const json& n : tools) named.push_back(spec.at("tools").at(n.get<std::string>()));
      tools = named;
    }
    const std::string prompt = tmpl.render(found->at("messages"), tools, true, found->at("kwargs"));
    const std::vector<uint32_t> got = t.encode(prompt, false);
    const std::vector<uint32_t> want = ids_of(r.at("ids"));
    // One BOS: the template writes <|ifm|begin_of_text|>; encoding the render WITH special
    // tokens would prepend a second (docs/probe-k2-2026-10-05.md, "BOS").
    const bool one_bos = !got.empty() && got[0] == 0 && (got.size() < 2 || got[1] != 0);
    const bool special_adds_one = t.encode(prompt, true).size() == got.size() + 1;
    if (got != want || !one_bos || !special_adds_one) {
      std::fprintf(stderr, "render %s: %zu ids (want %zu), equal %d, one BOS %d, add_special adds one %d\n",
                   name.c_str(), got.size(), want.size(), got == want, one_bos, special_adds_one);
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
    std::fprintf(stderr, "usage: k2_tokenizer_test <tests/tokenizer> <k2-snapshot-dir>\n");
    return 2;
  }
  const std::string dir = argv[1], snapshot = argv[2];
  const char* env = std::getenv("B70_K2_TOKENIZER_JSON");
  const std::string path = env != nullptr ? env : snapshot + "/tokenizer.json";
  if (!exists(path)) {
    std::printf("k2_tokenizer_test: SKIP (no %s; set B70_K2_TOKENIZER_JSON or -DB70_K2_SNAPSHOT_DIR)\n",
                path.c_str());
    return 77;
  }
  const json facts = json::parse(slurp(dir + "/k2_tokenizer.json"));
  const tok::Tokenizer t(path);

  CHECK_EQ(t.vocab_size(), facts.at("vocab_size_with_added").get<uint32_t>());
  CHECK_EQ(facts.at("bos_id").get<uint32_t>(), 0U);
  for (const auto& [tag, id] : facts.at("tag_ids").items()) {
    const auto got = t.token_to_id(tag);
    CHECK(got.has_value());
    CHECK_EQ(*got, id.get<uint32_t>());
    CHECK_EQ(t.id_to_token(id.get<uint32_t>()), tag);
  }
  if (exists(snapshot + "/generation_config.json")) {
    const json gen = json::parse(slurp(snapshot + "/generation_config.json"));
    CHECK_EQ(gen.at("eos_token_id"), json({1, 250019}));
    CHECK_EQ(t.decode({1}, false), std::string("<|ifm|endoftext|>"));
    CHECK_EQ(t.decode({250019}, false), std::string("<|ifm|im_end|>"));
  }

  int bad = check_cases(t, facts);
  bad += check_corpus(t, facts, dir);
  bad += check_renders(t, facts, dir, snapshot);
  CHECK_EQ(bad, 0);
  std::printf("k2_tokenizer_test OK (%s)\n", path.c_str());
  return 0;
}
