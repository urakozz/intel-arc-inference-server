// Spec 23e Task 1: Muse-Glimmer-30B's tokenizer through the engine's (the Rust `tokenizers` crate behind
// tok::Tokenizer) against transformers 5.19.0's, from tools/tokenizer/dump_muse.py's tests/tokenizer/muse_tokenizer.json:
//   muse_tokenizer_test <tests/tokenizer> <tests/tokenizer/muse> <tokenizer.json>
// tokenizer.json (28 MB, identical in the bf16 original and both Vishva007 repos - spec 23 §1.2) is not vendored: the
// HF cache's meta-models/Muse-Glimmer-30B @ a4e59da5, or the environment's B70_MUSE_TOKENIZER_JSON. Absent: SKIP (77).
//   - the file's sha256 (c9dbee66...), the id count 202048 (200000 BPE + 2048 added), every named added token and the
//     drafter's mask id (201818) both ways, BOS 200000 / pad 200018, generation_config.json's EOS {200001, 200008} -
//     <|eom|> (200007) is NOT among them (a turn holds several messages);
//   - a 24-text set (prose, code, digits, emoji, CJK, the scripts whose combining marks the \p{M} split keeps, every
//     special tag inline, ATEM markup, a tool output, empty): encode without special tokens, encode WITH them (the
//     post-processor ADDS BOS), both decodes, the streamer's pieces concatenating to the decode;
//   - tests/tokenizer/corpus.txt's 10240 lines: FNV-1a digests of the ids and the decodes, in 10 chunks;
//   - Review Focus 1, one BOS: every template case of muse_templates.json rendered by chat::Template and encoded
//     WITHOUT special tokens (the server's encode) equals apply_chat_template(tokenize=True)'s ids, with exactly one
//     200000, at position 0 - and encoding it WITH special tokens would add a second (why the server never does).
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
using Ids = std::vector<uint32_t>;

bool exists(const std::string& path) { return std::ifstream(path).good(); }

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

uint64_t fnv1a64(const uint8_t* data, size_t n, uint64_t h) {
  for (size_t i = 0; i < n; ++i) h = (h ^ data[i]) * 0x100000001b3ull;
  return h;
}

std::string hex64(uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
  return buf;
}

std::string streamed(const tok::Tokenizer& t, const Ids& ids) {
  tok::Streamer s(t);
  std::string out;
  for (uint32_t id : ids) out += s.push(id);
  return out + s.flush();
}

int check_cases(const tok::Tokenizer& t, const json& facts) {
  int bad = 0;
  for (const json& c : facts.at("cases")) {
    const std::string text = c.at("text");
    const Ids want = c.at("ids").get<Ids>();
    const Ids want_special = c.at("ids_special").get<Ids>();
    const bool enc = t.encode(text, false) == want;
    const bool enc_special = t.encode(text, true) == want_special;
    // The post-processor's TemplateProcessing: BOS, then the text's ids.
    const bool bos = !want_special.empty() && want_special[0] == 200000 && Ids(want_special.begin() + 1, want_special.end()) == want;
    const std::string decoded = t.decode(want, false);
    const bool dec = decoded == c.at("decoded").get<std::string>();
    const bool dec_skip = t.decode(want, true) == c.at("decoded_skip").get<std::string>();
    const bool stream = streamed(t, want) == decoded;
    if (!(enc && enc_special && bos && dec && dec_skip && stream)) {
      std::fprintf(stderr, "case %s: encode %d, encode+special %d, BOS added %d, decode %d, decode-skip %d, stream %d\n",
                   json(text).dump().substr(0, 80).c_str(), enc, enc_special, bos, dec, dec_skip, stream);
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
      const Ids got = t.encode(lines[i], false);
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

// Review Focus 1: the renders' ids - one BOS, the template's.
int check_renders(const tok::Tokenizer& t, const json& facts, const json& templates, const std::string& muse) {
  const chat::Template tmpl(muse);
  int bad = 0;
  size_t i = 0;
  for (const json& r : facts.at("renders")) {
    const json& c = templates.at("cases").at(i++);
    CHECK_EQ(c.at("name").get<std::string>(), r.at("case").get<std::string>());
    json kw = c.at("kwargs");
    kw["add_generation_prompt"] = c.at("add_generation_prompt").get<bool>();
    const std::string prompt = tmpl.render(c.at("messages"), c.at("tools"), true, kw);
    const Ids got = t.encode(prompt, false);
    const Ids want = r.at("ids").get<Ids>();
    size_t bos = 0;
    for (uint32_t id : got) bos += id == 200000;
    const Ids doubled = t.encode(prompt, true);
    const bool second_bos = doubled.size() == got.size() + 1 && doubled[0] == 200000 && doubled[1] == 200000;
    if (got != want || bos != 1 || got.empty() || got[0] != 200000 || !second_bos) {
      std::fprintf(stderr, "render %s: %zu ids (want %zu), equal %d, BOS count %zu, add_special doubles it %d\n",
                   r.at("case").get<std::string>().c_str(), got.size(), want.size(), got == want, bos, second_bos);
      ++bad;
    }
  }
  std::printf("renders: %zu cases' ids equal apply_chat_template(tokenize=True)'s, one BOS at 0 each, %d mismatches\n",
              facts.at("renders").size(), bad);
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: muse_tokenizer_test <tests/tokenizer> <tests/tokenizer/muse> <tokenizer.json>\n");
    return 2;
  }
  const std::string dir = argv[1], muse = argv[2];
  const char* env = std::getenv("B70_MUSE_TOKENIZER_JSON");
  const std::string path = env != nullptr && *env != '\0' ? env : argv[3];
  if (!exists(path)) {
    std::printf("muse_tokenizer_test: SKIP (no %s; set B70_MUSE_TOKENIZER_JSON)\n", path.c_str());
    return 77;
  }
  const json facts = json::parse(slurp(dir + "/muse_tokenizer.json"));
  const json templates = json::parse(slurp(dir + "/muse_templates.json"));
  const std::string file_sha = chat::sha256_hex(slurp(path));
  if (file_sha != facts.at("tokenizer_json_sha256").get<std::string>()) {
    std::fprintf(stderr, "%s has sha256 %s, not Muse-Glimmer's %s\n", path.c_str(), file_sha.c_str(),
                 facts.at("tokenizer_json_sha256").get<std::string>().c_str());
    return 1;
  }
  CHECK_EQ(file_sha.substr(0, 8), std::string("c9dbee66"));
  const tok::Tokenizer t(path);
  CHECK_EQ(t.vocab_size(), facts.at("vocab_size_with_added").get<uint32_t>());
  CHECK_EQ(t.vocab_size(), facts.at("len_tok").get<uint32_t>());
  CHECK_EQ(t.vocab_size(), 202048U);
  CHECK_EQ(facts.at("bos_id").get<uint32_t>(), 200000U);
  CHECK_EQ(facts.at("pad_id").get<uint32_t>(), 200018U);
  const json want_named = {{"<|begin_of_text|>", 200000}, {"<|end_of_text|>", 200001}, {"<|eom|>", 200007},
                           {"<|eot|>", 200008},         {"<|finetune_right_pad|>", 200018}, {"<|start|>", 200022},
                           {"<|message|>", 200023},     {"<|patch|>", 200092}, {"<|reserved_special_token_1818|>", 201818}};
  for (const auto& [tag, id] : want_named.items()) CHECK_EQ(facts.at("named_ids").at(tag), id);
  for (const auto& [tag, id] : facts.at("named_ids").items()) {
    const auto got = t.token_to_id(tag);
    CHECK(got.has_value());
    CHECK_EQ(*got, id.get<uint32_t>());
    CHECK_EQ(t.id_to_token(id.get<uint32_t>()), tag);
    CHECK_EQ(t.decode({id.get<uint32_t>()}, false), tag);
    CHECK_EQ(t.decode({id.get<uint32_t>()}, true), std::string());
  }
  const json gen = json::parse(slurp(muse + "/generation_config.json"));
  CHECK_EQ(gen.at("eos_token_id"), json({200001, 200008}));
  CHECK_EQ(facts.at("generation_eos"), json({200001, 200008}));
  std::printf("ids: 202048, the %zu named added tokens both ways, EOS {200001, 200008} (<|eom|> 200007 is not one)\n",
              facts.at("named_ids").size());

  int bad = check_cases(t, facts);
  bad += check_corpus(t, facts, dir);
  bad += check_renders(t, facts, templates, muse);
  CHECK_EQ(bad, 0);
  std::printf("muse_tokenizer_test OK (%s)\n", path.c_str());
  return 0;
}
