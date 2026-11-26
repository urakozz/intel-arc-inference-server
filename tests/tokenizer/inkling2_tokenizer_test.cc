// Spec 41f Task 1: Inkling 2 Mini's tokenizer and template through the engine's (the Rust `tokenizers` crate behind
// tok::Tokenizer, minja behind chat::Template) against Hugging Face's, from tools/tokenizer/dump_inkling2.py's facts:
//   inkling2_tokenizer_test <tests/tokenizer> <template dir> <tokenizer.json> [<A4 set dir>]
// <template dir> holds the checkpoint's chat_template.jinja and tokenizer_config.json (tests/tokenizer/inkling2,
// vendored). The environment's B70_INKLING2_TOKENIZER_JSON replaces the third argument. No tokenizer.json: SKIP (77).
// Checked:
//   - the id count (200058) and every content / message tag's id;
//   - a text set (code, digits, emoji, the tags inline, empty): encode without and with special tokens (no BOS is ever
//     added), decode with and without them, the streamer's pieces concatenating to the decode;
//   - ignore_merges: ~400 words that are whole vocabulary entries each encode to HF's ids (one id each);
//   - tests/tokenizer/corpus.txt's 10240 lines: FNV-1a digests of the ids and the decodes, in 10 chunks;
//   - every template case's render (inkling2_template_cases.json): the engine's renderer + encode equal HF's
//     apply_chat_template(tokenize = True) ids;
//   - with an A4 set dir (oracle-out/inkling/a4/set: tools/toolcall/make_set.py --from, the checkpoint's template):
//     every scenario's messages, tools and chat_template_kwargs rendered and encoded by the engine equal its .ids,
//     byte for byte. A set dir that does not exist is said and skipped (the set is made by the CPU queue).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
    const bool no_bos = ids_of(c.at("ids_special")) == want;
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
  std::printf("text set: %zu cases, %d mismatches\n", facts.at("cases").size(), bad);
  return bad;
}

int check_words(const tok::Tokenizer& t, const json& facts) {
  int bad = 0;
  for (const json& w : facts.at("words")) {
    const std::string text = w.at("text");
    if (t.encode(text, false) != ids_of(w.at("ids"))) {
      if (bad < 8) std::fprintf(stderr, "word %s: ids differ from HF's\n", json(text).dump().c_str());
      ++bad;
    }
  }
  std::printf("whole vocabulary words (ignore_merges): %zu, %d mismatches\n", facts.at("words").size(), bad);
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
    if (ids != chunk.at("ids").get<size_t>() || hex64(he) != chunk.at("encode_fnv1a64").get<std::string>() ||
        hex64(hd) != chunk.at("decode_fnv1a64").get<std::string>()) {
      std::fprintf(stderr, "corpus lines %zu..%zu differ from HF's\n", first, first + n - 1);
      ++bad;
    }
  }
  std::printf("corpus: %zu lines, %d chunk mismatches\n", lines.size(), bad);
  return bad;
}

int check_renders(const tok::Tokenizer& t, const json& facts, const std::string& dir, const chat::Template& tmpl) {
  const json spec = json::parse(slurp(dir + "/inkling2_template_cases.json"));
  int bad = 0;
  for (const json& r : facts.at("renders")) {
    const std::string name = r.at("case");
    const json* found = nullptr;
    for (const json& c : spec.at("cases"))
      if (c.at("name") == name) found = &c;
    CHECK(found != nullptr);
    json tools = found->at("tools");
    if (tools.is_array()) {
      json named = json::array();
      for (const json& n : tools) named.push_back(spec.at("tools").at(n.get<std::string>()));
      tools = named;
    }
    const std::vector<uint32_t> got = t.encode(tmpl.render(found->at("messages"), tools, true, found->at("kwargs")), false);
    if (got != ids_of(r.at("ids"))) {
      std::fprintf(stderr, "render %s: %zu ids (want %zu)\n", name.c_str(), got.size(), r.at("ids").size());
      ++bad;
    }
  }
  std::printf("renders: %zu, %d mismatches\n", facts.at("renders").size(), bad);
  return bad;
}

// The A4 set's scenarios: <name>.json (messages, tools, chat_template_kwargs) and <name>.ids, rendered by HF.
int check_a4(const tok::Tokenizer& t, const chat::Template& tmpl, const std::string& set) {
  if (!std::filesystem::is_directory(set)) {
    std::printf("A4 set: %s does not exist yet, not checked\n", set.c_str());
    return 0;
  }
  int bad = 0, n = 0;
  for (const auto& entry : std::filesystem::directory_iterator(set)) {
    const std::string path = entry.path().string();
    if (entry.path().extension() != ".json" || entry.path().filename() == "manifest.json") continue;
    const json s = json::parse(slurp(path));
    const std::string ids_path = path.substr(0, path.size() - 5) + ".ids";
    std::vector<uint32_t> want;
    std::istringstream in(slurp(ids_path));
    for (uint32_t id; in >> id;) want.push_back(id);
    const json kwargs = s.contains("chat_template_kwargs") ? s.at("chat_template_kwargs") : json::object();
    const bool think = s.contains("enable_thinking") ? s.at("enable_thinking").get<bool>() : true;
    const std::vector<uint32_t> got = t.encode(tmpl.render(s.at("messages"), s.at("tools"), think, kwargs), false);
    ++n;
    if (got != want) {
      size_t i = 0;
      while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
      std::fprintf(stderr, "A4 %s: %zu ids (want %zu), first difference at %zu\n", entry.path().filename().c_str(),
                   got.size(), want.size(), i);
      ++bad;
    }
  }
  std::printf("A4 set: %d scenarios, %d mismatches\n", n, bad);
  CHECK(n > 0);
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::fprintf(stderr, "usage: inkling2_tokenizer_test <tests/tokenizer> <template dir> <tokenizer.json> [<A4 set>]\n");
    return 2;
  }
  const std::string dir = argv[1], snapshot = argv[2];
  const char* env = std::getenv("B70_INKLING2_TOKENIZER_JSON");
  const std::string path = env != nullptr && *env != '\0' ? env : argv[3];
  if (!exists(path)) {
    std::printf("inkling2_tokenizer_test: SKIP (no %s; set B70_INKLING2_TOKENIZER_JSON)\n", path.c_str());
    return 77;
  }
  const json facts = json::parse(slurp(dir + "/inkling2_tokenizer.json"));
  const tok::Tokenizer t(path);
  CHECK_EQ(t.vocab_size(), facts.at("vocab_size_with_added").get<uint32_t>());
  for (const auto& [tag, id] : facts.at("tag_ids").items()) {
    CHECK(t.encode(tag, false) == std::vector<uint32_t>{id.get<uint32_t>()});
    CHECK_EQ(t.decode({id.get<uint32_t>()}, false), tag);
  }
  CHECK_EQ(facts.at("eos_id").get<uint32_t>(), 200006u);
  const chat::Template tmpl(snapshot);
  int bad = check_cases(t, facts) + check_words(t, facts) + check_corpus(t, facts, dir) +
            check_renders(t, facts, dir, tmpl);
  if (argc == 5) bad += check_a4(t, tmpl, argv[4]);
  CHECK_EQ(bad, 0);
  std::printf("inkling2_tokenizer_test OK\n");
  return 0;
}
