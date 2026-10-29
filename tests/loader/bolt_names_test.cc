// Spec 25b Task 4: the descriptor's tensor names against the served export's index (our g64 export's
// model.safetensors.index.json, a small file in the HF cache; no weight shard is read). argv[1] = that index; SKIP 77 when
// absent (the box's HF cache has it; the Mac's after the small-files download).
// The language names of the index (everything but the vision tower / projector) must equal bolt_expected_names(model::bolt())
// under the index's own trunk (`language_model.backbone.` in ours, `language_model.model.` in the GPTQ g128 spelling), plus
// one .g_idx beside each .qweight when the index ships any g_idx - exactly: the real 88-block pattern and the real E / I
// table, the real names.
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include "check.h"
#include "common/json.h"
#include "loader/bolt_repack.h"
#include "model/bolt.h"

int main(int argc, char** argv) {
  CHECK(argc > 1);
  struct stat st;
  if (stat(argv[1], &st) != 0) {
    std::printf("SKIP: no published index at %s\n", argv[1]);
    return 77;
  }
  std::ifstream f(argv[1]);
  std::stringstream s;
  s << f.rdbuf();
  const common::json::Value ix = common::json::parse(s.str());
  std::set<std::string> lang;
  size_t vision = 0;
  bool g_idx = false;
  for (const auto& [name, file] : ix.at("weight_map").obj()) {
    (void)file;
    if (loader::bolt_is_vision(name)) {
      ++vision;
    } else {
      lang.insert(name);
      g_idx = g_idx || (name.size() > 6 && name.compare(name.size() - 6, 6, ".g_idx") == 0);
    }
  }
  const std::string trunk =
      lang.count("language_model.backbone.embeddings.weight") ? "language_model.backbone." : "language_model.model.";
  const model::BoltDesc& d = model::bolt();
  std::set<std::string> want;
  for (const std::string& n : loader::bolt_expected_names(d, trunk)) {
    want.insert(n);
    if (g_idx && n.size() > 8 && n.compare(n.size() - 8, 8, ".qweight") == 0)
      want.insert(n.substr(0, n.size() - 8) + ".g_idx");
  }
  size_t missing = 0, extra = 0;
  for (const std::string& n : want)
    if (!lang.count(n) && missing++ < 3) std::fprintf(stderr, "missing from the index: %s\n", n.c_str());
  for (const std::string& n : lang)
    if (!want.count(n) && extra++ < 3) std::fprintf(stderr, "not in the descriptor: %s\n", n.c_str());
  CHECK_EQ(missing, size_t(0));
  CHECK_EQ(extra, size_t(0));
  CHECK_EQ(lang.size(), want.size());
  std::printf("bolt_names_test OK: the descriptor's %zu language names (trunk %s%s) equal the index's (+ %zu vision / "
              "projector tensors skipped by prefix)\n", want.size(), trunk.c_str(), g_idx ? ", + g_idx" : "", vision);
  return 0;
}
