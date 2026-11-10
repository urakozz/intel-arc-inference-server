// Engine routes for f32-judged replay. Tool arguments only; no injected engine routing.
// checkpoint ids original-f32-dump output layers [pp2] [prompt-length=4500] [gen=32]
#include <algorithm>
#include <cstring>
#include <filesystem>

#include "golden/golden_common.h"
#include "loader/dflash_fixture.h"
#include "runtime/qwen4exp_rig.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace {
namespace rq = runtime::qwen4exp;
using Tensors = dflash_fixture::Tensors;
template<class T> void put(Tensors& ts, const std::string& name, const char* dtype,
                           std::vector<uint64_t> shape, const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!bytes.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
  ts[name] = {dtype, std::move(shape), std::move(bytes)};
}
}

int main(int argc, char** argv) {
  if (argc < 6) return 2;
  qwen4exp_rig::Options o;
  o.layers = uint32_t(std::stoul(argv[5]));
  o.devices = argc > 6 && std::string(argv[6]) == "pp2" ? 2 : 1;
  o.max_len = 8192;
  o.int8_head = true;
  const uint32_t n = argc > 7 ? uint32_t(std::stoul(argv[7])) : 4500;
  const uint32_t gen = argc > 8 ? uint32_t(std::stoul(argv[8])) : 32;
  CHECK(n > 0 && n + gen <= o.max_len);
  const auto src = qwen4exp_rig::read_ids(argv[2], 248320);
  CHECK(!src.empty());
  std::vector<uint32_t> prompt(n), force;
  for (uint32_t p = 0; p < n; ++p) prompt[p] = src[p % src.size()];
  golden::Golden original(argv[3]);
  CHECK(original.dim("tokens", 1, 0) >= gen);
  const int32_t* tokens = original.i32("tokens", original.dim("tokens", 1, 0));
  for (uint32_t j = 0; j < gen; ++j) { CHECK(tokens[j] >= 0); force.push_back(uint32_t(tokens[j])); }
  const std::string snap = qwen4exp_rig::qwen4exp_snapshot(argv[1]);
  if (snap.empty()) return 77;
  qwen4exp_rig::Rig rig;
  qwen4exp_rig::build(rig, snap, o);
  auto& e = *rig.eng;
  const auto& d = e.model().desc;
  Tensors out;
  put(out, "trace.spec", "I32", {10}, std::vector<int32_t>{1, int32_t(d.layers), int32_t(n), int32_t(gen),
      int32_t(rq::kPfC), int32_t(o.max_len), 1, 1, int32_t(d.vocab_used), int32_t(d.vocab)});
  put(out, "prompt.ids", "I64", {n}, std::vector<int64_t>(prompt.begin(), prompt.end()));
  put(out, "tokens", "I64", {gen}, std::vector<int64_t>(force.begin(), force.end()));
  for (const std::string mode : {"prefill", "decode"}) {
    std::vector<std::vector<int32_t>> routes(d.layers, std::vector<int32_t>(size_t(n + gen) * d.top_k));
    const auto add = [&](const std::vector<uint32_t>& words, uint32_t pos, uint32_t count, bool pf) {
      for (uint32_t l = 0; l < d.layers; ++l)
        for (uint32_t r = 0; r < count; ++r) {
          const uint32_t* ids = words.data() + (pf ? rq::pf_route_at(l) / 4 + size_t(r) * rq::kRouteWords
                                                  : size_t(l) * rq::kRouteWords);
          for (uint32_t k = 0; k < d.top_k; ++k) routes[l][size_t(pos + r) * d.top_k + k] = int32_t(ids[k]);
        }
    };
    e.reset();
    if (mode == "prefill") {
      for (uint32_t p = 0; p < n; p += rq::kPfC) {
        const uint32_t count = std::min(rq::kPfC, n - p);
        e.prefill(std::vector<uint32_t>(prompt.begin() + p, prompt.begin() + p + count));
        add(e.read_prefill_routes(), p, count, true);
      }
    } else {
      for (uint32_t p = 0; p < n; ++p) { e.ingest({prompt[p]}); add(e.read_routes(), p, 1, false); }
    }
    for (uint32_t l = 0; l < d.layers; ++l) {
      const auto bytes = e.read_layer_state(l);
      put(out, mode + ".state.L" + std::to_string(l), "U8", {bytes.size()}, bytes);
    }
    std::vector<float> logits = e.read_logits();
    for (uint32_t j = 0; j < gen; ++j) {
      CHECK(force[j] < d.vocab_used);
      e.ingest({force[j]});
      add(e.read_routes(), n + j, 1, false);
      const auto row = e.read_logits();
      logits.insert(logits.end(), row.begin(), row.end());
    }
    put(out, mode + ".logits", "F32", {gen + 1, d.vocab}, logits);
    for (uint32_t l = 0; l < d.layers; ++l)
      put(out, mode + ".route.ids.L" + std::to_string(l), "I32", {n + gen, d.top_k}, routes[l]);
  }
  std::filesystem::create_directories(std::filesystem::path(argv[4]).parent_path());
  dflash_fixture::write_safetensors(argv[4], out);
  std::printf("captured prefill/decode engine routes: %u prompt + %u forced rows, %u layers -> %s\n", n, gen, d.layers, argv[4]);
}
