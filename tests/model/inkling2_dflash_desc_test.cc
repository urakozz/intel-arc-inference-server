// Plan 41k Task 1 (host): Inkling 2 Mini's DFlash 2 drafter - model::DFlashDesc read from the published config
// (tests/model/inkling2_dflash/config.json, incoai/Inkling-2-Mini-DFlash2 @ 4f9088e; SOURCE beside it has the
// checkpoint's tensor shapes), held to its kernels and to Inkling (model::check_inkling_dflash2). No device.
//
//   1. The published config -> DFlash 2 (vLLM form) with the selector's softcap after the correction: 5 layers, hidden
//      2304, 32 / 8 x 128, FF 6912, every layer sliding at 2048 and non-causal, RoPE 1e4, eps 1e-6, block 8, mask
//      200056, taps 5 / 23 / 41 / 48 (layer visits; fc K = 4 x 2304 = 9216), conv 16 x 2 (576 kernel-projection rows),
//      selector rank 256 / top 16, vocabulary 201024.
//   2. selector_softcap_mode: absent is vLLM's order (before), an unknown value refused.
//   3. Refused by check_inkling_dflash2, each naming the key: the softcap before the correction, other taps, 16 q
//      heads, intermediate 17408, block 16, a causal layer, a window over 2048, num_target_layers 64, the mask a
//      padded row, another hidden, another vocabulary.
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/json.h"
#include "model/dflash.h"

namespace {

bool throws_with(const std::function<void()>& f, const std::string& needle) {
  try {
    f();
  } catch (const std::exception& e) {
    if (std::string(e.what()).find(needle) != std::string::npos) {
      std::printf("  refused: %s\n", e.what());
      return true;
    }
    std::fprintf(stderr, "  threw, but without '%s': %s\n", needle.c_str(), e.what());
    return false;
  }
  std::fprintf(stderr, "  did not throw (wanted '%s')\n", needle.c_str());
  return false;
}

std::string slurp(const std::string& p) {
  std::ifstream f(p);
  CHECK(f.good());
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}
std::string with(std::string s, const std::string& from, const std::string& to) {
  const size_t at = s.find(from);
  CHECK(at != std::string::npos);
  return s.replace(at, from.size(), to);
}
model::DFlashDesc parse(const std::string& text) { return model::dflash_desc_from_config(common::json::parse(text)); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: inkling2_dflash_desc_test <drafter config.json>\n");
    return 2;
  }
  const std::string real = slurp(argv[1]);

  // --- 1. the published config ------------------------------------------------------------------------------------------
  const model::DFlashDesc d = parse(real);
  CHECK(d.v2() && !d.dspark() && d.architecture == "DFlash2DraftModel" && d.softcap_after && d.softcap == 30.0f);
  CHECK(d.layers == 5 && d.hidden == 2304 && d.q_heads == 32 && d.kv_heads == 8 && d.head_dim == 128);
  CHECK(d.inter == 6912 && d.vocab == 201024 && d.block == 8 && d.max_k() == 7 && d.mask_id == 200056);
  CHECK((d.taps == std::vector<uint32_t>{5, 23, 41, 48}) && d.num_target_layers == 48);
  CHECK(d.window == std::vector<uint32_t>(5, 2048) && d.causal == std::vector<bool>(5, false));
  CHECK(d.conv_group == 16 && d.conv_taps == 2 && d.kproj_n() == 576 && d.sel_rank == 256 && d.sel_top_k == 16);
  CHECK(d.eps == 1e-6f && d.theta == 10000.0 && d.neox);
  CHECK(d.fc_k(2304) == 9216 && d.qkv_n() == 4096 + 2048 && d.gate_up_n() == 13824);
  model::check_inkling_dflash2(d);
  std::puts("1. the published config: DFlash 2, 5 layers at 2304, 32 / 8 x 128, FF 6912, block 8, taps 5 / 23 / 41 / 48 "
            "(fc K 9216), mask 200056, window 2048 non-causal, RoPE 1e4, the softcap after the correction");

  // --- 2. the softcap mode ------------------------------------------------------------------------------------------------
  CHECK(!parse(with(real, "\"selector_softcap_mode\": \"after_correction\",", "")).softcap_after);
  CHECK(throws_with([&] { parse(with(real, "\"after_correction\"", "\"sometimes\"")); }, "selector_softcap_mode"));
  std::puts("2. selector_softcap_mode: absent = before the correction (vLLM's), unknown refused");

  // --- 3. refused by the kernels' shapes ----------------------------------------------------------------------------------
  const auto refused = [&](const std::string& text, const std::string& needle) {
    CHECK(throws_with([&] { model::check_inkling_dflash2(parse(text)); }, needle));
  };
  refused(with(real, "\"after_correction\"", "\"before_correction\""), "after the correction");
  refused(with(real, "\"target_layer_ids\": [\n      5,", "\"target_layer_ids\": [\n      4,"), "target_layer_ids");
  refused(with(real, "\"num_attention_heads\": 32", "\"num_attention_heads\": 16"), "are not 32 / 8 x 128");
  refused(with(real, "\"intermediate_size\": 6912", "\"intermediate_size\": 17408"), "intermediate_size 17408");
  refused(with(real, "\"block_size\": 8", "\"block_size\": 16"), "block_size 16");
  refused(with(real, "\"is_causal\": false", "\"is_causal\": true"), "is causal");
  refused(with(real, "\"sliding_window\": 2048", "\"sliding_window\": 4096"), "sliding at most 2048");
  refused(with(real, "\"num_target_layers\": 48", "\"num_target_layers\": 64"), "num_target_layers 64");
  refused(with(real, "\"mask_token_id\": 200056", "\"mask_token_id\": 200060"), "mask_token_id 200060");
  refused(with(real, "\"hidden_size\": 2304", "\"hidden_size\": 2560"), "hidden_size 2560");
  refused(with(real, "\"vocab_size\": 201024", "\"vocab_size\": 248320"), "vocab_size 248320");
  std::puts("inkling2_dflash_desc_test OK");
  return 0;
}
