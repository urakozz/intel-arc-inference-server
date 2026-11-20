// PERF-53 (plan 21-QFN-v3b Task 0 Step 1, host): Qwen3.8-Flash-Next's DFlash 2 drafter - model::DFlashDesc read from the
// published config (tests/model/flash_next_dflash/config.json, incoai/Qwen3.8-Flash-Next-DFlash2 @ a22a174; SOURCE beside
// it has the checkpoint's tensor shapes), held to its kernels (model::check_flash_next_dflash) and to Flash-Next
// (model::check_dflash_target against model::qwen4exp()'s shape: hidden 2560, 48 layers, 248320 rows). No device.
//
//   1. The published config -> DFlash 2 (vLLM form): 5 layers, hidden 2560, 32 / 8 x 128, FF 7680, every layer sliding
//      at 2048 and non-causal, RoPE 1e4, eps 1e-6, block 8 (K <= 7; the engine checks <= 3), mask 248077 (a padded row
//      of the target's embedding), taps 5 / 15 / 25 / 35 / 45 (fc K = 5 x 2560 = 12800: the four streams' mean a tap),
//      conv 16 x 2 (640 kernel-projection rows), selector rank 256 / top 16.
//   2. Held to Flash-Next: another hidden, another vocabulary, a tap past the 48 layers, num_target_layers 64.
//   3. Refused by check_flash_next_dflash: 16 q heads, intermediate 17408 (Qwen3.8's drafter), other taps, block 16,
//      a causal layer, window 1024, mask 248070, DFlash v1.
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/json.h"
#include "model/dflash.h"
#include "model/qwen4exp.h"

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
    std::fprintf(stderr, "usage: flash_next_dflash_desc_test <drafter config.json>\n");
    return 2;
  }
  const std::string real = slurp(argv[1]);
  const model::Qwen4ExpDesc& fn = model::qwen4exp();
  const model::DFlashTargetShape tshape{fn.name, fn.layers, fn.hidden, fn.vocab};

  // --- 1. the published config ------------------------------------------------------------------------------------------
  const model::DFlashDesc d = parse(real);
  CHECK(d.v2() && !d.dspark() && d.architecture == "DFlash2DraftModel");
  CHECK(d.layers == 5 && d.hidden == 2560 && d.q_heads == 32 && d.kv_heads == 8 && d.head_dim == 128);
  CHECK(d.inter == 7680 && d.vocab == 248320 && d.block == 8 && d.max_k() == 7 && d.mask_id == 248077);
  CHECK((d.taps == std::vector<uint32_t>{5, 15, 25, 35, 45}) && d.num_target_layers == 48);
  CHECK(d.window == std::vector<uint32_t>(5, 2048) && d.causal == std::vector<bool>(5, false));
  CHECK(d.conv_group == 16 && d.conv_taps == 2 && d.kproj_n() == 640 && d.sel_rank == 256 && d.sel_top_k == 16);
  CHECK(d.eps == 1e-6f && d.theta == 10000.0 && d.neox);
  CHECK(d.fc_k(fn.hidden) == 12800 && d.qkv_n() == 4096 + 2048 && d.gate_up_n() == 15360);
  model::check_flash_next_dflash(d);
  model::check_dflash_target(d, tshape);
  std::printf("1. the published config: DFlash 2, 5 layers at 2560, 32 / 8 x 128, FF 7680, block 8, taps 5..45 (fc K "
              "12800), mask 248077, window 2048 non-causal, RoPE 1e4 - held to its kernels and to %s\n", fn.name.c_str());

  // --- 2. held to Flash-Next ----------------------------------------------------------------------------------------------
  const auto misfit = [&](const std::string& text, const std::string& needle) {
    CHECK(throws_with([&] { model::check_dflash_target(parse(text), tshape); }, needle));
  };
  misfit(with(real, "\"hidden_size\": 2560", "\"hidden_size\": 5120"), "hidden_size 5120");
  misfit(with(real, "\"vocab_size\": 248320", "\"vocab_size\": 262144"), "vocab_size 262144");
  misfit(with(real, "45\n", "48\n"), "holds 48");
  misfit(with(real, "\"num_target_layers\": 48", "\"num_target_layers\": 64"), "num_target_layers 64");

  // --- 3. refused by the kernels' shapes ----------------------------------------------------------------------------------
  const auto refused = [&](const std::string& text, const std::string& needle) {
    CHECK(throws_with([&] { model::check_flash_next_dflash(parse(text)); }, needle));
  };
  refused(with(real, "\"num_attention_heads\": 32", "\"num_attention_heads\": 16"), "are not 32 / 8 x 128");
  refused(with(real, "\"intermediate_size\": 7680", "\"intermediate_size\": 17408"), "intermediate_size 17408");
  refused(with(real, "\"target_layer_ids\": [\n      5,", "\"target_layer_ids\": [\n      4,"), "target_layer_ids");
  refused(with(real, "\"block_size\": 8", "\"block_size\": 16"), "block_size 16");
  refused(with(real, "\"is_causal\": false", "\"is_causal\": true"), "is causal");
  refused(with(real, "\"sliding_window\": 2048", "\"sliding_window\": 1024"), "sliding_window 1024");
  refused(with(real, "\"mask_token_id\": 248077", "\"mask_token_id\": 248070"), "mask_token_id 248070");
  std::puts("flash_next_dflash_desc_test OK");
  return 0;
}
