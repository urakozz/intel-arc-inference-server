// Spec 26b Task 1 (host): the LiLiCorr drafter's descriptor - model::DFlashDesc's third form (DFlashForm::LiLiCorr) read
// from the published config (tests/model/lilicorr/config.json, nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr @
// 449d2414), held to its kernels (model::check_lilicorr) and to Bolt (model::check_dflash_target against model::bolt()'s
// shape). No device, no weights.
//
//   1. The published config -> the form: the vLLM DFlash 2 backbone (5 layers, hidden 4096, 32 / 2 x 128, FF 12288, every
//      layer sliding at 1024 - |q - k| <= 1023 -, non-causal, conv 2 taps / groups of 16, RoPE 1e4, eps 1e-6), sinks,
//      block 8 from the top level, mask 990, taps 1 / 18 / 35 / 51 / 68 / 85, no selector, the raw head (multiplier 1,
//      no cap), the LiLiCorr head 1024 / 2 layers / 8 heads / MLP 2048 / factor 1024, top-8, vector_eps 1e-4, logit
//      scale 8.0; check_lilicorr passes.
//   2. Held to Bolt (hidden 4096, vocab 131072, every tap < 88, the mask id a row): another hidden, another vocabulary,
//      a shorter target, a mask past the rows - each refused by name.
//   3. Refused by name: has_own_lm_head, a draft vocabulary, candidate_topk 6 / 32, lilicorr_enabled false, a causal
//      layer, no convolutions, an output_multiplier, another quant_method / quant_algo / group; by check_lilicorr:
//      block 16, window 2048, no sinks, top-4, another head width.
//   4. 19b's and 23f's rules unchanged: the Qwen3.8 DFlash 2 config (tests/model/dflash/qwen38_dflash2_config.json)
//      still refuses a sink bias and a quantization_config, and still parses to the vLLM form with no LiLiCorr field set.
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/json.h"
#include "model/bolt.h"
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
  if (argc < 3) {
    std::fprintf(stderr, "usage: lilicorr_desc_test <lilicorr config.json> <qwen38_dflash2_config.json>\n");
    return 2;
  }
  const std::string real = slurp(argv[1]);
  const model::BoltDesc& bolt = model::bolt();
  const model::DFlashTargetShape tshape{bolt.name, bolt.layers, bolt.hidden, bolt.vocab};

  // --- 1. the published config ------------------------------------------------------------------------------------------
  const model::DFlashDesc d = parse(real);
  {
    CHECK(d.lilicorr() && !d.muse() && d.form == model::DFlashForm::LiLiCorr && d.v2());
    CHECK(d.architecture == "LiLiCorrDraftModel" && model::is_dflash_architecture(d.architecture));
    CHECK(d.layers == 5 && d.hidden == 4096 && d.q_heads == 32 && d.kv_heads == 2 && d.head_dim == 128);
    CHECK(d.inter == 12288 && d.vocab == 131072 && d.block == 8 && d.max_k() == 7 && d.mask_id == 990);
    CHECK((d.taps == std::vector<uint32_t>{1, 18, 35, 51, 68, 85}));
    CHECK(d.window == std::vector<uint32_t>(5, 1024) && d.causal == std::vector<bool>(5, false));
    CHECK(!d.window_inclusive && d.context_reach(0) == 1023 && !d.norm_two_roundings);
    CHECK(d.sinks && d.conv_taps == 2 && d.conv_group == 16 && d.kproj_n() == 1024 && d.sel_rank == 0);
    CHECK(d.eps == 1e-6f && d.theta == 10000.0 && d.neox);
    CHECK(d.in_scale == 1.0f && d.out_mult == 1.0f && d.softcap == 0.0f);
    CHECK(d.cand_topk == 8 && d.lil_hidden == 1024 && d.lil_layers == 2 && d.lil_heads == 8 && d.lil_mlp == 2048);
    CHECK(d.factor_dim == 1024 && d.vector_eps == 1e-4f && d.lil_logit_scale == 8.0f);
    CHECK(d.lil_slots() == 7 && d.lil_rows() == 56 && d.lil_head_dim() == 128);
    CHECK(d.fc_k(bolt.hidden) == 24576 && d.qkv_n() == 4608 && d.gate_up_n() == 24576);
    model::check_lilicorr(d);
    std::printf("1. the published LiLiCorr config: the LiLiCorr form, 5 sliding layers of 1024 (|q - k| <= 1023) with "
                "sinks, block 8, taps 1/18/35/51/68/85, top-8 into a 1024-wide 2-layer head; check_lilicorr passes\n");
  }

  // --- 2. held to Bolt ----------------------------------------------------------------------------------------------------
  {
    model::check_dflash_target(d, tshape);
    CHECK(throws_with([&] { model::check_dflash_target(d, {"x", 88, 5120, 131072}); }, "hidden_size 4096"));
    CHECK(throws_with([&] { model::check_dflash_target(d, {"x", 88, 4096, 151936}); }, "vocab_size 131072"));
    CHECK(throws_with([&] { model::check_dflash_target(d, {"x", 80, 4096, 131072}); }, "target_layer_ids holds 85"));
    model::DFlashDesc m = d;
    m.mask_id = 131072;
    CHECK(throws_with([&] { model::check_dflash_target(m, tshape); }, "mask_token_id 131072"));
    std::printf("2. held to %s: hidden 4096, vocab 131072, every tap < %u, the mask id a row\n", bolt.name.c_str(),
                bolt.layers);
  }

  // --- 3. refusals by name ------------------------------------------------------------------------------------------------
  {
    const std::string dc0 = "\"lilicorr_enabled\": true,";
    CHECK(throws_with([&] { (void)parse(with(real, dc0, "\"lilicorr_enabled\": false,")); }, "lilicorr_enabled"));
    CHECK(throws_with([&] { (void)parse(with(real, dc0, dc0 + " \"has_own_lm_head\": true,")); }, "has_own_lm_head"));
    CHECK(throws_with([&] { (void)parse(with(real, "\"vocab_size\": 131072,", "\"vocab_size\": 131072, \"draft_vocab_size\": 32768,")); },
                      "draft_vocab_size"));
    CHECK(throws_with([&] { (void)parse(with(real, "\"lilicorr_candidate_topk\": 8", "\"lilicorr_candidate_topk\": 6")); },
                      "lilicorr_candidate_topk 6"));
    CHECK(throws_with([&] { (void)parse(with(real, "\"lilicorr_candidate_topk\": 8", "\"lilicorr_candidate_topk\": 32")); },
                      "lilicorr_candidate_topk 32"));
    CHECK(throws_with([&] { (void)parse(with(real, "\"causal\": false,", "\"causal\": true,")); }, "is causal"));
    CHECK(throws_with([&] { (void)parse(with(real, "\"conv_kernel_size\": 2,", "")); }, "conv_kernel_size"));
    CHECK(throws_with([&] { (void)parse(with(real, dc0, dc0 + " \"output_multiplier\": 2.0,")); }, "output_multiplier"));
    CHECK(throws_with([&] { (void)parse(with(real, "\"quant_method\": \"modelopt_fp4\"", "\"quant_method\": \"compressed-tensors\"")); },
                      "quant_method compressed-tensors"));
    CHECK(throws_with([&] { (void)parse(with(real, "\"quant_algo\": \"W4A16_NVFP4\"", "\"quant_algo\": \"NVFP4\"")); },
                      "quant_algo NVFP4"));
    CHECK(throws_with([&] { (void)parse(with(real, "\"group_size\": 16,", "\"group_size\": 32,")); }, "group_size 32"));
    // the kernels' shapes
    CHECK(throws_with([&] { model::check_lilicorr(parse(with(real, "\"block_size\": 8", "\"block_size\": 16"))); },
                      "block_size 16"));
    CHECK(throws_with([&] { model::check_lilicorr(parse(with(real, "\"swa_window_size\": 1024", "\"swa_window_size\": 2048"))); },
                      "sliding_window 2048"));
    CHECK(throws_with([&] { model::check_lilicorr(parse(with(real, "\"attention_sink_bias\": true,\n    \"projector",
                                                             "\"attention_sink_bias\": false,\n    \"projector"))); },
                      "attention_sink_bias"));
    CHECK(throws_with([&] { model::check_lilicorr(parse(with(real, "\"lilicorr_candidate_topk\": 8", "\"lilicorr_candidate_topk\": 4"))); },
                      "lilicorr_candidate_topk 4"));
    CHECK(throws_with([&] { model::check_lilicorr(parse(with(real, "\"lilicorr_hidden_size\": 1024", "\"lilicorr_hidden_size\": 512"))); },
                      "the LiLiCorr head is 512"));
    model::DFlashDesc v = d;
    v.form = model::DFlashForm::Vllm;
    CHECK(throws_with([&] { model::check_lilicorr(v); }, "not LiLiCorrDraftModel"));
    std::printf("3. refused by name: has_own_lm_head, a draft vocabulary, top-k 6 / 32, lilicorr_enabled false, a causal "
                "layer, no convolutions, a multiplier, another NVFP4 form; check_lilicorr: block 16, window 2048, no sinks, "
                "top-4, another head\n");
  }

  // --- 4. 19b / 23f unchanged ---------------------------------------------------------------------------------------------
  {
    const std::string q = slurp(argv[2]);
    const model::DFlashDesc v = parse(q);
    CHECK(v.form == model::DFlashForm::Vllm && v.v2() && !v.sinks && v.cand_topk == 0 && v.lil_hidden == 0);
    CHECK(v.sel_rank == 256 && v.sel_top_k == 16 && v.block == 8);
    CHECK(throws_with([&] { (void)parse(with(q, "\"block_size\": 8,", "\"block_size\": 8, \"attention_sink_bias\": true,")); },
                      "attention_sink_bias is not supported"));
    CHECK(throws_with([&] { (void)parse(with(q, "\"block_size\": 8,", "\"block_size\": 8,")
                                              .insert(1, "\"quantization_config\": {\"quant_method\": \"modelopt_fp4\"}, ")); },
                      "quantization_config: a quantised drafter"));
    std::printf("4. the vLLM form unchanged: a sink bias and a quantization_config still refused there\n");
  }
  std::printf("lilicorr_desc_test OK\n");
  return 0;
}
