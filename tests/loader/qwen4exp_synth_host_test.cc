// Spec 21b: the loader's host half over make_synth.py's synthetic checkpoints - the Python writer and the C++
// reader held to each other on real files, no device (host only; SKIP 77 without the data).
//   qwen4exp_synth_host_test <oracle-out-q4exp-synth>      (<dir>/{ours,intel}/ckpt and ckpt-ple-int8)
// For each form: config.json -> QuantConfig::parse and model::qwen4exp_desc at the checkpoint's 4 layers; the
// forms from the names (ours: int4 dense / shared, g64; Intel's: bf16, g128) = the config's group_size;
// assert_quant_invariants; every name both ways with the MTP head; every layer and the head repacked, each
// layer's bytes = loader::q4_layer_bytes; the PLE I64 tensors = the formula for the synthetic base; the PLE
// int8 file read per head and held to the descriptor (check_q4_ple); nothing unconsumed; the plan of the
// 4-layer model with the head fits one 32.53 GB card.
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "check.h"
#include "common/json.h"
#include "loader/qwen4exp_layout.h"
#include "loader/qwen4exp_ple.h"
#include "loader/qwen4exp_repack.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/qwen4exp.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  const std::string root = argc > 1 ? argv[1] : "oracle-out-q4exp-synth";
  if (!fs::exists(fs::path(root) / "ours" / "ckpt" / "config.json") ||
      !fs::exists(fs::path(root) / "intel" / "ckpt" / "config.json")) {
    std::printf("SKIP: no synthetic checkpoints under %s (make_synth.py; box: r31.synth)\n", root.c_str());
    return 77;
  }
  for (const char* form : {"ours", "intel"}) {
    const std::string snap = root + "/" + form + "/ckpt/", ple = root + "/" + form + "/ckpt-ple-int8/";
    std::ifstream f(snap + "config.json");
    std::stringstream s;
    s << f.rdbuf();
    const common::json::Value cfg = common::json::parse(s.str());
    const loader::QuantConfig qc = loader::check_qwen4exp_checkpoint_config(cfg);
    loader::SafetensorsSet set(snap);
    const model::Qwen4ExpDesc pre = model::qwen4exp_desc(cfg, model::Q4Forms{});
    const model::Qwen4ExpDesc d = model::qwen4exp_desc(cfg, loader::q4_forms(set, pre));
    const bool ours = std::string(form) == "ours";
    CHECK_EQ(d.layers, 4u);
    CHECK(d.forms.dense == (ours ? model::Q4Form::Int4 : model::Q4Form::Bf16));
    CHECK(d.forms.shared == d.forms.dense && d.forms.mtp_experts == model::Q4Form::Bf16);
    CHECK_EQ(d.forms.expert_group, ours ? 64u : 128u);
    CHECK_EQ(qc.group_size, d.forms.expert_group);
    const loader::QuantScan scan = loader::assert_quant_invariants(set);
    loader::check_quant_scan(qc, scan);
    loader::Q4Checkpoint ck(d, set);
    ck.check_names(true);
    loader::Q4HostLayer h;
    for (uint32_t l = 0; l < d.layers; ++l) {
      ck.repack_layer(l, h);
      CHECK_EQ(h.bytes().total(), loader::q4_layer_bytes(d, l).total());
    }
    ck.repack_mtp(h);
    CHECK_EQ(h.bytes().total(), loader::q4_mtp_layer_bytes(d).total());
    (void)ck.embed();
    (void)ck.lm_head();
    CHECK_EQ(ck.final_mixer().size(), loader::q4_final_mixer_bytes(d));
    CHECK_EQ(ck.mtp_fc().size(), loader::q4_mtp_fc_offsets(d).total);
    CHECK_EQ(ck.mtp_mixer().size(), loader::q4_final_mixer_bytes(d));
    const loader::Q4PleConstants c = ck.ple_constants();
    CHECK(c.multipliers == loader::q4_ple_multipliers(d.vocab, d.ngram, 0, d.ple_seed));
    CHECK(c.sizes == loader::q4_ple_primes(d.ple_base, d.ple_heads, 0));
    CHECK(c.offsets == loader::q4_ple_offsets(c.sizes));
    const size_t shards = ck.skip_ple_shards();
    CHECK_EQ(shards, size_t(128));
    std::string names;
    const size_t left = ck.unconsumed(&names);
    if (left) std::fprintf(stderr, "unconsumed: %s\n", names.c_str());
    CHECK_EQ(left, size_t(0));
    const loader::Q4PleHost table(ple);
    loader::check_q4_ple(d, table);
    CHECK_EQ(table.bytes(), loader::q4_ple_table_bytes(c.sizes, d.ple_dim, table.scale()));
    const std::array<size_t, runtime::kPpDevices> caps = {32530000000ull, 32530000000ull};
    runtime::qwen4exp::require_fits(runtime::qwen4exp::plan(d, model::Q4Placement::one(d), 32768, true, true), caps,
                                    1500000000ull);
    std::printf("  %s: %zu tensors, forms dense %s / shared %s / experts g%u, 4 layers + the MTP head repacked "
                "(bytes = the layout), 128 PLE shards skipped by design, the int8 file (%s scales, %zu B) = the "
                "descriptor, 0 unconsumed\n",
                form, set.tensors().size(), model::q4_form_name(d.forms.dense), model::q4_form_name(d.forms.shared),
                d.forms.expert_group, loader::q4_ple_scale_name(table.scale()), table.bytes());
  }
  std::printf("qwen4exp_synth_host_test OK\n");
  return 0;
}
