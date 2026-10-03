// Every row of model::Qwen35's linear table must have a compiled device
// binary. The table (shape + S + layout) and src/kernels/CMakeLists.txt's
// variant matrix are two lists maintained by hand, and plan 3 binds ~650
// kernels off the first one - a row whose .bin was never compiled is a
// zeModuleCreate failure 64 layers into a load, not a build error. This test
// makes it a build-tree error instead. No device, no checkpoint, milliseconds.
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>
#include "check.h"
#include "kernels/kernels.h"
#include "model/model_desc.h"
#include "model/qwen35.h"

int main() {
  using model::LinearId;
  using model::Qwen35;
  size_t checked = 0, missing = 0;
  // The eight table rows, plus the int4 `lm_head` row - which is NOT in the
  // table (LinearId::LmHead is one ordinal and `table()` is indexed by it), so
  // the loop below would never see it and the missing binary would surface as
  // a zeModuleCreate failure 13 s into a load of the RTN checkpoint. `lm_head`
  // is the one linear whose kind is a property of the checkpoint rather than of
  // the model (model/qwen35.h), so BOTH of its kinds must be compiled for the
  // engine to accept both checkpoints, and this is the list that says so.
  auto rows = [] {
    std::vector<model::FusedLinear> v;
    for (size_t i = 0; i < size_t(LinearId::kCount); ++i) v.push_back(model::qwen38().linear(LinearId(i)));
    v.push_back(model::qwen38().lm_head(model::WeightKind::Int4));
    v.push_back(model::qwen38().lm_head(model::WeightKind::Int8));   // spec 9: made at load
    // Spec 14: Agnes's two folded MLP rows (its other six are Qwen3.8's, checked above).
    v.push_back(model::agnes().linear(LinearId::GateUp));
    v.push_back(model::agnes().linear(LinearId::Down));
    return v;
  }();
  for (const model::FusedLinear& fl : rows) {
    const model::GemvShape& s = fl.shape;
    // bf16 rows have one tiled layout and no split-K variant; their `layout`
    // and `S` columns are fillers (docs/13-loader.md), so they are not part of
    // the name. What they DO carry is a tiling - a work-group width and a K
    // split - and it is per shape (spec 1.5 lever L2): the name below therefore
    // goes through the same `gemv_bf16_tiling` the runtime binds with, so this
    // test checks the binary runtime::build will actually open, not a sibling.
    const std::string variant =
        fl.kind == model::WeightKind::Int4   ? kernels::gemv_variant(1, s.K, s.N, s.S, s.layout)
        : fl.kind == model::WeightKind::Int8 ? kernels::gemv_i8w_variant(1, s.K, s.N)
                                             : kernels::gemv_bf16_variant(1, s.K, s.N, kernels::gemv_bf16_tiling(s.N));
    const std::string path = kernels::path(variant);
    if (!std::filesystem::exists(path)) {
      std::fprintf(stderr, "no device binary for table row %zu (%s, %s): %s\n", checked,
                   fl.parts[0].c_str(),
                   fl.kind == model::WeightKind::Int4   ? "int4"
                   : fl.kind == model::WeightKind::Int8 ? "int8"
                                                        : "bf16",
                   path.c_str());
      ++missing;
    }
    ++checked;
  }
  CHECK_EQ(missing, size_t(0));
  CHECK_EQ(checked, size_t(12));   // 8 table rows + int4/int8 lm_head + Agnes's 2 MLP rows
  // Spec 14: the per-model binaries that bake the MLP intermediate / GDN layer count.
  for (const std::string& v : {kernels::prep_silu_mul_variant(1, model::agnes().intermediate),
                               std::string("pf_silu_mul_I19456"),
                               kernels::gdn_step_slots_variant(1, model::agnes().gdn_layers),
                               kernels::argmax_stage1_variant(1, model::agnes().vocab_used)}) {
    if (!std::filesystem::exists(kernels::path(v))) {
      std::fprintf(stderr, "no device binary for Agnes: %s\n", kernels::path(v).c_str());
      ++missing;
    }
  }
  CHECK_EQ(missing, size_t(0));
  std::printf("kernel_table_test OK (%zu rows have a compiled binary: 8 table + int4/int8 lm_head"
              " + Agnes's 2; Agnes's silu / slots variants present)\n",
              checked);
  return 0;
}
