// Every row of model::Qwen35's linear table must have a compiled device
// binary. The table (shape + S + layout) and src/kernels/CMakeLists.txt's
// variant matrix are two lists maintained by hand, and plan 3 binds ~650
// kernels off the first one - a row whose .bin was never compiled is a
// zeModuleCreate failure 64 layers into a load, not a build error. This test
// makes it a build-tree error instead. No device, no checkpoint, milliseconds.
#include <cstdio>
#include <filesystem>
#include <string>
#include "check.h"
#include "kernels/kernels.h"
#include "model/qwen35.h"

int main() {
  using model::LinearId;
  using model::Qwen35;
  size_t checked = 0, missing = 0;
  for (size_t i = 0; i < size_t(LinearId::kCount); ++i) {
    const model::FusedLinear& fl = Qwen35::linear(LinearId(i));
    const model::GemvShape& s = fl.shape;
    // bf16 rows have one tiled layout and no split-K variant; their `layout`
    // and `S` columns are fillers (docs/13-loader.md), so they are not part of
    // the name.
    const std::string variant = fl.kind == model::WeightKind::Int4
                                    ? kernels::gemv_variant(1, s.K, s.N, s.S, s.layout)
                                    : kernels::gemv_bf16_variant(1, s.K, s.N);
    const std::string path = kernels::path(variant);
    if (!std::filesystem::exists(path)) {
      std::fprintf(stderr, "no device binary for table row %zu (%s): %s\n", i,
                   fl.parts[0].c_str(), path.c_str());
      ++missing;
    }
    ++checked;
  }
  CHECK_EQ(missing, size_t(0));
  CHECK_EQ(checked, size_t(8));
  std::printf("kernel_table_test OK (%zu table rows have a compiled binary)\n", checked);
  return 0;
}
