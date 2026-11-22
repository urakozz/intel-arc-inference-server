// The prompt pass's kernels need no scratch (spill or private memory). The prompt pass runs on its own immediate
// list (prefill::Context), and the driver keeps the scratch that list first got: once a decode list with a larger
// scratch (int8 KV's attn_decode_v2_kv8, DFlash's df_attn_part) has run on the card, a prompt-pass kernel that spills
// reads and writes freed scratch - DEVICE_LOST on a served int8 DFlash prompt (pf_gdn_scan_dpas_split at 128 GRF,
// 512 B a thread). Every entry of every pf_* binary named on the command line (the walk's, B70_PREFILL_KERNELS) must
// report no scratch.
//
// usage: prefill_scratch_test <binary>...   (kernels::path names; others than pf_* are skipped)
#include <cstdio>
#include <string>
#include <vector>

#include <level_zero/ze_api.h>

#include "kernels/kernels.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/module.h"

int main(int argc, char** argv) {
  l0::Context ctx(0);
  size_t bins = 0, entries = 0, bad = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string v = argv[i];
    if (!v.starts_with("pf_")) continue;
    ++bins;
    l0::Module m(ctx, kernels::path(v));
    uint32_t n = 0;
    zeModuleGetKernelNames(m.handle(), &n, nullptr);
    std::vector<const char*> names(n);
    zeModuleGetKernelNames(m.handle(), &n, names.data());
    for (const char* name : names) {
      l0::Kernel k(m, name);
      ze_kernel_properties_t kp{};
      kp.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
      zeKernelGetProperties(k.handle(), &kp);
      ++entries;
      if (kp.spillMemSize == 0 && kp.privateMemSize == 0) continue;
      ++bad;
      std::printf("FAIL %s/%s: spill %u B, private %u B a thread\n", v.c_str(), name, kp.spillMemSize,
                  kp.privateMemSize);
    }
  }
  std::printf("%zu binaries, %zu entries, %zu with scratch\n", bins, entries, bad);
  const bool ok = bins > 0 && bad == 0;
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
