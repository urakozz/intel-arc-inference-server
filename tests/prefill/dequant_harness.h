#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/int4.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

inline std::string dequant_variant(uint32_t K, uint32_t N, uint32_t layout,
                                   uint32_t transposed) {
  return "pf_dequant_tile_K" + std::to_string(K) + "_N" + std::to_string(N) +
         "_L" + std::to_string(layout) + "_T" + std::to_string(transposed);
}

// Upload one canonical int4 tensor in either loader layout, launch one
// subgroup per (N tile, K group), then read the whole bf16 output back.
inline std::vector<uint16_t> run_dequant(l0::Context& ctx, l0::Queue& q,
                                         l0::Fence& fence,
                                         const common::Int4Gptq& weights,
                                         uint32_t K, uint32_t N,
                                         uint32_t layout,
                                         uint32_t transposed) {
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const std::vector<uint32_t> tiled = layout ? weights.tiled() : std::vector<uint32_t>{};
  const uint32_t* source = layout ? tiled.data() : weights.qweight.data();
  const size_t source_bytes = layout ? tiled.size() * sizeof(uint32_t)
                                     : weights.qweight.size() * sizeof(uint32_t);
  l0::Mem source_buf(ctx, l0::MemKind::Device, source_bytes);
  imm.copy(source_buf.ptr(), source, source_bytes);

  l0::Mem scales_buf(ctx, l0::MemKind::Device, weights.scales.size() * sizeof(uint16_t));
  if (!layout)
    imm.copy(scales_buf.ptr(), weights.scales.data(), weights.scales.size() * sizeof(uint16_t));

  std::vector<uint16_t> got(size_t(K) * N);
  l0::Mem out_buf(ctx, l0::MemKind::Device, got.size() * sizeof(uint16_t));
  l0::Module mod(ctx, kernels::path(dequant_variant(K, N, layout, transposed)));
  l0::Kernel kernel = mod.kernel("pf_dequant_tile");
  kernel.group_size(16);
  kernel.arg_ptr(0, source_buf.ptr());
  kernel.arg_ptr(1, layout ? nullptr : scales_buf.ptr());
  kernel.arg_ptr(2, out_buf.ptr());

  l0::CmdList list = l0::CmdList::regular(ctx);
  list.launch(kernel, N / 16, K / 64);
  list.close();
  q.execute(list, &fence);
  fence.wait();
  imm.copy(got.data(), out_buf.ptr(), got.size() * sizeof(uint16_t));
  return got;
}
