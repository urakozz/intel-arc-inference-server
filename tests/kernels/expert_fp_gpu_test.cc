#include <bit>
#include <cstdio>

#include "check.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "loader/expert_fp.h"

int main() {
  l0::Context ctx(0);
  auto cmd = l0::CmdList::immediate(ctx);
  using common::ExpertFpFormat;
  for (const auto format : {ExpertFpFormat::Nvfp4, ExpertFpFormat::Mxfp8}) {
    const unsigned bits = common::expert_fp_code_bits(format), group = common::expert_fp_group(format);
    const unsigned per_word = 32 / bits, tile_words = common::expert_fp_tile_words(format);
    constexpr unsigned tiles = 4;
    loader::ExpertFpMatrix m{format, 64, 16 * tiles, std::vector<uint32_t>(tiles * tile_words)};
    for (unsigned t = 0; t < tiles; ++t) {
      auto* tile = m.words.data() + t * tile_words;
      for (unsigned lane = 0; lane < 16; ++lane) {
        for (unsigned k = 0; k < 64; ++k) {
          unsigned code = (k + lane * 13 + t * 97) & ((1u << bits) - 1);
          if (bits == 8 && (code & 127) == 127) code -= 1;
          tile[k / per_word * 16 + lane] |= code << (k % per_word * bits);
        }
        for (unsigned b = 0; b < 64 / group; ++b) {
          const unsigned scale = (t * 64 + lane + b * 43) % (bits == 4 ? 127 : 255);
          tile[32 * bits + b * 4 + lane / 4] |= scale << (8 * (lane % 4));
        }
      }
    }
    std::vector<uint32_t> out(tiles * 64 * 16 * 2);
    l0::Mem input(ctx, l0::MemKind::Device, m.words.size() * 4);
    l0::Mem output(ctx, l0::MemKind::Device, out.size() * 4);
    cmd.copy(input.ptr(), m.words.data(), m.words.size() * 4);
    l0::Module mod(ctx, kernels::path(bits == 4 ? "expert_fp_bytes_NVFP4" : "expert_fp_bytes_MXFP8"));
    auto kernel = mod.kernel("expert_fp_bytes");
    kernel.group_size(16);
    kernel.arg_ptr(0, input.ptr());
    kernel.arg_ptr(1, output.ptr());
    cmd.launch(kernel, 64, tiles);
    cmd.copy(out.data(), output.ptr(), out.size() * 4);
    for (unsigned t = 0; t < tiles; ++t)
      for (unsigned k = 0; k < 64; ++k)
        for (unsigned lane = 0; lane < 16; ++lane) {
          const unsigned i = 2 * ((t * 64 + k) * 16 + lane);
          const unsigned n = t * 16 + lane;
          CHECK_EQ(out[i], std::bit_cast<uint32_t>(m.block_f32(n, k)));
          CHECK_EQ(out[i + 1], m.block_bf16(n, k));
        }
  }
  std::puts("GPU expert block decode matches FP32 and BF16 host bits");
}
