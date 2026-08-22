// Round-trips 1 MiB host -> device -> host through the immediate list.
#include <cstdint>
#include <cstring>
#include <vector>
#include "check.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"

int main() {
  l0::Context ctx(0);
  std::printf("device: %s, EUs: %u\n", ctx.name().c_str(), ctx.eu_count());
  CHECK(ctx.eu_count() > 0);

  const size_t n = 1u << 20;
  std::vector<uint8_t> src(n), dst(n, 0);
  for (size_t i = 0; i < n; ++i) src[i] = static_cast<uint8_t>(i * 7 + 3);

  l0::Mem dev(ctx, l0::MemKind::Device, n);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(dev.ptr(), src.data(), n);
  imm.copy(dst.data(), dev.ptr(), n);
  CHECK_EQ(std::memcmp(src.data(), dst.data(), n), 0);

  l0::Mem shared(ctx, l0::MemKind::Shared, 64);
  imm.fill(shared.ptr(), 0xA5A5A5A5u, 64);
  CHECK_EQ(shared.as<uint32_t>()[0], 0xA5A5A5A5u);
  CHECK_EQ(shared.as<uint32_t>()[15], 0xA5A5A5A5u);
  std::puts("copy_test OK");
  return 0;
}
