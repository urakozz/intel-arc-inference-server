// A cold session clears the drafter's state without rebuilding its immutable RoPE table.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include "check.h"
#include "common/json.h"
#include "kernels/dflash_ref.h"
#include "runtime/dflash/dflash_buffers.h"

int main(int argc, char** argv) {
  std::ifstream f(argc > 1 ? argv[1] : "tests/model/dflash/qwen38_dflash2_config.json");
  std::stringstream text;
  text << f.rdbuf();
  const auto d = model::dflash_desc_from_config(common::json::parse(text.str()));
  constexpr uint32_t max_len = 131072;
  l0::Context ctx(0);
  auto imm = l0::CmdList::immediate(ctx);
  runtime::dflash::DFlashBuffers b(ctx, d, 5120, 248320, max_len, loader::DraftFormat::Int8, true);
  auto expected = dfref::rope_table(d.theta, d.head_dim, max_len + kernels::dflash::kM);
  std::vector<float> rope(expected.size());
  imm.copy(rope.data(), b.rope.ptr(), b.rope.size());
  CHECK(std::memcmp(rope.data(), expected.data(), b.rope.size()) == 0);

  // Guard both ends of the immutable allocation: zero() must not write any of it.
  expected.front() = 123.0f;
  expected.back() = -456.0f;
  imm.copy(b.rope.ptr(), expected.data(), b.rope.size());
  imm.fill(b.ring_k.ptr(), 0xFFFFFFFFu, b.ring_k.size());
  imm.fill(b.ring_v.ptr(), 0xFFFFFFFFu, b.ring_v.size());
  std::memset(b.dctl.ptr(), 0xFF, b.dctl.size());
  std::memset(b.out.ptr(), 0xFF, b.out.size());
  const auto start = std::chrono::steady_clock::now();
  b.zero(imm);
  std::printf("DFlash reset at %u rows: %.3f ms\n", max_len,
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
  imm.copy(rope.data(), b.rope.ptr(), b.rope.size());
  CHECK(std::memcmp(rope.data(), expected.data(), b.rope.size()) == 0);
  std::vector<uint32_t> ring(b.ring_k.size() / sizeof(uint32_t));
  for (auto* mem : {&b.ring_k, &b.ring_v}) {
    imm.copy(ring.data(), mem->ptr(), mem->size());
    CHECK(std::all_of(ring.begin(), ring.end(), [](uint32_t v) { return v == 0; }));
  }
  for (auto* mem : {&b.dctl, &b.out}) {
    const auto* p = mem->as<uint8_t>();
    CHECK(std::all_of(p, p + mem->size(), [](uint8_t v) { return v == 0; }));
  }
  std::puts("dflash_buffers_test: PASS");
}
