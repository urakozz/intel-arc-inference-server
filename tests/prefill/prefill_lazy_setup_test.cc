// 3g only: record cold/warm short prefill and later fallback for bytewise before/after comparison.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "check.h"
#include "l0/cmdlist.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"

int main(int argc, char** argv) {
  CHECK(argc == 4 || argc == 5);
  const bool baseline = argc == 5 && std::strcmp(argv[4], "--baseline") == 0;
  CHECK(argc == 4 || baseline);
  CHECK(std::strcmp(argv[2], "bf16") == 0 || std::strcmp(argv[2], "int8") == 0);
  const bool kv8 = std::strcmp(argv[2], "int8") == 0;
  constexpr uint32_t capacity = 4096;
  constexpr size_t slab_bytes = 35651584;
  l0::Context ctx(0);
  runtime::Engine eng(ctx, loader::load(ctx, argv[1], capacity, false, loader::LmHeadForm::Int8),
                      capacity, false, kv8 ? runtime::KvCache::Int8 : runtime::KvCache::Bf16);
  eng.set_prefill_backend(runtime::PrefillBackend::L0);
  eng.prepare_prefill();
  const size_t raw_bytes = kv8 ? size_t(16777216) : 0;  // K/V, 4096 positions, 4 heads x 256 bf16.
  CHECK_EQ(eng.prefill_scratch()->lazy_bytes(), raw_bytes);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  std::ofstream dump(argv[3], std::ios::binary);
  CHECK(dump.good());
  const uint32_t lengths[] = {237, 237, 300, 321, 385, 237};
  size_t previous_lazy = 0;
  for (size_t i = 0; i < 6; ++i) {
    const uint32_t M = lengths[i];
    std::vector<uint32_t> ids(M);
    for (uint32_t j = 0; j < M; ++j) ids[j] = 1000 + j;
    eng.reset();
    const auto t0 = std::chrono::steady_clock::now();
    eng.prefill(ids);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const auto* pf = eng.prefill_scratch();
    const size_t lazy = pf->lazy_bytes();
    CHECK_EQ(lazy, raw_bytes + ((baseline || i >= 3) ? slab_bytes : 0));
    if (i == 3) CHECK_EQ(lazy - previous_lazy, baseline ? size_t(0) : slab_bytes);
    if (i > 3) CHECK_EQ(lazy, previous_lazy);
    previous_lazy = lazy;
    CHECK_EQ(eng.pos(), M);
    std::vector<float> logits(model::Qwen35::kVocab);
    imm.copy(logits.data(), pf->logits.ptr(), logits.size() * sizeof(float));
    for (float x : logits) CHECK(std::isfinite(x));
    const uint32_t header[] = {M, eng.control()->cur_token[0]};
    dump.write(reinterpret_cast<const char*>(header), sizeof header);
    dump.write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * sizeof(float)));
    CHECK(dump.good());
    std::printf("3g case=%zu kv=%s rows=%u wall_ms=%.6f lazy_bytes=%zu prefill_bytes=%zu id=%u\n",
                i, argv[2], M, ms, lazy, eng.memory_use().prefill_scratch, header[1]);
  }
  dump.close();
  CHECK(dump.good());
  std::puts("prefill_lazy_setup_test PASS");
}
