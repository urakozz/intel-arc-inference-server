#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "l0/cmdlist.h"
#include "loader/qwen4exp_ple.h"

// The Level Zero half of the PLE table (loader/qwen4exp_ple.h): pinning the int8 file into 32 host-USM ranges
// (q and scales per head), the tag-and-readback alias check, the sampled readback of the data, and the device
// pointer table. Box only (the Mac links no device); tests/loader/qwen4exp_load_checkpoint_test.cc runs it.
namespace loader {
namespace {

// Rows of a head compared with the file after the copy: the first, the last and every 4096th.
size_t compare_rows(const Q4PleHost& h, uint32_t head, const int8_t* q, const uint8_t* s) {
  const uint64_t n = h.rows(head);
  const size_t sb = q4_ple_scale_bytes(h.scale());
  const auto* fs = static_cast<const uint8_t*>(h.s(head));
  std::vector<uint64_t> rows;
  for (uint64_t r = 0; r < n; r += 4096) rows.push_back(r);
  if (n > 0 && rows.back() != n - 1) rows.push_back(n - 1);
  for (uint64_t r : rows)
    if (std::memcmp(q + r * h.dim(), h.q(head) + r * h.dim(), h.dim()) != 0 ||
        std::memcmp(s + r * sb, fs + r * sb, sb) != 0)
      throw std::runtime_error("load_q4_ple: head " + std::to_string(head) + " row " + std::to_string(r) +
                               " reads back different from the file after the copy into host USM");
  const size_t checked = rows.size();
  return checked;
}

}  // namespace

Q4PleTable load_q4_ple(l0::Context& ctx, const model::Qwen4ExpDesc& d, const std::string& ple_dir) {
  const auto t0 = std::chrono::steady_clock::now();
  const Q4PleHost h(ple_dir);
  check_q4_ple(d, h);
  Q4PleTable t;
  t.scale = h.scale();
  t.bytes = h.bytes();
  t.mem_before = q4_mem_available();
  check_q4_ple_host_fit(t.bytes, t.mem_before);

  // 1. Every range, then a tag at the start of every 2 MiB page of every range, then every tag read back:
  //    an allocation that aliased another's pages shows the later range's tag (spec 22 P0.5's check).
  const uint32_t H = h.heads();
  for (uint32_t i = 0; i < H; ++i) {
    t.q.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Host, h.q_bytes(i), kQ4PlePage));
    t.s.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Host, h.s_bytes(i), kQ4PlePage));
  }
  const auto range = [&](uint32_t r) -> l0::Mem& { return r < H ? *t.q[r] : *t.s[r - H]; };
  for (uint32_t r = 0; r < 2 * H; ++r) {
    l0::Mem& m = range(r);
    for (uint64_t p = 0; p * kQ4PlePage + 8 <= m.size(); ++p) {
      const uint64_t tag = q4_ple_tag(r, p);
      std::memcpy(static_cast<uint8_t*>(m.ptr()) + p * kQ4PlePage, &tag, 8);
    }
  }
  for (uint32_t r = 0; r < 2 * H; ++r) {
    l0::Mem& m = range(r);
    for (uint64_t p = 0; p * kQ4PlePage + 8 <= m.size(); ++p) {
      uint64_t got = 0;
      std::memcpy(&got, static_cast<const uint8_t*>(m.ptr()) + p * kQ4PlePage, 8);
      if (got != q4_ple_tag(r, p))
        throw std::runtime_error("load_q4_ple: host range " + std::to_string(r) + " page " + std::to_string(p) +
                                 " reads back another tag - two pinned allocations alias (spec 22 P0.5)");
      t.tag_checksum += got;
    }
  }
  // 2. The data, head by head; sampled rows compared with the file; each page's first word recorded.
  for (uint32_t i = 0; i < H; ++i) {
    std::memcpy(t.q[i]->ptr(), h.q(i), h.q_bytes(i));
    std::memcpy(t.s[i]->ptr(), h.s(i), h.s_bytes(i));
    t.rows_checked += compare_rows(h, i, t.q[i]->as<int8_t>(), t.s[i]->as<uint8_t>());
  }
  for (uint32_t r = 0; r < 2 * H; ++r) {
    l0::Mem& m = range(r);
    for (uint64_t p = 0; p * kQ4PlePage + 8 <= m.size(); ++p) {
      uint64_t w = 0;
      std::memcpy(&w, static_cast<const uint8_t*>(m.ptr()) + p * kQ4PlePage, 8);
      t.page_words.push_back(w);
    }
  }
  // 3. The device pointer table: q[0..H), s[0..H) as u64 - what q4_ple_gather indexes by head.
  std::vector<uint64_t> ptrs(2 * H);
  for (uint32_t r = 0; r < 2 * H; ++r) ptrs[r] = reinterpret_cast<uint64_t>(range(r).ptr());
  t.ptrs = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, ptrs.size() * 8);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(t.ptrs->ptr(), ptrs.data(), ptrs.size() * 8);
  t.mem_after = q4_mem_available();
  t.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return t;
}

}  // namespace loader
