#pragma once
// Fix round A (2026-11-01): the size limit on one host <-> device copy.
//
// NEO 26.35 copies a plain (non-USM) host source of 512 MB or more by mapping it
// directly - no staging. On the box's fault-mode VM the kernel's userptr re-pin of
// such a range can hit the 1 s HMM page-in timeout (-EBUSY) under host memory
// pressure, which the GPU sees as a CAT error and an engine reset (card-1
// load_checkpoint_test DEVICE_LOST, 2026-10-31). Every copy larger than
// kMaxCopyChunk is therefore handed to the driver as consecutive pieces of at most
// kMaxCopyChunk bytes, in order, on the same list - the bytes that land are the
// same, only the number of driver calls changes (l0::CmdList::copy,
// prefill::Context::copy).
#include <algorithm>
#include <cstddef>

namespace common {

constexpr size_t kMaxCopyChunk = size_t(256) << 20;   // 256 MiB, half NEO's direct-map threshold

// How many pieces `bytes` is cut into with pieces of at most `chunk` bytes (0 for 0 bytes).
constexpr size_t copy_chunk_count(size_t bytes, size_t chunk = kMaxCopyChunk) {
  return chunk == 0 ? 0 : bytes / chunk + (bytes % chunk != 0);
}

// Calls f(offset, length) for each piece, in increasing offset order: every piece is
// `chunk` bytes but the last, which holds the remainder. No call for 0 bytes.
template <class F>
void for_each_copy_chunk(size_t bytes, size_t chunk, F&& f) {
  if (chunk == 0) return;
  for (size_t off = 0; off < bytes; off += chunk) f(off, std::min(chunk, bytes - off));
}

}  // namespace common
