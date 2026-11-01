// Fix round A: the arithmetic that cuts a large host <-> device copy into pieces of at
// most common::kMaxCopyChunk bytes (src/common/copy_chunks.h). Host only.
#include <cstdio>
#include <utility>
#include <vector>
#include "check.h"
#include "common/copy_chunks.h"

namespace {
using Piece = std::pair<size_t, size_t>;
std::vector<Piece> cut(size_t bytes, size_t chunk) {
  std::vector<Piece> v;
  common::for_each_copy_chunk(bytes, chunk, [&](size_t off, size_t n) { v.emplace_back(off, n); });
  return v;
}
// The pieces tile [0, bytes) in order, each non-empty and at most `chunk`, and their
// number is copy_chunk_count.
void check_tiles(size_t bytes, size_t chunk) {
  const auto v = cut(bytes, chunk);
  CHECK_EQ(v.size(), common::copy_chunk_count(bytes, chunk));
  size_t at = 0;
  for (const auto& p : v) {
    CHECK_EQ(p.first, at);
    CHECK(p.second > 0 && p.second <= chunk);
    at += p.second;
  }
  CHECK_EQ(at, bytes);
}
}  // namespace

int main() {
  const size_t C = common::kMaxCopyChunk;
  CHECK_EQ(C, size_t(256) << 20);

  // 0 bytes: no piece.
  CHECK_EQ(common::copy_chunk_count(0), size_t(0));
  CHECK(cut(0, C).empty());

  // Less than one chunk: one piece, the whole copy.
  CHECK((cut(1, C) == std::vector<Piece>{{0, 1}}));
  CHECK((cut(C - 1, C) == std::vector<Piece>{{0, C - 1}}));

  // Exact multiples: full pieces only.
  CHECK((cut(C, C) == std::vector<Piece>{{0, C}}));
  CHECK((cut(3 * C, C) == std::vector<Piece>{{0, C}, {C, C}, {2 * C, C}}));

  // A remainder: full pieces, then the rest.
  CHECK((cut(C + 1, C) == std::vector<Piece>{{0, C}, {C, 1}}));
  CHECK((cut(2 * C + 7, C) == std::vector<Piece>{{0, C}, {C, C}, {2 * C, 7}}));

  // The real tensors: Qwen3.8 lm_head / embed bf16 (248320 x 5120 x 2 = 2.54 GB) and the
  // int8 lm_head (1.27 GB).
  const size_t embed = size_t(248320) * 5120 * 2, head8 = size_t(248320) * 5120;
  CHECK_EQ(common::copy_chunk_count(embed), size_t(10));
  CHECK_EQ(common::copy_chunk_count(head8), size_t(5));
  check_tiles(embed, C);
  check_tiles(head8, C);

  // Small chunks, every length near the boundaries.
  for (size_t chunk : {size_t(1), size_t(3), size_t(8)})
    for (size_t b = 0; b <= 40; ++b) check_tiles(b, chunk);

  // A zero chunk is never used, but must not loop.
  CHECK_EQ(common::copy_chunk_count(5, 0), size_t(0));
  CHECK(cut(5, 0).empty());

  std::printf("copy_chunks_test: PASS\n");
  return 0;
}
