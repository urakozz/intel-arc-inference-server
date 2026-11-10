// Open-work 3o: the prompt's state after prefill, saved and swapped between KV forms, to tell where l0-int8 prefill
// x int8 KV leaves the other configurations. One decision row of a forced context (probe_row_taps' numbering).
//   probe_kv_state <snapshot> <prompt ids> <force ids> <row> <stem> [--prefill-backend l0|l0-int8] [--lm-head bf16|int8]
//                  [--max-len N] [--restore STEM2 [--from bf16|int8]]
// Without --restore: prefill the prompt (P ids), write <stem>.kv (Engine::save_kv(0, P), the engine's own form) and
// <stem>.state (save_state), then ingest force[0, row) and write <stem>.logits.f32 (the row deciding force[row]).
// With --restore: load STEM2.kv / STEM2.state in place of the prefill, then the same ingests. --from names STEM2's KV
// form when it is not the engine's (B70_KV_CACHE): bf16 -> int8 is the writer's chain on the host
// (common::kv8::encode of each bf16 row), int8 -> bf16 dequantises, un-rotates and rounds to bf16 (the value an
// int8 reader sees, in the bf16 cache); the converted KV is also written to <stem>.kv.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "common/kv8.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/lm_head_int8.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "model/qwen35.h"
#include "runtime/engine.h"
#include "runtime/prefill_backend.h"

namespace {
using Ids = std::vector<uint32_t>;
Ids read_ids(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("probe_kv_state: cannot read " + path);
  Ids v;
  uint32_t x = 0;
  while (in >> x) v.push_back(x);
  return v;
}
std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("probe_kv_state: cannot read " + path);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}
void write(const std::string& path, const void* p, size_t n) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f || std::fwrite(p, 1, n, f) != n) throw std::runtime_error("probe_kv_state: cannot write " + path);
  std::fclose(f);
}
float bf16f(uint16_t b) {
  const uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t rne_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
constexpr uint32_t kHeads = 4, kDim = 256;

// bf16 [L][n][4][256] K then V  ->  int8 rows [L][n][4][256] then fp16 scales [L][n][4], K then V.
std::vector<uint8_t> to_int8(const std::vector<uint8_t>& src, uint32_t L, uint32_t n) {
  const size_t rows = size_t(L) * n * kHeads;
  if (src.size() != 2 * rows * kDim * 2) throw std::runtime_error("probe_kv_state: the bf16 KV file's size");
  std::vector<uint8_t> out(2 * rows * (kDim + 2));
  for (int kv = 0; kv < 2; ++kv) {
    const auto* s = reinterpret_cast<const uint16_t*>(src.data()) + kv * rows * kDim;
    auto* q = reinterpret_cast<int8_t*>(out.data() + kv * rows * (kDim + 2));
    auto* sc = reinterpret_cast<uint16_t*>(out.data() + kv * rows * (kDim + 2) + rows * kDim);
    float x[kDim];
    for (size_t r = 0; r < rows; ++r) {
      for (uint32_t i = 0; i < kDim; ++i) x[i] = bf16f(s[r * kDim + i]);
      sc[r] = common::kv8::encode(x, q + r * kDim);
    }
  }
  return out;
}
std::vector<uint8_t> to_bf16(const std::vector<uint8_t>& src, uint32_t L, uint32_t n) {
  const size_t rows = size_t(L) * n * kHeads;
  if (src.size() != 2 * rows * (kDim + 2)) throw std::runtime_error("probe_kv_state: the int8 KV file's size");
  std::vector<uint8_t> out(2 * rows * kDim * 2);
  for (int kv = 0; kv < 2; ++kv) {
    const auto* q = reinterpret_cast<const int8_t*>(src.data() + kv * rows * (kDim + 2));
    const auto* sc = reinterpret_cast<const uint16_t*>(src.data() + kv * rows * (kDim + 2) + rows * kDim);
    auto* d = reinterpret_cast<uint16_t*>(out.data()) + kv * rows * kDim;
    float y[kDim], x[kDim];
    for (size_t r = 0; r < rows; ++r) {
      for (uint32_t i = 0; i < kDim; ++i) y[i] = common::kv8::dequant(q[r * kDim + i], sc[r]);
      common::kv8::unrotate(y, x);
      for (uint32_t i = 0; i < kDim; ++i) d[r * kDim + i] = rne_bf16(x[i]);
    }
  }
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: probe_kv_state <snapshot> <prompt ids> <force ids> <row> <stem> [--prefill-backend B] "
                 "[--lm-head F] [--max-len N] [--restore STEM2 [--from bf16|int8]]\n");
    return 2;
  }
  uint32_t max_len = 16384;
  loader::LmHeadForm head = loader::LmHeadForm::Checkpoint;
  runtime::PrefillBackend backend = runtime::PrefillBackend::L0;
  std::string restore, from;
  for (int i = 6; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--max-len") max_len = uint32_t(std::stoul(argv[i + 1]));
    else if (a == "--lm-head") {
      if (!loader::parse_lm_head_form(argv[i + 1], head)) return 2;
    } else if (a == "--prefill-backend") {
      if (!runtime::parse_prefill_backend(argv[i + 1], backend)) return 2;
    } else if (a == "--restore") restore = argv[i + 1];
    else if (a == "--from") from = argv[i + 1];
    else return 2;
  }
  using model::Qwen35;
  const Ids prompt = read_ids(argv[2]), force = read_ids(argv[3]);
  const size_t row = std::stoul(argv[4]);
  const std::string stem = argv[5];
  if (row > force.size()) throw std::runtime_error("probe_kv_state: row past the forced ids");
  l0::Context ctx(0);
  runtime::Engine eng(ctx, loader::load(ctx, loader::resolve_snapshot(argv[1]), max_len, false, head), max_len);
  eng.set_prefill_backend(backend);
  eng.prepare_prefill();
  const bool int8 = eng.kv_cache() == runtime::KvCache::Int8;
  const uint32_t P = uint32_t(prompt.size()), L = eng.model().desc->fa_layers;
  eng.reset();
  if (restore.empty()) {
    eng.prefill(prompt);
    std::vector<uint8_t> kv(eng.kv_bytes(P)), st(eng.state_bytes());
    eng.save_kv(0, P, kv.data());
    eng.save_state(st.data());
    write(stem + ".kv", kv.data(), kv.size());
    write(stem + ".state", st.data(), st.size());
  } else {
    std::vector<uint8_t> kv = read_file(restore + ".kv");
    const std::vector<uint8_t> st = read_file(restore + ".state");
    if (from == "bf16" && int8) kv = to_int8(kv, L, P);
    else if (from == "int8" && !int8) kv = to_bf16(kv, L, P);
    else if (!from.empty()) throw std::runtime_error("probe_kv_state: --from " + from + " is the engine's own form");
    if (kv.size() != eng.kv_bytes(P) || st.size() != eng.state_bytes())
      throw std::runtime_error("probe_kv_state: " + restore + " does not fit this engine's KV / state");
    if (!from.empty()) write(stem + ".kv", kv.data(), kv.size());
    eng.load_kv(0, P, kv.data());
    eng.load_state(st.data(), P);
  }
  if (row > 0) eng.ingest(Ids(force.begin(), force.begin() + row));
  std::vector<float> lg(Qwen35::kVocab);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(lg.data(), eng.buffers().logits.ptr(), lg.size() * 4);
  write(stem + ".logits.f32", lg.data(), lg.size() * 4);
  const uint32_t want = row < force.size() ? force[row] : 0u;
  uint32_t t1 = want == 0 ? 1 : 0;   // the best id other than the forced one
  for (uint32_t i = 0; i < Qwen35::kVocabUsed; ++i)
    if (i != want && lg[i] > lg[t1]) t1 = i;
  const std::string how = restore.empty() ? "prefilled" : "restored " + restore + (from.empty() ? "" : " from " + from);
  std::printf("%s: kv %s, %s, row %zu: forced %u (%.5f), best other %u (%.5f), forced - other %+.5f\n",
              stem.c_str(), int8 ? "int8" : "bf16", how.c_str(), row, want, lg[want], t1, lg[t1], lg[want] - lg[t1]);
  return 0;
}
