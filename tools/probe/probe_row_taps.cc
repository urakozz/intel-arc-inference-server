// Spec 28 (the special-token logit gap): the engine's PLAIN decode row at one position of a forced context, with the
// per-layer residual taps, for a layer bisection against a CPU reference (tools/oracle/row_taps_ref.py).
//   probe_row_taps <snapshot> <cases> [--lm-head bf16|int8] [--prefill-backend l0|l0-int8] [--max-len N]
// Cases file, one case a line: "<label> <prompt ids file> <force ids file> <row> <prefill len> <out stem>".
// The context is prompt + force[0, row): the probe prefills its first <prefill len> ids (the test's own prompt, so
// the rest is decoded as the test's plain stream decodes it), ingests the others one at a time, and writes from the
// last ingest (the row that decides force[row]):
//   <out stem>.logits.f32  f32 [kVocab]        the logits row
//   <out stem>.taps.bf16   bf16 [64][5120]     runtime/capture.h's debug_resid: tap[L] = resid after layer L's mixer
//                                              (layer L's MLP output not yet folded in)
//   <out stem>.resid.bf16  bf16 [5120]         the final hidden state before the final norm
//   <out stem>.x.bf16      bf16 [5120]         the final-normalised hidden (the lm_head's input)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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
  if (!in) throw std::runtime_error("probe_row_taps: cannot read " + path);
  Ids v;
  uint32_t x = 0;
  while (in >> x) v.push_back(x);
  return v;
}
void write(const std::string& path, const void* p, size_t n) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f || std::fwrite(p, 1, n, f) != n) throw std::runtime_error("probe_row_taps: cannot write " + path);
  std::fclose(f);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: probe_row_taps <snapshot> <cases> [--lm-head bf16|int8] [--prefill-backend l0|l0-int8] "
                 "[--max-len N]\n");
    return 2;
  }
  uint32_t max_len = 16384;
  loader::LmHeadForm head = loader::LmHeadForm::Checkpoint;
  runtime::PrefillBackend backend = runtime::PrefillBackend::L0;
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--max-len") max_len = uint32_t(std::stoul(argv[i + 1]));
    else if (a == "--lm-head") {
      if (!loader::parse_lm_head_form(argv[i + 1], head)) return 2;
    } else if (a == "--prefill-backend") {
      if (!runtime::parse_prefill_backend(argv[i + 1], backend)) return 2;
    } else return 2;
  }
  using model::Qwen35;
  l0::Context ctx(0);
  runtime::Engine eng(ctx, loader::load(ctx, loader::resolve_snapshot(argv[1]), max_len, false, head), max_len,
                      /*debug_resid=*/true);
  eng.set_prefill_backend(backend);
  eng.prepare_prefill();
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const uint32_t H = eng.model().desc->hidden;
  std::vector<float> row(Qwen35::kVocab);
  std::vector<uint16_t> resid(H), x(H);
  std::ifstream in(argv[2]);
  for (std::string line; std::getline(in, line);) {
    std::istringstream s(line);
    std::string label, pf, ff, stem;
    size_t r = 0, P = 0;
    if (!(s >> label >> pf >> ff >> r >> P >> stem)) continue;
    Ids ids = read_ids(pf);
    const Ids force = read_ids(ff);
    if (r > force.size()) throw std::runtime_error(label + ": row past the forced ids");
    ids.insert(ids.end(), force.begin(), force.begin() + r);
    if (P == 0 || P >= ids.size()) throw std::runtime_error(label + ": the prefill must leave at least one decode");
    eng.reset();
    eng.prefill(Ids(ids.begin(), ids.begin() + P));
    for (size_t i = P; i < ids.size(); ++i) eng.ingest({ids[i]});
    imm.copy(row.data(), eng.buffers().logits.ptr(), row.size() * 4);
    imm.copy(resid.data(), eng.buffers().resid.ptr(), size_t(H) * 2);
    imm.copy(x.data(), eng.buffers().x.ptr(), size_t(H) * 2);
    const std::vector<uint16_t> taps = eng.read_debug_resid();
    write(stem + ".logits.f32", row.data(), row.size() * 4);
    write(stem + ".taps.bf16", taps.data(), taps.size() * 2);
    write(stem + ".resid.bf16", resid.data(), resid.size() * 2);
    write(stem + ".x.bf16", x.data(), x.size() * 2);
    uint32_t t1 = 0;
    for (uint32_t i = 1; i < Qwen35::kVocabUsed; ++i)
      if (row[i] > row[t1]) t1 = i;
    std::printf("%s: %zu ids (%zu prefilled), row %zu: top %u (%.5f), forced %u (%.5f)\n", label.c_str(), ids.size(), P,
                r, t1, row[t1], r < force.size() ? force[r] : 0u, r < force.size() ? row[force[r]] : 0.f);
    std::fflush(stdout);
  }
  return 0;
}
