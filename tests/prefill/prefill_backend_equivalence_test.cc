// Spec 2.1 §2 bar 2: prefilling on the L0 backend leaves the SAME state as prefilling on
// sycl-tla -- kv_k, kv_v, gdn_state, conv_ring, the last logits row and the control block,
// bitwise, with +0.0 and -0.0 counted as equal (spec §3.4: P·V's 256-padding can flip the sign
// of an exact zero) and their count printed. Tokens are a consequence (cur_token is in the
// control block), so this test IS the token gate across backends.
//
// Spec 6 (plan 6b): the L0 walk's default attention is pf_flash_attn, a different algorithm
// sycl-tla cannot run; this bar is about the GEMM backends, so it pins the composed attention
// on both sides (set_attn_mode_for_test), which is B70_PREFILL_ATTN=composed exactly.
//
// Order matters for S3's assertion: every L0 session runs BEFORE any sycl-tla session, so a
// fresh engine's SYCL side is provably never built by the L0 backend.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/prefill/attn.h"
#include "runtime/prefill_backend.h"

namespace {
using golden::read_ids;
using model::Qwen35;
using runtime::PrefillBackend;
constexpr uint32_t kMaxLen = 16384;
const char* const kPrompts[] = {"prose", "code", "cjk"};

struct Snap {
  std::vector<uint16_t> kv_k, kv_v, conv_ring;   // bf16
  std::vector<float> gdn_state, logits;           // fp32
  std::vector<uint8_t> control;
};
template <class T>
std::vector<T> grab(l0::CmdList& imm, const l0::Mem& m) {
  std::vector<T> v(m.size() / sizeof(T));
  imm.copy(v.data(), m.ptr(), v.size() * sizeof(T));
  return v;
}
Snap snapshot(runtime::Engine& eng, l0::CmdList& imm) {
  Snap s;
  s.kv_k = grab<uint16_t>(imm, eng.buffers().kv_k);
  s.kv_v = grab<uint16_t>(imm, eng.buffers().kv_v);
  s.conv_ring = grab<uint16_t>(imm, eng.buffers().conv_ring);
  s.gdn_state = grab<float>(imm, eng.buffers().gdn_state);
  s.control = grab<uint8_t>(imm, eng.buffers().control);
  const runtime::PrefillScratch* pf = eng.prefill_scratch();
  CHECK(pf != nullptr);
  s.logits = grab<float>(imm, pf->logits);
  return s;
}

struct Tally { size_t words = 0, sign_zero = 0; size_t first = 0; };
bool zero16(uint16_t v) { return (v & 0x7FFFu) == 0; }
bool zero32(float v) { uint32_t u; std::memcpy(&u, &v, 4); return (u & 0x7FFFFFFFu) == 0; }
Tally cmp16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  Tally t;
  CHECK_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] == b[i]) continue;
    if (zero16(a[i]) && zero16(b[i])) { ++t.sign_zero; continue; }
    if (t.words++ == 0) t.first = i;
  }
  return t;
}
Tally cmp32(const std::vector<float>& a, const std::vector<float>& b) {
  Tally t;
  CHECK_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], 4) == 0) continue;
    if (zero32(a[i]) && zero32(b[i])) { ++t.sign_zero; continue; }
    if (t.words++ == 0) t.first = i;
  }
  return t;
}
void verdict(const char* buf, const Tally& t) {
  std::printf("    %-10s %s  (%zu words differ, %zu sign-of-zero)\n", buf,
              t.words == 0 ? "equal" : "**DIFFERS**", t.words, t.sign_zero);
  if (t.words) std::fprintf(stderr, "    first difference at element %zu\n", t.first);
  CHECK_EQ(t.words, size_t(0));
}
void compare(const Snap& l0, const Snap& tla) {
  verdict("kv_k", cmp16(l0.kv_k, tla.kv_k));
  verdict("kv_v", cmp16(l0.kv_v, tla.kv_v));
  verdict("conv_ring", cmp16(l0.conv_ring, tla.conv_ring));
  verdict("gdn_state", cmp32(l0.gdn_state, tla.gdn_state));
  verdict("logits", cmp32(l0.logits, tla.logits));
  CHECK(l0.control == tla.control);   // pos, n_active, cur_token, out_token: exact bytes
  std::puts("    control    equal");
}

// A deterministic 4096-id prompt: two full 2048-row chunks, the bench shape, no ties to any
// golden set. Ordinary ids only (2 .. kVocabUsed), never a special token.
std::vector<uint32_t> synthetic(uint32_t n) {
  std::vector<uint32_t> ids(n);
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (uint32_t i = 0; i < n; ++i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    ids[i] = 2 + uint32_t((s >> 33) % (Qwen35::kVocabUsed - 2));
  }
  return ids;
}

struct Case { std::string name; std::vector<uint32_t> ids; uint32_t chunk; };

Snap session(runtime::Engine& eng, l0::CmdList& imm, PrefillBackend b, const Case& c) {
  eng.reset();
  eng.set_prefill_backend(b);
  eng.prefill(c.ids, c.chunk);
  CHECK_EQ(eng.pos(), uint32_t(c.ids.size()));
  return snapshot(eng, imm);
}

// Bar 2's third family: the prose prompt in 16-id slices, alternating backend slice by slice
// (sycl-tla, l0, sycl-tla, ...), against the same slicing on sycl-tla alone. Both use one
// prefill() call per slice, so the only variable is the backend.
Snap sliced(runtime::Engine& eng, l0::CmdList& imm, const std::vector<uint32_t>& ids,
            bool alternate) {
  eng.reset();
  for (size_t off = 0, k = 0; off < ids.size(); off += 16, ++k) {
    eng.set_prefill_backend(alternate && (k % 2 == 1) ? PrefillBackend::L0
                                                       : PrefillBackend::SyclTla);
    const size_t n = std::min<size_t>(16, ids.size() - off);
    eng.prefill(std::vector<uint32_t>(ids.begin() + off, ids.begin() + off + n), 16);
  }
  CHECK_EQ(eng.pos(), uint32_t(ids.size()));
  return snapshot(eng, imm);
}
}  // namespace

int main(int argc, char** argv) {
  const std::string pdir = argc > 1 ? argv[1] : "tests/golden/prompts";
  const std::string snap = argc > 2 ? argv[2] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  runtime::prefill::set_attn_mode_for_test(runtime::prefill::AttnMode::Composed);
  l0::Context ctx(0);
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen);
  runtime::Engine eng(ctx, std::move(model), kMaxLen);
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  std::vector<Case> cases;
  for (const char* p : kPrompts) cases.push_back({p, read_ids(pdir + "/" + p + ".ids"), 0});
  cases.push_back({"synthetic-4096", synthetic(4096), 0});

  // Every L0 session first (S3 asserts the SYCL side is still absent here), then sycl-tla.
  std::vector<Snap> l0;
  for (const Case& c : cases) l0.push_back(session(eng, imm, PrefillBackend::L0, c));
  // S3: the L0 backend never built the SYCL side. This must be checked BEFORE the first
  // sycl-tla session, which builds it for the rest of the process.
  CHECK(!eng.prefill_sycl_side_created());
  std::puts("l0 sessions done: no SYCL side was created (spec 2.1 §3.5)");
  for (size_t i = 0; i < cases.size(); ++i) {
    std::printf("\n== %s: %zu ids at chunk %u, l0 vs sycl-tla\n", cases[i].name.c_str(),
                cases[i].ids.size(), cases[i].chunk ? cases[i].chunk : runtime::PrefillScratch::kC);
    const Snap tla = session(eng, imm, PrefillBackend::SyclTla, cases[i]);
    compare(l0[i], tla);
  }
  std::printf("\n== mixed: prose in 16-id slices, alternating backends, vs sycl-tla alone\n");
  const Snap mixed = sliced(eng, imm, cases[0].ids, /*alternate=*/true);
  const Snap plain = sliced(eng, imm, cases[0].ids, /*alternate=*/false);
  compare(mixed, plain);

  std::printf("\nprefill_backend_equivalence_test OK: %zu cases, l0 == sycl-tla (sign of zero excepted)\n",
              cases.size() + 1);
  return 0;
}
