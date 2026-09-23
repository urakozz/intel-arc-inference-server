// prefill_determinism_test - spec 2 §6.4: the same prefill, twice from a
// reset, **bitwise**.
//
// The claim is narrow and it is the one that makes every other prefill number
// reproducible: no kernel on this path uses a floating-point atomic, the
// sycl-tla GEMM is instantiated without split-K, and the walk's launch order is
// fixed by an in-order L0 immediate list plus an in-order SYCL queue. So two
// prefills of the same ids from the same zeroed state must leave the five
// persistent buffers and `cur_token` byte for byte the same.
//
// Three widths (kC, 1024, 16), the same result required at EACH width. Not
// ACROSS widths: `gdn_chunk_test` case 4 covers that for the GDN family, and
// the attention's `npad` rounding legitimately differs between chunkings.
//
// Then, deliberately, the **run-C property** (replay_determinism_test's, and
// for its reason): a third prefill after a `reset()` but with `PrefillScratch`
// NOT re-zeroed must give the same bytes. That is the standing proof that no
// prefill kernel reads scratch it has not first written. The scratch is
// therefore never zeroed here - zeroing it would mask exactly the regression
// this case exists to catch.
//
// argv: [1] prompt dir, [2] snapshot,
//       [3] prefill backend, sycl-tla or l0 (default: the build's) -- spec 2.1
//       §2 bar 3 is this gate run on each backend (plan 9e Task 1).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
#include "runtime/prefill/backend.h"
#include "runtime/prefill/gdn.h"
#include "runtime/prefill_backend.h"

namespace {
using golden::read_ids;

constexpr uint32_t kMaxLen = 16384;
const char* const kPrompts[] = {"prose", "code", "cjk"};
const uint32_t kChunks[] = {0 /* = kC */, 1024, 16};

// replay_determinism_test's, copied with its diff-reporting body: the count and
// the first offset are what turn "not identical" into something to look at.
void same_bytes(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, const char* what) {
  CHECK_EQ(a.size(), b.size());
  size_t diff = 0, first = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) {
      if (diff == 0) first = i;
      ++diff;
    }
  if (diff != 0)
    std::fprintf(stderr, "%s: %zu of %zu bytes differ, first at byte %zu\n", what, diff, a.size(),
                 first);
  CHECK_EQ(diff, size_t(0));
}

struct State {
  std::vector<uint8_t> gdn_state, conv_ring, kv_k, kv_v, control;
};

State snapshot(runtime::Engine& eng, l0::CmdList& imm) {
  State s;
  auto grab = [&](std::vector<uint8_t>& dst, const l0::Mem& m) {
    dst.resize(m.size());
    imm.copy(dst.data(), m.ptr(), m.size());
  };
  grab(s.gdn_state, eng.buffers().gdn_state);
  grab(s.conv_ring, eng.buffers().conv_ring);
  grab(s.kv_k, eng.buffers().kv_k);
  grab(s.kv_v, eng.buffers().kv_v);
  grab(s.control, eng.buffers().control);
  return s;
}

void compare(const State& a, const State& b, const char* what) {
  same_bytes(a.gdn_state, b.gdn_state, (std::string(what) + " gdn_state").c_str());
  same_bytes(a.conv_ring, b.conv_ring, (std::string(what) + " conv_ring").c_str());
  same_bytes(a.kv_k, b.kv_k, (std::string(what) + " kv_k").c_str());
  same_bytes(a.kv_v, b.kv_v, (std::string(what) + " kv_v").c_str());
  same_bytes(a.control, b.control, (std::string(what) + " control").c_str());
}

}  // namespace

int main(int argc, char** argv) {
  const std::string pdir = argc > 1 ? argv[1] : "tests/golden/prompts";
  const std::string snap = argc > 2 ? argv[2] : "Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ";
  // argv[3]: the prefill backend (spec 2.1) -- sycl-tla or l0; default = the build's.
  runtime::PrefillBackend backend = runtime::prefill::default_prefill_backend();
  if (argc > 3) CHECK(runtime::parse_prefill_backend(argv[3], backend));

  l0::Context ctx(0);
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen);
  runtime::Engine eng(ctx, std::move(model), kMaxLen);
  eng.set_prefill_backend(backend);
  std::printf("prefill backend: %s\n", runtime::prefill_backend_name(backend));
  // The scan selector's DISPATCH PROOF (parity-program design §11). A gate run
  // under `B70_PREFILL_GDN_SCAN` is not evidence unless it says which kernel it
  // launched: the 2026-09-21 "green" split record was executing `pf_gdn_scan`.
  {
    const char* const sel = std::getenv("B70_PREFILL_GDN_SCAN");
    std::printf("gdn scan selector: B70_PREFILL_GDN_SCAN=%s -> entry %s\n",
                sel && *sel ? sel : "(unset)", runtime::prefill::gdn_scan_entry_name());
  }
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  uint32_t cases = 0;
  for (const char* pname : kPrompts) {
    const std::vector<uint32_t> ids = read_ids(pdir + "/" + pname + ".ids");
    for (uint32_t chunk : kChunks) {
      const uint32_t C = chunk ? chunk : runtime::PrefillScratch::kC;
      eng.reset();
      eng.prefill(ids, chunk);
      const State a = snapshot(eng, imm);
      const uint32_t tok_a = eng.buffers().control.as<runtime::Control>()->cur_token[0];

      eng.reset();
      eng.prefill(ids, chunk);
      const State b = snapshot(eng, imm);
      const uint32_t tok_b = eng.buffers().control.as<runtime::Control>()->cur_token[0];

      const std::string what = std::string(pname) + " C=" + std::to_string(C) + " run B";
      compare(a, b, what.c_str());
      CHECK_EQ(tok_a, tok_b);

      // Run C: no re-zeroing of PrefillScratch between B and C, on purpose.
      eng.reset();
      eng.prefill(ids, chunk);
      const State c = snapshot(eng, imm);
      const uint32_t tok_c = eng.buffers().control.as<runtime::Control>()->cur_token[0];
      compare(a, c, (std::string(pname) + " C=" + std::to_string(C) + " run C").c_str());
      CHECK_EQ(tok_a, tok_c);

      std::printf("  %-6s C=%-5u %zu ids: gdn_state (%zu B), conv_ring (%zu B), kv_k/kv_v"
                  " (%zu B each) and control bitwise identical over 3 runs; cur_token %u\n",
                  pname, C, ids.size(), a.gdn_state.size(), a.conv_ring.size(), a.kv_k.size(),
                  tok_a);
      ++cases;
    }
  }
  // The other half of the dispatch proof: the entry a scan launch was really
  // built with, read back after the walk rather than before it.
  CHECK(runtime::prefill::gdn_scan_launched_entry() != nullptr);
  CHECK_EQ(std::string(runtime::prefill::gdn_scan_launched_entry()),
           std::string(runtime::prefill::gdn_scan_entry_name()));
  std::printf("gdn scan entry LAUNCHED: %s\n", runtime::prefill::gdn_scan_launched_entry());
  std::printf("prefill_determinism_test OK: %u (prompt, chunk) case(s) x 3 runs from reset,"
              " bitwise identical persistent state and first generated id; PrefillScratch"
              " deliberately never re-zeroed (the run-C property)\n", cases);
  return 0;
}
