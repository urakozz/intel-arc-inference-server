#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/model_desc.h"
#include "runtime/pipeline_plan.h"

// Spec 16c (pipeline parallel PREFILL across two B70s): the chunk pipeline as device-free
// arithmetic and order - which chunks, what the host does in which order, where the hand-off
// rows land - beside runtime/pipeline_plan.h, so tests/runtime/pipeline_prefill_plan_test.cc
// and tests/runtime/pp_prefill_protocol_test.cc (the order run on host threads standing in
// for the two devices, also under ThreadSanitizer) prove it on any host. The device half is
// runtime/prefill/pipeline_prefill.cc; it runs exactly the steps below, through
// pp_prefill_run.
//
// **The pipeline (spec 16 §3.3).** Chunk j runs layers [0, s) on device 0 - the embedding,
// then layer s's fold (16b's cut) - and hands its folded residual rows and norm sums into
// landing slot j % 2 on device 1; device 1 waits for them (a cross-device event device 0
// signals after the rows), copies them in and runs layers [s, L) from layer s's
// norm-finish. Device 0 starts chunk j + 1 as soon as it has finished chunk j: the two cards
// work on consecutive chunks at once.
//
// **Every per-chunk resource is doubled, indexed by j % 2** (kPpPfDepth): the landing slot,
// each device's prefill Control block (the walk's kernels read pos / n_active from it while
// the host writes the next chunk's), device 0's ids buffer, the ready event, each device's
// done event and its block-end shadow (spec 7, below). A chunk's resources are reused by
// chunk j + 2, so:
//
// **The back-pressure rule.** The host appends chunk j to either device only after it has
// seen BOTH devices finish chunk j - 2. Device 0 is then never more than two chunks ahead
// of device 1, nothing of chunk j - 2 is overwritten while it is still read, and the host
// resets an event only once nothing waits on it or will signal it. Device 0 waits for
// device 1 only when both slots are taken (chunk j - 1's rows waiting, chunk j - 2's still
// being worked on) - the slower card is the pipeline's rate either way.
//
// **Spec 7 under the pipeline.** With a block hook, every chunk ends at a block end or at
// the prompt end (prefill_chunk_rows). The hook of a mid-prompt block end (chunk j) runs
// once both devices have finished chunk j - while chunk j + 1 may already be running on
// them. So a hooked chunk copies each device's GDN state and conv ring into that device's
// shadow j % 2 right after its own layers (in order on its list), and the hook's
// save_state reads the shadows; the KV rows [0, end) are final once chunk j is done (later
// chunks write rows >= end) and are read live. A hook that throws leaves the state at its
// end exactly as one card does: the lists drain and the shadows are copied back.
namespace runtime {

// Spec 7's block (Engine::kBlock, asserted equal where both are visible).
inline constexpr uint32_t kPpPfBlock = 2048;
// Chunk resources are indexed by chunk % kPpPfDepth (the doubling above).
inline constexpr uint32_t kPpPfDepth = 2;

// One prefill chunk: absolute positions [pos, pos + rows). `hook`: a mid-prompt block end
// with a block hook set - hook(pos + rows, true) runs after it, from the shadows. The
// prompt's end (the last chunk) is never `hook`: its hook runs after the head, live.
struct PpChunk {
  uint32_t pos = 0, rows = 0;
  bool hook = false;
  uint32_t end() const { return pos + rows; }
};
// Engine::prefill's chunks of `n` ids from `base` (runtime::prefill_chunk_rows - the same
// function): throws std::invalid_argument for n == 0 or chunk == 0.
std::vector<PpChunk> pp_prefill_chunks(uint32_t base, size_t n, uint32_t chunk, bool hooked,
                                       uint32_t block = kPpPfBlock);

// --- the host's order ---------------------------------------------------------------------
//
//   Run0 j   append device 0's chunk j (ids and Control j % 2, the walk, the hand-off out,
//            the ready and done events)
//   Run1 j   append device 1's chunk j (Control j % 2, the wait on ready j % 2, the hand-off
//            in, the walk, the done event)
//   Wait0 j / Wait1 j   wait (bounded) until that device has finished chunk j
//   Hook j   spec 7's hook at chunk j's end, from the shadows
//   Head     append the head on device 1 (final norm of the last row, lm_head, argmax)
//   Drain    wait (bounded) until both lists are idle
//
// For j in [0, n): [j >= 2: Wait0 j-2, Wait1 j-2, Hook j-2 if hooked], Run0 j, Run1 j; then
// [n >= 2: Wait0 n-2, Wait1 n-2, Hook n-2 if hooked], Head, Wait0 n-1, Wait1 n-1, Drain.
enum class PpPfOp : uint8_t { Run0, Run1, Wait0, Wait1, Hook, Head, Drain };
struct PpPfStep {
  PpPfOp op;
  uint32_t chunk;   // the chunk Run / Wait / Hook act on (Head: the last; Drain: n)
  bool operator==(const PpPfStep& o) const { return op == o.op && chunk == o.chunk; }
};
const char* pp_pf_op_name(PpPfOp op);
std::vector<PpPfStep> pp_prefill_schedule(const std::vector<PpChunk>& chunks);

// Every rule the order must keep, checked: "" when `steps` keeps them all, else the first
// one broken. Each chunk is run and waited once per device, in order, a device's wait after
// its run, device 1's run after device 0's; Run of chunk j only after both waits of j - 2
// (the back-pressure rule); a Hook exactly for each hooked chunk, after both its waits and
// before anything of chunk j + 2 and before the Head; the Head once, after device 1's last
// run; the Drain last. PipelineEngine::prefill asserts it on the order it runs, and the
// protocol test shows a mutated order is caught.
std::string pp_prefill_check(const std::vector<PpPfStep>& steps, const std::vector<PpChunk>& chunks);

// Runs `steps` over a driver - the ONE executor the engine (runtime/prefill/pipeline_prefill.cc)
// and the host-thread protocol test share:
//   drv.run(dev, chunk, j)   Run0 / Run1
//   drv.wait(dev, j)         Wait0 / Wait1; throws when its bound passes
//   drv.hook(chunk, j)       Hook
//   drv.head(chunk, j)       Head (the last chunk)
//   drv.drain()              Drain; throws when its bound passes
template <class Driver>
void pp_prefill_run(const std::vector<PpChunk>& chunks, const std::vector<PpPfStep>& steps,
                    Driver& drv) {
  for (const PpPfStep& s : steps) {
    switch (s.op) {
      case PpPfOp::Run0: drv.run(0u, chunks.at(s.chunk), s.chunk); break;
      case PpPfOp::Run1: drv.run(1u, chunks.at(s.chunk), s.chunk); break;
      case PpPfOp::Wait0: drv.wait(0u, s.chunk); break;
      case PpPfOp::Wait1: drv.wait(1u, s.chunk); break;
      case PpPfOp::Hook: drv.hook(chunks.at(s.chunk), s.chunk); break;
      case PpPfOp::Head: drv.head(chunks.at(s.chunk), s.chunk); break;
      case PpPfOp::Drain: drv.drain(); break;
    }
  }
}

// --- the hand-off buffers ------------------------------------------------------------------
//
// Device 1's prefill landing allocation: kPpPfDepth slots, each the rows of one chunk
// (kC x hidden bf16 - a chunk moves rows x hidden), then on its own page the norm sums
// (PrefillScratch::norm_sumsq, whole: [kNormGroups][kC] fp32) and the peer path's stamp
// word, then the peer flag on a page of its own; each slot a multiple of kPpLandingAlign.
// Only device 0 writes it (the copy engine or pp_send), device 1 only reads it - spec 16
// §2's rules for the peer-written buffer, as 16b's decode landing keeps them.
struct PpPfLinkLayout {
  size_t resid_bytes = 0;   // kC x hidden bf16
  size_t sumsq_bytes = 0;   // PrefillScratch::norm_sumsq's size
  size_t sumsq_off = 0;     // in a slot: page-aligned, after the rows
  size_t stamp_off = 0;     // the word right after the sums
  size_t flag_off = 0;      // a page of its own
  size_t slot_bytes = 0;    // one slot, a multiple of kPpLandingAlign
  size_t total = 0;         // kPpPfDepth slots
  size_t slot(uint32_t chunk) const { return size_t(chunk % kPpPfDepth) * slot_bytes; }
};
PpPfLinkLayout pp_prefill_link_layout(const model::ModelDesc& d);
// The prefill pipeline's own device bytes beyond the prefill scratch, per device: device 0
// its two Control blocks, the chunk timestamps and pp_send's per-slot counters; device 1 the
// landing slots, its two Control blocks, the timestamps and pp_recv's per-slot state words.
size_t pp_prefill_link_bytes(const model::ModelDesc& d, uint32_t device);
// One device's block-end shadows (allocated only with a block hook): kPpPfDepth copies of
// its GDN state and conv ring.
size_t pp_prefill_shadow_bytes(const model::ModelDesc& d, const PpStage& st);

}  // namespace runtime
