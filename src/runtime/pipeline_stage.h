#pragma once
// PipelineEngine's one-device half (spec 16b), shared by the two files that define its
// members: runtime/pipeline_engine.cc (decode, b70_runtime) and runtime/prefill/
// pipeline_prefill.cc (spec 16c's prefill, b70_prefill_host). Private to them - nothing
// else includes this.
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"
#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/pipeline_engine.h"
#include "runtime/pipeline_plan.h"

namespace runtime {

struct PipelineEngine::Stage {
  Stage(l0::Context& c, loader::LoadedModel m, uint32_t max_len, const PpStage& r, KvCache kv)
      : ctx(c),
        model(std::move(m)),
        range(r),
        persist(c, max_len, *model.desc, kv, r.gdn, r.fa),
        scratch(c, max_len, *model.desc),
        buffers(persist, scratch),
        queue(c),
        fence(queue),
        imm(l0::CmdList::immediate(c)),
        ctl(buffers.control.as<Control>()) {}
  l0::Context& ctx;
  loader::LoadedModel model;
  PpStage range;
  // Declaration order is construction order: `buffers` binds references into the two
  // above it (Engine's rule).
  PersistentBuffers persist;
  DecodeScratch scratch;
  DecodeBuffers buffers;
  l0::Queue queue;
  l0::Fence fence;
  mutable l0::CmdList imm;
  Control* ctl;
  std::optional<CapturedStep> step;
  // Spec 16d, MTP on only (the head on device 1). Device 1: the head's buffers (MtpBuffers:
  // its control, its own verify slots, KV layer, hh / dh, draft logits) and the draft lists;
  // device 0: its verify slots alone (pp_gdn_spec_bytes). Both: the verify lists at M = 1..4.
  std::unique_ptr<MtpBuffers> mtp;
  std::unique_ptr<l0::Mem> gdn_spec;
  std::vector<CapturedStep> verify, draft;
  // The verify slots this device's lists bind (MtpBuffers::gdn_spec or gdn_spec above).
  const l0::Mem* spec() const { return mtp ? &mtp->gdn_spec : gdn_spec.get(); }
};

}  // namespace runtime
