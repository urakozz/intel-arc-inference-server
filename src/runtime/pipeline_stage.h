#pragma once
// PipelineEngine's one-device half (spec 16b), shared by the two files that define its
// members: runtime/pipeline_engine.cc (decode, b70_runtime) and runtime/prefill/
// pipeline_prefill.cc (spec 16c's prefill, b70_prefill_host). Private to them - nothing
// else includes this.
#include <optional>
#include <utility>

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
};

}  // namespace runtime
