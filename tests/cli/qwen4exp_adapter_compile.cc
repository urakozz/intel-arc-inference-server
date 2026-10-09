// Spec 21e Task 4: spec 16d's cli::pp::PipelineEngineAdapterT instantiated over runtime::qwen4exp::Qwen4ExpEngine - every
// member (prefill, the greedy and sampled steps, step_many's draft / verify / accept / deferred commit, step_drafts's
// lookup path, truncate_to, the prefix cache's snapshot forwards, the block hook, ingest) compiled against the engine's
// host calls: the interface the adapter's header names is the engine's, exactly (no third copy of the speculative loop).
// Compiled by the Mac's Level Zero syntax check (tools/mac/l0_syntax.sh) and as an object library on the box (linked
// for real by b70-serve, which instantiates the same template).
#include "cli/pipeline_serve_adapter.h"
#include "runtime/qwen4exp/qwen4exp_engine.h"

template struct cli::pp::PipelineEngineAdapterT<runtime::qwen4exp::Qwen4ExpEngine>;

static_assert(runtime::qwen4exp::Qwen4ExpEngine::kMaxDraft == 3, "the adapter's pinned rows assume K <= 3");
static_assert(runtime::qwen4exp::Qwen4ExpEngine::kBlock == 2048, "spec 7's block");
