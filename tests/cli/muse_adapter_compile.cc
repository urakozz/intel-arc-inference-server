// Spec 23e Task 5: cli::muse::MuseEngineAdapterT instantiated over runtime::muse::MuseEngine - every member (prefill,
// the greedy and sampled steps with the model's cap, the prefix cache's snapshot forwards, the block hook, ingest)
// compiled against the engine's host calls: the interface the host test's fake has (tests/runtime/muse_snapshot_test.cc)
// is the engine's, exactly. Compiled by the Mac's Level Zero syntax check (tools/mac/l0_syntax.sh) and as an object
// library on the box (linked for real by b70-serve, which instantiates the same template).
#include "cli/muse_serve.h"
#include "runtime/muse/muse_engine.h"

template struct cli::muse::MuseEngineAdapterT<runtime::muse::MuseEngine>;

static_assert(runtime::muse::MuseEngine::kBlock == 2048, "spec 7's block = the prefill chunk");
