#pragma once
#include "runtime/prefill_backend.h"

namespace runtime::prefill {

// What Engine::prefill runs when nothing was set: l0 for every model (operator, 2026-10-27);
// l0-int8 is opt-in (--prefill-backend l0-int8).
inline PrefillBackend default_prefill_backend() { return PrefillBackend::L0; }

}  // namespace runtime::prefill
