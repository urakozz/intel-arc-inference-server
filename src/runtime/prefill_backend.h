#pragma once
#include <string>

// Which backend Engine::prefill() runs a chunk on (spec 2.1 §3.2 / §3.5). Header-only and
// dependency-free so runtime/engine.h (the decode library) can name it without linking the
// prefill host library.
namespace runtime {

enum class PrefillBackend {
  SyclTla,   // spec 2's path: dequant -> host wait -> sycl-tla GEMM -> host wait
  L0,        // spec 2.1: every GEMM on the Level Zero list, no SYCL, no host waits
};

inline const char* prefill_backend_name(PrefillBackend b) {
  return b == PrefillBackend::SyclTla ? "sycl-tla" : "l0";
}

// The two spellings the CLIs accept. Returns false on anything else.
inline bool parse_prefill_backend(const std::string& s, PrefillBackend& out) {
  if (s == "sycl-tla") { out = PrefillBackend::SyclTla; return true; }
  if (s == "l0") { out = PrefillBackend::L0; return true; }
  return false;
}

}  // namespace runtime
