#pragma once
#include <string>

// Which backend Engine::prefill() runs a chunk on (spec 2.1 §3.2 / §3.5). Header-only and
// dependency-free so runtime/engine.h (the decode library) can name it without linking the
// prefill host library.
namespace runtime {

enum class PrefillBackend {
  SyclTla,   // spec 2's path: dequant -> host wait -> sycl-tla GEMM -> host wait
  L0,        // spec 2.1: every GEMM on the Level Zero list, no SYCL, no host waits
  L0Int8,    // spec 5: L0, with every int4 linear on the h8 int8 path
};

// L0Int8 is L0 for everything but the int4 linears: attention, GDN, norms, the
// head, replay and the launch accounting of everything else. Every site that
// asks "is this the L0 walk" asks this, never `== PrefillBackend::L0`.
inline bool is_l0(PrefillBackend b) { return b == PrefillBackend::L0 || b == PrefillBackend::L0Int8; }

inline const char* prefill_backend_name(PrefillBackend b) {
  return b == PrefillBackend::SyclTla ? "sycl-tla" : b == PrefillBackend::L0 ? "l0" : "l0-int8";
}

// The three spellings the CLIs accept. Returns false on anything else.
inline bool parse_prefill_backend(const std::string& s, PrefillBackend& out) {
  if (s == "sycl-tla") { out = PrefillBackend::SyclTla; return true; }
  if (s == "l0") { out = PrefillBackend::L0; return true; }
  if (s == "l0-int8") { out = PrefillBackend::L0Int8; return true; }
  return false;
}

}  // namespace runtime
