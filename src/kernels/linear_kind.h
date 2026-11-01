#pragma once
// linear_kind.h - spec 28b's rule for which linear kernel a list at M rows binds (no kernel-directory
// dependency, so runtime headers can carry it). Documented in runtime/gemv_kind.h, which re-exports it.
namespace kernels {

enum class LinearKind : unsigned char { Gemv, SmallM };
// Spec 28 §2: Gemv (gemv / gemv_i8w, today's binaries) at M = 1, SmallM (gemm_sm) at M >= 2.
constexpr LinearKind gemv_kind(unsigned M) { return M >= 2 ? LinearKind::SmallM : LinearKind::Gemv; }

}  // namespace kernels
