#pragma once
// P-D's torch-free shim (docs/probe-prefill-vllm-parity-2026-09-14.md addendum
// §A3.4, Phase 1 §2.1). `chunk_gated_delta_rule_kernels_xe2.hpp` (the vendor
// header, read-only, never modified) `#include <torch/all.h>` unconditionally
// at its top, and its OUTER wrapper `chunk_gated_delta_rule_impl_xe2` (a plain,
// non-template function) is fully typechecked at parse time whether or not
// anything calls it. This probe never calls that wrapper -- it calls
// `gdn::kernel_launcher<T, StateT>` directly, the torch-free template Phase 1
// found confined to lines 1-1504 of the header -- so this header only needs
// to make the WRAPPER's signature and body TYPECHECK, never actually run.
//
// This is therefore not a rewrite of vendor code and not a real ATen: it is
// the smallest set of names `torch::`/`at::`/`TORCH_CHECK` the wrapper
// mentions (enumerated by grep, docs/probe-prefill-vllm-parity-2026-09-14.md
// Phase 1 §2.1), each backed by a stand-in with just enough behaviour to
// compile. `torch::zeros` is never invoked by this probe (the probe allocates
// its own USM scratch) and throws if it ever is, so a silent behavioural gap
// cannot hide a wrong number.
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace c10 {
enum class ScalarType { Float, BFloat16, Half, Byte };
inline std::ostream& operator<<(std::ostream& os, ScalarType t) {
  switch (t) {
    case ScalarType::Float: return os << "float32";
    case ScalarType::BFloat16: return os << "bfloat16";
    case ScalarType::Half: return os << "float16";
    default: return os << "byte";
  }
}
}  // namespace c10

namespace at {
using ScalarType = c10::ScalarType;
inline constexpr ScalarType kFloat = ScalarType::Float;
inline constexpr ScalarType kBFloat16 = ScalarType::BFloat16;
inline constexpr ScalarType kHalf = ScalarType::Half;
inline constexpr ScalarType kByte = ScalarType::Byte;
}  // namespace at

namespace torch {
using at::kBFloat16;
using at::kByte;
using at::kFloat;
using at::kHalf;
using ScalarType = at::ScalarType;

struct Device {};

struct TensorOptions {
  ScalarType dtype_ = at::kFloat;
  Device device_{};
  TensorOptions dtype(ScalarType t) const {
    TensorOptions o = *this;
    o.dtype_ = t;
    return o;
  }
  TensorOptions device(Device d) const {
    TensorOptions o = *this;
    o.device_ = d;
    return o;
  }
  TensorOptions requires_grad(bool) const { return *this; }
};
inline TensorOptions dtype(ScalarType t) { return TensorOptions{}.dtype(t); }
// The vendor wrapper also calls `torch::dtype(dtype)` where `dtype` is
// already a TensorOptions (`auto dtype = core_attn_out.dtype();` in real
// ATen returns a TypeMeta, and `torch::dtype(TypeMeta)` is a distinct real
// overload from `torch::dtype(ScalarType)`; this stub's `Tensor::dtype()`
// returns a TensorOptions directly, so the matching overload here just
// passes it through).
inline TensorOptions dtype(const TensorOptions& o) { return o; }

// A view, not an owning tensor: `ptr_` is USM the probe already owns. Every
// method here is exactly what the vendor wrapper's SIGNATURE calls -- see the
// grep in the file header comment above.
class Tensor {
 public:
  Tensor() = default;
  Tensor(void* ptr, std::vector<int64_t> sizes, std::vector<int64_t> strides, ScalarType st,
        Device dev = {})
      : ptr_(ptr), sizes_(std::move(sizes)), strides_(std::move(strides)), st_(st), dev_(dev) {}
  int64_t size(int64_t d) const { return sizes_.at(size_t(d)); }
  int64_t stride(int64_t d) const { return strides_.at(size_t(d)); }
  int64_t dim() const { return int64_t(sizes_.size()); }
  ScalarType scalar_type() const { return st_; }
  TensorOptions dtype() const { return TensorOptions{}.dtype(st_).device(dev_); }
  Device device() const { return dev_; }
  void* data_ptr() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
  std::vector<int64_t> sizes_, strides_;
  ScalarType st_ = at::kFloat;
  Device dev_{};
};

// Never called by this probe (kernel_launcher is invoked directly with USM
// the probe allocates itself); present only so chunk_gated_delta_rule_impl_xe2
// -- unused, but not a template, so fully typechecked -- compiles.
inline Tensor zeros(std::vector<int64_t>, TensorOptions) {
  throw std::runtime_error(
      "torch_stub::zeros: not implemented -- P-D calls kernel_launcher directly "
      "and never executes chunk_gated_delta_rule_impl_xe2");
}

template <class... Args>
inline void check_impl(bool cond, const char* file, int line, Args&&... args) {
  if (cond) return;
  std::ostringstream oss;
  oss << file << ":" << line << ": ";
  (oss << ... << args);
  throw std::runtime_error(oss.str());
}
}  // namespace torch

#define TORCH_CHECK(cond, ...) ::torch::check_impl((cond), __FILE__, __LINE__, ##__VA_ARGS__)
