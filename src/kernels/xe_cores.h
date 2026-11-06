#pragma once
// The Xe-core count the kernels are built for - the B70's 32 (docs/01-hardware.md): the grids
// assume it (kernels.h's gemv_bf16 tiny-N rule), and the hardware gate (cli/hardware_check.h)
// refuses a card that shows another. Its own header so host-only code (the gate's test) can
// read it without kernels.h's B70_KERNEL_DIR.
namespace kernels {
inline constexpr unsigned kBuiltXeCores = 32;
}  // namespace kernels
