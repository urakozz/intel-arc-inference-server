#pragma once
// Spec 19b: the DFlash kernels' fixed geometry as plain constants (no kernel paths - kernels/dflash_kernels.h has the
// binary names), so host-only code (the planner's sizes, the control block) can share them without B70_KERNEL_DIR.
namespace kernels::dflash {

inline constexpr unsigned kM = 8;          // the block's rows (the list's M): anchor + up to 7 drafts
inline constexpr unsigned kR = kM - 1;     // the head's rows (mask rows 1..7)
inline constexpr unsigned kTopK = 16, kQs = 16;
inline constexpr unsigned kWg = 256;       // df_prep's, df_topk's, df_select's work-group
inline constexpr unsigned kAttnPb = 128, kAttnWg = 128;   // df_attn_part: ring offsets per block, its work-group
inline constexpr unsigned kTopkChunk = 4096;              // df_topk_stage1's logits per work-group

// The DFlash control block's word indices (runtime/dflash/dflash_control.h holds the struct to them).
inline constexpr unsigned kDctlK = 0, kDctlProjPos = 1, kDctlProjN = 2, kDctlTemp = 3, kDctlSeedLo = 4,
                          kDctlSeedHi = 5;

}  // namespace kernels::dflash
