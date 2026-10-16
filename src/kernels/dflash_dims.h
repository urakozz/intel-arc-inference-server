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

// Spec 23f: Muse-Glimmer's assistant (model::DFlashForm::Muse) runs the same lists at its block of 16 rows (the anchor
// + 15 drafts: spec 23 decision 10), df_attn's blocks of 64 ring offsets (64 q pairs x 128 offsets would overflow its
// local memory) and a ring of window + 1 positions rounded up to them: row 0 reads p - window .. p - 1 (|q - k| <=
// window, decision 13), so df_attn runs at WIN = window + 1 and RING >= WIN.
inline constexpr unsigned kMuseM = 16;
inline constexpr unsigned kMuseR = kMuseM - 1;
inline constexpr unsigned kMuseAttnPb = 64;
inline constexpr unsigned list_rows(bool muse) { return muse ? kMuseM : kM; }
inline constexpr unsigned attn_pb(bool muse) { return muse ? kMuseAttnPb : kAttnPb; }
inline constexpr unsigned muse_ring(unsigned window) { return (window + 1 + kMuseAttnPb - 1) / kMuseAttnPb * kMuseAttnPb; }

}  // namespace kernels::dflash
