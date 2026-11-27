#pragma once
// Spec 19b: the DFlash kernels' fixed geometry as plain constants (no kernel paths - kernels/dflash_kernels.h has the
// binary names), so host-only code (the planner's sizes, the control block) can share them without kernels::kernel_dir().
namespace kernels::dflash {

inline constexpr unsigned kM = 8;          // the block's rows (the list's M): anchor + up to 7 drafts
inline constexpr unsigned kR = kM - 1;     // the head's rows (mask rows 1..7)
inline constexpr unsigned kTopK = 16, kQs = 16;
inline constexpr unsigned kWg = 256;       // df_prep's, df_topk's, df_select's work-group
inline constexpr unsigned kAttnPb = 128, kAttnWg = 128;   // df_attn_part: ring offsets per block, its work-group
// Plan 19v3b: the batched draft's blocks - one a sequence slot of a card (runtime::SlotDims::kMaxSeqs).
inline constexpr unsigned kSeqs = 4;
// Plan 40a Task 4: the batched draft's widest list - kSeqs blocks of kM rows, block z at rows kM z .. (the int4 drafter
// with the candidate head; the other forms keep one list of kM rows).
inline constexpr unsigned kSeqsRows = kSeqs * kM;
inline constexpr unsigned kTopkChunk = 4096;              // df_topk_stage1's logits per work-group

// Grid split-K for the int4 layout-1 block linears. Consumers fold these slices inside their existing launch.
inline constexpr unsigned int4_slices(unsigned K, unsigned N) {
  if ((K == 5120 && N == 6144) || (K == 4096 && N == 5120) || (K == 17408 && N == 5120)) return 4;
  if ((K == 5120 && (N == 34816 || N == 2048)) || (K == 25600 && N == 5120)) return 8;
  if (K == 5120 && (N == 1280 || N == 256)) return 16;
  // Spec 19 v3c: a TP card's half (16 q / 4 kv heads, the MLP's 8704) - the one-card linear's S (df_prep.cl's FOLD build folds it)
  if ((K == 5120 && N == 3072) || (K == 2048 && N == 5120) || (K == 8704 && N == 5120)) return 4;
  if (K == 5120 && (N == 17408 || N == 1024)) return 8;
  // Inkling's 2304-wide inputs have 36 groups, so 8 and 16 cannot split them.
  if ((K == 2304 && (N == 6144 || N == 13824 || N == 2048)) ||
      (N == 2304 && (K == 4096 || K == 6912))) return 4;
  if (K == 2304 && (N == 576 || N == 256)) return 12;
  if (K == 9216 && N == 2304) return 8;
  return 1;
}

// The DFlash control block's word indices (runtime/dflash/dflash_control.h holds the struct to them).
inline constexpr unsigned kDctlK = 0, kDctlProjPos = 1, kDctlProjN = 2, kDctlTemp = 3, kDctlSeedLo = 4,
                          kDctlSeedHi = 5;

}  // namespace kernels::dflash
