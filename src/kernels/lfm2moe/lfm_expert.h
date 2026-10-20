// lfm_expert.h - spec 24c: THE expert address of LFM2.5-24B-A2B's MoE kernels (spec 15 decision 3: one macro, so
// a later layout - spec 22's table-addressed experts - changes one line). A layer's routed experts are ONE
// allocation each for gate||up and down (loader::LfmLayer::experts_gate_up / experts_down), expert e's layout-1
// block at e x the block's words (loader::lfm::expert_gate_up_offset / expert_down_offset: 3,342,336 B and
// 1,671,168 B at the real widths). Included by lfm_moe.cl after GU_BLK_U32 / DN_BLK_U32 are defined.
#ifndef LFM_EXPERT_H
#define LFM_EXPERT_H
#if !defined(GU_BLK_U32) || !defined(DN_BLK_U32)
#error "lfm_expert.h: define GU_BLK_U32 and DN_BLK_U32 (the blocks' u32 words) first"
#endif
#define LFM_EXPERT_GU(base, id) ((base) + (size_t)(id) * GU_BLK_U32)
#define LFM_EXPERT_DN(base, id) ((base) + (size_t)(id) * DN_BLK_U32)
#endif
