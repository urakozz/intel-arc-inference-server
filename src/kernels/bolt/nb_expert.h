// nb_expert.h - spec 25c: THE expert address of Nemotron 3.5 Bolt's MoE kernels (spec 15 decision 3: one macro, so a
// later layout - spec 22's table-addressed experts - changes one line). A MoE block's routed experts are ONE allocation
// (loader::BoltLayer::experts) of E blocks, expert e's block at e x block(I) words: its up (K 1024, N I, layout 1) then its
// down (K I, N 1024, layout 1), I x 544 B each (loader/bolt_layout.h expert_up_offset / expert_dn_offset). Included by
// nb_moe.cl after MOE_I is defined.
#ifndef NB_EXPERT_H
#define NB_EXPERT_H
#ifndef MOE_I
#error "nb_expert.h: define MOE_I (the block's routed intermediate) first"
#endif
#define NB_UP_U32 ((size_t)(MOE_I / 16) * (1024 / 64) * 136)   /* = I x 544 B / 4 */
#define NB_BLK_U32 (2 * NB_UP_U32)
#define NB_EXPERT_UP(base, id) ((base) + (size_t)(id) * NB_BLK_U32)
#define NB_EXPERT_DN(base, id) (NB_EXPERT_UP(base, id) + NB_UP_U32)
#endif
