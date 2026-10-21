# The model description as data - no device, no I/O, no library dependency.
# Plan 3's capture code walks this table. The one header it borrows is
# `loader/small_layout.h`, a code-free set of block offsets shared with the
# loader and (plan 3) the kernels so the small-tensor table's destination
# offsets have exactly one definition; it adds no link dependency.
add_library(b70_model STATIC qwen35.cc model_desc.cc)
target_include_directories(b70_model PUBLIC ${CMAKE_SOURCE_DIR}/src)

# --- Spec 18b: K2-Horizon ------------------------------------------------------------
# K2-Horizon as data, BESIDE qwen3_5 (spec 18 §5.1): it shares GemvShape / WeightKind /
# Fuse from qwen35.h and nothing of the Qwen layer plan. Host only, like b70_model.
add_library(b70_k2_model STATIC k2_horizon.cc)
target_include_directories(b70_k2_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_k2_model PUBLIC b70_model)

# --- Spec 20c: Kolibri-1 -----------------------------------------------------------------
# Kolibri-1 as data, BESIDE qwen3_5 and K2 (spec 20 §4): its own table, sharing GemvShape /
# WeightKind / Fuse from qwen35.h. Host only.
add_library(b70_kolibri_model STATIC kolibri1.cc)
target_include_directories(b70_kolibri_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_kolibri_model PUBLIC b70_model)

# --- Spec 21b: Qwen3.8-Flash-Next (qwen4_exp) ----------------------------------------------------
# The family as data, BESIDE qwen3_5, K2 and Kolibri (spec 21 §6): its own table, sharing GemvShape /
# WeightKind / Fuse from qwen35.h. Host only.
add_library(b70_qwen4exp_model STATIC qwen4exp.cc)
target_include_directories(b70_qwen4exp_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_qwen4exp_model PUBLIC b70_model)

# --- Spec 23b: Muse-Glimmer-30B ------------------------------------------------------------------
# The text model as data, BESIDE qwen3_5, K2, Kolibri and qwen4_exp (spec 23 §5.1): its own table, sharing
# GemvShape / WeightKind / Fuse from qwen35.h. Host only.
add_library(b70_muse_model STATIC muse_glimmer.cc)
target_include_directories(b70_muse_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_muse_model PUBLIC b70_model)

# --- Spec 24b: LFM2.5-24B-A2B (lfm2_moe) ----------------------------------------------------------
# The model as data, BESIDE qwen3_5, K2, Kolibri, qwen4_exp and Muse-Glimmer (spec 24 §6): its own table,
# sharing GemvShape / WeightKind / Fuse from qwen35.h. Host only.
add_library(b70_lfm2moe_model STATIC lfm2moe.cc)
target_include_directories(b70_lfm2moe_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_lfm2moe_model PUBLIC b70_model)

# --- Spec 19b: the DFlash / DFlash 2 drafter (spec 19 §2) -----------------------------------------
# The drafter's config as data and its checks against the target it drafts for (model/dflash.h). Host only.
add_library(b70_dflash_model STATIC dflash.cc)
target_include_directories(b70_dflash_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_dflash_model PUBLIC b70_model)

# --- Spec 25b: Nemotron 3.5 Bolt 74B-A9B (nemotron_h_omni / nemotron_h_puzzle) --------------------------------------
# The text model as data, BESIDE qwen3_5, K2, Kolibri, qwen4_exp, Muse-Glimmer and LFM2.5 (spec 25 §8): its own table
# (per-block types, E and I per layer), sharing GemvShape / WeightKind / Fuse from qwen35.h. Host only.
add_library(b70_bolt_model STATIC bolt.cc)
target_include_directories(b70_bolt_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_bolt_model PUBLIC b70_model)
