# The model description as data - no device, no I/O, no library dependency.
# Plan 3's capture code walks this table. The one header it borrows is
# `loader/small_layout.h`, a code-free set of block offsets shared with the
# loader and (plan 3) the kernels so the small-tensor table's destination
# offsets have exactly one definition; it adds no link dependency.
add_library(b70_model STATIC qwen35.cc model_desc.cc)
target_include_directories(b70_model PUBLIC ${CMAKE_SOURCE_DIR}/src)

# --- Spec 20c: Kolibri-1 -----------------------------------------------------------------
# Kolibri-1 as data, BESIDE qwen3_5 (spec 20 §4): its own table, sharing GemvShape /
# WeightKind / Fuse from qwen35.h. Host only.
add_library(b70_kolibri_model STATIC kolibri1.cc)
target_include_directories(b70_kolibri_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_kolibri_model PUBLIC b70_model)

# --- Spec 21-QFNb: Qwen3.8-Flash-Next (qwen4_exp) ----------------------------------------------------
# The family as data, BESIDE qwen3_5 and Kolibri (spec 21-QFN §6): its own table, sharing GemvShape /
# WeightKind / Fuse from qwen35.h. Host only.
add_library(b70_qwen4exp_model STATIC qwen4exp.cc)
target_include_directories(b70_qwen4exp_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_qwen4exp_model PUBLIC b70_model)

# --- Spec 19b: the DFlash / DFlash 2 drafter (spec 19 §2) -----------------------------------------
# The drafter's config as data and its checks against the target it drafts for (model/dflash.h). Host only.
add_library(b70_dflash_model STATIC dflash.cc)
target_include_directories(b70_dflash_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_dflash_model PUBLIC b70_model)

# Static reflection (P2996, common/json_reflect.h): Kolibri and Flash-Next hold config.json to their descriptors.
foreach(_t b70_kolibri_model b70_qwen4exp_model)
  target_compile_options(${_t} PRIVATE -freflection)
endforeach()
