# Spec 19b: the DFlash drafter on the card.
#
# b70_dflash_plan: the buffers' sizes, the planner's term (runtime::DrafterPlan) and the RoPE table as device-free
# arithmetic (dflash_sizes.cc) - host only, so tests/runtime/dflash_plan_test.cc runs on the Mac.
# Spec 26b: + the LiLiCorr drafter's (lilicorr_sizes.cc: its buffers, launch counts, ring snapshot, key and Bolt's reserve
# terms - runtime::bolt::BoltReserve, hence b70_bolt_plan).
add_library(b70_dflash_plan STATIC dflash_sizes.cc lilicorr_sizes.cc)
target_include_directories(b70_dflash_plan PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_dflash_plan PUBLIC b70_dflash_model b70_plan b70_bolt_plan)

# b70_dflash_runtime: DFlashBuffers and the captured draft / projection lists (Level Zero).
# Spec 19c: + DFlashDrafter (dflash_drafter.cc), the drafter behind runtime::TapDrafter in the Engine's loop.
# Spec 23f: + MuseDrafter (muse_drafter.cc), Muse-Glimmer's assistant behind runtime::TapSink beside MuseEngine.
# Spec 26b: + LiliDrafter (lilicorr_drafter.cc), the LiLiCorr drafter behind runtime::TapSink beside Bolt (26c binds it).
add_library(b70_dflash_runtime STATIC dflash_buffers.cc dflash_capture.cc dflash_drafter.cc muse_drafter.cc
            lilicorr_drafter.cc)
target_include_directories(b70_dflash_runtime PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_dflash_runtime PUBLIC b70_l0 b70_dflash_loader b70_dflash_plan b70_runtime)
b70_target_kernel_dir(b70_dflash_runtime)
