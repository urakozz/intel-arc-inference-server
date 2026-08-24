add_ocloc_kernel(noop SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/noop.cl)
add_ocloc_kernel(bw_sum SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/bw_sum.cl)
add_ocloc_kernel(ctrl_read SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ctrl_read.cl)

set(GEMV_CL ${CMAKE_CURRENT_SOURCE_DIR}/gemv.cl)
function(add_gemv_variant M K N S L)
  add_ocloc_kernel(gemv_M${M}_K${K}_N${N}_S${S}_L${L} SOURCE ${GEMV_CL}
                   DEFINES M=${M} K=${K} N=${N} S=${S} LAYOUT=${L})
endfunction()
# Probe matrix (spec 1 §4.2): the five int4 shapes the 27B runs after load-time
# fusion, both layouts, S in {1,2,4,8,16}. All S divide K/64 (96, 80 and 272).
foreach(L 0 1)
  foreach(S 1 2 4 8 16)
    add_gemv_variant(1 6144 5120  ${S} ${L})
    add_gemv_variant(1 5120 14336 ${S} ${L})
    add_gemv_variant(1 5120 16384 ${S} ${L})
    add_gemv_variant(1 5120 34816 ${S} ${L})
    add_gemv_variant(1 17408 5120 ${S} ${L})
  endforeach()
endforeach()
add_gemv_variant(2 6144 5120 1 0)   # spec 1 §9: the M loop must compile at M=2

set(GEMV_BF16_CL ${CMAKE_CURRENT_SOURCE_DIR}/gemv_bf16.cl)
function(add_gemv_bf16_variant M K N)
  add_ocloc_kernel(gemv_bf16_M${M}_K${K}_N${N} SOURCE ${GEMV_BF16_CL} DEFINES M=${M} K=${K} N=${N})
endfunction()
add_gemv_bf16_variant(1 5120 248320)   # lm_head
add_gemv_bf16_variant(1 5120 128)      # in_proj_a || in_proj_b, zero-padded 96 -> 128

# The between-GEMV kernels: three entry points in one file (prep.cl), so each
# binary carries all three and the runtime loads the one it asked for. Only
# prep_res_norm has a shape to vary; the other two bake the model's dimensions.
#
# `-cl-fp32-correctly-rounded-divide-sqrt` is not decoration: prep's references
# are compared **bit-exactly**, and by default OpenCL allows 2.5 ulp on `/` and
# `sqrt`. Measured without it, the device's `1/sqrt(mean+eps)` sat 1-2 ulp under
# the host's and one x_out element in 5120 landed on the wrong side of a
# round-to-nearest tie (docs/12-kernels.md, "Rounding discipline").
set(PREP_CL ${CMAKE_CURRENT_SOURCE_DIR}/prep.cl)
set(PREP_OPTS -cl-fp32-correctly-rounded-divide-sqrt)
function(add_prep_res_norm M K SP)
  add_ocloc_kernel(prep_res_norm_M${M}_K${K}_SP${SP} SOURCE ${PREP_CL}
                   DEFINES M=${M} K=${K} S_PREV=${SP} OPTIONS ${PREP_OPTS})
endfunction()
add_prep_res_norm(1 5120 0)    # first norm of a layer: no mixer partials to fold in
add_prep_res_norm(1 5120 16)   # after out_proj / down (S = 16)
add_prep_res_norm(2 5120 16)   # spec 1 §9: the M loop must compile at M=2
add_ocloc_kernel(prep_silu_mul_M1 SOURCE ${PREP_CL} DEFINES M=1 OPTIONS ${PREP_OPTS})
add_ocloc_kernel(prep_gated_head_M1 SOURCE ${PREP_CL} DEFINES M=1 OPTIONS ${PREP_OPTS})
