# Prefill-only kernels (spec 2). None is bound by src/runtime/capture.cc and
# none changes the decode list's 774 launches or 19 modules.
set(PF_DEQUANT_CL ${CMAKE_CURRENT_SOURCE_DIR}/dequant.cl)

function(add_pf_dequant K N L T)
  add_ocloc_kernel(pf_dequant_tile_K${K}_N${N}_L${L}_T${T} SOURCE ${PF_DEQUANT_CL}
                   DEFINES K=${K} N=${N} LAYOUT=${L} TRANSPOSED=${T})
endfunction()

# The unit-test shape, both layouts x both output orientations.
foreach(L 0 1)
  foreach(T 0 1)
    add_pf_dequant(256 64 ${L} ${T})
  endforeach()
endforeach()

# Production dimensions: both loader layouts and both store geometries are
# built so the probe measures rather than assumes either contrast.
set(PF_SHAPES "5120 16384" "6144 5120" "5120 34816"
              "17408 5120" "5120 14336" "5120 248320")
foreach(row IN LISTS PF_SHAPES)
  separate_arguments(f UNIX_COMMAND "${row}")
  list(GET f 0 K)
  list(GET f 1 N)
  foreach(L 0 1)
    foreach(T 0 1)
      add_pf_dequant(${K} ${N} ${L} ${T})
    endforeach()
  endforeach()
endforeach()
