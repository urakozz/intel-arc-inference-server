# P2 - sycl-tla stock bf16 GEMM at the prefill shapes (spec 2 §4)

grade: iterate (card 1, ZE_AFFINITY_MASK=1, card 0 may be held)

## Pre-registered predictions (written 2026-09-04, BEFORE any measurement)

1. **≥ 90 TFLOP/s at M = 4096 on gate‖up** (spec §4's P2 row; vLLM's implied
   98.2 TFLOP/s sustained is the yardstick, `explorer-2 §4`).
2. TFLOP/s rises monotonically in M over {512, 1024, 2048, 4096} at every
   shape: weight bytes amortise over more rows, and the work-group tile is
   `Shape<_256,_256,_32>` (`00_bmg_gemm.cpp:364`), so M = 512 is only two
   tiles deep and M < 256 would not fill one.
3. `out/o_proj` (6144×5120) is the worst cell at every M - the smallest N,
   and the shape decode also finds hardest (`docs/12`, 533 GB/s).
4. Two runs of the same cell are **bitwise identical**. Source basis, not
   hope: the configuration omits the `TileScheduler_` template argument, so
   it is the data-parallel `PersistentScheduler`, and
   `include/cutlass/gemm/kernel/xe_gemm.hpp:82-87` static_asserts
   *"Intel Xe does not support specializing the tile scheduler"* - Stream-K
   routes to a **different** kernel (`xe_gemm_cooperative.hpp`) which this
   probe does not instantiate. There is no atomic accumulation into C.
5. `Gemm::get_workspace_size(arguments) == 0`. It has to be: the queue
   variant treats a non-zero workspace as a hard error
   (`00_bmg_gemm_with_sycl_queue.cpp:263-265`).

## What prediction 1 implies for the whole spec - pre-registered, DERIVED

Per-token GEMM FLOPs, from `model::Qwen35`'s table:
GDN layer = QkvZ + OutProj + GateUp + Down = 0.76546 GFLOP/row, ×48;
FA layer = Qkv + OProj + GateUp + Down = 0.74449 GFLOP/row, ×16.
**Σ = 48.654 GFLOP/row.** (Reconciles with `explorer-2 §1`'s 48.97
GFLOP/token, which additionally counts the conv, the recurrence and the a‖b
linear - the two are different quantities, not two values for one.)

At C = 4096 that is **199.3 TFLOP of GEMM**. At exactly 90 TFLOP/s the GEMM
term alone is **2.215 s** - **6.7 % ABOVE vLLM's 2.076 s for the entire
pp4096** - before dequant, attention, GDN, norms or interop. So:

> **If P2 measures 90 TFLOP/s, the composed ceiling lands under vLLM and spec
> §2's second bullet fires.** To reach 2.076 s device-side the GEMM term must
> run at **≥ 96 TFLOP/s** *and* everything else must be free. This is written
> down before the measurement so T6's conclusion cannot be a hindsight
> rationalisation of whatever number comes out.
