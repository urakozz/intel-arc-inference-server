# DFlash 8-row verify on the int8 KV cache: no defect in the M = 5..8 binaries (2026-11-13)

Branch `dflash-kv8-verify-m8`, card 1. Logs: `~/b70/logs/dflash-kv8-verify-m8/`.

## The question

`dflash_gpu_kv8_test`, t1_define-rotation, K = 7 and auto K: generated id 200 is 248068, where the plain int8-KV row
and the int8-simulating f32 judge both pick 248046 by 34.6 bf16 ulp. It was identical at K = 3 and on the bf16 cache.
The suspects were spec 19v3a's M = 5..8 int8 binaries (the int8 writer `attn_prep_M<M>_KV8`, v2's int8 pair, and
`attn_verify_dpas_M<M>_T32_KV8`) and the all-depth verify dispatch.

## What was measured

1. **The exact path is bitwise at every M** (`dflash_verify_kv8_test`, part H, new). With the verify lists forced to
   v2 attention and gemv linears, each verify row does the M = 1 list's arithmetic. At every fixed K = 1..7 the DFlash
   stream equals the plain stream. The KV rows the verify lists wrote (int8 codes and fp16 scales, every FA layer)
   also equal the plain run's bit for bit. This held on t1_define-rotation (960 + 256) and on long32k cut to 32,000
   (+ 64). So the M = 2..8 int8 writer, v2's int8 pair at M rows, the multi-row GDN step and the commit are exact.
   Log: `dv_main2.log`.
2. **The DPAS kernel at M = 5..8 is as accurate as at M = 2..4** (`attn_verify_dpas_test` part D, real FA-layer-15
   dump, now printed per M; `kernel_d.log`). On int8, against fp64: 0.9999998 at 512 keys, 0.99999994 at 4096 and
   32768, and 0.9999984 at 130816. Within each depth the value is the same for every M, the M = 2..4 PAIR build
   included. bf16 shows the same picture. PASS.
3. **The flip needs both approximations, at any M** (`dflash_verify_kv8_test` part A, t1_define-rotation, 256 ids):

   | lists | K = 3 | K = 4 | K = 5 | K = 6 | K = 7 |
   |---|---|---|---|---|---|
   | production rule | identical | id 200 flips | flips | flips | flips |
   | v2 attention + gemm_sm | identical | identical | identical | identical | identical |
   | DPAS attention + gemv | identical | identical | identical | identical | identical |
   | DPAS attention + gemm_sm | **id 200 flips** | flips | flips | flips | flips |

   Every flip is 248068 for 248046, rank 1, 34.63 ulp. At K = 3 the 4-row PAIR kernel flips the same id as soon as it
   is bound. The production K = 3 is identical only because the table (`runtime/verify_attn.h`) binds v2 for int8
   M = 2..4 below 4096 keys, while spec 19v3a binds DPAS for int8 M = 5..8 from the first key.
4. **The row is unstable even for a correct computation** (`probe_verify_rows`, now on the default KV cache, with
   every verify row's watched gap; `probe2.log`, `probe3.log`). The row's gap moves with history alone. The row that
   decides id 200 is row 1 of the DFlash iteration at position 1158. On the DFlash run's history it gives +68 to +70
   ulp for 248068 under every arm, v2 + gemv included. On the plain history the same row gives -32 to -37 under every
   arm. Row 0 on the same state also moves by about 50 ulp between the two histories (-55.8 against -3.9). The
   context is the model emitting `<|im_end|>\n<|im_start|>assistant\n` in the middle of a reply. Its standard-bf16
   reference gap is 6.5 ulp against f32's 34.6, which is why the judge prints "bf16-UNSTABLE". On the bf16 cache, an
   M = 8 verify row at the next position moves its layer-61 tap to cosine 0.80 against the plain row.

## Verdict

There is no defect in the int8 M = 5..8 binaries or in the dispatch. The divergence comes from the two approximations
the operator already accepted: DPAS verify attention (spec 28c) and gemm_sm verify linears (spec 28b). Together they
shift the written history slightly, and this one bf16-unstable row amplifies that shift. Either approximation alone
stays identical here. The 4-row DPAS kernel does exactly the same, so this is not specific to 8 rows. "Identical on
bf16 KV" is the bf16 stream's own luck: the bf16 run takes a different path through this context. No engine code was
changed.

## Options (operator)

- (a) Accept: record a reviewed entry for `dflash_gpu_int8_t1_define-rotation` row 200 (248046 / 248068, 34.63 ulp).
  This follows the same mechanism as the 2026-11-05 t5_test-box entries.
- (b) Put int8 M = 5..8 back on v2 below 4096 keys, as M = 2..4 are. That costs the shallow-depth DFlash speed spec
  19v3a measured, and it only moves the luck: v2 + gemm_sm can flip other unstable rows.
- (c) Revisit the bf16-unstable allowance (max(8, |plain - f32|)) for rows where the bf16 reference is this far from
  f32.
