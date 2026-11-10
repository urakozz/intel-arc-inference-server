# Flash-Next decode/reference diagnostic, 2026-11-10

Do not merge on these results. The sampled parameter identities pass, but decode already changes discrete routes on the first token. The 4500-token, 37-layer baseline is not established by this short check, so the failed prefill gates cannot yet be attributed specifically to prefill convolution or GDN.

The bounded probe uses Intel's checkpoint, four layers, the first eight `q4exp_4k.ids` tokens, and teacher-forced single-token calls throughout. Engine decode uses the int8 head and a 256-position allocation. Three CPU arms reuse the same engine capture: ordinary fp32; fp32 with only int4 products rounded through bf16; and bf16 activations with the int8 head's dequantized products and accumulation kept in fp32. The bf16 arm retains the established reference's bf16 int4 products, so it is a precision control, not a bitwise engine emulator.

The snapshot is `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`. Its PLE sibling and the physical `/ple` mount source resolve to the same index file: device/inode `66306:31588778`, 2145 bytes, SHA256 `d5bc173ae4cdb58a69514c3d5441f65256ceffb6f83c60e9c040da56451f1583`. All 128 selected PLE row IDs, 20,480 int8 bytes, 128 scales and 20,480 bf16 dequantized elements match between Python, the C++ mmap reader and the engine's actual pinned table. This verifies the selected rows; it is not a hash of the whole 51.8 GB table.

All 4,915,200 products in the device-packed layer-0 expert-0 gate, up and down matrices match the checkpoint's direct fp32 `(q - 8) * scale` values bitwise. The g128-to-g64 expansion duplicates scales without rounding. Decode GEMV sums integer-times-activation products and applies the f16 scale in fp32. Thus adding a bf16 weight-product rounding would change the sampled engine weights. Intel's dense and shared weights are bf16 as shipped. This is a full-matrix sample of one expert, not an exhaustive identity check of every expert.

The first observed persistent-state difference is already row 0, layer 0, before any PLE, MoE route difference or QSA/RoPE:

| Row 0 comparison against fp32 | Cosine distance | Relative L2 | Maximum absolute error |
| --- | ---: | ---: | ---: |
| L0 convolution live history | 2.126488708e-6 | 0.002151155 | 0.106773376 |
| L0 GDN state | 7.785735508e-6 | 0.004696130 | 0.010767043 |
| H after L0 | 1.133839452e-5 | 0.004850232 | 0.001078084 |

With bf16 activations, row-0 L0 convolution agrees except for four of its 10,240 live elements: distance `3.041777941e-11`, relative L2 `7.799724574e-6`, maximum error `0.001953125`. This supports an activation-rounding boundary at the earliest observed convolution difference. It does not prove that subsequent GDN/HC arithmetic is correct: that arm's row-0 L0 GDN distance is still `1.149223823e-5`.

The first route-set change occurs at row 0, L2; L0 and L1 route sets still match. The fp32 reference includes expert 338 and the engine includes 272. Reference probabilities are `p338=0.007880102843`, `p5=0.007828897797`, `p272=0.007792126387`; the reference rank-10/rank-11 gap is `3.677140921e-5`. Engine P10 and P11 both equal `0.007793190423`. Before this route, H after L1 has distance `1.625884472e-5`; L2's pre-MoE convolution/GDN states have distances `4.719166630e-6` / `1.197851942e-5`. H after L2 has distance `2.966554425e-5`. The bf16 control also changes the L2 route on row 0, at an exact cut tie. This precedes the first QSA layer, so RoPE precision cannot cause the first route change.

Fp32 route changes occur at `(row, layer)` `(0,2), (1,1), (4,2), (6,2)`. At row 6, H after L1 has distance `2.437264668e-5`, rising to `0.001273526138` after L2 and `0.001461730690` after L3; logits distance is `0.001580820332` (relative L2 `0.056311066`). This localizes an error increase to a layer with a route change, consistent with route amplification; no same-route intervention was run to prove causality. All eight engine logit argmax IDs nevertheless equal the fp32 arm's IDs on these forced prefix contexts.

Rounding only int4 products to bf16 preserves the same four route-set changes. Its maximum logits distance is `0.001580945419`, versus `0.001580820332` for direct fp32; it does not remove the disagreement. No reference default, gate tolerance or engine implementation was changed.

The next attribution decision stays with decode/reference arithmetic and routing. Replay the earliest GDN/HC/router boundary with identical intermediate inputs before interpreting the full-context token anomaly or blaming prefill-specific recurrence. The short result does not identify the cause of the reported 37-layer rank-22/33-ulp decision. Later QSA/RoPE precision and long-context amplification remain open.

Validation: box C++ probe build passed; all three eight-row CPU controls completed with all identity assertions passing (`ALLDONE rc=0`). CPU controls use the shared CPU-model lock, 16 threads and the 64g/no-swap container. GPU capture used the card-1 lock; load took 282.2 seconds, including 192.1 seconds pinning PLE. Mac `py_compile`, shell syntax and whitespace checks passed. Initial diagnostic startup failures are preserved in the logs; no engine rerun was needed after the successful capture.

Reproduce on the box, detached: `tools/probe/detach.sh ~/b70/logs/q4exp-f32/diagnose/reproduce/driver.log bash tools/probe/qwen4exp_decode_diagnose.sh reproduce`. To reuse the unchanged capture, append `reference-only`; optional subsequent arguments select `fp32`, `rounded`, or `bf16`. Outputs are under the worktree's `oracle-out-q4exp-decode-diagnose`; successful logs are `~/b70/logs/q4exp-f32/diagnose/attempt1/engine.log` and `~/b70/logs/q4exp-f32/diagnose/reference4/`. All 504 finite measurements are in [the CSV](q4exp-f32-decode-diagnostic-2026-11-10.csv). `route_missing` rows record the missing-expert count in `max_abs`; their relative L2 against zero is not a useful numerical error measure. Empty tensors are omitted from measurements, and GDN's dead convolution slots are masked to zero on both sides.
