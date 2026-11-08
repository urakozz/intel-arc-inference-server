# 3e: int8 KV prefill

Baseline: main ba99cb9f, compiled unchanged in this worktree's `build-before`
on the box with 16 build jobs. Card 1 through `tools/gpu_lock.sh card 1`.
Qwen3.8-27B GPTQ, l0 prefill, 2048-row chunks, 15,257 synthetic ids, one card,
16 generated ids, max_len 131072. Each KV form warmed before its instrumented run.

| measurement, ms | bf16 KV | int8 KV |
|---|---:|---:|
| unprofiled prefill | 9832.1 | 20355.6 |
| instrumented prefill wall | 10392.0 | 20880.6 |
| flash-attention GPU | 1192.6 | 11876.6 |
| attention prep GPU | 50.7 | 85.2 |
| attention gate GPU | 33.6 | 45.2 |
| all GPU kernels | 9570.7 | 20073.3 |
| GPU kernels excluding flash | 8378.1 | 8196.7 |

The flash kernel accounts for the extra time: +10,684.0 ms of GPU execution,
against +10,488.6 ms of instrumented wall time. The bf16 and int8 kernels
already share KT=64, RPW=8, HPW=6 and the DPAS/online-softmax structure.
The int8 kernel constructs its K operand with scattered vector loads and its
V operand with scalar loads, while bf16 uses 2D block reads. This is the first
load-path hypothesis to test; the scale arithmetic and accumulation order stay.

Logs: `~/b70/logs/3e/profile-{bf16,int8}-before.log` and
`profile-warmup-{bf16,int8}-before.log`. Both profiles completed. The driver later
exited 1 because its script was updated in place during execution; subsequent
jobs must run fixed script copies. This does not invalidate the completed
profile tables or benchmark rows.


## First candidate: 2D int8 block reads

K uses transposed 32-bit block reads, V uses the 8-bit transform read. Raw
int8 values widen exactly to bf16 in registers; the scales, accumulation
order, tile order and probability rounding are unchanged. No runtime switch.

The detached build and comparison completed with `ALLDONE rc=0`. Every
fp32 output word (including padding) and gated bf16 word matched the original
kernel for nine shapes, through a last position of 128000. Initial 15257-id
unprofiled times were bf16 9875.6 ms and int8 10721.0 ms (8.56% behind bf16).
Profiled flash GPU time fell from 11876.6 to 1904.8 ms (bf16 1179.8 ms).
This is a substantial improvement but does not yet meet the approximately
5% target. Next: full-chunk bitwise cases at long positions, register-resident
Q / GRF compile probes, the required engine gates and interleaved paired runs.

Log: `~/b70/logs/3e/block-read-driver.log`.
