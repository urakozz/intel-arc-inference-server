# Spec 12b - int8 KV cache in the engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** store K and V as int8 with the scheme plan 12a chose, behind `--kv-cache bf16|int8`; quantise in the writers, dequantise in the readers; spec 7 and spec 8 interplay; gates Q2-Q5 and the speed bars.

**Architecture:**
- **Writers:** `attn_prep` (decode, `src/kernels/attn.cl`) and `pf_attn_prep` (prefill, `src/kernels/prefill/pf_attn_prep.cl`) rotate their K and V rows with the 256-point Hadamard (the operator's `rotkv` ruling, spec 12 §8), quantise per token per head with fp16 scales, and write the scales; q is rotated the same way before attention.
- **Readers:** decode attention v2 (`src/kernels/attn_v2.cl`, spec 10) and `pf_flash_attn` dequantise while loading, and **un-rotate the attention output per head before the sigmoid output gate** (spec 12 §8); v1 decode and the composed prefill path stay bf16-KV only (references).
- **Buffers:** `src/runtime/buffers.{h,cc}` allocate int8 KV plus scales when selected; the MTP head's KV (spec 8) follows.
- **Snapshots:** `Engine::save_kv`/`load_kv` and `kv_bytes()` (spec 7) copy the int8 rows and the scales.

**Tech Stack:** OpenCL C (ocloc), C++20, Level Zero, ctest.

**Spec:** `docs/superpowers/specs/2026-09-28-spec12-int8-kv-cache-design.md` (§3, §4 Q2-Q5, §5). Needs plan 12a's verdict (`docs/probe-int8-kv-2026-09-28.md`).

**Status (2026-10-05): Tasks 1-3 written blind on the Mac, branch `spec12b-int8-kv`; Task 4 and
every card-side check open.** The operator set Review Focus 0 aside: 12b was written before the
Qwen3.8 repeat of 12a, which runs on the Mac CPU (Task A below) and re-derives the PROVISIONAL
tolerances. As built: spec 12 §9. What runs on the card, in order: box-validation-queue row 11.

### Task A: the 12a repeat on Qwen3.8, on the Mac CPU (tonight)

- **Download** the bf16 base checkpoint into the Mac's HF cache (it was downloading on
  2026-10-05, snapshot `1d4bf0f`): `uvx --from huggingface_hub hf download Qwen/Qwen3.8-27B`
  (~54 GB; the script refuses to start until every shard the index names is present).
- **Run**, from the repo (or this branch's worktree) root, detached:
  `nohup tools/oracle/kv8_qwen38_repeat.sh > /dev/null 2>&1 &`, then
  `tail -f oracle-out-12a-qwen38/repeat.log`. It starts a container `kv8-ref` from
  `agnes-ref-img:latest` (torch 2.14.1+cpu, transformers 5.15.0, g++; the image the Agnes run
  used) capped at 28 GB, streams the model a layer at a time, and runs, each step resumable
  (an existing output is skipped): the probe's unit test and a tokenizer round trip; the three
  golden prompts with the 2026-08-24 oracle's 32 greedy ids teacher-forced (six variants);
  the two A4 prompts `t1_define-linear_l0` (963 ids) and `t2_explain-box` (1023); long32k's
  first 4096 ids (five variants, with the q/K/V capture); the fp64 replay of that capture at
  1k/2k/4k real and 8k/16k/32k tiled; then `oracle-out-12a-qwen38/summary.md`.
  `STEPS="golden a4"` runs a subset; `LONG_N=8192` the 8192-id capture §6 asks for (about
  twice the long step, so only if the night allows).
- **Expected time** (derived from the Agnes run on this Mac, §7 of the 12a record, at 64
  layers against 72): golden ~10 min per prompt, A4 ~25-60 min each, long 4096 x 5 ~2.5-3 h,
  replay ~0.5-1.5 h - **about 5-7 h** in all, peak RSS ~25 GiB.
- **What decides the 12b tolerances** (12a §4-§6 against these numbers):
  1. the stop rule: `rotkv`'s golden decision-row **cos mean** against `l0-int8`'s
     **0.999931742** (Agnes: 0.9999467) - below it on every scheme is a stop;
  2. Q2 (`kv8_vs_oracle_test`, PROVISIONAL 1e-4): the golden **1 - cos mean** of `rotkv`
     (Agnes 5.3e-5) and its **KL mean** (1.9e-4; the KL bar 1e-3);
  3. Q3 (`flash_long_kv8_test`, PROVISIONAL 5e-4): long4k's **2048-4095 bucket 1 - cos mean**
     (Agnes 3.1e-4) and the **last row** (1.3e-4), with the replay flat from 4k to 32k
     (`rotkv` rel L2 mean against `ctl_eager`'s);
  4. argmax: the non-near-tie differences (gap > 0.05) in the long buckets (bar ≤ 0.5 % of
     rows);
  5. the scheme itself: `rotkv` still best on Qwen3.8 (the replay's K:pt / V:pt diagnostics
     show whether K's outlier channels and V's per-token term behave as on Agnes). If another
     scheme wins, spec 12 §8's ruling is the operator's to revisit before the box run.
  Then edit the PROVISIONAL constants (`tools/probe/kv8_vs_oracle.sh` `KV8_TOL`,
  `flash_long_test.cc` `kTol`, and the gated-rows bar in `kv8_kernels_test.cc`) and record the
  run as `docs/probe-int8-kv-qwen38-<date>.md`.

## Global Constraints

- Branch `spec12b-int8-kv` from main; box tree `~/b70-inference-server-spec12b`. Copy `tools/box.env` if missing; never commit it. Symlink `oracle-out*` dirs.
- Every GPU command under `flock ~/b70-gpu.lock`, detached, polled. Interleaved pairs, median of 3, `uptime` recorded.
- `--kv-cache bf16` (the default until the gates pass) is bitwise today's engine. No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

0. **Do not start before the Qwen3.8 repeat of 12a** (spec 12 §8, "Before 12b"): the scheme and tolerances are confirmed on Qwen3.8 first.

1. **Every KV writer and reader** is covered: decode attn_prep, prefill pf_attn_prep, the MTP head's KV fill in prefill and in verify (spec 8), decode v2, pf_flash_attn, save_kv/load_kv. A grep for `kv_k`/`kv_v` bindings in `capture.cc` and `step.cc` lists them; each gets a test.
2. **Mixing forms is refused:** a snapshot saved at bf16 cannot be loaded into an int8 engine (spec 7's cache keys gain the KV form).
3. **Verify rows bitwise equal to M = 1 (spec 8 M2)** still hold with int8 KV (quantisation per row is row-local).
4. **Scales at the positions MTP rejects:** stale int8 rows past `pos` are rewritten before read, as with bf16.
5. **The composed prefill path** with int8 KV is rejected with a clear error, not silently wrong.

---

### Task 1: buffers, writers, flag

Written blind (2026-10-05): `KvCache` / `KvLayout` (runtime/buffer_sizes.h), the buffers, the
planner, `attn_prep_kv8` (decode, S1, prefill) in `src/kernels/kv8.cl`, `--kv-cache` in both
CLIs; host references `src/common/kv8.h` + `tests/kernels/kv8_ref.h`; the writer ran bitwise
on the Mac's UHD 630 (indicative). Card: `kv8_kernels_test` part W.

- [ ] Buffers and the memory report line; `attn_prep` / `pf_attn_prep` rotate (K, V and q) and quantise; `--kv-cache` in both CLIs; unit tests of the writers against a host reference (`tests/kernels/*` style). **Commit** `runtime: int8 KV cache buffers and writers (spec 12)`.

### Task 2: readers

Written blind: `attn_decode_v2_kv8` / `attn_reduce_v2_kv8` and `pf_flash_attn_kv8` +
`pf_attn_gate_kv8` (kv8.cl); the un-rotating gate ran on the Mac. The int8-KV cases live in
the new `kv8_kernels_test` (parts D and F, against kv8_ref.h's fp64 attention) rather than in
`attn_v2_test` / `pf_flash_attn_test`, whose sources stay untouched.

- [ ] decode v2 and `pf_flash_attn` read int8 + scales; `attn_v2_test` and `pf_flash_attn_test` gain int8-KV cases (cosine against their bf16-KV outputs >= the tolerance 12a proposed). **Commit** `kernels: attention reads the int8 KV cache (spec 12)`.

### Task 3: spec 7 and spec 8 interplay

Written blind: save_kv / load_kv / kv_bytes over KvLayout (rows, then scales); the prefix
cache's hash root is the KV form (`prefix_cache_test` on the Mac); the MTP head's KV in the
same form (verify fill, draft, prefill `step_mtp_kv`); `snapshot_test` and `mtp_verify_test`
read the layout; their `_kv8` twins are registered (`ctest -L kv8`).

- [ ] save_kv/load_kv/kv_bytes; the cache key's KV form; the MTP head's KV; `snapshot_test`, `prefix_gpu_*`, `mtp_verify_test` with int8 KV (Review Focus 2-4). **Commit** `runtime: snapshots and MTP with the int8 KV cache (spec 12)`.

### Task 4: gates, speed, record

- [ ] **Step 1:** Q2 golden gates (`l0`, `l0-int8`) with int8 KV; Q3 `flash_long_test` oracle-closeness at 32k with 12a's tolerance and passkey 3/3 at 120k; Q4 A4 >= 25/36 and 256 greedy ids diverging only at near-ties; Q5 determinism/replay; full suite.
- [ ] **Step 2:** interleaved pairs bf16-KV vs int8-KV: decode at 32k / 64k / 128k (bars >= 1.1x / 1.15x / 1.2x), prefill at depth (<= 3 % regression), the memory line at 131072.
- [ ] **Step 3:** make int8 the `b70-serve` default only if every gate passes; BENCHMARKS section "int8 KV cache (spec 12)"; spec 12 amendment §8; README depth line.
- [ ] **Step 4: Commit** `spec 12: int8 KV cache gates and speed (the record)`.

**Gate for the plan:** Q2-Q5 green, speed measured (misses recorded), full suite green.
