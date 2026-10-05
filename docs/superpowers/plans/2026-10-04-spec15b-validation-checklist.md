# Spec 15b - R0 on the box: Qwen3.8 and Agnes unchanged

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. One box session, in this order; a step that fails stops the session there (fix on the branch, re-run from step 2).

**Goal:** prove branch `spec15b-model-desc` changes nothing on the card for Qwen3.8 and Agnes (spec 15 §5 R0), then merge. Nothing here measures Ornith: its descriptor exists but `loader::load` refuses it ("MoE not implemented (spec 15c)").

**Spec:** `docs/superpowers/specs/2026-10-04-spec15-ornith-moe-design.md` (§3 decision 2, §5 R0, §7 15b). **Plan:** `docs/superpowers/plans/2026-10-04-spec15b-model-desc-all-shapes.md`.

**Order with spec 14:** main carries spec 14 **unvalidated on the card** (its checklist, `2026-10-03-spec14-validation-checklist.md`, is pending). Run spec 14's session on main first; this session's "main" reference below is the main build spec 14's session validated (or main at `c104247` if both run in one sitting, spec 14's G0 first). A failure here that also fails on main is spec 14's, not 15b's.

**Branch commits (written on the Mac, nothing ran on a card):**

| commit | what |
|---|---|
| `a314c97` | `ModelDesc` carries hidden, FA q/kv heads, GDN k/v heads, `FfnKind` + `MoeDesc`, `mtp_intermediate`, the small-tensor tables and their layout (`loader::make_small_layout`, `kQwen38Small` pins the old offsets); Ornith's row; `model::require_loadable`. Does not build the device tree alone. |
| `b3a8729`, `5adda67` | runtime, prefill, engine and loader read those shapes from the descriptor (`5adda67`: `require_loadable` runs before the quant-config parse); `x` = max(intermediate, 2 x hidden), prefill `mixer_out` = max(value dims, hidden) - identical bytes on Qwen3.8 and Agnes |
| `9d714c7` | kernel variant names take the shapes (`kernels/shape_suffix.h`; empty suffix at Qwen3.8's shapes); `variant_names_test` |

## What the Mac already proved (2026-10-05)

| check | result |
|---|---|
| host tests (Apple clang 21, `-std=c++17 -Wall -Wextra -Werror`) | 17/18 + 2 Agnes template modes - as main, plus the new `variant_names_test`; `template_test`'s default mode needs the Qwen3.8 snapshot (absent on the Mac, as on main) |
| `model_desc_test` | Qwen3.8 and Agnes shapes, derived widths, small layout = `kQwen38Small`, MTP bytes 849,398,784, empty kernel suffixes; Ornith's row (40 / 30 / 10 layers, 2048, 16 / 2, 16 / 32, conv 8192, MoE 256 x top-8 x 512 + shared 512 with gate, vocab_used 248077) and `require_loadable` throwing "MoE not implemented (spec 15c)" |
| `variant_names_test` | every shape-baking variant name the runtime builds equals main's at Qwen3.8 and Agnes; Ornith's carry `_D2048` / `_GK16V32` / `_Q16KV2` |
| Level Zero syntax check (`-fsyntax-only`, open-source L0 headers) | 144/151 clean - the 7 SYCL / cutlass probes, as on main |
| `tools/kernel_cmdlines` against main (`c104247`) | **identical**: default 271, `-DB70_AGNES_SWEEP=ON` 289, `-DB70_DECODE_EXTRA_M=ON` 271. No `.cl` source changed (`git diff c104247 -- 'src/kernels/*.cl' 'src/kernels/prefill/*.cl'` is empty), so the binaries should be byte-identical. |

Nothing was linked against Level Zero and nothing ran on a GPU.

## Global constraints

- Box workflow as always: `BOX` from `tools/box.env`; per-branch tree `tools/box.sh dir`; every GPU command under `flock ~/b70-gpu.lock`, detached (`tools/probe/detach.sh <log> <cmd>`), polled with a background until-loop; idle-box protocol before any timing; device 0.
- Builds `-j44` (`JOBS=44`). No `rm -rf`. Commit on `spec15b-model-desc`; merge only after step 6.

## Steps

### 1. Build

```
tools/box.sh dir                         # the spec15b-model-desc tree
JOBS=44 tools/box.sh build               # -Werror, the full tree (device code links here first)
```
Pass: the build is clean. This is the first time `capture.cc`, `prefill/*.cc`, `engine*.cc`, `buffers.cc` and `loader.cc` compile with the real toolchain against the routed descriptor (the Mac only syntax-checked them).

### 2. Kernel binaries checksum-identical to main's

```
ssh $BOX 'cd <main tree> && cmake --build build -j44'     # main as validated by spec 14's session
ssh $BOX 'cd <main tree>/build/kernels && sha256sum *.bin | sort -k2' > /tmp/main.sha
ssh $BOX 'cd <spec15b tree>/build/kernels && sha256sum *.bin | sort -k2' > /tmp/branch.sha
diff /tmp/main.sha /tmp/branch.sha                         # must print nothing
```
Pass: **no difference at all** - the same file set and every sha256 equal (15b adds no binary; Ornith's arrive in 15c). Repeat with `CMAKE_ARGS=-DB70_AGNES_SWEEP=ON` only if spec 14's session still builds the sweep.

### 3. The host and table tests

```
ctest -R 'model_desc_test|variant_names_test|qwen35_test|kernel_table_test|agnes_fold_test|template_test'
```
Pass: all green (`kernel_table_test` finds every Qwen3.8 row's binary; the first two re-run what the Mac ran).

### 4. The full Qwen3.8 suite, bitwise against main

```
JOBS=44 tools/box.sh test ''             # ctest -LE agnes, detached, under the lock
```
Pass: everything green, and bitwise against main's record of the same build (step 2's main tree run the same way, or spec 14's G0 record):
- `buffers_test`: the byte table unchanged (`x` 8 x 17408 x 2, `mixer_out` 2048 x 6144 x 2, pf_s / pf_p at `s_heads()` = 6, every total as pinned) - the routed sizes are the old numbers.
- `replay_determinism_test` (+ `_i8head`): `kernel_count` 774, 19 modules, replay bitwise.
- `golden_gate_test`, `golden_gate_i8head_test`: same verdict lines, per-layer cosines identical to the last digit.
- `prefill_gate_*` (l0, l0-int8), `prefill_consistency_test`, `prefill_split_test`, `prefill_smoke_test` (launch arithmetic 8449 / l0-int8 8705: the per-kv-head terms now read `fa_kv_heads` = 4), `attn_chunk_test`, `gdn_chunk_test`: same results as main.
- `load_checkpoint_test`: the small blocks at `kQwen38Small`'s offsets, the MTP head's 15 tensors / 849,398,784 B (now `ModelDesc::mtp_checkpoint_bytes()`), the W line within 2% with the same delta as main (the lm_head byte terms are now computed from `hidden`).
- `mtp_verify_test`, `mtp_head_test`, `mtp_prefill_test`, `snapshot_test`, `prefix_gpu_test`, `engine_smoke_test`: as main.

### 5. Agnes, bitwise against main

```
ctest -L agnes                           # the spec 14 registrations, under the lock
```
Pass: every Agnes test that passed in spec 14's session passes here with identical output (`load_agnes_test` incl. `mtp_checkpoint_bytes` = 849,398,784; `golden_gate_agnes*`, `prefill_gate_agnes*`, `replay_determinism_agnes`, `mtp_*_agnes`). If spec 14's session has not run, these are spec 14's gates and run there first.

### 6. Ornith refuses to load, then the record

```
# a directory holding only Ornith's config.json is enough: the refusal precedes any tensor read
ssh $BOX 'mkdir -p ~/ornith-cfg && cd ~/ornith-cfg && uvx --from huggingface_hub hf download ornith-ai/Ornith-1.5-35B-A3B config.json --local-dir .'
build/src/cli/b70-decode ~/ornith-cfg --bench --depth 16 --tg 1 2>&1 | tail -2
```
Pass: it fails with `ornith-1.5-35b-a3b (Qwen3_5MoeForConditionalGeneration): MoE not implemented (spec 15c)` (from `loader::load` -> `model::require_loadable`, ahead of the quantisation-config parse and of any tensor read).

Then: one paragraph in the spec 15 record (or `docs/probe-ornith-*.md` when 15a creates it) - R0 passed, the checksum compare, the suite and Agnes results with their commits - and offer the fast-forward merge of `spec15b-model-desc` into main.

## Pending until this session

- R0 itself: steps 2, 4 and 5 (the binaries, the Qwen3.8 suite and the Agnes gates bitwise).
- The device-code build of the routed runtime (step 1).
- Ornith's provisional values, which this session does NOT replace (spec 15c's P0 does): the int4 rows' `{S, layout}` (copied from Qwen3.8's map), `doc_w` 0 (no int4 checkpoint), `kNormGroups` 20 / `kConvRing` 16 / the a||b pad to 128 kept at hidden 2048, and the shared-expert reading of the GateUp / Down table rows.
