# Spec 15b - ModelDesc for every per-model shape

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** extend spec 14's `ModelDesc` from layer counts and intermediate size to every shape that differs between Qwen3.5-family models — hidden size, attention q / kv heads, GDN v-heads (and conv dim), FFN kind (dense or MoE with experts, top-k, expert and shared intermediate) — with Qwen3.8 and Agnes bitwise unchanged (R0).

**Architecture:** today `model::Qwen35` keeps hidden 5120, 24/4 attention heads, 16/48 GDN heads and the per-layer linear table as `constexpr`; spec 14's `ModelDesc` (branch `spec14-agnes`) carries layers, GDN/FA counts and intermediate. This plan moves the rest into the descriptor, routes buffers, the capture, prefill and the kernel variant names through it, and adds an `FfnKind { Dense, Moe }` with the MoE fields (unused until 15c). Kernels already take shapes as CMake defines; the variant names gain the new parameters.

**Tech Stack:** C++17 (the project standard), CMake, OpenCL C (defines only).

**Spec:** `docs/superpowers/specs/2026-10-04-spec15-ornith-moe-design.md` (§3 decision 2, §4.3, §5 R0, §7 15b). Depends on spec 14 (`spec14-agnes`) merged — or branch from it if it has not merged yet, and say so.

## Global Constraints

- Branch `spec15b-model-desc` from main (or from `spec14-agnes` until it merges); commit there; no merge, no push.
- **Writable without the box** (spec 14's write-phase method): host code compiled and host tests run with Apple clang on the Mac; Level Zero sources syntax-checked against the open-source headers; kernel command lines checked with `tools/kernel_cmdlines` (from spec 14). **R0 itself is validated on the box** later: every existing registration, golden and replay bitwise, Qwen3.8 and Agnes kernel binaries checksum-identical to main's. Add the R0 steps to a validation checklist (Task 4).
- No `rm -rf`. No `tools/box.sh` unless the box is available.

## Review Focus

1. **The blast radius:** `grep -rn "5120\|kHidden\|24u\|kFaQHeads\|kFaKvHeads\|kGdnVHeads\|48\b\|10240\|12288"` over `src tests tools` before and after; every remaining literal is either a genuinely shared constant (head_dim 256, GDN head dim 128, vocab) with a comment, or routed.
2. **Derived sizes** that hide the shapes: GDN conv channels (`2 x k_heads x 128 + v_heads x 128`: 10240 for Qwen3.8, 8192 for Ornith), q_proj width (2 x q_heads x 256 with the output gate), the residual and norm buffer widths, `attn_part` and the decode attention's heads-per-kv-head (6 for Qwen3.8, 8 for Ornith).
3. **Kernel command lines:** all Qwen3.8 and Agnes kernels keep identical names and command lines (`tools/kernel_cmdlines` diff empty); new variants only appear when a descriptor asks for them.
4. **`FfnKind::Moe`** carries `experts`, `top_k`, `expert_intermediate`, `shared_intermediate`, `has_shared_gate`; nothing reads it yet except the descriptor test.
5. **Ornith's descriptor exists** (`Qwen3_5MoeForConditionalGeneration` → it) but loading it throws "MoE not implemented (spec 15c)" until 15c.

---

### Task 1: the descriptor fields and Ornith's row

**Files:** `src/model/model_desc.{h,cc}`, `tests/model/model_desc_test.cc`.

- [ ] Failing test (Qwen3.8, Agnes, Ornith rows; derived sizes of Review Focus 2) → implement → pass on the Mac. **Commit** `model: ModelDesc carries hidden, heads, GDN heads and the FFN kind (spec 15b)`.

### Task 2: route the runtime and prefill

**Files:** `src/model/qwen35.{h,cc}`, `src/runtime/{buffers,capture,engine}.*`, `src/runtime/prefill/*`, `src/loader/*`.

- [ ] Replace each use (Review Focus 1, 2); host tests on the Mac; L0 syntax check. **Commit** `runtime: per-model shapes from ModelDesc (spec 15b)`.

### Task 3: kernel variant names

**Files:** `src/kernels/CMakeLists.txt`, `src/kernels/prefill/CMakeLists.txt`, `src/kernels/kernels.h`.

- [ ] Variant names take the descriptor's shapes; Review Focus 3 (empty diff for Qwen3.8 and Agnes). **Commit** `kernels: variant names from ModelDesc shapes (spec 15b)`.

### Task 4: the R0 checklist

- [ ] `docs/superpowers/plans/2026-10-04-spec15b-validation-checklist.md`: build, kernel checksums against main, the full suite, golden and replay bitwise, on Qwen3.8 and Agnes; each with its pass condition. **Commit** `plans: spec 15b R0 checklist for the box`.

**Gate for the plan:** host tests pass on the Mac, the kernel command-line diff is empty for Qwen3.8 and Agnes, the checklist is written. R0 on the card is pending until the box.
