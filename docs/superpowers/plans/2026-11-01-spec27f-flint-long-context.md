# Spec 27f - Flint long context: int8 KV for the 14 full layers, the size model, max-len auto capped at 262,144, and gate K (int8 KV quality within 262,144)

**Status (2026-11-01): plan written, nothing built.** Needs 27c and 27d. This plan carries the operator's first goal (usable context >= 242k). **Decided (operator, 2026-11-01, spec 27 D3): Flint's context is 262,144 - its vendor limit - with int8 KV; no option, probe or gate beyond it** (330k+ is K2-Horizon's target, not Flint's).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed, `git -c gpg.program=$(brew --prefix)/bin/gpg commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** halve the KV bytes of the full layers (28,672 -> 14,560 B per position) in decode **and** prefill, so one card holds the decided 262,144 with room (spec 27 section 3: 3.8 GB of int8 KV, ~4 GB spare; bf16 only marginal) and decode at depth reads half the bytes (the KV read equals the weights at ~104k tokens); max-len auto = min(262,144, memory); **gate K** records that int8 KV holds quality against bf16 KV within 262,144.

**Architecture:** spec 27 sections 3, 7.3, 9 (K). Template: K2's hd128 int8 KV (`src/kernels/k2/k2_kv8.cl`, `src/common/kv8.h` namespace `hd128`: f16 scale per (position, kv head), Hadamard rotation, K scale multiplies the score, V scale folds into P before the bf16 round), spec 12 (`2026-09-28-spec12-int8-kv-cache-design.md`), spec 10 (decode attention at depth, split-K TGT 32), `src/cli/max_len.h`, `src/runtime/memory_plan.*`.

**New files:** `src/kernels/flint/fl_kv8.cl` (prep write path: quantise k and v rows of full layers; sliding rings stay bf16), `sink_attn_kv8_decode` / `_reduce` and `sink_pf_attn_kv8` (the 27c / 27d sink kernels with the int8 reader), host twin `src/common/kv8.h` additions only if the hd128 twin lacks a function (existing functions untouched).

## Global Constraints
- Branch `spec27f-flint-long-context`; signed commits; no merge, no push; GPU jobs under `~/b70-gpu.lock`.
- F0 as in 27c. **No new user flag:** the KV form is a capture rule (int8 for the full layers, D3; bf16 only for the diagnostic arm and gate K's comparison); max-len `auto` = min(262,144, the free memory in 256-quanta); `--max-len N` stays the one existing control and is refused above 262,144 by name (the cap is `max_position_embeddings`, read by `FlintDesc`, not a constant).
- Rings (35 x 4096 x 2048 B = 0.29 GB) stay bf16 (D4).

---

### Task 1: the twins and the int8 kernels (Mac, then card)
- [ ] **Step 1:** host twin of the int8 write (scale `f16_rne(amax/127)` per head row after the Hadamard, rounding as the K2 twin) and of the sink reader; failing tests (round trip error bound; sink = -inf bitwise the no-sink int8 kernel). **Step 2:** `fl_kv8` and the int8 attention kernels: F2 against the bf16 kernels within the int8 bound and against fp64; ring wrap and window skip unaffected (sliding layers do not use int8). **Commit each.**

### Task 2: the size model and max-len auto
- [ ] **Step 1:** `flint_sizes` int8 terms (14,560 B per position); `runtime::plan`'s bisection (monotone in max-len) with prefill scratch and the 1.5 GB reserve; tests: what one card would fit (int8 ~545k, bf16 ~277k at the 32.53 GB Level Zero reports; spec 27 section 3, within the stated reserve) and that max-len auto returns min(262,144, that) - 262,144 for int8 on one card - and `--max-len 262145` is refused by name; `--pp 2` likewise capped at 262,144. **Commit.**

### Task 3: the engine paths and B4 (box)
- [ ] **Step 1:** decode and prefill with int8 full-layer KV on the synthetic real-width checkpoint then the real one: F3 against the bf16-KV engine (greedy, tie-aware, near-miss rule) at depths 2k / 16k / 64k; F4: replay bitwise, splits at 64 bitwise, chunked = whole, restore at block ends bitwise (the int8 block is 29.8 MB per 2048 positions). **Step 2:** the snapshot size and `kv_form` key for the prefix store change with the KV form (a different `store_root`, never a mixed store).

### Task 4: gate K - int8 KV quality within 262,144 (box)
- [ ] **Step 1: the sets.** A passkey / needle set (a short fact planted in a long agentic transcript, retrieved by a question; 20 positions per depth) at depths 32k, 128k, 200k and 262,144 (no depth beyond it, D3); perplexity of a long transcript (the last 4k tokens' NLL given the full prefix) at the same depths; each for bf16 KV and int8 KV (the bf16 arm at 262,144 on `--pp 2` if one card does not fit it); prefill of the long prompts in 2048 chunks (hours at the larger depths: detached, one long run per depth ladder, the prefix cache reused between depths).
- [ ] **Step 2: the rule.** Gate K passes at depth d if int8 passkey accuracy >= bf16's minus one prompt of twenty and the perplexity ratio int8 / bf16 <= 1.02. It decides only the KV form, never the length: if it fails at some depth, the record goes to the operator (bf16 KV with a shorter one-card context, or `--pp 2`), nothing changes automatically. Record the table in `docs/probe-flint-long-context-2026-11-<dd>.md`. **Commit.**

### Task 5: speed at depth (box)
- [ ] Decode tok/s and prefill tok/s at the ladder up to 262,144, int8 vs bf16 KV, against spec 27 section 3's roofline table (119 t/s ceiling at 262k int8); achieved GB/s of the attention part vs spec 10's 540 GB/s; if a split-K retune (TGT) helps at 128k-262k with 14 layers of 4 kv heads, probe it (interleaved pairs) before changing a constant. **Commit** into the spec's "27f as built" section.

## Gates
- F0, F2 (int8 kernels), F3 (int8 vs bf16 KV), F4 (incl. int8 restores), **K** (the depth table and the rule above, up to 262,144), max-len auto = 262,144 on one card with int8 KV and the refusal above it, the plan numbers matching section 3.
