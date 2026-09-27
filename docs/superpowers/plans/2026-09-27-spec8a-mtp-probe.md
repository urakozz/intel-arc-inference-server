# Spec 8a - MTP probe: the reference head, acceptance, and the verify cost

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** measure what spec 8's decision rests on (P0): a CPU reference of the MTP head, its acceptance rate at draft depths 1-3 (greedy and sampled), the main model's verify step at M = 1..4, the draft-step cost, the GDN slot copy and the host round trip; then choose K and the commit mechanism, or stop.

**Architecture:** a Python reference head in `tools/oracle/` (CPU, in the oracle container) fed by the main model's hidden states from the existing oracle dump; M-variant timing through probe binaries that capture the existing decode list at M = 2..4 (the kernels already have the M loop; `gdn_step_M2` is already compiled for a test).

**Tech Stack:** Python 3 + torch (CPU, the oracle container), C++20 + Level Zero probes, CMake.

**Spec:** `docs/specs/2026-09-27-spec8-mtp-speculative-decoding-design.md` (§1, §3.2-3.4, §5 P0, §6 8a, the stopping rule).

## Global Constraints

- `tools/box.env` is untracked: never commit it or its contents. No `rm -rf`.
- Box: `tools/box.sh sync|build|test|run`, `-j44`, device 0; long runs detached (`tools/probe/detach.sh`), poll for `ALLDONE`.
- The oracle container: `tools/oracle/run_in_container.sh` (repo at /ws, HF cache at /hf, /tmp at /scratch). CPU runs of the 27B model are slow: size them (a 2k-token forward is minutes to an hour; say how long before launching anything over an hour).
- Timing: warm-up, median of 3, interleaved pairs for ratios.
- Numbers **measured** unless marked **derived**/**estimated**.
- Probe code only: no change to `src/` behaviour. Commit on branch `spec8a-mtp-probe`; no merge, no push.

## Review Focus

1. **Which hidden state feeds the head.** `h_t` before or after the main model's final norm, and the order of the concatenation (`embed ‖ hidden` vs `hidden ‖ embed`) in `mtp.fc`: take both from the reference implementation (vLLM's `qwen3_5_mtp.py` / `qwen3_next_mtp.py`, whichever matches this checkpoint's `config.json` architecture), cite the file and line in the doc, and prove it: the head's top-1 on real text must beat a wrong wiring by a wide margin (run the wrong wiring once as a control).
2. **The head's own attention and RoPE position:** the head's layer runs at position t + 1 (the drafted token's) or t; take it from the reference, and check by acceptance (the wrong one collapses it).
3. **Sampled acceptance** must be measured with opencode's actual `temperature`/`top_p`/`top_k` from the recorded log (spec 7 plan 7a, `tests/golden/opencode/session1/`); if that log is not there yet, use temperature 0.6, top_p 0.95, top_k 20 (the checkpoint's `generation_config.json`: read it and use its values) and say so.
4. **The M-variant timing must use the real captured list**, not kernels in isolation: the lm_head GEMV, argmax and every layer at M rows, since the win is weights read once for M rows.
5. The chained-draft acceptance at depths 2 and 3 uses the head's **own** output hidden as the next step's `h`, as it would run in the engine, not the main model's hidden (that would overstate acceptance).

---

### Task 1: the reference head

**Files:**
- Create: `tools/oracle/mtp_ref.py` (loads `mtp.*` from the checkpoint via the safetensors index; implements `fc`, the norms, the one transformer layer (a full-attention layer as the main model's, reuse `tools/oracle`'s layer code if it has one; otherwise port from the reference), `mtp.norm`, the shared `lm_head`), `tools/oracle/test_mtp_ref.py`
- Modify: `tools/oracle/dump.py` only if needed to save per-position final hidden states for a prompt (`--hidden-out FILE`)

- [ ] **Step 1:** find the reference (vLLM source on the box or in the oracle container: `find / -name "*mtp*.py" -path "*vllm*" 2>/dev/null`); write down the wiring (Review Focus 1, 2) in the doc skeleton `docs/probe-mtp-2026-09-27.md`.
- [ ] **Step 2:** `test_mtp_ref.py`: shapes and dtypes of the 15 tensors against `docs/03-models.md`; a 16-token smoke run finite.
- [ ] **Step 3:** implement; run on the `code` golden prompt plus its 32 greedy tokens (from the oracle golden set): per position, the head's argmax against the true next-next token. Also the wrong-wiring control.
- [ ] **Step 4: Commit** `oracle: a CPU reference of the MTP head (spec 8 P0)`.

### Task 2: acceptance rates

**Files:** Create `tools/oracle/mtp_accept.py`

For a token sequence (prompt + continuation), with the main model's per-position logits and hiddens from one teacher-forced CPU forward: depth 1 acceptance = the fraction of positions where the head's draft equals the main model's greedy token; depths 2 and 3 by chaining the head on its own hidden (Review Focus 5); sampled: the expected acceptance `Σ_x min(p(x), q(x))` per position after the same filters (exact, no sampling noise), chained the same way with the draft sampled from q (fixed seed).

- [ ] **Step 1:** run on the three golden prompts' greedy continuations (256 tokens each) and on up to 2000 tokens of assistant output from the recorded opencode log (take the `response_text` of the longest turns, tokenised, with their prompts' last 2k tokens as context; if the log is absent, use the A4 tool-call set's bf16 oracle outputs, `tools/toolcall/`). Size the CPU time first.
- [ ] **Step 2:** table: a at depths 1..3, greedy and sampled, per source; the derived expected tokens per iteration.
- [ ] **Step 3: Commit** `probe: MTP acceptance rates by depth (spec 8 P0)`.

### Task 3: the verify step at M = 1..4, the draft cost, the slot copy

**Files:**
- Create: `tools/probe/probe_verify_m.cc` (+ registration in `tools/probe/CMakeLists.txt`)
- Modify (probe-only compile of variants): `src/kernels/CMakeLists.txt` to also build M = 2, 3, 4 variants of every decode-list kernel, **behind a CMake option** `B70_DECODE_EXTRA_M` default OFF (so production builds are unchanged); `src/runtime/capture.cc` to take the list's M from a parameter instead of `kCapM`, default 1 (a pure refactor at M = 1: `replay_determinism_test` and the golden gates must be bitwise unchanged).

`probe_verify_m`: load the model (max_len 16384), prefill 4096 ids, capture the decode list at M, time 64 replays at M = 1, 2, 3, 4 (interleaved order, median of 3); report ms per step and ms per row. Draft cost, derived from measured pieces: time one M = 1 replay of `lm_head` alone (the existing profile/`profile_capture_test` machinery if it splits phases; otherwise a bf16 GEMV of the head's shapes: `fc [5120,10240]`, one layer's q/k/v/o/gate/up/down in bf16 at M = 1, and `lm_head`) and sum. Slot copy: a 151 MB device-to-device copy on an immediate list, median of 5. Host round trip: submit + fence of an empty-ish list plus a 4-byte readback, and a 5 MB logits readback.

- [ ] **Step 1:** the capture refactor, with the M = 1 gates run first (bitwise, `replay_determinism_test` + golden gates).
- [ ] **Step 2:** the probe; run detached on the idle box.
- [ ] **Step 3:** table in the doc; the derived speedup per K from Task 2's a and these costs (spec §1's formula with measured inputs).
- [ ] **Step 4: Commit** `probe: verify step at M = 1..4, draft and commit costs (spec 8 P0)`.

### Task 4: the decision

- [ ] **Step 1:** in the doc: the chosen K (highest derived speedup, ties to the smaller K), the commit mechanism (copy if the copy costs < 1 % of a step, else the `Control` index), and the stopping-rule verdict (greedy depth-1 acceptance < 0.5 on the opencode outputs, or M = 2 verify > 1.5 plain steps → stop).
- [ ] **Step 2:** `docs/README.md` line for the doc.
- [ ] **Step 3: Commit** `probe: MTP P0 verdict - K and the commit mechanism (spec 8)`.

**Gate for the plan:** the doc committed with every P0 number, the M = 1 capture refactor bitwise-neutral (full suite green), the verdict. Hand back with the tables.
