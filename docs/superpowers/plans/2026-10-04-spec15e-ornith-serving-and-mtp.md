# Spec 15e - Ornith in the server: template, tool calls, MoE MTP, the record

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `b70-serve ornith-…` serves chat with the checkpoint's template, reasoning and tool calls; the MoE MTP head drafts through spec 8's machinery; A4, prefix caching, determinism and long-context gates (R4); the comparison rows; the record.

**Architecture:** the server, prefix cache (spec 7), MTP (spec 8), int8 head (spec 9) and decode attention v2 (spec 10) run unchanged on the descriptor; this plan adds what Ornith needs: its template / tokenizer check, the MTP head's MoE layer (15c's kernels at M = 1) and spec 8's verify at M = K + 1 over MoE layers (up to 8 x (K + 1) distinct experts), measured.

**Tech Stack:** C++17, OpenCL C, Level Zero, Python (A4, oracle), uvx llama-benchy.

**Spec:** `docs/superpowers/specs/2026-10-04-spec15-ornith-moe-design.md` (§4.5, §5 R4, §6 recorded rows, §7 15e). Needs 15c and 15d merged, 15a's MTP reference dump (`oracle-out-ornith-mtp/`).

**Status (2026-10-05): Tasks 1 and 2 written blind on the Mac** (branch `spec15e-ornith-serving`;
as built: spec 15 §12). Every card-side step - Task 2's M1 / M2 / acceptance / verify cost, the
server's greedy chat against `b70-decode`, and all of Task 3 - is box work: box-validation-queue
row 16. Mac-side: `tools/mac_check.sh --base main --kernels` green (host tests incl.
`template_ornith_test`, `ornith_server_test`, `ornith_mtp_head_test`, `ornith_mtp_names_test`;
every C++ source against the Level Zero headers; 80 added kernel command lines and none moved;
every variant through clang; `moe.cl` at M = 4 bitwise M = 1's per row on the Mac's GPU,
indicative). Deviations from the text below, each recorded in §12:
- **Template**: Ornith's is Qwen3.5's, not Qwen3.8's; no parser addition (the same Qwen XML tool
  calls). Rendered byte-identical to transformers on six lists; the tokenizer.json diff found its
  added tokens stop at `</think>` (248070 ids, the descriptor masks from 248077).
- **The MTP head's experts ship bf16**: quantised at load (RTN int4 g64 sym) so 15c's kernels run
  them; no new MoE kernel. `moe.cl` needed no change for M > 1 - only new `_M2..4` binaries.
- **No default K for Ornith**: `--mtp auto` keeps Qwen3.8's cost table and says so (Review Focus
  2 is unmeasured).
- **The short greedy chat through the server** is the existing `golden_server_test` pointed at
  the Ornith checkpoint by hand (it does not skip without one, so it is not registered for Ornith).

## Global Constraints

- Branch `spec15e-ornith-serving` from main; box tree `~/b70-inference-server-spec15e`. Copy `tools/box.env` if missing; never commit it. Symlink `oracle-out*` dirs.
- Every GPU command under `flock ~/b70-gpu.lock`, detached, polled. `uptime` with timings.
- R0 holds. No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **Template and tokenizer:** diff Ornith's `chat_template.jinja` and `tokenizer.json` against Qwen3.8's (Agnes's template was byte-identical but its tokenizer differed); the tool-call format against `src/server/toolcall`; a render test against HF's `apply_chat_template` on 5 message lists.
2. **MTP verify over MoE:** the verify step at M = K + 1 routes each row independently; measure its cost at K = 1..3 (it may grow faster than the dense model's 1.17 / 1.52 / 1.74 steps) before choosing a default K.
3. **Spec 8 M2 on Ornith:** verify rows bitwise equal to M = 1 decode (the MoE kernels must be row-independent for this to hold).
4. **Long context:** passkey at 120k and 250k (Ornith's KV is 20 KiB per position; 262144 fits).
5. **The comparison rows** use the operator's llama-benchy flags and state the card, clocks if known, quant and head form on every row.

---

### Task 1: template, tokenizer, tool calls

- [ ] Review Focus 1; a parser addition only if the format differs (with spec 7a's streaming tests); a short greedy chat through the server equal to `b70-decode` on the same ids. **Commit** `server: Ornith chat template and tool calls (spec 15e)`.

### Task 2: the MoE MTP head

- [ ] The MTP head's MoE layer through 15c's kernels; M1 (head against `oracle-out-ornith-mtp/`), M2 (verify rows bitwise), the verify cost at K = 1..3 (Review Focus 2); acceptance on the golden and A4 prompts; D1-style speed rows. **Commit** `mtp: Ornith's MoE draft head (spec 15e)`.

### Task 3: R4 and the record

- [ ] **Step 1:** A4 on Ornith against bf16 Ornith (15a's reference), prefix caching (spec 7's C2 on Ornith), determinism and replay, passkey at 120k and 250k.
- [ ] **Step 2:** the comparison rows: `b70-serve` with llama-benchy (the operator's flags: pp4096, tg256, depth 1, `--no-cache --exact-tg --latency-mode generation`), against vLLM if it serves Ornith on XPU and llama.cpp SYCL with an Ornith GGUF, same card; prefix-caching rows at 4k / 16k / 32k.
- [ ] **Step 3:** `docs/BENCHMARKS.md` "Ornith 1.5 MoE (spec 15)" completed; spec 15 amendment §10 with every gate and number; `README.md` (the scope section names the MoE model; an Ornith row); `docs/03-models.md`, `docs/13-loader.md`.
- [ ] **Commit** `spec 15: Ornith served - MTP, A4, long context, comparison rows (the record)`.

**Gate for the plan:** R4 green, MTP gates green, the comparison rows recorded, R0 green, the record committed.
