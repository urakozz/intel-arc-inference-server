# Spec 8c - MTP in the server: acceptance, greedy and sampled

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `b70_serve --mtp K` generates with speculative decoding: host-side acceptance (greedy and lossless sampling), `EngineIface::step_many()`, correct stop/`max_tokens`/streaming, spec 7's prefix cache working with MTP on; gates M3, M4, M5 and speed bars D1, D2; the record.

**Architecture:** a pure host module `src/server/spec_accept.{h,cc}` implements the acceptance rule over logits rows (unit-testable without a GPU); `EngineAdapter::step_many` drives plan 8b's `draft` / `verify` / `commit`, reading back ids always and logits rows only when sampling; `server.cc`'s loop consumes 1..K+1 ids per call.

**Tech Stack:** C++20, nlohmann::json, Level Zero readbacks, Python 3 for the replay comparison.

**Spec:** `docs/specs/2026-09-27-spec8-mtp-speculative-decoding-design.md` (§3.3 step 3, §3.5, §3.6, §4 M3-M5, §5 D1-D2). Needs plan 8b (`Engine::draft/verify/commit`, `mtp_logits_device`, `verify_logits_device`) and spec 7 plan 7c merged (the prefix cache, `tools/prefix/replay_log.py`, `tests/golden/opencode/session1/`).

## Global Constraints

- `tools/box.env` is untracked: never commit it or its contents. No `rm -rf`.
- Box: `tools/box.sh sync|build|test|run`, `-j44`, device 0; long runs detached, poll for `ALLDONE`.
- `--mtp 0` is exactly today's server.
- The filter applied to p and q is the one `sample()` in `src/cli/serve_adapters.h` applies (temperature, top-k, top-p, `vocab_used` mask): factor it into a shared function, do not re-implement it.
- Numbers **measured** unless marked **derived**/**estimated**.
- Commit on branch `spec8c-mtp-server`; no merge, no push.

## Review Focus

1. **A stop string or EOS inside the accepted run:** tokens after it must not be emitted, and the engine must be committed only up to the last kept token (so the next request's prefix-cache resident ids are right). Test with the mock engine scripting an EOS at accepted position 2 of 4.
2. **`max_tokens` reached mid-run:** same truncation and commit. Test.
3. **Top-p leaving a single candidate** (p or q one-hot after filtering) and **q(d) = 0 for a draft sampled from q** (impossible by construction; assert) - the rule must not divide by zero; the residual `max(0, p - q)` all zero (p == q) cannot happen after a rejection, but guard by sampling from p.
4. **Seeded reproducibility across `step_many` sizes:** the RNG draws in a fixed order per iteration (draft samples, then acceptance uniforms, then the residual/bonus sample), so the same seed and the same device logits give the same output. Test on the host with scripted logits.
5. **Streaming tool calls with bursts of up to K + 1 tokens:** the tool-call `OutputStream` (spec 7a) receives the pieces one token at a time in order. Test with the mock.

---

### Task 1: the acceptance rule (host only)

**Files:** Create `src/server/spec_accept.h`, `src/server/spec_accept.cc`; modify `src/cli/serve_adapters.h` (factor the filter out of `sample()` into `filter_probs(logits, vocab_used, Sampling) -> sparse probs`, used by both); test `tests/server/spec_accept_test.cc` (host-only).

**Interfaces (produces):**
```cpp
namespace server {
struct AcceptResult { uint32_t accepted; uint32_t next_token; };  // j drafts kept, then one
                                                                 // token from p (residual/bonus)
AcceptResult accept_greedy(const uint32_t* verify_argmax, const uint32_t* drafts, uint32_t k);
AcceptResult accept_sampled(const float* p_rows /*[k+1][V]*/, const float* q_rows /*[k][V]*/,
                            const uint32_t* drafts, uint32_t k, uint32_t vocab_used,
                            const Sampling& s, std::mt19937_64& rng);
uint32_t sample_draft(const float* q_row, uint32_t vocab_used, const Sampling& s,
                      std::mt19937_64& rng);
}
```
Tests (M4): greedy cases (all accepted, first rejected, middle rejected); sampled: synthetic p, q over a 50-token vocab, 1e6 draws of the full procedure, chi-square of the emitted first token against filtered p, p-value >= 0.01 (fixed seed, so the test is deterministic), with and without top-k/top-p; Review Focus 3 and 4.

- [ ] Steps: failing tests → implement → PASS → commit `server: speculative acceptance, greedy and lossless sampled (spec 8 §3.3, M4)`.

### Task 2: `step_many` and the loop

**Files:** Modify `src/server/deps.h` (`EngineIface::step_many(const Sampling&) -> std::vector<uint32_t>` and `truncate_to(uint32_t pos)` for Review Focus 1-2, `mtp_k()`), `src/cli/serve_adapters.h` (`EngineAdapter`: greedy: `draft(K)`, `verify(K)`, read `verify_ids` + drafts (4 x (2K+1) bytes), `accept_greedy`, `commit`; sampled: `draft` one step at a time with `sample_draft` writing each draft back, `verify`, read back logits rows, `accept_sampled`, `commit`), `src/server/server.cc` (consume the vector token by token through the existing EOS/stop/max_tokens/stream logic; on early stop, `truncate_to`), `src/cli/b70_serve.cc` (`--mtp K`, default from plan 8a's verdict; `--mtp` requires the head loaded), `tests/server/mock.h` (a scripted `step_many`).

Test `tests/server/mtp_server_test.cc` (mock): Review Focus 1, 2, 5; `--mtp 0` responses byte-identical to today's; `usage.completion_tokens` counts emitted tokens only.

- [ ] Steps: failing tests → implement → PASS (all server tests) → commit `server: --mtp K, step_many, truncation at stops (spec 8 §3.6)`.

### Task 3: M3 and M5 on the card

**Files:** Create `tests/server/mtp_gpu_test.cc` (checkpoint label, real engine, no HTTP).

- M3: greedy 256 tokens with `--mtp K` against `--mtp 0` on the three golden prompts and the 36 A4 tool-call prompts (`tools/toolcall/`); identical, or diverging only at a near-tie under the golden rule (reuse the comparison helper from `tests/prefill/prefill_gate_test.cc`); report the count of divergences and the acceptance rate per prompt.
- M5: the same greedy run twice, bitwise; spec 7's C2 (`tests/server/prefix_gpu_test.cc`) with MTP on.

- [ ] Steps: write → run on the box (`l0-int8`; `l0` for M3 as well) → full suite green → commit `tests: MTP greedy is lossless, and prefix caching holds (spec 8 M3, M5)`.

### Task 4: D1, D2 and the record

- [ ] **Step 1: D1:** greedy decode tokens/s, `--mtp K` against `--mtp 0`, interleaved pairs, at 4k depth on the golden prompts and on the opencode replay (`tools/prefix/replay_log.py` with a `--decode-tps` report); bar >= 1.4x; recorded at 32k and 64k.
- [ ] **Step 2: D2:** sampled at opencode's temperature (from the log's requests): recorded, bar >= 1.25x.
- [ ] **Step 3:** the record: `docs/BENCHMARKS.md` section "MTP speculative decoding (spec 8)" (P0 summary, acceptance, D1/D2 tables); spec 8 amendment §8 with every result; `README.md` headline decode row updated if D1 passes (decode t/s with `--mtp K`, and the plain number beside it); `docs/03-models.md` and `docs/13-loader.md` lines that say the head is not loaded.
- [ ] **Step 4: Commit** `spec 8: MTP results, D1/D2 (the record)`.

**Gate for the plan:** M3, M4, M5 passing, full suite green, D1/D2 measured (a miss is recorded and handed back), the record committed.
