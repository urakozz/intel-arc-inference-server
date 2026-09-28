# Spec 8 - MTP speculative decoding, greedy and sampled

**Status:** design, 2026-09-27, for operator review.

**Order, set by the operator:** spec 5 (int8 prefill linears) is the
foundation; spec 6 (flash attention, 128k) and spec 7 (prefix caching) are
built on it. This spec is the third feature on top. It is designed in
parallel with spec 7; its code starts after spec 7's plan 7b (engine
snapshots) has merged, because both change `runtime::Engine`'s state
handling.

Every number is **measured** unless marked **derived** or **estimated**.

## 1. Why

Decode is bandwidth-bound: one token reads every weight once. It measures
29.45 t/s at shallow depth (vLLM 31.01), 20.6 t/s at 32k, 10.2 t/s at 128k
(spec 6 §8). An agentic turn generates hundreds to thousands of tokens
(reasoning plus tool calls), so decode is most of a turn's wall time once
spec 7 removes the re-prefill.

The checkpoint carries a **multi-token-prediction (MTP) head**
(`docs/03-models.md`): `mtp.*`, 15 tensors, all bf16, 0.849 GB: `mtp.fc`
`[5120, 10240]` over `pre_fc_norm_embedding(embed(t+1))` ‖
`pre_fc_norm_hidden(h_t)`, one full-attention transformer layer at the main
model's shapes, `mtp.norm`; it shares `embed_tokens` and `lm_head` with the
main model. The loader skips it today (`docs/13-loader.md`).

Speculative decoding drafts k tokens with the head, then **verifies** the
pending token plus the k drafts in one main-model step of M = k + 1 rows.
The weights are read once for all M rows, so a verify step costs little
more than a single-token step, and every accepted draft is a token for free.

Phase 0 built for this: every decode kernel has an `M ∈ [1, 8]` loop over
**consecutive positions** (`docs/specs/2026-08-22-phase0-decode-core-design.md`
§9: `gdn_step<M>` advances the recurrent state token by token, `attn_decode`
applies the causal bound `position <= pos + m`), `DecodeScratch::kM` = 8 is
allocated, and only M = 1 is compiled (`capture.cc`, `kCapM`).

**Estimated gain** (derived, not measured): a draft step reads the head plus
`lm_head`, 0.85 + 2.54 GB, about 20 % of a main step's bytes; a verify step
at M = k + 1 is assumed at 1.05 to 1.15 main steps. With per-token
acceptance a = 0.8, expected tokens per iteration (1 - a^(k+1)) / (1 - a):
k = 1: 1.8 tokens for ~1.3 steps, 1.4x; k = 2: 2.4 for ~1.5, 1.6x; k = 3:
3.0 for ~1.7, 1.7x. P0 measures a, the verify cost and the draft cost.

## 2. The decision

**Approach 1 (operator, 2026-09-27): two captured lists, acceptance decided
on the host, greedy and sampled.** Rejected in the design discussion:

- **Everything on the card** (draft, verify, accept, commit in one list):
  saves one host round trip per iteration (~0.5 ms of ~60 ms, derived) but
  needs device-side top-k/top-p sampling over 248k logits. Possible later if
  P0 shows the host round trip matters.
- **k = 1 fused into the verify list** (vLLM's shape): simplest, capped at
  two tokens per iteration.

**Sampling is in scope** (operator, 2026-09-27): opencode sends a
temperature, and greedy-only MTP would then never engage.

## 3. Design

### 3.1 Loading the head

`loader::load` keeps `mtp.*` (bf16, from `model_extra_tensors.safetensors`
as the index names it) behind `--mtp`; without it, nothing changes. The
head's attention layer gets its own KV cache, `[max_len][4][256]` bf16 K and
V: 4 MiB per 1k positions, 512 MiB at 131072 (derived). The load report
adds both lines.

### 3.2 Prefill fills the head's KV

The head attends over its own KV at earlier positions, so prefill must run
the head over every prompt position: for position t, input
`embed(id[t+1])` ‖ `h_t` (the main model's last hidden at t, before the
final norm, per the reference implementation checked in P0). The chunk's
last position has no `id[t+1]` yet: it is filled by the first verify step.
One extra layer plus `fc` per chunk: under 2 % of prefill work (derived).

### 3.3 One iteration

State at the start: `pos` = n, the pending token `x_n` (sampled, not yet
fed), the main hidden `h_{n-1}` and the head's state.

1. **Draft list** (captured once, M = 1, replayed k times): the head on
   `(x, h)` gives draft logits q_i and the next draft `d_i` (argmax when
   greedy; when sampling, the host samples from q_i and writes `d_i` back,
   the ingest protocol `EngineAdapter` uses today). Its output hidden is
   the next draft step's `h`.
2. **Verify list** (captured once, M = k + 1): the main model on
   `x_n, d_1 .. d_k` at positions `n .. n + k`. It writes logits p_0 .. p_k,
   the main hidden of every row, and **a GDN state and conv ring snapshot
   after each row** (see 3.4). KV is written at `n .. n + k` as usual.
3. **Accept, on the host.** Greedy: accept d_i while `d_i == argmax p_{i-1}`;
   the first mismatch position takes `argmax p` as its token; all accepted
   plus one bonus token. Sampling (Leviathan et al. 2023): after the same
   top-k / top-p / temperature filter on both p and q, accept d_i with
   probability `min(1, p(d_i) / q(d_i))`; on the first rejection sample from
   `normalise(max(0, p - q))`; if all k are accepted, sample the bonus token
   from p_k. The output distribution equals plain sampling from p. The host
   reads back the M argmax ids always, and the (2k + 1) logit rows only when
   sampling (~5 MB at k = 3, derived).
4. **Commit** (a small list): `pos = n + j + 1` for j accepted drafts; the
   GDN state and conv ring of row j become the live state; the head's KV
   and hidden for the accepted rows are kept. KV beyond the new `pos` is
   stale and never read (the causal bound).

### 3.4 Rolling back the recurrent state

`gdn_step<M>` keeps the state in registers across the M loop and writes it
once. For rollback it also writes the state after each row m into slot m
of `gdn_state_spec [M][48][48][128][128]` fp32 (151 MB per slot, 604 MB at
M = 4, derived). Commit makes slot j live either by copying it into
`gdn_state` (151 MB, ~0.3 ms at device bandwidth, derived) or by an index
the kernels read from `Control`; P0 measures the copy and 8b chooses. The
conv ring needs no copy: it is a 16-deep ring indexed by position (depth
16 >= M + 3, phase 0 §9.4). After a commit to `pos = n + j + 1`, the next
step reads history slots for positions `n + j - 2 .. n + j`, all accepted;
the slots of rejected positions `n + j + 1 .. n + k` are rewritten by that
step before anything reads them as history (checked by M2).

### 3.5 Interaction with prefix caching (spec 7)

The head's KV is part of the session state: spec 7's KV blocks and
snapshots carry it (17 KV layers instead of 16 when `--mtp` is on), and its
restore gates run with MTP on. This is the case vLLM gets wrong
(`docs/BENCHMARKS.md`: prefix caching is silently disabled under MTP).

### 3.6 Server

`b70_serve --mtp K` (K in 1..3, 0 = off, default chosen by P0). The
generation loop in `server.cc` calls `EngineIface::step_many()`, which runs
one iteration and returns 1 to K + 1 ids; streaming, stop strings, tool-call
parsing and `max_tokens` see them one at a time (tokens past a stop or
`max_tokens` are dropped and `pos` is rewound to the last kept token by the
same commit). The seed makes a sampled MTP run reproducible, but not equal to
a sampled non-MTP run with the same seed (the random draws differ).

## 4. Correctness gates

- **M1, the head.** Head logits against a CPU reference of the MTP head
  (added to `tools/oracle/`, following the reference implementation
  identified in P0) on the golden prompts: per-position cosine >= 0.999
  and the reference's argmax in the engine's top 2.
- **M2, verify at M = k + 1.** For k = 1..3: each row's logits against M = 1
  decode of the same tokens, cosine >= 0.99999; each row's GDN state slot
  against the M = 1 state after the same token, max abs error recorded,
  cosine >= 0.99999; after a commit with j < k accepted, 64 further M = 1
  greedy tokens equal those of a run that never saw the rejected drafts.
- **M3, greedy is lossless.** Greedy with MTP against greedy without, 256
  tokens on the golden prompts and on the A4 tool-call set: identical where
  no near-tie separates them; divergences only at near-ties under the golden
  rule (M = k + 1 rows are not bit-identical to M = 1, as chunked prefill is
  not to M = 1).
- **M4, sampling is exact.** Host unit tests: the acceptance rule on
  synthetic p and q (1e6 draws, chi-square p-value >= 0.01 against p), with
  and without top-k/top-p; a seeded end-to-end run is reproducible bitwise.
- **M5.** Determinism and replay bitwise with MTP on; every existing
  registration unchanged with MTP off; spec 7's C2 passes with MTP on.

## 5. Speed bars

Idle box, device 0, interleaved pairs, median of 3.

- **P0, first.** Acceptance rate a at draft depths 1, 2, 3 on the outputs
  of the recorded opencode session (spec 7 plan 7a) and on the golden
  prompts, measured with the CPU reference head, greedy and at the
  temperature opencode sends. Verify-step time at M = 1, 2, 3, 4 (the
  M-variants compiled, the main list captured at each). Draft-step time.
  The GDN slot copy. The host round trip.
- **D1.** Greedy decode at 4k depth with `--mtp K*`: **>= 1.4x** plain
  decode tokens per second, on the golden prompts and on the opencode
  replay; recorded at 32k and 64k.
- **D2.** Sampled decode at opencode's temperature: recorded, and >= 1.25x.
- **D3.** `--mtp 0`: decode and prefill within 1 % of today.

## 6. Stages

- **8a, probe:** P0; the CPU reference head and M1's reference numbers;
  choose K and the commit mechanism.
- **8b, engine:** the loader, the head's kernels and KV, prefill filling it,
  the draft, verify and commit lists, per-row GDN slots; M1, M2, M5 (engine
  part); D3.
- **8c, server:** acceptance (greedy and sampled) on the host, `step_many`,
  `--mtp`, spec 7 interplay; M3, M4, M5; D1, D2; the record.

**Stopping rule.** If P0 finds greedy acceptance at depth 1 below 0.5 on the
opencode outputs, or a verify step at M = 2 above 1.5 plain steps, the gain
is under 1.2x (derived); stop and record for the operator's call before 8b.

## 7. Out of scope

- Draft trees (several candidates per position), EAGLE-style heads.
- Device-side acceptance and sampling (§2).
- Quantising the head or pruning `lm_head`'s vocabulary for drafts (a later
  lever: `lm_head` is 75 % of a draft step's bytes, derived).
- Batching several requests.

## 8. Amendment: what plan 8b built (2026-09-28)

Numbers: `docs/probe-mtp-2026-09-27.md` §6. Each item overrides the text above where
they differ.

- **A1, the head's wiring (§3.2, from P0).** The head's hidden input is the main model's
  hidden **after** the final norm (the engine's `b.x`, what `lm_head` reads), not before;
  fc's input is `cat(pre_fc_norm_embedding(embed(x[t+1])), pre_fc_norm_hidden(h_t))`,
  embedding first; the head's layer runs at position t for the pair (h_t, x[t+1]), +1 per
  chained draft, and chained drafts take the head's own post-`mtp.norm` hidden.
- **A2, K is runtime-selectable 1..3.** P0's K = 3 is provisional until the opencode log
  exists, so the engine captures verify lists at M = 1..4 and three draft lists (one per
  draft index, each writing its own logits row), and M2 is gated at every K.
- **A3, the prefill gate compares like inputs.** Plan 8b asked for the head's KV after
  prefill against M = 1 decode fills at cosine >= 0.9999 per row. That comparison carries the
  main model's own l0-int8-prefill-vs-decode difference (median 0.99971; 15 of 4095 rows
  below 0.9, the main hidden at cosine 0.27 at one of them), which predates spec 8. The gate
  is the M = 1 decode head on the prefill's own hidden rows (worst 0.9999965); the
  end-to-end comparison is recorded.
- **A4, the commit mechanism is the Control index (§3.4).** The 151 MB copy measured 1.66 %
  of a step in P0. `gdn_state` is slot 0 and `MtpBuffers::gdn_spec` holds slots 1..3
  (453 MB, not [M] slots of 604 MB); `Control::gdn_live` names the live slot; verify row m
  writes slot (live + m) % 4, `commit(j)` sets live = (live + j) % 4, and M = 1 with slots is
  in place. Paths that read `gdn_state` directly (prefill, `load_state`) first copy the live
  slot into slot 0, once, only when it is not already 0.
- **A5, the head's KV bookkeeping lives in the verify list (§3.3 step 4).** The verify
  list's tail is a K/V-only fill of the head over its M rows at positions n-1 .. n+M-2, from
  (h_{n-1}, x_n), (h_n, d_1), ... - the main hidden, so after commit(j) every head-KV row
  below the new pos - 1 comes from the main model, and the chained drafts' own rows are
  overwritten before they are read. verify(0) + commit(0) is the plain step with MTP on, so
  the invariant holds under any mix of plain and speculative steps. Prefill does the same
  per chunk (§3.2) and computes only K/V (fc, input norm, the k||v columns, attn_prep).
  The head runs on its own control block, so its argmax chains the drafts with the main
  model's kernels unchanged. Draft ids land in `Control::cur_token[1..k]` (the verify
  inputs), verify ids in `Control::out_token[0..k]`.
- **A6, M = k + 1 rows are bitwise equal to M = 1** (M2: logits and GDN slots, every K), so
  §4 M3's caveat does not arise on decode; greedy MTP should be exactly lossless (8c's M3).
- **A7, the verify step costs 1.17 / 1.52 / 1.74 plain steps at M = 2 / 3 / 4** in
  production (P0: 1.171 / 1.519 / 1.743), not §1's assumed 1.05-1.15. verify(0) costs 1.3 %
  over the plain list (the head's K/V fill). Two follow-up levers: the int4 GEMV at M = 3/4
  and `attn_decode`'s per-row walk of the KV.
- **A8, spec 7 interplay (§3.5).** With the head, `save_state`/`load_state` carry the LIVE GDN
  slot, the conv ring and the head's input hidden h_{pos-1} (+10 240 B); `save_kv`/`load_kv`
  carry 17 KV layers, the head's last. Spec 7b's C1 case A passes with MTP on, including
  speculative iterations after the restore.
- **A9, scope limits.** The head loads only as the published checkpoint's 15 bf16 tensors
  (the RTN checkpoint's 29 are refused by name); the MTP prefill runs on the L0 backends
  only; MTP lists are compiled at max_len 16384 (`B70_MTP`, default ON, adds binaries only).
  Device memory with the head: +0.849 GB weights, +0.523 GB `MtpBuffers`, +21 MB prefill
  hidden rows (lazy).

## 9. Amendment: what plan 8c measured (2026-09-28)

Numbers: `docs/BENCHMARKS.md`, "MTP speculative decoding (spec 8)". The box was shared
throughout, so every speed row is grade iterate. All runs are at max_len 16384.

- **A10, the server (§3.6).** Acceptance runs on the host in `src/server/spec_accept`.
  - The p/q filter is `server::filter_probs`, the one `sample()` now calls.
  - `EngineIface` gains `mtp_k`, `step_many` and `truncate_to`. Their defaults keep
    `--mtp 0` exactly today's loop.
  - `EngineAdapter` defers `commit(j)` to its next engine call. When a stop, EOS or
    `max_tokens` lands inside a burst, `truncate_to` commits fewer ids. The id that ends
    the request stays consumed, as it does with `step()`, so the session holds exactly
    what plain steps would have left.
  - Sampled drafts use a host pick hook on `Engine::draft`. `--mtp` defaults to 0.
- **A11, M3 is exact identity.** Greedy with `--mtp K` against `--mtp 0`: 256 ids on 39
  prompts at K = 1..3, on l0-int8 and on l0. **0 divergences.**
  - M4 passes: host chi-square p-values 0.10-0.62, and the seeded sampled run is
    bitwise.
  - M5 passes for the repeated run.
  - M5's spec 7 C2 with MTP on: after the 7c merge, `prefix_gpu_mtp_test` runs
    `prefix_gpu_test` on l0-int8 with the head loaded and `EngineAdapter` at `--mtp 3`
    (argv[6] = K); sequences a-g pass (2026-09-28).
- **A12, D1 misses on the golden prompts and passes on the agentic proxy.**
  - At 4k, greedy golden geomeans are 1.297x / 1.244x / 1.186x at K = 1 / 2 / 3.
    Prose at K = 3 falls to 0.970x.
  - The four P0 tool-call scenarios give 1.428x / 1.512x / 1.633x.
  - At 12k the golden rows fall to 1.264x / 1.179x / 1.099x.
  - The cause is the verify cost (A7) on low-acceptance text.
  - The opencode replay, which the bar names, is pending the log.
  - 32k/64k cannot run until the MTP lists are compiled beyond 16384.
- **A13, D2 passes.** Sampled at T 1.0, top-p 0.95, top-k 20 (the checkpoint's defaults;
  the log's own temperature is pending):
  - golden: 1.330x / 1.203x / 1.370x
  - tool-call: 1.410x / 1.481x / 1.543x
- **A14, choosing K is workload-dependent.**
  - The rule is the same as P0's: pick K by the opencode log's acceptance once it
    exists.
  - Agentic output that copies context favours K = 3. Free text favours K = 1.
  - A per-request adaptive K (drop K after rejections) is the cheap follow-up. The
    M = 3/4 int4 GEMV lever (A7) is the structural one.
