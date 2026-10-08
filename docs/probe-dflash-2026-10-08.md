# Probe: DFlash P0 on Qwen3.8 - acceptance (spec 19a, Tasks 2-3), 2026-10-08

`z-lab/Qwen3.8-27B-DFlash2` drafting from the target's taps, teacher-forced on the target's
own greedy continuations, on the Mac CPU (`tools/oracle/dflash_p0.sh`, container
`agnes-ref-img`, Qwen3.8-27B bf16 snapshot `1d4bf0f`), 2026-10-06 12:40 to 2026-10-08 02:54.
Raw tables: `oracle-out-19a/accept/summary.md` (gitignored); the per-anchor histograms are in
`oracle-out-19a/accept/accept.<arm>.json`.

**What this answers:** how many tokens one verify of a K-token DFlash block yields (E_K, the
accepted drafts + 1), per corpus and per drafter format. **What it does not:** the cost of that
verify at M = K + 1 = 5..8 rows and of the drafter's own forward on the card - plan 19a Task 4,
box only. The formal go / no-go (Task 5, spec 19 §7's stopping rule) needs those costs; §3 is a
provisional projection from spec 8's measured M = 2..4 verify costs and an extrapolation.

## 1. Acceptance (E_K, tokens per verify step)

| corpus | anchors | E_1 | E_2 | E_3 | E_4 | E_5 | E_6 | E_7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A4 tool calls (34 scenarios, 2613 ids) | 2409 | 1.99 | 2.96 | 3.88 | 4.77 | 5.63 | 6.44 | **7.21** |
| golden code | 26 | 1.88 | 2.88 | 3.60 | 4.36 | 5.07 | 5.82 | 6.15 |
| golden prose | 26 | 1.73 | 2.13 | 2.19 | 2.19 | 2.19 | 2.19 | 2.20 |
| golden cjk | 26 | 1.38 | 1.50 | 1.50 | 1.58 | 1.58 | 1.58 | 1.54 |
| all | 2487 | 1.98 | 2.93 | 3.84 | 4.71 | 5.55 | 6.34 | 7.09 |

(bf16 drafter, full vocabulary. Per-depth acceptance at K = 7 on A4: 0.99 0.98 0.96 0.95 0.96
0.95 0.93 - it barely falls with depth, so E_K is still rising at K = 7, the largest the tool
measured.)

- **Agentic tool-call output is where DFlash shines:** a 7-token block is almost always
  accepted whole (7.21 of 8 possible). These continuations restate code and names from the
  prompt, so they are unusually predictable; the number is an upper bound for long sessions
  until the box records fresh A4 outputs.
- **Code (6.15 at K = 7) also pays; prose saturates at ~2.2 by K = 3; cjk at ~1.5.**
- The golden sets are small (26 anchors each); the prose / reasoning number rests on them
  (plan 19a's status: no larger prose set with recorded greedy output exists on the Mac).

## 2. Drafter format and draft vocabulary (all corpora, E_K)

| arm | E_1 | E_3 | E_5 | E_7 |
|---|---:|---:|---:|---:|
| bf16 | 1.98 | 3.84 | 5.55 | 7.09 |
| int8 (per-channel RTN, simulated) | 1.98 | 3.84 | 5.55 | 7.10 |
| int8h (int8 drafter + int8 head) | 1.98 | 3.84 | 5.55 | 7.09 |
| w4a16 (`syvai/Qwen3.8-27B-DFlash2-W4A16`) | 1.98 | 3.84 | 5.55 | 7.10 |
| bf16, V′ 32k ranked (`-v32k-r`) | 1.98 | 3.84 | 5.56 | 7.11 |
| bf16, V′ 64k ranked | 1.98 | 3.84 | 5.55 | 7.10 |
| bf16, V′ 128k ranked | 1.98 | 3.84 | 5.55 | 7.09 |
| bf16, V′ 64k by id (unranked) | 1.96 | 3.75 | 5.38 | 6.83 |
| bf16, V′ 32k by id (unranked) | 1.90 | 3.42 | 4.67 | 5.71 |

- **Quantising the drafter costs nothing measurable:** int8 and 4-bit equal bf16 to 0.01.
  The engine's drafter should therefore be our own format - AutoRound W4A16 g64 symmetric,
  re-quantised by us (the operator's one-format rule; syvai's checkpoint proves the 4-bit
  acceptance, it is not the format we ship) - or int8 if a W4A16 g64 drafter is not ready.
- **A ranked 32k draft vocabulary looks free** (7.11 vs 7.09) and would shrink the draft head
  ~7.6x (32768 of 248320 rows); by id it loses up to 1.4 tokens at K = 7. **Caveat: the ranking
  is in-sample** - `oracle-out-19a/draft_vocab.ranked.ids` was ranked (`tools/draft_vocab/rank.py`)
  on the same A4 and golden sources it is scored on, so the `-r` rows are an upper bound for a
  ranked V′. A ranking from other traffic (e.g. the opencode request logs) must be re-scored
  before the 32k head is the default. cjk: unranked 32k / 64k drop it to 1.19 (ranked: 1.62).

## 3. Provisional projection (derived - the verify and draft costs are not measured)

Costs in plain decode steps (M = 1 = 1.00). Measured (spec 8 P0, docs/probe-mtp-2026-09-27.md):
verify M = 2 / 3 / 4 = 1.17 / 1.52 / 1.74, an MTP draft step 0.185. **Assumed:** verify grows
by M = 3 -> 4's slope (0.22 per row) to M = 8 = 2.62; one DFlash block (the 3.5 GB bf16
drafter at int8, ~1.75 GB, plus K rows of the ranked 32k head) ~0.2 plain steps. MTP's side:
`--mtp auto` at the measured MTP acceptance (tool-call ~0.97 -> K = 3; golden 0.82 -> K = 3;
prose 0.44 -> K = 1), E_K = 1 + a + a^2 + ... (one alpha for all depths, as `--mtp auto`
does - it over-rates MTP's deep drafts slightly, which favours MTP here).

| corpus | DFlash best K | DFlash tokens / step-time | MTP auto | DFlash vs MTP |
|---|---|---:|---:|---:|
| A4 tool calls | K = 7: 7.21 / (2.62 + 0.2) | 2.56 | K = 3: 3.82 / 2.30 = 1.66 | **+54 %** |
| golden code | K = 7: 6.15 / 2.82 | 2.18 | K = 3: 3.04 / 2.30 = 1.32 | **+65 %** |
| golden prose | K = 2: 2.13 / (1.52 + 0.2) | 1.24 | K = 1: 1.44 / 1.36 = 1.06 | **+17 %** |

**Provisional verdict: go on the coding and agentic sets** - well above spec 19 §7's +10 %
bar with the extrapolated costs; if verify at M = 8 costs a third more (3.5 plain steps), A4
falls to ~+17 % (7.21 / 3.7 = 1.95 vs 1.66) - still above the bar, with little margin. **Prose is marginal** (+17 % on 52 anchors; within the extrapolation's
error) - the adaptive K should fall back to MTP or plain decode there.

**What the box must settle (plan 19a Task 4)** before 19b is formally a "go": verify(M) for
M = 5..8 on the int8 head at depths 4k and 32k (GEMV may leave the bandwidth-bound regime
past M = 4, which is exactly where the extrapolation is weakest), the drafter's GEMV pass at
M = 8 in int8 and in W4A16 g64, and the head's K rows. Then re-run this table with measured
costs (Task 5) and record the verdict in spec 19. The probes are built (plan 19a Task 4's
status): `probe_mtp_steps <snap> <depth> 32 3 int8 off <max_len> 8` and `probe_draft_cost`, run
by box queue row 29 (`tools/box_validate.sh --only r29.cost`, box-day plan Session 6b).

## 4. Run notes

- Dump: 37 sources in 12 forwards; the first two at a 28 GiB container cap took 2.75 h and
  9.6 h (memory pressure); restarted at 48 GiB (batches 2.6-2.9 h), raised live to 54 / 60 GiB
  (batches 1.75 h, then 0.8-1.1 h for the shorter last batches; peak RSS 53.3 GiB). Plan 19a's
  ~5 h estimate assumed 12.5 GiB peak; the real peak was ~4x that.
- Accept: bf16 1.7 h, int8 1.2 h, int8h 1.7 h, w4a16 2.0 h.
- Recorded continuations equal the bf16 greedy decode at 0.995 of all ids (A4 0.998; golden
  0.875-0.969, the 2026-08-24 oracle's ids against today's bf16 reference).
