# Spec 19a - DFlash P0: acceptance, verify cost, the projection

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** decide whether a DFlash drafter pays on our card before any engine kernel is written: the greedy acceptance-length distribution of `z-lab/Qwen3.8-27B-DFlash2` on our prompts (teacher-forced), the verify cost at M = 5..8, and the projected tokens/s against `--mtp auto`; stop and record if it does not clear the bar.

**Architecture:** spec 19 §2 (reference semantics), §7 19a. A CPU PyTorch port of the drafter (DFlash and DFlash 2 in one module) is fed the target's tap hidden states at every position of [prompt + greedy output]; its drafts against the real next greedy ids give the acceptance lengths a live greedy run would see. Taps come from the CPU reference model (layer-streamed, `tools/oracle/stream.py`, runs on the Mac) or from the engine (a debug dump, on the box); verify cost comes from `probe_mtp_steps` on the box.

**Tech Stack:** Python 3 + torch (CPU, the oracle container), safetensors; C++17 / Level Zero for the probe.

**Spec:** `docs/superpowers/specs/2026-10-05-spec19-draft-model-speculative-decoding-design.md` (§1-§3, §4 decisions 2 and 6, §6 D1, §7 19a). Precedent: `tools/oracle/mtp_accept.py` (spec 8 P0's teacher-forced MTP acceptance).

## Global Constraints

- Branch `spec19a-dflash-p0` from main. Mac work in the oracle container (`tools/oracle/run_in_container.sh`, containers capped at 28 GB RAM); box work in its own tree, `tools/box.env` copied if missing and never committed, every GPU command under `flock ~/b70-gpu.lock`, detached, polled, interleaved pairs.
- The drafter checkpoint: `z-lab/Qwen3.8-27B-DFlash2` (3.85 GB, Apache-2.0) in the HF cache; the target reference: the Qwen3.8 bf16 checkpoint in the HF cache. Never downloads from code.
- The port is written from the semantics in spec 19 §2 and vLLM's Apache-2.0 sources (`qwen3_dflash.py`, `qwen3_dflash2.py`, `dflash2/speculator.py`); nothing copied without its licence header.
- No recursive force deletes. Commit on the branch; no merge, no push.

## Review Focus

1. **The tap offset:** `target_layer_ids` i means the residual stream AFTER layer i (vLLM uses i + 1 as the aux-hidden index, i.e. the input of layer i+1); a test pins it against the reference model's per-layer outputs.
2. **The block construction:** context = taps of committed positions < p; queries = [token at p, mask x K] at positions p .. p+K; non-causal within the block (`is_causal: false`), sliding window 2048 over the context.
3. **DFlash 2's conv taps** look back exactly one row of the block (`row % (1+K) >= tap`); K < 7 is legal and changes only the block length.
4. **The selector walk** at temperature 0 is argmax over `unary + bilinear(prev, h, cand)`, `prev` = the anchor token at row 0; ties to the lower candidate index.
5. **The draft-only vocabulary arm** (decision 6) counts a miss whenever the target's next id is outside V′; V′ built exactly as `loader::select_draft_vocab` builds it (added tokens always in).

---

### Task 1: the reference drafter (Mac)

- [ ] `tools/oracle/dflash_ref.py`: load the drafter's safetensors and config, share the target's `embed_tokens` and `lm_head` (read from the target snapshot), implement §2 (fc → hidden_norm → per-layer k/v + k_norm + RoPE for the context; the block forward with conv (DFlash 2) and non-causal attention; final norm; head top-16; selector walk). Unit tests (`test_dflash_ref.py`) on a tiny random-weight config: shapes, the conv's one-row look-back, the walk's tie rule, K < block - 1. **Commit** `oracle: DFlash / DFlash 2 reference drafter (spec 19a)`.

### Task 2: taps from the CPU reference (Mac)

- [ ] `tools/oracle/dump_taps.py` on `stream.py`: run the bf16 target over [prompt + the engine's recorded greedy output] for the golden prompts (`tests/golden/prompts`), the A4 tool-call set (`tests/golden/toolcall`) and a prose / reasoning set, and save per position the residual after layers 5, 19, 33, 47, 61 (bf16) plus the greedy next ids. Review Focus 1's test. **Commit** `oracle: tap dumps for the DFlash P0 (spec 19a)`.

### Task 3: teacher-forced acceptance (Mac)

- [ ] `tools/oracle/dflash_accept.py`: at every position draft K = 1..7 from the taps, compare with the next greedy ids, report the acceptance-length distribution and mean tokens per verify per corpus; arms: drafter bf16, drafter int8 per-channel RTN (simulated), `syvai/Qwen3.8-27B-DFlash2-W4A16` if the operator downloads it, draft vocabulary V′ = 32k / 64k / 128k (Review Focus 5). Record in `docs/probe-dflash-<date>.md`. **Commit** `probe: DFlash acceptance on our prompts (spec 19a)`.

### Task 4: verify cost at M = 5..8 (box)

- [ ] (the probe build is in: Status below; the measurement is box queue row 29) Build the `gemv` / `gemv_i8w` / attention / GDN-slot variants at M = 5..8 the verify needs (additive binaries; existing command lines unchanged), extend `probe_mtp_steps` to M = 8, measure verify(M) for M = 1..8 on the int8 head at depths 4k and 32k (interleaved pairs, median of 3). Also time an int8 and a bf16 GEMV pass over the drafter's weights at M = 8 for the draft cost. **Commit** `probe: verify cost at M = 5..8 (spec 19a)`.

### Task 5: the projection and the verdict

- [ ] Per corpus: tokens/s at the best K for DFlash (acceptance from Task 3, costs from Task 4) against `--mtp auto`'s (spec 8 §10 tables at the measured acceptance). Apply spec 19 §7's stopping rule (≥ 10 % over `--mtp auto` on the coding and reasoning sets). Spec 19 amendment with the verdict; if it stops, record spec 8 A7's M-row GEMV lever as the prerequisite and re-run Task 5 after it. **Commit** `spec 19: P0 verdict`.

**Gate for the plan:** the verdict recorded with its numbers; 19b starts only on a "go". Tasks 1-3 run without the box.

## Status (2026-10-08, Task 4's probe build)

- **Task 4: the probe build is in, the measurement is not** (branch `spec19a-task4-probe`, written
  blind on the Mac; box queue row 29, stage `r29.cost`, box-day plan Session 6b).
  - **Kernels** (`src/kernels/CMakeLists.txt`, `B70_VERIFY_M8`, default ON, additive: no existing
    command line moved, and the preprocessed sources of the 25 existing variants of the two edited
    files are unchanged): Qwen3.8's verify list at M = 5..8 - the five int4 GEMVs at their
    production {S, layout}, the int8 head, a||b, the norm pair, SiLU, the gated norm, embed,
    argmax, `attn_prep` and attention v2 (`attn_v2.cl`: M <= 8, was <= 4), the MTP head's KV fill
    (fc, q||k||v, the strided norm, the ZERO_RESID fold, `attn_prep_S1`) and `gdn_step_slots_M<M>_N8`
    (`gdn_step.cl`: `N_SLOTS` overridable, 4 unless set; `kernels::gdn_step_slots_variant`'s
    `slots`). The int8 head at M = 5..8 only (no bf16 head, no int8 KV, no v1 attention at M > 4).
  - **Runtime:** `MtpBuffers` / `MtpDims::sizes` take a slot count (default 4 = every engine and
    stage, unchanged); `build_verify` accepts M up to it. The engine, the pipeline engine and
    `--mtp` / `--mtp auto` are untouched: their lists stay M = 1..4 over 4 slots.
  - **`probe_mtp_steps ... <max_len> 8`:** the M = 5..8 lists over a probe-owned 8-slot
    `MtpBuffers` (+1.06 GB), driven as `Engine::verify` + `commit(0)` drive theirs; pos rewound to
    the depth before every arm; a second table of interleaved pairs (verify M = 1 then M, order
    alternating by round, median of 3). `max_m` defaults to 4: the old probe's arms and output.
  - **`probe_draft_cost`:** one DFlash2 block's GEMVs on random weights of the checkpoint's shapes
    (`tools/probe/draft_cost_shapes.h`, read from z-lab/Qwen3.8-27B-DFlash2's config and
    safetensors header: 5 layers x {attn / mlp conv kernel projections 5120 -> 1280, q||k||v
    5120 -> 6144, o_proj 4096 -> 5120, gate||up 5120 -> 34816, down 17408 -> 5120, the commit's
    ctx k||v 5120 -> 2048} + fc 25600 -> 5120 + the selector's 5120 -> 256, at M = 8) in int8,
    bf16 and int4 g64 (the int4 {S} of the new shapes are PROVISIONAL picks, not swept), and the
    int8 head over the 7 draft rows at V 248320 and 32768. GEMVs only (the drafter's attention,
    norms, convolutions and selector walk are not in it).
  - **Mac gates** (`B70_JOBS=2 tools/mac_check.sh --base main --quick --kernels`): host 72 pass /
    0 fail (incl. the new `verify_m8_names_test` and `memory_plan_test`'s 8-slot sizes), L0 syntax
    98 / 0, `kernel_cmdlines` 630 variants +111 / -0 / ~0, OpenCL syntax 614 / 0, the Mac GPU 8
    agree (`gemv_i8w` at M = 8 K 5120 N 1280 and `argmax` at M = 8 among them; indicative).
  - **Not done here:** a bitwise check of the M = 5..8 rows (spec 19 D3 is 19c's); the commit
    `probe: verify cost at M = 5..8 (spec 19a)` with the box's numbers; Task 5.

## Status (2026-10-08)

- **Tasks 2-3 done** (Mac CPU, 2026-10-06 12:40 to 10-08 02:54): `docs/probe-dflash-2026-10-08.md`.
  E_7 = 7.21 on A4, 6.15 code, 2.2 prose; int8 / w4a16 drafters and the ranked 32k V′ equal bf16.
  Provisional projection (verify at M = 5..8 extrapolated): go on code and agentic output, prose
  marginal. **Task 4 (box) and Task 5 decide formally.**

## Status (2026-10-06)

- **Task 1** merged (`dflash_ref.py`).
- **Tasks 2-3: the tools are in, the runs are not** (branch `spec19a-tasks23`). `dump_taps.py`,
  `dflash_accept.py`, their tests on tiny models (Review Focus 1: a dumped tap is layer i's
  output, bitwise, on a tiny Qwen3.5 checkpoint; Review Focus 2-4: the batched drafter equals
  `dflash_ref.draft_block` per anchor; Review Focus 5: `select_draft_vocab` ported line for
  line and checked on `draft_vocab_test.cc`'s cases) and the driver `tools/oracle/dflash_p0.sh`
  (README "The DFlash P0"). The dumps need a 28 GB container and wait for the spec 12a repeat;
  estimated ~5 h of dumps and ~0.4 h per drafter arm (dry run, assumed rates).
- **The corpora on the Mac:** golden prose / code / cjk with the 32 greedy ids of the
  2026-08-24 oracle (26 anchors each), A4 (34 scenarios, 2613 output ids, all tool calls). **No
  prose / reasoning set with recorded greedy output exists here:** spec 8 P0's 256-id golden
  continuations and the A4 bf16 ids are on the box, and a CPU decode of new continuations
  costs ~6.5 min per token (streamed). `GOLDEN_CONT_DIR` / `EXTRA_SOURCES` take them when they
  are copied over (or recorded by `b70-decode` on the box); until then the prose number rests
  on 52 anchors.
- **Tasks 4-5 open.** The projection needs, per corpus: E_K for K = 1..7 at the chosen arm
  (Task 3), verify(M) for M = 1..8 on the int8 head (Task 4; M = 5..8 unmeasured), the draft
  cost of one block at the arm's precision (Task 4's GEMV pass over the drafter + the head's
  K rows), and `--mtp auto`'s tokens/s at the same corpora's MTP acceptance (spec 8 §10).
