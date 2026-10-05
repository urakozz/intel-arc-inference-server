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

- [ ] Build the `gemv` / `gemv_i8w` / attention / GDN-slot variants at M = 5..8 the verify needs (additive binaries; existing command lines unchanged), extend `probe_mtp_steps` to M = 8, measure verify(M) for M = 1..8 on the int8 head at depths 4k and 32k (interleaved pairs, median of 3). Also time an int8 and a bf16 GEMV pass over the drafter's weights at M = 8 for the draft cost. **Commit** `probe: verify cost at M = 5..8 (spec 19a)`.

### Task 5: the projection and the verdict

- [ ] Per corpus: tokens/s at the best K for DFlash (acceptance from Task 3, costs from Task 4) against `--mtp auto`'s (spec 8 §10 tables at the measured acceptance). Apply spec 19 §7's stopping rule (≥ 10 % over `--mtp auto` on the coding and reasoning sets). Spec 19 amendment with the verdict; if it stops, record spec 8 A7's M-row GEMV lever as the prerequisite and re-run Task 5 after it. **Commit** `spec 19: P0 verdict`.

**Gate for the plan:** the verdict recorded with its numbers; 19b starts only on a "go". Tasks 1-3 run without the box.
