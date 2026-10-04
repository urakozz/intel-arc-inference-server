# Spec 14 - phase 2: validate Agnes support on the box

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. One box session, in this order; a step that fails stops the session at that step (fix on the branch, re-run from the step that failed - G0 first if anything shared changed).

**Goal:** prove branch `spec14-agnes` on the card - Qwen3.8 bitwise unchanged (G0), Agnes 3.0 Flash correct (G1-G5), the provisional GEMV rows measured and replaced, speed recorded - then merge.

**Spec:** `docs/superpowers/specs/2026-10-02-spec14-agnes-3-flash-design.md` (§4 gates, §5 speed, §6 phase 2). **Phase 1 record:** `docs/probe-agnes-2026-10-03.md` (what ran on the Mac).

**Branch commits (phase 1, written on the Mac, nothing ran on a card):** `be21676` ModelDesc; `3aa5dbf` oracle + fold proof; `e5a71e9` Agnes descriptor, loader fold, kernel variants, test registrations; `061a6e5` template and tool calls; `42a41bf` box instruments (`probe_gemv --agnes`, passkey `MAX_LEN`, A4 reference); `2f8c994` Agnes `vocab_used` 248089.

## Global constraints

- Box workflow as always: `BOX` from `tools/box.env`; per-branch tree `tools/box.sh dir`; every GPU command under `flock ~/b70-gpu.lock`, detached (`tools/probe/detach.sh <log> <cmd>`), polled with a background until-loop; idle-box protocol (DRM-holder check) before any timing; device 0 for every timed row; interleaved control/candidate pairs after a warm-up, median of 3, `uptime` recorded.
- Builds `-j44` (`JOBS=44`).
- No `rm -rf`. Commit on `spec14-agnes`; merge only after step 12.
- **Every number below written as "PENDING" is a phase-1 placeholder.** Replace it with the measured value or the gate fails.

## Provisional values this session replaces (each by name)

| # | where | value now | replaced by |
|---|---|---|---|
| P1 | `src/model/model_desc.cc` `make_agnes()` GateUp row | `{5120, 38912, S 8, L 0}` (copied from Qwen3.8's gate‖up cell) | step 4 sweep |
| P2 | same, Down row | `{19456, 5120, S 4, L 0}` (copied from Qwen3.8's down cell) | step 4 sweep |
| P3 | `src/kernels/CMakeLists.txt` `add_gemv_variant` cells `K5120 N38912 S8 L0` / `K19456 N5120 S4 L0` | extra defines `GEMV_DEQ_SHIFT=1` / `GEMV_BLOCK2D=8 GEMV_DEQ_SHIFT=1` (copied) | step 4: keep only if the copied cell beats or ties the plain cell (`probe_gemv --agnes` prints both) |
| P4 | `model::agnes().doc_w` | 18.344234624e9 B, summed from the safetensors headers | step 3: the loader's W line must land within 2% (it prints the delta) |
| P5 | `model::agnes().vocab_used` | 248089 (tokenizer.json: 248077 + 12 specials) | step 3: `parity_test` on Agnes's tokenizer; step 6 golden gates |
| P6 | `tests/tokenizer/agnes_template_*.txt` | rendered on the Mac with transformers 4.57.6 `render_jinja_template` (no tokenizer.json there) | step 3: re-dump with `apply_chat_template` (transformers 5, container) and `git diff` - must be empty |
| P7 | spec §5 derived speeds | ~26 t/s decode at 4k, pp4096 ~1850-1900 t/s | step 9: measured rows |
| P8 | decode list size | 870 launches (12 x 72 + 6, derived) | step 6: `golden_gate_agnes_test` asserts it |

## Steps

### 1. Setup

```
tools/box.sh dir                                            # the spec14-agnes tree
ssh $BOX 'df -h ~; free -g'
# the Mac already holds the complete checkpoint (verified 2026-10-04): copy it over the LAN
# instead of downloading again; fall back to `hf download` on the box only if this fails
rsync -a --info=progress2 ~/.cache/huggingface/hub/models--urakozz--Agnes-3.0-Flash-W4A16-AutoRound-GPTQ \
  $BOX:~/.cache/huggingface/hub/
ssh $BOX 'uvx --from huggingface_hub hf download urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ'   # no-op if the copy is complete
ssh $BOX 'uvx --from huggingface_hub hf download Agnes-AI/Agnes-3.0-Flash --dry-run'               # size first
```
Pass: the Agnes snapshot complete (`model.safetensors.index.json` names 6 files, all present, no `.incomplete`); the bf16 `Agnes-AI/Agnes-3.0-Flash` downloaded if it fits (else A4's reference is the dequantised W4A16, labelled so). Qwen3.8 gate checkpoint and `oracle-out-primary` present as before.

### 2. Build, and G0 - nothing moves for Qwen3.8

```
JOBS=44 tools/box.sh build                                   # spec14-agnes, -Werror
# main's kernels, same compiler, same flags, in a main tree (or a fresh `git worktree` of main):
ssh $BOX 'cd <main tree> && git pull && cmake --build build -j44'
ssh $BOX 'cd <main tree>/build/kernels && sha256sum *.bin | sort -k2' > /tmp/main.sha
ssh $BOX 'cd <spec14 tree>/build/kernels && sha256sum *.bin | sort -k2' > /tmp/branch.sha
join -1 2 -2 2 /tmp/main.sha /tmp/branch.sha | awk '$2 != $3'      # must print nothing
comm -13 <(awk '{print $2}' /tmp/main.sha) <(awk '{print $2}' /tmp/branch.sha)   # the added binaries
```
Pass: **every** main binary present on the branch with an identical sha256; the added binaries are exactly the 32 `tools/kernel_cmdlines` lists (`cmake -S tools/kernel_cmdlines -B /tmp/kc -DTREE=$PWD`, diff against main: argmax_stage1 _V248089 x4, gdn_step_slots _G54 x4, gemv at K5120 N38912 / K19456 N5120 for M 1-4, pf_dequant_slab x2, pf_dequant_tile x8, pf_quant_had_K19456, pf_silu_mul_I19456, prep_silu_mul _I19456 x4). The Mac proved the command lines identical (239/239) and the preprocessed `prep.cl` / `pf_prep.cl` identical at Qwen3.8's defines; if a `prep_silu_mul_M*` / `pf_silu_mul` checksum still differs, ocloc embeds line information - rule on it in the record, and the bitwise gates below decide.

```
JOBS=44 tools/box.sh test ''   # i.e. ctest -LE agnes on the branch build, detached, under the lock
```
Pass: the full Qwen3.8 suite green (`ctest -LE agnes`); `golden_gate_test`, `golden_gate_i8head_test`, `prefill_gate_*`, `replay_determinism_test` (+`_i8head`) **bitwise identical** to main's last record (same verdict lines, same per-layer cosines printed to the last digit). Includes the routed Qwen3.8 assertions (`buffers_test` byte table, `prefill_smoke_test` launch arithmetic 8449 / l0-int8 8705, `kernel_count` 774).

### 3. Host and loader checks on Agnes

```
ctest -R 'model_desc_test|agnes_fold_test|kernel_table_test|template_agnes_test|template_test|qwen35_test'
tools/oracle/run_in_container.sh 'python3 tools/oracle/test_agnes.py "$SNAP"'           # ORACLE_MODEL=models--urakozz--Agnes-3.0-Flash-W4A16-AutoRound-GPTQ
tools/oracle/run_in_container.sh 'python3 tools/oracle/test_agnes_fold.py'
tools/oracle/run_in_container.sh 'python3 tools/tokenizer/dump_agnes_template.py "$SNAP"' && git diff --exit-code tests/tokenizer   # P6
ctest -R load_agnes_test                                     # device; under the lock
build/tests/parity_test <dir with Agnes corpus ids>          # P5: re-make corpus.ids with Agnes's tokenizer.json first (tools/tokenizer/dump_parity.py)
```
Pass: `test_agnes.py` **4/4** (check 4: `build_reference` strict key match - pending on the Mac); `load_agnes_test` OK (72 layers, folded shapes, join readbacks exact, MTP head bound, 131072 refused) and its loader report's W check within 2% (P4); template re-dump empty diff (P6); the Rust tokenizer's parity on Agnes's `tokenizer.json`.

### 4. GEMV tuning of the two new shapes (P1-P3)

```
CMAKE_ARGS=-DB70_AGNES_SWEEP=ON JOBS=44 tools/box.sh build
tools/probe/detach.sh ~/agnes-gemv.log flock ~/b70-gpu.lock build/tools/probe/probe_gemv --agnes
```
Pass: every row within tolerance; the decision lines name `L` and `S` per shape. For each shape: if the pick equals the provisional `{S, L}`, keep it and record the GB/s; otherwise replace P1/P2 in `make_agnes()` **and** the cell in `add_gemv_variant` (P3), and note: a gate'‖up' S other than 8 also needs a `prep_silu_mul` `SILU_S` variant (`Capture::check_sizes` asserts 8), a down' S other than 4 the matching `prep_res_fold_M*_K5120_SP<S>_G20`. Review Focus 2 of plan 14c: each picked config's GB/s against bytes; the int4 GEMVs run ~590 GB/s on Qwen3.8 - a row much below is a finding. Then rebuild without the sweep and re-run step 2's checksum compare (only Agnes binaries may change).

### 5. Kernel tests at the new shapes (G1, card half)

```
ctest -R 'gemv_test|prep_test|pf_dequant_slab_test|pf_int8_test|argmax_test'
```
Pass: all green - `gemv_test` now includes `{1,5120,38912,8,0}` and `{1,19456,5120,4,0}`, `prep_test` the `_I19456` silu (random and exact cases), `pf_dequant_slab_test` 5120x38912 and 19456x5120, `pf_int8_test` K = 19456 (quant, requant, colmax, gemm).

### 6. Oracle dumps, then the golden gates (G1 CPU on real inputs, G2)

```
# golden sets for Agnes, unfolded reference (agnes.py), detached, ~1 h, 61+ GiB peak - check free -g first
ORACLE_MODEL=models--urakozz--Agnes-3.0-Flash-W4A16-AutoRound-GPTQ OUT_DIR=oracle-out-agnes \
  setsid nohup tools/oracle/golden.sh > oracle-out-agnes/golden.log 2>&1 </dev/null &
# real MLP inputs for the fold proof (one prompt), then the proof on the snapshot
tools/oracle/run_in_container.sh 'python3 tools/oracle/dump.py "$SNAP" --prompt tests/golden/prompts/prose.ids \
  --out /scratch/agnes-prose.safetensors --mlp-in /ws/oracle-out-agnes/mlp_in.safetensors --gen 4'
tools/oracle/run_in_container.sh 'python3 tools/oracle/agnes_fold.py --proof "$SNAP" --inputs /ws/oracle-out-agnes/mlp_in.safetensors'
# modeling_agnes.py against our reference (Review Focus 1 of plan 14a): 64-id prompt, logits per position
#   (write the 20-line comparison in the container: AutoModelForCausalLM trust_remote_code on the
#    dequantised state dict vs agnes.build_reference; layer by layer if RAM is short)
ctest -L agnes -R 'golden_gate_agnes|prefill_gate_agnes'
```
Pass: G1 (CPU, real inputs) cosine >= 0.999999 on layers 0, 3, 35, 71; `modeling_agnes.py` vs our reference cosine >= 0.99999 and the same argmax at every position; `golden_gate_agnes_test`, `golden_gate_agnes_i8head_test`, `prefill_gate_agnes_l0_test`, `prefill_gate_agnes_int8_test` pass with the golden tie rule (P8 asserted there). Note the prompt ids are Qwen3.8's tokenisation (P5 note) - valid for the gate, which only needs both sides to see the same ids.

### 7. G4 - the features on Agnes

```
ctest -L agnes -R 'replay_determinism_agnes|prefill_split_agnes|snapshot_agnes|prefix_gpu_agnes|mtp_verify_agnes'
tools/oracle/run_in_container.sh 'python3 tools/oracle/mtp_ref.py --dump "$SNAP" --prompts tests/golden/prompts \
  --cont <engine greedy cont256 dir> --out /ws/oracle-out-agnes-mtp'      # as oracle-out-spec8b was made
ctest -R mtp_head_agnes_test
```
Pass: replay bitwise; prefill split l0 / l0-int8 within their bars; snapshot C1 bitwise; prefix C2 (`prefix_gpu_agnes_int8_test`); MTP M2 (`mtp_verify_agnes_test`, its routed sizes 54 / 18 layers) and M1 (`mtp_head_agnes_test` against `oracle-out-agnes-mtp`); spec 9 L3 is `golden_gate_agnes_i8head_test` (step 6); spec 10 v2 is the default decode attention every Agnes gate ran.

### 8. G5 - passkey at ~60k (Agnes's max_len ceiling)

```
MODEL=urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ ORACLE_MODEL=models--urakozz--Agnes-3.0-Flash-W4A16-AutoRound-GPTQ \
  MAX_LEN=65536 N_TARGET=60000 tools/probe/detach.sh ~/agnes-passkey.log tools/probe/passkey.sh l0-int8 l0
```
Pass: `passkey l0-int8: 3/3` and `passkey l0: 3/3` (ids from Agnes's own tokenizer - `passkey.py` uses the snapshot's).

### 9. A4 (G3) and speed (P7)

```
# A4: re-make the set's ids with Agnes's tokenizer (make_set.py on the Agnes snapshot) into tests/golden/toolcall-agnes
tools/oracle/run_in_container.sh 'python3 tools/toolcall/make_set.py "$SNAP" tests/golden/toolcall-agnes'
tools/oracle/run_in_container.sh 'python3 tools/toolcall/oracle_generate.py <Agnes-AI/Agnes-3.0-Flash snapshot> tests/golden/toolcall-agnes /scratch/agnes-toolcall-ref'
ZE_AFFINITY_MASK=0 tools/toolcall/engine_generate.sh <agnes snap> tests/golden/toolcall-agnes ~/agnes-toolcall-out l0-int8
python3 tools/toolcall/score.py ...   (as spec 5's A4 rows)
# speed: decode at 4k / 32k / 60k depth (max_len 65536), pp4096, load time, memory report; --mtp 1..3 greedy
#   b70-decode <agnes> --bench ... and tools/probe/serve_benchy.sh with MODEL=<agnes> MAX_LEN=65536
```
Pass: A4 match count recorded next to Qwen3.8's 25/36 (no bar, spec §4 G3; the reference labelled bf16 or dequantised); speed rows recorded against the derived ~26 t/s / ~1850-1900 t/s (no bar). Review Focus 4 of plan 14c: the memory report at 65536 with MTP lists and the int8 head fits.

### 10. The vLLM row (PR #57003)

```
# the box's vLLM XPU image with the PR branch's eight files overlaid (gh pr diff 57003), --language-model-only --max-model-len 65536
llama-benchy --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation   # vLLM
MODEL=<agnes> MAX_LEN=65536 SERVE_ARGS='--lm-head bf16 --mtp 0' tools/probe/serve_benchy.sh <same flags>   # byte-matched
MODEL=<agnes> MAX_LEN=65536 tools/probe/serve_benchy.sh <same flags>                                       # serving defaults
```
Pass: three rows recorded, same checkpoint files, same max_len, never both servers on the card at once. Note in the record: the engine's tool-schema JSON is key-sorted (nlohmann::json), vLLM's is the caller's order - not a factor for llama-benchy (no tools).

### 11. Prefix caching through the server

`llama-benchy --enable-prefix-caching` at depth 4k / 16k / 32k, cache on vs off, as spec 7's rows. Pass: recorded.

### 12. The record, then merge

- `docs/BENCHMARKS.md` "Agnes 3.0 Flash (spec 14)": steps 4, 9-11.
- Spec 14 amendment §8: every gate's result (G0 checksum compare, G1-G5, A4), every P-row's final value.
- `README.md` (scope names both models; an Agnes row), `docs/03-models.md` (the second model), `docs/13-loader.md` (strike "unvalidated" from the Agnes section).
- Strike `provisional_tuning` (and its comment) from `make_agnes()` once P1-P3 are measured.
- `git switch main && git merge --ff-only spec14-agnes` (or the operator's merge), push.

## Items phase 1 could only partly check (all above, collected)

- Host C++: compiled with Apple clang `-std=c++17 -Wall -Wextra -Werror` and run where host-only (16 tests - qwen35, model_desc, json, int4, safetensors, quant, dequant_fixture, lm_head_int8, agnes_fold, openai, toolcall, prefix_cache, prefix_server, spec_accept, mtp_server, protocol - plus template_test in both Agnes modes; its Qwen3.8-snapshot mode needs that snapshot, absent on the Mac); everything touching Level Zero only syntax-checked (`-fsyntax-only` against the open-source headers: 143/150 sources clean, the 7 SYCL-only probes as on main). Nothing linked against Level Zero, nothing ran on a GPU.
- OpenCL: Agnes variants syntax-checked with clang's OpenCL front end (Intel built-ins aside); Qwen3.8's preprocessed `prep.cl` / `pf_prep.cl` identical to main at 8 define sets.
- Python: `test_agnes.py` 3/4 (no transformers 5 on x86_64 macOS), `test_agnes_fold.py` OK, the fold proof on real packed layers with synthetic inputs.
- Not attempted on the Mac: `modeling_agnes.py` (remote code, transformers 5), the full-model forward, every device path.
