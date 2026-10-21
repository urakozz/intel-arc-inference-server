# Spec 25a - Nemotron 3.5 Bolt 74B-A9B: the facts, the CPU reference, golden sets, the A4 references, the tap dumps

**Status (2026-10-21): built on the Mac (branch `spec25a-bolt-reference`; spec 25 §13 "25a as built") - B1 on the tiny puzzles, the facts sheet `docs/probe-bolt-2026-10-21.md`, the prompts, the A4 set, `bolt_oracle.sh`; no real weight read (not downloaded): Task 7's Steps 2-3 are the box's.** Box queue row **57** (rows 57-62 are spec 25's, 63-65 spec 26's).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** the reference every later spec 25 / 26 stage is graded against: a facts sheet with every rounding point of spec 25 §2 pinned (the remote code `R:line`, the training kernels' semantics, the native 5.19.0 class `N:line` as a cross-check); `tools/oracle/bolt_ref.py`, the checkpoint's own remote modelling code layer-streamed with its kernel path (`cuda_kernels_forward`) enabled on the CPU through plain-torch implementations of the five training-kernel functions it calls (`tools/oracle/bolt_mamba_ops.py`) - **bitwise** the unmodified remote code outside the Mamba mixer, and inside it bitwise the engine's twin order and within a measured fp32 bound of the remote torch fallback, on tiny random puzzle models (B1); golden sets on SergiioB's g128 checkpoint dequantised (the engine-format reference) and on the BF16 original; the A4 set rendered by Bolt's template with both references; the tap dumps spec 26a needs; gate Q recorded.

**Architecture:** spec 25 §2, §9 B1 / B5, §10 25a; spec 21a's / 24a's arrangement (`tools/oracle/stream.py`'s `attach` materialising one block at a time, routed experts read lazily - only the selected ones - dequantised from int4 by `stream.dequant_t`, routes recorded by hooks, `layer_major` for batches). New here: (1) the model class is the **remote code** from the checkpoint snapshot (imported from the HF cache at run time, never vendored - spec 25 Decision 20), its `mamba_ssm` / `causal_conv1d` imports satisfied by a stub package that routes to `bolt_mamba_ops.py`, and the mixer's dispatch (R:735-738: the kernel path only on CUDA) patched to take `cuda_kernels_forward` on the CPU - the torch fallback stays callable for the comparisons; (2) the cache is an `NemotronHHybridDynamicCache` built by us with dtype fp32 and passed explicitly (R:1664-1667 would build it in the model's dtype); (3) Decision 1's semantics: the kernel path passes `dt_limit` (0, inf) (R:449, R:524) and never clamps; the fallback's clamp (R:595, R:643) is left in R and exercised by the comparisons, never edited.

**Tech Stack:** Python 3, torch 2.14.1 CPU, **transformers 5.19.0** through spec 21a's site (`tools/oracle/qwen4exp_env.sh <dir>`), safetensors; the Mac's `agnes-ref-img` container (8 GB, 4 CPUs) for tests and tiny runs; the box CPU (`tools/oracle/run_in_container.sh`) for real weights.

**Spec:** `docs/superpowers/specs/2026-10-20-spec25-nemotron-bolt-design.md` (§1, §2, §5.1, §7, §9 B1 / B5, §10, §12 decisions 1, 2, 10, 11, 20). Precedents: plans 24a (task for task) and 23a (gate Q as a recorded measurement, tap dumps), `tools/oracle/lfm2moe_ref.py`, `tools/oracle/kolibri_ref.py` (the fp32 sigmoid router with a selection bias), `tools/oracle/dump_taps.py` / `muse_taps.py` (the dump format 26a reads), `tools/toolcall/{make_set,score,oracle_generate}.py`, `tools/toolcall/a4_ref.sh`, `tools/box_validate/lfm2moe_oracle.sh`.

## Dependencies and branch points

- **None on the engine.** No C++, no kernel. First of spec 25's plans: 25q, 25b, 25c and 26a depend on it.
- **Weights.** No public tiny model of `nemotron_h_puzzle` exists (checked 2026-10-20 by the spec writer; re-check in Task 1): Task 2 builds random tiny puzzle models from the real config's structure. The real checkpoints are box-side: SergiioB's g128 (41.9 GB, `hf download SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym --revision b9f381d20cbd4cede9f729c2647d2542c4a35997`) and the BF16 original (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16`, 63 shards, ~150 GB: `df -h ~` first). On the Mac only the small files are read (already in the HF cache: config, index, modelling / configuration files, tokenizer, template, generation config, README, the reasoning parser) plus range reads of shard headers and of a few tensors (a few MB; Task 4).
- **Our g64 export** (25q) does not exist yet: the `export` mode runs on it when it does.

## Global Constraints

- Branch `spec25a-bolt-reference` from main; signed commits; no merge, no push. Box tree automatic; `tools/box.env` copied from the main checkout if missing, never read into a log, never printed, never committed.
- **One reference version:** transformers 5.19.0 (`PINNED_TRANSFORMERS`), refused otherwise by name; the remote code from the pinned snapshot (`b9f381d2`; NVIDIA's 24GB repo ships the byte-identical file `de290021...` - the sha is checked).
- **Nothing of the model is vendored** (Decision 20): configs are built in code from the cached `config.json` at run time; tests that need the real config SKIP (exit 77) without the HF cache.
- **The reference's attention is eager**; flash is the engine's and held to fp64.
- Python tests in `agnes-ref-img` with `--memory 8g --cpus 4 -e OMP_NUM_THREADS=3`, the 5.19.0 site on `PYTHONPATH`. One oracle container at a time on the Mac; the CPU model lock `/tmp/b70-cpu-model.lock` (python `fcntl`) around every real-weight run on the Mac.
- Box CPU runs: the oracle container, `MemAvailable` >= the script's floor, one at a time, detached (`tools/probe/detach.sh`), polled; no GPU, no GPU lock.
- Every number in docs is measured, or marked derived / estimated / proposed / published.
- No `rm -rf`. Nothing about third-party engines beyond what spec 25 cites.

## Review Focus

1. **Bitwise where it can be, measured where it cannot.** Outside the Mamba mixer the streamed port equals the unmodified remote code bit for bit (fp32 and bf16), prompt and cached decode, chunks = one forward. Inside it the port (R's kernel path + `bolt_mamba_ops`) equals the engine twin's order bitwise, and R's torch fallback in fp32 within a bound this plan measures and records (the same function summed in other orders); the clamp is the one semantic difference, exercised only on the variant built to trigger it. Chunks = one forward for the port's own Mamba state.
2. **The Mamba-2 chain.** `in_proj` split z ‖ xBC ‖ dt (R:478-482); the conv state holds the RAW xBC (R:568-572), taps oldest first, bias then SiLU then ONE rounding; `dt = softplus(dt_raw + dt_bias)` with mamba_ssm's threshold 20 and NO clamp; `A = -exp(A_log)` in fp32; heads to groups `h / 16`; the state update and `y = C·S + D·x` in fp32, y rounded once; the gated norm over groups of 1024 with `norm_before_gate` false (`y · silu(z)` normed, not `norm(y) · silu(z)`). A test per point, each one able to fail (swap z and xBC, drop the bias, gate before the norm, group 8192 instead of 1024, the clamp back in).
3. **The router's fp32 chain and the combine.** fp32 logits from widened bf16 (R:1044), sigmoid, `sel = s + bias`, top-9 by `sel` (the set; torch's `sorted=False` order recorded, not trusted), `w = s / (Σ + 1e-20) x 5.0` in fp32; the combine in ascending expert id, product rounded to fp32 then added, one rounding (R:925-950); relu² on bf16; the shared expert on the NORMED hidden (R:964), added after `fc2`. Ties planted at the 9th / 10th `sel`.
4. **The tap convention.** `taps[i]` = the residual after block i = `output_hidden_states[i + 1]` (R:1481-1482: the tuple holds block INPUTS), for i in [1, 18, 35, 51, 68, 85]; a test pins it on the tiny model at the layers that exist there.
5. **NoPE and the untied head.** Attention has no positional term and no q / k norm (a RoPE added by mistake changes every attention output); the head is `lm_head`, not the embedding.

---

### Task 1: the remote code end to end in 5.19.0 on a tiny config (Mac)

**Files:**
- Create: `tools/oracle/bolt_tiny_check.py`, `tools/oracle/bolt_remote.py` (imports the remote modules from the snapshot under a private package name, installs the `mamba_ssm` stub, builds a `NemotronHPuzzleForCausalLM` from a config dict - the omni wrapper is never built)

- [x] **Step 1: write `bolt_remote.py`** (`load_remote(snapshot) -> module`, `build_lm(cfg_dict, dtype) -> NemotronHPuzzleForCausalLM`, `fp32_cache(model, batch) -> NemotronHHybridDynamicCache` with dtype fp32) **and `bolt_tiny_check.py <snapshot>`**: a random tiny puzzle config with the real model's first 12 block types (`M E M E M E M * E M E M`), hidden 256, Mamba 8 heads x 32 (d_inner 256), `n_groups` 2, state 16, conv 4, attention 16 q / 1 kv x 16, MoE E per layer {16, 20, 24} with I {64, 128, 192}, top-9, latent 64, shared 128, vocab 512; fp32 and bf16; a 24-id prompt with `generate(max_new_tokens=8, do_sample=False)` using our fp32 cache, then the 32 ids as one uncached forward: the last 8 rows' argmax equal; the logits' max |diff| printed in fp32 and bf16 (recorded - 24a found cached vs uncached not bitwise in transformers itself even in fp32); print which mamba path ran (the torch fallback unpatched, `cuda_kernels_forward` with the dispatch patch - R:735-738) and whether the stub's functions were called. Exit 0 only if all hold.
- [x] **Step 2: run on the Mac** (the container command of plan 24a Task 1 with the Bolt snapshot mounted read-only). Expected: `tiny check: OK`. If 5.19.0 cannot build the remote class, stop and report (no version switch).
- [x] **Step 3: commit** `git commit -S -m "oracle: Bolt's remote code end to end on a tiny puzzle config in transformers 5.19.0 (spec 25a)"`.

### Task 2: the training-kernel ops and the tiny variants (Mac)

**Files:**
- Create: `tools/oracle/bolt_mamba_ops.py`, `tools/oracle/bolt_make_tiny.py`, `tools/oracle/test_bolt_make_tiny.py`

**Interfaces:**
- Produces (`bolt_mamba_ops.py`, imported by `bolt_ref.py`, 25c's fixture generator and 26a): `rmsnorm_gated(x, z, w, eps, group)` (mamba_ssm `rmsnorm_fn` with `norm_before_gate=False`: `v = f32(x) * silu(f32(z))`, per group `v * (1 / sqrt(mean(v²) + eps)) * w`, cast to x's dtype); `conv_update(x_new, ring_cols, w, b)` / `conv_seq(x, w, b, state)` (fp32 taps oldest first, `+ b`, SiLU, one rounding); `ssm_update(S, x, dt_raw, A, B, C, D, dt_bias)` and `ssm_chunk(x, dt_raw, A, B, C, D, dt_bias, S0, chunk)` (fp32 internals, softplus threshold 20, `dt_limit (0, inf)`); `softplus_mamba`; `A_from_log(A_log)` (torch's fp32 `exp`); and a `stub_package()` that exposes them under `mamba_ssm.ops.triton.{layernorm_gated,selective_state_update,ssd_combined}` and `causal_conv1d` names so R's imports resolve. The port runs R's `cuda_kernels_forward` with these (the dispatch patched for the CPU); R's torch fallback is untouched and stays the comparison (with its clamp) - Decision 1.
- Produces (`bolt_make_tiny.py <out> --variant NAME [--seed 0]`, `save_pretrained`-shaped, the checkpoint's tensor names under `language_model.model.*`, the real tokenizer / template copied from the HF cache at run time):
  - `pattern12`: Task 1's config, bf16, non-zero `e_score_correction_bias` ~ 4.5 + N(0, 0.02) fp32 (the checkpoint's scale), `dt_bias` in [-10.5, 8.5], `A_log` in [-4.7, 4.4] (the checkpoint's ranges), F32 `A_log` / `D` / `dt_bias` holding bf16 values;
  - `pattern12-fp32`;
  - `ties`: two experts with identical router rows and biases in one MoE layer (exact `sel` ties at the 9th / 10th);
  - `dtclamp`: half the heads' `dt_bias` below -6.9 (softplus < 0.001 at dt ~ 0) - where the clamp matters;
  - `int4-g64`: every linear RTN int4 g64 sym in the `auto_round:auto_gptq` bytes (`src/loader/rtn.h`'s rule, **signed** scales allowed by the packer's option `--signed-scales` - AutoRound's form, a group's extreme mapped to -8);
  - `int4-g128-gptq`: SergiioB's spelling - `quant_method` gptq, group 128, `g_idx` = `i // 128`, `qzeros` 0x77777777, F16 signed scales, the `dynamic` object with the restating `+:` `out_proj` rule.
- [x] **Step 1: the failing tests** (`test_bolt_make_tiny.py`; and unit tests in the same file for the ops): each variant loads through `bolt_remote.build_lm`; the dequant of both int4 variants equals `(q - 8) x scale` exactly; `rmsnorm_gated` equals NVIDIA's documented formula on random data and DIFFERS from `norm_before_gate=True` and from a group of 8192; `conv_update` with the bias dropped or the SiLU before the bias differs; `ssm_update` with the clamp on differs on `dtclamp`. Run: FAIL (`ModuleNotFoundError`).
- [x] **Step 2: implement; PASS. Commit** `git commit -S -m "oracle: Bolt's training-kernel ops (gated norm, conv, SSM update / chunk, no dt clamp) and tiny puzzle variants incl. SergiioB's g128 GPTQ spelling (spec 25a)"`.

### Task 3: `bolt_ref.py` and B1 (Mac)

**Files:**
- Create: `tools/oracle/bolt_ref.py`, `tools/oracle/test_bolt_ref.py`
- Modify: `tools/oracle/README.md` (a section "The Nemotron 3.5 Bolt reference" and the Files table rows)

**Interfaces:**

```python
# tools/oracle/bolt_ref.py
PINNED_TRANSFORMERS = "5.19.0"
REMOTE_SHA = "de290021..."                     # modeling_nemotron_h.py, checked at import
def config(snapshot: str, layers: int | None = None) -> dict   # llm_config; layers=N truncates block_configs[:N]
                                                               # (N >= 8: the first attention block is 7 - a DynamicCache-
                                                               # style length needs one; smaller N refused by name)
def expected_names(cfg, form: str) -> list[str]  # "bf16" (the original) | "gptq-g128" (SergiioB) | "ar-g64" (ours)
def build_streamed(snapshot, cfg, weights: str = "int4") -> model   # "int4": dequantised on demand (g128 or g64), the
                                                                     # engine-format reference; "bf16": the original
class Recorder:   # per block and row: h after the block; per Mamba block the SSM state and conv columns after the prompt;
                  # per MoE block route ids (ascending), w, sel of ranks 0..9, the 9th-10th gap, Σ; taps after 1, 18, 35,
                  # 51, 68, 85; logits rows
def layer_major(model, pf)
# restated ops (25c's host twin is generated from these; each tested bitwise against the module it restates)
def rms_norm(x, w, eps)                # R:760-765, one rounding
def route(x, gate_w, bias, k=9)        # R:1040-1052 -> ids (ascending), w (fp32), sel10, gap, sum
def combine(d_rows, w, ids)            # R:925-950: ascending id, product rounded to fp32, then add, one rounding
def moe_block(x, layer)                # §2.4 end to end
def mamba_step(x, layer, state) / mamba_seq(x, layer, state)   # §2.2 via bolt_mamba_ops
def main()                             # run | ppl | facts | hfcheck | taps | gateq
```

`run <snapshot> --prompt <ids> --out <p>.golden.safetensors --gen 32 [--layers N] [--weights int4|bf16] [--chunk C] [--taps] [--cache fp32|bf16]`: the golden layout of plan 24a (`h.L<l>` for the last `--act-tail` rows, `route.{ids,w,sel,gap}.L<l>`, `ssm.L<l>` fp32 [128][64][128] and `conv.L<l>` bf16 [3][10240] after the prompt, `logits` tail + generated rows, `tokens`, `nll`), plus with `--taps` `tap.L{1,18,35,51,68,85}` bf16 for every row (dump_taps' names).

- [x] **Step 1: the failing tests** (each against the unmodified remote code un-streamed; SKIP with a message when the HF cache lacks the snapshot):
  - `test_non_mamba_bitwise` on `pattern12` and `pattern12-fp32`: with the Mamba mixer's outputs fed from one shared source, every other op's output bitwise R's (Review Focus 1);
  - `test_mamba_vs_fallback_fp32`: a 40-id and a 400-id prompt, then 8 cached steps - the port against R's torch fallback (clamp off by construction: `pattern12-fp32`'s `dt_bias` keeps every head above the clamp) within the measured bound (max relative error per state and output printed; the bar recorded as found);
  - `test_mamba_twin_order`: `bolt_mamba_ops` against an fp64 evaluation and against the summation order 25c's twin will use - bitwise the latter;
  - `test_chunked_equals_whole`: the port fed chunks of 1, 2, 3, 64, 127, 128, 129 and 255 through the cache against one forward - the attention / MoE / norm outputs bitwise, the Mamba state and outputs within the chunk scan's measured bound (the SSD chunk size 128's edge);
  - `test_bf16_differences_recorded`: in bf16 the port and the remote torch path differ; the differing ops are exactly the ones spec 25 §2.0 lists (each switched off in turn makes its difference vanish); printed per op;
  - `test_dt_clamp`: on `dtclamp` the port (no clamp) and R's fallback (clamp) differ far beyond the bound; with the ops' `dt_clamp=True` test switch they agree within it (fp32);
  - `test_mamba_chain` (Review Focus 2's mutations), `test_router_chain` + `test_route_ties` (Review Focus 3; torch's tie order recorded), `test_combine_order` (ascending id equals R; rank order differs on > 0 rows, printed), `test_relu2_bf16`, `test_shared_on_normed_hidden`, `test_latent_projections`;
  - `test_nope` (adding a RoPE changes attention outputs), `test_untied_head`, `test_tap_convention` (Review Focus 4 at layers 1 and 8 of the tiny model);
  - `test_int4_both_spellings`: `build_streamed(weights="int4")` on `int4-g64` and `int4-g128-gptq` equals the port on the dequantised twin (the same weights as plain tensors) bitwise; the g128 dequant equals the g64-expanded dequant bitwise (the expansion is exact).
- [x] **Step 2: run, FAIL; implement until PASS.** `hfcheck <snapshot> --layers N` runs R (layer-streamed) against the port on real weights (Task 7).
- [x] **Step 3: README section; commit** `git commit -S -m "oracle: bolt_ref.py - Bolt's remote code layer-streamed with the training-kernel ops, B1 on the tiny puzzles - bitwise outside the Mamba mixer, the fallback's bound measured inside it (spec 25a)"`.

### Task 4: the facts sheet (Mac: source, small files, range reads)

**Files:**
- Create: `docs/probe-bolt-2026-10-<dd>.md`, `tools/oracle/bolt_facts.py` (prints the tables; `--headers` range-reads every shard header of SergiioB's repo at the pinned revision into a cache dir outside the tree; `--sample` reads named tensors' data by range)
- Modify: `docs/README.md` (one line)

- [x] **Step 1: the rounding chain**, one row per point with `R:` / `N:` / mamba_ssm citations: the block (R:802-840), the norm (R:760-765), the Mamba mixer (R:384-532 the kernel path, R:535-726 the torch path; spec 25 §2.0's three readings side by side), attention (R:1077-1188), the MoE (R:884-965), the router (R:968-1063), the model loop and the tuple convention (R:1470-1513), the head (R:1731), the cache dtype (R:1664-1667), the puzzle config (RP:68-100).
- [x] **Step 2: the checkpoint facts:** the 112,940 names grouped (spec 25 §1.3), dtypes and shapes from every shard header (range reads, ~2 MB each), the per-layer (type, E, I) table, the 849 vision names, the derived parameter counts and bytes of spec 25 §3 re-derived from the headers, the `quantization_config` (the `+:` rule quoted), the sampled tensors (qzeros, g_idx, signed / subnormal scale counts, `A_log` / `D` / `dt_bias` ranges, `e_score_correction_bias` ranges); **the dt-clamp census**: per Mamba layer the heads with `softplus(dt_bias) < 0.001` from the 40 `dt_bias` tensors (512 B each by range read).
- [x] **Step 3: tokens and template:** `chat_template.jinja` sha256 `6521afc7...` and its minja feature list (spec 25 §7); rendered examples (no system, system, tools, a tool call with string / number / bool / null / list / dict arguments, tool results grouped, reasoning before / after the last user turn with `truncate_history_thinking` both ways, `enable_thinking` false, `medium_effort`) through `apply_chat_template` - 25e's fixtures; the special / added ids; `fix_mistral_regex`'s no-op finding; whether the golden prompt texts re-tokenise.
- [x] **Step 4: torch's CPU `topk(sorted=False)` on tied `sel`** at the 9th / 10th of 240 / 496 (planted ties, 50 rows): which index wins - the engine's rule is the lower id; exact ties are undetermined.
- [x] **Step 5: commit** `git commit -S -m "probe: Nemotron 3.5 Bolt facts - rounding chain, checkpoint (headers, signed scales, the dt-clamp census), tokens, template, tie order (spec 25a)"`.

### Task 5: the golden prompts (Mac)

**Files:**
- Create: `tools/oracle/bolt_prompts.py`, `tests/golden/prompts/bolt_agentic.json`, the ids (`tests/golden/prompts/bolt_{short,4k,32k,agentic}.ids`)

- [x] **Step 1:** `bolt_prompts.py <snapshot>`: `short` = `prose.txt` + `code.txt` encoded by Bolt's tokenizer without BOS (the template writes none); `4k` / `32k` as plan 24a's; `agentic` = an original multi-turn coding session written for this repo (system prompt, a user task about this repository, assistant turns with `reasoning_content` and Qwen3-Coder XML tool calls given as structured `tool_calls`, tool results holding two files of `src/runtime/`), rendered by the checkpoint's template with `add_generation_prompt=True`, `enable_thinking` true, to ~12k ids. Prints each length; refuses a `32k` shorter than 32768.
- [x] **Step 2: commit** the script, the JSON and the ids: `git commit -S -m "tests: Bolt golden prompts - short, 4k, 32k and an agentic session with Qwen3-Coder XML calls (spec 25a)"`.

### Task 6: the A4 set (Mac)

**Files:**
- Modify: `tools/toolcall/make_set.py` (`--model bolt`), `tools/toolcall/oracle_generate.py` (`--model bolt` -> `bolt_ref`), `tools/toolcall/a4_ref.sh` (`bolt set|ref|status`), `tools/toolcall/test_make_set.py`, `tools/toolcall/test_score.py` (Bolt's cases)
- Create: `tests/golden/toolcall-bolt/` (the 36 scenarios' `.ids` / `.json`)

- [x] **Step 1: the failing tests:** a scenario re-rendered by Bolt's template equals `apply_chat_template` byte for byte; `score.py`'s existing Qwen XML format scores Bolt's outputs (output opening inside reasoning, `</think>`, then content and `<tool_call>` blocks; a call inside reasoning is reasoning; a multi-line parameter value; a value containing `</parameter` text).
- [x] **Step 2: implement** (`make_set.py --model bolt --from tests/golden/toolcall --tok <snapshot>`; `oracle_generate.py --model bolt --weights int4|bf16`, 2048 new ids default - reasoning first; `a4_ref.sh bolt set|ref|status`). PASS. **Commit** `git commit -S -m "toolcall: the A4 set through Bolt's template, the bolt reference generator (spec 25a)"`.

### Task 7: the box CPU runner and the real-weight runs

**Files:**
- Create: `tools/box_validate/bolt_oracle.sh`
- Modify: `tools/box_validate/data.sh` (`have bolt_g128`, `have bolt_bf16`, `have oracle_bolt`, `have oracle_bolt_bf16`, `have bolt_a4_ref`, `have oracle_bolt_taps`), `tools/oracle/README.md`

**Interfaces:**
- Produces: `bolt_oracle.sh <data tree> tests|ppl|hfcheck|golden|golden-bf16|export <dir>|a4|a4-bf16|taps|gateq` (lfm2moe_oracle.sh's shape: resumable per prompt into `*.partial/`, `MemAvailable` floor `BOLT_REF_MIN_GB` - 8 tiny, 48 g128, 96 bf16, ESTIMATED; exit 77 when short or the snapshot is absent; `DRY_RUN=1` prints every command, the floor and an ESTIMATED time per prompt). `golden` -> `oracle-out-bolt/` on SergiioB's checkpoint dequantised (the engine-format reference every B3 gate reads); `golden-bf16` -> `oracle-out-bolt-bf16/`; `export <dir>` -> `oracle-out-bolt-<name>/` (25q's exports, 25b's synthetics); `taps` -> `oracle-out-bolt-taps/` in dump_taps' format over [prompt + the g128 reference's greedy continuation] for the A4 scenarios (512 new ids), `short_code`, `short_prose` and the agentic session (spec 26a's input); `gateq` -> spec 25 B5's table (KL / top-1 on held-out rows, A4 agreement / parse failures, g128 vs BF16).

- [x] **Step 1: write the script and the `have` lines;** `DRY_RUN=1` prints the plan.
- [ ] **Step 2 (box CPU): B1's real-weight half** - `ppl` (prose / code / agentic; finite; code < 10 ESTIMATED ceiling, recorded per role as 24a found it necessary) on g128 and BF16; `hfcheck --layers 12` (R vs the port on real weights in fp32 mode where RAM allows, else bf16 with the listed differences: 2048 ids + 4 cached steps).
- [ ] **Step 3 (box CPU): the golden sets, the A4 references, the taps, gate Q** - each log's gap summary (the 9th / 10th `sel` per layer: `B70_BOLT_TIE_TOL`'s evidence, Decision 11), the SSM-state magnitudes per layer, times and RSS. Gate Q is a recorded measurement (spec 25 Decision 3).
- [x] **Step 4: commit** `git commit -S -m "oracle: bolt_oracle.sh - resumable golden runs on g128 / BF16 / exports, perplexity, the A4 references, tap dumps, gate Q (spec 25a)"` (outputs are git-ignored data).

### Task 8: docs, the box queue row

**Files:**
- Modify: spec 25 (a section "25a as built"), this plan's status line, `docs/superpowers/plans/box-validation-queue.md` (row 57), `tools/box_validate/stages.sh` (its `row 57` block)

- [x] **Step 1:** "25a as built": the ops module's departures from R, the dt-clamp census, the bf16 difference table, the tie order, the gap distributions, per-prompt times.
- [x] **Step 2: the queue row 57** and its stages (`python3 tools/box_validate/test_box_validate.py` passes; `tools/box_validate.sh --dry-run --only r57` prints them):

```text
row 57 "spec 25a - Nemotron 3.5 Bolt: the facts, the CPU reference (B1), golden sets, the A4 references, the tap dumps, gate Q (plan 25a)"
stage r57.host       57 default cpu  oracle_image                 -          "bolt_oracle.sh tests: test_bolt_ref.py, test_bolt_make_tiny.py, the toolcall tests"
stage r57.ppl        57 optin   cpu  bolt_g128,oracle_image       -          "perplexity on prose / code / agentic, g128 and BF16 (recorded)"
stage r57.hfcheck    57 optin   cpu  bolt_g128,oracle_image       -          "hfcheck --layers 12: the remote code vs the port on real weights"
stage r57.golden     57 optin   cpu  bolt_g128,oracle_image       -          "golden sets on SergiioB's g128 dequantised -> oracle-out-bolt (gap lines per layer)"
stage r57.golden_bf16 57 optin  cpu  bolt_bf16,oracle_image       -          "golden sets on the BF16 original -> oracle-out-bolt-bf16 (hours)"
stage r57.a4_ref     57 optin   cpu  bolt_g128,bolt_bf16,oracle_image  -     "the A4 references (g128 and BF16) -> oracle-out-bolt-a4 (days, resumable)"
stage r57.taps       57 optin   cpu  bolt_g128,oracle_image       r57.golden "tap dumps for spec 26a -> oracle-out-bolt-taps"
stage r57.gateq      57 optin   cpu  bolt_a4_ref,oracle_image     r57.a4_ref "gate Q: g128 vs BF16 (KL / top-1 / A4), recorded"
rownote 57 "Download first: SergiioB's g128 (41.9 GB) and nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 (~150 GB): df -h ~ first; MemAvailable >= 48 / 96 GB."
```
- [x] **Step 3: commit** `git commit -S -m "docs: spec 25 (25a as built), box queue row 57 - the Bolt reference"`.

**Gate for the plan:** Mac - Task 1's tiny check OK; every test of `test_bolt_ref.py` / `test_bolt_make_tiny.py` / the toolcall tests green with the 5.19.0 site; the facts sheet complete (every §2 point cited, the header tables, the dt-clamp census, the template facts, the tie order). Box CPU - perplexity sane, `hfcheck` within its stated mode, the golden sets / A4 references / tap dumps written, gate Q recorded. Hand back the facts sheet's tables, the gap summaries, the dt-clamp census and the bf16 difference table.
