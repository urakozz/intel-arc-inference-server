# Spec 15a - Ornith: the CPU reference, router facts, and the quantisation study

**Status (2026-10-06): the quant source is decided - the operator's AutoRound checkpoint
already exists** (spec 15 §13): `urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ` (published
2026-08-30, AutoRound 0.15.0, GPTQ packing, int4 g64 sym, iters 400, calibrated on
`opencode-instruct`). Decision 1 (A) holds; Task 4 Step 2 is that checkpoint. Every header /
index fact Task 1 Step 2 asked for is in spec 15 §13 (read by range requests, no download).

### Task 0 (Mac, written; run once the checkpoint is downloaded): the reference from the int4 checkpoint

Golden sets the engine can match token for token come from the int4 checkpoint itself,
dequantised by the repo's rule - spec 18a's argument for K2 - so R1 / R2 / R3 / M1 do not wait
for the box or the 71.9 GB bf16 base. `tools/oracle/ornith_ref.py` (transformers 5.15's
`Qwen3_5MoeForCausalLM`, layer-streamed, routed experts dequantised on demand; tested on tiny
random weights by `tools/oracle/test_ornith_ref.py`, all pass) writes `oracle-out-ornith/`
(prose / code / cjk golden files with `router_logits.L*`, `shared_gate_logits.L*`,
`route.{ids,w,gap}.L*` for every layer and forward) and `oracle-out-ornith-mtp/` (the MoE MTP
head's M1 reference and the 33-id continuations `mtp_head_ornith_test` reads).
`tools/oracle/README.md` "The Ornith reference" has the details.

- **Download** (the Mac's HF cache; the operator started it 2026-10-06):
  `uvx --from huggingface_hub hf download urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ`.
  **Disk:** 22.99 GB (5 shards 21.3 GB + `model_extra_tensors.safetensors` 1.69 GB + 20 MB of
  tokenizer / configs); the outputs are < 1 GB (each golden file ~0.2 GB: GDN states 63 MB,
  logits (T + 32) x 248320 x 4 B ~75 MB, activations and routes; each M1 file 32 MB).
- **Run** (from the repo root on the Mac, detached, resumable, `facts` first - it stops if the
  downloaded checkpoint is not the one spec 15 §13 read):
  `tools/oracle/ornith_golden.sh`, then `tools/oracle/ornith_golden.sh --status`. Container
  `ornith-ref` from `agnes-ref-img:latest`, capped at 28 GB, `ORACLE_THREADS` 12. It refuses to
  start while another `agnes-ref-img` container (e.g. the 12a repeat) runs.
- **RAM (estimated):** embed + lm_head 2.0 GB resident, the one shared expert buffer pair 1.6
  GB, one or two layers of dequantised mixer weights (~0.07 GB each), logits rows ~0.1 GB, the
  MTP head at the end ~1.8 GB (its 257 experts in bf16 + RTN temporaries): **~6-8 GiB RSS**,
  the rest of the 28 GB cap is page cache over the shards.
- **Time (estimated, idle Mac):** the prompt forward dequantises the experts its ~40-60 tokens
  route to (most of 256 per layer: ~1.3 GB bf16 per layer at stream.py's 1-3 GB/s) - **~1 min**;
  a decode step re-dequantises the mixer linears (~2.8 GB bf16) and 9 experts per layer and
  reads the 1 GB head - **~2-5 s**, 32 steps ~1-3 min; the head ~10 s. **~3-5 min per prompt
  warm, ~15-25 min for the three** plus the first cold read of 23 GB; on the loaded Mac,
  longer.
- **Then:** copy `oracle-out-ornith*` to the box tree (`--push-data` or rsync) for rows 10 /
  13 / 16 / 19 (`golden_gate_ornith*`, `ornith_decode_test` R2, `prefill_gate_ornith*`,
  `mtp_head_ornith_test`); read R2's tolerance off the logs' gap line (Task 3's statistic,
  from the int4 checkpoint's own routes).

Still box-side: Task 2's bf16-base reference (R5's quantisation damage, the A4 reference) and
Task 4's RTN arm.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a layer-streamed CPU reference for Ornith 1.5 35B-A3B (R1) with golden sets; the router's exact formula and its statistics on real prompts (the input to 15c/15d's kernel design and to R2's tolerance); the quantisation study R5 (RTN int4 g64 now, the operator's AutoRound g64 when it exists).

**Architecture:** transformers' `qwen3_5_moe` model driven by spec 14's `tools/oracle/stream.py` (the model is built empty; each layer's weights are materialised just before its forward and dropped after), on the bf16 checkpoint. A quantisation simulator dequantises int4 g64 tensors (RTN made here; AutoRound read from its checkpoint) into the same streamed forward.

**Tech Stack:** Python 3, torch 2.14.1 CPU, transformers 5.15.0 (the box's oracle image; the Mac's `agnes-ref` container has the same pins).

**Spec:** `docs/superpowers/specs/2026-10-04-spec15-ornith-moe-design.md` (§1, §3 decision 1, §4.1, §5 R1, R2, R5, §7 15a, §9).

## Global Constraints

- **Two phases.** Phase M (Mac, now): code and unit tests on synthetic tensors, the tensor-index and header checks (range requests, no full download). Phase B (box CPU, when it is back): everything that needs the 71.9 GB bf16 checkpoint. The Mac has no room for it (operator, 2026-10-04): never download it there.
- Branch `spec15a-ornith-reference` from main; commit there; no merge, no push. Box tree `~/b70-inference-server-spec15a` (automatic). Copy `tools/box.env` if missing; never commit it.
- Needs spec 14's oracle tooling (`tools/oracle/stream.py`, the C++ dequant speedup): base the branch on `spec14-agnes` until that merges, and say so in the first commit message.
- Box CPU runs: the oracle container (`tools/oracle/run_in_container.sh`), `free -g` >= 70 GB available, one at a time, detached, polled with a background until-loop. No GPU.
- No `rm -rf`.

## Review Focus

1. **The router formula is read from the transformers code, not assumed:** softmax over 256 then top-8, or top-8 then softmax; whether the top-8 weights are renormalised (`norm_topk_prob` or its equivalent in `qwen3_5_moe`); the shared expert's gate (`sigmoid(shared_expert_gate · x)` scaling the shared expert's output); the dtype of each step. Cite file:line in the installed transformers.
2. **The fused expert tensors' layout:** `experts.gate_up_proj [256, 1024, 2048]` - which half is gate and which up, and whether the 1024 rows interleave them; `experts.down_proj [256, 2048, 512]` - its orientation (out x in). Prove both on one expert against the reference forward.
3. **Router near-ties:** the gap between the 8th and 9th router logits per (layer, token) over the golden and A4 prompts: its distribution sets R2's tolerance (how often bf16 rounding could swap the 8th expert).
4. **RTN for the fused tensors** quantises per expert, per output row, groups of 64 along K, symmetric, exactly as the engine will store them; the router, shared-expert gate, `in_proj_a/b`, norms, embeddings and (separately) `lm_head` stay bf16.
5. **Memory on the box:** a layer of Ornith in bf16 is ~1.6 GB of routed experts; stream it, never the whole model.

---

### Task 1 (Mac): the reference code and its tests

**Files:** `tools/oracle/ornith.py` (config translation and any naming glue for `qwen3_5_moe`; detection in `tools/oracle/dump.py`), `tools/oracle/moe_quant.py` (RTN int4 g64 for fused and per-expert tensors; dequant), `tools/oracle/test_ornith.py`, `tools/oracle/test_moe_quant.py`.

- [ ] **Step 1:** tests on synthetic tensors: the router formula (as read, Review Focus 1) on random logits against a hand reference; RTN round-trip bound per group; fused-tensor slicing per expert (Review Focus 2) on a toy [4, 8, 16] tensor.
- [ ] **Step 2:** the tensor-index and header check against the real repo by range requests (as done for spec 15's shapes): every key classified (attention, GDN, router, experts, shared expert, MTP, vision, embed, head), counts per layer, dtypes; written into `docs/probe-ornith-2026-10-04.md`.
- [ ] **Step 3:** run the tests in the Mac's `agnes-ref` container (or a fresh one from its image). **Commit** `oracle: Ornith (qwen3_5_moe) reference glue and RTN g64 for fused experts (spec 15a)`.

### Task 2 (box CPU): the reference and golden sets (R1)

- [ ] **Step 1:** download `ornith-ai/Ornith-1.5-35B-A3B` (bf16) on the box (`df -h ~` first).
- [ ] **Step 2:** Review Focus 2 on the real checkpoint (one expert of layer 0 sliced and run against the full layer's forward).
- [ ] **Step 3:** golden sets: prose, code, cjk + 32 greedy tokens, layer-streamed, into `oracle-out-ornith/` (gitignored); timing per prompt. Tokenisation: check whether Ornith's tokenizer gives the committed `tests/golden/prompts/*.ids` (as Agnes's did); if not, write Ornith ids beside them and say so.
- [ ] **Step 4:** the MTP head reference dump (its MoE layer) into `oracle-out-ornith-mtp/`.
- [ ] **Commit** `oracle: Ornith golden sets (spec 15 R1)`.

### Task 3 (box CPU): router statistics

- [ ] On the golden prompts and two A4 prompts: per layer, the histogram of tokens per expert (min / median / max rows per expert for a 2048-token chunk, derived from the per-token choices), the fraction of experts never chosen, and Review Focus 3's 8th/9th gap distribution. Doc section; proposed R2 tolerance. **Commit** `probe: Ornith router statistics (spec 15a)`.

### Task 4 (box CPU): the quantisation study (R5)

- [ ] **Step 1:** RTN int4 g64 (Review Focus 4) simulated in the streamed forward: logits relative L2 and argmax match against bf16 on the golden prompts, beside Qwen3.8's 7.77 % (docs/probe-w4a8 §15.2); also with `lm_head` int8 (spec 9).
- [ ] **Step 2: operator step.** The operator's AutoRound W4A16 g64 run (as for Qwen3.8 and Agnes; the hours-long one). When the checkpoint exists: its tensor format (per-expert `qweight` or fused), then the same measurement as Step 1. Until then this step is pending in the doc.
- [ ] **Step 3:** the verdict for spec 15 decision 1 (AutoRound expected; RTN only if it is within the measured noise of AutoRound), recorded for the operator. `docs/README.md` line. **Commit** `probe: Ornith quantisation, RTN vs AutoRound g64 (spec 15 R5)`.

**Gate for the plan:** R1's golden sets and MTP dump, the router formula and statistics, R5 for RTN (and AutoRound when available). Hand back with the tables.
