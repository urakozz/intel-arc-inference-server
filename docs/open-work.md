# Open work

The one list of what is not finished. An item leaves this page when it is done: it moves to **Closed recently**
(one line, with its numbers). Details live in the spec docs (`docs/superpowers/specs/`) and the box test list
(`docs/superpowers/plans/box-validation-queue.md`, rows `rN`); this page only says what is left and what closes it.
Merges go in after the Mac checks; the full box validation runs on main on demand (operator 2026-11-07).

Last updated: 2026-11-11. Order of focus: Qwen3.8-27B, then Flash-Next (implemented, tuned, fused, fast), then Bolt.
Low-priority and ruled-out work that may come back: [low-priority-work.md](low-priority-work.md).

## Now, in order

| # | what | state | closes when |
|---|---|---|---|
| 4 | **Flash-Next at full size - the expert-tiering probe** (from saber-engine, 2026-11-07). Flash-Next's 48 layers do not fit two 32 GB cards whole: keep the most-used experts in VRAM by a recorded route profile and read the rest in place, from (a) pinned host RAM over PCIe 3.0 (saber) or (b) card 1's VRAM over P2P (basalt); experts stay int4. The box's 121 GB of RAM holds every expert, so the VRAM hit rate decides it. Bar on one card: XeStrata, ~28 t/s | routing traces recorded 2026-11-09 (39 sources, 474 MB); the probe itself not started | the measured hit rate and t/s; then the operator decides whether spec 22 comes back |
| 5 | **The quiet-box measurement session** - one record-grade speed run at the end, on an idle box and the newest driver. It measures every speed number (tensor parallel, 3l, TP + slots and replica throughput: tonight's numbers ran beside CPU jobs and are not records) and settles the open speed choices: list B (card hand-off copy vs peer, int8 KV, int8 prefill, draft vocabulary, Flash-Next's MTP norm and eager attention: the winner stays, the rest is deleted), 4096-row prefill chunks (+4.2 %, under the +5 % bar: the operator decides), MTP acceptance cold vs restored from the prefix cache (< 1 % closes spec 16), and our adaptive MTP depth against saber's rule. The stages: `tools/box_validate.sh --with optin` (`--list` shows them); parked models' stages wait for those models | after everything above it | measured; README.md, docs/BENCHMARKS.md and docs/18-running-models.md refreshed |
| 9 | **`--max-num-seqs N`: how many requests at once is the user's choice, the context follows** (operator 2026-11-11; replaces slots appearing only with an asked `--max-len`). N = 1 (the default): one slot at the largest context that fits; on two cards that is TP. N > 1: on two cards replica (TP keeps one slot: TP + slots was ruled out), on one card its slots; slot 0 gets the largest context that fits beside N - 1 slots of 32768 (on one card 13c's layout, e.g. 131072 + 3 x 32768; replica the same per card), and an explicit `--max-len` that does not fit beside them fails at startup with the plan's numbers. Limits: 4 slots a card (one card up to 4, replica up to 8); the batched step keeps its 8-row budget (MTP drafts share it). A request that reaches its slot's end is logged, not cut silently. Same name and meaning as vLLM's flag; a user choice, not a tuning handle | after the replica row | `b70-serve --max-num-seqs 1 / 2 / 4` (and 8 on two cards) start with the printed slot layout; a too-long `--max-len` refused; 13c's and replica's slot checks pass |

### The needed stages that have never passed (validation run `f73b249`, 2026-11-07: 232 PASS, 75 SKIP, 16 FAIL)

Each row closes with a PASS (or moves to Waiting with its reason). A gate we touch gets a negative control - a deliberately broken input it must FAIL on (saber's rule: a gate nobody has watched fail is not a gate).

| group | stages | what they need |
|---|---|---|
| **Qwen3.8** | Proof check for the 3 accepted `mtp_gpu_kv8_test` rows: rerun those contexts with the bf16 prompt pass and the int8 cache - if they pass, the int8 prompt pass is the cause as accepted; if not, it is a new failure | after the replica row; ~20 min on a card plus new f32 references |
| **Replica + per-card slots** (merged 2026-11-10, untested on the cards) | `--parallel replica` with an asked `--max-len` gives each card 13c's slots. Left: `box_validate.sh --only r77 --force` (G0 not rerun for these merges), 4-8 requests at once on `--parallel replica --max-len 32768` (by hand: `r78.batch` has no replica mode) with first differences by the f32 judge, throughput against one card's slots and plain replica | both cards after TP + slots, tonight |

## Waiting - only a box run or a CPU job left

| spec | what | proved by | state |
|---|---|---|---|
| **Flash-Next** (next after Qwen3.8) | the golden-set remake with block scores (`r30.golden` / `r30.golden_layers`), which `r32.partial` and `r33.golden_real` wait for (`qwen4exp_partial_test`, `qwen4exp_golden_prefill_intel_test` FAIL on the old 1e-3 gap rule until then); `r34.fixture` / `r34.accept` / `r34.a4_ref` (CPU), then `r34.a4` (needs the whole model loading: item 4); `r59.load_real`. Passed since 2026-11-07: `r31.load`, `r32.k3` / `golden` / `pp` / `cli`, `r33.prefill` / `pp` | CPU references (the remake ~90 GB RAM, holds both cards), item 4 | held in the CPU queue (`#HOLD-sj q4exp-golden-scores`) until the Qwen3.8 and TP + slots references are done |
| **Bolt / LiLiCorr** (parked, operator 2026-11-11: until Flash-Next is implemented, tuned, fused and speed-optimised) | `r60.golden` / `r61.golden` judged by Bolt's f32 reference (the tooling is on branch `bolt-f32-ref`; its CPU job held as `#HOLD-bolt-parked-until-flash-next`); `r61.golden`'s conv-column drift (0.987 vs 0.999); `r64.draft` (LiLiCorr's exp now `df_exp`: confirm on a card); `r65.prefill`'s one-position ring cosine (likely a routing near-tie flip); `r65.serve`, `r57.*`, `r62.a4` / `agentic`, `r60.golden32k`, `r61.golden32k` | Bolt's f32 reference, a card run | parked |
| **K2 / Kolibri** (parked) | `r25.a4_ref` / `r25.a4` (waits for K2 bf16 hot, 75 GB), `r21.passkey` (FAILed: read its log first), `r28.a4` (Kolibri's 156 GB bf16 download), `r28.passkey`. Spec 28m (the MoE draft-checking probe) waits for an MoE model in focus | the models back in focus | parked |
| **16** pipeline | two-card serving: MTP, lookup, prefix cache | row 27 | the next validation run |
| **28** speculative serving | `r2.auto_rows` / `r12.a4` and the stages that compared a speculative arm against `--spec off` (`r6.greedy_long`, `r8.auto_rows`, `r16.mtp_rows`, `r27.cli_mtp`, `r55.serve`, `r55.decode`, `r56.pp_serve`, `r70.d2`, `r70.speed`) now judge each first difference by the f32 judge (`serve_judge.py`); a difference without its reference SKIPs. Their serve contexts' f32 references: the `serve-judge-f32` CPU queue job | the next validation run, then those references | merged (2fb3fd68) |
| **8** MTP | `mtp_head_test`: the MTP head's draft logits against a CPU reference never made (`tools/oracle/mtp_ref.py --dump`, reads Qwen3.8's bf16 original) - SKIPs until then | a CPU job, then the next run | to queue |
| **19** DFlash | the CPU half of the drafter reference (`r54.d1_ref`): the 2x-noise near-tie window is enabled (`WALK_TIE_NOISE_X = 2.0`) | `r54.d1_ref` | passes in the re-test |
| *(operator)* | **A real opencode session on Qwen3.8** against `b70-serve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --parallel replica --prefix-split-last --log-requests ~/b70/logs/opencode-session`. The recording feeds the replica speed record, `r2.opencode` and `r12.d5` | the operator, when the cards are free | not recorded yet |
| *(idea, optional)* | **Is Hadamard-rotated int4 closer to bf16?** nchapman's `Qwen3.8-27B-W4A16-HGPTQ` needs the rotation applied to every quantised layer's input at runtime - our engine does not, so it must NOT be loaded as is. First a CPU comparison against our AutoRound export (KL / top-1 against bf16 on the golden and tool-call prompts); only if clearly closer, the rotation in the engine | CPU job, optional | idea |

## Closed recently

| date | spec | what |
|---|---|---|
| 2026-11-11 | Qwen3.8 | **`mtp_gpu_kv8_test` PASS** with its 3 f32-rejected rows accepted as the int8 prompt pass + int8 cache known fail (operator) |
| 2026-11-11 | Qwen3.8 | **Long-context MTP verify passes:** f32 references for the 32k and 130k contexts (8.1 h on the CPU, 37 GiB peak), then `mtp_verify_32k_test`, `_128k_test` and `mtp_verify_kv8_test` PASS on one card |
| 2026-11-11 | 7 | **TP + slots ruled out** (operator): with MTP, 4 requests at once gave 121 t/s against replica's 206, and its batched rows had crossing failures and a 100-ulp f32-rejected id; kept on branch `tp-slots`, notes in [low-priority-work.md](low-priority-work.md) |
| 2026-11-11 | 3l (2) | **Short prefill chunks deep in a session 4.5-4.9x faster**: split-K flash attention for chunks of <= 128 rows; a 16-row turn at 128k 441 -> 98 ms (bf16), 526 -> 108 ms (int8); the f32 judge rejects 0 ids ([record](probes/probe-longctx-decode-2026-11-09.md)) |
| 2026-11-10 | 17b-d | **Tensor parallel for Qwen3.8 on two B70s merged**: decode 1.61x one card, prefill 1.42-1.75x, serve `--spec auto` ~1.46x |
| 2026-11-10 | 17 | **TP on by default** (`--parallel auto`): the f32 judge rejects TP's ids less often than one card's own ([record](probes/tp-twin-count-2026-11-10.md)) |
| 2026-11-10 | 28 | **Qwen3.8's long MTP tests PASS judged** (`mtp_gpu_*`, `mtp_verify_test`, `dflash_*`, `prefix_gpu_mtp_test`); the tool-call test 29 / 36 PASS |
| 2026-11-10 | 3o | **l0-int8 prefill + int8 KV: a known fail** (~3 % of exact-copy tool calls), startup warning; the cheap fix is in [low-priority-work.md](low-priority-work.md) ([record](probes/a4-int8-prefill-int8-kv-2026-11-10.md)) |
| 2026-11-10 | 21 Flash-Next | **Flash-Next prefill PASS against an f32 reference** (37 layers on two cards and 18 on one) ([record](probes/q4exp-f32-aligned-routes-2026-11-10.md)) |
| 2026-11-10 | 13 | **Several requests at once: 13b replica and 13c slots merged**; one card 1 / 2 / 4 requests 72 / 106 / 125 t/s |
| 2026-11-10 | Qwen3.8 | `r11.near_ties`, `r66.check`, `slots_test` / `slots_nomtp_test` PASS |
| 2026-11-10 | Ornith | `r16.mtp` PASS judged |
| 2026-11-10 | decode | **3h: int8 decode attention at long context +24 %** at 150k, now faster than bf16 ([record](probes/probe-longctx-decode-2026-11-09.md)) |
| 2026-11-10 | prefill | **3n: GDN prefill scan -30 %** (bitwise), whole prefill -1.3 % ([record](probes/probe-gdn-prefill-2026-11-10.md)) |
| 2026-11-10 | prefill | **3m / 3k: where prefill time goes** - probes that led to 3n ([records](probes/probe-prefill-chunk-split-2026-11-10.md)) |
| 2026-11-10 | prefill | **3j: int8 prefill scratch sized to the longest slot** (-384 MiB) |
| 2026-11-10 | tools | CPU reference chain's silent job deaths fixed; two-card test stages see both GPUs; HF snapshot paths follow refs/main |
| 2026-11-09 | decode | **3f: packed int8 decode reader** -15 % at 80k, -20 % at 150k; 3c's anatomy probe led to it ([record](probes/probe-longctx-decode-2026-11-09.md)) |
| 2026-11-09 | prefill | **3e: int8 KV long prefill 2-5.8x faster**, within 1-4.4 % of bf16 ([record](probes/probe-kv8-prefill-2026-11-08.md)) |
| 2026-11-09 | prefill | **3d / 3g / 3i: short-prefill leftovers** - ~8 % faster short turns, 34 MiB less setup ([record](probes/probe-fused-small-chunk-2026-11-08.md)) |
| 2026-11-08 | prefill | **3a / 3b: short prefill chunks** - a 237-id served turn 741 -> 299 ms (3a), a 1-id turn 228 -> 73 ms (3b), bitwise |
| 2026-11-07 | validation | **Full validation on main** (`f73b249`): 232 PASS, 75 SKIP, 16 FAIL; runs on demand since |
| 2026-11-07 | spec 8 | Deeper MTP drafts (K 4-6) only +2-5 %: not taken up |
| 2026-11-07 | models | **Agnes dropped** (no evidence it is used or better) |
| 2026-11-07 | prefill | **W4A4: no** - doubles the output error |
| 2026-11-07 | tools | validation reuses unchanged PASSes and skips doc-only changes |
| 2026-11-06 | 17 | 17a probe: tensor parallel GO (derived 1.68x one card) |
| 2026-11-06 | engine | **Zero-config cleanup**: 21 flags, 11 env vars, 45 tests removed; B70-only hardware check |
| 2026-11-06 | box | model storage tiers (NVMe / 8 TB / 3 TB) and the box layout under `~/b70` |
| 2026-11-06 | 5 | 5d: 4096-row prefill chunks built (opt-in; the default is item 5's call) |
| 2026-11-06 | 25/26 | Bolt's `--spec lookup` / `dflash` on the zero-config surface |
| 2026-11-06 | 20 | Kolibri tool sessions: 20 turns, 0 stalls, 25 valid calls |
