# Open work

The one list of what is not finished. An item leaves this page when it is done: it moves to **Closed recently**
(one line, with its numbers). Details live in the spec docs (`docs/superpowers/specs/`) and the box test list
(`docs/superpowers/plans/box-validation-queue.md`, rows `rN`); this page only says what is left and what closes it.
Merges go in after the Mac checks; the full box validation runs on main on demand (operator 2026-11-07).

Last updated: 2026-11-12. Order of focus: Qwen3.8-27B, then Flash-Next (implemented, tuned, fused, fast), then Bolt.
Low-priority and ruled-out work that may come back: [low-priority-work.md](low-priority-work.md).

## Now, in order

| # | what | state | closes when |
|---|---|---|---|
| 11 | **Pipeline parallel: its full serving test** (operator 2026-11-11: right after item 10, then close) | `--parallel pp` splits the layers between the cards; the engine is built, its full serving test never ran. Run validation row 27 with Qwen3.8 at the full context with the bf16 cache (262,144 split over both cards): MTP across the split, the prefix cache on two cards, the passkey at ~250k | `tools/box_validate.sh --only g0.build,r27` all PASS |
| 13 | **Qwen3.8: choose the MTP guesses per step** (operator 2026-11-11: after item 11) | Today K (0-3 guesses) is chosen per request from its recent acceptance. Choose it every step instead: the K with the most expected accepted tokens per millisecond, from the head's own confidence in each guess (calibrated against how often guesses at that position and confidence were accepted) and the measured cost of verifying 1-3 guesses. Confident stretches (code) verify all 3, uncertain ones stop early; output never changes. Idea from basalt (62 -> 83 t/s on code there) | `--mtp auto` with the per-step choice beats today's on the opencode replay and loses nowhere by more than 1 %; greedy output identical |
| 12 | **Qwen3.8: make the DFlash drafter fast** (2026-11-11) | `--spec dflash` is built and correct but lost to MTP (0.88x) on 2026-10-27, when one draft-and-check round cost ~3.9 plain steps; specs 28b-28d have since cut it to ~1.4-1.9 (derived), and the R9700 vLLM fork's round costs ~1.15 at the same 8-row block (its "11 tokens a step" is batch occupancy; it gets ~5 tokens a round, as we do - `docs/probes/dflash-vs-sly-2026-11-11.md`). The plan is `docs/superpowers/plans/2026-11-11-spec19v2-dflash-speed.md`: measure again first, then one submit a round, an int4 drafter, a smaller draft head and the draft list's small kernels | the plan's Task 8: an iteration <= 1.35 plain steps and `--spec dflash` >= 1.10x `--mtp auto` on the opencode replay |
| 4 | **Flash-Next at full size - our own weights, then the expert tier on two cards** ([spec 21 v2](superpowers/specs/2026-11-11-spec21v2-flash-next-v2-design.md), [plan 21v2a](superpowers/plans/2026-11-11-spec21v2a-whole-model-tiering.md), then [21v2b](superpowers/plans/2026-11-12-spec21v2b-tensor-parallel.md)). The operator ruled on 2026-11-12: re-quantise the model ourselves (AutoRound int4 g64 from the 360 GB bf16 original, head and MTP experts included), split the layers first and tensor parallel after, and aim past Strata's 66 t/s on two B70s at 4-bit quality. The whole model needs ~70 GB against ~62 on two cards, so the most-used experts stay in VRAM and the rest are read from pinned host memory | the bf16 original is not on the box yet; next: its download check and the AutoRound toolchain on the tiny model, then sizing the run on one card | `b70-serve` serves the whole model on our weights, judged by the f32 reference, at 30 t/s or more on `pp`; then 50 t/s decode and 1000 t/s prefill on `tp` (21v2b) |
| 5 | **The quiet-box measurement session** - one record-grade speed run at the end, on an idle box and the newest driver. It measures every speed number (tensor parallel, 3l, TP + slots and replica throughput: tonight's numbers ran beside CPU jobs and are not records) and settles the open speed choices: list B (card hand-off copy vs peer, int8 KV, int8 prefill, draft vocabulary, Flash-Next's MTP norm and eager attention: the winner stays, the rest is deleted), 4096-row prefill chunks (+4.2 %, under the +5 % bar: the operator decides), MTP acceptance cold vs a prefix cache hit (< 1 % closes spec 16), and our adaptive MTP depth against saber's rule. The stages: `tools/box_validate.sh --with optin` (`--list` shows them); parked models' stages wait for those models | after everything above it | measured; README.md, docs/BENCHMARKS.md and docs/18-running-models.md refreshed |

## Waiting - only a box run or a CPU job left

| spec | what | proved by | state |
|---|---|---|---|
| **Flash-Next** (next after Qwen3.8) | the golden-set remake with block scores (`r30.golden` / `r30.golden_layers`), which `r32.partial` and `r33.golden_real` wait for (`qwen4exp_partial_test`, `qwen4exp_golden_prefill_intel_test` FAIL on the old 1e-3 gap rule until then); `r34.fixture` / `r34.accept` / `r34.a4_ref` (CPU), then `r34.a4` (needs the whole model: item 4); `r31.load_real`. Passed since 2026-11-07: `r31.load`, `r32.k3` / `golden` / `pp` / `cli`, `r33.prefill` / `pp`. What comes after these is [plan 21v2a](superpowers/plans/2026-11-11-spec21v2a-whole-model-tiering.md) | CPU references (the remake ~90 GB RAM, holds both cards; plan 21v2a Task 7 adds the 48-layer f32 reference of our export to the same job), item 4 | held in the CPU queue (`#HOLD-sj q4exp-golden-scores`) until the Qwen3.8 and TP + slots references are done |
| **Bolt / LiLiCorr** (parked, operator 2026-11-11: until Flash-Next is implemented, tuned, fused and speed-optimised) | `r60.golden` / `r61.golden` judged by Bolt's f32 reference (the tooling is on branch `bolt-f32-ref`; its CPU job held as `#HOLD-bolt-parked-until-flash-next`); `r61.golden`'s conv-column drift (0.987 vs 0.999); `r64.draft` (LiLiCorr's exp now `df_exp`: confirm on a card); `r65.prefill`'s one-position ring cosine (likely a routing near-tie flip); `r65.serve`, `r57.*`, `r62.a4` / `agentic`, `r60.golden32k`, `r61.golden32k` | Bolt's f32 reference, a card run | parked |
| **K2 / Kolibri** (parked) | `r25.a4_ref` / `r25.a4` (waits for K2 bf16 hot, 75 GB), `r21.passkey` (FAILed: read its log first), `r28.a4` (Kolibri's 156 GB bf16 download), `r28.passkey`. Spec 28m (the MoE draft-checking probe) waits for an MoE model in focus | the models back in focus | parked |
| **28** speculative serving | `r2.auto_rows` / `r12.a4` and the stages that compared a speculative arm against `--spec off` (`r6.greedy_long`, `r8.auto_rows`, `r16.mtp_rows`, `r27.cli_mtp`, `r55.serve`, `r55.decode`, `r56.pp_serve`, `r70.d2`, `r70.speed`) now judge each first difference by the f32 judge (`serve_judge.py`); a difference without its reference SKIPs. Their serve contexts' f32 references: the `serve-judge-f32` CPU queue job | the next validation run, then those references | merged (2fb3fd68) |
| **8** MTP | `mtp_head_test`: the MTP head's guesses against a CPU reference of the head, never made (`tools/oracle/mtp_ref.py --dump` from Qwen's bf16 original, on the box; its inputs `oracle-out-spec8b/` do not exist yet either) - SKIPs until then. Checks speed only: the main model verifies every guess | a CPU job (~1-2 h), run beside item 11's card runs, then the test | queued with item 11 |
| **19** DFlash | the CPU half of the drafter reference (`r54.d1_ref`): the 2x-noise near-tie window is enabled (`WALK_TIE_NOISE_X = 2.0`) | `r54.d1_ref` | passes in the re-test |
| *(idea, optional)* | **Is Hadamard-rotated int4 closer to bf16?** nchapman's `Qwen3.8-27B-W4A16-HGPTQ` needs the rotation applied to every quantised layer's input at runtime - our engine does not, so it must NOT be loaded as is. First a CPU comparison against our AutoRound export (KL / top-1 against bf16 on the golden and tool-call prompts); only if clearly closer, the rotation in the engine | CPU job, optional | idea |

## Closed recently

| date | spec | what |
|---|---|---|
| 2026-11-11 | 10 | **TP int8 prefill implemented and validated on `tp-int8-prefill`:** zero-pad down 8704 → 9216 for the 1024 rotation; f32 gate 40/40 contexts PASS (64 tokens, worst difference 6.70 / 8 ulp), negative control PASS, server 39/39 plus default MTP smoke PASS. Idle medians at 4k / 16k: 3323 / 3048 t/s vs TP bf16 2935 / 2673 (+13.2% / +14.0%). The 512-wide alternative was removed; int8 KV keeps its known-fail warning. [Record](probes/tp-int8-prefill-2026-11-11.md) |
| 2026-11-11 | 13 | **Replica with per-card slots passes on the cards:** row 77 all PASS on main (two at a time vs alone, two chat clients, a snapshot served on the other card, disconnects / stop strings / streaming, and 1 / 4 / 8 requests at once with `--max-num-seqs 8`: every difference passes the f32 judge, 1 request awaits its f32 reference). Rough speed from that run (128-token replies, cold prompts): MTP 1 / 4 / 8 requests ~44 / 117 / 132 t/s in total, one request alone 78 t/s while generating; the record run is item 5 |
| 2026-11-11 | 7 | **The prefix cache hang fixed:** the prefix cache is now 2 GiB pinned pieces - one 32 GiB allocation, mapped by the second card through a shared import, could overflow the kernel's 4 GiB mapping segments; the kernel answered "busy" and the GPU driver retried forever. TP bf16 32 GiB, TP int8 16 GiB and replica pass with prefix cache hits (324 hits, 0 hangs) |
| 2026-11-11 | 9 | **`--max-num-seqs N` merged:** one card up to 4 requests, two cards (replica) up to 8; startup prints each card's slots (bf16 cache, one card: 171008 alone, 46080 + 3 x 32768 at N = 4); a too-long `--max-len` and TP with N > 1 are refused; the slot tests pass |
| 2026-11-11 | Qwen3.8 | **Real opencode sessions recorded** (operator, 2026-11-07 / 08): `~/b70/logs/opencode-session-int8` (78 requests), `opencode-session` (21), `opencode-session-2` (4) on the box - the replays (`r2.opencode`, `r12.d5`) take `OPENCODE_LOG=` one of them |
| 2026-11-11 | Qwen3.8 | **The int8 prompt pass is the cause of the 3 accepted `mtp_gpu_kv8_test` rows:** the same test with the bf16 prompt pass (`l0`) and the int8 cache has 0 failing runs; t4_comment-loader is identical to the plain steps at K 1/2/3, t5_test-linear_l0 and t2_explain-openai differ only at near-ties under 0.11 ulp (other positions), so none of the three f32-rejected rows (gaps 20.94 / 23.86 / 13.01) appears |
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
