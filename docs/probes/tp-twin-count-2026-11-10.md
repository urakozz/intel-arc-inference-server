# The tensor-parallel twin count, 2026-11-10 (open-work item 6; decisions D10, D18 -> D3)

Operator 2026-11-10: D11 accepted; D10 (17c: TP prefill, int8 KV, `long` id 62) and D18 (17d: served TP `--spec auto`,
`toolcall/t5_test-loader` row 77) are decided by this count; D3 (`cli::kTpAuto = true`: `b70-serve --parallel auto`
-> tp for Qwen3.8 on two B70s) is switched on only if TP passes.

Why a count: TP (tensor parallel - every layer split over both cards, the halves added after each row-parallel
linear) sums each linear's K in two halves, so its rounding differs from one card's and its greedy ids leave one
card's at a few sensitive rows. One card's own legitimate paths do the same to each other (17c's
`tools/probe/tp_prefill_kv8_drift_17c.sh`: one card's prefill vs its token-by-token path diverge at ids 45 / 76
under int8 KV). The question is not whether TP diverges but whether its divergences land on a wrong id (one the f32
judge rejects) more often than one card's own paths' do.

## The rule (fixed and committed before any result)

**Contexts** (41 prompts): the golden prompts `prose`, `code`, `cjk` (tests/golden/prompts/*.ids), the 36 tool-call
prompts (tests/golden/toolcall/*.ids - with the golden three, the 39 requests 17d's `r81.serve` used), `long`
(2820 ids) and `long32k[:6000]`. Both KV forms: bf16 (default) and `--kv-cache int8`. Greedy, 256 generated ids,
`--max-len 16384`, the int8 head (b70-decode's and b70-serve's).

**Paths** (each one run of the real CLI on the whole box; one card = `--parallel 1`):

| arm | command |
|---|---|
| one-pf / tp-pf | `b70-decode --prefill` (prefill, then plain decode) |
| one-in / tp-in | `b70-decode` without `--prefill` (the prompt token by token: decode replays) |
| one-mtp / tp-mtp | `b70-decode --prefill --spec auto` (MTP, AdaptiveK) |
| serve one / one-auto / tp / tp-auto | `b70-serve --spec off` / `--spec auto`, the golden + tool-call sets in one server (prefix cache on, as `r81.serve`) |

**Pairs** - in each, the first index where the two streams differ (the FIRST divergence only; what follows it is a
different context):

| class | pairs (reference first) |
|---|---|
| TP vs its one-card twin | TP prefill (one-pf, tp-pf); TP plain decode (one-in, tp-in); TP MTP (one-mtp, tp-mtp); served TP `--spec off` (one, tp); served TP `--spec auto` (one-auto, tp-auto) |
| one card's own legitimate path pairs | prefill vs token by token (one-in, one-pf); MTP vs plain (one-pf, one-mtp); served `--spec off` vs `--spec auto` (one, one-auto) |

**The judge.** Each first divergence at index d is one forced context: the prompt + the shared ids [0, d), one forced
id (the reference side's); its f32 reference (`tools/box_validate/f32_refs.sh forced-qwen`, `tools/oracle/qwen_forced.py`:
Qwen3.8 in fp32 on the CPU, the int8 head - key prefix `tp_twin_`) gives row 0, the decision at d. Each side's id is
accepted or rejected by `judge_row`'s rule (golden_common.h / serve_judge.py: the f32 top, or any id within one bf16
ulp of it when the top-2 gap is within one ulp - the 1-bf16-ulp near-tie). A rejected id is printed with its gap
below the f32 top in bf16 ulp of the top.

**The count.**
- TP's rejected count: the TP-side ids the judge rejects at the first divergences of the TP twin pairs.
- One card's own-path rejected count: the ids (either side - both are one card's legitimate paths) the judge rejects
  at the first divergences of one card's own pairs.
- **TP passes if its rejected count is no more than one card's own-path rejected count** (both KV forms and every
  path together). Reported beside it, per path and KV form: the pairs compared, the divergences, the rejections, and
  the rate per divergence and per pair - the two classes have different numbers of pairs (5 TP pair kinds, 3 own),
  so the raw count is the verdict (the operator's rule) and the normalised rates are its fairness caveat; if the two
  disagree, that goes to the operator with the verdict. The one-card-side rejections inside the TP pairs (one card
  wrong where TP is right) are reported too, not counted on either side.
- A divergence whose reference is missing is not counted on either side; the record says so (none may remain at
  the verdict).
- D11 / D18 by name: D11's row is a TP-MTP vs plain-TP difference (17d), D18's row 77 follows an accepted first
  difference at index 10 - neither is a first divergence of these pairs by construction; where the same rows recur as
  first divergences here, they are counted like any other.

Tools: `tools/probe/tp_twin_count.sh` (the arms, on the box, under `tools/gpu_lock.sh box`), records through
`serve_client.py record` / `serve_run.sh`; `tools/box_validate/twin_count.py` (first divergences, the dumps, the
judge, the tables).

## Results

**The arms** (2026-11-10 09:33-15:32, box `~/b70/twin-count/run1`, log `~/b70/logs/twin-count/arms.log`): every
b70-decode run and every server arm completed, none failed - one card under `gpu_lock.sh card 1`, TP under
`gpu_lock.sh box`. 154 distinct forced contexts dumped to `oracle-forced-dumps/tp-twin` (the same divergence reached
by two pairs is one context).

| class | path | KV | pairs | identical | first divergences |
|---|---|---|---:|---:|---:|
| TP | prefill | bf16 / int8 | 41 / 41 | 15 / 16 | 26 / 25 |
| TP | plain decode | bf16 / int8 | 41 / 41 | 12 / 12 | 29 / 29 |
| TP | MTP | bf16 / int8 | 41 / 41 | 19 / 14 | 22 / 27 |
| TP | served `--spec off` | bf16 / int8 | 39 / 39 | 15 / 16 | 24 / 23 |
| TP | served `--spec auto` | bf16 / int8 | 39 / 39 | 19 / 14 | 20 / 25 |
| own | prefill vs token by token | bf16 / int8 | 41 / 41 | 12 / 14 | 29 / 27 |
| own | MTP vs plain | bf16 / int8 | 41 / 41 | 22 / 19 | 19 / 22 |
| own | served off vs auto | bf16 / int8 | 39 / 39 | 22 / 19 | 17 / 20 |

Totals: TP 402 pairs, 250 first divergences; one card's own 242 pairs, 134 first divergences. Per pair the classes
diverge alike (TP 0.62, own 0.55): in 256 greedy ids a legitimate rounding change leaves the stream about as often
as TP does.

**The f32 references** (`f32_refs.sh forced-qwen`, QWEN_KEYS_RE `^tp_twin_`, 15:44-16:56, 4287 s, peak RSS 36.1 GiB;
it waited for 95 GB of free RAM first): 154 judges in `oracle-out-qwen-f32-i8head-forced`. `twin_count.py`'s output:
`~/b70/twin-count/run1/count.txt` / `count.json`.

| class | path | KV | first divergences | both ids accepted | rejected (counted) | the one-card side rejected (not counted) |
|---|---|---|---:|---:|---:|---:|
| TP | prefill | bf16 / int8 | 26 / 25 | 11 / 11 | 6 / 9 | 9 / 7 |
| TP | plain decode | bf16 / int8 | 29 / 29 | 15 / 11 | 9 / 11 | 6 / 7 |
| TP | MTP | bf16 / int8 | 22 / 27 | 11 / 13 | 8 / 6 | 4 / 9 |
| TP | served `--spec off` | bf16 / int8 | 24 / 23 | 10 / 10 | 6 / 8 | 8 / 7 |
| TP | served `--spec auto` | bf16 / int8 | 20 / 25 | 10 / 12 | 7 / 6 | 4 / 8 |
| own | prefill vs token by token | bf16 / int8 | 29 / 27 | 13 / 17 | 17 / 10 | - |
| own | MTP vs plain | bf16 / int8 | 19 / 22 | 8 / 10 | 11 / 13 | - |
| own | served off vs auto | bf16 / int8 | 17 / 20 | 6 / 9 | 11 / 12 | - |

| class | KV | pairs | first divergences | rejected | per divergence | per pair |
|---|---|---:|---:|---:|---:|---:|
| TP | bf16 / int8 / both | 201 / 201 / 402 | 121 / 129 / 250 | 36 / 40 / **76** | 0.298 / 0.310 / **0.304** | 0.179 / 0.199 / **0.189** |
| own | bf16 / int8 / both | 121 / 121 / 242 | 65 / 69 / 134 | 39 / 35 / **74** | 0.600 / 0.507 / **0.552** | 0.322 / 0.289 / **0.306** |

**Verdict by the committed rule: FAIL by two** - TP's rejected count 76 > one card's own 74. **The normalised rates
disagree** (to the operator, as the rule says): per first divergence TP's id is rejected 0.30 times, one card's own
paths' 0.55; per pair compared 0.19 against 0.31.

**Which TP path carries the excess: none.** Path by path TP is rejected less often than the matching own pair (both
KV forms together): TP prefill 15, TP plain decode 20, TP MTP 14, served TP off 14, served TP auto 13, against one
card's prefill vs token by token 27, MTP vs plain 24, served off vs auto 23. The raw total exceeds one card's only
because TP has five pair kinds against three (402 pairs against 242). Further facts:
- At the TP-vs-one-card divergences the judge rejects TP's id 76 times and one card's 69 times: the two sides are
  about equally often wrong where they part - what two legitimate roundings do, not a TP fault.
- The rejected ids' gaps: TP side median 6.9 ulp (34 of 76 above 8 ulp, the largest 50.3), one-card side in TP pairs
  median 4.8 (27 of 69 above 8, the largest 57.9), own pairs median 6.3 (29 of 74 above 8, the largest 113.6).
- Counted by distinct context (the same divergence reached by several pairs - e.g. one prefill divergence shows in TP
  prefill, served TP off and the MTP pairs - counted once): TP 36 distinct rejected ids, one card's own 46.
- D10's row (`long` id 62, 271 vs the f32 top 198, 22.96 ulp): one card itself takes the rejected 271 there - int8
  KV MTP vs plain on one card (one-mtp 271), and bf16 one-card prefill takes 271 at id 124 (40.97 ulp below 95802,
  where token-by-token takes 96079, 113.58 below). D18's request (`t5_test-loader`): the first divergences of every
  pair on it are near-misses of 0.4 - 9.0 ulp on both sides (one card's own `--spec off` arm is the rejected side at
  index 89, 2.69 ulp; served one-card off vs auto at int8 186, 9.01) - row 77 is not a first divergence here.

Every rejected row (TP's counted, the one-card side in TP pairs marked "not counted", both sides of own pairs):

| class | KV | path | context | index | rejected (arm id: ulp below the f32 top) | f32 top | the other side |
|---|---|---|---|---:|---|---:|---|
| TP | bf16 | TP MTP | test/long | 62 | tp-mtp 271: 22.96 | 198 | one-mtp accepted |
| TP | bf16 | TP MTP | toolcall/t2_explain-loader | 32 | one-mtp 198: 2.55 (not counted) | 248068 | tp-mtp accepted |
| TP | bf16 | TP MTP | toolcall/t2_explain-rotation | 84 | tp-mtp 74455: 10.82 | 27345 | one-mtp accepted |
| TP | bf16 | TP MTP | toolcall/t4_comment-box | 146 | tp-mtp 29: 7.03 | 248068 | one-mtp accepted |
| TP | bf16 | TP MTP | toolcall/t4_comment-linear_l0 | 112 | tp-mtp 248068: 14.86 | 74455 | one-mtp accepted |
| TP | bf16 | TP MTP | toolcall/t4_comment-rotation | 248 | tp-mtp 248045: 5.87; one-mtp 846: 3.92 (not counted) | 248046 | both rejected |
| TP | bf16 | TP MTP | toolcall/t5_test-loader | 10 | tp-mtp 79114: 2.59 | 28 | one-mtp accepted |
| TP | bf16 | TP MTP | toolcall/t5_test-openai | 45 | tp-mtp 248068: 4.47 | 846 | one-mtp accepted |
| TP | bf16 | TP MTP | toolcall/t5_test-step | 237 | one-mtp 846: 4.38 (not counted) | 248045 | tp-mtp accepted |
| TP | bf16 | TP MTP | toolcall/t6_usage-box | 37 | tp-mtp 248046: 2.28 | 248068 | one-mtp accepted |
| TP | bf16 | TP MTP | toolcall/t6_usage-linear_l0 | 41 | one-mtp 198: 9.38 (not counted) | 21966 | tp-mtp accepted |
| TP | int8 | TP MTP | test/long | 62 | one-mtp 271: 22.96 (not counted) | 198 | tp-mtp accepted |
| TP | int8 | TP MTP | toolcall/t2_explain-loader | 32 | tp-mtp 4560: 8.12 | 248068 | one-mtp accepted |
| TP | int8 | TP MTP | toolcall/t2_explain-openai | 46 | one-mtp 846: 1.38 (not counted) | 5097 | tp-mtp accepted |
| TP | int8 | TP MTP | toolcall/t2_explain-rotation | 201 | tp-mtp 248046: 32.78 | 74455 | one-mtp accepted |
| TP | int8 | TP MTP | toolcall/t3_rename-step | 198 | one-mtp 248068: 13.68 (not counted) | 846 | tp-mtp accepted |
| TP | int8 | TP MTP | toolcall/t4_comment-box | 147 | one-mtp 198: 7.42 (not counted) | 248068 | tp-mtp accepted |
| TP | int8 | TP MTP | toolcall/t4_comment-linear_l0 | 112 | one-mtp 248068: 14.86 (not counted) | 74455 | tp-mtp accepted |
| TP | int8 | TP MTP | toolcall/t4_comment-openai | 45 | one-mtp 846: 6.26 (not counted) | 248068 | tp-mtp accepted |
| TP | int8 | TP MTP | toolcall/t5_test-loader | 89 | tp-mtp 1669: 2.69 | 248046 | one-mtp accepted |
| TP | int8 | TP MTP | toolcall/t5_test-rotation | 47 | tp-mtp 248045: 6.59 | 846 | one-mtp accepted |
| TP | int8 | TP MTP | toolcall/t5_test-step | 237 | one-mtp 846: 4.38 (not counted) | 248045 | tp-mtp accepted |
| TP | int8 | TP MTP | toolcall/t6_usage-linear_l0 | 41 | tp-mtp 198: 9.38; one-mtp 5920: 4.05 (not counted) | 21966 | both rejected |
| TP | int8 | TP MTP | toolcall/t6_usage-loader | 41 | one-mtp 248068: 4.79 (not counted) | 198 | tp-mtp accepted |
| TP | int8 | TP MTP | toolcall/t6_usage-step | 253 | tp-mtp 248046: 1.21 | 248068 | one-mtp accepted |
| TP | bf16 | TP plain decode | test/long | 61 | tp-in 271: 50.26 | 25 | one-in accepted |
| TP | bf16 | TP plain decode | toolcall/t1_define-linear_l0 | 175 | tp-in 248046: 3.68 | 248068 | one-in accepted |
| TP | bf16 | TP plain decode | toolcall/t1_define-rotation | 200 | tp-in 248068: 38.14 | 248046 | one-in accepted |
| TP | bf16 | TP plain decode | toolcall/t2_explain-loader | 32 | one-in 61369: 57.88 (not counted) | 248068 | tp-in accepted |
| TP | bf16 | TP plain decode | toolcall/t2_explain-openai | 46 | one-in 846: 1.38 (not counted) | 5097 | tp-in accepted |
| TP | bf16 | TP plain decode | toolcall/t2_explain-rotation | 84 | tp-in 51319: 2.19; one-in 74455: 10.82 (not counted) | 27345 | both rejected |
| TP | bf16 | TP plain decode | toolcall/t3_rename-step | 115 | tp-in 846: 14.87 | 248045 | one-in accepted |
| TP | bf16 | TP plain decode | toolcall/t4_comment-loader | 157 | tp-in 74455: 1.62 | 846 | one-in accepted |
| TP | bf16 | TP plain decode | toolcall/t4_comment-openai | 45 | one-in 846: 6.26 (not counted) | 248068 | tp-in accepted |
| TP | bf16 | TP plain decode | toolcall/t4_comment-rotation | 248 | one-in 846: 3.92 (not counted) | 248046 | tp-in accepted |
| TP | bf16 | TP plain decode | toolcall/t5_test-box | 212 | one-in 248046: 10.52 (not counted) | 248068 | tp-in accepted |
| TP | bf16 | TP plain decode | toolcall/t5_test-openai | 45 | tp-in 248068: 4.47 | 846 | one-in accepted |
| TP | bf16 | TP plain decode | toolcall/t6_usage-box | 31 | tp-in 846: 13.61 | 248068 | one-in accepted |
| TP | bf16 | TP plain decode | toolcall/t6_usage-linear_l0 | 41 | tp-in 198: 9.38 | 21966 | one-in accepted |
| TP | int8 | TP plain decode | test/long | 125 | one-in 271: 22.67 (not counted) | 96755 | tp-in accepted |
| TP | int8 | TP plain decode | toolcall/t1_define-box | 114 | tp-in 846: 8.31 | 74455 | one-in accepted |
| TP | int8 | TP plain decode | toolcall/t1_define-linear_l0 | 91 | one-in 248068: 11.24 (not counted) | 846 | tp-in accepted |
| TP | int8 | TP plain decode | toolcall/t1_define-openai | 40 | tp-in 846: 31.11 | 248045 | one-in accepted |
| TP | int8 | TP plain decode | toolcall/t1_define-rotation | 200 | one-in 248068: 38.14 (not counted) | 248046 | tp-in accepted |
| TP | int8 | TP plain decode | toolcall/t2_explain-linear_l0 | 10 | tp-in 28: 6.70 | 79114 | one-in accepted |
| TP | int8 | TP plain decode | toolcall/t2_explain-loader | 32 | tp-in 248046: 30.67 | 248068 | one-in accepted |
| TP | int8 | TP plain decode | toolcall/t2_explain-rotation | 84 | tp-in 74455: 10.82 | 27345 | one-in accepted |
| TP | int8 | TP plain decode | toolcall/t2_explain-step | 10 | tp-in 28: 8.05 | 79114 | one-in accepted |
| TP | int8 | TP plain decode | toolcall/t3_rename-rotation | 81 | tp-in 12244: 1.07 | 4364 | one-in accepted |
| TP | int8 | TP plain decode | toolcall/t3_rename-step | 198 | one-in 74455: 2.37 (not counted) | 846 | tp-in accepted |
| TP | int8 | TP plain decode | toolcall/t4_comment-openai | 45 | tp-in 846: 6.26 | 248068 | one-in accepted |
| TP | int8 | TP plain decode | toolcall/t4_comment-rotation | 248 | one-in 846: 3.92 (not counted) | 248046 | tp-in accepted |
| TP | int8 | TP plain decode | toolcall/t5_test-box | 79 | one-in 846: 36.51 (not counted) | 248045 | tp-in accepted |
| TP | int8 | TP plain decode | toolcall/t5_test-linear_l0 | 142 | tp-in 248058: 1.21 | 248068 | one-in accepted |
| TP | int8 | TP plain decode | toolcall/t6_usage-linear_l0 | 41 | tp-in 198: 9.38 | 21966 | one-in accepted |
| TP | int8 | TP plain decode | toolcall/t6_usage-loader | 41 | one-in 248068: 4.79 (not counted) | 198 | tp-in accepted |
| TP | int8 | TP plain decode | toolcall/t6_usage-step | 61 | tp-in 846: 8.55 | 198 | one-in accepted |
| TP | bf16 | TP prefill | test/long | 124 | one-pf 271: 40.97 (not counted) | 95802 | tp-pf accepted |
| TP | bf16 | TP prefill | toolcall/t1_define-step | 138 | one-pf 248068: 2.76 (not counted) | 248046 | tp-pf accepted |
| TP | bf16 | TP prefill | toolcall/t2_explain-loader | 228 | one-pf 248045: 1.22 (not counted) | 248068 | tp-pf accepted |
| TP | bf16 | TP prefill | toolcall/t2_explain-openai | 144 | one-pf 248045: 8.02 (not counted) | 846 | tp-pf accepted |
| TP | bf16 | TP prefill | toolcall/t2_explain-rotation | 84 | tp-pf 74455: 10.82 | 27345 | one-pf accepted |
| TP | bf16 | TP prefill | toolcall/t2_explain-step | 200 | tp-pf 248068: 3.50 | 198 | one-pf accepted |
| TP | bf16 | TP prefill | toolcall/t3_rename-step | 115 | one-pf 846: 14.87 (not counted) | 248045 | tp-pf accepted |
| TP | bf16 | TP prefill | toolcall/t4_comment-box | 146 | one-pf 29: 7.03 (not counted) | 248068 | tp-pf accepted |
| TP | bf16 | TP prefill | toolcall/t4_comment-loader | 157 | one-pf 74455: 1.62 (not counted) | 846 | tp-pf accepted |
| TP | bf16 | TP prefill | toolcall/t4_comment-openai | 45 | tp-pf 846: 6.26 | 248068 | one-pf accepted |
| TP | bf16 | TP prefill | toolcall/t5_test-loader | 89 | one-pf 1669: 2.69 (not counted) | 248046 | tp-pf accepted |
| TP | bf16 | TP prefill | toolcall/t5_test-openai | 45 | tp-pf 248068: 4.47 | 846 | one-pf accepted |
| TP | bf16 | TP prefill | toolcall/t5_test-step | 237 | tp-pf 846: 4.38 | 248045 | one-pf accepted |
| TP | bf16 | TP prefill | toolcall/t6_usage-linear_l0 | 41 | one-pf 198: 9.38 (not counted) | 21966 | tp-pf accepted |
| TP | bf16 | TP prefill | toolcall/t6_usage-loader | 41 | tp-pf 248068: 4.79 | 198 | one-pf accepted |
| TP | int8 | TP prefill | test/long | 62 | tp-pf 271: 22.96 | 198 | one-pf accepted |
| TP | int8 | TP prefill | toolcall/t1_define-linear_l0 | 91 | tp-pf 74455: 4.36; one-pf 248068: 11.24 (not counted) | 846 | both rejected |
| TP | int8 | TP prefill | toolcall/t1_define-rotation | 200 | tp-pf 248068: 38.14 | 248046 | one-pf accepted |
| TP | int8 | TP prefill | toolcall/t2_explain-loader | 32 | tp-pf 4560: 8.12 | 248068 | one-pf accepted |
| TP | int8 | TP prefill | toolcall/t2_explain-openai | 46 | one-pf 846: 1.38 (not counted) | 5097 | tp-pf accepted |
| TP | int8 | TP prefill | toolcall/t2_explain-rotation | 201 | one-pf 248046: 32.78 (not counted) | 74455 | tp-pf accepted |
| TP | int8 | TP prefill | toolcall/t3_rename-step | 198 | one-pf 74455: 2.37 (not counted) | 846 | tp-pf accepted |
| TP | int8 | TP prefill | toolcall/t4_comment-box | 146 | tp-pf 29: 7.03 | 248068 | one-pf accepted |
| TP | int8 | TP prefill | toolcall/t4_comment-openai | 45 | tp-pf 846: 6.26 | 248068 | one-pf accepted |
| TP | int8 | TP prefill | toolcall/t5_test-loader | 167 | one-pf 27690: 1.12 (not counted) | 3425 | tp-pf accepted |
| TP | int8 | TP prefill | toolcall/t5_test-openai | 91 | tp-pf 1877: 14.88 | 248068 | one-pf accepted |
| TP | int8 | TP prefill | toolcall/t5_test-rotation | 47 | tp-pf 248045: 6.59 | 846 | one-pf accepted |
| TP | int8 | TP prefill | toolcall/t6_usage-linear_l0 | 41 | tp-pf 198: 9.38; one-pf 5920: 4.05 (not counted) | 21966 | both rejected |
| TP | int8 | TP prefill | toolcall/t6_usage-loader | 219 | one-pf 248068: 10.92 (not counted) | 248044 | tp-pf accepted |
| TP | bf16 | served TP --spec auto | toolcall/t2_explain-loader | 32 | one-auto 198: 2.55 (not counted) | 248068 | tp-auto accepted |
| TP | bf16 | served TP --spec auto | toolcall/t2_explain-rotation | 84 | tp-auto 74455: 10.82 | 27345 | one-auto accepted |
| TP | bf16 | served TP --spec auto | toolcall/t4_comment-box | 146 | tp-auto 29: 7.03 | 248068 | one-auto accepted |
| TP | bf16 | served TP --spec auto | toolcall/t4_comment-linear_l0 | 112 | tp-auto 248068: 14.86 | 74455 | one-auto accepted |
| TP | bf16 | served TP --spec auto | toolcall/t4_comment-rotation | 248 | tp-auto 248045: 5.87; one-auto 846: 3.92 (not counted) | 248046 | both rejected |
| TP | bf16 | served TP --spec auto | toolcall/t5_test-loader | 10 | tp-auto 79114: 2.59 | 28 | one-auto accepted |
| TP | bf16 | served TP --spec auto | toolcall/t5_test-openai | 45 | tp-auto 248068: 4.47 | 846 | one-auto accepted |
| TP | bf16 | served TP --spec auto | toolcall/t5_test-step | 237 | one-auto 846: 4.38 (not counted) | 248045 | tp-auto accepted |
| TP | bf16 | served TP --spec auto | toolcall/t6_usage-box | 37 | tp-auto 248046: 2.28 | 248068 | one-auto accepted |
| TP | bf16 | served TP --spec auto | toolcall/t6_usage-linear_l0 | 41 | one-auto 198: 9.38 (not counted) | 21966 | tp-auto accepted |
| TP | int8 | served TP --spec auto | toolcall/t2_explain-loader | 32 | tp-auto 4560: 8.12 | 248068 | one-auto accepted |
| TP | int8 | served TP --spec auto | toolcall/t2_explain-openai | 46 | one-auto 846: 1.38 (not counted) | 5097 | tp-auto accepted |
| TP | int8 | served TP --spec auto | toolcall/t2_explain-rotation | 201 | tp-auto 248046: 32.78 | 74455 | one-auto accepted |
| TP | int8 | served TP --spec auto | toolcall/t3_rename-step | 198 | one-auto 248068: 13.68 (not counted) | 846 | tp-auto accepted |
| TP | int8 | served TP --spec auto | toolcall/t4_comment-box | 147 | one-auto 198: 7.42 (not counted) | 248068 | tp-auto accepted |
| TP | int8 | served TP --spec auto | toolcall/t4_comment-linear_l0 | 112 | one-auto 248068: 14.86 (not counted) | 74455 | tp-auto accepted |
| TP | int8 | served TP --spec auto | toolcall/t4_comment-openai | 45 | one-auto 846: 6.26 (not counted) | 248068 | tp-auto accepted |
| TP | int8 | served TP --spec auto | toolcall/t5_test-loader | 89 | tp-auto 1669: 2.69 | 248046 | one-auto accepted |
| TP | int8 | served TP --spec auto | toolcall/t5_test-rotation | 47 | tp-auto 248045: 6.59 | 846 | one-auto accepted |
| TP | int8 | served TP --spec auto | toolcall/t5_test-step | 237 | one-auto 846: 4.38 (not counted) | 248045 | tp-auto accepted |
| TP | int8 | served TP --spec auto | toolcall/t6_usage-linear_l0 | 41 | tp-auto 198: 9.38; one-auto 5920: 4.05 (not counted) | 21966 | both rejected |
| TP | int8 | served TP --spec auto | toolcall/t6_usage-loader | 41 | one-auto 248068: 4.79 (not counted) | 198 | tp-auto accepted |
| TP | int8 | served TP --spec auto | toolcall/t6_usage-step | 253 | tp-auto 248046: 1.21 | 248068 | one-auto accepted |
| TP | bf16 | served TP --spec off | toolcall/t1_define-step | 138 | one 248068: 2.76 (not counted) | 248046 | tp accepted |
| TP | bf16 | served TP --spec off | toolcall/t2_explain-loader | 228 | one 248045: 1.22 (not counted) | 248068 | tp accepted |
| TP | bf16 | served TP --spec off | toolcall/t2_explain-openai | 144 | one 248045: 8.02 (not counted) | 846 | tp accepted |
| TP | bf16 | served TP --spec off | toolcall/t2_explain-rotation | 84 | tp 74455: 10.82 | 27345 | one accepted |
| TP | bf16 | served TP --spec off | toolcall/t2_explain-step | 200 | tp 248068: 3.50 | 198 | one accepted |
| TP | bf16 | served TP --spec off | toolcall/t3_rename-step | 115 | one 846: 14.87 (not counted) | 248045 | tp accepted |
| TP | bf16 | served TP --spec off | toolcall/t4_comment-box | 146 | one 29: 7.03 (not counted) | 248068 | tp accepted |
| TP | bf16 | served TP --spec off | toolcall/t4_comment-loader | 157 | one 74455: 1.62 (not counted) | 846 | tp accepted |
| TP | bf16 | served TP --spec off | toolcall/t4_comment-openai | 45 | tp 846: 6.26 | 248068 | one accepted |
| TP | bf16 | served TP --spec off | toolcall/t5_test-loader | 89 | one 1669: 2.69 (not counted) | 248046 | tp accepted |
| TP | bf16 | served TP --spec off | toolcall/t5_test-openai | 45 | tp 248068: 4.47 | 846 | one accepted |
| TP | bf16 | served TP --spec off | toolcall/t5_test-step | 237 | tp 846: 4.38 | 248045 | one accepted |
| TP | bf16 | served TP --spec off | toolcall/t6_usage-linear_l0 | 41 | one 198: 9.38 (not counted) | 21966 | tp accepted |
| TP | bf16 | served TP --spec off | toolcall/t6_usage-loader | 41 | tp 248068: 4.79 | 198 | one accepted |
| TP | int8 | served TP --spec off | toolcall/t1_define-linear_l0 | 91 | tp 74455: 4.36; one 248068: 11.24 (not counted) | 846 | both rejected |
| TP | int8 | served TP --spec off | toolcall/t1_define-rotation | 200 | tp 248068: 38.14 | 248046 | one accepted |
| TP | int8 | served TP --spec off | toolcall/t2_explain-loader | 32 | tp 4560: 8.12 | 248068 | one accepted |
| TP | int8 | served TP --spec off | toolcall/t2_explain-openai | 46 | one 846: 1.38 (not counted) | 5097 | tp accepted |
| TP | int8 | served TP --spec off | toolcall/t2_explain-rotation | 201 | one 248046: 32.78 (not counted) | 74455 | tp accepted |
| TP | int8 | served TP --spec off | toolcall/t3_rename-step | 198 | one 74455: 2.37 (not counted) | 846 | tp accepted |
| TP | int8 | served TP --spec off | toolcall/t4_comment-box | 146 | tp 29: 7.03 | 248068 | one accepted |
| TP | int8 | served TP --spec off | toolcall/t4_comment-openai | 45 | tp 846: 6.26 | 248068 | one accepted |
| TP | int8 | served TP --spec off | toolcall/t5_test-loader | 167 | one 27690: 1.12 (not counted) | 3425 | tp accepted |
| TP | int8 | served TP --spec off | toolcall/t5_test-openai | 91 | tp 1877: 14.88 | 248068 | one accepted |
| TP | int8 | served TP --spec off | toolcall/t5_test-rotation | 47 | tp 248045: 6.59 | 846 | one accepted |
| TP | int8 | served TP --spec off | toolcall/t6_usage-linear_l0 | 41 | tp 198: 9.38; one 5920: 4.05 (not counted) | 21966 | both rejected |
| TP | int8 | served TP --spec off | toolcall/t6_usage-loader | 219 | one 248068: 10.92 (not counted) | 248044 | tp accepted |
| own | bf16 | MTP vs plain | toolcall/t1_define-step | 138 | one-pf 248068: 2.76 | 248046 | one-mtp accepted |
| own | bf16 | MTP vs plain | toolcall/t2_explain-loader | 32 | one-mtp 198: 2.55 | 248068 | one-pf accepted |
| own | bf16 | MTP vs plain | toolcall/t2_explain-openai | 144 | one-pf 248045: 8.02 | 846 | one-mtp accepted |
| own | bf16 | MTP vs plain | toolcall/t3_rename-step | 115 | one-pf 846: 14.87 | 248045 | one-mtp accepted |
| own | bf16 | MTP vs plain | toolcall/t4_comment-box | 146 | one-pf 29: 7.03 | 248068 | one-mtp accepted |
| own | bf16 | MTP vs plain | toolcall/t4_comment-linear_l0 | 112 | one-pf 248068: 14.86 | 74455 | one-mtp accepted |
| own | bf16 | MTP vs plain | toolcall/t4_comment-loader | 157 | one-pf 74455: 1.62 | 846 | one-mtp accepted |
| own | bf16 | MTP vs plain | toolcall/t4_comment-openai | 45 | one-mtp 846: 6.26 | 248068 | one-pf accepted |
| own | bf16 | MTP vs plain | toolcall/t5_test-loader | 89 | one-pf 1669: 2.69 | 248046 | one-mtp accepted |
| own | bf16 | MTP vs plain | toolcall/t5_test-step | 237 | one-mtp 846: 4.38 | 248045 | one-pf accepted |
| own | bf16 | MTP vs plain | toolcall/t6_usage-box | 37 | one-pf 248046: 2.28 | 248068 | one-mtp accepted |
| own | int8 | MTP vs plain | test/long | 62 | one-mtp 271: 22.96 | 198 | one-pf accepted |
| own | int8 | MTP vs plain | toolcall/t2_explain-loader | 228 | one-pf 248045: 1.22 | 248068 | one-mtp accepted |
| own | int8 | MTP vs plain | toolcall/t2_explain-rotation | 201 | one-pf 248046: 32.78 | 74455 | one-mtp accepted |
| own | int8 | MTP vs plain | toolcall/t3_rename-step | 198 | one-mtp 248068: 13.68; one-pf 74455: 2.37 | 846 | both rejected |
| own | int8 | MTP vs plain | toolcall/t4_comment-box | 146 | one-mtp 29: 7.03 | 248068 | one-pf accepted |
| own | int8 | MTP vs plain | toolcall/t4_comment-linear_l0 | 112 | one-mtp 248068: 14.86 | 74455 | one-pf accepted |
| own | int8 | MTP vs plain | toolcall/t4_comment-openai | 45 | one-mtp 846: 6.26 | 248068 | one-pf accepted |
| own | int8 | MTP vs plain | toolcall/t5_test-loader | 186 | one-pf 248046: 9.01 | 198 | one-mtp accepted |
| own | int8 | MTP vs plain | toolcall/t5_test-openai | 45 | one-pf 248068: 4.47 | 846 | one-mtp accepted |
| own | int8 | MTP vs plain | toolcall/t5_test-step | 237 | one-mtp 846: 4.38 | 248045 | one-pf accepted |
| own | int8 | MTP vs plain | toolcall/t6_usage-linear_l0 | 99 | one-pf 248046: 5.69 | 248068 | one-mtp accepted |
| own | int8 | MTP vs plain | toolcall/t6_usage-loader | 41 | one-mtp 248068: 4.79 | 198 | one-pf accepted |
| own | bf16 | prefill vs token by token | test/long | 124 | one-pf 271: 40.97; one-in 96079: 113.58 | 95802 | both rejected |
| own | bf16 | prefill vs token by token | toolcall/t1_define-linear_l0 | 91 | one-pf 248068: 11.24 | 846 | one-in accepted |
| own | bf16 | prefill vs token by token | toolcall/t1_define-rotation | 200 | one-pf 248068: 38.14 | 248046 | one-in accepted |
| own | bf16 | prefill vs token by token | toolcall/t1_define-step | 138 | one-pf 248068: 2.76 | 248046 | one-in accepted |
| own | bf16 | prefill vs token by token | toolcall/t2_explain-loader | 32 | one-in 61369: 57.88 | 248068 | one-pf accepted |
| own | bf16 | prefill vs token by token | toolcall/t2_explain-openai | 46 | one-in 846: 1.38 | 5097 | one-pf accepted |
| own | bf16 | prefill vs token by token | toolcall/t2_explain-rotation | 84 | one-in 74455: 10.82 | 27345 | one-pf accepted |
| own | bf16 | prefill vs token by token | toolcall/t3_rename-step | 115 | one-pf 846: 14.87 | 248045 | one-in accepted |
| own | bf16 | prefill vs token by token | toolcall/t4_comment-box | 146 | one-pf 29: 7.03 | 248068 | one-in accepted |
| own | bf16 | prefill vs token by token | toolcall/t4_comment-linear_l0 | 112 | one-pf 248068: 14.86 | 74455 | one-in accepted |
| own | bf16 | prefill vs token by token | toolcall/t4_comment-loader | 157 | one-pf 74455: 1.62 | 846 | one-in accepted |
| own | bf16 | prefill vs token by token | toolcall/t4_comment-openai | 45 | one-in 846: 6.26 | 248068 | one-pf accepted |
| own | bf16 | prefill vs token by token | toolcall/t5_test-openai | 233 | one-in 248068: 1.39 | 248045 | one-pf accepted |
| own | bf16 | prefill vs token by token | toolcall/t5_test-step | 237 | one-in 846: 4.38 | 248045 | one-pf accepted |
| own | bf16 | prefill vs token by token | toolcall/t6_usage-box | 31 | one-pf 846: 13.61 | 248068 | one-in accepted |
| own | bf16 | prefill vs token by token | toolcall/t6_usage-linear_l0 | 41 | one-pf 198: 9.38 | 21966 | one-in accepted |
| own | int8 | prefill vs token by token | test/long | 125 | one-in 271: 22.67 | 96755 | one-pf accepted |
| own | int8 | prefill vs token by token | toolcall/t1_define-linear_l0 | 93 | one-in 40: 1.77 | 2665 | one-pf accepted |
| own | int8 | prefill vs token by token | toolcall/t1_define-openai | 196 | one-in 846: 37.42 | 248045 | one-pf accepted |
| own | int8 | prefill vs token by token | toolcall/t1_define-rotation | 200 | one-in 248068: 38.14 | 248046 | one-pf accepted |
| own | int8 | prefill vs token by token | toolcall/t2_explain-openai | 46 | one-pf 846: 1.38 | 5097 | one-in accepted |
| own | int8 | prefill vs token by token | toolcall/t5_test-box | 79 | one-in 846: 36.51 | 248045 | one-pf accepted |
| own | int8 | prefill vs token by token | toolcall/t5_test-openai | 45 | one-pf 248068: 4.47 | 846 | one-in accepted |
| own | int8 | prefill vs token by token | toolcall/t5_test-step | 237 | one-in 846: 4.38 | 248045 | one-pf accepted |
| own | int8 | prefill vs token by token | toolcall/t6_usage-linear_l0 | 41 | one-pf 5920: 4.05 | 21966 | one-in accepted |
| own | int8 | prefill vs token by token | toolcall/t6_usage-loader | 41 | one-in 248068: 4.79 | 198 | one-pf accepted |
| own | bf16 | served --spec off vs --spec auto | toolcall/t1_define-step | 138 | one 248068: 2.76 | 248046 | one-auto accepted |
| own | bf16 | served --spec off vs --spec auto | toolcall/t2_explain-loader | 32 | one-auto 198: 2.55 | 248068 | one accepted |
| own | bf16 | served --spec off vs --spec auto | toolcall/t2_explain-openai | 144 | one 248045: 8.02 | 846 | one-auto accepted |
| own | bf16 | served --spec off vs --spec auto | toolcall/t3_rename-step | 115 | one 846: 14.87 | 248045 | one-auto accepted |
| own | bf16 | served --spec off vs --spec auto | toolcall/t4_comment-box | 146 | one 29: 7.03 | 248068 | one-auto accepted |
| own | bf16 | served --spec off vs --spec auto | toolcall/t4_comment-linear_l0 | 112 | one 248068: 14.86 | 74455 | one-auto accepted |
| own | bf16 | served --spec off vs --spec auto | toolcall/t4_comment-loader | 157 | one 74455: 1.62 | 846 | one-auto accepted |
| own | bf16 | served --spec off vs --spec auto | toolcall/t4_comment-openai | 45 | one-auto 846: 6.26 | 248068 | one accepted |
| own | bf16 | served --spec off vs --spec auto | toolcall/t5_test-loader | 89 | one 1669: 2.69 | 248046 | one-auto accepted |
| own | bf16 | served --spec off vs --spec auto | toolcall/t5_test-step | 237 | one-auto 846: 4.38 | 248045 | one accepted |
| own | bf16 | served --spec off vs --spec auto | toolcall/t6_usage-box | 37 | one 248046: 2.28 | 248068 | one-auto accepted |
| own | int8 | served --spec off vs --spec auto | toolcall/t2_explain-loader | 228 | one 248045: 1.22 | 248068 | one-auto accepted |
| own | int8 | served --spec off vs --spec auto | toolcall/t2_explain-rotation | 201 | one 248046: 32.78 | 74455 | one-auto accepted |
| own | int8 | served --spec off vs --spec auto | toolcall/t3_rename-step | 198 | one-auto 248068: 13.68; one 74455: 2.37 | 846 | both rejected |
| own | int8 | served --spec off vs --spec auto | toolcall/t4_comment-box | 146 | one-auto 29: 7.03 | 248068 | one accepted |
| own | int8 | served --spec off vs --spec auto | toolcall/t4_comment-linear_l0 | 112 | one-auto 248068: 14.86 | 74455 | one accepted |
| own | int8 | served --spec off vs --spec auto | toolcall/t4_comment-openai | 45 | one-auto 846: 6.26 | 248068 | one accepted |
| own | int8 | served --spec off vs --spec auto | toolcall/t5_test-loader | 186 | one 248046: 9.01 | 198 | one-auto accepted |
| own | int8 | served --spec off vs --spec auto | toolcall/t5_test-openai | 45 | one 248068: 4.47 | 846 | one-auto accepted |
| own | int8 | served --spec off vs --spec auto | toolcall/t5_test-step | 237 | one-auto 846: 4.38 | 248045 | one accepted |
| own | int8 | served --spec off vs --spec auto | toolcall/t6_usage-linear_l0 | 99 | one 248046: 5.69 | 248068 | one-auto accepted |
| own | int8 | served --spec off vs --spec auto | toolcall/t6_usage-loader | 41 | one-auto 248068: 4.79 | 198 | one accepted |
