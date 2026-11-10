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
as TP does. The judge's verdicts: (pending the f32 references, `f32_refs.sh forced-qwen`, CPU).
