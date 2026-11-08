# Running the models

How to download, serve, decode and benchmark each model the engine knows, what each
one supports today, and which of those configurations have actually run on a B70.
Flags and environment variables are explained in the project README's tables; this
page only says which ones matter for which model.

Every number is **measured** (on the card) or **derived** (from the memory planner or
byte counts), as in the specs.

## Read this first: what "measured" means here

How a change reaches `main`: the box is the check (`tools/box_check.sh` before every merge:
build, kernel command lines, `ctest -LE checkpoint` on a free card), and the box test queue
(`tools/box_validate.sh`, one "row" = the box tests of one spec) runs the model-level stages:
golden sets, serving, prefix cache, speculative decoding. A test row that passes proves the
configuration **runs and is correct**; it is not a speed row. The rows still open are listed
in [open-work.md](open-work.md); the queue itself is
[superpowers/plans/box-validation-queue.md](superpowers/plans/box-validation-queue.md).

- **Measured** below means a number on the card, with its row in
  [BENCHMARKS.md](BENCHMARKS.md). Qwen3.8 has the full set (prefill, decode, depth, prefix
  cache, MTP, DFlash). The other models have box test rows but mostly no speed row yet; the
  speed numbers for every model are to be re-taken on a quiet box after the 2026-11-04
  driver / kernel upgrade, so older rows come from compute runtime 26.35 + IGC 2.41.5.
- **Derived** means arithmetic from the memory planner or byte counts.
- Commands marked `# no speed row yet` run in the box test stages but have no speed
  number in BENCHMARKS; the expected numbers beside them are derived.
- Models dropped on 2026-10-27 (LFM2.5, Muse-Glimmer) and on 2026-11-07 (Agnes 3.0 Flash, spec 14: an
  `AgnesForConditionalGeneration` config.json is refused by name) are no longer supported and are not in
  the table; LiLiCorr, dropped the same day, came back on 2026-11-01 as Bolt's draft model
  (`--spec dflash`, Bolt's section). Flint (spec 27) and Nanbeige (spec 29) are parked (2026-11-04), not
  built. Nemotron 3.5 Bolt has its own section below; the table's columns are the older five.

| | Qwen3.8-27B | Ornith 1.5 35B-A3B | K2-Horizon MoVA-36B-A4B | Kolibri-1 (spec 20) | Qwen3.8-Flash-Next (spec 21) |
|---|---|---|---|---|---|
| checkpoint | published | published (spec 15 §13) | published | published (ours, int4 g64) | Intel's interim W4A16 (181 GB); ours not made (21q) |
| `b70-serve` | yes | yes | yes (spec 18d) | yes (spec 20e), **two cards** by default | `--layers N` only (spec 21e; the whole model refused naming spec 22) |
| `b70-serve --parallel pp` (two cards, spec 16d) | yes | yes | **refused** (one card) | its default (its own engine, spec 20e) | yes, `--layers N` (spec 21e) |
| `b70-decode` | yes | yes | yes | decode and prefill (spec 20d), **two cards** (the default) | decode (spec 21c) and prefill (spec 21d), **`--layers N` required** (the whole model needs spec 22), one card or `--parallel pp` |
| speed rows in BENCHMARKS | yes | no | no | no | no |
| MTP head (`--spec mtp`; `--spec auto` = adaptive MTP) | yes | yes (MoE head) | none (`auto` = off) | none (`auto` = off) | `b70-serve --spec mtp`, a fixed K = `--spec-max` 1..3 (spec 21e; `auto` = off until spec 22's numbers); `b70-decode` refuses it |
| `--spec lookup` | yes | yes (needs the head) | no | no | yes (needs the head; spec 21e) |
| `--spec dflash` (spec 19c / 19d) | yes, one card or `--parallel pp` (copy hand-off), bf16 KV | no (its v1 drafter is a later stage) | no drafter exists (refused) | refused | refused |
| `--kv-cache int8` | yes, gates pending | not built | yes, gates pending (spec 18e) | refused (bf16 only) | refused (decision 8: bf16 first) |
| prefill backends | `l0` (default), `l0-int8` | `l0` (default); `l0-int8` opt-in, its accuracy gates diagnostic (MoE routing) | `l0` only | `l0` only (spec 20d) | `l0` only (spec 21d; `l0-int8` refused) |
| KV per position (bf16) | 64 KiB | 20 KiB | 192 KiB | 20 KiB (+ 335.5 MB of sliding rings) | 24.75 KiB (12 QSA layers: K + V 2 KiB + 64 B of indexer keys each) |
| `--max-len auto`, int8 head (derived) | ~170k with MTP (the default); ~201k with `--spec off` | 262144, with or without MTP | ~46k served (the prefill planned); `b70-decode` ~50k decode-only | 262144 on two cards (split 25), the prefill planned or not | at `--layers 19` (Intel's forms, one card) 78336 decode-only; at `--layers 18` 70144 with the prefill planned |
| with `--kv-cache int8` (derived) | 262144, with or without MTP | - | ~90k served; `b70-decode` ~98k decode-only | - | - |

The `--max-len auto` lengths are the planner's at the default 1.5 GB reserve on the
32.53 GB card; the reserve itself is an estimate the box has not confirmed, and nothing
above 131072 has run on the card. Both CLIs print the length they chose and the memory
plan at startup.

## Common to every model

**Build** as in the README (`cmake`, then `ctest`). The engine never downloads; fetch a
checkpoint into the Hugging Face cache once:

```sh
uvx --from huggingface_hub hf download <org>/<name>
```

Both CLIs then take the repo id (looked up under `$HF_HOME`, default
`~/.cache/huggingface`) or a snapshot directory. The model is chosen from the
checkpoint's `config.json`; there is no model flag.

**What you choose, what the engine decides** (the zero-config cleanup, operator 2026-11-06). You
choose the model, the speculative decoding (`--spec`, default `auto`: the model's validated
proposer), the context length (`--max-len`) and the parallel mode (`--parallel auto|1|pp`, default
`auto`: one card when the model and the context fit it, else two; `--pipeline-split` overrides the
cut). The engine decides the rest - the int8 `lm_head`, the memory it leaves free (1.5 GB), the
draft model's form (int8), the speculative cost tables, the served name (the repo id), the tokenizer,
chat template and generation defaults (the snapshot's) - and refuses at startup, by name, what does
not fit or is not built.

**Which cards.** One or two Intel Arc Pro B70s (PCI id 0xE223), with peer access between them when
there are two: the hardware gate checks the vendor, the device id, the Xe-core count and peer access
before any weight is read (the Arc Pro B65 and B60 are refused as not yet tested on this box; more
than two cards are refused). One card is device 0, two are devices 0 and 1 of what
`ZE_AFFINITY_MASK` shows - `ZE_AFFINITY_MASK=1` runs on the second card alone.

**One model per server process.** The prefix cache (`--prefix-cache-gb`, system RAM, on
by default) belongs to that process and that model.

**Connecting a client.** `b70-serve` is OpenAI compatible: point the client (opencode or
any other) at `http://<host>:8000/v1`; the server lists the model under its repo id (or the
snapshot directory's name) and answers to any model name in a request. For
agentic sessions with a thinking model, add `--prefix-split-last` so each turn restores
exactly where the next one diverges (README, `b70-serve` flags).

**Benchmarking.** The command behind the vLLM comparison rows, against a running
server (no `--prefix-split-last`; it is for sessions, not benchmarks):

```sh
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model <repo id> \
    --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation
```

Both its `tg` and its `pp` figures read the same way as vLLM's: since 2026-09-27 the server
sends the role frame with the first token, after prefill, as vLLM does (docs/BENCHMARKS.md,
"llama-benchy's own pp figure against our server"). Its `pp` is HTTP-inclusive;
`b70-decode --bench --prefill-length` gives the device-side prefill time. The same-flags
comparison against vLLM is queue row 41 (docs/BENCHMARKS.md, "llama-benchy comparisons").

**Token ids for `b70-decode --ids`** are whitespace-separated, made with the checkpoint's
own tokenizer (needs `transformers`; `<snapshot>` is the snapshot directory in the cache):

```sh
python3 tools/oracle/tokenize.py <snapshot> encode prompt.txt > prompt.ids
python3 tools/oracle/tokenize.py <snapshot> decode $(cat out.ids)
```

`tests/golden/prompts/{prose,code,cjk}.ids` are ready-made ids for the Qwen tokenizer
(Qwen3.8's).

**Two cards (`--parallel pp`, spec 16).** Qwen3.8 and Ornith run with their layers split over
GPUs 0 and 1 of what `ZE_AFFINITY_MASK` shows: `b70-decode --parallel pp` (decode, spec 16b; the
two-card prefill with `--prefill`, spec 16c; `--spec mtp`, spec 16d) and `b70-serve --parallel pp` (spec
16d, everything the server does on one card); `--parallel auto` picks two cards by itself when the
model and the asked `--max-len` do not fit one. One answer is not faster on two cards; each card
holds about half the model, so `--max-len auto` reaches the trained 262144 tokens on every one
of them (derived, int8 head; with MTP too), and a long prompt prefills ~1.8x faster
(derived). The MTP head runs on the second card with its own copy of the embedding (2.54 GB).
K2-Horizon is one card only; Kolibri-1 runs on two cards by default in `b70-decode` (spec 20c / 20d)
and `b70-serve` (spec 20e, its own engine). Its box test rows (22, 23, 27, 28) are partly open (`open-work.md`); the speed
of the two-card prefill (row 23) is not measured:

```sh
# every command below: no speed row yet
Q=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
# an agentic session over both cards: 262144 tokens with MTP (derived; one card: ~170k)
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve $Q --kv-cache int8 --host 0.0.0.0 --port 8000 --parallel pp --prefix-split-last
# the same by itself: --parallel auto takes two cards when 262144 does not fit one
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve $Q --kv-cache int8 --max-len 262144 --prefix-split-last
# the same with the int8 KV cache and the peer hand-off (both cards ~16 GB, derived)
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve $Q --parallel pp --kv-cache int8 --pipeline-handoff peer
# b70-decode: the same ids as one card, greedy, with MTP across the split
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode $Q --ids tests/golden/prompts/prose.ids --n 64 \
    --prefill --parallel pp --spec mtp
```

Startup prints the split (`split: auto -> N`), the plan per card and both cards' memory lines
(and `parallel: auto -> ...` when `auto` chose).

## Speculative decoding and how its output is judged

A *proposer* guesses the next few tokens and the main model checks them all in one pass (a
"verify" step of M rows), so every correct guess is a token for free. `--spec` picks the
proposer: `mtp` (the model's own MTP head, spec 8), `dflash` (a small draft model, spec 19),
`lookup` (the continuation of the longest earlier repeat in the request, spec 19e) and
`dflash-lookup` (lookup inside DFlash, spec 19f). Since spec 28b the verify step of every decode
list at M >= 2 runs the small-batch int4 GEMM `gemm_sm` (M = 1 keeps the GEMV, bitwise as
before); spec 28d moved the DFlash drafter's int8 linears and Flash-Next's verify lists onto it
too. The DPAS verify-attention kernel (spec 28c) is built but **switched off**
(`kVerifyAttnTable` is all `kDpasNever`) until a real error at 8 rows is fixed, so verify
attention runs the v2 decode kernel at every M.

Because the verify rows no longer run the same kernel as a plain step, speculative output
is not compared bit for bit with plain decode. It is graded by the **f32 judge**
(`tests/server/spec_judge.h`, doc 14): a speculative stream passes when it is no further from
a float32 reference than plain decode on the same context, or when every miss is a near-miss
(`golden::NearMiss`, within 8 bf16 ulps of the f32 top). Plain, unchanged code paths still gate bitwise.

## Qwen3.8-27B

`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`: the model everything is tuned for. 64
layers (48 delta net, 16 attention), int4 g64, bf16 `lm_head` and MTP head in the file.

**On the card (measured):** decode (29.45 t/s at depth 4096 with the bf16 head, 31.21 t/s
with the int8 head; 20.1 t/s at 128k), prefill (2125.12 t/s at 4096), the 131072
context (28.1 GB) with passkey at 5 / 50 / 95 % of 120k, the prefix cache and
`--prefix-split-last` (a turn at 60k of history in 1.23 s), the int8 `lm_head` (the only head
the CLIs run since the zero-config cleanup), decode attention v2, MTP at a fixed K 1 / 2 / 3 at
`--max-len 16384`, the golden gates and the tool-call set.

**Never run on the card:** `--max-len auto` (the `b70-serve` default; row 7),
`--prefix-cache-gb auto` (the default; row 9), the adaptive MTP of `--spec auto` (the `b70-serve`
default; row 2), MTP above 16384 (row 6), `b70-decode --spec mtp` (row 6), `--draft-vocab` (row 8),
`--kv-cache int8` (row 11), `--spec lookup` (row 12).

```sh
uvx --from huggingface_hub hf download urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
Q=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ

# agentic session, the configuration that has run on the card: 131072 context, an explicit
# prefix cache size (spec 7 measured 32 GiB; pick what your RAM allows), no proposer
./build/src/cli/b70-serve $Q --kv-cache int8 --host 0.0.0.0 --port 8000 \
    --max-len 131072 --prefix-cache-gb 32 --prefix-split-last --spec off

# the default for agentic use: the largest context that fits beside the MTP head (~170k, derived),
# the cache sized from the machine's RAM, MTP choosing 0-3 guesses per step
./build/src/cli/b70-serve $Q --kv-cache int8 --prefix-split-last                                   # no speed row yet

# the full trained context, 262144, with the int8 KV cache (accuracy gates pending)
./build/src/cli/b70-serve $Q --prefix-split-last --kv-cache int8                   # no speed row yet

# prompt lookup instead of the MTP guesses (one proposer per server)
./build/src/cli/b70-serve $Q --kv-cache int8 --prefix-split-last --spec lookup                     # no speed row yet
```

```sh
# the BENCHMARKS.md rows: prefill 4096, then decode 256 at that depth (the int8 head; the
# byte-matched bf16-head rows predate the zero-config cleanup)
./build/src/cli/b70-decode $Q --bench --prefill-length 4096 --tg 256
./build/src/cli/b70-decode $Q --bench --depth 4096 --tg 256 --max-len 131072  # decode at a deep cache

# greedy ids in, ids out
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/prose.ids --n 64 --prefill > out.ids
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/prose.ids --n 64 --prefill --spec mtp  # no speed row yet

# per-launch anatomy of the decode step
./build/src/cli/b70-decode $Q --profile --depth 4096 --steps 32 --repeats 5
```

**Limits.** The trained context is 262144; with the bf16 KV cache one card holds ~201k plain and
~170k with the MTP head (derived) and 131072 is the largest length measured. `--kv-cache int8`
needs the `l0` or `l0-int8` prefill. MTP above 16384 and the adaptive MTP's cost table for the
int8 head are derived, not measured. `b70-decode` keeps `--max-len 16384` and `--spec off` by
default so its rows stay comparable.

**Relevant variables:** `B70_KV_CACHE`, `B70_PREFILL_PROFILE` for prefill phase timings,
`B70_GIT_SHA` for bench rows.

### DFlash draft model (spec 19) - on the card since 2026-10-27

`z-lab/Qwen3.8-27B-DFlash2` drafts a block of up to 7 tokens per step from Qwen3.8's own hidden
states (the outputs of layers 5 / 19 / 33 / 47 / 61). Its acceptance on our prompts is measured on the Mac CPU (spec 19a
P0, `docs/probes/probe-dflash-2026-10-08.md`: 7.21 tokens per verify at K = 7 on agentic tool calls, 6.15 on code, ~2.2 on
prose; int8 and 4-bit drafters equal bf16). The engine runs it since spec 19b-19d; it has run on the card
since 2026-10-27 (box queue rows 54, 55, 56, measured in BENCHMARKS "DFlash"), and the verdict is an option, not the default.
Since spec 28d the drafter's int8 linears run on the small-batch GEMM `gemm_sm_i8w` (draft step 16.3 -> 8.9 ms).
`--spec dflash-lookup` (spec 19f) adds prompt lookup inside DFlash for edit / copy turns:

```sh
# one card: the drafter beside the target (int8 at load, 2.05 GB + 1.06 GB of verify slots), K per request (auto, <= 7)
./build/src/cli/b70-serve $Q --prefix-split-last \
    --spec dflash --draft-model z-lab/Qwen3.8-27B-DFlash2                 # no speed row yet
# two cards: the drafter on the second card beside lm_head (the copy hand-off only)
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve $Q --parallel pp --spec dflash   # no speed row yet
# greedy without the server (one card), K <= 7 per step
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/code.ids --n 256 --prefill \
    --spec dflash --spec-max 7                                            # no speed row yet
```

- **Defaults (spec 19 §12, chosen from P0):** the drafter quantised to int8 at load (P0: int8 = bf16 acceptance at half the
  bytes; the zero-config cleanup removed `--draft-format`),
  `--spec-max 7` (the drafter's block; P0's acceptance still rises at K = 7 on tool calls and code, and the per-request
  policy drops K on prose), `--draft-vocab off` (P0's free-looking ranked 32k was ranked in-sample; by id it loses up to
  1.4 tokens at K = 7 - r56.d5_dv measures it). `--spec auto` stays the adaptive MTP: `dflash` becomes Qwen3.8's
  `auto` only if it beats the adaptive MTP by >= 10 % on the opencode recording (D5, r56.d5_opencode).
- **Refused, by name:** `--kv-cache int8` (kv8.cl is built for M <= 4), `--parallel pp --pipeline-handoff peer`,
  `b70-decode --parallel pp`, every other model.
- **The prefix cache** keeps the drafter's ring (41.9 MB) with every snapshot (spec 19 decision 5 as ruled in 19d), so a
  conversation restored from a request's prompt-end or request-end snapshot drafts exactly as an unbroken one; the store
  is keyed by the drafter. **Caveat:** a snapshot at a block end inside a long prefill holds the ring as that prefill
  left it - the prefill projects only its last 2048 positions - so a restore there drafts worse (never differently)
  until 2048 new positions have refilled the ring (spec 19 §12).

Re-running the acceptance measurement (a Mac or the box CPU; ~20-25 h on a 64 GB Mac with the container at
~54 GB - the first run's numbers are in `docs/probes/probe-dflash-2026-10-08.md` §4):

```sh
# downloads (HF cache): the target in bf16 (~54 GB), the drafter (3.85 GB); optional 4-bit drafter arm
uvx --from huggingface_hub hf download Qwen/Qwen3.8-27B
uvx --from huggingface_hub hf download z-lab/Qwen3.8-27B-DFlash2
uvx --from huggingface_hub hf download syvai/Qwen3.8-27B-DFlash2-W4A16
# plans and estimates only, no model loaded
DRY_RUN=1 tools/oracle/dflash_p0.sh
# the run: tap dumps, the draft-vocabulary ranking, acceptance per drafter arm (resumable, detached)
MEM=54g nohup tools/oracle/dflash_p0.sh > /dev/null 2>&1 &
tail -f oracle-out-19a/p0.log                       # then oracle-out-19a/accept/summary.md
# one arm again, e.g. after a new ranking
STEPS="accept" ARMS=int8 tools/oracle/dflash_p0.sh
```

## Ornith 1.5 35B-A3B

`ornith-ai/Ornith-1.5-35B-A3B` is published in bf16 only (71.9 GB), which the engine
refuses: it loads int4 GPTQ. The checkpoint is the operator's AutoRound W4A16 export,
[`urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ`](https://huggingface.co/urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ)
(22.99 GB: int4 g64 symmetric, GPTQ packing, experts per expert as
`experts.E.{gate,up,down}_proj`, the routers and the MTP head bf16, and - unlike Qwen3.8's
export - `in_proj_a` / `in_proj_b` int4 too, which the engine loads as an int4 a||b; spec 15
§13 lists every fact read from it).

The first mixture-of-experts model (spec 15): 40 layers (30 delta net, 10 attention),
hidden 2048, 256 experts with top-8 plus a shared expert, ~3 B parameters active per token.
Decode (15c), prefill (15d) and serving with its own chat template (Qwen3.5's), Qwen XML
tool calls and the MoE MTP head (15e) are built.

**Status: built; no speed row yet** (box test rows 10, 13, 16, 19). Expected (derived): ~1.86 GB
read per token, so the bandwidth ceiling is far above Qwen3.8's (~270 t/s), and launches,
not bytes, will set the decode rate; 526 launches per token. Prefill estimated at 3-6k t/s.
Weights 19.45 GB with the int8 head (derived), +0.50 GB for the MTP head. For comparison,
the checkpoint's model card reports vLLM XPU on a B70 at 99.8 t/s decode (tg256, c1) and
7090 t/s prefill (pp4096, c1) - not measured here.

```sh
O=urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ
uvx --from huggingface_hub hf download $O                 # 22.99 GB

# every command below: no speed row yet
# agentic session: the full trained context, 262144, fits with or without MTP (derived)
./build/src/cli/b70-serve $O --kv-cache int8 --host 0.0.0.0 --port 8000 --prefix-split-last

./build/src/cli/b70-decode $O --bench --depth 4096 --tg 256
./build/src/cli/b70-decode $O --bench --prefill-length 4096 --tg 256
./build/src/cli/b70-decode $O --bench --prefill-length 4096 --tg 256 --prefill-backend l0
./build/src/cli/b70-decode $O --ids prompt.ids --n 64 --prefill > out.ids
./build/src/cli/b70-decode $O --profile --depth 4096
```

**Limits.**

- `--kv-cache int8` is not built at Ornith's attention shape (capture names the missing
  binary); it is not needed for context, since 262144 fits in bf16.
- Prefill runs on `l0` (the default) and `l0-int8` (opt-in). On this MoE model `l0-int8` loses routing
  accuracy against f32 (the box, 2026-10-30: 1314 differing expert sets on the prefill path against the
  bf16 reference's 1140), and no 8-bit activation scheme studied won it back, so its accuracy gates are
  diagnostic - printed, not gated (operator ruling 2026-10-31); use it for speed only. Its kernels'
  own gate (`prefill_gate_ornith_l0-int8`, judged by the quantisation-aware f32 reference) still binds.
- The adaptive MTP (`--spec auto`) decides with Qwen3.8's cost table until the card measures Ornith's (the
  startup line says so); no default K is chosen for Ornith. The head's bf16 experts are
  quantised to int4 at load (round to nearest), which can change acceptance, never output.
- The checkpoint's `tokenizer.json` defines Qwen3.8's 248077 ids, the count the greedy
  argmax masks from, so `b70-serve` prints no vocabulary note. It tokenises the committed
  `tests/golden/prompts` (prose, code, cjk) to the committed ids; its pre-tokenizer splits
  combining marks differently from the bf16 base checkpoint's file (`\p{M}`), which only
  text with combining marks sees.

**Relevant variables:** `B70_KV_CACHE` must stay `bf16`; `B70_PREFILL_PROFILE` applies.

## K2-Horizon MoVA-36B-A4B

`urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ` (spec 18; 22.22 GB): a second
engine beside the Qwen one. 48 layers, every one full attention (32 q-heads, 8 kv-heads,
head dim 128); layers 3-47 route values through 64 value experts (MoVA, top-4) and the FFN
through 100 experts (top-8) plus a shared one. Its own tokenizer (250,624 ids, BOS 0) and
chat template. No MTP head. Trained context 524288.

**Status: built; no speed row yet** (box test rows 14, 15, 17, 21, 25). Decode (18b) and prefill (18c)
run through `b70-decode`; `b70-serve` serves it (18d: its chat template, tool calls and
reasoning, K2's own engine behind the server, KV-only prefix-cache snapshots - host-tested only).
Expected
(derived): 717 launches per token, 3.15 GB read per token with the int8 head (roofline
~190 t/s from bytes alone), prefill ~3,400 t/s at 4096. For reference, vLLM on **two**
cards (pipeline parallel, fp8 KV) measured 44.43 t/s decode.

```sh
uvx --from huggingface_hub hf download urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ
K=urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ

# every command below: no speed row yet
# K2 ids start with BOS 0: encode with --bos
python3 tools/oracle/tokenize.py <k2-snapshot> encode --bos prompt.txt > k2.ids

./build/src/cli/b70-decode $K --ids k2.ids --n 32                    # decode only, one replay per id
./build/src/cli/b70-decode $K --ids k2.ids --n 32 --prefill          # prefill on l0
./build/src/cli/b70-decode $K --bench --depth 4096 --tg 256
./build/src/cli/b70-decode $K --bench --prefill-length 4096 --tg 256
./build/src/cli/b70-decode $K --ids k2.ids --n 32 --prefill --max-len auto   # ~46k, derived

# serving (spec 18d): int8 head, prefix cache on, --max-len auto (~46k bf16 KV, ~90k int8 KV)
./build/src/cli/b70-serve $K --kv-cache int8 --port 8000
./build/src/cli/b70-serve $K --kv-cache int8
# K2's template variables per request: "chat_template_kwargs": {"tool_call_format": "xml" |
# "xml_typed" | "json", "reasoning_effort": "high" | "medium" | "low"}
```

**Limits.**

- Served on one card only (`--parallel pp` refused; `--parallel auto` is one card), no llama-benchy
  row yet (queue row 25). `b70-serve` refuses `--spec mtp` / `lookup` / `dflash` and a non-`l0`
  `--prefill-backend` by name; its `--spec auto` is off.
- Context: the bf16 KV cache costs 192 KiB per position, so `--max-len auto` gives ~49.9k
  decode-only, ~45.8k when the run prefills (the int8 head) - derived. `b70-decode`'s default is
  16384. Long context needs int8 KV or two cards (spec 18e, not built).
- Prefill runs on `l0` only: `--prefill-backend l0-int8` is refused by name.
  Omitting `--prefill-backend` is fine.
- Refused: `--spec mtp` (no head), `--profile` (not built yet). `--kv-cache int8` is built (spec 18e,
  gates pending).
  `--spec lookup` has nothing to run on: it needs verify lists, which only an MTP head
  provides today.
- Which attention matches the reference better is decided by the f32 judge (the MoE gates
  are graded against an f32 reference, with a one-bf16-ulp near-tie rule, since routing
  flips on near-ties): `flash` is the attention (the eager diagnostic arm, `B70_K2_ATTN=eager`, went with the
  zero-config cleanup; the f32 judge had decided).

**Relevant variables:** `B70_PREFILL_PROFILE` times K2's prefill phases but `b70-decode` does not
print the table for K2 yet. `B70_KV_CACHE` does not apply to K2 (`--kv-cache`).

## Kolibri-1 (spec 20)

Aleph Alpha's Kolibri-1 (`model_type kolibri1`): 50 MoE layers (384 experts top 6 + an ungated shared
expert), hidden 2560, 48 / 4 heads x 128, 40 sliding-window layers (513 keys, RoPE) and 10 full NoPE
layers, sandwich norms, a 128000-id vocabulary, trained context 262144. Our int4 checkpoint is
`urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ` (AutoRound g64 sym, every routed AND shared expert int4 - the
engine reads the shared expert as one more int4 block, operator ruling 2026-10-31); a synthetic checkpoint in
the same form comes from `tools/quantize/kolibri/make_synth.py`.

**Status: built; no speed row yet** (box test rows 24, 26 and 28). Decode (spec 20c), prefill (spec 20d) and
serving (spec 20e, `b70-serve`), on **two cards**: at int4 the model holds ~42.5 GB of weights, so two cards are Kolibri's
default (`--parallel auto`) and `--parallel 1` is refused unless the model fits one card (a synthetic checkpoint, or `--layers N`:
development mode, the first N layers only - e.g. 30, ~26 GB). Expected (derived): 756 launches per
token (856 with `B70_KOLIBRI_ATTN=eager`), ~2.39 GB read per token with int4 attention and the int8 head -
a roofline near 250 t/s on two cards. Prefill (`--prefill`, `--prefill-length N`) runs chunks of up to
2048 positions on the `l0` backend only: 2252 launches per chunk (45 a layer: the attention linears as
bf16 slabs + `pf_gemm`, the windowed flash attention over the 4096-slot ring, the 384 experts sorted and
dequantised to bf16 in 512 MiB weight batches for the grouped GEMMs) + 5 for the head, ~0.9 GB of prefill
scratch a card; on two cards each chunk crosses once, by copy, and the cards take turns (no overlap yet).

```sh
K=urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ          # spec 20b

# every command below: no speed row yet; needs two cards
./build/src/cli/b70-decode $K --ids k.ids --n 32                                  # two cards, split by bytes
./build/src/cli/b70-decode $K --ids k.ids --n 32 --pipeline-handoff peer --pipeline-split 25
./build/src/cli/b70-decode $K --bench --depth 4096 --tg 256 --max-len 40960
./build/src/cli/b70-decode $K --parallel 1 --layers 30 --ids k.ids --n 8          # one card, 30 layers
B70_KOLIBRI_ATTN=eager ./build/src/cli/b70-decode $K --bench --depth 4096 --tg 256
./build/src/cli/b70-decode $K --ids k.ids --n 32 --prefill                        # spec 20d: prefilled
./build/src/cli/b70-decode $K --bench --prefill-length 32768 --tg 16 --max-len 40960   # the pp row
./build/src/cli/b70-decode $K --ids k.ids --n 32 --prefill --prefill-chunk 1000
```

**Serving (spec 20e, `b70-serve`).** The same engine behind the OpenAI server, on two cards by default:
Kolibri's ChatML template (the checkpoint's `tokenizer_config.json` string; byte-identical to transformers'
`apply_chat_template` on 16 cases, measured on the Mac), its 128k tokenizer (127998 ids; the head's last two
rows have no token and are never emitted). The model opens its reasoning with `<think>` - it comes back as
`reasoning_content`; `chat_template_kwargs` `reasoning_effort` (`none` turns thinking off; `minimal`, `low`,
`medium`, `high` - the default -, `xhigh`, `max`) or `enable_thinking: false` reach the template. Tool calls
are hermes JSON in `<tool_call>` and come back as OpenAI tool calls (`finish_reason` `tool_calls`); a call
that does not parse comes back as content. A request without `temperature` / `top_p` / `top_k` samples with
the checkpoint's `generation_config.json` (1.0 / 0.97 / 128); `"temperature": 0` is greedy. The prefix cache
works as for every model: each snapshot carries the 40 sliding rings' last 512 positions (41.9 MB, derived)
beside the full layers' KV (20 KiB a position). `--max-len auto` reaches 262144 on two cards.

```sh
# every command below: no speed row yet; needs two cards
./build/src/cli/b70-serve $K --kv-cache int8                                    # two cards, --max-len auto (262144), int8 head
./build/src/cli/b70-serve $K --kv-cache int8 --pipeline-handoff peer --max-len 131072
./build/src/cli/b70-serve $K --kv-cache int8 --prefix-split-last                # agentic sessions: the next turn restores at len - 1
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":
  [{"role": "user", "content": "Wie spät ist es in Berlin?"}], "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

**KL4 (box):** `tools/probe/kolibri_passkey.sh` (passkey 3 / 3 at 262144 on two cards) and the
German A4-style tool-call set `tests/golden/toolcall-kolibri-de` (`tools/toolcall/engine_generate.sh`, then
`score.py --format hermes` against the bf16 source's reference, `oracle_generate.py --model kolibri`).

**Limits.** Prefill on `--prefill-backend l0` only (`l0-int8` refused: its h8 linears rotate in
1024-k Hadamard blocks and the hidden is 2560), `--prefill-chunk`
at most 2048; no `--spec mtp` / `lookup` / `dflash` (no head, no verify lists; its `--spec auto` is off),
`--kv-cache int8` refused (20 KiB a position: bf16 only), no `--profile`, context capped at the trained
262144 (decision 3) - `b70-decode` and `b70-serve` alike. The
bench prompt is the first 42 ids of `de_prose.txt` through Kolibri's tokenizer (spec 20e).

**Relevant variables:** `B70_KOLIBRI_ATTN` (`flash` | `eager`, decode and prefill alike), `B70_KOL_TIE_TOL`
(the golden test).

## Qwen3.8-Flash-Next (spec 21)

Qwen's `Qwen3.8-Flash-Next` (`model_type qwen4_exp`): 48 layers (36 GatedDeltaNet with a sigmoid output gate, 12 QSA
sparse-attention layers that attend the top 512 blocks of 4 positions plus the open block), a 512-expert / top-10
MoE in every layer, 4-stream hyper-connections instead of a residual, and a per-layer n-gram embedding (PLE) whose
51.8 GB int8 table lives in host memory and is read by the card directly. Intel's interim checkpoint
`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` (int4 g128 experts, bf16 dense layers) loads through spec 21b's exact
g64 expansion; its PLE table must be converted once (`tools/box_validate/qwen4exp_oracle.sh <data> ple-int8`, ~51.8 GB
beside the snapshot). Our own AutoRound g64 export is spec 21q's.

**Status: built; no speed row yet** (box test rows 31, 32, 33 and 34). Decode (spec 21c), prefill (spec 21d) and serving
(spec 21e): **the whole
model does not fit two B70s** (its routed experts alone are 64.17 GB at int4 g64) and is refused, naming spec 22
(the expert-offload tier); `--layers N` (development mode: the first N layers, then the final mixer and the head) or
`--layers auto` (the planner's N, derived: 18-19 of Intel's layers on one card, 38-39 on two, by `--max-len`) is
required. Expected (derived): 779 launches per token at 48 layers, 75 at `--layers 4`; `B70_Q4_ATTN=eager` binds the
reference's attention chain (one launch for flash's two in each QSA layer) in decode and the eager sparse attention
in prefill. `--prefill` / `--prefill-length N` run the prompt in chunks of up to 2048 (`--prefill-chunk C`) on the
`l0` backend: spec 6's flash for the positions that see at most 2051 others, the indexer's own selection and a sparse
flash past them; 358 / 361 / 360 launches a chunk at `--layers 4` (all dense / straddling 2050 / all sparse), ~3950
at 48 layers (derived). The prefill scratch is ~1.5 GB a card (the 512 MiB expert weight batch, the score rows that
grow with `--max-len`), planned into `--layers auto` / `--max-len auto` when a run prefills: with it Intel's forms fit
18 layers on one card and 37 on two at 32768 (derived).

**Serving (spec 21e).** `b70-serve <checkpoint> --layers N` serves OpenAI chat on one card or two (`--parallel pp`) through
the Qwen chat path: the template is Qwen3.8's byte for byte, XML tool calls and `<think>` reasoning as for Qwen3.8,
greedy unless the request asks (decision 11 open). **The tokenizer must be the original's** (`Qwen/Qwen3.8-Flash-Next`'s
`tokenizer.json`, whose split adds `\p{M}`), and it is the snapshot's: Intel's checkpoint ships Qwen3.8's file, which
`b70-serve` refuses - serve a snapshot directory that ships the original's (our own export, spec 21q; the zero-config
cleanup removed `--tokenizer`). On the box that directory is `~/b70/models/qwen3.8-flash-next/`: links to every file
of Intel's snapshot except `tokenizer.json`, which links to the original's; `qwen3.8-flash-next-ple-int8/` beside it
links the PLE int8 file.
The prefix cache keeps this family's state with each snapshot (115.7 MB: the 36 GDN states and conv rows, the PLE
history, the QSA open block's raw keys; + 21 KB with the head) and 25,344 B a position in its blocks (+ 2,112 with the
head), one layout on one card or two. `--spec mtp --spec-max K` (K 1-3, a fixed K) drafts with the MTP head (vLLM's form;
`pre_fc_norm_hidden` as one RMS over the 4 streams, `B70_Q4_MTP_NORM=per_stream` for the other form; draft steps after
the first reuse step 0's selection) and verifies K + 1 rows in one list (790-795 launches at 48 layers, 86-91 at
`--layers 4`, derived); its `--spec auto` is off until spec 22 measures the acceptance on the whole model. The head's bf16 experts are RTN-quantised to int4 g64 at load; on two cards it runs on the second and
reads the first card's embedding over peer.

```sh
I=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound

# every command below: no speed row yet; needs the PLE int8 file beside the snapshot (<snapshot>-ple-int8/)
./build/src/cli/b70-decode $I --layers 18 --ids tests/golden/prompts/q4exp_short.ids --n 32
./build/src/cli/b70-decode $I --layers 38 --parallel pp --bench --depth 4096 --tg 256
./build/src/cli/b70-decode $I --layers auto --max-len auto --ids tests/golden/prompts/q4exp_short.ids --n 8
# spec 21d: the prompt prefilled in chunks (l0 only), then decode; the pp row
./build/src/cli/b70-decode $I --layers 18 --ids tests/golden/prompts/q4exp_4k.ids --n 32 --prefill
./build/src/cli/b70-decode $I --layers 37 --parallel pp --bench --prefill-length 32768 --tg 16 --max-len 33024
# spec 21e: served truncated from a snapshot that ships the original's tokenizer.json, MTP K = 2, one card / two
F=~/b70/models/qwen3.8-flash-next   # Intel's files, the original's tokenizer.json
./build/src/cli/b70-serve $F --kv-cache int8 --layers 18 --spec mtp --spec-max 2
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve $F --kv-cache int8 --layers 37 --parallel pp --spec mtp --spec-max 2
```

**Limits.** Prefill on `l0` only (`--prefill-backend l0-int8`: spec 5's h8 linears rotate in whole 1024-k Hadamard
blocks and the hidden is 2560), `--prefill-chunk` at most 2048; the two cards
cross each chunk by copy, sequentially (spec 16c's overlapped chunk pipeline is a later lever); MTP in `b70-serve`
only, at a fixed K (`b70-decode --spec mtp` and `--draft-vocab` refused), the whole model refused in both CLIs
(spec 22), bf16 KV only (`--kv-cache int8` refused, decision 8), `reasoning_effort` not forwarded to the template (as
for Qwen3.8),
no `--profile`, context capped at the trained 262144.

**Relevant variables:** `B70_Q4_ATTN` (`flash` | `eager`, decode and prefill alike), `B70_Q4_MTP_NORM` (`single` |
`per_stream`),
`B70_Q4_TIE_TOL` / `B70_Q4_SEL_TOL` (the golden test's routing and selection near-tie tolerances, 1e-3 proposed).

## Nemotron 3.5 Bolt 74B-A9B (spec 25)

NVIDIA's hybrid of 40 Mamba-2 blocks, 8 NoPE attention blocks (GQA 32 / 2, head_dim 128) and 40 latent MoE blocks (top-9
of 240..496 relu² experts in a 1024 latent, a shared expert), 88 blocks, vocabulary 131072; served from our own int4 g64 export
`urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ` @ `8932aa8d` (AutoRound, GPTQ packing; 41 GB on disk). The
loader also reads the GPTQ g128 spelling through its exact g64 expansion (the synthetic checkpoints test it).

**Status: decode, prefill and serving built; no speed row yet** (box test rows 60 / 61 / 62; spec 25c / 25d / 25e, as built in
spec 25 §15 / §16 / §17). The real model runs on **two cards** (Decision 4: two cards are its `--parallel auto`, split 48
by bytes; `--parallel 1` is refused naming the bytes); the N 12 synthetic checkpoints and `--layers N` fit one card. Without `--prefill`
a prompt goes through the decode list one replay per id; `--prefill` / `--bench --prefill-length N` run the `l0` prefill
(spec 25d). Expected (derived): 653 launches a token (645 with `B70_BOLT_ATTN=eager`, 533 with `B70_BOLT_FUSE=all`, +2 with
`--pipeline-handoff peer`). P0 (`probe_bolt_step`) runs first on the box and decides the GEMV `{S, layout}` picks and which
arms are worth keeping.

```sh
S=oracle-out-bolt-synth/n12-g128                   # a synthetic checkpoint (25b's make_synth.py, 12 blocks)
P=tests/golden/prompts/bolt_short.ids

# every command below: no speed row yet
./build/tools/probe/probe_bolt_step --depth 4096 --rounds 9 --sweep-s --pair      # P0: the skeleton lists (no model)
./build/src/cli/b70-decode $S --ids $P --n 32                                     # one card, int8 head
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <snapshot> --ids $P --n 32          # two cards (the default for 88 blocks)
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <snapshot> --bench --depth 4096 --tg 256 --max-len 8192
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <snapshot> --profile --depth 4096 --steps 32 --repeats 5
B70_BOLT_FUSE=all ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <snapshot> --bench --depth 4096 --tg 256   # 533 launches
B70_BOLT_SSM=bf16 ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <snapshot> --bench --depth 4096 --tg 256   # Decision 2's arm
```

**Serving (spec 25e).** `b70-serve <snapshot>` runs the engine on **two cards by default** (`--parallel 1` only for a synthetic /
small checkpoint; `--pipeline-split`, `--pipeline-handoff` as `b70-decode` takes them), `--max-len auto` up to 262144 with
the prefill scratch planned in (Decision 14), the int8 head by default, prefill on `l0` only:

```sh
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve <snapshot> --kv-cache int8 --port 8000 --prefix-cache-gb auto --log-requests ~/bolt-log
```

- **Chat template.** The snapshot's `chat_template.jinja` (sha256 `6521afc7...`, NVIDIA's, byte-identical in our
  export; its sha256 is printed and compared in the startup line). ChatML, no BOS, and the system block is ALWAYS
  rendered (`<|im_start|>system\n<|im_end|>\n` with no system message). `chat_template_kwargs`: `enable_thinking` (default
  true: the prompt ends `<think>\n`; false: `<think></think>`), `medium_effort` (appends `{reasoning effort: efficient}` to
  the last user message), `truncate_history_thinking` (default true: the reasoning of turns before the last user message is
  dropped), each a boolean (a 400 otherwise); the OpenAI `reasoning_effort` maps onto them - `none` / `minimal` ->
  `enable_thinking` false, `low` / `medium` -> `medium_effort` true, `high` / `xhigh` -> the default, anything else a 400 -,
  and an explicit kwarg wins. Tool schemas and call arguments render with sorted keys (the server's JSON objects are
  sorted: the prefix cache misses at a call whose arguments the model wrote in another order). Python's spelling for a
  history argument (`True`, `None`) is the template's own.
- **Output.** The output opens inside reasoning when the prompt ends `<think>\n` (`reasoning_content` in the response,
  VERBATIM - the template re-adds `<think>\n` + reasoning + `</think>`, so a stripped reasoning would drift every later
  prompt), up to the first `</think>`; then content and Qwen3-Coder XML tool calls (`<tool_call>\n<function=NAME>\n<parameter=
  KEY>\nVALUE\n</parameter>...`, values typed by the tool's schema first). With thinking off everything is content, a stray
  `</think>` included. `chat_template_kwargs.force_nonempty_content: true` turns reasoning that never closed (or ended with
  no content) into content - held back from the stream until decided. A call cut by `max_tokens` or EOS is dropped, never
  leaked as content; a call inside reasoning is reasoning (the `ultra_v3` reasoning parser's rules; spec 25 §7 and §17).
- **Sampling.** A request that names no `temperature` / `top_p` / `top_k` samples with NVIDIA's published T 1.0, top-p 0.95,
  no top-k, and generation ends on EOS **2 or 11** - from the snapshot's `generation_config.json` when it carries sampling
  keys (ours does: NVIDIA's file), else the published values. A seeded
  sampled request is reproducible request by request; `temperature: 0` is the device argmax.
- **Prefix cache.** A snapshot at an aligned 2048-position block end (and at a request's end) carries, per Mamba block, the
  SSM state (fp32: 4 MiB; 2 MiB with `B70_BOLT_SSM=bf16`, which is another store key) and the three raw conv columns
  before the position - 170,229,760 B in all (derived) - and the 8 attention blocks' KV, 8,192 B a position (one state per
  block end: a 64k-token prompt writes 32 of them, ~6.0 GB with the KV, derived - size `--prefix-cache-gb` for it). A hit restores
  the largest block end <= the common prefix and prefills the rest. A new user message re-renders every earlier assistant
  turn without its reasoning (`truncate_history_thinking`), so the cache diverges at the first assistant turn before the
  last user message; within one user turn (the agentic tool loop) the re-render equals what was generated.
- **Request.** Image / video / audio content parts are refused (text only).

**Speculative decoding (spec 26c; restored 2026-11-01, not yet run on a card - box rows 63-65).** Bolt has no MTP head, but
its decode list also runs as a *verify* pass of M = K + 1 <= 8 rows (each row bitwise a plain decode step, so greedy output
is the plain run's), and two proposers feed it on `b70-serve`:

- `--spec lookup` - prompt lookup (the continuation of the longest earlier repeat in the request, no extra model);
- `--spec dflash` - NVIDIA's LiLiCorr draft model (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr`, 918 MiB, from
  the HF cache; `--draft-model PATH` points elsewhere) on the last card beside the main model's head: it reads six of Bolt's
  hidden-state rows ("taps") and drafts a block of up to 7 ids in one pass.

`--spec-max 1..7` caps the guesses (default 7); K is picked per request from how often its guesses were right. Everything
else is the engine's: the draft model's linears int8 per row (exact bf16 and int4 g64 are test arms), its draft distribution
vLLM's (the candidate lattice), the verify's experts read once per block for all rows (the union form; the plain M-row form
is its bitwise test reference). Bolt's `--spec auto` stays **off** until row 65's D5 measures `dflash` >= 10 % faster than
both `off` and `lookup` on the agentic set (and nowhere > 3 % slower). Two cards cross by the copy hand-off only
(`--pipeline-handoff peer` is refused with a proposer). `b70-decode` decodes Bolt plainly (`--spec` there is refused).

```sh
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve <snapshot> --spec dflash      # the LiLiCorr drafter, K <= 7 adaptive
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve <snapshot> --kv-cache int8 --spec lookup --spec-max 4
```

**Limits.** No `--spec mtp` (no MTP head) and no `--spec dflash-lookup` (Qwen3.8's), no `--draft-vocab`, bf16 KV only
(`--kv-cache int8` refused, Decision 17), prefill on `l0` only (`--prefill-backend l0-int8` refused), context at most
262144, `--layers` is `b70-decode`'s development mode only.

**Relevant variables:** `B70_BOLT_ATTN` (`flash` | `eager`), `B70_BOLT_FUSE` (`none` | `all` | a comma list of `router`,
`route`, `conv`), `B70_BOLT_SSM` (`fp32` | `bf16`), `B70_BOLT_TIE_TOL` (the golden gate's near-tie tolerance, 1e-3 proposed),
`B70_BOLT_A_EXP` (the reference's `A`: `torch` | `engine`), `B70_BOLT_SNAPSHOT` (`template_bolt_test`'s,
`bolt_tokenizer_test`'s and `bolt_request_test`'s snapshot).
