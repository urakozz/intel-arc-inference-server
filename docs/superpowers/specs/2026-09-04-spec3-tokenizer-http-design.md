# Spec 3 - Tokenizer, chat template, HTTP server

Status: design for review, 2026-09-04. Companion to spec 2 (prefill);
independent of it - every stream here builds and is gated against the
**decode-only engine that exists today**, and prefill plugs in underneath
`Engine::prefill()` when spec 2 lands. Operator rulings that shape it:
no dependency rule (inherit what exists); delegate, divide and conquer;
the goal is a custom engine faster than vLLM that is **usable for real
serving** (Open WebUI, opencode).

Authority for what the checkpoint ships and for the tokenizer options is
`docs/11-tokenizer-and-chat-template.md` (read 2026-08-22 from the
snapshot on the box); this spec adopts its recommendations and adds the
server. Every number is measured unless marked derived/estimated.

## 1. Where we start

- **No tokenizer, no server, no stop handling.** `b70-decode` is ids-in,
  ids-out (`src/cli/b70_decode.cc:19`); prompts become ids through
  `tools/oracle/tokenize.py` (transformers, run in the reference
  container). The generate loop has no EOS check. No HTTP code exists
  (`src/` = `cli common kernels l0 loader model runtime`).
- **The checkpoint ships** (docs/11 §"What the checkpoint ships"):
  `tokenizer.json` (Qwen2Tokenizer byte-level BPE, 248,044 vocab +
  247,587 merges + 33 added tokens = 248,077 ids), `chat_template.jinja`
  (8.9 KB, separate file - ChatML with a thinking toggle, tools, macros,
  `raise_exception`), `eos_token_id: [248046, 248044]` (both stop),
  sampling defaults `temperature 1.0, top_k 20, top_p 0.95`,
  `vocab_size 248320` - `lm_head` padded by 243 rows, so ids ≥ 248,077
  must be masked.
- **The pre-tokenizer regex** is the Qwen2 pattern with Unicode categories
  and a case-insensitive contraction group (docs/11 quotes it verbatim);
  `std::regex` cannot express it.
- **What the benchmark actually needs** (explorer 4, from llama-benchy's
  source): only `POST /v1/chat/completions`, both non-streaming (warmup +
  coherence test) and streaming (`stream: true`, SSE `data:` frames,
  `choices[0].delta.content`, `data: [DONE]`); accurate
  `usage.prompt_tokens` (it calibrates `--pp`); `min_tokens`/`ignore_eos`
  honoured for `--exact-tg`; **an unbuffered first token** (pp is derived
  from time-to-first-response); a genuinely correct answer to "What is
  the capital of France?" - the coherence gate runs before any timing and
  exits on failure. `/v1/models` is not called by the standing command.
  README's v1 definition of done additionally names `/v1/completions`.

## 2. Goals, bars, stopping rule

**Goal:** the engine becomes a server that llama-benchy, Open WebUI and
opencode can talk to, with the tokenizer and template exact to the
reference implementation, and **the HTTP path costing nothing measurable
against the CLI bench**.

**Bars (hard):**
1. **Tokenizer parity:** byte-identical ids to Python `tokenizers` on a
   ≥ 10k-line corpus spanning ASCII, CJK, emoji, code, whitespace runs and
   every special token; `decode(encode(x)) == x`; the streaming detokeniser
   fed one id at a time concatenates to the one-shot decode and never
   emits U+FFFD for valid input (docs/11 "Acceptance test").
2. **Template parity:** a 3-turn conversation (system/user/assistant/user)
   rendered byte-for-byte equal to `transformers.apply_chat_template` for
   *this* `chat_template.jinja`, with `enable_thinking` both ways.
3. **Golden gate through the server:** the three golden prompts sent as
   raw completions produce the same greedy ids as `b70-decode --ids`
   (the tie-aware gate semantics unchanged).
4. **The benchmark runs end to end:** the standing llama-benchy command
   passes its coherence test and prints pp/tg rows.
5. **HTTP costs ≤ 2% of tg:** `tg256` over HTTP within 2% of the CLI
   record row on the same checkpoint and box (the per-token host path -
   detokenise + SSE write - overlaps the next replay; measured, not
   assumed).
6. Suite green; `-Wall -Wextra -Werror`; decode's 774/19 and gate rows
   untouched; nothing in `src/runtime` or `src/kernels` changes except the
   two engine API additions in §3.4.

**Performance bar for this spec:** bar 5 only. The pp number through
HTTP is spec 2's gate re-taken on this surface, not this spec's.

**Stopping rule:** streams land independently; the gate (§6) runs when all
four have; short on any bar → memo; nothing beyond it without a ruling.

## 3. Design - four streams, three inherited dependencies

### 3.1 Tokenizer (`src/tokenizer/`) - inherit the reference

**HF `tokenizers` (Rust) behind a four-function C interface** -
`encode`, `decode`, `token_to_piece`, `free` - exactly docs/11's
recommendation. It *is* the reference, so parity is by construction and
the parity corpus (bar 1) is what makes it replaceable later. The crate
builds once into a static `.a` (~10 MB) via a Cargo step in CMake, cached
like the kernels. Added tokens, byte-level pre/post-processing and the
Unicode regex are the crate's, not ours. `src/tokenizer/` depends on
nothing else in `src/`.

Rejected: hand-written BPE (docs/11: "the end state, not the starting
point" - and the operator has since ruled learning is not a goal);
`openvino_tokenizers` (drags the OpenVINO runtime); `tokenizers-cpp`
(same crate plus a shim we would not use).

### 3.2 Chat template and streaming detokeniser - inherit `minja`

**`google/minja`** (header-only Jinja subset; what `openvino.genai`,
`model_server` and `llama.cpp` use) renders `chat_template.jinja` with the
request's `messages`, `tools` (rendered if present, never parsed from
output - out of scope), and `enable_thinking` (default per the template).
Bar 2 is the test; if this template uses a construct minja lacks, the
fix is a per-model fallback as `openvino.genai` does, recorded.

**Streaming detokeniser** (`src/tokenizer/streamer.*`): the hold-and-flush
algorithm docs/11 quotes from `openvino.genai/text_streamer.cpp:7-47` -
decode the pending run, hold if it ends in `\xEF\xBF\xBD`, emit the
suffix past the printed cursor, flush on newline. `push(id)` returns zero
or more UTF-8-complete strings.

### 3.3 HTTP server (`src/server/`) - single header libraries, single stream

- **`cpp-httplib`** (single header; `llama.cpp`'s choice) for HTTP/1.1 +
  chunked SSE; **`nlohmann/json`** (single header) for the OpenAI schema.
  Both vendored as pinned single headers under `third_party/`.
- **Endpoints:** `POST /v1/chat/completions` (stream and non-stream),
  `POST /v1/completions` (README DoD; same engine path, no template),
  `GET /v1/models` (one entry: the served name). Nothing else in v1.
- **Request fields honoured:** `messages`/`prompt`, `max_tokens`,
  `min_tokens`, `ignore_eos`, `stream`, `stream_options.include_usage`,
  `stop` (strings, matched on decoded text), `n = 1` only,
  `temperature`/`top_k`/`top_p`/`seed` (see §3.5). Unknown fields are
  ignored, not rejected (vLLM extensions like `return_token_ids` and
  `cache_prompt` arrive from llama-benchy).
- **Response:** OpenAI shapes; `usage.prompt_tokens` = the tokenizer's
  count of the rendered prompt (load-bearing for the benchmark's
  `--adapt-prompt`); `completion_tokens` exact; streaming emits **one
  SSE frame per generated token, written immediately** (no buffering -
  bar 4's pp depends on it), a final `finish_reason` frame, an optional
  `usage` frame when asked, then `data: [DONE]`.
- **Single stream, serialized:** one engine, one request at a time; a
  second request waits in a FIFO (bounded; 503 beyond it). No
  scheduler, no batching - `docs/04-architecture.md` "Server" says why.
- **Stop:** generation halts on any id in `eos_token_id` (unless
  `ignore_eos`), on a matched `stop` string, or at `max_tokens`;
  `min_tokens` suppresses EOS until reached.
- **Errors:** JSON error bodies with OpenAI's shape; a request that fails
  template rendering or exceeds `max_model_len` is rejected before
  touching the engine.

### 3.4 Engine API - AMENDED 2026-09-06 (planning, after spec 2 landed): no `src/runtime` change at all

What exists (`src/runtime/engine.h`, read 2026-09-06): `reset()` (zeroes the
persistent group), `ingest(ids)` (one replay per id - spec 2 did **not**
change its body), `prefill(ids, chunk)` (spec 2; compiled into
`b70_prefill_host`, reachable only by a target that calls
`b70_link_prefill()`; 1408 t/s iterate at C = 2048), and
`generate(n, on_token)` - token *i* is read from `cur_token[0]` **before**
replay *i*, so `generate(1)` is exactly the `step()` this spec asked for:
it returns the id the previous fence sampled and runs one replay that
samples the next. The server's loop is therefore

```
reset → prefill(prompt ids)   [ingest(ids) when B70_HAVE_PREFILL is 0]
      → repeat: id = generate(1)[0]; streamer.push(id) → SSE frame; stop checks
```

and the `(n+1)`-th token that `generate` leaves pending in the control
block is what the next `generate(1)` returns - two calls are one continuous
stream (engine.h:89-97). **No engine method is added**; bar 6's "nothing in
`src/runtime` changes" tightens to *nothing at all*. `b70-serve` links what
`b70-decode` links, prefill component included, and is the second product
binary that reaches `Engine::prefill`.

Consequence for the gate (§4): the llama-benchy `pp` row is no longer
"prefill = decode replay per id" - it is spec 2's device-side 1408 t/s
re-taken **through HTTP**, and the delta between the two is this spec's
measurement of the tokenizer + template + HTTP + first-frame path.

### 3.5 Sampling - greedy on device; temperature/top-k/top-p on the host, last

The device sampler is argmax (`argmax_stage2` writes `cur_token`). Greedy
is the benchmark path and the default. **Real serving wants sampling**
(the checkpoint's own defaults are `temperature 1.0, top_k 20,
top_p 0.95`), so the design keeps a host path: when a request asks for
sampling, the replay's logits (248,320 fp32 = 993 KB) are read back, the
host masks ids ≥ 248,077, applies temperature/top-k/top-p with a seeded
RNG, and writes `cur_token[0]` before the next replay - which is exactly
the ingest protocol `Control` already supports. **Cost, derived:** ~1 MB
over PCIe + a 248k-element partial sort ≈ 0.2-0.4 ms/token on a 31 ms
step, 1% class; measured before it is offered. It is the **last** task
in the ladder and ships only if it measures under 2% of tg; greedy is
never slowed by it.

### 3.6 Streams (delegable, independent)

| stream | owns | interface | tested without |
|---|---|---|---|
| **T1 tokenizer** | Rust crate + C FFI + CMake/Cargo step; parity corpus generator (`tools/tokenizer/dump_parity.py`, runs in the reference container) | `tok::encode(str) → ids`, `decode(ids) → str`, `token_to_piece(id)`; `tests/tokenizer/parity_test` | everything else |
| **T2 template + streamer** | minja vendoring; `chat::render(messages, tools, enable_thinking) → str`; `Streamer::push(id) → vector<string>` | `tests/tokenizer/template_test`, `streamer_test` | T3, engine |
| **T3 server** | httplib/json vendoring; schema; SSE; FIFO; stop logic; `/v1/models` | `server::run(engine, tok, chat, opts)`; `tests/server/protocol_test` against a **mock engine** that returns scripted ids | T1/T2 (mocked), the GPU |
| **T4 integration + gate** | wiring; `b70-serve` binary (`--model <snapshot> --port --host --max-len --device`); the golden-through-server test; the llama-benchy run; the HTTP-vs-CLI tg control; host sampling (§3.5) | the gate | - |

T1-T3 run in parallel; T4 is sequential after them.

## 4. The gate

On the idle box, RTN checkpoint: bars 1-6. Record `tg256` through HTTP
beside the CLI row in `docs/BENCHMARKS` with the HTTP delta; record the
llama-benchy `pp` row **labelled "prefill = decode replay per id" if spec 2
has not landed** - it is the honest number for the server as shipped.
Pass → tag `spec3-done`; README v1 definition-of-done row updated.

## 5. Constraints

- Tokenizer/template/server code never touches `src/kernels`/`src/l0`;
  `src/tokenizer` and `src/server` are leaves (docs/04 component rule).
- Python in `tools/` only (parity dumps run in the reference container,
  never on the serving path). The server never downloads anything: the
  model is a snapshot path or a repo id resolved through the existing
  loader (`B70_TEST_SNAPSHOT`-style explicit path preferred in tests).
- Pinned versions for the crate, minja, httplib, json - recorded in one
  `third_party/VERSIONS` file.
- Box workflow `tools/box.sh`; Rust toolchain on the box verified in T1's
  first task (install path recorded, user-local, no sudo).
- Every number measured/derived/estimated with grade; no two values for
  one quantity.

## 6. Out of scope

Prefill (spec 2); session continuation / prefix caching (queued after
this spec per docs/04 - the design keeps the last state resident, which is
its prerequisite); continuous batching / concurrency > 1; tool-call
*parsing* of model output; vision/audio inputs; a hand-written BPE;
authentication; TLS.

## 7. Risks

- **The template uses a Jinja construct minja lacks.** Bar 2 finds it on
  day one; mitigation is a recorded per-model fallback, as
  `openvino.genai` does.
- **Rust toolchain on the box.** T1's first task installs `rustup`
  user-local and proves the static link; if the box cannot build it, the
  `.a` is built on the Mac for `x86_64-unknown-linux-gnu` and copied -
  recorded as a build deviation.
- **`usage.prompt_tokens` disagreement with llama-benchy's local
  tokenizer** would silently mis-calibrate pp; bar 1 makes disagreement
  impossible by construction (same crate), and the warmup delta is
  logged.
- **SSE buffering by a proxy or by httplib defaults** would corrupt pp;
  the protocol test asserts the first frame's write timestamp precedes
  the second token's generation.
- **Host sampling slower than derived.** It is last and gated on its own
  measurement; greedy is unaffected either way.
