# Tokenizer and chat template

`/v1/chat/completions` cannot exist without a byte-level BPE encoder, a Jinja
chat template, and a detokeniser that streams partial UTF-8 safely. All three
are host code, and all three are places where owning the dependencies and
shipping something that works pull in opposite directions.

## What the checkpoint ships

Read from the checkpoint snapshot:

| File | Contents |
|---|---|
| `tokenizer.json` | HF `tokenizers` format, `tokenizer_class: Qwen2Tokenizer`. BPE with **248 044 vocab + 247 587 merges**, plus **33 added tokens** → 248 077 ids |
| `chat_template.jinja` | the chat template, **8.9 KB, a separate file** - `tokenizer_config.json` has no `chat_template` key |
| `tokenizer_config.json` | `eos_token <\|im_end\|>`, `pad_token <\|endoftext\|>`, no `bos_token` |
| `generation_config.json` | `eos_token_id: [248046, 248044]` = `<\|im_end\|>`, `<\|endoftext\|>` - **both** stop generation; `bos_token_id 248044`; sampling defaults `temperature 1.0, top_k 20, top_p 0.95` |
| `config.json` | `vocab_size 248320` - `lm_head` is **padded by 243 rows**; the sampler must mask ids ≥ 248 077 |

Pre-tokenizer (verbatim from `tokenizer.json`): a `Split` with the regex

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

followed by `ByteLevel` (`add_prefix_space: false`, `use_regex: false`);
decoder is `ByteLevel`. This is the Qwen2 pattern - note `\p{N}` matches
**single** digits (not runs of three as in GPT-4's pattern) and the
`(?i:…)` contraction group needs case-insensitive matching. A hand-written
matcher must reproduce exactly this, including the `\s+(?!\S)` lookahead.


### `pretokenize_regex` discrepancy

The checkpoint's `tokenizer_config.json` `pretokenize_regex` contains `\p{M}`
twice where `tokenizer.json` does not. Both patterns are verbatim:

`tokenizer.json`:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

`tokenizer_config.json`:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

The crate reads `tokenizer.json`; the committed parity corpus is the arbiter.
Any future hand-written BPE must match `tokenizer.json`'s pattern, not the
config's.

Added tokens: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<think>`,
`</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`,
`</tool_response>`, the FIM set, and the vision/audio/TTS markers.

Template features: ChatML (`<|im_start|>role\n…<|im_end|>`), **thinking mode**
wired to `enable_thinking`, **tools**, and multimodal content handling that
calls `raise_exception` on images in system messages. It uses Jinja
`macro`, `namespace`, `is string` / `is iterable` / `is mapping` tests and
`raise_exception` - all of which `minja` supports, but verify against *this*
file in the acceptance test rather than assuming.

## Where it sits in the request path

```
request JSON ─► chat template ─► BPE encode ─► prompt token ids ─► prefill
                                                                     │
SSE chunk ◄── UTF-8-safe incremental detokenise ◄── token id ◄── decode replay
```

- **Encode**: once per request, on the host, before prefill. Cost is
  irrelevant against a 4k-token prefill.
- **Detokenise**: once per generated token. Also host work, but **off the GPU's
  critical path** - it overlaps the next replay. It must not block the submit.
- **Stop check**: compare each token id against the `eos_token_id` list, plus
  user `stop` strings matched on the decoded text.

Nothing here touches the decode command list. The control block (doc 04)
carries `cur_token` out; the host detokenises it while the next replay runs.

## Options for the BPE encoder

| Option | Parity with HF | Dependency cost | Static binary? | Verdict |
|---|---|---|---|---|
| **HF `tokenizers` (Rust) via C FFI** | exact - it *is* the reference | a Rust toolchain in the build; a 16 MB static archive | yes (a Rust static lib links fine) | **what is built** |
| `tokenizers-cpp` (MLC) | exact - wraps the same Rust crate, adds SentencePiece | same Rust dep plus a C++ shim we'd only half use | yes | acceptable; buys little over the crate directly |
| `openvino_tokenizers` | exact | drags in the OpenVINO runtime, the thing this project exists to not depend on | no | rejected |
| **Hand-written byte-level BPE in C++** | must be *proven* - the pre-tokenizer regex and Unicode categories are where ports diverge | zero; needs a Unicode-aware regex (or a hand-compiled matcher for Qwen's fixed pattern) | yes | **the end state if owning every dependency matters**, not the starting point |


### What is built

**The first option**: HF `tokenizers` (Rust) via C FFI, as the leaf
`src/tokenizer/` library, behind a four-function C interface (`encode`,
`decode`, `token_to_piece`, `free`) so that nothing else knows which
implementation is behind it. The crate is pinned to `tokenizers 0.22.2`; its
C++ RAII wrapper is `tok::Tokenizer`. The committed Python-reference corpus has
10,240 cases and reports zero encode and zero decode mismatches.

The release static archive is 16,217,384 bytes and a fresh Cargo build takes
50.36 s. Normal CMake builds cache Cargo outputs in `build/tokenizer-rs`;
`B70_TOKENIZER` is `AUTO` by default and may be set to `ON` to require the
component or `OFF` to omit it.

Replace it with a hand-written BPE only once the parity test proves the
replacement. **The parity test is the deliverable that makes the swap safe**, so
it was written first and it is kept forever.

`chat::Template` runs over `google/minja` and `nlohmann/json v3.12.0`, both
pinned by source URL and header SHA-256 in `third_party/VERSIONS`. The
`transformers` parity vectors for the gate checkpoint match byte for byte:
thinking on is 565 bytes, thinking off is 367 bytes, and the one-function tools
case is 1547 bytes.

The checkpoint's `chat_template.jinja` SHA-256
`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`
uses Jinja's unsupported `is undefined` test. `chat::Template` selects a
hash-gated fallback for exactly that source and no other; it changes that one
construct to minja's equivalent `is not defined` without editing the
checkpoint's template file.

`tok::Streamer` is the hold-and-flush incremental detokeniser described under
"Streaming detokenisation" below.

### What it costs at run time

`b70-serve` consumes this tokenizer, template and streamer on the real
checkpoint. `golden_server_test` reproduces `b70-decode --ids --prefill`'s exact
ids on all three golden prompts, both sides on the same engine path.

The host cost is **not measurably present**. Decode over HTTP measured
29.902 t/s against the same session's CLI control of 29.31, i.e. 102.0%, against
a 2% bar registered before the measurement. Prefill over HTTP measured
2915.83 ms for 4096 tokens against the CLI's 2919.6, a derived difference of
-3.77 ms, indistinguishable from zero. Encode plus HTTP framing cost nothing on
top of the device-side prefill, and detokenisation overlaps the next replay.

Host sampling (temperature, top-k, top-p) measured 0.537 ms/token against a
0.62 ms bar registered before the measurement, and ships on by default. Every
row and its grade is in [BENCHMARKS.md](BENCHMARKS.md).

## Chat template

Qwen templates are real Jinja: loops over messages, `if` on roles, tool-call
blocks, a thinking toggle. Hand-substituting `{role}\n{content}` will break on
the first multi-turn request.

Use **`google/minja`**, a header-only C++ Jinja subset built for exactly this.
It is what `openvino.genai` uses, and therefore what OpenVINO Model Server uses;
`llama.cpp` uses it too. One header, no runtime dependency, static-friendly. If
a template uses a construct minja lacks, `openvino.genai` keeps a per-model
fallback map - copy the idea, not the file, which is what the hash-gated
fallback above is.

Tool-call *parsing* of model output is not implemented. Template *rendering* of
tool definitions in the prompt is, and has parity vectors.

**Agnes 3.0 Flash (spec 14, 2026-10-03).** Its `chat_template.jinja` is byte-identical
to Qwen3.8's (the same sha256 above), so it takes the same fallback, renders the same
Qwen XML tool-call format and needs no parser addition. `template_agnes_test` renders
five message lists (`tests/tokenizer/agnes_template_cases.json`: plain, thinking, tools,
a tool call in history, tool responses) through Agnes's snapshot against transformers'
render (`tools/tokenizer/dump_agnes_template.py`); details in
`docs/probe-agnes-2026-10-03.md`.

## Streaming detokenisation

Byte-level BPE tokens are byte sequences, not characters. A multi-byte
code point (any CJK character, most emoji) can straddle two tokens, so decoding
token-by-token emits U+FFFD replacement characters mid-character. The standard fix, as `openvino.genai`'s text streamer implements it:

1. keep the ids since the last flush;
2. decode the whole pending run, not the last token;
3. if the decoded text ends in `\xEF\xBF\xBD`, it is incomplete - hold it;
4. otherwise emit the suffix after `m_printed_len` and advance the cursor;
5. flush on newline to bound the held run.

OpenVINO Model Server subclasses that streamer; vLLM does the same thing with a
prefix-diff over `(prefix_offset, read_offset)`. It is implemented once in
`src/tokenizer/streamer.*`: the SSE layer calls `push(id)` and receives zero or
more UTF-8-complete strings. On the 10,240-case committed corpus `tok::Streamer`
has zero concatenation mismatches and zero spurious U+FFFD emissions, and its
measured longest hold is 4 ids.

## Acceptance test

Parity against Python `tokenizers` on a corpus, produced once by a script in
`tools/` and checked into `tests/tokenizer/`:

- ≥ 10 k lines spanning ASCII, CJK, emoji, code, whitespace runs, and every
  special token - **byte-identical token ids** to HF on encode;
- decode(encode(x)) == x for every line;
- the streaming detokeniser, fed the ids one at a time, concatenates to the
  same string as one-shot decode, and never emits U+FFFD for valid input;
- the chat template rendered for a 3-turn conversation equals
  `tokenizer.apply_chat_template(...)` from `transformers`, byte for byte.

Any tokenizer implementation, the Rust one or a later hand-written one, must
pass this before it is wired to the server. That is what makes the dependency
swappable.

## What this costs the design

- `src/tokenizer/` is a leaf: it depends on nothing in `src/` and nothing in
  `src/` except `server/` depends on it. Keep it that way.
- The build gains a Rust step. Cache it the way `ccache` caches the C++ side
  (doc 09); it is one crate, compiled once.
- `tools/` gains a Python script that dumps golden vectors. Python appears in
  `tools/` only, as doc 04 requires.
- Owning every dependency is deferred for this one component, explicitly, with
  the parity test as the path back. That is a named trade-off: a correct server
  now against a dependency-free one later.
