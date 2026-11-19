# Tokenizer and chat template

This doc covers the host-side text handling that turns an OpenAI-style chat request into token ids and the
generated ids back into streamed text. It is for anyone working on the server, adding a model with a new
template, or wondering why a prompt's ids differ from what another implementation produces.

`/v1/chat/completions` needs three pieces of host code before the GPU sees anything: a byte-level BPE encoder
(byte-pair encoding: text is split into bytes and frequent byte pairs are merged into vocabulary entries), a Jinja
chat template (the model's own script for laying out roles, tools and thinking markers as text), and a
detokeniser that streams partial UTF-8 safely. For each of them, owning every dependency and shipping something
that works today pull in opposite directions. The choices below say which way each one went.

## What the checkpoint ships

The Qwen3.8 checkpoint snapshot contains:

| File | Contents |
|---|---|
| `tokenizer.json` | HF `tokenizers` format, `tokenizer_class: Qwen2Tokenizer`. BPE with 248 044 vocab entries and 247 587 merges, plus 33 added tokens, so 248 077 ids |
| `chat_template.jinja` | the chat template, 8.9 KB, as a separate file: `tokenizer_config.json` has no `chat_template` key |
| `tokenizer_config.json` | `eos_token <\|im_end\|>`, `pad_token <\|endoftext\|>`, no `bos_token` |
| `generation_config.json` | `eos_token_id: [248046, 248044]`, which are `<\|im_end\|>` and `<\|endoftext\|>`; both stop generation. `bos_token_id 248044`; sampling defaults `temperature 1.0, top_k 20, top_p 0.95` |
| `config.json` | `vocab_size 248320`: `lm_head` is padded by 243 rows, so the sampler must mask ids ≥ 248 077 |

The pre-tokenizer (the step that cuts text into words before BPE runs on each) is a `Split` with this regex,
verbatim from `tokenizer.json`:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

followed by `ByteLevel` (`add_prefix_space: false`, `use_regex: false`); the decoder is `ByteLevel`. This is the
Qwen2 pattern. Unlike GPT-4's pattern, `\p{N}` matches single digits rather than runs of three, and the `(?i:…)`
contraction group needs case-insensitive matching. A hand-written matcher would have to reproduce exactly this,
including the `\s+(?!\S)` lookahead.

### The two pre-tokenizer patterns disagree

The checkpoint's `tokenizer_config.json` also carries a `pretokenize_regex`, and it contains `\p{M}` (combining
marks) twice where `tokenizer.json` does not. Both patterns, verbatim:

`tokenizer.json`:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

`tokenizer_config.json`:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

The engine reads `tokenizer.json`, and the committed parity corpus decides what is right. Any future hand-written
BPE must match `tokenizer.json`'s pattern, not the config's.

The added tokens are `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, `<tool_call>`,
`</tool_call>`, `<tool_response>`, `</tool_response>`, the FIM (fill-in-the-middle) set, and the vision, audio and
TTS markers.

The template implements ChatML (`<|im_start|>role\n…<|im_end|>`), a thinking mode wired to `enable_thinking`,
tools, and multimodal content handling that calls `raise_exception` on images in system messages. It uses Jinja
`macro`, `namespace`, the `is string` / `is iterable` / `is mapping` tests and `raise_exception`. `minja` supports
all of these, but the acceptance test checks against this exact file rather than assuming.

## Where it sits in the request path

```
request JSON ─► chat template ─► BPE encode ─► prompt token ids ─► prefill
                                                                     │
SSE chunk ◄── UTF-8-safe incremental detokenise ◄── token id ◄── decode replay
```

Encoding runs once per request, on the host, before prefill (the pass that reads the prompt). Its cost is
irrelevant next to a 4k-token prefill. Detokenising runs once per generated token, also on the host, but off the
GPU's critical path: the decode step is a recorded command list replayed once per token, and the host decodes one
token while the next replay runs, so it must never block the submit. The stop check compares each token id against
the `eos_token_id` list and matches the user's `stop` strings on the decoded text.

Nothing here touches the decode command list. The control block ([04-architecture.md](04-architecture.md))
carries `cur_token` out, and the host detokenises it while the next replay runs.

## Choosing the BPE encoder

| Option | Parity with HF | Dependency cost | Static binary? | Verdict |
|---|---|---|---|---|
| HF `tokenizers` (Rust) via C FFI | exact: it is the reference | a Rust toolchain in the build; a 16 MB static archive | yes (a Rust static lib links fine) | what is built |
| `tokenizers-cpp` (MLC) | exact: wraps the same Rust crate, adds SentencePiece | the same Rust dependency plus a C++ shim we would only half use | yes | acceptable, but buys little over the crate directly |
| Hand-written byte-level BPE in C++ | must be proven: the pre-tokenizer regex and Unicode categories are where ports diverge | none, but needs a Unicode-aware regex or a hand-compiled matcher for Qwen's fixed pattern | yes | the end state if owning every dependency matters, not the starting point |

### What is built

The engine uses HF `tokenizers` through C FFI, as the leaf library `src/tokenizer/`. A small C interface
(`b70_tok.h`: `b70_tok_encode`, `b70_tok_decode`, the two id/token lookups, the vocabulary size and the frees)
hides which implementation is behind it from the rest of the code. The crate is pinned to `tokenizers 0.22.2`,
and its C++ RAII wrapper is `tok::Tokenizer`. The committed Python-reference corpus has 10,240 cases and shows
zero encode and zero decode mismatches.

The release static archive is 16,217,384 bytes and a fresh Cargo build takes 50.36 s. Normal CMake builds cache
Cargo outputs in `build/tokenizer-rs`. The CMake option `B70_TOKENIZER` is `AUTO` by default; set it to `ON` to
require the component or `OFF` to leave it out.

A hand-written BPE could replace the crate only if the parity test proves it equal. That test is what makes the
swap safe, so it was written first and is kept permanently.

## Chat template

Qwen's templates are real Jinja: loops over messages, `if` on roles, tool-call blocks, a thinking toggle.
Substituting `{role}\n{content}` by hand breaks on the first multi-turn request.

The renderer is `google/minja`, a header-only C++ subset of Jinja built for exactly this job. It is one header with no runtime dependency and links
statically. `chat::Template` runs over `google/minja` and `nlohmann/json v3.12.0`, both
pinned by source URL and header SHA-256 in `third_party/VERSIONS`. The `transformers` parity vectors for the
Qwen3.8 checkpoint match byte for byte: thinking on renders 565 bytes, thinking off 367 bytes, and the
one-function tools case 1547 bytes.

### Where minja is not Jinja

The checkpoint's `chat_template.jinja` (SHA-256
`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`) uses Jinja's `is undefined` test, which minja
does not support. A per-model fallback is the usual way to handle such cases, and `chat::Template` does that:
for exactly that template hash, and no other, it rewrites that one construct to minja's
equivalent `is not defined`, without editing the checkpoint's file.

Beyond that, the bundled minja is patched in the places where it behaved differently from Jinja: Undefined kept
apart from None, `is sameas`, `str.split()`, the `replace` filter, `dict`, index attribute getters, empty mappings
being falsy, `none`, and the tool-call capability probe. Each patch is listed in `third_party/VERSIONS` and tested
against Jinja2 by `minja_ext_test`. The patches were first needed by a model's template that has since been
removed.

A request's `chat_template_kwargs` reach the template through `chat::Template::render(..., kwargs)` for models
whose `server::ChatFormat` reads them. Kolibri-1 reads them (including `reasoning_effort`); Qwen only takes
`enable_thinking`. Ornith and Kolibri-1 have their own template tests, `template_ornith_test` and
`template_kolibri_test`.

### Tool calls

Rendering tool definitions into the prompt is covered by the parity vectors. Parsing tool calls out of the model's
output is server code, one parser per format in `src/server/toolcall*.cc`: Qwen's XML format (Qwen3.8, Ornith,
Flash-Next) and Kolibri-1's hermes JSON inside `<tool_call>`.

### Key order is part of the prompt

A chat request is parsed with `nlohmann::ordered_json`, not `nlohmann::json`, whose objects are sorted maps. The
template therefore sees the client's `messages`, `tools` and tool-call arguments in the order the client wrote
them, and a parsed call keeps the model's argument order. This matters because the template prints tools with
`tojson` and iterates arguments with `| items`. With sorted keys, the prompt ids differed from what transformers
and vLLM render for the same request (0 of 36 agentic prompts matched), and a client echoing a call back had it
re-rendered in a different order from the one the model wrote, so the next turn diverged inside the
prefix-cached span. With ordered keys, `prompt_order_test` matches 36 of 36 agentic prompts and 12 of 12 edit and
copy prompts against the reference tokenizer's ids.

## Streaming detokenisation

Byte-level BPE tokens are byte sequences, not characters. A multi-byte code point (any CJK character, most emoji)
can straddle two tokens, so decoding one token at a time emits U+FFFD replacement characters in the middle of a
character. The standard fix is a hold-and-flush streamer:

1. keep the ids since the last flush;
2. decode the whole pending run, not just the last token;
3. if the decoded text ends in `\xEF\xBF\xBD`, it is incomplete, so hold it;
4. otherwise emit the suffix after `m_printed_len` and advance the cursor;
5. flush on a newline to bound how long a run can be held.

vLLM does the same thing with a prefix diff over
`(prefix_offset, read_offset)`. Here it is `tok::Streamer` in `src/tokenizer/streamer.*`: the SSE layer calls
`push(id)` and gets back zero or more UTF-8-complete strings. On the 10,240-case corpus it has zero concatenation
mismatches and zero spurious U+FFFD emissions, and the longest run it ever holds is 4 ids.

## Acceptance test

Any tokenizer implementation, the Rust crate or a later hand-written one, must pass a parity test against Python
`tokenizers` before it is wired to the server. That is what keeps the dependency swappable. The corpus is
produced once by a script in `tools/` and checked into `tests/tokenizer/`, and the test requires:

- byte-identical token ids to HF on encode, over at least 10 k lines spanning ASCII, CJK, emoji, code, whitespace
  runs and every special token;
- `decode(encode(x)) == x` for every line;
- the streaming detokeniser, fed the ids one at a time, concatenates to the same string as a one-shot decode and
  never emits U+FFFD for valid input;
- the chat template rendered for a 3-turn conversation equals `tokenizer.apply_chat_template(...)` from
  `transformers`, byte for byte.

## What it costs at run time

`b70-serve` uses this tokenizer, template and streamer on the real checkpoint. `golden_server_test` checks that
the server produces exactly the ids `b70-decode --ids --prefill` produces on all three golden prompts, with both
sides on the same engine path.

The host cost does not show up in measurements. These numbers are from the first Qwen3.8 server measurements, on
an older driver stack, each against a bar set before the run. Decode over HTTP measured 29.902 t/s against
29.31 t/s for the CLI in the same session (102.0%, against a 2% bar). Prefill over HTTP took 2915.83 ms for 4096
tokens against the CLI's 2919.6 ms, a difference of -3.77 ms, indistinguishable from zero. Encoding and HTTP
framing add nothing on top of the device-side prefill, and detokenisation overlaps the next replay.

Host sampling (temperature, top-k, top-p) measured 0.537 ms/token against a 0.62 ms bar, and every request that
asks for sampling uses it.

## What it costs the design

- `src/tokenizer/` is a leaf: it depends on nothing else in `src/`, and nothing in `src/` except `server/` depends
  on it. Keep it that way.
- The build gains a Rust step. It is cached the way `ccache` caches the C++ side
  ([09-vllm-patch-postmortem.md](09-vllm-patch-postmortem.md)); it is one crate, compiled once.
- `tools/` gains a Python script that dumps the golden vectors. Python appears only in `tools/`, as
  [04-architecture.md](04-architecture.md) requires.
- Owning every dependency is deliberately deferred for this one component, with the parity test as the way back.
  The trade-off is a correct server now against a dependency-free one later.
