# Spec 7 - prefix caching for long agentic sessions

**Status:** design, 2026-09-27, for operator review.

**Order, set by the operator:** spec 5 (int8 prefill linears) is the
foundation, spec 6 (flash attention, 128k) the first feature on top of it.
This spec is the second. MTP comes after.

Every number is **measured** unless marked **derived** or **estimated**.

## 1. Why

The workload is long agentic coding sessions driven by **opencode**
(operator, 2026-09-27; memory: target-workload-agentic-coding). Every request
today runs `engine.reset()` and prefills the whole prompt
(`src/server/server.cc:146`). opencode resends the whole conversation each
turn, so a turn at 60k tokens of history re-prefills 60k tokens: about 50 s
at spec 6's rates at depth (pp32768 1499 t/s, pp65536 1122 t/s; derived).

Two facts about this model and this client decide the design:

1. **The GDN state cannot be rewound.** Qwen3.8 is hybrid. Its 16 FA layers
   keep a KV cache, and any prefix of it stays valid when `pos` is moved
   back. Its 48 GDN layers keep a recurrent state (`gdn_state`, fp32,
   150.99 MB) and a conv ring (`conv_ring`, 15.73 MB), valid only at the
   exact position they were computed to. Reuse therefore needs **snapshots**
   of that state at chosen positions.
2. **opencode interleaves side requests** (title generation, subagents) that
   diverge from the main session near position 0. The engine has one KV
   cache, so each side request overwrites the main session's KV. A resident
   session alone is evicted every few turns.

The chat template may also drop earlier turns' `<think>` blocks, in which
case the resent prompt does **not** start with "previous prompt + generated
ids", and plain session continuation (`docs/04-architecture.md`, "Not built:
prefix caching", stage 1) misses on every turn.

## 2. The decision

**Approach 2 (operator, 2026-09-27): one resident session on the card, and a
write-through store in pinned host RAM of KV blocks and state snapshots.**
Rejected in the design discussion:

- **Resident session only, snapshots on the card:** every opencode side
  request evicts the main session; only plain back-and-forth chat benefits.
- **Several resident sessions on the card:** a second 128k KV cache does not
  fit (spec 6 §8: 28.1 of 32.5 GB at 131072), so side slots would be 8k to
  16k with a captured decode list each. More code, and it overlaps batching.

## 3. Design

### 3.1 The store

In pinned host memory (`l0::Mem` host allocation: host-resident,
device-visible), with a byte budget `--prefix-cache-gb` (default 32, `0`
disables the cache and the server behaves exactly as today).

- **KV block:** positions `[2048 b, 2048 (b + 1))`, aligned to absolute
  position; K and V of the 16 FA layers, `[16][2048][4][256]` bf16 each,
  **128 MiB per block** (derived).
- **State snapshot:** `gdn_state` + `conv_ring` + `pos`, **166.7 MB**
  (derived). Taken:
  - at every block end during prefill;
  - at the end of every prompt, carrying the KV of positions since the last
    block boundary (at most 2047, at most 128 MiB);
  - at the end of every request, after the response is sent, carrying the KV
    of the positions since the last stored boundary, generated tokens
    included.
- **Keys.** Every entry stores the full token ids it covers and a 64-bit
  hash chain over them. Lookup matches on the hash, then **compares the ids
  exactly**, so a collision cannot produce a wrong hit.
- **Structure and eviction.** Entries form a tree: a block's parent is the
  block before it, a snapshot's parent is the last block it depends on. LRU
  eviction removes **leaves only**, so no snapshot ever outlives the KV blocks
  it needs. Entries in use by the current request are pinned against
  eviction.

### 3.2 Write-through

During every prefill, as each block completes, its KV and a state snapshot
are copied device to host on the prefill immediate list (301 MB per 2048
chunk, ~25 ms at an estimated 12 GB/s PCIe 3 x16, against ~1.4 s of
prefill; derived). Eviction from the card therefore copies nothing: the
main session is already on the host when a side request overwrites it.

The prefill of a tail starting at `p` runs its first chunk to the next
absolute 2048 boundary, then full chunks, so every completed block is a
storable block. Chunking is otherwise unchanged (`PrefillScratch::kC`).

### 3.3 A request

1. `d` = the common prefix of the new prompt's ids with the **resident**
   session's ids (prompt and generated).
2. Restart point `p`, the largest candidate with `p <= len - 1`:
   - the resident `pos`, if the resident ids are a prefix of the prompt
     (pure continuation: nothing is copied);
   - otherwise the deepest host snapshot whose ids are a prefix of the prompt;
   - otherwise `p = 0`: a cold prefill after `reset()`. (The card holds a
     valid GDN state only at the resident `pos`, so a divergence below it
     always restarts from a host snapshot or from 0.)
3. Restore: the snapshot's state to the card; the KV positions `[d, p)` from
   the host blocks and the snapshot's partial KV (positions below `d` are
   already correct on the card); `control.pos = p`.
4. Prefill `ids[p:]`, at least one token, so the first generated token always
   comes from a real forward pass (`cur_token` is never restored).
5. `usage.prompt_tokens_details.cached_tokens = p` (the OpenAI field).

A prompt whose history keeps its `<think>` blocks restarts from the previous
request's end snapshot; one whose history drops them restarts from the
previous prompt's end snapshot. Both are the same lookup.

### 3.4 Where the code changes

- **`runtime::Engine`:** `snapshot_to(HostSnapshot&)`,
  `restore_from(const HostSnapshot&, uint32_t pos)`, `copy_kv_to_host` /
  `copy_kv_from_host` for a position range, and a per-block callback in the
  prefill chunk loop. Device-host copies on the prefill immediate list. The
  captured decode list is not touched: it reads `pos` and the persistent
  buffers, which restore writes in place.
- **`src/server/prefix_cache.{h,cc}` (new):** the store, the lookup and the
  LRU. Pure host code behind an interface, unit-testable without a GPU.
- **`server::EngineIface`:** gains the snapshot, restore and KV-range calls;
  `EngineAdapter` implements them.
- **`server.cc`:** `reset()` + full prefill becomes lookup, restore, tail
  prefill; the request-end snapshot is taken after the last chunk is sent.
- **`b70_serve`:** `--prefix-cache-gb`. `b70_decode` and the benchmarks are
  unchanged.

### 3.5 Tool calls and reasoning in the response (prerequisite)

`b70_serve` returns the generated text as `content`, with no `tool_calls` and
no `reasoning_content`, so opencode cannot drive it, and C3 needs opencode.
The checkpoint emits the **Qwen XML tool-call format** (`<tool_call>` /
`<function=NAME>` / `<parameter=KEY>` ... `</parameter>` / `</function>` /
`</tool_call>`; `tools/toolcall/score.py`), the format vLLM's `qwen3_coder`
and `qwen3_xml` parsers both read (two implementations of one format; the
second is the streaming-robust one). `hermes` (JSON inside `<tool_call>`) is
Qwen2.5's and wrong here.

- A C++ parser in `src/server/toolcall.{h,cc}` with `qwen3_xml`'s
  semantics: each parameter's text is converted by the tool's JSON schema
  type (string: as is, one leading and one trailing newline stripped;
  integer / number / boolean / object / array: parsed as JSON, falling back to
  the string); an unterminated call at the end of generation is still emitted
  if its function tag is complete.
- Reasoning: with thinking on, the prompt ends in `<think>\n`, so the output
  up to `</think>` is `reasoning_content`, the rest `content`.
- Non-streaming: `message.reasoning_content`, `message.content` (null when
  empty and there are tool calls), `message.tool_calls`, finish reason
  `tool_calls` when there are any. Streaming: `reasoning_content` and
  `content` deltas as text arrives; each tool call as one delta (id, name
  and complete `arguments`) when its `</function>` arrives.
- `/v1/completions` is unchanged (raw text).

This matters to the cache as well: the template keeps earlier turns' thinking
when the client sends `reasoning_content` back (`preserve_thinking`, default
on), and then the resent history matches the generated ids further.

## 4. Correctness gates

- **C1, bitwise restore.** Prefill 4096 + 300 ids and snapshot. `reset()`,
  overwrite the persistent buffers with a second prompt, restore, generate 64
  greedy tokens: **bitwise identical** to generating straight on, and the KV
  cache byte-identical over `[0, pos)`. Also at a block boundary (p = 4096)
  and from the request-end snapshot.
- **C2, cached against cold.** Multi-turn sequences whose divergence falls at
  a prompt end, mid-block, and exactly on a block boundary. The cached run's
  logits and greedy tokens against a cold prefill of the same prompt, under
  the tie-aware golden rule. Not bitwise: the tail's chunk boundaries differ
  from the cold run's, as in the chunk-1000 and chunk-16 gates.
- **C3, a real opencode session.** One real opencode session recorded as a
  request log (title and subagent requests included) and replayed against
  `b70_serve` with the cache on and off: greedy outputs agree under the
  tie-aware rule, every tool call parses as valid JSON with the same name and
  arguments, and `cached_tokens` is logged per request.
- **C4, host unit tests** (no GPU): LRU and budget; a snapshot never outlives
  its blocks; entries in use are not evicted; exact-ids comparison under a
  forced hash collision; a prompt equal to a snapshot position still prefills
  one token; `--prefix-cache-gb 0` equals today's behaviour.
- **C6, tool-call output.** Host tests of the parser on the golden
  tool-call outputs (`tests/golden/toolcall/`) and on partial and malformed
  text, streamed and not; every call's name and arguments equal
  `tools/toolcall/score.py`'s reading of the same text.
- **C5.** Every existing registration passes unchanged, the full suite green.

## 5. Speed bars

Idle box, device 0, median of 3.

- **P0, baseline (first).** Host-device copy bandwidth both ways for pinned
  host memory on the T5810, the largest pinned host allocation the driver
  gives, one snapshot's copy time, the short-tail prefill floor (~0.7 s,
  probe-w4a8 §15.6), and on the recorded opencode log: how far each request's
  ids match the previous requests' (the hit rate this design can reach).
- **S1.** A continuation turn with 60k tokens of history and a 1k-token tail
  reaches its first token in **<= 1.5 s** (cold today: ~50 s, derived).
- **S2.** The main session returning after a side request, at 60k tokens:
  restore **<= 1 s**, plus the tail.
- **S3.** Write-through costs cold pp4096 **<= 3 %**, interleaved pairs
  against `--prefix-cache-gb 0`.
- **Recorded:** time to first token per request over the C3 replay, cache on
  and off.

## 6. Stages

- **7a, output and probe:** §3.5 and C6; a request log in `b70_serve`;
  the recorded opencode session; P0.
- **7b, engine:** snapshot, restore, KV-range copies, the per-block
  callback, aligned tail chunking; C1; S3.
- **7c, server:** `prefix_cache`, the request path, `cached_tokens`,
  `--prefix-cache-gb`; C2 to C5; S1, S2 and the C3 record.

**Stopping rule.** If P0 finds host-device bandwidth under 4 GB/s, or a
pinned allocation under 8 GB, S2 is not reachable as designed; stop and
record for the operator's call before 7b.

## 7. Out of scope

- Several resident sessions, batching, sharing across concurrent requests.
- Cache persistence across server restarts (disk).
- The ~0.7 s short-tail floor itself: measured here, fixed separately. With
  the cache it becomes most of a typical turn's time to first token.
- MTP: next.

## 8. Amendment: results (plan 7c, 2026-09-28)

Built on branch `spec7c-server-prefix-cache` (rebased on main b7368b2). Records:
`docs/probe-prefix-cache-2026-09-27.md` §8 (S1, S2) and §9 (C3 on a synthetic session),
`docs/BENCHMARKS.md` "Prefix caching (spec 7)".

- **As built.** `src/server/prefix_cache.{h,cc}`: `PrefixCache` (the store and `plan()`)
  and `PrefixSession` (the request path over `EngineIface`); `b70-serve
  --prefix-cache-gb N` (default 32, one pinned host allocation carved first-fit; `0` =
  exactly the old server). A restore loads whole store entries, so `kv_from` is the
  common prefix with the resident session rounded down to the entry start (at most one
  extra block, ~10 ms). Missing blocks of a chain (positions crossed during generation)
  are stored at the next store call, so a request-end snapshot's KV stays below one block.
- **Review Focus 5 / C4 "prefills one token".** Holding a snapshot at `len - 1` needs the
  last prompt id fed separately. Built as an opt-in (`--prefix-split-last`, chat only):
  the tail prefills to `len - 1` and the last id goes through one decode replay. It is
  **off by default** because it is not the cold run's arithmetic: `golden_server_test`
  failed on `cjk` with it on (completions therefore never split), and on the synthetic C3
  replay 2 of its 4 divergences fail the tie rule. Default: the prompt-end snapshot is at
  `len`, restarts are `<= len - 1`, so the same prompt again restores at the block end below
  it. Mock tests cover both paths.
- **C1** (plan 7b) as recorded there. **C4**: `prefix_cache_test`, `prefix_server_test`
  pass. **C2**: `prefix_gpu_{l0,int8}_test` (default) and `prefix_gpu_split_{l0,int8}_test`:
  every sequence a-f passes on l0-int8 both ways and on l0 with the split; **on l0 without
  the split, sequence e (restore at 5064 after a side request) misses one row: row 30 of
  32, cached 1756 = the cold run's runner-up at 0.108 bf16 ulp, a near-tie the l0 rule
  (allowance 0) does not accept.** Deterministic (twice). For the operator: accept the
  near-tie allowance on l0 for C2, or not.
  **Operator ruling (2026-09-28): accepted.** `prefix_gpu_l0_test` runs with a near-tie
  allowance of 1, the same as l0-int8: the flip is the GDN-chunk-boundary rounding class
  proven in `docs/probe-prefill-continuation-2026-09-28.md` (where a prefill's chunk
  boundaries fall changes its rounding, and the model amplifies that at a few positions;
  the CPU oracle shows the same from chunking alone; not a cache fault). The split registration `prefix_gpu_split_l0_test` keeps allowance 0.
- **C3**: the recorded opencode session does not exist yet; **pending the log**. On a
  synthetic opencode-shaped log (15 requests, 2 side), default path: 15/15 bitwise equal to
  the cold server, all tool calls equal, time to first token 17.46 s vs 45.44 s.
- **C5**: full suite 104/105 (2026-09-28, load 1.0): only `prefix_gpu_l0_test` fails, the l0 near-tie above; every pre-existing registration passes unchanged.
- **S1 met**: 1233.4 ms (default), 1294.0 ms (split), bar 1500. **S2 met**: restore
  347.7 ms of a 3933 MB KV + the state, bar 1000; time to first token 1581.2 ms. **S3**
  (7b): 2.03%.

## Amendment: `--prefix-cache-gb auto` (2026-10-05)

The store is pinned **system RAM**, so its size is bounded by the machine, not the card, and an
oversized pin cannot be swapped. `b70-serve --prefix-cache-gb` now defaults to `auto`
(`src/cli/prefix_cache_size.h`): min(32 GiB, half of MemTotal, MemAvailable - 8 GiB), read
after the model is loaded; below 4 GiB the cache is off. The startup line prints the choice
and its inputs; an explicit N pins exactly N GiB and warns when that exceeds MemAvailable.
`prefix_cache_size_test` (host) pins the rule. With at least 64 GiB of RAM and 40 GiB
available, auto chooses 32, the old default, so behaviour on such a machine is unchanged.
