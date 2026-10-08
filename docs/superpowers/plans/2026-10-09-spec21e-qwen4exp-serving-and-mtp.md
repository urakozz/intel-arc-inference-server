# Spec 21e - Qwen3.8-Flash-Next served: template and tokens, prefix-cache snapshots, the MTP head, draft / verify

**Status (2026-10-09): planned; nothing built.** Built blind on the Mac; box queue row 34 (it renumbers at build
time if taken). Tasks 1-2's host halves need neither 21c nor 21d and can run first. The full model is served
only after spec 22: until then `b70-serve` runs the synthetic checkpoints and Intel's truncated with `--layers N`,
and the full-model rows (A4, passkey, the engine's acceptance and speed) are spec 22's.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `b70-serve <qwen4exp> --layers N` serves OpenAI chat for agentic sessions on one or two cards: the checkpoint's template (Qwen3.8's, sha256 `c3cf9e34...`) and tokenizer through the server's existing Qwen path (XML tool calls, reasoning, the Qwen parser), spec 7's prefix cache with this family's state (GDN states and conv rows, the PLE history and conv rows, the indexer's open-block raw keys) and blocks (KV + compressed indexer keys), the MTP head (vLLM's form) drafting through spec 8's draft / verify / commit, verify at M = K + 1 with every row's own selection and top-10; F4's restores and F5 (verify rows bitwise M = 1, greedy lossless, acceptance from the CPU reference for both `pre_fc_norm_hidden` forms); the tool-call set and its CPU reference prepared for spec 22's A4 run.

**Architecture:** spec 21 §4.5, §4.6 (prefix cache, MTP rows), §8 21e. The server's request path is unchanged: `server::ChatFormat::for_model_type("qwen4_exp")` already reads as `Kind::Qwen` (anything not listed, `src/server/chat_format.h`), whose parser is `server::OutputStream` (Qwen XML) and whose reasoning keys on the prompt's `<think>\n`. The engine side reuses spec 16d's adapter: `cli::pp::PipelineEngineAdapterT<Engine>` (`src/cli/pipeline_serve_adapter.h`) is EngineAdapter's logic over a host-side engine interface - reset, prefill, ingest, generate(1), pos, max_len, kBlock, kMaxDraft, state_bytes, kv_bytes, kv_cache, save/load_state, save/load_kv, set_block_hook, pending, set_token, read_logits_into(host, rows), read_draft_logits_into(host, i), host_rows, mtp, max_verify_k, draft(k, pick), draft_ids, verify(k), verify_ids, commit(j, t), set_draft_input(i, id) - so `Qwen4ExpEngine` implements exactly that interface and `b70-serve` instantiates `PipelineEngineAdapterT<runtime::qwen4exp::Qwen4ExpEngine>`: no third copy of the speculative loop, greedy and sampled acceptance (`server::accept_greedy` / `accept_sampled` / `accept_point_mass`) as they are.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, nlohmann::json, minja, the Rust `tokenizers` crate; Python 3 in `agnes-ref-img` with 21a's 5.19.0 site (the template / tokenizer dumps, the acceptance runs, the A4 reference).

**Spec:** `docs/superpowers/specs/2026-10-09-spec21-qwen4exp-design.md` (§1 tokens, §2.5, §4.5, §4.6, §7 F4 F5, §8 21e, §10 decisions 4, 5, 6, 9, 11). Spec 8 (draft / verify / commit, M2 / M3, §11 draft vocabulary), spec 7 (the prefix cache's state and blocks), spec 16 §8-§10 (snapshot layouts identical under `--pp 1` / `--pp 2`; MTP across the split, 16d). Precedents: plan 15e (Ornith's MoE MTP head, its bf16 experts RTN at load, `golden_server_test` pointed by hand), plan 20e Task 3 (`KolibriEngine`'s snapshot calls, `kolibri_snapshot_{,gpu_}test`), `src/runtime/engine.h:210-259` (spec 8's API and its contracts), `src/cli/pipeline_serve_adapter.h`, `tests/runtime/{mtp_verify_test,mtp_head_test,ornith_mtp_test,pp_mtp_test}.cc`, `tools/toolcall/{a4_ref.sh,oracle_generate.py}`.

## Dependencies and branch points

- **Plans 21c and 21d merged** for Tasks 2-5 (the engine, its prefill and block hook). Task 1 is host-only.
- **21a's MTP port and acceptance** (`tools/oracle/qwen4exp_mtp.py`: `MtpHead`, `chain`, `accept`) is the reference for M1 and for the acceptance numbers; the head's semantics are vLLM's (spec 21 decisions 1, 4, 5).
- **The MTP head's experts:** 21b loads Intel's bf16 head experts RTN-quantised at load (spec 15e's precedent); decision 6 may replace them with 21q's. Every MoE kernel serves them unchanged (int4 layout-1 blocks).
- **Decision 4 (`pre_fc_norm_hidden`):** vLLM's single RMS over 10240 is the default; the per-stream form is built behind `B70_Q4_MTP_NORM=single|per_stream` (read at construction, anything else throws) so F5's acceptance is measured for both (from the CPU reference now; from the engine after spec 22).
- **Decision 5 (draft attention):** draft steps after the first reuse step 0's selection - the head's draft lists 2..k bind step 0's list buffer and carry no score / select launches.
- **Decision 11 (sampling defaults) is open.** Until it is ruled, `b70-serve` leaves `server::Options::sampling_defaults` unset for `qwen4_exp` (the Qwen family's greedy-unless-asked, `src/server/deps.h:45-52`); the ruling flips the one line Kolibri's dispatch has (`options.sampling_defaults = defaults`, `src/cli/b70_serve.cc:399`) with `generation_config.json`'s T 1.0 / top-p 0.95 / top-k 20, and `qwen4exp_server_test` pins whichever is ruled.
- **Decision 9 (context against cache):** `--max-len auto` here plans what fits beside the truncated model; the full-model default is spec 22's (its decision 5).
- **`reasoning_effort`:** the template reads it (spec 21 §1), but the server's `Kind::Qwen` forwards only `enable_thinking` (`ChatFormat::template_kwargs()` is false for Qwen) - exactly as for Qwen3.8 today. Forwarding `chat_template_kwargs` for the Qwen kind is a server-wide change for both models, not this plan's; recorded as a follow-up.
- **Spec 22:** the full model, the engine's acceptance and verify cost on it, A4 and passkey. This plan leaves ready: the verify list's per-layer union of experts read back (`read_verify_routes`), which is spec 22 P0.6's "MTP verify" term measured on the card.

## Global Constraints

- Branch `spec21e-qwen4exp-serving` from main; box tree automatic; `tools/box.env` copied if missing, never committed, never printed; `oracle-out*` symlinked.
- **K0 for the server path:** `template_test`, `toolcall_test`, `protocol_test`, `golden_server_test`, `prefix_server_test`, `mtp_server_test`, `lookup_server_test`, `pp_serve_test`, `kolibri_server_test`, `k2_server_test` unchanged; Qwen3.8's server behaviour unchanged.
- **F0:** every existing binary unchanged; new binaries are this family's M = 2..4 variants and the head's - additions only.
- One weight format; the forms 21b reads. Every number measured, or marked derived / estimated / proposed.
- Box: `flock ~/b70-gpu.lock` (once for both cards), detached, polled; interleaved pairs, median of 3, `uptime`; the operator's llama-benchy flags for any server row.
- Mac checks `tools/mac_check.sh --base main --kernels`. No `rm -rf`. Signed commits on the branch; no merge, no push.

## Review Focus

1. **A restore at any position continues bitwise.** The state at position p is everything the next token reads that is not a per-position block: 36 GDN recurrent states (fp32, 113.2 MB) and their conv rows (the last 3 positions), the PLE id history (positions p - 1, p - 2) and its 9 conv rows (p - 9 .. p - 1), and per QSA layer the open block's raw keys (positions `4 floor(p / 4) .. p - 1`, at most 3). The blocks are per position: the KV rows and the compressed keys of the complete blocks `[begin / 4, floor(end / 4))`. Restores at block ends (2048, 4096) and at request ends (2049, 2050, 2051 - the QSA cut -, 2052, 5000) continue bitwise as the cold run; positions before 0 in the PLE history read as EOS, as a cold run's do.
2. **Verify rows are decode rows.** At M = K + 1, row r's arithmetic is exactly M = 1's at its position (spec 8 M2): every kernel row-independent - the HC kernels, the PLE gather (each row's own history from the chunk of drafts and the ring), the indexer (8-slot tail ring, 21c's Review Focus 3), each row's own selection and top-10, the MoE's 11 slots per row. Rows past the accepted j are stale and never read: the KV, keys, tail-ring slots, PLE rings and GDN conv rows are position-indexed and overwritten before any later read; the GDN recurrent state takes row j's slot (`Control::gdn_live`, Qwen3.8's `gdn_step_slots`).
3. **The head is vLLM's head.** Inputs `R_i` (the main model's pre-mixer 4-stream `H` after the last layer's combine, materialised - the final mixer's `combine_norm` already writes it) and `t_{i+1}`; `fc_embedding(gemma_rms_2560(embed(t)))`; `fc_hidden` per stream after ONE RMS over 10240 (decision 4's default); `e` added to every stream (unit injection); one QSA layer with its own KV, indexer, keys and tail ring, its own 512 experts + shared; its own final mixer -> the shared `lm_head`; the next draft step reads the head's own pre-mixer `H`. M1 holds the engine's head to `qwen4exp_mtp.py` on the same `R` and token.
4. **The head's KV is filled for every position the main model has.** As spec 8 does for Qwen3.8 (`engine.h:222-224`: the head's KV rewritten at n-1..n+k-1 by verify; prefill's MTP KV fill), the head layer runs on (R_p, t_{p+1}) for every prompt row (21d's walk gains the head's pass per chunk when MTP is on) and for every verify row; its indexer and selection likewise. A draft never attends a head-KV row the head has not written.
5. **Snapshots and MTP across two cards.** Spec 16b's rule: the host layouts under `--pp 2` are `--pp 1`'s byte for byte (layers in layer order, whichever device holds them; the head's KV and keys last); the head lives on the last device; an entry saved under one placement restores under the other.

---

### Task 1: tokens and the template (host)

**Files:**
- Create: `tests/tokenizer/qwen4exp_tokenizer_test.cc`, `tools/tokenizer/dump_qwen4exp.py`, `tests/server/qwen4exp_server_test.cc`
- Modify: `tests/CMakeLists.txt` (a block `# ==== Spec 21e: Qwen3.8-Flash-Next served (label qwen4exp) ==== (begin)` after 21d's)
- Test: `qwen4exp_tokenizer_test`, `qwen4exp_server_test`; K0: `template_test`, `toolcall_test`

**Interfaces:**
- Consumes: 21a's facts (tokenizer.json identical to Qwen3.8's or the diff; the template's sha256), `tok::Tokenizer`, `chat::Template`, `server::ChatFormat`, `server::OutputStream`, `tests/server/mock.h`.

- [ ] **Step 1: the tests.** `qwen4exp_tokenizer_test <tokenizer.json>` (the snapshot's file through the CMake cache path `B70_Q4EXP_TOKENIZER_JSON` or the environment variable of that name; absent: disabled on the Mac, SKIP 77 elsewhere): if 21a found it byte-identical to Qwen3.8's, the test asserts the sha256 equality and `vocab_used` 248077; otherwise `dump_qwen4exp.py` writes a K2-style fixture (24 texts, the added tokens both ways, three chat renders' ids) and the test holds the server's encode / decode to it. The template: the checkpoint's `chat_template.jinja` (or `tokenizer_config.json`'s string) hashes to `c3cf9e34...`, so `chat::Template`'s existing fallback for that hash applies (`src/tokenizer/chat_template.cc:140-142`) and `template_test`'s Qwen3.8 cases ARE this model's renders - the test asserts the hash and renders two cases through the snapshot's own file. `qwen4exp_server_test` (mock engine and tokenizer): `ChatFormat::for_model_type("qwen4_exp")` is `Kind::Qwen`; a tool-call history and parallel calls parse back to `tool_calls`; reasoning with `enable_thinking` on and off; both EOS ids (248046, 248044) stop with `finish_reason: stop`; a request without sampling fields runs greedy (decision 11's interim, Dependencies).
- [ ] **Step 2: register, run, expect FAIL where unbuilt; implement; PASS** (`ctest --preset mac-host -R '^(qwen4exp_tokenizer|qwen4exp_server|template|toolcall)_test$'`).
- [ ] **Step 3: commit** `git commit -S -m "tokenizer, server: Qwen3.8-Flash-Next's tokens and template are Qwen3.8's - the hash, the renders, the Qwen parser (spec 21e)"`.

### Task 2: prefix-cache snapshots

**Files:**
- Create: `tests/runtime/qwen4exp_snapshot_test.cc` (host: the layouts), `tests/runtime/qwen4exp_snapshot_gpu_test.cc` (card)
- Modify: `src/runtime/qwen4exp/qwen4exp_sizes.{h,cc}` (the host layouts: `state_runs`, `kv_runs`, `state_snapshot_bytes`, `kv_snapshot_bytes`), `src/runtime/qwen4exp/qwen4exp_engine.{h,cc}` and `qwen4exp_prefill_engine.cc` (the snapshot calls), `tests/CMakeLists.txt`
- Test: `qwen4exp_snapshot_test` (host), `qwen4exp_snapshot_gpu_test` (card)

**Interfaces:**
- Consumes: 21c / 21d's engine (`prefill`, `ingest`, `generate`, `set_block_hook`, `kBlock`), `runtime::kolibri`'s `SnapRun` shape as the model to copy.
- Produces:

```cpp
namespace runtime::qwen4exp {
enum class SnapTensor : uint32_t { Kv, IdxKeys, IdxTail, GdnState, ConvRing, PleIds, PleRing, MtpKv, MtpIdxKeys, MtpIdxTail };
struct SnapRun { uint32_t device = 0; SnapTensor tensor; size_t offset = 0, bytes = 0; bool zero = false; };
// state  [GDN states, layer order][GDN conv rows of p-3..p-1][PLE ids p-1, p-2 (EOS before 0)][PLE conv rows p-9..p-1]
//        [per QSA layer: raw keys of 4 floor(p/4) .. p-1, padded to 3 rows][the head's tail when MTP is on]
// kv     [per QSA layer, layer order: K | V rows [begin, end)][per QSA layer: keys of blocks [begin/4, floor(end/4))][the head's, last]
size_t state_snapshot_bytes(const model::Qwen4ExpDesc& d, bool mtp);   // ~115.6 MB (derived: GDN states 113.2 + their conv rows 2.2 + PLE 0.18)
size_t kv_snapshot_bytes(const model::Qwen4ExpDesc& d, uint32_t begin, uint32_t end, bool mtp);
std::vector<SnapRun> state_runs(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t pos, bool mtp);
std::vector<SnapRun> kv_runs(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t max_len,
                             uint32_t begin, uint32_t end, bool mtp);
}
// Qwen4ExpEngine: size_t state_bytes() const; size_t kv_bytes(uint32_t n) const; runtime::KvCache kv_cache() const;  // Bf16
//                 void save_state(void*); void load_state(const void*, uint32_t pos);
//                 void save_kv(uint32_t begin, uint32_t end, void*); void load_kv(uint32_t begin, uint32_t end, const void*);
```

`kv_bytes(n)` is the block form the prefix cache asks for (`begin` a multiple of `kBlock`, so of 4; `end` any position); the snapshot calls run on the immediate lists with no step in flight (Kolibri's rule).

- [ ] **Step 1: the failing host test** `qwen4exp_snapshot_test`: the runs at pos 0, 3, 4, 2051, 5000 under `Q4Placement::one` and `two(d, 2)` at `--layers 4`: total bytes equal `state_snapshot_bytes` / `kv_snapshot_bytes`; the host order identical under both placements (Review Focus 5); the PLE ids of positions before 0 are written to the host as EOS 248044 by `save_state` and loaded back as such (a cold run's history is EOS, not zero), the conv rows before 0 as zeros (`zero` runs); the open block's raw keys at pos 2051 (3 rows: 2048, 2049, 2050) and at 2052 (0 rows).
- [ ] **Step 2: implement; PASS.** Then the card test `qwen4exp_snapshot_gpu_test <ckpt> <ids>` (synthetic 4 layers under `--pp 1` and `--pp 2 --pipeline-split 2`; Intel's `--layers 18` when present): a 5000-id prompt prefilled cold, logits / KV / keys / states and 32 greedy ids recorded; then Review Focus 1's restores (block ends from the hook's saves, request ends 2049 / 2050 / 2051 / 2052 / 5000 taken by `save_state` + `save_kv`) each followed by the tail prefill and 32 greedy ids: bitwise the cold run; a state saved under `--pp 2` restored under `--pp 1` and the reverse. Registered (`LABELS "checkpoint;qwen4exp"`, SKIP 77 without data).
- [ ] **Step 3: commit** `git commit -S -m "runtime: Qwen3.8-Flash-Next prefix-cache snapshots - GDN, PLE and indexer state, KV and compressed-key blocks, one layout on one card or two (spec 21e)"`.

### Task 3: the MTP head and draft / verify / commit

**Files:**
- Create: `src/kernels/qwen4exp/q4_mtp.cl` (the fusion: `gemma_rms` of the embedding, the single or per-stream RMS over the 4 streams, `+ e` on every stream), `tests/runtime/qwen4exp_mtp_test.cc`, `tests/kernels/qwen4exp_mtp_names_test.cc`, `tools/oracle/qwen4exp_mtp_fixture.py` (writes the M1 data: `R`, tokens, the head's logits and pre-mixer `H` from 21a's `MtpHead`, both norm forms)
- Modify: `src/kernels/qwen4exp_kernels.h` (a spec 21e block: the M = 2..4 names, the head's), `src/kernels/CMakeLists.txt` (a block `# ==== Spec 21e: Qwen3.8-Flash-Next MTP (B70_Q4EXP and B70_MTP) ==== (begin) / (end)`: every 21c kernel at M = 2..4, `gdn_step_slots_M{1..4}` and `attn_prep_M{2..4}_Q24KV2` / `_S1_Q24KV2` bound by name or added, `prep_gated_head_M{2..4}_SIG`, the GEMVs at M = 2..4, `gemv_bf16` at `{2560, 2560}` M = 1 and 4 for the two fc's, `q4_mtp`), `src/runtime/qwen4exp/qwen4exp_{buffers,capture,engine}.{h,cc}`, `src/runtime/qwen4exp/qwen4exp_prefill.cc` (the head's pass per chunk, Review Focus 4), `src/runtime/qwen4exp/qwen4exp_sizes.{h,cc}` (the head's KV / keys / scratch, the verify slots, the launches of each list), `tests/CMakeLists.txt`
- Test: `qwen4exp_mtp_names_test` (host), `qwen4exp_mtp_test` (card)

**Interfaces:**
- Consumes: 21b's head weights (`Q4DevicePart::mtp`, `mtp_fc`), 21c's kernels at M rows, Qwen3.8's `gdn_step_slots` (spec 8's state slots, `Control::gdn_live`), spec 8's draft vocabulary option (`loader::select_draft_vocab`, the compact head) unchanged.
- Produces (`Qwen4ExpEngine`, spec 8's contracts as `engine.h:210-259` states them, so `PipelineEngineAdapterT` drives it):

```cpp
static constexpr uint32_t kMaxDraft = 3;                    // verify at M <= 4
bool mtp() const;  uint32_t max_verify_k() const;
void draft(uint32_t k, const std::function<uint32_t(uint32_t i)>& pick = {});  // step 0 selects; steps 1.. reuse its list
const std::vector<uint32_t>& draft_ids() const;
void verify(uint32_t k);  const uint32_t* verify_ids() const;  void commit(uint32_t j, uint32_t next_token);
void set_draft_input(uint32_t i, uint32_t id);             // spec 19e's prompt lookup: drafts from outside
void read_logits_into(float* host, uint32_t rows);  void read_draft_logits_into(float* host, uint32_t i);
float* host_rows(size_t floats);                            // pinned rows for the sampled path
std::vector<uint32_t> read_verify_routes();                 // [layers][M][32]: the per-layer expert union (spec 22 P0.6's MTP term)
enum class MtpNorm { Single, PerStream };  MtpNorm mtp_norm() const;   // B70_Q4_MTP_NORM
```

- [ ] **Step 1: the failing host test** `qwen4exp_mtp_names_test`: every name the draft lists (k = 1..3), the verify lists (M = 1..4) and the prefill's head pass bind is a target of the 21e block (`${B70_Q4EXP_MTP_KERNELS}`); the launch counts per list asserted from `qwen4exp_sizes` (the head: fusion, its QSA layer, its MoE, its mixer, the head GEMV and argmax).
- [ ] **Step 2: implement** the head's buffers and lists, draft / verify / commit (GDN state slots as spec 8: the verify's `gdn_step_slots_M<k+1>` writes row r's state into slot `(live + r) % 4`, commit sets `gdn_live`), the head's pass in prefill, `set_draft_input`. The names test PASSES.
- [ ] **Step 3: the card test** `qwen4exp_mtp_test <ckpt> <oracle-out-q4exp-mtp> <ids>` (synthetic with `--mtp`; Intel's `--layers 18` when present):
  - **M1:** the head on the fixture's `R` and tokens, both norm forms: draft ids tie-aware equal to `qwen4exp_mtp.py`, logits cosine >= 0.9999 (PROPOSED), the head's pre-mixer `H` within the HC chain's bar;
  - **M2 (F5):** verify rows at M = 2, 3, 4 bitwise equal to M = 1 decode at the same positions - logits, routes, selections, KV, keys, GDN state of the accepted row (Review Focus 2);
  - **M3 (F5):** greedy generation of 64 ids with `--mtp 1`, 2, 3 bitwise equal to without (lossless), on the short and the 4k prompts (the selection active), one card and two;
  - the verify cost: step time at K = 1..3 against M = 1 and the per-layer union of experts over the verify rows (`read_verify_routes`), printed per layer - recorded, the number spec 22's miss term multiplies.
- [ ] **Step 4: acceptance from the reference** (box CPU; F5's numbers until spec 22): `qwen4exp_mtp.py accept` (21a) on Intel's checkpoint for the golden set, the 36 A4 scenarios and code, K = 1..3, `norm` single and per_stream -> `oracle-out-q4exp-mtp/accept_{single,per_stream}.json`; the table goes to "21e as built" as decision 4's evidence.
- [ ] **Step 5: Mac gate and commit** `git commit -S -m "mtp: the Qwen3.8-Flash-Next head (vLLM's form) - draft with step 0's selection, verify at M = K + 1, commit; M1-M3 (spec 21e)"`.

### Task 4: the server

**Files:**
- Create: `src/cli/qwen4exp_serve.h` (the dispatch's body: settle, load, engine, adapter, options), `tests/cli/qwen4exp_adapter_compile.cc` (instantiates `cli::pp::PipelineEngineAdapterT<runtime::qwen4exp::Qwen4ExpEngine>` - compiled by the Level Zero syntax check on the Mac, linked on the box)
- Modify: `src/cli/b70_serve.cc` (dispatch on `model_type` `qwen4_exp`; 21c's refusal lifted), `src/cli/CMakeLists.txt`, `tests/CMakeLists.txt` (21c's `cli_reject_qwen4exp_serve` removed; `cli_reject_serve_qwen4exp_*` added)
- Test: `cli_reject_serve_qwen4exp_*` (binary), `golden_server_test` pointed by hand (card)

- [ ] **Step 1:** `b70-serve <qwen4exp>`: `--layers N|auto` (required until spec 22: without it refused before the device naming the bytes and spec 22), `--ple-dir`, `--pp 1|2` / `--pipeline-split` / `--pipeline-handoff` and `--max-len N|auto` as `b70-decode` (21c), `--lm-head int8` default (spec 9, as every served model), the prefix cache on by the existing flags (`kv_form` 0), `--mtp K` (1..3) and `--spec mtp|lookup` through the adapter, EOS from `generation_config.json` ([248046, 248044], read for every model already), `ChatFormat::for_model_type` (Qwen), `Options::sampling_defaults` per decision 11's interim. Refused before the device, by name: `--mtp auto` (the `MtpCost` table is Qwen3.8's and the verify cost here is measured only in Task 3 Step 3 on the truncated model - auto waits for spec 22's full-model numbers), `--kv-cache int8` (decision 8), `--prefill-backend l0-int8 | sycl-tla`.
- [ ] **Step 2: the server end to end** on the synthetic checkpoint (box): `golden_server_test` pointed at it by hand (as 15e did for Ornith: it does not SKIP without a checkpoint, so it is not registered) - one greedy chat through the server equals `b70-decode --layers 4 --prefill` on the same rendered ids; with `--mtp 2` the same ids; with the prefix cache, the second identical request restores and returns the same ids.
- [ ] **Step 3: commit** `git commit -S -m "server: b70-serve runs Qwen3.8-Flash-Next --layers N on one card or two - PipelineEngineAdapterT over Qwen4ExpEngine, prefix cache, MTP (spec 21e)"`.

### Task 5: the tool-call set and its CPU reference (prepared for spec 22)

**Files:**
- Modify: `tools/toolcall/oracle_generate.py` (`MODELS` gains `"qwen4exp": "qwen4_exp"`, the reference `qwen4exp_ref.py`'s streamed model, 192 new ids), `tools/toolcall/a4_ref.sh` (`qwen4exp set | ref | status`: the set is Qwen3.8's `tests/golden/toolcall` re-rendered by the checkpoint's template - identical ids if 21a found the template and tokenizer identical, said in the log), `tools/toolcall/test_oracle_generate.py` (the model check), `tools/box_validate/data.sh` (`have oracle_q4exp_a4`)

- [ ] **Step 1:** the failing test (`--model qwen4exp` refuses a snapshot of another `model_type`; the tiny model runs one scenario); implement; PASS.
- [ ] **Step 2 (box CPU, opt-in, hours):** `tools/toolcall/a4_ref.sh qwen4exp set` then `ref` on Intel's checkpoint -> `oracle-out-q4exp-a4/` (36 `<name>.bf16.{ids,txt}`); `score.py` parses every reference call (Qwen XML; 0 parse failures - the hard bar on the reference). The engine's A4 run waits for spec 22 (the full model).
- [ ] **Step 3: commit** `git commit -S -m "toolcall: the Qwen3.8-Flash-Next A4 set and its CPU reference, ready for spec 22's full-model run (spec 21e)"`.

### Task 6: the queue row and the record

**Files:**
- Modify: `docs/superpowers/plans/box-validation-queue.md` (row 34), `tools/box_validate/stages.sh` (`row 34` block), the spec's "21e as built" section, `docs/BENCHMARKS.md` ("Qwen3.8-Flash-Next (spec 21)": the MTP rows on the truncated model), `README.md` (scope: the family built, served truncated, whole after spec 22), `docs/03-models.md`, `docs/13-loader.md`, `docs/19-running-models.md`

- [ ] **Step 1: the queue row** (34): the 21e block's binaries never compiled; stages `r34.host`, `r34.k1` (`kbins` + names), `r34.snapshot` (`qwen4exp_snapshot_gpu_test`), `r34.mtp` (`qwen4exp_mtp_test`: M1-M3, the verify cost), `r34.accept` (opt-in cpu: Task 3 Step 4), `r34.serve` (b70-serve on the synthetic: one greedy chat = `b70-decode`, `--mtp 2`, the prefix-cache repeat), `r34.a4_ref` (opt-in cpu: Task 5 Step 2); `rownote`s: A4 on the engine, passkey at the context spec 22 chooses, the llama-benchy rows and the full-model acceptance and speed are spec 22's. `python3 tools/box_validate/test_box_validate.py` passes.
- [ ] **Step 2: the record** - "21e as built" (every gate and number of 21c-21e on the truncated model, decision 4's acceptance table, the verify cost), BENCHMARKS, README, docs/03, docs/13, docs/19.
- [ ] **Step 3: commit** `git commit -S -m "spec 21: Qwen3.8-Flash-Next served truncated - snapshots, MTP, the record; the whole model waits for spec 22"`.

**Gate for the plan:** Mac - `qwen4exp_tokenizer_test`, `qwen4exp_server_test`, `qwen4exp_snapshot_test`, `qwen4exp_mtp_names_test` green, the adapter instantiation through the Level Zero syntax check, the K0 server tests unchanged, cmdlines additions only. Box - F4's restores bitwise (one card, two, across placements); F5's M1 / M2 / M3; the verify cost and the per-layer expert union recorded; the server's greedy chat equal to `b70-decode`; the reference's acceptance for both norm forms recorded (decision 4); the A4 reference ready for spec 22.
