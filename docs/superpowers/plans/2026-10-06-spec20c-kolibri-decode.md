# Spec 20c - Kolibri-1 decode (one card for development, two cards for the model)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kolibri-1 decodes through replayed Level Zero lists: its own model table, loader, kernels and engine (K2's pattern, spec 18 §5.1), first on one card (a real-width synthetic checkpoint, or the real checkpoint truncated to its first N layers), then across two cards with spec 16b's hand-off; KL2 and KL3 on the decode path; the decode speed rows.

**Architecture:** spec 20 §4. A model beside qwen3_5 and K2: `model::Kolibri1Desc` (`src/model/kolibri1.{h,cc}`), `loader::load_kolibri1()` (`src/loader/kolibri1_*`), `runtime::kolibri::KolibriEngine` (`src/runtime/kolibri/`), Kolibri-only kernels in `src/kernels/kolibri/` named `kol_*` (host half `src/kernels/kolibri_kernels.h`). Reused at Kolibri's shapes from new CMake lines only: `gemv.cl` (int4 attention), `gemv_bf16.cl` (router, bf16 attention, bf16 head), `gemv_i8w.cl` (int8 head, spec 9), `prep.cl`'s `prep_res_fold` (plain and `ZERO_RESID`), `embed_gather.cl`, `argmax.cl`. New: the 384-expert router, the MoE block with Kolibri's combine and a bf16 shared expert, the sandwich-norm kernels, the attention prep (per-head q/k norm, RoPE only in sliding layers), decode attention over a sliding ring and over the full layers' growing KV (flash, and the reference's eager chain behind a switch). Per layer 15 launches; one captured list per device; layers `[0, s)` on device 0, `[s, L)` plus the head on device 1.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake/ctest; Python 3 + torch in `agnes-ref-img` for the synthetic checkpoint, the fixtures and the golden sets.

**Spec:** `docs/superpowers/specs/2026-10-05-spec20-kolibri-1-design.md` (§1, §2, §4, §5 KL2-KL3, §6 20c, §7 decisions 2-4, §10 as built). Facts: `docs/probe-kolibri-2026-10-05.md`. Reference: `tools/oracle/kolibri_ref.py` (20a). Precedents to copy structure from: `src/model/k2_horizon.{h,cc}`, `src/loader/k2_{layout.h,repack.*,rope.cc,loader.*}`, `src/runtime/k2/`, `src/kernels/k2/k2_{prep,moe,attn,attn_eager}.cl`, `src/cli/k2_decode.h`, `tests/golden/k2_golden_test.cc`, `tools/box_validate/k2_oracle.sh`, `tools/mac/clrun/k2_run.cc`; spec 18 §10-§10.1 (as built, the eager switch).

## Dependencies and branch points

- **Spec 16b (two-card decode) is merged** (`0792415..dd2fc87`; spec 16 §8 "16b as built"). 16a's probe has NOT run, so 16b ships both hand-offs, `copy` (default) and `peer`. Kolibri runs its own engine (as K2, which `runtime::PipelineEngine` refuses), so it cannot be a `PipelineEngine` stage pair; it **reuses 16b's pieces** and adds only what its descriptor and its size force:

  | 16b piece (on main) | how Kolibri uses it |
  |---|---|
  | `l0::Context(const Context& primary, uint32_t device_index)`, `l0::Context::gpu_count()`, `l0::Context::can_access_peer(const Context&)` (`src/l0/context.h`) | unchanged: device 1 is a view in device 0's context; peer access checked before the load |
  | `l0::SyncEvent` (`src/l0/sync_event.h`), `l0::CmdList::barrier_signal` / `wait_event`, `l0::Fence::wait_for` | unchanged: the `copy` hand-off's cross-device event; every step's fence wait bounded |
  | `runtime::PpHandoff`, `kDefaultPpHandoff`, `parse_pp_handoff`, `kPpDevices`, `PipelineOptions` (`timeout_ms`, `spin_limit`) | unchanged |
  | `runtime::pp_balance(layer_bytes, dev0_fixed, dev1_fixed)` (`src/runtime/pipeline_plan.h`; descriptor-free, documented as the K2 / non-`ModelDesc` entry) | **the split by bytes**: Kolibri feeds it its per-layer bytes at the session's max_len (Task 5) - `--pipeline-split auto` is 16b's rule, not a second one |
  | `runtime::PpLandingLayout`, `kPpPage`, `kPpLandingAlign`, `kPpStateWords`, `runtime::PipelineLink`, `runtime::StageLink` (`src/runtime/capture.h`), `PipelineLink::binding(spin_limit)`, `zero()` | the hand-off buffers and their binding, through two **descriptor-free overloads added to 16b's files** (Task 6): `pp_landing_layout(size_t resid_bytes, size_t sumsq_bytes)` and `PipelineLink(l0::Context&, l0::Context&, const PpLandingLayout&, PpHandoff)` - the `ModelDesc` forms delegate to them, byte-for-byte as today |
  | `pp_handoff.cl` (binary `pp_handoff`: `pp_send` / `pp_recv`, word counts at run time) | unchanged under `--pipeline-handoff peer`: Kolibri's cut hands off `resid` (M x 2560 bf16) and the norm sums (20 x M fp32), the same two regions as Qwen's cut |
  | the token back and `pos` (spec 16 §8): after the fence the host copies device 1's Control block into device 0's; a host sampler writes both (`PipelineEngine::set_token`) | the same rule in `KolibriEngine` (`set_token`), and the same bounded-failure rule (a lost hand-off host-signals the event, throws naming it, marks the engine until `reset()`) |
  | `cli::PipelineArgs`, `parse_pipeline_split`, `parse_pipeline_handoff`, the device-count parser, `require_two_devices` (`src/cli/pipeline_args.h`) | the same flags and parsers; Kolibri's own refusals replace `check_pipeline`'s (which refuses prefill under PP for the Qwen engine; Kolibri prefills on two cards in 20d) |

  **New for Kolibri, and why:** placement at load (`load_kolibri1` puts each layer straight onto its device: 16b's load-then-place, `runtime::place_stages`, needs the whole model on device 0 first, ~42 GB here - the case `pipeline_place.h` leaves to "that model's spec"); its stage capture (its own kernels; `runtime::build_stage` walks a `ModelDesc`); its per-layer byte list for `pp_balance` (`pp_weights` / `pp_stages` / `require_split` are `ModelDesc`'s and carry the Qwen GDN / FA constraint).

  **The switches are 16b's, as renamed by the operator on 2026-10-06:** `--pp N` / `--pipeline-parallel-size N`, `--pipeline-split auto|N`, `--pipeline-handoff copy|peer`; the bench's prefill flags are `--prefill-length N`, `--prefill-chunk C`, `--prefill-backend B`. **Kolibri's default is `--pp 2`**: the real model needs two cards at int4 (~42.5 GB). `--pp 1` is accepted only when the planner says the model fits one card (the synthetic checkpoints, `--layers N`), otherwise refused before the device naming the bytes and `--pp 2`; with one GPU visible, `require_two_devices` refuses by name. Tasks 1-5 need no second card.
- **Spec 20b (the real int4 checkpoint, decision 1 open).** Every real-checkpoint gate SKIPs (77) until `urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ` and `oracle-out-kolibri/` exist. Task 1's synthetic real-width checkpoint makes KL2 / KL3 runnable before 20b.
- **Decision 2 (attention int4 vs bf16) is open: both arms are built.** The loader reads the arm from the checkpoint (layer 0's `self_attn.q_proj.qweight` present: int4; `.weight`: bf16; every attention projection of every layer must agree, else refused naming the tensor). Branch point: `model::KolAttnForm` - the int4 arm binds `gemv.cl` rows and `prep_res_fold ... _SP<oproj_s>_G20_Z`; the bf16 arm binds `gemv_bf16.cl` rows and `prep_res_fold_M1_K2560_SP1_G20_Z`. Both arms' binaries are built; both synthetic checkpoints are gated.
- **Decision 3 (context) is open:** `max_len` above the trained 262144 is refused naming spec 20 decision 3.
- **Decision 4 (shared experts) is open:** this plan serves them bf16 (§3.1's checkpoint form, the proposal). The int8 arm, if chosen, is a load-time quantisation with `gemv_i8w.cl`'s W8A16 arithmetic inside the MoE kernels' shared slot - not built here.

## Global Constraints

- Branch `spec20c-kolibri-decode` from main; box tree automatic; `tools/box.env` copied if missing, never committed, never printed; `oracle-out*` symlinked into the box tree (`tools/box_validate/data.sh`).
- **K0, nothing moves:** every existing kernel binary keeps its name and command line (`tools/kernel_cmdlines`: additions only; the box compares sha256); Qwen3.8, Agnes, Ornith and K2 suites unchanged. No existing `.cl` is edited: reused sources run at Kolibri's shapes from new CMake lines.
- **One weight format:** AutoRound W4A16 int4 g64 symmetric, `auto_round:auto_gptq` packing (`w = (q - 8) * scale`, `qzeros` 0x77777777), as `tools/quantize_kolibri1.sh` (AutoRound 0.17.0 @ 6afaecdb) writes it; bf16 for the router, `expert_bias`, shared experts, norms, embedding and `lm_head` (and attention in decision 2's bf16 arm). No asymmetric loader, no compressed-tensors kernel path, no new kernel family for another quant format.
- **Semantics are 20a's, pinned** (`docs/probe-kolibri-2026-10-05.md`): plain-`w` RMSNorm with x̂ rounded to bf16 before `× w`; sandwich norms; fp32 router logits never rounded to bf16; top-6 on `logit + expert_bias`, ties to the lower id; weights `sigmoid(logit)` in fp32, not renormalised; combine ascending id in fp32 (`fp32(w_e) × fp32(bf16 y_e)`), `+ fp32(bf16 shared)`, one rounding; ungated shared expert; RoPE (neox, theta 1e4, all 128 dims) only in sliding layers; NoPE full layers at 4, 9, ..., 49; window 513 keys including the query (`i - 513 < j <= i`); fp32 head logits; EOS {127906, 127901}; no BOS.
- Kolibri-only kernels in `src/kernels/kolibri/`, named `kol_*`, every shape define in the binary name; reused sources' binaries named by their existing helpers (`kernels::gemv_variant`, ...) or with a `kol_` prefix when the existing name carries another model's vocabulary.
- Every number in docs is measured, or marked derived / estimated / proposed.
- Box: every GPU command under `flock ~/b70-gpu.lock` (once for both cards in a two-card job), detached, polled; interleaved pairs, median of 3, `uptime` recorded; `-j44` builds.
- Mac checks: `tools/mac_check.sh --base main --kernels` (host tests, Level Zero syntax, kernel command lines, OpenCL syntax, Mac GPU runs of the portable kernels). Python tests run in `agnes-ref-img` with `--memory 28g`.
- No `rm -rf`. Signed commits on the branch; no merge, no push.

## Review Focus

1. **The sandwich order at the residual adds.** `x += post_attn_norm(attn)` and `x += post_ffn_norm(moe)`: the post norms normalise the sub-block's own bf16 output before the add, and the next pre-norm reads the sum. A kernel that folds the MoE output into the residual before `post_ffn_norm` (as `k2_moe_down` / `moe_down` do) is wrong. Task 4's host test drives a post-norm weight of 0 (the sub-block must vanish) and of 2 (doubled contribution), as KL0 did.
2. **The ring at its wrap and at the window edge.** Key `pos - 512` is visible, `pos - 513` is not; keys whose slots wrap past slot 4095 back to 0 are read in absolute order. Task 4's kernel tests run positions 0, 1, 511, 512, 513, 4095, 4096, 4097 and 9000 (a window wholly wrapped) against the host reference.
3. **Router padding and ties.** Experts 384-511 (the padded slots of `router_n() = 512`) can never be selected however negative the real selections are; exact ties at the cut go to the lower id; the bias moves selection but never the weight. Task 4's tests: all 384 logits `-1e30` with zero bias, two experts tied at rank 6, a bias that reorders the top-6 without changing any weight.
4. **NoPE really is position-free.** A full layer's attention output is invariant to the absolute position offset of the whole sequence (KL0's test): Task 4's attention-prep test checks a full layer's k equals its pre-norm-normed value bit for bit at positions 0 and 100000.
5. **The two-card cut is invisible.** Logits, route rows, the full layers' KV and the rings after a prompt and 32 greedy tokens are bitwise identical between `--pp 1` and `--pp 2 --pipeline-split 3` (synthetic) and between `--pipeline-split 25` and `20` on the real model, under both `--pipeline-handoff copy` and `peer` (Task 6). A stale hand-off buffer (spec 16 §2's silent-receive lesson) shows here.

---

### Task 1: the synthetic checkpoint, the fixtures and the golden sets (Python, no card)

**Files:**
- Create: `tools/quantize/kolibri/make_synth.py`, `tools/oracle/kolibri_fixture.py`, `tools/oracle/kolibri_chat_ids.py`, `tools/box_validate/kolibri_oracle.sh`, `tests/golden/prompts/de_prose.txt`, `tests/golden/prompts/de_chat.json`
- Modify: `tools/quantize/kolibri/test_kolibri_quant.py` (two tests), `tools/oracle/README.md` (rows), `tools/box_validate/data.sh` (`have oracle_kolibri`, `have oracle_kolibri_synth`)
- Test: `tools/quantize/kolibri/test_kolibri_quant.py::test_synth_pack_roundtrip`, `::test_synth_config_prefix`

**Interfaces:**
- Produces: a real-width checkpoint directory (`config.json` with `num_hidden_layers` N and `layer_types[:N]`, `quantization_config` as AutoRound 0.17.0 writes it, safetensors shards + index, the tokenizer files) that `kolibri_ref.py` and `loader::load_kolibri1` both read; `tests/kernels/kolibri_fixture.h` (generated, committed); the golden sets `oracle-out-kolibri-synth/{int4attn,bf16attn}/{ckpt/,prose.ids,de_prose.ids,<p>.golden.safetensors,<p>.log}` and `oracle-out-kolibri/{prose,code,de_prose,de_chat}.{ids,golden.safetensors,log}` in kolibri_ref.py's `run` layout (`resid.L*`, `mixer.L*`, `mlp.L*` for prompt rows, `route.moe.{ids,w,gap}.L*` for every row, `logits`, `tokens`, `nll`).

- [ ] **Step 1: the failing tests.** In `test_kolibri_quant.py`:

```python
def test_synth_pack_roundtrip():
    # make_synth's RTN packer against dequant.py's rule: w_hat = (q - 8) * scale
    import make_synth as S, torch
    g = torch.Generator().manual_seed(0)
    w = torch.randn(128, 256, generator=g).to(torch.bfloat16)          # [N, K]
    qw, sc, qz = S.pack_rtn_g64(w)                                     # I32 [K/8, N], F16 [K/64, N], I32
    assert qw.shape == (32, 128) and sc.shape == (4, 128) and (qz == 0x77777777).all()
    deq = S.dequant(qw, sc)                                            # [N, K] fp32, dequant.py's rule
    rel = (deq - w.float()).norm() / w.float().norm()
    assert rel < 0.15
    # nibble order: element k of a column sits in word k // 8, bits 4 (k % 8) - dequant.py's rule
    assert torch.equal(S.dequant(qw, sc)[:, 0], (((qw[0] & 0xF) - 8).float() * sc[0].float()))

def test_synth_config_prefix():
    import make_synth as S
    d = S.synth_config(layers=5, attn="bf16")
    assert d["num_hidden_layers"] == 5 and d["layer_types"][4] == "full_attention"
    assert d["layer_types"][:4] == ["sliding_attention"] * 4 and d["hidden_size"] == 2560
    assert d["quantization_config"]["packing_format"] == "auto_round:auto_gptq"
    assert all(d["quantization_config"]["extra_config"][f"model.layers.{i}.self_attn.q_proj"]["bits"] == 16
               for i in range(5))
```

- [ ] **Step 2: run, expect FAIL** (`ModuleNotFoundError: make_synth`):

```sh
docker run --rm --memory 8g -e OMP_NUM_THREADS=3 -v "$PWD":/ws -w /ws \
  agnes-ref-img:latest python3 tools/quantize/kolibri/test_kolibri_quant.py
```

- [ ] **Step 3: `make_synth.py`.** `make_synth.py <out> --layers N --attn int4|bf16 --tokenizer <dir> [--seed 0]`: the vendored `third_party/kolibri1/config.json` with `num_hidden_layers = N`, `layer_types = real[:N]` (N in 1..50; the default 5 holds both kinds), every other key unchanged; weights drawn as `make_tiny.py` draws them (norms `1 + 0.2 N(0,1)`, `expert_bias` `0.5 N(0,1)`, linears `N(0,1) / sqrt(K)`, the router `3 N(0,1) / sqrt(K)`, embedding `N(0,1)`, `lm_head` `N(0,1) / sqrt(2560)`) at the REAL widths; every routed expert linear (and attention in the int4 arm) RTN-packed by `pack_rtn_g64` (per 64-row group of K: `scale = max|w| / 8` as f16, `q = clamp(round(w / scale) + 8, 0, 15)`, nibbles low-first along K, `qweight` I32 [K/8, N], `scales` F16 [K/64, N], `qzeros` all 0x77777777) - the bytes AutoRound's `auto_round:auto_gptq` export has; `quantization_config` = `{quant_method: "auto-round", packing_format: "auto_round:auto_gptq", bits: 4, group_size: 64, sym: true, extra_config: {<shared experts>: {bits: 16}, <attention in the bf16 arm>: {bits: 16}}}`; `dequant(qweight, scales)` exposed for the test. One layer is generated, packed and written at a time (peak RAM ~8 GB). Then `tools/quantize/kolibri/check.py <out> --attn <arm>` (no `--source`: there is no bf16 original; names, dtypes, shapes, `qzeros`, g64 scales and `quantization_config` are what it checks, the packer's arithmetic is Step 1's test) must print `ACCEPTED`.
- [ ] **Step 4: run the tests, expect PASS**; then build one synth and check it (box CPU or Mac, ~15 min, ~4.3 GB):

```sh
docker run --rm --memory 28g -e OMP_NUM_THREADS=3 -v "$PWD":/ws -v "$HOME/b70-data":/data -w /ws agnes-ref-img:latest \
  sh -c 'python3 tools/quantize/kolibri/make_synth.py /data/synth-check --layers 1 --attn int4 --tokenizer tools/oracle/third_party/kolibri1 && \
         python3 tools/quantize/kolibri/check.py /data/synth-check --attn int4'
```
Expected: `ACCEPTED`.

- [ ] **Step 5: the kernel-chain fixture.** `tools/oracle/kolibri_fixture.py > tests/kernels/kolibri_fixture.h`: imports `kolibri_ref` and runs ITS functions (`rms_norm`, `route`, `moe`'s combine, `rope_cos_sin` + `rotate_half`, `visible`, `attention`) in bf16 mode on seeded inputs at the real widths, and writes the inputs and outputs as `uint16_t` / `float` / `uint32_t` arrays: (a) one hidden row through `rms_norm` with weights 1, 0, 2 and random; (b) a q head and a k head through `rms_norm` (q/k norm) then RoPE at positions 0, 1, 513, 100000; (c) `route` on 4 rows of 384 fp32 logits + bias, including a tie at the cut and a bias-reordered row; (d) the combine of 6 expert rows + a shared row; (e) eager `attention` for a sliding layer at query positions 5, 512, 513, 700 (window 513) and a full layer at 700, GQA 12. Header comment: the command and torch version. `tests/kernels/kolibri_ref.h` (Task 4) is graded against it.
- [ ] **Step 6: prompts and the oracle script.** `tests/golden/prompts/de_prose.txt`: an original German paragraph written for this repo (about 300 words, plain prose on the history of a fictional town's railway; no quoted third-party text). `tests/golden/prompts/de_chat.json`: `{"messages": [system: "Du bist ein hilfreicher Assistent.", user: "Wie ist das Wetter in Heidelberg?"], "tools": [get_weather(city: string)], "reasoning_effort": "high"}`; `tools/oracle/kolibri_chat_ids.py <snapshot> <json>` prints `apply_chat_template(messages, tools=..., add_generation_prompt=True, reasoning_effort=...)` ids. `tools/box_validate/kolibri_oracle.sh <data tree> synth|real` (k2_oracle.sh's shape: per prompt resumable into `oracle-out-kolibri*.partial/`, moved when complete, `MemAvailable` check `KOL_REF_MIN_GB` default 24 for synth / 48 for real, exit 77 when short): `synth` builds `int4attn/ckpt` and `bf16attn/ckpt` with `make_synth.py --layers 5` if absent, then `tokenize.py <ckpt> encode` (no `--bos`) for `prose`, `de_prose`, then `kolibri_ref.py run <ckpt> --prompt <p>.ids --out <p>.golden.safetensors --gen 32`; `real` does the same on `KOL_ORACLE_MODEL` (default `models--urakozz--Kolibri-1-W4A16-g64-AutoRound-GPTQ`) for `prose`, `code`, `de_prose` and `de_chat` (ids from `kolibri_chat_ids.py`), then prints every log's `MoE 6th/7th selection gap` line (the tie tolerance's evidence). It also writes `tests/golden/prompts/kolibri_bench.ids`'s source: the first 42 ids of `de_prose.ids` (Task 5 bakes them).
- [ ] **Step 7: README rows, data.sh `have` lines, commit.**

```sh
git add tools/quantize/kolibri/make_synth.py tools/quantize/kolibri/test_kolibri_quant.py tools/oracle/kolibri_fixture.py \
  tools/oracle/kolibri_chat_ids.py tools/box_validate/kolibri_oracle.sh tools/box_validate/data.sh tools/oracle/README.md \
  tests/golden/prompts/de_prose.txt tests/golden/prompts/de_chat.json tests/kernels/kolibri_fixture.h
git commit -S -m "oracle: Kolibri-1 synthetic real-width checkpoint, kernel-chain fixture, golden-set script (spec 20c)"
```

### Task 2: the model table

**Files:**
- Create: `src/model/kolibri1.h`, `src/model/kolibri1.cc`, `tests/model/kolibri1_test.cc`, `tests/model/kolibri1/config.json` (copy of `tools/oracle/third_party/kolibri1/config.json`), `tests/model/kolibri1/synth5_bf16attn.json` (Task 1's synth config)
- Modify: `src/model/CMakeLists.txt` (library `b70_kolibri_model`), `tests/CMakeLists.txt` (a new block `# ==== Spec 20c: Kolibri-1 decode (label kolibri) ==== (begin)` at the end)
- Test: `kolibri1_test`

**Interfaces:**
- Consumes: `model::GemvShape`, `model::WeightKind`, `model::Fuse` (`src/model/qwen35.h`), `common::json::Value`.
- Produces (used by every later task and by 20d / 20e):

```cpp
namespace model {
enum class KolAttnForm { Int4, Bf16 };            // spec 20 decision 2, the checkpoint's
enum class KolLinearId { Qkv, OProj, LmHead, kCount };
struct KolLinear { KolLinearId id; GemvShape shape; WeightKind kind; Fuse fuse; std::vector<std::string> parts; };
struct KolPlacement {                              // layers [0, split) on device 0, [split, layers) on device 1
  uint32_t devices = 1, split = 0;                 // devices == 1: split == layers
  uint32_t first(uint32_t dev) const;  uint32_t end(uint32_t dev) const;  uint32_t device_of(uint32_t layer) const;
};
struct Kolibri1Desc {
  std::string name, architecture, model_type;     // "kolibri-1", "Kolibri1ForCausalLM", "kolibri1"
  uint32_t layers = 0;                             // 50; a synthetic checkpoint: 1..50, layer_types a prefix
  uint32_t hidden = 0, q_heads = 0, kv_heads = 0, head_dim = 0;
  uint32_t experts = 0, top_k = 0, moe_inter = 0, shared_inter = 0;
  uint32_t vocab = 0, vocab_used = 0;              // 128000, 128000 (argmax over every row, as the reference)
  uint32_t window = 0, full_every = 0;             // 513, 5
  uint32_t trained_max_len = 0;                    // 262144
  std::vector<uint32_t> eos;                       // {127906, 127901}
  double rope_theta = 0; float rms_eps = 0;
  KolAttnForm attn = KolAttnForm::Int4;
  uint32_t qkv_s = 0, oproj_s = 0;                 // int4 arm split-K, PROVISIONAL (S2, S4) until Task 7's sweep
  bool is_sliding(uint32_t l) const { return l % full_every != full_every - 1; }
  uint32_t full_before(uint32_t l) const;          // full layers in [0, l)
  uint32_t sliding_before(uint32_t l) const;       // sliding layers in [0, l)
  uint32_t q_n() const { return q_heads * head_dim; }        // 6144
  uint32_t kv_n() const { return kv_heads * head_dim; }      // 512
  uint32_t qkv_n() const { return q_n() + 2 * kv_n(); }      // 7168: q | k | v
  uint32_t gqa() const { return q_heads / kv_heads; }        // 12
  uint32_t router_n() const;                       // 512: next power of two >= experts; rows >= experts zero
  static constexpr uint32_t kRouteLanes = 256;     // the route work-group (the Mac's cap too)
  uint32_t route_epl() const { return router_n() / kRouteLanes; }   // 2 experts per lane
  static constexpr uint32_t kRing = 4096;          // sliding-ring slots: pow2 >= kPfC (2048) + window - 1
  KolLinear linear(KolLinearId id) const;
  std::vector<KolLinearId> layer_linears() const;  // {Qkv, OProj}
  static std::string layer_prefix(uint32_t l);     // "model.layers.<l>."
  size_t rope_table_bytes(uint32_t max_len) const { return size_t(max_len) * 2 * (head_dim / 2) * 4; }
};
const Kolibri1Desc& kolibri1();                    // the published model, attn Int4
// config.json -> the descriptor: layers and layer_types read (prefix of the real pattern), the
// attention form given by the loader (tensor names decide it), everything else held to kolibri1().
Kolibri1Desc kolibri1_desc(const common::json::Value& config, KolAttnForm attn);
void check_kolibri1_config(const Kolibri1Desc& d, const common::json::Value& config);   // throws naming the key
bool is_kolibri1_model_type(const std::string& model_type);                               // "kolibri1"
}
```

- [ ] **Step 1: the failing test** `tests/model/kolibri1_test.cc` (argv[1] = the real config, argv[2] = the synth one), using `tests/check.h`'s `CHECK`:
  - `kolibri1()`: hidden 2560, 48 / 4 heads x 128, gqa 12, qkv_n 7168, experts 384, top_k 6, moe_inter 512, shared_inter 512, vocab 128000, window 513, eos {127906, 127901}, rope_theta 1e4, rms_eps 1e-6, router_n 512, route_epl 2, trained_max_len 262144;
  - `is_sliding`: false exactly at 4, 9, ..., 49; `full_before(50) == 10`, `sliding_before(50) == 40`;
  - `check_kolibri1_config` passes the real config; refuses (message names the key) each of: `norm_topk_prob: true`, `num_experts: 256`, `sliding_window: 512`, `head_dim: 64`, `tie_word_embeddings: true`, `layer_types` with a full layer at 3, `rope_parameters.rope_type: "yarn"`, `hidden_act: "gelu"`, `num_hidden_layers: 51`;
  - `kolibri1_desc(synth5, Bf16)`: layers 5, attn Bf16; `kolibri1_desc(synth5 with layer_types[4] = "sliding_attention")` throws naming `layer_types`;
  - `KolPlacement{2, 25}`: `device_of(24) == 0`, `device_of(25) == 1`, `first(1) == 25`, `end(1) == 50`; `KolPlacement{1, 50}.end(0) == 50`; a split of 0 or 50 with 2 devices throws.
- [ ] **Step 2: register and run, expect FAIL (link error):**

```cmake
add_executable(kolibri1_test model/kolibri1_test.cc)
target_include_directories(kolibri1_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kolibri1_test PRIVATE b70_kolibri_model)
add_test(NAME kolibri1_test COMMAND kolibri1_test ${CMAKE_SOURCE_DIR}/tests/model/kolibri1/config.json
  ${CMAKE_SOURCE_DIR}/tests/model/kolibri1/synth5_bf16attn.json)
```

```sh
cmake --preset mac-host && cmake --build --preset mac-host && ctest --preset mac-host -R '^kolibri1_test$'
```

- [ ] **Step 3: implement** `kolibri1.cc` (numbers from config.json and the facts sheet; `check_kolibri1_config` mirrors `check_k2_config`'s style; `KolPlacement` checks in a free function `validate(const KolPlacement&, const Kolibri1Desc&)` called by its users).
- [ ] **Step 4: run, expect PASS.**
- [ ] **Step 5: commit** `git commit -S -m "model: Kolibri-1 table - sliding / NoPE pattern, 384-expert router on 512 slots, placement (spec 20c)"`.

### Task 3: the loader (host half on the Mac, the upload on the card)

**Files:**
- Create: `src/loader/kolibri1_layout.h`, `src/loader/kolibri1_repack.{h,cc}`, `src/loader/kolibri1_rope.cc`, `src/loader/kolibri1_loader.{h,cc}`, `tests/loader/kolibri1_repack_test.cc`, `tests/loader/kolibri1_rope_test.cc`, `tests/loader/kolibri1_load_checkpoint_test.cc`
- Modify: `src/loader/CMakeLists.txt` (library `b70_kolibri_loader`), `tests/CMakeLists.txt` (the 20c block)
- Test: `kolibri1_repack_test`, `kolibri1_rope_test` (host); `kolibri1_load_checkpoint_test` (card, label `checkpoint;kolibri`)

**Interfaces:**
- Consumes: Task 2's descriptor; `loader::resolve_snapshot`, `QuantConfig::parse`, `SafetensorsSet`, `assert_quant_invariants`, `LinearSrc::classify`, `DeviceWeight` (`src/loader/loader.h`), `LmHeadForm` + the int8 quantiser (`src/loader/lm_head_int8.h`), `common::repack_int4_layout0_cols`, `repack_int4_layout1_cols`, `cols_interleave16`, `repack_bf16_tiled` (`src/common/repack.h`).
- Produces:

```cpp
namespace loader {
// kolibri1_layout.h - device-free sizes (the planner's and the loader's one formula)
struct KolLayerBytes { size_t qkv, oproj, router, bias, norms, gate_up, down, shared_gate_up, shared_down; size_t total() const; };
KolLayerBytes kol_layer_bytes(const model::Kolibri1Desc& d);
size_t kol_embed_bytes(const model::Kolibri1Desc& d);                 // bf16 [vocab][hidden]
size_t kol_lm_head_bytes(const model::Kolibri1Desc& d, bool int8);   // bf16 tiled, or int8 + fp32 row scales
// kolibri1_repack.h - host only (a synthetic checkpoint in the test)
KolAttnForm kol_attn_form(const SafetensorsSet& st, const model::Kolibri1Desc& d);  // all-or-nothing, throws naming the tensor
std::vector<std::string> kol_expected_names(const model::Kolibri1Desc& d, model::KolAttnForm a);   // = kolibri_ref.expected_names with int4 suffixes
std::vector<float> kol_rope_table(const model::Kolibri1Desc& d, uint32_t max_len);   // fp32 [max_len][2][64], bf16-valued
// kolibri1_loader.h
struct KolLayer {
  DeviceWeight qkv, oproj;                       // int4 layout 0 (S cells from the desc) or bf16 tiled
  std::unique_ptr<l0::Mem> norms;                // fp32 plain w: input | post_attn | post_attention | post_ffn [hidden], q_norm | k_norm [head_dim]
  std::unique_ptr<DeviceWeight> router;          // bf16 tiled {hidden, 512}, rows >= 384 zero
  std::unique_ptr<l0::Mem> bias;                 // fp32 [512], expert_bias widened, 0 past 384
  std::unique_ptr<l0::Mem> gate_up, down;        // int4 layout-1 blocks [384][hidden x 2I] (gate||up interleave16), [384][I x hidden]
  std::unique_ptr<l0::Mem> shared_gate_up, shared_down;   // bf16 tiled {hidden, 2I} interleave16, {I, hidden}
};
struct KolDevicePart {
  uint32_t device = 0, first = 0, end = 0;       // layers [first, end) live here
  std::vector<KolLayer> layers;
  std::unique_ptr<l0::Mem> embed;                // device 0 only
  std::unique_ptr<l0::Mem> final_norm;           // last device only
  std::unique_ptr<DeviceWeight> lm_head;         // last device only
  std::unique_ptr<l0::Mem> rope;                 // any device holding a sliding layer
  size_t bytes = 0;                              // == the planner's figure for this device
};
struct KolLoadedModel {
  model::Kolibri1Desc desc; model::KolPlacement placement;
  std::vector<KolDevicePart> parts;              // one per device
  uint32_t max_len = 0, trained_max_len = 0;
  size_t unconsumed = 0, read_per_token = 0;     // read_per_token: derived, both cards
};
bool is_kolibri1_checkpoint(const std::string& snapshot_dir);
// devices.size() == placement.devices; layers_limit > 0 loads only layers [0, layers_limit) of the
// checkpoint (development mode, spec 20 §4) and makes desc.layers = layers_limit.
KolLoadedModel load_kolibri1(const std::vector<l0::Context*>& devices, const std::string& snapshot_or_repo,
                             uint32_t max_len, const model::KolPlacement& placement,
                             LmHeadForm lm_head = LmHeadForm::Checkpoint, uint32_t layers_limit = 0);
}
```

- [ ] **Step 1: failing host tests.** `kolibri1_repack_test` builds a synthetic 2-layer checkpoint in memory at the real widths of one expert group but only 8 experts (the descriptor field `experts` set to 8 in a test copy; widths unchanged) in both attention arms, through `SafetensorsSet` on a temp dir, and checks: `kol_attn_form` returns Int4 / Bf16, and throws naming `model.layers.1.self_attn.o_proj.weight` when layer 1 mixes; every expert block's dequantised words equal `(q - 8) * scale` of the source, block e at offset `e * kol_layer_bytes(d).gate_up / experts` (Review Focus 2 of plan 15c: test expert slices individually); gate||up columns interleaved in 16s (gate tile t, then up tile t); the shared expert's bf16 tiles equal `repack_bf16_tiled` of the concatenated interleave; router rows 8..15 (of the test's 16) zero; bias widened exactly and zero past the experts; norms widened verbatim (plain `w`, no `+1`); `kol_expected_names` equals the file's names both ways (a missing and an extra name each refused by name). `kolibri1_rope_test`: `kol_rope_table` at positions 0, 1, 513, 262143 equals the fixture's (Task 1 (b)) cos/sin values bit for bit, and the table is `max_len * 512` bytes.
- [ ] **Step 2: register, run, expect FAIL.** In the 20c block:

```cmake
foreach(t kolibri1_repack_test kolibri1_rope_test)
  add_executable(${t} loader/${t}.cc)
  target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(${t} PRIVATE b70_kolibri_loader)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
```
`ctest --preset mac-host -R '^kolibri1_(repack|rope)_test$'`.
- [ ] **Step 3: implement** the layout, repack, rope (`k2_rope_table`'s fp32 steps with theta 1e4) and the loader. `load_kolibri1`: config held (`check_kolibri1_config`), `max_len <= trained_max_len` (else refused naming spec 20 decision 3), quant invariants, names both ways, zero unconsumed (layers past `layers_limit` count as consumed-by-design and are listed in one note line), each part's `bytes` equal to `kol_layer_bytes` / embed / head formulas; one report line per device. Derived per-layer bytes (the test asserts the formula): int4 attention 830,794,752 B (experts 802,160,640; shared 7,864,320; attention 18,104,320; router 2,621,440; bias and norms 44,032), bf16 attention 880,847,872 B; embedding 655,360,000; head bf16 655,360,000 / int8 328,192,000. Real model, int4 attention, int8 head, split 25: device 0 21,425,228,800 B, device 1 21,098,071,040 B (derived).
- [ ] **Step 4: run host tests, expect PASS.** Then the card test `kolibri1_load_checkpoint_test <snapshot> [int8] [layers]` (one device, `layers` default all): loads, prints the report, asserts 0 unconsumed and `bytes` per part. Registered:

```cmake
set(B70_KOLIBRI_SNAPSHOT "urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ" CACHE STRING "Kolibri-1 int4 checkpoint (spec 20b)")
set(B70_KOLIBRI_SYNTH_DIR "${CMAKE_SOURCE_DIR}/oracle-out-kolibri-synth" CACHE PATH "Task 1's synthetic checkpoints and golden sets")
add_executable(kolibri1_load_checkpoint_test loader/kolibri1_load_checkpoint_test.cc)
target_include_directories(kolibri1_load_checkpoint_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kolibri1_load_checkpoint_test PRIVATE b70_kolibri_loader b70_l0)
add_test(NAME kolibri1_load_synth_int4attn_test COMMAND kolibri1_load_checkpoint_test ${B70_KOLIBRI_SYNTH_DIR}/int4attn/ckpt)
add_test(NAME kolibri1_load_synth_bf16attn_test COMMAND kolibri1_load_checkpoint_test ${B70_KOLIBRI_SYNTH_DIR}/bf16attn/ckpt int8)
add_test(NAME kolibri1_load_partial_test COMMAND kolibri1_load_checkpoint_test ${B70_KOLIBRI_SNAPSHOT} bf16 30)
set_tests_properties(kolibri1_load_synth_int4attn_test kolibri1_load_synth_bf16attn_test kolibri1_load_partial_test
  PROPERTIES LABELS "checkpoint;kolibri" SKIP_RETURN_CODE 77 TIMEOUT 1800)
```
(SKIP 77 when the directory or snapshot is absent.)
- [ ] **Step 5: commit** `git commit -S -m "loader: Kolibri-1 - expert blocks, bf16 shared expert, padded router, both attention arms, per-device parts (spec 20c)"`.

### Task 4: the kernels and their host reference

**Files:**
- Create: `src/kernels/kolibri/kol_prep.cl`, `src/kernels/kolibri/kol_moe.cl`, `src/kernels/kolibri/kol_attn.cl`, `src/kernels/kolibri/kol_attn_eager.cl`, `src/kernels/kolibri_kernels.h`, `tests/kernels/kolibri_ref.h`, `tests/kernels/kolibri_ref_test.cc`, `tests/kernels/kolibri_variant_names_test.cc`, `tests/kernels/kolibri_kernels_test.cc`, `tools/mac/clrun/kolibri_run.cc`
- Modify: `src/kernels/CMakeLists.txt` (a block `# ==== Spec 20c: Kolibri-1 decode ==== (begin) / (end)` after spec 18e's, option `B70_KOLIBRI` ON), `tools/mac_check.sh` (section 5 runs `kolibri_run`), `tools/mac/opencl/intel_shim.h` only if a new block-read shape is used (say so in the commit), `tests/CMakeLists.txt`
- Test: `kolibri_ref_test`, `kolibri_variant_names_test` (host); `kolibri_kernels_test` (card); `kolibri_run` (Mac GPU, indicative)

**Interfaces:**
- Consumes: Task 2's descriptor; Task 1's `tests/kernels/kolibri_fixture.h`.
- Produces (`src/kernels/kolibri_kernels.h`, namespace `kernels::kolibri`):

```cpp
inline constexpr unsigned kNormG = 20, kNormW = 20, kNormWg = 256;   // prep_res_fold FOLD_G / stage-B grids at K 2560
inline constexpr unsigned kUpKs = 4, kDnKs = 2;                      // kol_moe_gate_up / kol_moe_down K splits
inline constexpr unsigned kAttnTgt = 32, kAttnWg = 128, kAttnPart = 130;
namespace route {                                    // the route row, u32 words per (layer, token)
inline constexpr unsigned kWords = 32, kIds = 0 /* 6 ids ascending */, kWeights = 8 /* fp32 sigmoid(logit) */,
                          kSel = 16 /* fp32 logit + bias */, kNext = 24 /* sel of rank 6 */, kLogitMin = 25 /* diagnostics */;
}
std::string norm_variant(unsigned M);                                // "kol_norm_M1_K2560_G20_W20"
std::string post_add_variant(unsigned M);                            // "kol_post_add_M1_K2560_G20"
std::string attn_prep_variant(unsigned M, unsigned S, bool sliding); // "kol_attn_prep_M1_N7168_S2_Q48KV4_R4096" | "_F"
std::string route_variant(unsigned M);                               // "kol_route_M1_E384_T6_N512_L256"
std::string moe_variant(unsigned M);                                 // "kol_moe_M1_E384_T6_D2560_I512_SH"
std::string attn_variant(unsigned M, bool sliding);                  // "kol_attn_M1_T32_Q48KV4_W513_R4096" | "_F"
std::string attn_eager_variant(unsigned M, bool sliding);            // "kol_attn_eager_M1_T32_Q48KV4_W513_R4096" | "_F"
std::string fold_zero_variant(unsigned M, unsigned S);               // prep_res_fold_variant(M, 2560, S, 20) + "_Z"
std::vector<std::string> decode_variants(model::KolAttnForm a, bool int8_head, bool eager);   // what a list binds
```

The kernels (every chain is the reference's, `kolibri_ref.py` op for op; `tests/kernels/kolibri_ref.h` repeats each one; edit the two together):

```text
kol_prep.cl  (no sub-group functions: runs on the Mac)
  kol_norm_finish(sumsq, resid, w, x)              grid (kNormW, M), WG 256
      rstd = 1 / sqrt(Σ_{g ascending} sumsq[g][m] / 2560 + 1e-6)
      x[m][k] = rne(f32(rne(f32(resid[m][k]) · rstd)) · w[k])          two roundings (x̂ bf16, then × w)
  kol_post_add(sumsq_a, a, w, resid, sumsq_out)    grid (kNormG, M), WG 256: work-group g owns chunk g (128 columns)
      n = rne(f32(rne(f32(a[m][k]) · rstd_a)) · w[k]);  r = rne(f32(resid[m][k]) + f32(n));  resid[m][k] = r
      sumsq_out[g][m] = prep_res_fold's per-chunk tree of f32(r)²       (same chunking, same tree)
  kol_attn_prep(ctrl, partials, qkn, rope, attn_q, kv_k, kv_v)   grid (48 + 4 + 4, M), WG 128; defines QKV_S, SLIDING, RING
      x_b = rne(Σ_{s<QKV_S} partials[s][m][col])                      the linear's bf16 output, per head of 128
      q, k heads: y = rne(f32(rne(f32(x_b) · rstd_h)) · w_qk[d]),  rstd_h over the head's 128 (tree)
      SLIDING: y = rne(f32(rne(y · cos)) + f32(rne(rot(y) · sin)))  rot(y)[d] = d < 64 ? -y[d + 64] : y[d - 64]
      q -> attn_q[m][h][d] (fp32, a bf16 value);  k, v (v = x_b) -> row (SLIDING ? (pos + m) & (RING - 1) : pos + m)
kol_moe.cl  (sub-group block reads under cl_intel_subgroups, plain loads otherwise: runs on the Mac)
  kol_route(logits, bias, route)                   grid (1, M), WG 256, lane l owns experts l and l + 256
      sel_e = logits[m][e] + bias[e] (e < 384), -INF for 384 <= e < 512; rank = #{j : sel_j > sel_e or (== and j < e)}
      top 6 by rank -> ids ascending; w = 1 / (1 + exp(-logits[m][e])) fp32 (never rounded); kSel, kNext
  kol_moe_gate_up(route, x, w_gu, w_sh_gu, h)      grid (7 x 2I / 64, M), WG 64 x kUpKs
      slots 0..5: int4 layout-1 block ids[j];  slot 6: the shared expert's bf16 tiles
      h[m][slot][i] = rne(f32(rne(silu(f32(rne(Σ gate))))) · f32(rne(Σ up)))
  kol_moe_down(route, h, w_dn, w_sh_dn, mo)        grid (2560 / 16, M), WG 16 x 7 x kDnKs;  #pragma OPENCL FP_CONTRACT OFF
      acc = 0.f;  for j = 0..5 (ascending id): acc = acc + f32(rne(Σ down_j)) · w_j     (mul rounded, then add)
      mo[m][n] = rne(acc + f32(rne(Σ shared down)))                     NOT folded into resid (post_ffn_norm first)
kol_attn.cl  (spec 10 v2 = k2_attn.cl's structure; sub-groups and 2D block reads: card only)
  kol_attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part)   grid (4, TGT), WG 128;  kol_attn_reduce(ctrl, attn_part, attn_out)  grid (48, M)
      keys [lo, hi], hi = pos + m, lo = SLIDING ? max(0, hi - 512) : 0; key p at row SLIDING ? p & (RING - 1) : p
      SLIDING: blocks are absolute 64-key blocks [64 b, 64 b + 64) (keys < lo masked), so no 2D read crosses the ring's end
      attn_out[m][h·128 + d] = rne(acc / sm)                             no gate
kol_attn_eager.cl  (k2_attn_eager.cl's four launches without the gate; plain OpenCL C: runs on the Mac)
      s = rne(f32(rne(q·k)) · 128^-0.5); p = rne(e · (1 / Σ e)), e = exp_torch(s - max), Σ in torch's 8-lane order over the
      row's keys indexed from lo (the cached decode pass's index); o = rne(Σ p·v)
```

Reused at Kolibri's shapes (new CMake lines; names from the existing helpers):

```cmake
option(B70_KOLIBRI "build the Kolibri-1 decode variants (spec 20c)" ON)
if(B70_KOLIBRI)
  set(KOL_CL ${CMAKE_CURRENT_SOURCE_DIR}/kolibri)
  add_gemv_variant(1 2560 7168 2 0)            # q||k||v int4        PROVISIONAL S2 L0
  # o_proj int4 PROVISIONAL S4 L0 is gemv_M1_K6144_N2560_S4_L0, the binary K2's dense down already
  # builds (spec 18b block); prep_res_fold_M1_K2560_SP0_G20 likewise. Built here only without K2, so
  # no target is defined twice:
  if(NOT B70_K2)
    add_gemv_variant(1 6144 2560 4 0)
    add_prep_res_fold(1 2560 0 20)
  endif()
  add_gemv_bf16_variant(1 2560 7168 64 1)      # q||k||v bf16 (decision 2's bf16 arm)
  add_gemv_bf16_variant(1 6144 2560 64 1)      # o_proj bf16
  add_gemv_bf16_variant(1 2560 512 16 16)      # the router, rows 384..511 zero; fp32 logits
  add_gemv_bf16_variant(1 2560 128000 64 1)    # lm_head bf16 (fp32 accumulation and logits = head_dtype float32)
  add_gemv_i8w_variant(1 2560 128000)          # lm_head int8 (spec 9)
  # prep_res_fold_M1_K2560_SP0_G20 is bound after embed, after every MoE block and after the hand-off
  foreach(SP 1 4)
    add_ocloc_kernel(prep_res_fold_M1_K2560_SP${SP}_G20_Z SOURCE ${PREP_CL}
                     DEFINES M=1 K=2560 S_PREV=${SP} FOLD_G=20 ZERO_RESID=1)
  endforeach()
  add_ocloc_kernel(kol_embed_gather_M1_D2560_V128000 SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/embed_gather.cl
                   DEFINES ${CTRL_DEFINES} M=1 HIDDEN=2560 VOCAB=128000)
  set(KOL_ARGMAX_DEFINES ${CTRL_DEFINES} VOCAB=128000 VOCAB_USED=128000)
  add_ocloc_kernel(kol_argmax_stage1_M1_N128000_V128000 SOURCE ${ARGMAX_CL} DEFINES ${KOL_ARGMAX_DEFINES} M=1)
  add_ocloc_kernel(kol_argmax_stage2_N128000 SOURCE ${ARGMAX_CL} DEFINES ${KOL_ARGMAX_DEFINES})
  add_ocloc_kernel(kol_norm_M1_K2560_G20_W20 SOURCE ${KOL_CL}/kol_prep.cl DEFINES M=1 K=2560 NORM_G=20 NORM_WGS=20)
  add_ocloc_kernel(kol_post_add_M1_K2560_G20 SOURCE ${KOL_CL}/kol_prep.cl DEFINES M=1 K=2560 POST_G=20)
  set(KOL_HEADS Q_HEADS=48 KV_HEADS=4 HD=128)
  foreach(S 1 2)          # S 2: the int4 arm's q||k||v split-K; S 1: the bf16 arm (gemv_bf16 writes [M][N])
    add_ocloc_kernel(kol_attn_prep_M1_N7168_S${S}_Q48KV4_R4096 SOURCE ${KOL_CL}/kol_prep.cl
                     DEFINES ${CTRL_DEFINES} M=1 QKV_N=7168 QKV_S=${S} ${KOL_HEADS} SLIDING=1 RING=4096)
    add_ocloc_kernel(kol_attn_prep_M1_N7168_S${S}_Q48KV4_F SOURCE ${KOL_CL}/kol_prep.cl
                     DEFINES ${CTRL_DEFINES} M=1 QKV_N=7168 QKV_S=${S} ${KOL_HEADS} SLIDING=0 RING=0)
  endforeach()
  add_ocloc_kernel(kol_route_M1_E384_T6_N512_L256 SOURCE ${KOL_CL}/kol_moe.cl
                   DEFINES M=1 ROUTE_E=384 ROUTE_K=6 ROUTE_WG=256 ROUTE_EPL=2 ROUTE_LN=512)
  add_ocloc_kernel(kol_moe_M1_E384_T6_D2560_I512_SH SOURCE ${KOL_CL}/kol_moe.cl
                   DEFINES M=1 MOE_E=384 MOE_K=6 HIDDEN=2560 INTER=512 UP_KS=4 DN_KS=2)
  add_ocloc_kernel(kol_attn_M1_T${ATTN_V2_TGT}_Q48KV4_W513_R4096 SOURCE ${KOL_CL}/kol_attn.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} Q_HEADS=48 KV_HEADS=4 WINDOW=513 RING=4096)
  add_ocloc_kernel(kol_attn_M1_T${ATTN_V2_TGT}_Q48KV4_F SOURCE ${KOL_CL}/kol_attn.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} Q_HEADS=48 KV_HEADS=4 WINDOW=0 RING=0)
  add_ocloc_kernel(kol_attn_eager_M1_T${ATTN_V2_TGT}_Q48KV4_W513_R4096 SOURCE ${KOL_CL}/kol_attn_eager.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} Q_HEADS=48 KV_HEADS=4 WINDOW=513 RING=4096)
  add_ocloc_kernel(kol_attn_eager_M1_T${ATTN_V2_TGT}_Q48KV4_F SOURCE ${KOL_CL}/kol_attn_eager.cl
                   DEFINES ${CTRL_DEFINES} M=1 TGT=${ATTN_V2_TGT} Q_HEADS=48 KV_HEADS=4 WINDOW=0 RING=0)
endif()
```
(23 new binaries with K2 on, 25 without: 1 (2) gemv, 4 gemv_bf16, 1 gemv_i8w, 2 (3) folds, embed, 2 argmax, norm, post_add, 4 attention preps, route, MoE, 4 attention. `${ATTN_V2_TGT}` is 32, as `kAttnTgt`.)

- [ ] **Step 1: the failing host test** `tests/kernels/kolibri_ref_test.cc`: `kolibri_ref.h`'s chains against `kolibri_fixture.h` bit for bit - (a) `norm_finish` for w = 1, 0, 2, random (a zero post-norm weight makes `post_add` leave `resid` unchanged; 2 doubles the normalised contribution - Review Focus 1); (b) q/k norm + RoPE at 0, 1, 513, 100000, and for a full layer the k equals the normed head bit for bit at positions 0 and 100000 (Review Focus 4); (c) `route` - ids, fp32 weights within 2 ulp of torch's `sigmoid` (OpenCL `exp` vs torch's), the tie at the cut to the lower id, the bias row, and the all-`-1e30` row never selecting 384..511 (Review Focus 3); (d) the combine (an fma in place of mul + add is caught by a crafted pair whose fused result rounds differently); (e) eager attention bitwise at 5, 512, 513, 700 (sliding) and 700 (full), and flash's fp64 form within the cosine 0.99999 bar of spec 6 K1. Plus `ring_rows(lo, hi)` (the host's key -> row map) over positions 0, 1, 511, 512, 513, 4095, 4096, 4097, 9000 (Review Focus 2).
- [ ] **Step 2: register, run, expect FAIL:** `add_test(NAME kolibri_ref_test COMMAND kolibri_ref_test)` (host, links nothing but headers); `ctest --preset mac-host -R '^kolibri_ref_test$'`.
- [ ] **Step 3: write `kolibri_ref.h`** until the test passes, then the four `.cl` sources (chains above), `kolibri_kernels.h`, the CMake block, `kolibri_variant_names_test` (every name `decode_variants` returns for each (attn form, head, eager) is a target of the block: `${B70_KOLIBRI_DECODE_KERNELS}` passed as argv, as `k2_variant_names_test`).
- [ ] **Step 4: Mac GPU driver** `tools/mac/clrun/kolibri_run.cc`: builds `kol_prep.cl`, `kol_moe.cl` (gate||up at kUpKs 4 = 256 lanes, down at kDnKs 2 = 224 lanes, the card binary's defines, the router at 256), `kol_attn_eager.cl` at Kolibri's real shapes and runs them against `kolibri_ref.h` on random inputs plus the fixture's cases; prints `kolibri_run: <kernel> exact` or the first differing index. Add `kolibri_run` to section 5's driver loop in `tools/mac_check.sh`.
- [ ] **Step 5: the card test** `kolibri_kernels_test` (label `kolibri`, no checkpoint): every binary of the block at the real shapes against `kolibri_ref.h` - bitwise for prep, route, MoE (the int4 slots and the bf16 shared slot each tested with the other slots' weights zero), eager attention; flash decode at depths 1, 513, 4096 + 7 (wrapped ring), 30000 (full) within spec 10's bars; `kol_attn_decode` sliding at pos 9000 equal to a linear-cache run of the same 513 keys bitwise (the ring is only addressing).
- [ ] **Step 6: Mac gate.** `tools/mac_check.sh --base main --kernels`: host PASS (incl. `kolibri1_test`, `kolibri1_repack_test`, `kolibri1_rope_test`, `kolibri_ref_test`, `kolibri_variant_names_test`), l0 PASS, cmdlines `+N / -0 / ~0` (N = the block's count, recorded in the commit message), opencl PASS, `kolibri_run` exact on every kernel it runs.
- [ ] **Step 7: commit** `git commit -S -m "kernels: Kolibri-1 decode - sandwich norms, q/k norm + sliding RoPE, ring and NoPE attention, 384-expert route, MoE with a bf16 shared slot (spec 20c)"`.

### Task 5: `KolibriEngine` on one card, the CLI, KL2 / KL3 on decode

**Files:**
- Create: `src/runtime/kolibri/kolibri_sizes.{h,cc}`, `src/runtime/kolibri/kolibri_buffers.{h,cc}`, `src/runtime/kolibri/kolibri_capture.{h,cc}`, `src/runtime/kolibri/kolibri_engine.{h,cc}`, `src/runtime/kolibri/CMakeLists.txt` (libraries `b70_kolibri_plan`, `b70_kolibri_runtime`), `src/cli/kolibri_decode.h`, `tests/runtime/kolibri_plan_test.cc`, `tests/runtime/kolibri_decode_test.cc`, `tests/golden/kolibri_golden_test.cc`, `tests/golden/kolibri_partial_test.cc`
- Modify: `src/runtime/CMakeLists.txt` (`add_subdirectory(kolibri)`), `src/cli/b70_decode.cc` (dispatch), `src/cli/b70_serve.cc` (refusal), `tests/CMakeLists.txt`
- Test: `kolibri_plan_test` (host); `kolibri_decode_test`, `kolibri_golden_test`, `kolibri_partial_test`, `cli_reject_kolibri_*` (card / binary)

**Interfaces:**
- Consumes: Tasks 2-4.
- Produces:

```cpp
namespace runtime::kolibri {
enum class KolAttn { Flash, Eager };
inline constexpr KolAttn kDefaultKolAttn = KolAttn::Flash;     // until the box A/B (spec 18 §10.1's rule)
KolAttn kolibri_attn();                                        // B70_KOLIBRI_ATTN=flash|eager, unknown value throws
// The split, by 16b's rule: runtime::pp_balance over Kolibri's per-layer bytes at max_len.
// layer_bytes[l] = kol_layer_bytes(d).total() + (is_sliding(l) ? ring_bytes_per_layer(d) : max_len x 2 x kv_n x 2);
// dev0_fixed = embedding + RoPE + decode scratch + pp_link bytes (device 0); dev1_fixed = final norm + lm_head +
// RoPE + decode scratch + pp_link bytes (device 1).
std::vector<size_t> pp_layer_bytes(const model::Kolibri1Desc& d, uint32_t max_len);
runtime::PpBalance pp_split(const model::Kolibri1Desc& d, uint32_t max_len, bool int8_head, runtime::PpHandoff h);
struct DevicePlan : MemoryComponents { uint32_t device; size_t weights = 0, rope = 0, link = 0; };
std::vector<DevicePlan> plan(const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t max_len,
                             bool int8_head, bool debug_tap = false);
uint32_t max_len_that_fits(const model::Kolibri1Desc& d, const model::KolPlacement& p, bool int8_head,
                           const std::array<size_t, runtime::kPpDevices>& device_bytes, size_t reserve);
                                                               // min over devices, quantum kMaxLenQuantum, <= trained 262144
runtime::PpChoice pp_split_and_len(const model::Kolibri1Desc& d, bool int8_head,
                                   const std::array<size_t, runtime::kPpDevices>& device_bytes, size_t reserve);
                                                               // --pipeline-split auto with --max-len auto: pp_auto_split_and_len's rule
bool fits_one_card(const model::Kolibri1Desc& d, bool int8_head, size_t device_bytes, size_t reserve);   // --pp 1 allowed?
// "pipeline plan at max_len N, split s (layers [0, s) | [s, L)):" + one line per device - runtime::pp_describe's text.
std::string describe(const std::vector<DevicePlan>& p, const model::KolPlacement& pl, uint32_t max_len,
                     const std::array<size_t, runtime::kPpDevices>& device_bytes, size_t reserve);
size_t decode_launches(const model::Kolibri1Desc& d, const model::KolPlacement& p, KolAttn a, runtime::PpHandoff h);
//   per layer 15 (flash) / 17 (eager); + embed + fold on device 0; + 4 head = 756 / 856 at 50 layers on one card AND on
//   two: the cut is 16b's (device 0 ends with layer s-1's kol_post_add, which already wrote resid and the sums layer s's
//   norm reads - they are what crosses), so the two lists' compute launches add up to the one-card count;
//   --pipeline-handoff peer adds pp_send and pp_recv (758 / 858)
inline size_t full_kv_bytes_per_pos(const model::Kolibri1Desc& d) { return size_t(d.full_before(d.layers)) * 2 * d.kv_n() * 2; }  // 20480
inline size_t ring_bytes_per_layer(const model::Kolibri1Desc& d) { return size_t(model::Kolibri1Desc::kRing) * 2 * d.kv_n() * 2; } // 8 MiB

struct KolibriBuffers {                                        // one per device
  KolibriBuffers(l0::Context& ctx, const model::Kolibri1Desc& d, const model::KolPlacement& p, uint32_t device,
                 uint32_t max_len, KolAttn attn = kolibri_attn());
  l0::Mem control, kv_full_k, kv_full_v, ring_k, ring_v;     // persistent, zeroed by zero()
  l0::Mem resid, x, a, mo, partials, sumsq_a, sumsq_r, attn_q, attn_part, attn_out, logits_r, routes, moe_h, logits, argmax_part;
  std::unique_ptr<l0::Mem> attn_scores;                      // eager only
  void* full_k_layer(uint32_t layer) const; void* ring_k_layer(uint32_t layer) const;   // and _v
  void zero(l0::CmdList& imm);
};
class KolibriEngine {
 public:
  KolibriEngine(std::vector<l0::Context*> devices, loader::KolLoadedModel model, uint32_t max_len, bool debug_tap = false);
  void reset();
  void ingest(const std::vector<uint32_t>& ids);
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});
  std::vector<uint16_t> read_debug_resid();   // bf16 [layers][hidden] of the last replay (debug_tap)
  std::vector<uint32_t> read_routes();        // u32 [layers][32]
  std::vector<float> read_logits();           // fp32 [128000]
  std::vector<uint16_t> read_kv(uint32_t layer, uint32_t first, uint32_t count, bool v);   // full: rows; sliding: absolute positions via the ring
  MemoryComponents memory_use(uint32_t device) const; std::string memory_line() const;
  uint32_t pos() const; uint32_t max_len() const; size_t launches() const;
};
}
```

The list per layer (asserted count, Review Focus 1's order): `kol_norm_finish` (input_layernorm) · q||k||v GEMV · `kol_attn_prep` · `kol_attn_decode` + `kol_attn_reduce` (or eager's four) · o_proj GEMV · `prep_res_fold ..._Z` into `a` (Σa²) · `kol_post_add` (post_attn_norm, resid += , Σresid²) · `kol_norm_finish` (post_attention_layernorm) · router `gemv_bf16` · `kol_route` · `kol_moe_gate_up` · `kol_moe_down` (into `mo`) · `prep_res_fold_M1_K2560_SP0_G20` over `mo` (Σmo²) · `kol_post_add` (post_ffn_norm). Head: `kol_norm_finish` (model.norm) · lm_head (`gemv_bf16` / `gemv_i8w`) · `kol_argmax_stage1` · `kol_argmax_stage2`.

- [ ] **Step 1: failing host test** `kolibri_plan_test`: `decode_launches` 756 / 856 (flash / eager) at 50 layers on one card and on two with `copy`, 758 / 858 with `peer`; the per-device plan at max_len 262144, int4 attention, int8 head, split 25: weights as Task 3's, full KV 2,684,354,560 B per device (5 full layers each), rings 167,772,160 B per device (20 sliding layers each), RoPE 134,217,728 B per device; `pp_split` at 262144 (int8 head) returns 25 - device 0 ~24.11 GB against ~23.78 GB, where 24 and 26 leave a heavier side of ~25.15 / ~25.48 GB (derived; the test asserts `runtime::pp_balance`'s answer on `pp_layer_bytes`, and that `pp_balance` itself is the function called - no second balance rule); `max_len_that_fits` on 32.53 GB cards with a 1.5 GB reserve returns 262144 (capped by decision 3) on two cards; `fits_one_card` is false for the real model and true for the synthetic 5-layer one and for `--layers 30`; a `KolPlacement` split outside [1, layers - 1] throws.
- [ ] **Step 2: register and run, expect FAIL;** implement `kolibri_sizes.cc`; PASS.
- [ ] **Step 3: buffers, capture, engine (one device).** `capture` walks layers `[first, end)` of the device's part, asserts `decode_launches`, writes per-layer route rows and (debug) the tap after each layer's last launch. `generate` = K2Engine's contract (id i is the previous fence's; the (n+1)-th pending). The bench prompt: `tests/golden/prompts/kolibri_bench.ids` (Task 1) baked as `kKolibriBenchPrompt` in `kolibri_decode.h` (every id < 128000).
- [ ] **Step 4: CLI.** `b70-decode` dispatches on `model_type` `kolibri1` (`cli::kolibri::is_kolibri`, placed before `cli::check_pipeline` and before K2's check, so the Qwen engine's pipeline refusals never see Kolibri) to `cli::kolibri::run_decode`: `--ids`, `--n`, `--bench [--depth N] [--tg N]`, `--lm-head bf16|int8` (default bf16), `--max-len N|auto`, `--layers N` (development mode: `layers_limit`, prints the truncation), and 16b's switches parsed by 16b's parsers into a `cli::PipelineArgs` - `--pp N` / `--pipeline-parallel-size N` (**default 2 for Kolibri**; 1 or 2), `--pipeline-split auto|N`, `--pipeline-handoff copy|peer`. Until Task 6 lands, `--pp 2` is refused naming Task 6 and the synthetic / `--layers` runs take `--pp 1`. Refused before the device, each by name: `--pp 1` when `fits_one_card` is false ("Kolibri-1 holds ~42.5 GB of weights at int4, one card 32.5 GB: run it with --pp 2"); `--prefill` / `--prefill-length` / `--prefill-chunk` / `--prefill-backend` (spec 20d); `--mtp` (Kolibri has no MTP head); `--kv-cache int8` (Kolibri's KV is 20 KiB per position: bf16 only); `--profile` (Task 8); `--device N` with `--pp 2` (16b's rule: GPUs 0 and 1 of what Level Zero shows). `b70-serve` refuses `kolibri1` before the device naming spec 20e. Registered in the 20c block (tests/model/kolibri1/ holds config.json only - the fit check needs only the descriptor):

```cmake
b70_cli_reject(cli_reject_kolibri_one_card ${CMAKE_CURRENT_SOURCE_DIR}/model/kolibri1 --ids ${B70_OK_IDS} --n 4 --pp 1 EXPECT "--pp 2")
b70_cli_reject(cli_reject_kolibri_prefill ${CMAKE_CURRENT_SOURCE_DIR}/model/kolibri1 --ids ${B70_OK_IDS} --n 4 --prefill EXPECT "spec 20d")
b70_cli_reject(cli_reject_kolibri_mtp ${CMAKE_CURRENT_SOURCE_DIR}/model/kolibri1 --ids ${B70_OK_IDS} --n 4 --mtp 1 EXPECT "no MTP head")
b70_cli_reject(cli_reject_kolibri_kv8 ${CMAKE_CURRENT_SOURCE_DIR}/model/kolibri1 --ids ${B70_OK_IDS} --n 4 --kv-cache int8 EXPECT "bf16 only")
```
(and `cli_reject_kolibri_serve` through b70-serve with `EXPECT "spec 20e"`, as K2's serve refusal test does).
- [ ] **Step 5: the card gates.**
  - `kolibri_decode_test <ckpt> <ids> [int8]`: plan == allocation per device; K3 replay determinism - two `reset` + `ingest` + `generate(32)` runs give bitwise equal logits, route rows, full KV rows and both rings; the ring at pos 600 holds exactly positions 88..599 in their slots (read back through `read_kv`).
  - `kolibri_golden_test <ckpt> <oracle dir> [int8]` (k2_golden_test's copy): ingest the prompt with M = 1 replays, `generate(32)`; KL2 = golden_common.h's tie-aware token rule (determined rows exact, tie rows in the argmax set) plus the routing diagnostic per layer against `route.moe.{ids,w,gap}.L*`: a set difference where the reference's gap exceeds `tie_tol()` (`B70_KOL_TIE_TOL`, 1e-2 PROPOSED, set from kolibri_oracle.sh's printed gap distribution) FAILS; prints first differing row per layer, near-tie count, worst weight difference.
  - `kolibri_partial_test <real ckpt> <oracle-out-kolibri> <N>`: development mode (spec 20 §4): load `layers_limit = N` (default 30, one card), ingest each golden prompt, compare the tap's resid rows of layers 0..N-1 with `resid.L*` (cosine per row: median >= 0.9998, min >= 0.99, PROPOSED, printed per layer) and the routing diagnostic for layers < N.
  - Registered (label `checkpoint;golden;kolibri`, SKIP 77 without data): `kolibri_decode_synth_test`, `kolibri_golden_synth_int4attn_test`, `kolibri_golden_synth_bf16attn_test`, their `_eager` twins (`ENVIRONMENT B70_KOLIBRI_ATTN=eager`), `kolibri_golden_synth_i8head_test`, `kolibri_partial_test` (real, N = 30), `kolibri_partial_eager_test`.
- [ ] **Step 6: Mac gate** (`tools/mac_check.sh --base main --kernels`, all sections PASS) and **commit** `git commit -S -m "runtime: KolibriEngine decodes on one card - synthetic checkpoint and the real one truncated; KL2, KL3 on decode (spec 20c)"`.

### Task 6: two cards (`--pp 2`, on spec 16b's pieces)

**Files:**
- Modify: `src/runtime/pipeline_plan.{h,cc}` (the descriptor-free `pp_landing_layout` overload), `src/runtime/pipeline_engine.{h,cc}` (the `PipelineLink` constructor from a layout), `tests/runtime/pipeline_plan_test.cc` (the overload == the `ModelDesc` form on Qwen3.8 / Agnes / Ornith), `src/runtime/kolibri/kolibri_engine.{h,cc}`, `src/runtime/kolibri/kolibri_capture.{h,cc}`, `src/loader/kolibri1_loader.cc` (parts on two devices), `src/cli/kolibri_decode.h`, `tests/CMakeLists.txt`
- Create: `tests/runtime/kolibri_pp_test.cc`
- Test: `kolibri_pp_test`, `kolibri_golden_test` on the real checkpoint (two cards); K0: `pipeline_plan_test`, `pp_protocol_test`, `pipeline_args_test`, `pp_decode_test` unchanged

**Interfaces:**
- Consumes (16b, on main): `l0::Context(const Context&, uint32_t)`, `gpu_count`, `can_access_peer`, `l0::SyncEvent`, `CmdList::barrier_signal` / `wait_event`, `Fence::wait_for`, `runtime::PipelineLink`, `StageLink`, `PipelineLink::binding(uint32_t spin_limit)`, `PipelineLink::zero(l0::CmdList&, l0::CmdList&)`, `PipelineOptions`, `PpHandoff`, the `pp_handoff` binary, `cli::require_two_devices`.
- Produces:

```cpp
// src/runtime/pipeline_plan.h (16b's file; the ModelDesc form now returns pp_landing_layout(kM x hidden x 2, norm_sumsq's size))
PpLandingLayout pp_landing_layout(size_t resid_bytes, size_t sumsq_bytes);
// src/runtime/pipeline_engine.h (16b's file; the ModelDesc constructor delegates to this one)
PipelineLink(l0::Context& d0, l0::Context& d1, const PpLandingLayout& layout, PpHandoff mode);
// src/runtime/kolibri/kolibri_capture.h
CapturedStep build(l0::Context& ctx, const loader::KolDevicePart& part, const model::Kolibri1Desc& d,
                   KolibriBuffers& b, const runtime::StageLink* link,   // null on one card
                   l0::Mem* tap = nullptr);
// KolibriEngine: constructor gains `const runtime::PipelineOptions& opt = {}`; adds
void set_token(uint32_t id);                    // both Control blocks (PipelineEngine::set_token's rule)
uint32_t split() const;  runtime::PpHandoff handoff() const;
void drop_next_handoff();                       // P4 test hook, as PipelineEngine's
```

The cut is 16b's: device 0's list is embed, its layers, and ends after layer s-1's `kol_post_add` - which already wrote `resid` and `sumsq_r` (the 20 chunk sums layer s's `kol_norm_finish` reads); `copy`: two device-to-device copies of those into the landing buffer (`pp_landing_layout(1 x 2560 x 2, 20 x 4)`) and `barrier_signal(event)`; device 1's list starts with `wait_event(event)` and two local copies into its `resid` / `sumsq_r`, then layer s's `kol_norm_finish`. `peer`: `pp_send` / `pp_recv` (binary `pp_handoff`, unchanged) with `resid_words` 1280 and `sumsq_words` 20. After device 1's fence (`wait_for(timeout_ms)`) the host copies device 1's Control into device 0's. A step that times out host-signals the event, throws naming the hand-off and marks the engine until `reset()`.

- [ ] **Step 1: the failing tests.** `pipeline_plan_test` gains: `pp_landing_layout(resid, sumsq)` equals `pp_landing_layout(desc)` field for field for Qwen3.8, Agnes and Ornith, and for Kolibri's sizes the regions are page-aligned with the flag on a page of its own. `kolibri_pp_test <synth ckpt> <real ckpt> <ids>`: (a) synthetic, `--pp 1` vs `--pp 2 --pipeline-split 3`, each hand-off: prompt + 32 greedy tokens, logits, route rows, full KV and rings bitwise equal (Review Focus 5), Control blocks equal after every phase; (b) real checkpoint, `--pipeline-split 25` vs `20`: the same, bitwise; (c) P4 - `drop_next_handoff()` throws within the bounds and later steps throw until `reset()`; a device pair without peer access refused naming `zeDeviceCanAccessPeer`; (d) replay determinism with two cards. (The flags are the test's engine options; the CLI parses them into the same values.)
- [ ] **Step 2: register** (`LABELS "checkpoint;kolibri;pp"`, SKIP 77 with one GPU or no data); `pipeline_plan_test` runs on the Mac (`ctest --preset mac-host -R '^pipeline_plan_test$'`): FAIL until Step 3.
- [ ] **Step 3: implement** the two overloads in 16b's files (delegation only; 16b's tests unchanged), the loader's parts on two devices (`load_kolibri1` with two contexts: each layer read from the shards straight onto its device - no load-then-place), per-device `KolibriBuffers` and lists, the link and the Control mirror; `cli::kolibri::run_decode` with `--pp 2`: `require_two_devices(l0::Context::gpu_count())`, the view, the peer check before the load, `pp_split` / `pp_split_and_len` for `--pipeline-split auto` and `--max-len auto`, `describe` printed; `memory_line` one line per device.
- [ ] **Step 4: real-checkpoint KL2 on two cards:** register `kolibri_golden_test` against `${B70_KOLIBRI_SNAPSHOT}` / `${CMAKE_SOURCE_DIR}/oracle-out-kolibri` (`kolibri_golden_test`, `_i8head`, `_eager`), and `kolibri_decode_test` on it (K3).
- [ ] **Step 5: Mac gate and commit** `git commit -S -m "runtime: Kolibri-1 across two cards on spec 16b's split by bytes and hand-offs; bitwise across splits (spec 20c)"`.

### Task 7: docs, the box queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-05-spec20-kolibri-1-design.md` (new §11 "20c as built"), this plan's status line, `docs/superpowers/plans/box-validation-queue.md` (a row), `tools/box_validate/stages.sh` (its `row N` block), `docs/19-running-models.md` (a Kolibri row: two cards, decode only)

- [ ] **Step 1:** §11: structure, layouts, launch counts, the ring decision (4096 slots, 335.5 MB on 40 layers, derived, against §2's 513-slot ~42 MB: a 513 ring cannot hold a prefill chunk's own keys), the router's 256 x 2 lanes (§4's "512-lane variant" as 512 slots on 256 lanes), the flash / eager switch, known deviations (flash's fp32 probabilities vs the reference's eager bf16; eager's 8-lane softmax index follows the cached decode pass, which differs from the prompt pass's in sliding layers once pos >= 513 - ulp-level; sigmoid's `exp` vs torch's within 2 ulp in fp32 weights).
- [ ] **Step 2: the queue row** (next free number; 23 at this writing - 22 is spec 16b's): merged = branch and §11; unvalidated = every binary of the 20c kernel block never compiled by ocloc (list them), `kol_attn.cl`'s sub-group / 2D path, the loader on real data, the 756 / 757-launch lists, two-card hand-off; Mac-side = the host tests and `kolibri_run`; how = runbook stages `rN.oracle_synth` (opt-in cpu: `tools/box_validate/kolibri_oracle.sh $DATA synth`), `rN.k1` (`kolibri_kernels_test`), `rN.load`, `rN.k3` (`kolibri_decode_*`), `rN.golden` (synth, both arms, flash and eager), `rN.partial` (real, one card; needs 20b), `rN.pp` (two cards, `--pp 2` with each `--pipeline-handoff`; after row 22 has shown 16b's hand-offs on the card), `rN.speed` (opt-in); `rownote`s for what needs 20b's checkpoint (SKIP "missing data"). `python3 tools/box_validate/test_box_validate.py` passes; `tools/box_validate.sh --dry-run --only rN` prints the stages.
- [ ] **Step 3: commit** `git commit -S -m "docs: spec 20 §11 (20c as built), box queue row N - Kolibri-1 decode"`.

### Task 8: decode speed (box)

- [ ] Two cards, real checkpoint: `b70-decode <snap> --pp 2 --bench --depth D --tg 256 --max-len 40960` for D = 4096 and 32768, `--lm-head bf16` and `int8`, `--pipeline-handoff copy` and `peer` (16b's S1 question on Kolibri's 5 KB hand-off), `B70_KOLIBRI_ATTN` flash and eager, interleaved pairs (`tools/box_validate/interleave.sh -r 3`), median of 3, `uptime` and idle grade recorded; launches per token; per-device step time; the `{S, layout}` sweep of the two int4 attention rows (S 1 / 2 / 4, L 0 / 1) if the int4 arm ships. Against the derived roofline: int4 attention + int8 head reads ~2.39 GB per token (experts 0.63, bf16 shared 0.39, attention 0.91, routers 0.13, head 0.33) - ~250 t/s at ~600 GB/s; bf16 attention ~4.89 GB, ~120 t/s (derived). §2's ~2.0 GB counted the shared experts at int4 (0.1 GB); decision 4 keeps them bf16. `docs/BENCHMARKS.md` section "Kolibri-1 (spec 20)". **Commit** `git commit -S -m "spec 20c: Kolibri-1 decode speed"`.

**Gate for the plan:** Mac - every host test and `kolibri_run` green, `kernel_cmdlines` additions only. Box - K0 (sha256 of every pre-existing binary unchanged, existing suites unchanged); kernel tests green; KL3 (replay bitwise) and KL2 (tie-aware tokens, routing diagnostic clean) on both synthetic arms; the partial forward on the real checkpoint within the proposed bars (after 20b); two cards bitwise equal to one card and across splits (after 16b); speed rows recorded.
