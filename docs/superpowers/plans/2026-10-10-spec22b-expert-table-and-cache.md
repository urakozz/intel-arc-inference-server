# Spec 22b - the expert table, the pinned host mirror and the static cache; Qwen3.8-Flash-Next whole on two cards

**Status (2026-10-10): planned; begins only on the operator's go after plan 22a's record** (spec 22 §3's stopping
rule). Tasks 1 and 4's O0 half are also plan 22m's prerequisite (the coding mask addresses its kept experts through
the same table) and may begin on decision 9 alone (Dependencies). Box queue row 37 (renumbers at build time if taken).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the routed experts are addressed through a device-side table (`[layer][1024]` u64: 512 gate‖up pointers,
512 down pointers) that the expert kernels read instead of `id x stride`; each card keeps a byte-sized per-layer cache
of whole experts in VRAM filled at load from a routing profile, and every other expert lives in a pinned host mirror
that the kernels read zero-copy (decision 1's proposal) - no CPU expert compute, no host action between layers, the
captured lists unchanged per step. Gates O0 (every expert resident: bitwise spec 21's engine), O1 (residency 100 / 50 /
0 %, random and adversarial: bitwise), O3 (mirror checksums, the arena alias check, a hang-free refusal when pinning
fails); then **the full 48-layer model end to end on two cards** - decode and prefill, F3 against 21a's full-model
golden sets - and its first speed rows against P0.7's projection.

**Architecture:** spec 22 §4 (the cache, the host mirror, the indirection table, the miss path), §5 O0, O1, O3.
- **The indirection point** is 21c's / 21d's macro pairs: `Q4_EXPERT_GU` / `Q4_EXPERT_DN` in
  `src/kernels/qwen4exp/q4_moe.cl:195-196` (decode, verify, the MTP head) and in `src/kernels/qwen4exp/q4_pf_moe.cl:85-86`
  (the prefill's weight-batch dequant, `q4_pf_dequant_gu` / `_dn`). Under a new define `Q4_EXPERT_TABLE` both pairs
  become the table read plan 22a's probe measured (`((__global const uint*)(intptr_t)((__global const ulong*)(base))[(id)])`):
  `base` - the kernel's existing `w` argument - is then the layer's table row (gate‖up entries at `+0`, down at `+512`)
  instead of the layer's block base. New binaries carry `_TBL`; every existing binary is preprocessed token for token as
  main's (F0; the precedent is `prep.cl`'s `GDN_GATE_SIGMOID`, spec 21 §14).
- **The bind sites** change from the block base to the table row: `qwen4exp_capture.cc:549` / `:559` (decode, verify,
  the head's MoE) and `qwen4exp_prefill.cc:375` / `:378` (the prefill's dequant). The table's contents change only at
  load here (and between steps in 22c); the arguments are bound once at capture.
- **Where an expert lives** (`ExpertMap`, host): per (layer, expert) `Vram{slot}` - the layer's slot arena on the card
  that holds the layer: the loader's `Q4Layer::gate_up` / `down` become arenas of `n_l` blocks instead of 512 -, or
  `Host{hslot}` - the layer's pinned mirror range, gate‖up and down blocks of the mirrored experts packed in hslot
  order -, or `None` (plan 22m's dropped experts: the entry points at a NaN-filled poison block). An all-resident layer
  is `n_l = 512` with the identity map: today's allocation and today's addresses.
- **The fill** (decisions 2 and 8): `runtime::qwen4exp::expert_budget` (plan 22a Task 1) gives each card's bytes for
  routed experts; the per-layer counts `n_l` are uniform or from the profile's own curves (a greedy fill by marginal
  profile hits per slot over the card's layers); the resident experts of a layer are its top `n_l` by profile count
  (ties to the lower id). The profile is 22a's `spec22-profile-1` file (`offload_curve.py profile`), read from
  `--expert-profile FILE`, `$B70_Q4_PROFILE` or `<snapshot>-expert-profile.safetensors`.
- **The mirror** (decision 4): `all` (every expert pinned, 64.17 GB) or `misses` (only the non-resident experts:
  `(1 - f) x 64.17 GB` - the proposal while the PLE table's 51.8 GB shares the box's ~101 GiB pinned cap; 22c's swaps add
  a spare pool). Pinned per layer as one host-USM range (2 MiB aligned), one context for both cards (21b's PLE rule:
  host USM allocated once, visible to every device of the context).
- **Integrity** (O3; spec 22 §1's xe hazard): every VRAM arena and every mirror range is tagged per 2 MiB page before
  its data and read back by the device (`q4_page_check`, a generalised `q4_ple_check` over any number of ranges); every
  expert carries a 64-bit checksum of its repacked bytes, recomputed from its home after the copy; a pinned allocation
  that fails (or exceeds the host rule `mirror + PLE + 16 GiB <= MemAvailable`) is refused before any list exists,
  naming the bytes.
- **The full model:** `load_qwen4exp` without `--layers` loads all 48 layers with the tier on (the planner no longer
  refuses it; it refuses only when the non-routed part does not fit or the host rule fails), placement by `pp_split`;
  `b70-decode` runs it (`--prefill`, `--bench`); `b70-serve` still refuses it, now naming plan 22d.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake/ctest; Python 3 for the profile file in tests.

**Spec:** `docs/superpowers/specs/2026-10-09-spec22-moe-expert-offload-design.md` (§4, §5 O0 O1 O3 O4, §6 22b, §7
decisions 1, 2, 4, 6, 8) and plan 22a's record `docs/probe-offload-<date>.md` (the decisions' evidence). Spec 21 §13
(the layouts, the PLE's host-USM pinning and alias check in `loader/qwen4exp_ple_usm.cc`, the planner), §14 (the decode
list, the macro pair), §15 (the prefill walk, the dequant batches), §16 (verify at M = K + 1, the head's MoE).
Precedents: `loader/qwen4exp_ple.{h,cc}` + `qwen4exp_ple_usm.cc` (a host-USM table, its tags, its device pointer table
and `q4_ple_check`), plan 21c Task 2 (one existing source gains a define, new binaries by name), plan 21b's planner
tests.

## Dependencies and branch points

- **Plan 22a's record and the operator's go.** Decisions this plan builds, with its default when the decision is
  still open (each behind a switch so the other arm is measurable):
  1. the miss path: **zero-copy** (proposed). If decision 1 rules gather staging (22a's arm K), Task 1 Step 5 adds the
     `_STG` form; copy-engine staging as spec 22 §3 writes it (arm S) needs a per-layer host step the captured list
     does not have, and is not built (see the hand-back's ambiguities).
  2. the per-layer budget: `--expert-budget uniform|curve`, default **curve** if 22a's held-out curves gain over uniform
     by >= 0.5 point of h at the two-card f, else uniform (recorded).
  4. the mirror: `--expert-mirror all|misses`, default **misses** (the PLE table and the full mirror exceed the
     measured cap - spec 22 §1; 22a's `pin` gives the numbers).
  6. the MTP head's experts: **resident** (1.34 GB; the head's table row is the identity over its own 512 blocks).
  8. the profile: 22a's `profile.safetensors` from the CPU reference's traces (Intel's checkpoint), shipped beside the
     checkpoint; `--expert-profile none` fills by id order (test mode, printed as such).
- **Rows 31-34 PASS** (21b's loader, 21c's decode incl. `r32.partial`, 21d's prefill, 21e's verify and head): this
  plan changes their binding sites and must leave their gates green.
- **For plan 22m only** (the coding mask without the tier): Task 1 and Task 4's O0 half are the table it needs (a
  loader that skips dropped experts must remap ids to compacted blocks: the table is that remap). If decision 9 is
  ruled before the operator's go, those two pieces are built first on this plan's branch and the rest waits.
- **Not here:** adaptive swaps and prefill streaming (22c), serving the full model, MTP / snapshots / A4 on it, `--max-len
  auto` with the cache, the record (22d), the mask (22m).

## Global Constraints

- Branch `spec22b-expert-table` from main; box tree automatic; `tools/box.env` copied if missing, never committed,
  never printed; `oracle-out*` symlinked.
- **Commit after each step that changes a file** (signed `git commit -S`; never bypass). No `rm -rf`. No merge, no push.
- **F0:** every existing kernel binary keeps its name, command line and bytes (`tools/kernel_cmdlines` additions only;
  G0's sha256 on the box). `q4_moe.cl` and `q4_pf_moe.cl` gain `#if Q4_EXPERT_TABLE ... #else <today's two lines>
  #endif`; every existing variant of both preprocesses token for token as main's (`clang -E -P`, checked on the Mac and
  recorded); if G0 shows any moved, the table form moves to a copy (`q4_moe_tbl.cl`) and the sources are restored.
- **No host action between layers; nothing changes the captured lists per step.** The table is written at load (and,
  in 22c, between steps); a step never waits for the host.
- **Determinism:** residency never changes arithmetic - the bytes are the same, the work-groups and their order are the
  same; only the address differs. No atomic feeds a result.
- Every number measured, or marked derived / estimated / proposed. Box: every GPU command under `flock
  ~/b70-gpu.lock` (once for both cards), detached, polled; interleaved pairs after warm-up, median of 3, `uptime` and
  idle grade; `-j44`. **Host RAM:** the full model's runs pin the mirror and the PLE table - no CPU oracle beside them;
  `free -g` checked first.
- Mac checks `tools/mac_check.sh --base main --kernels` (host, Level Zero syntax, cmdlines, OpenCL syntax, Mac GPU runs).

## Review Focus

1. **O0 is a code identity.** With the identity map the table entry of expert e is `gate_up + e x 1,740,800` /
   `down + e x 870,400` - exactly `Q4_EXPERT_GU` / `_DN` today. `B70_Q4_EXPERTS=direct` binds 21c's / 21d's binaries
   (legal only when every expert is resident, refused otherwise); the default `table` binds `_TBL`; the two runs are
   bitwise (ids, logits, routes, selections, every state) in decode, prefill and verify.
2. **O1 is bitwise, including the adversarial sets.** Forced residency (`B70_Q4_RESIDENT=all|none|random:<f>:<seed>|
   adversarial-profile|adversarial-prompt`): 0 %, 50 % (two seeds), 100 %, the profile's most-routed experts on host,
   and exactly the experts a first run of the prompt routed to on host (read from `read_routes` /
   `read_prefill_routes`). Every run equals the 100 % run bit for bit; a difference is a bug, never a tolerance.
3. **The table rows the kernels read are the rows the capture bound.** gate‖up launches bind `row`, down launches `row +
   512`; the head's MoE binds its own row; on two cards each card's table holds only its layers' rows and its own VRAM
   addresses (host mirror addresses are context-wide). A test reads every bound row back and checks it against
   `ExpertMap` for each layer, both devices.
4. **Nothing reads a slot or a mirror page before its data and tags are checked.** The load order is: allocate, tag,
   read tags back on the device, copy the data, checksum from the home, write the table, capture. A failed check throws
   naming the layer, the expert and the page; nothing is captured.
5. **The planner tells the truth.** The plan with the tier = non-routed + the cache arenas + reserve per card, and the
   host side = mirror + PLE; `memory_line()` prints the cache (experts and GB per card), the mirror and f; the planned
   bytes equal the allocated bytes (21b's rule).

---

### Task 1: the table form of the expert kernels

**Files:**
- Modify: `src/kernels/qwen4exp/q4_moe.cl`, `src/kernels/qwen4exp/q4_pf_moe.cl` (the define), `src/kernels/qwen4exp_kernels.h`
  (`moe_variant(M, shared_bf16, bool table = false)`, `pf_moe_variant(bool table = false)`, the `decode_variants` /
  prefill / MTP name lists gain the table flag; a new `page_check_variant()`), `src/kernels/CMakeLists.txt` (a block
  `# ==== Spec 22b: the expert table (model::qwen4exp()) ==== (begin) / (end)` after spec 21e's: `q4_moe_M<1..4>_E512_T10_D2560_I640_{SH4,SHB}_TBL`,
  `q4_pf_moe_E512_T10_D2560_I640_L256_TBL`, `q4_page_check`), `tests/kernels/qwen4exp_kernels_test.cc`,
  `tests/kernels/qwen4exp_pf_kernels_test.cc`, `tests/kernels/qwen4exp_variant_names_test.cc`,
  `tests/kernels/qwen4exp_pf_variant_names_test.cc`, `tools/mac/clrun/qwen4exp_run.cc`, `tests/CMakeLists.txt`
- Create: `src/kernels/qwen4exp/q4_tier.cl` (`q4_page_check(ptrs, first_page, out, pages, ranges)`: one lane per 2 MiB
  page of any number of ranges - `q4_ple_check`'s rule without its fixed 32)
- Test: the names tests (host), `qwen4exp_kernels_test` / `qwen4exp_pf_kernels_test` (card), `qwen4exp_run` (Mac GPU)

**Interfaces:**
- Produces: the `_TBL` binaries (kernel signatures unchanged: `w` is the table row); `q4_page_check`.
- [ ] **Step 1: failing names tests** (every `_TBL` name the lists return is a target; the existing names unchanged). FAIL.
- [ ] **Step 2: the define, the CMake block, `q4_tier.cl`**; names PASS; the preprocessed-source check of every existing
  `q4_moe` / `q4_pf_moe` variant (`clang -E -P` against main's, token for token) recorded in the commit. Commit `git
  commit -S -m "kernels: the expert table - q4_moe / q4_pf_moe read a routed expert's blocks through a u64 table row
  (_TBL), existing binaries unchanged; q4_page_check (spec 22b)"`.
- [ ] **Step 3: the card tests.** `qwen4exp_kernels_test` gains `table` cases at M = 1..4, both shared forms: the table
  over (a) the identity, (b) a permuted VRAM slot order, (c) every routed slot in a host-USM copy, (d) 1 / 3 / 9 of 10
  on host - each bitwise the direct binary's `h` and `y`; `qwen4exp_pf_kernels_test` the same for the dequant batches
  (a chunk's touched experts split across VRAM and host, untouched experts' entries pointing at an unmapped-looking
  poison address that must never be read - the kernel's early return); `q4_page_check` over 3 host + 2 device ranges.
- [ ] **Step 4: Mac gate** (`qwen4exp_run`: the table cases on the Mac GPU, host memory as "host"; `tools/mac_check.sh
  --base main --kernels` exit 0, cmdlines `+10 / -0 / ~0` (8 `q4_moe` `_TBL`, the prefill's, `q4_page_check`) or the count recorded). Commit `git commit -S -m "tests:
  table-addressed experts bitwise the direct ones at M = 1..4 and in the prefill dequant (spec 22b)"`.
- [ ] **Step 5 (only if decision 1 rules gather staging):** `_STG` variants - the MoE launch pair preceded by
  `q4_expert_stage(route, table_row, bounce)` (22a's `po_stage`: slots whose entry is outside the card's arena range are
  copied into bounce slot k, the expert kernels read bounce slot k for those), +1 launch a MoE (+1 per QSA / GDN layer
  in `device_launches`); the same bitwise cases. Recorded as not built otherwise.

### Task 2: the profile, the fill and the planner with a cache (host)

**Files:**
- Create: `src/loader/qwen4exp_profile.{h,cc}`, `src/runtime/qwen4exp/qwen4exp_experts.{h,cc}` (the host half:
  `ExpertMap`, the fill, the table rows' values), `tests/runtime/qwen4exp_tier_test.cc`,
  `tools/oracle/qwen4exp_profile_fixture.py` -> `tests/loader/testdata/q4_profile_tiny.safetensors` (a committed
  48 x 512 profile from a seeded generator, and an 8 x 512 one)
- Modify: `src/runtime/qwen4exp/qwen4exp_sizes.{h,cc}` (`plan` with a tier: `ExpertTierPlan`; `require_fits` and
  `fits` with the tier; `describe` prints the cache and the mirror), `src/runtime/qwen4exp/CMakeLists.txt`,
  `src/loader/CMakeLists.txt`, `tests/runtime/qwen4exp_plan_test.cc`, `tests/CMakeLists.txt`
- Test: `qwen4exp_tier_test`, `qwen4exp_plan_test` (host)

**Interfaces:**

```cpp
namespace loader {
struct Q4Profile { uint32_t layers = 0, experts = 0; std::vector<uint32_t> counts, mtp_counts; std::string source; };
Q4Profile load_q4_profile(const std::string& file, const model::Qwen4ExpDesc& d);  // format "spec22-profile-1";
                                                    // refused by name: wrong shape, wrong format, a different model
std::string q4_profile_path(const std::string& snapshot, const std::string& flag);  // flag, $B70_Q4_PROFILE, beside
}
namespace runtime::qwen4exp {
enum class BudgetPolicy { Uniform, Curve };          // decision 2
enum class MirrorPolicy { All, Misses };             // decision 4
struct ExpertHome { enum Kind : uint8_t { Vram, Host, None } kind; uint32_t slot; };
struct ExpertMap {                                   // [layer][512] (+ the head's 512 last), per device placement
  std::vector<ExpertHome> home; std::vector<uint32_t> n_vram, n_host;   // per layer
  const ExpertHome& at(uint32_t layer, uint32_t e) const;
};
struct ExpertTierPlan {
  bool on = false; BudgetPolicy budget = BudgetPolicy::Curve; MirrorPolicy mirror = MirrorPolicy::Misses;
  std::array<size_t, kPpDevices> cache_bytes{};      // from expert_budget (22a) or --expert-cache <GB>
  size_t spare_host = 0;                             // 22c's swap pool (0 here)
};
// The fill: n_l per layer from the budget and the policy, then the top n_l experts by profile count (ties to the
// lower id). `forced` (the test hook B70_Q4_RESIDENT) overrides which experts are resident, never the arena sizes.
ExpertMap fill(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, const ExpertTierPlan& t,
               const loader::Q4Profile* profile, const std::string& forced = "");
size_t mirror_bytes(const ExpertMap& m, MirrorPolicy mp);   // the pinned host bytes the tier needs
}
```

- [ ] **Step 1: failing tests.** `qwen4exp_tier_test`: the uniform fill gives every layer `floor(cache / layers /
  2,611,200)` slots; the curve fill on a profile where layer 0 routes 10 experts and layer 1 all 512 gives layer 0 ten
  slots and the rest to layer 1; the top-n choice with planted ties takes the lower id; `forced` = `none` / `all` /
  `random:0.5:7` (deterministic: the same set every call) / `adversarial-profile`; `mirror_bytes(misses)` = the
  non-resident experts x 2,611,200; the profile loader refuses each wrong file by name. `qwen4exp_plan_test`: the full
  48 layers on two cards with the tier at 32768 fits (non-routed + arenas + reserve = capacity within one slot), its
  `describe` lines; without the tier the refusal text unchanged; the host rule refuses `mirror all` + the PLE at 128 GB
  of RAM with 16 GiB headroom, naming both. FAIL.
- [ ] **Step 2: implement**; PASS. Commit `git commit -S -m "runtime: the expert map - a byte-sized per-layer cache
  filled from a routing profile, the mirror's bytes, the planner with the tier (spec 22b)"`.

### Task 3: the loader - arenas, the pinned mirror, the table, the checks

**Files:**
- Create: `src/loader/qwen4exp_experts.{h,cc}` (host half: the per-expert repack routed to its home, checksums),
  `src/loader/qwen4exp_mirror_usm.cc` (the device half: the pinned ranges, tags, `q4_page_check`, the table upload -
  `qwen4exp_ple_usm.cc`'s split), `tests/loader/qwen4exp_load_tier_test.cc` (card)
- Modify: `src/loader/qwen4exp_loader.{h,cc}` (`load_qwen4exp` gains `const runtime::qwen4exp::ExpertTierPlan& tier`
  and the profile path, defaulting to off = today's load; `Q4LoadedModel` gains `Q4ExpertTier tier`), `src/loader/CMakeLists.txt`,
  `tests/CMakeLists.txt`
- Test: `qwen4exp_load_tier_test` (card, label `checkpoint;qwen4exp`)

**Interfaces:**

```cpp
namespace loader {
struct Q4ExpertTier {                               // owned by Q4LoadedModel; off when !plan.on
  runtime::qwen4exp::ExpertMap map;
  std::vector<std::unique_ptr<l0::Mem>> mirror;     // per layer one pinned range (gate||up blocks, then down blocks)
  std::vector<std::unique_ptr<l0::Mem>> table;      // per device: u64 [its layers (+ the head)][1024]
  std::unique_ptr<l0::Mem> poison;                  // 22m's None entries (NaN bf16-scaled blocks)
  std::vector<uint64_t> checksums;                  // [layer][512]: of the repacked gate||up then down bytes
  size_t mirror_bytes = 0, mem_before = 0, mem_after = 0; double seconds = 0;
  uint64_t table_row(uint32_t layer) const;         // device address of the layer's row (on its device)
};
}
```

- [ ] **Step 1: the failing card test** `qwen4exp_load_tier_test <synth ckpt>` (ours and Intel's form, 4 layers + the
  head): forced residency 100 / 50 / 0 % and `misses` / `all`: every expert's gate‖up and down blocks read back from its
  home (VRAM slot by a copy, host range directly) equal the repack 21b's `qwen4exp_repack_test` writes; the checksums
  match; `q4_page_check` reads every page's tag of every arena and range on each device; the table rows read back equal
  `ExpertMap` (Review Focus 3); the planned bytes = the allocated bytes; `B70_Q4_PIN_LIMIT_GB=1` (a test hook capping
  the mirror's pinned bytes) is refused before any list, naming the bytes, within 1 s (no hang); the 21b load tests
  unchanged with the tier off. FAIL.
- [ ] **Step 2: implement** (Review Focus 4's order); PASS on the Mac's host half (`qwen4exp_synth_host_test` unchanged)
  and through the Level Zero syntax check. Commit `git commit -S -m "loader: Qwen3.8-Flash-Next's expert tier - VRAM slot
  arenas, the pinned host mirror, the device table, tags and checksums (spec 22b)"`.

### Task 4: the engine binds the table; the CLI; O0 and O1

**Files:**
- Modify: `src/runtime/qwen4exp/qwen4exp_capture.{h,cc}` (`:549` / `:559` bind `tier.table_row(l)` / `+ 512 x 8`; the
  head's MoE its row), `src/runtime/qwen4exp/qwen4exp_prefill.cc` (`:375` / `:378` likewise), `src/runtime/qwen4exp/qwen4exp_engine.{h,cc}`
  (`B70_Q4_EXPERTS=table|direct` read at construction - anything else throws, `direct` with a non-resident expert
  throws; `memory_line()` with the cache and the mirror; a test accessor `read_table_row(layer)`),
  `src/cli/qwen4exp_decode.h` (`--expert-cache auto|off|<GB>`, `--expert-profile FILE|none`, `--expert-mirror
  all|misses`, `--expert-budget uniform|curve`; `refuse_whole` only when the tier is off or cannot fit; the tier on by
  default exactly when the planned model does not fit), `src/cli/qwen4exp_serve.h` (`check_serve`'s whole-model refusal
  names plan 22d), `tests/CMakeLists.txt`
- Create: `tests/runtime/qwen4exp_tier_o1_test.cc`
- Test: `qwen4exp_tier_o1_test` (card); every existing qwen4exp card test (K0)

**Interfaces:**
- Produces: `b70-decode <qwen4exp> [--layers N]` with the tier; `B70_Q4_RESIDENT` honoured with `--layers N` (O1's
  forced residency on models that fit); the engine's `memory_line()` adds `experts: <vram>/<total> resident (f <x>),
  mirror <GB> pinned`.
- [ ] **Step 1: the failing test** `qwen4exp_tier_o1_test <ckpt> <ids> [intel ckpt]`:
  - **O0:** `direct` against `table` (identity) - one card and `--pp 2` (copy and peer), decode 32 tokens after a
    prefill of 2200 ids (chunks of 2048 and 64), MTP verify at K = 1..3 when the head is loaded: ids, logits, routes,
    selections, KV, keys, GDN / PLE state bitwise.
  - **O1:** Review Focus 2's sets on the synthetic checkpoints (one card, two) and on Intel's `--layers 18` (one card)
    and `--layers 37` (two): every run bitwise the 100 % run (decode, prefill, verify).
  - Review Focus 3's row read-back on both devices. FAIL (no binding).
- [ ] **Step 2: implement**; the existing qwen4exp card tests' expectations unchanged (their runs use `table` with the
  identity: O0 holds them). Commit `git commit -S -m "runtime: Qwen4ExpEngine reads its experts through the table - O0
  bitwise the direct binaries, O1 bitwise across residency (spec 22b)"`.
- [ ] **Step 3: the CLI and its refusals** (`cli_reject_qwen4exp_full` now expects the tier's refusal only with
  `--expert-cache off`; new `cli_reject_qwen4exp_expert_direct` - `B70_Q4_EXPERTS=direct` with a cache - and
  `_profile` - a wrong profile file). Mac gate. Commit `git commit -S -m "cli: b70-decode <qwen4exp> loads the whole
  model through the expert tier - --expert-cache / --expert-profile / --expert-mirror / --expert-budget (spec 22b)"`.

### Task 5: the whole model on two cards (box)

**Files:**
- Create: `tests/golden/qwen4exp_full_golden_test.cc` (or a `full` mode of `qwen4exp_golden_test`, whichever keeps the
  file smaller - said in the commit)
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: the test** `qwen4exp_full_golden_test <intel ckpt> <oracle-out-q4exp> [--pp 2]`: the whole model, the
  tier on (the profile from 22a), the int8 head; per golden prompt (`q4exp_short`, `4k`, `8k`, `32k`, `agentic`) the
  prompt by prefill then `generate(32)`: the tie-aware token gate (golden_common.h), the routing diagnostic per layer
  (`B70_Q4_TIE_TOL`), gate S per QSA layer (`B70_Q4_SEL_TOL`), the H tap per layer against `H.L*` (median cosine >=
  0.9998, min >= 0.99 - 21c's partial bars, PROPOSED); the load line (bytes per card, experts resident per card, the
  mirror, MemAvailable, seconds). Registered `qwen4exp_full_golden_intel_test` (label `checkpoint;golden;qwen4exp;pp`,
  SKIP 77 without two cards, Intel's checkpoint, its PLE file, 21a's full sets or the host RAM).
- [ ] **Step 2 (box):** O3 at real scale (the mirror and the PLE pinned together - the tag and checksum lines), the
  golden gate, then the first speed rows: `b70-decode --bench --depth 4096 / 32768 --tg 256` and `--prefill-length 4096
  / 32768`, `--pp 2`, int8 head, interleaved against nothing (absolute rows) and against 22a's projection (the ratio
  printed; O4's bars are set from P0, recorded only). Commit `git commit -S -m "spec 22b: Qwen3.8-Flash-Next whole on
  two cards - F3 against the full reference, O3 at scale, the first decode / prefill rows"`.

### Task 6: docs, the queue row

**Files:**
- Modify: `docs/superpowers/specs/2026-10-09-spec22-moe-expert-offload-design.md` (a "22b as built" section; decisions
  1, 2, 4, 6, 8 as built), `docs/superpowers/plans/box-validation-queue.md` (row 37), `tools/box_validate/stages.sh` (a
  `row 37` block), `tools/box_validate/data.sh` (`have q4exp_profile`), `docs/19-running-models.md` (the whole model:
  the flags, the RAM it pins), this plan's status line

- [ ] **Step 1: the row** - stages `r37.k0` (G0: every existing `q4_moe` / `q4_pf_moe` binary unchanged), `r37.host`,
  `r37.k1` (`kbins` of the new binaries + the kernel tests' table cases), `r37.load` (`qwen4exp_load_tier_test` on the
  synthetics), `r37.o1` (`qwen4exp_tier_o1_test`: synthetics, then Intel's 18 / 37 layers), `r37.reject`, `r37.full`
  (Task 5's golden test; RAM rule in its rownote: no CPU oracle live, `free -g` >= mirror + PLE + 16 GiB), opt-in
  `r37.speed` (Task 5 Step 2's rows); `test_box_validate.py` passes; `--dry-run --only r37` prints them. Commit `git
  commit -S -m "box: queue row 37 - spec 22b, the expert table and the static cache, the whole model on two cards"`.
- [ ] **Step 2: "22b as built"** after the box run (the measured f per card, h on the golden prompts from
  `read_routes` against the map, the rows). Commit `git commit -S -m "docs: spec 22 (22b as built)"`.

**Gate for the plan:** Mac - names tests, `qwen4exp_tier_test`, `qwen4exp_plan_test`, `qwen4exp_run`'s table cases
exact, the preprocessed-source identity of every existing `q4_moe` / `q4_pf_moe` variant, cmdlines additions only,
`mac_check.sh --kernels` exit 0. Box - G0; the table cases bitwise; O0 and O1 bitwise on the synthetics and on Intel's
18 / 37 layers, one card and two; O3 (tags, checksums, the refusal without a hang); the whole model's F3 on two cards
against 21a's full sets; the first speed rows recorded beside 22a's projection.
