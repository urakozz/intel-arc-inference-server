# Spec 22a - the expert-offload probe (P0): the links, zero-copy reads, pinned capacity, the hit-rate curve, the pruning curve, the projection

**Status (2026-10-10): planned, not started.** Box queue row 36 (it renumbers at build time if taken). Everything
that can be written and checked without the box is (Tasks 1-4, Mac gates); every number is the box's (Task 5).
Plans 22b-22d begin only on the operator's go after this plan's record (spec 22 §3, the stopping rule); plan 22m (the
coding mask) begins on decision 9, whatever the tier's verdict.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** measure everything spec 22's tier depends on and project its speed before any engine work: the PCIe links
both B70s train at (P0.1), copy-engine host-to-device rates (P0.2), GPU kernels reading experts straight from pinned
host USM with the expert GEMV's own access pattern - alone, mixed with VRAM slots, replayed, against copy-engine
staging (P0.3), host RAM bandwidth under that load (P0.4), pinned capacity and the alias check (P0.5), the hit-rate
curve from spec 21a's routing traces (P0.6: static fill, simulated swaps, MTP verify unions, prefill bytes), the
pruning curve for the coding mask (P0.8: REAP scores, masked reference runs at 50 / 62.5 / 75 / 87.5 % kept, KL,
argmax, A4 tool-call accuracy) and the projection with the stopping rule (P0.7: stop if two-card decode at 32k
projects below 40 t/s). The record is `docs/probe-offload-<date>.md`; it feeds decisions 1-9.

**Architecture:** no engine change except one host-only planner function. Four tools:
- `tools/probe/probe_offload.cc` - one Level Zero binary on the repo's `l0::` wrappers (one context over both cards,
  spec 16 decision 1), modes `copy`, `zc`, `replay`, `hostbw`, `pin`, `alias`, `hold`, each printing `po <mode> ...`
  lines a stage greps. Its expert kernels are **21c's `q4_moe_gate_up` / `q4_moe_down` token for token** with the
  two address macros swapped for a table read (`tools/probe/probe_offload_moe.cl`, generated from
  `src/kernels/qwen4exp/q4_moe.cl` by `tools/probe/mk_offload_kernel.py`, committed, drift-checked), so the probe's
  zero-copy arm reads host USM exactly as 22b's engine would, and its all-VRAM arm is checked bitwise against 21c's own
  binary `q4_moe_M1_E512_T10_D2560_I640_SH4`.
- `tools/probe/offload_links.sh` - P0.1's `lspci` / `dmidecode` / `lscpu` capture.
- `tools/oracle/offload_curve.py` - the trace analyser (P0.6), the REAP scorer and mask writer (P0.8), the profile
  writer 22b loads (decision 8), the prune report and the projection (P0.7).
- `tools/oracle/qwen4exp_ref.py --expert-mask FILE` and its `maskeval` subcommand, `tools/toolcall/oracle_generate.py
  --expert-mask` - the masked reference runs of P0.8.

The projection's byte model is not re-derived in Python: `runtime::qwen4exp::expert_budget` (Task 1, host-only, in the
planner that refuses the full model today) computes what every device has left for routed experts at a context, and
`qwen4exp_plan_test --budget-json` prints it; 22b's cache sizes from the same function.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake/ctest; bash; Python 3 (numpy, safetensors) and, for the
masked reference, torch in `agnes-ref-img` with 21a's transformers 5.19.0 site (`tools/oracle/qwen4exp_env.sh`).

**Spec:** `docs/superpowers/specs/2026-10-09-spec22-moe-expert-offload-design.md` (§1, §2, §3 P0.1-P0.8 and the
stopping rule, §4b, §7 decisions 1-9). Spec 21 §12 (21a's `trace`: per routed token the fp32 router probability `p`, the
expert output norm `onorm`, the MTP head's `mtp_ids` / `mtp_p` / `mtp_onorm` - `TRACE_FORMAT` in `qwen4exp_ref.py`), §13
(the layouts: `q4_gate_up_offset` / `q4_down_offset`, 1,740,800 / 870,400 B a block, 1,336,934,400 B a layer; the PLE's
16 + 16 host-USM ranges with the tag-and-readback check; the memory plan), §14 (`Q4_EXPERT_GU` / `Q4_EXPERT_DN`; the PLE
rate `r32.ple_rate`), §15 (the prefill's 512-expert sort; `q4_pf_moe.cl` holds a second macro pair), §16
(`read_verify_routes`; the A4 set and its reference). Precedents: plan 16a (a probe plan: probe binaries in
`tools/probe/`, one context over both cards, bounded waits), `tools/probe/probe_host_copy.cc` (spec 7 P0's pinned copies
and the `MemAvailable - 16 GiB` cap), `tools/probe/probe_bw.cc`, `docs/probe-prefix-cache-2026-09-27.md` §1-§2,
`docs/probe-eagle3-k2-2026-10-08.md` (K2's routing locality: M consecutive rows touch 12.46 / 15.75 / 18.55 distinct
experts a layer at M = 2 / 3 / 4 against the iid 15.36 / 22.13 / 28.36 - the shape P0.6's verify term should show).

## Dependencies and branch points

- **Row 30 (spec 21a on the box CPU):** `r30.traces` (`oracle-out-q4exp-traces/*.routes.safetensors`, `have
  oracle_q4exp_traces`: 39 or more files - the 36 A4 scenarios, `q4exp_agentic`, `code`, `prose`, + the opencode
  recording when `OPENCODE_LOG` was set) is P0.6's and P0.8's input; `r30.golden` (`oracle-out-q4exp`, the full
  model's golden logits) is P0.8's KL reference. Without them Task 5's CPU stages SKIP "missing data".
- **Row 32:** `r32.k1` (21c's `q4_moe_M1_E512_T10_D2560_I640_SH4` compiled - the probe's control binary) and
  `r32.ple_rate` (spec 22 §1's first host-USM number; P0.3 is its expert-shaped successor). `r32.partial` is P0.6's
  cross-check: the engine's routes for layers [0, 18) equal the reference's (its routing diagnostic), and 21a's Mac
  test holds the `trace` routes equal to `run`'s - so no new route dump is written here.
- **Row 31:** `r31.load_real` (the 51.8 GB PLE table pinned: time, MemAvailable before / after) is P0.5's start.
- **Row 34:** `r34.accept` (decision 4's acceptance from the CPU reference: P0.7's MTP rows) and `r34.a4_ref`
  (`oracle-out-q4exp-a4`, `have oracle_q4exp_a4`: the full model's A4 outputs, P0.8's accuracy reference). P0.7 uses
  an acceptance range (0.5-0.8 at K = 1, PROPOSED) when `r34.accept` has not run.
- **Row 22** (`r22.p1`: two cards, one context, peer access) before any "both cards" arm.
- **Nothing in `src/kernels/` changes** (G0 trivially; kernel_cmdlines +0 / -0 / ~0). The only `src/` change is Task 1's
  host-only planner function.

## Global Constraints

- Branch `spec22a-offload-probe` from main; box tree automatic (`tools/box.sh dir`); `tools/box.env` copied from the
  main checkout if missing, never committed, never printed; `oracle-out*` symlinked (`tools/box_validate/data.sh link`).
- **Commit after each step that changes a file** (signed: `git commit -S`; never bypass signing or hooks). No `rm
  -rf`. No merge, no push.
- Every number measured, or marked derived / estimated / proposed / published; cite file and line for every derived
  byte count. The record's tables carry the date, the commit, `uptime`, the idle grade and the GPU clock state.
- **Box protocol:** every GPU command under `flock ~/b70-gpu.lock` (once for both cards in a two-card arm); detached
  (`tools/probe/detach.sh`), polled with bounded loops; both cards free of other DRM holders for timed arms (the idle
  protocol); a ~0.5 s warm-up burst, then control and candidate as **interleaved pairs**, median of 5 with the range
  (paired ratios, never sequential arms). `-j44` builds.
- **Host RAM is shared:** P0.5's capacity arm and any pinned arm above 16 GB run only with no CPU oracle live and
  `free -g` >= 110 GB available (the probe refuses above `MemAvailable - 16 GiB`, printing the cap: spec 7's rule).
  CPU oracle stages (P0.8) run one at a time with `free -g` >= 70 GB, never beside the pinned arms.
- **Every wait is bounded** (fences `Fence::wait_for`, events polled with a timeout): a hang is a FAIL line, never a
  stuck box.
- Mac checks: `tools/mac_check.sh --base main --quick` (host tests, Level Zero syntax of `tools/probe/`, cmdlines),
  plus Task 2's hand-run OpenCL syntax of the probe's two `.cl` files (section 4 lists `src/kernels` only) and its Mac
  GPU run; Python tests in `agnes-ref-img` (8 GB / 4 CPUs, the operator's cap) with the 5.19.0 site.

## Review Focus

1. **The zero-copy arm reads what the engine will read.** `probe_offload_moe.cl` is `q4_moe.cl` with exactly the two
   macro lines replaced (the generator refuses anything else; the drift test reruns it); its all-VRAM arm with an
   identity table is bitwise `q4_moe_M1_E512_T10_D2560_I640_SH4`'s output, every replay. A miss is a whole expert's
   layout-1 blocks in host USM, read by the same work-groups in the same order.
2. **Mixed launches answer max-or-sum.** For 1 / 2 / 3 of the 10 routed slots on host, the launch time against the
   all-VRAM launch and against the all-host launch decides between spec 22 §2's serial and overlapped formulas; P0.7
   uses the law measured, and prints which.
3. **The visibility hazard is tested, not assumed.** In a replayed list run 10^4 times, (a) the output is compared every
   replay; (b) between replays the host rewrites a host-USM expert with a new pattern and the next replay must see it;
   (c) the device-side page tags of every pinned range are read back. Any mismatch is reported with the replay index and
   page, and stops the arm.
4. **Held-out profiles.** P0.6's static-fill hit rate is scored on transcripts not used to build the profile (halves
   by a fixed hash of the source name; the A4 scenarios split across both halves); the in-sample number is printed
   beside it, never instead of it.
5. **The pruning curve's mask is the engine's rule.** The reference's masked router sets the dropped experts' logits
   to -inf **before** the fp32 softmax over 512, takes the top 10 of what remains and renormalises over them - the
   router's own rule on the kept set (spec 22 §4b). Tested: an all-kept mask is bitwise the unmasked run; a mask that
   drops only experts the unmasked run never selected is bitwise the unmasked run.
6. **The stopping rule is computed, not argued.** `offload_curve.py project` prints `STOP` or `GO-CANDIDATE` from the
   measured numbers and the rule's own inputs (two cards, 32k, the best fill, the measured mixed-launch law, P measured
   per card); the operator's go is separate.

---

### Task 1: the expert budget in the planner (host)

**Files:**
- Modify: `src/runtime/qwen4exp/qwen4exp_sizes.{h,cc}`, `tests/runtime/qwen4exp_plan_test.cc`
- Test: `qwen4exp_plan_test` (host)

**Interfaces:**
- Consumes: 21b's `plan`, `pp_split`, `require_fits`, `loader/qwen4exp_layout.h` (`q4_gate_up_block_bytes`,
  `q4_down_block_bytes`, `q4_mtp_layer_bytes`).
- Produces (used by Task 4's projection and by plan 22b's cache sizing):

```cpp
namespace runtime::qwen4exp {
// Spec 22 §2 (derived): what each device has left for ROUTED experts once everything else of the plan - the
// non-routed weights, RoPE, KV + indexer keys, state, decode (and with `prefill` the prefill) scratch, the link -
// and the reserve are placed. The MTP head's experts count as non-routed (decision 6's proposal: resident).
struct ExpertBudget {
  uint32_t max_len = 0, devices = 0;
  bool mtp = false, prefill = false, int8_head = false;
  std::array<size_t, kPpDevices> non_routed{};   // per device: plan(...) minus its layers' gate_up + down blocks
  std::array<size_t, kPpDevices> cache{};        // max(0, capacity - reserve - non_routed)
  std::array<uint32_t, kPpDevices> first{}, end{};
  size_t routed_total = 0;                       // layers x 512 x (1,740,800 + 870,400)
  double f() const;                              // min(1, sum(cache) / routed_total): the resident fraction
};
ExpertBudget expert_budget(const model::Qwen4ExpDesc& d, const model::Q4Placement& p, uint32_t max_len,
                           bool int8_head, bool mtp, bool prefill,
                           const std::array<size_t, kPpDevices>& device_bytes, size_t reserve);
std::string budget_json(const ExpertBudget& b);  // one line: {"max_len":..,"devices":..,"cache":[..],"f":..,...}
}
```

- [ ] **Step 1: the failing test.** `qwen4exp_plan_test` gains a budget case: for the published 48 layers, ours and
  Intel's forms (21b's `Q4Placement::two(d, pp_split(...).split)`, 32.53 GB cards, the 1.5 GB reserve, int8 head),
  at max_len 4096 / 32768 / 131072 / 262144, with and without the MTP head and the prefill scratch: `non_routed +
  routed blocks == plan()`'s per-device total byte for byte; `cache` = capacity - reserve - non_routed; one card
  (`Q4Placement::one`) likewise. It prints every case as `budget-json {...}` and, with `--budget-json FILE`, writes
  them (one per line) to FILE. FAIL (no function).
- [ ] **Step 2: implement**; PASS on the Mac (`ctest --preset mac-host -R '^qwen4exp_plan_test$'`). The printed f for
  ours at 32768 on two cards with the head and the prefill scratch is spec 22 §2's 0.83 or the record says by how much
  it differs and why (derived; no bar). Commit `git commit -S -m "runtime: the routed-expert budget per device -
  what the plan leaves for spec 22's cache (spec 22a)"`.

### Task 2: the probe binary and its kernels (Mac: syntax and an indicative run)

**Files:**
- Create: `tools/probe/probe_offload.cc`, `tools/probe/probe_offload.cl`, `tools/probe/mk_offload_kernel.py`,
  `tools/probe/probe_offload_moe.cl` (generated, committed), `tools/probe/offload_links.sh`,
  `tools/mac/clrun/offload_run.cc`
- Modify: `tools/probe/CMakeLists.txt` (a block `# --- spec 22 P0 (2026-10-10): the expert-offload probe ---`, inside
  `if(B70_Q4EXP)`: the two kernels, `probe_offload` linking `b70_l0` and `b70_model`, depending on 21c's
  `q4_moe_M1_E512_T10_D2560_I640_SH4` target), `tests/CMakeLists.txt` (`probe_offload_kernel_drift_test`: `python3
  tools/probe/mk_offload_kernel.py --check`, label host), `tools/mac_check.sh` (section 5's loop gains `offload`)
- Test: `probe_offload_kernel_drift_test` (host), `offload_run` (Mac GPU, indicative)

**Interfaces:**
- Consumes: `l0::Context` (+ the view constructor for device 1), `l0::Mem` (`MemKind::Device` / `Host`),
  `l0::CmdList::immediate` / `regular` with an ordinal, `l0::Queue`, `l0::SyncEvent`, `Fence::wait_for`;
  `loader/qwen4exp_layout.h` and `model::qwen4exp()` for every size; 21c's q4_moe binary and route-row layout
  (`kernels::qwen4exp::route`: ids at word 0, weights at 16, the shared gate at 26).
- Produces:
  - `mk_offload_kernel.py [--check]`: reads `src/kernels/qwen4exp/q4_moe.cl`, replaces exactly the two lines
    `#define Q4_EXPERT_GU(base, id) ...` / `#define Q4_EXPERT_DN(base, id) ...` with the table form
    `#define Q4_EXPERT_GU(base, id) ((__global const uint*)(intptr_t)((__global const ulong*)(base))[(id)])` (DN the
    same; `base` is then the layer's u64 table: gate||up pointers `[0, 512)`, down pointers `[512, 1024)` - the down
    launch is bound at `table + 512`), prefixes a provenance comment (the source's sha256), and writes
    `tools/probe/probe_offload_moe.cl`; it exits non-zero if either line is missing or appears twice; `--check`
    compares instead of writing.
  - `probe_offload.cl`: `po_page_check(ptrs, first_page, out, pages)` (one lane per 2 MiB page: the first u64 of each
    page of each range through a u64 pointer table - `q4_ple_check`'s form for any number of ranges),
    `po_page_tag(ptr, pages, salt)` (a device arena's tags written by the device), `po_read_sum(ptr, words, out)` (a
    wide `uint4` read and fold: the streaming ceiling, `probe_bw`'s pattern on a host or device pointer), `po_stage(route, table, bounce)` (arm K: per routed slot whose entry is not in the VRAM range given, the expert's gate‖up and down blocks copied into bounce slot `slot` with `uint4` reads).
  - `probe_offload <mode> [--dev 0|1|both] [options]`, every result one line `po <mode> dev=<d> <key>=<value> ...`:

| mode | spec | what it times / checks |
|---|---|---|
| `copy` | P0.2 | pinned host -> device copies of 0.87 MB (one down), 1.74 MB (gate‖up), 2.61 MB (an expert), 16 / 64 / 256 experts as one copy and as per-expert copies, 1 GB; on an immediate list and inside a regular list replayed; on the compute ordinal and on the device's copy-only queue group (found with `zeDeviceGetCommandQueueGroupProperties`: `COPY` without `COMPUTE`; printed, `none` if absent); device 0, device 1, both concurrently (one thread per device, a start barrier, the aggregate and each); the first-byte latency of one 2.6 MB copy and of a 4-byte copy (submit to fence) |
| `zc` | P0.3 | M = 1, a replayed list of P launch pairs (gate‖up then down, each pair routed to 10 distinct experts of its own 512-expert layer image, P = 1 / 2 / 7 / 26 pairs ≈ 10 / 20 / 70 / 260 experts read); arms: **A** 21c's binary over a VRAM layer (control), **B** the table kernel with an identity table over the same VRAM (the indirection's own cost), **C** every routed slot's table entry in host USM, **D1-D3** 1 / 2 / 3 of the 10 slots on host, the rest VRAM, **S** the misses of D1-D3 / C staged by the copy queue into a VRAM bounce buffer (an event), then B's kernel - decision 1's comparison as spec 22 §3 writes it (the host knows the misses here: an upper bound, since the engine's list has no per-layer host step), **K** the realisable staging inside a list - a gather kernel (`po_stage`, wide reads) copies the slots whose table entries are host pointers into a VRAM bounce buffer, then B's kernel reads the bounce slots; `po_read_sum` over the same bytes as the ceiling; per card, then both cards concurrently; A-vs-each as interleaved pairs; outputs compared bitwise with A's every run |
| `replay` | P0.3 | arm D2 and arm C in a regular list replayed 10^4 times: `y` read back and compared every replay; every 100th replay the host rewrites one host expert with a new pattern (and its expected `y`, computed once by arm A on a VRAM copy) and the next replay must match the new one; per card and both |
| `hostbw` | P0.4 | host memory bandwidth: a STREAM-triad over 3 x 2 GiB on 1 and on all hardware threads with the GPUs idle; then arm C on both cards concurrently (the zero-copy aggregate) with the CPU idle, with one thread running a decode-sized host task (a 64 MiB working set copied in a loop - PROPOSED stand-in for the server's per-token host work), and with the all-thread triad |
| `pin` | P0.5 | the largest pinned host allocation as one range (doubling from 1 GiB, then bisection to 1 GiB resolution, never above `MemAvailable - 16 GiB`), then as 48 per-layer ranges of 1,336,934,400 B plus the PLE's 16 + 16 shard sizes of 21b's real table (`loader/qwen4exp_ple.h`'s rule): where it fails, MemAvailable before / after, seconds per GB pinned; each range tagged per 2 MiB page by the host and read back by `po_page_check` on each card |
| `alias` | P0.5 | a model-sized allocation history on device 0 (allocate and free 18 layers' allocations at 21b's sizes, then the int8 head and the PLE pointer table), then a device arena of `--arena-gb 22` (one allocation; `l0::Mem` asks relaxed limits): `po_page_tag` writes a salted tag at every 2 MiB page, the host reads every page's tag back through a copy, and a second salt confirms; any page holding another page's tag is an alias, printed with both addresses; repeated on device 1 |
| `hold` | P0.1 | arm `copy`'s 1 GB copy in a loop for `--seconds S` on the named cards (for `LnkSta` under load) |

  - `offload_links.sh [--under-load S]`: for 04:00.0 and 08:00.0, every bridge above each up to its root port (from
    `/sys/bus/pci/devices/<bdf>`'s path), `lspci -vv -s <bdf>`'s `LnkCap` / `LnkSta` (generation and width), idle and -
    with `--under-load` - while `probe_offload hold` runs; `lscpu`; `sudo -n dmidecode -t memory` (prints the command
    to run by hand when `sudo -n` fails); the derived host RAM ceiling (channels x MT/s x 8 B); one line `links: dev0
    Gen<g> x<w>, dev1 Gen<g> x<w>` and `STOP: <card> trains below Gen3 x8` when it does.

- [ ] **Step 1: the generator and its drift test** (host): `mk_offload_kernel.py`, the generated file,
  `probe_offload_kernel_drift_test` registered; `ctest --preset mac-host -R probe_offload_kernel_drift_test` PASS; a
  hand edit of the generated file makes it FAIL (checked once, reverted). Commit `git commit -S -m "probe: the
  expert kernels read through a pointer table - q4_moe.cl with its two address lines swapped, drift-checked (spec 22a)"`.
- [ ] **Step 2: `probe_offload.cc` and `probe_offload.cl`, the CMake block.** Sizes only from the layout header; the
  route rows written by the host (the 10 ids of each pair, weights 0.1 in bf16, the shared gate 0); every arm's
  `y` read back; every wait bounded. Commit `git commit -S -m "probe: probe_offload - copy-engine rates, zero-copy
  expert reads mixed with VRAM, replayed visibility, host RAM under load, pinned capacity, the alias check (spec 22a)"`.
- [ ] **Step 3: Mac gates.** `tools/mac_check.sh --base main --quick` exit 0 (Level Zero syntax of the new source;
  cmdlines +0 / -0 / ~0); by hand, recorded in the commit message: `clang -x cl -cl-std=CL3.0 -fsyntax-only -include
  tools/mac/opencl/intel_shim.h` over both `.cl` files with the q4_moe defines of 21c's `_SH4` line and with
  `-Dcl_intel_subgroups` unset; `offload_run` on the Mac GPU: the table kernel over (identity, a permuted slot order,
  host-allocated blocks) bitwise `q4_moe.cl`'s plain-load build at the real shapes - `offload_run: q4_moe table exact`.
  Commit `git commit -S -m "mac: offload_run - the table-addressed expert kernels bitwise the direct ones (spec 22a)"`.
- [ ] **Step 4: `offload_links.sh`** (`bash -n`, and its parser run over a saved `lspci -vv` sample in a heredoc-free
  fixture `tools/probe/testdata/lspci_b70.txt` - a recorded or hand-written sample, said which). Commit `git commit -S
  -m "probe: offload_links.sh - the B70s' PCIe links, bridges, root ports and host memory channels (spec 22a)"`.

### Task 3: the trace analyser and the profile (P0.6, Mac)

**Files:**
- Create: `tools/oracle/offload_curve.py`, `tools/oracle/test_offload_curve.py`
- Test: `test_offload_curve.py` (in `agnes-ref-img`)

**Interfaces:**
- Consumes: `<traces>/<name>.routes.safetensors` (21a's `TRACE_FORMAT`: `ids` i32 [T][L][10] ascending, `p` f32,
  `onorm` f32, `mtp_ids` / `mtp_p` / `mtp_onorm` [T][10], row T-1 of the head -1 / 0; metadata `layers`, `experts`,
  `top_k`, `source`); Task 1's `budget-json` lines; P0.2's copy time per expert (a number on the command line).
- Produces (`offload_curve.py <cmd> --traces DIR ...`; every table also written as JSON beside its text for the record):
  - `counts`: per (layer, expert) routed counts, per source and pooled; the distinct experts per layer per source;
    the Gini and the top-k share per layer.
  - `curve [--split hash|none] [--f 0.30:1.00:0.01]`: the hold-out split (Review Focus 4: sources by `sha256(name)`
    parity, A4 scenarios balanced across halves; the other half scores); h(f) for **uniform** bytes per layer and for
    **per-layer budgets** from the curves (a greedy fill by the marginal held-out hits per byte over all layers -
    every expert of a layer has the same bytes, so per byte = per slot), in-sample and held out, per source class (A4,
    agentic, code, prose, opencode); the f at which each policy reaches h = 0.90 / 0.95 / 0.99.
  - `swap --f F --S 1,4,16,64 --B <MB list> --copy-ms X --step-ms Y`: the adaptive simulation - every S steps up to B
    bytes of the window's most-routed non-resident experts replace the least-routed residents (window = the last W
    steps, W in {64, 256, 1024}); an incoming expert serves from the step after `ceil(B / copy rate / step time)` steps
    (P0.2's measured copy time, decode's step time from P0.7's model), a victim's slot is reused only after its last
    in-flight step; h against the static fill at the same f; the gain per (S, B) - decision 3's evidence.
  - `mtp --K 1,2,3,4 --f F`: per layer the union of experts over M = K + 1 consecutive rows (teacher-forced rows stand
    for accepted drafts) and its iid prior; the misses per verify step at the static fill; the MTP head's own routes
    (`mtp_ids`): distinct experts, and the hit rate if they were cached with the rest (decision 6).
  - `prefill --chunk 2048,4096 --f 0.6:0.9:0.05`: unique experts per layer per chunk, the streamed bytes per chunk
    at f under the static fill (the experts the chunk touches that are not resident x 2,611,200 B).
  - `profile --out FILE [--sources ...]`: the routing profile 22b's loader reads (decision 8's proposal): safetensors
    `counts` u32 [L][512], `mtp_counts` u32 [512], metadata `{model, checkpoint, checkpoint_revision, traces (names +
    sha256), sources, tokens, format: "spec22-profile-1"}`.
- [ ] **Step 1: the failing tests** `test_offload_curve.py` on synthetic traces built in the test with known answers:
  a trace where layer l routes only experts [0, 10 + l): h(f) is analytic; a hold-out split that sees disjoint
  experts scores h = 0 held out and 1 in sample; a swap simulation where the copy never lands (`--copy-ms` = inf)
  equals the static fill exactly; M = K + 1 unions on repeated rows equal 10, on disjoint rows 10 M; prefill chunks of
  one repeated row stream 10 experts a layer. FAIL.
- [ ] **Step 2: implement**; PASS. Then on the tiny model's own traces: `qwen4exp_ref.py trace` on a 16-expert /
  top-4 tiny made by `qwen4exp_make_tiny.py` (the downloaded tiny has 4 experts / top-2: too few for a curve) over
  `q4exp_agentic` and `code` (truncated to 512 ids), then every subcommand runs, `counts` sums to T x L x k and
  `profile` round-trips. Commit `git commit -S -m "oracle: offload_curve.py - hit-rate curves from 21a's traces
  (static held out, swaps, MTP verify unions, prefill bytes) and the routing profile (spec 22 P0.6)"`.

### Task 4: the pruning curve's tools and the projection (P0.8, P0.7, Mac)

**Files:**
- Modify: `tools/oracle/qwen4exp_ref.py` (`--expert-mask FILE` on `run` / `ppl` / `trace`; a `maskeval` subcommand),
  `tools/oracle/test_qwen4exp_ref.py`, `tools/toolcall/oracle_generate.py` (`--expert-mask FILE`, qwen4exp only; the
  output names gain `.mask-<tag>`), `tools/toolcall/test_oracle_generate.py`, `tools/toolcall/a4_ref.sh`
  (`qwen4exp ref` honours `MASK=FILE`), `tools/box_validate/qwen4exp_oracle.sh` (modes `prune`, `prune-a4`),
  `tools/oracle/offload_curve.py` (subcommands `reap`, `prune-report`, `project`), `tools/oracle/test_offload_curve.py`
- Test: `test_qwen4exp_ref.py`, `test_oracle_generate.py`, `test_offload_curve.py`

**Interfaces:**
- **The mask file** (spec 22 §4b; 22m's loader reads the same file): safetensors `keep` u8 [L][512] (1 = kept),
  `mtp_keep` u8 [512] (all ones unless asked: 22m's open point), metadata `{model, checkpoint_revision, criterion:
  "reap" | "freq", kept_fraction, per_layer: "uniform", calibration (source names + sha256 of each trace), format:
  "spec22-mask-1"}`; every layer keeps >= `top_k` experts (refused otherwise).
- `offload_curve.py reap --traces DIR --sources A4,agentic,opencode[,code] --kept 0.5,0.625,0.75,0.875 --out DIR
  [--criterion reap|freq] [--holdout-a4]`: score(l, e) = Σ over the calibration tokens routed to e of `p x onorm`
  (REAP; `freq` counts tokens - the weaker proxy, written for comparison); per layer keep the top `round(512 x kept)`
  by score (ties to the lower id); writes `mask-<criterion>-<kept>.safetensors` and the overlap between the REAP and
  frequency masks per layer. `--holdout-a4` calibrates without the A4 scenarios (the held-out arm; Review Focus 4's
  rule applied to P0.8).
- `qwen4exp_ref.py ... --expert-mask FILE`: a forward hook on every `layer.mlp.gate` that replaces its `(logits,
  scores, idx)` with the masked rule (Review Focus 5) from the same logits; the MTP head's router likewise with
  `mtp_keep`. `qwen4exp_ref.py maskeval <snapshot> --expert-mask FILE --golden <p>.golden.safetensors [--ple ...]`:
  teacher-forced over the golden's prompt and generated ids, against its stored logits rows: per row KL(full ||
  masked) in fp32 and argmax agreement, printed as `maskeval <p> kept=<f> kl_mean=.. kl_p99=.. argmax=..` and written
  as JSON. For the A4 set the same over each scenario's prompt + the full reference's continuation (`<name>.bf16.ids`).
- `offload_curve.py prune-report --dir <maskeval JSONs> --a4 <score.py outputs>`: the curve table - kept fraction x
  {KL mean / p99, argmax agreement on golden and on A4, A4 tool-call accuracy (score.py's match rate against the full
  reference) where generated, the experts' bytes at int4 g64, fits two cards whole (from Task 1's budget at 32k)}.
- `offload_curve.py project --budget FILE --p0 FILE --curve FILE [--accept FILE]`: P0.7's table (Task 5 Step 7).
- [ ] **Step 1: the failing tests.** `test_qwen4exp_ref.py`: on the 16-expert tiny, (a) an all-ones mask is bitwise
  the unmasked `run` (logits and every layer's H); (b) a mask dropping only experts the unmasked run never routed to
  (read from its own trace) is bitwise the unmasked run; (c) the hook's `(scores, idx)` equal `router()` restated with
  the dropped logits at -inf, including a planted tie at the 10th; (d) a mask keeping fewer than `top_k` in a layer is
  refused by name; (e) `maskeval` of an all-ones mask prints KL 0 and argmax 1.0. `test_oracle_generate.py`: the mask
  reaches the reference (the tiny's batched path, ids differ from unmasked when (b)'s complement is dropped).
  `test_offload_curve.py`: `reap` on a synthetic trace keeps the analytically top experts, ties to the lower id, and
  `--holdout-a4` changes the calibration set only. FAIL.
- [ ] **Step 2: implement**; PASS in `agnes-ref-img` with the 5.19.0 site. Commit `git commit -S -m "oracle: the
  masked reference (router logits -inf before the softmax, renormalised over the kept top-10) and REAP masks from 21a's
  traces (spec 22 P0.8)"`.
- [ ] **Step 3: `qwen4exp_oracle.sh prune` / `prune-a4`** (`DRY_RUN=1` prints each): `prune` = `reap` at the four
  fractions (both criteria, with and without `--holdout-a4`), then `maskeval` for each REAP mask over `q4exp_short`,
  `q4exp_4k`, `q4exp_agentic` (and `q4exp_32k` with `PRUNE_32K=1`) and over the 36 A4 scenarios' teacher-forced
  continuations -> `<data>/oracle-out-q4exp-prune/`; `prune-a4` = `a4_ref.sh qwen4exp ref` with `MASK=` each REAP
  mask whose experts fit two cards whole (Task 1's budget: the kept experts' bytes <= the two-card cache at 32k with the
  head) -> `<data>/oracle-out-q4exp-a4/<name>.mask-<tag>.{ids,txt}`, then `score.py` against the full reference.
  Resumable per prompt / scenario, `Q4_REF_MIN_GB` as the other modes. Times ESTIMATED in its DRY_RUN table from row
  30's per-prompt walls. Commit `git commit -S -m "box_validate: qwen4exp_oracle.sh prune / prune-a4 - spec 22 P0.8's
  masked runs (spec 22a)"`.
- [ ] **Step 4: `project`** with its tests (synthetic inputs: P = 12 GB/s, BW = 590 GB/s, D = 3.712 GB and h from
  spec 22 §2's table reproduce its serial and overlapped t/s columns to 0.1; the stopping line flips at 40 t/s). The
  projection: decode t/s at max_len 4096 / 32768 / 131072 = f from Task 1's budget (two cards, the head, the prefill
  scratch; ours and Intel's forms), h from P0.6's best held-out policy at that f (static per-layer, or static + swaps
  when the simulation gains), the law P0.3 measured (sum: serial; max: overlapped; between: the measured mix as an
  interpolation, printed), P per card from P0.3 arm C and BW from row 32's measured per-layer rate when present (else
  the 590 GB/s roofline at 77.6 %, spec 21 §3, marked); with MTP at K = 1..3 from `r34.accept` (else the 0.5-0.8 range):
  a verify step's misses from P0.6's unions; prefill t/s at chunk 2048 (and 4096 as a derived row: 21d's walk stops at
  2048) - the bus term (streamed bytes / P0.2's copy rate, both cards concurrently under spec 16c's pipeline) against
  row 33's measured compute per chunk; one card (f from the one-card budget) as a row. The last line is `STOP` or
  `GO-CANDIDATE` (Review Focus 6). Commit `git commit -S -m "oracle: the offload projection and the stopping rule
  (spec 22 P0.7)"`.

### Task 5: the box (row 36) and the record

**Files:**
- Modify: `docs/superpowers/plans/box-validation-queue.md` (row 36), `tools/box_validate/stages.sh` (a `row 36` block
  after row 34's - row 35 belongs to the W4A4 probe and is not touched), `tools/box_validate/data.sh` (`have
  oracle_q4exp_prune`: the maskeval JSONs; `have probe_offload_bin`), `tools/box_validate/test_box_validate.py` only if
  a new stage kind needs it, `docs/box-day-plan.md` (a spec 22 session after spec 21's rows: cards, RAM rule, order)
- Create: `docs/probe-offload-<date>.md`
- Modify (at the end): `docs/superpowers/specs/2026-10-09-spec22-moe-expert-offload-design.md` (§1's facts: the
  measured links, P, the PLE rate from `r32.ple_rate`; a "P0 as measured" section; decisions 1-9 each with its evidence
  and the proposal), `docs/README.md` (the probe doc's line), this plan's status line

The stages (`stage <id> 36 <kind> <cpu|gpu> <have> <after> "..."`):

| stage | kind | needs | what |
|---|---|---|---|
| `r36.k0` | default cpu | g0 | G0 trivially: kernel_cmdlines +0 / -0 / ~0; Task 1 changes no device code |
| `r36.host` | default cpu | - | `qwen4exp_plan_test` (the budget lines -> `$OUT/budget.json`), `probe_offload_kernel_drift_test`, `test_offload_curve.py` |
| `r36.build` | default gpu | r32.k1 | `probe_offload` and its two kernels built (their first ocloc compile) |
| `r36.links` | default cpu | - | `offload_links.sh` idle (P0.1); `dmidecode` is manual when `sudo -n` fails (the command printed); **a STOP line stops the row** |
| `r36.links_load` | default gpu | r36.links, r22.p1 | `offload_links.sh --under-load 20` (both cards copying) |
| `r36.copy` | default gpu | r36.build, r22.p1 | `probe_offload copy --dev 0`, `1`, `both` (P0.2) |
| `r36.zc` | default gpu | r36.copy | `probe_offload zc --dev 0`, `1`, `both` (P0.3: arms A, B, C, D1-D3, S, the ceiling; interleaved pairs) |
| `r36.replay` | default gpu | r36.zc | `probe_offload replay --dev both --n 10000` (P0.3's visibility) |
| `r36.hostbw` | default gpu | r36.zc | `probe_offload hostbw` (P0.4) |
| `r36.pin` | default gpu | r36.build | `probe_offload pin` (P0.5; SKIP "host RAM" when `free -g` < 110 GB available or a CPU oracle is live) |
| `r36.alias` | default gpu | r36.build | `probe_offload alias --arena-gb 22 --dev 0`, `1` (P0.5) |
| `r36.curve` | default cpu | oracle_q4exp_traces | `offload_curve.py counts / curve / mtp / prefill`, then `swap` with `r36.copy`'s per-expert copy time (P0.6); `profile` -> `$DATA/oracle-out-q4exp-traces/profile.safetensors` |
| `r36.reap` | default cpu | oracle_q4exp_traces | `offload_curve.py reap` (the masks, both criteria, held-out arm) -> `$DATA/oracle-out-q4exp-prune/masks/` |
| `r36.prune` | optin cpu | q4exp_intel, oracle_q4exp, r36.reap | `qwen4exp_oracle.sh prune` (P0.8's KL / argmax; HOURS, ESTIMATED) |
| `r36.prune_a4` | optin cpu | q4exp_intel, oracle_q4exp_a4, r36.reap | `qwen4exp_oracle.sh prune-a4` (A4 accuracy at the fractions that fit; HOURS) |
| `r36.project` | default cpu | r36.host, r36.zc, r36.curve | `offload_curve.py prune-report` (when r36.prune ran) and `project` -> the STOP / GO-CANDIDATE line |

- [ ] **Step 1: the row and its stages** (`python3 tools/box_validate/test_box_validate.py` passes; `tools/box_validate.sh
  --dry-run --only r36` prints every command; `--dry-run --with r36.prune,r36.prune_a4` the opt-ins). The rownotes:
  the data each stage needs (rows 30, 32, 34), the RAM rule (pinned arms never beside a CPU oracle), the session
  (both cards idle, RECORD grade). Commit `git commit -S -m "box: queue row 36 - spec 22 P0, the expert-offload probe
  (spec 22a)"`.
- [ ] **Step 2 (box): P0.1** - `r36.links`, `r36.links_load`. A card below Gen3 x8: stop and report (spec 22 §3).
- [ ] **Step 3 (box): P0.2-P0.5** - `r36.copy`, `r36.zc`, `r36.replay`, `r36.hostbw`, `r36.pin`, `r36.alias`; any
  mismatch in `replay` or `alias` is reported first, with its replay / page.
- [ ] **Step 4 (box CPU): P0.6** - `r36.curve`; the cross-check is `r32.partial`'s routing diagnostic (Dependencies).
- [ ] **Step 5 (box CPU): P0.8** - `r36.reap`, then `--with r36.prune,r36.prune_a4` (hours; one CPU oracle at a time,
  never beside Step 3's pinned arms).
- [ ] **Step 6: the record** `docs/probe-offload-<date>.md`: P0.1-P0.8's tables (the links, copy rates by size /
  list / ordinal / card / both, zero-copy arms with their paired ratios against A, the mixed-launch law, staging vs
  zero-copy, replay and alias results, host bandwidth idle and loaded, pinned capacity and its failure point, the h(f)
  curves uniform vs per-layer, held out vs in sample, swaps' gain per (S, B), the verify unions against the iid
  prior and K2's measured shape, prefill bytes, the pruning curve), with commit, dates, `uptime`, idle grade.
  Commit `git commit -S -m "probe: spec 22 P0 measured - links, copy and zero-copy rates, capacity, hit-rate and
  pruning curves (spec 22a)"`.
- [ ] **Step 7: the projection and the decisions** - `r36.project`; the record's last section: the projection table
  (decode 4k / 32k / 128k with and without MTP, prefill 2048 / 4096, one card), the STOP / GO-CANDIDATE line, and per
  decision 1-9 the evidence and the proposal (1 the miss path from arms C / D vs K, with S as the bound; 2 uniform vs per-layer; 3 swaps
  from `swap`; 4 the host RAM budget from `pin`; 5 the context default from the projection; 6 the head's experts from
  `mtp`; 7 prefill streaming from `prefill` and `copy`'s concurrent rate; 8 the profile's source - this row's
  `profile.safetensors`; 9 the coding mask's kept fraction from the pruning curve against the cache's projected
  speed). If STOP: the tier's levers first (HC int8, int8 KV, a smaller context), as spec 22 §3 says. Spec 22's status
  line and §1 updated; `docs/README.md`. Commit `git commit -S -m "spec 22: P0's record and projection - the go /
  no-go evidence and decisions 1-9's proposals (spec 22a)"`.

**Gate for the plan:** Mac - `qwen4exp_plan_test` (budget = plan byte for byte), the drift test, `offload_run` exact,
`test_offload_curve.py`, `test_qwen4exp_ref.py` (the mask identities), `test_oracle_generate.py`,
`test_box_validate.py`, `mac_check.sh --quick` exit 0. Box - every P0 table filled from measured runs (or a SKIP with
its reason), no unexplained mismatch in `replay` / `alias`, the projection's STOP / GO-CANDIDATE line, the record and
spec 22's decisions section written. The operator's go / no-go follows; this plan does not decide it.
