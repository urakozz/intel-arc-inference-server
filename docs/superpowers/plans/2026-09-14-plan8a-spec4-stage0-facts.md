# Spec 4 / Stage 0 - K2-Horizon facts before building Implementation Plan

**Status (2026-09-14): written, NOT dispatched.** Implementation waits for the operator's ruling on the prefill GEMM direction; these plans may be adjusted after it.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Settle, by measurement, the four facts spec 4 builds on - the checkpoint passes our quant checks, the device accepts the allocations we plan, K2's golden prompts exist as K2 token ids with an explicit BOS decision, and the fixed cost of one small launch is known - plus the operator-gated oracle feasibility check.

**Architecture:** One host+device probe (`tools/probe/k2_stage0_probe`) reusing the loader's public helpers for the quant checks and plain Level Zero allocations for the size checks; one probe for launch overhead reusing `gemv.cl` under a probe-only variant name; one tokenizer script run in the box's Python venv; one container import check that runs only on the operator's go. Results go to one facts document the later plans cite.

**Tech Stack:** C++17 + Level Zero (`src/l0`), `loader::QuantConfig`, `loader::SafetensorsSet`, ocloc AOT via `add_ocloc_kernel`, Python 3.14 with `transformers 5.14.1` in `~/auto-round/.venv` on the box, the reference image `vllm-xpu-env-next-p314-t215-vxkp0`.

**Spec:** `docs/superpowers/specs/2026-09-14-spec4-k2-horizon-decode-core-design.md` (§1, §3.4, §4 "Prompts" and "CPU oracle", §5 stage 0, §8 risks 1/3/4/5)

## Global Constraints

- Branch `spec1.7-codex-exp`; never push. Every commit message ends with `Claude-Session: `.
- The Mac never compiles: build and test only via `tools/box.sh` (JOBS 44). Box `user@box`; copy files with scp, never pipe heredocs over ssh.
- **Anything on the box longer than about a minute launches detached** (`setsid nohup <cmd> > $HOME/<name>.log 2>&1 < /dev/null &`) and is polled - the box's WiFi drops several times an hour.
- **Never start a container without the operator's explicit go.** Never kill a process you did not start. Never delete anything on the box.
- GPU work with `ZE_AFFINITY_MASK=1` unless a task says otherwise.
- Checkpoint: `urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ`, snapshot `c0fd9971ad6d3bb4caedf726e798a02b8df9bcb5` (refs/main), at `/home/user/.cache/huggingface/hub/models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ/snapshots/c0fd9971ad6d3bb4caedf726e798a02b8df9bcb5`.
- Do not modify anything the 27B uses: `src/loader/*.cc`, `src/model/qwen35.*`, `src/runtime/*`, existing `src/kernels/*.cl` and existing test files stay byte-identical. New files and appended CMake blocks only.
- Every number labelled measured / derived / estimated; never two values for one quantity.

---

## File structure

| path | responsibility |
|---|---|
| `tools/probe/k2_stage0_probe.cc` | quant config + invariants on the K2 checkpoint; device max allocation; allocate-and-touch the planned KV and expert buffers |
| `tools/probe/k2_launch_probe.cc` | fixed cost per launch of a tiny int4 GEMV in one replayed list |
| `tools/probe/CMakeLists.txt` (append) | the two probe targets and the probe-only GEMV variant |
| `tools/oracle/tokenize_k2.py` | K2 tokenizer ids for the three golden texts, with the BOS decision recorded |
| `tests/golden/k2/prompts/{prose,code,cjk}.ids` | committed K2 prompt ids |
| `tools/oracle/k2_import_check.py` | container check: the model's remote code imports and its parameter count matches |
| `docs/k2-stage0-facts-2026-09-14.md` | every measured fact, cited by plans 8b-8e |

---

### Task 1: The K2 checkpoint passes our quant checks, and the device takes our allocations

**Files:**
- Create: `tools/probe/k2_stage0_probe.cc`
- Modify: `tools/probe/CMakeLists.txt` (append one block)
- Create: `docs/k2-stage0-facts-2026-09-14.md` (§1 and §2)

**Interfaces:**
- Consumes: `loader::resolve_snapshot(const std::string&)`, `loader::QuantConfig::parse(const common::json::Value&)`, `common::json::parse(const std::string&)`, `loader::SafetensorsSet(const std::string&)`, `loader::assert_quant_invariants(const SafetensorsSet&) -> QuantScan`, `l0::Context(uint32_t)`, `l0::Context::props().maxMemAllocSize`, `l0::Mem(Context&, MemKind, size_t)`, `l0::CmdList::immediate(Context&)`, `CmdList::copy(dst, src, bytes)`.
- Produces: the facts plans 8b and 8e rely on - whether `QuantConfig::parse` accepts K2's config unchanged, the device's `maxMemAllocSize`, and whether two 1,610,612,736 B KV buffers plus one 196,608,000 B expert buffer can coexist.

- [ ] **Step 1: Write the probe**

```cpp
// tools/probe/k2_stage0_probe.cc - spec 4 stage 0, facts (a) and (b).
// (a) K2's quantization_config passes loader::QuantConfig::parse and every
//     qzeros/g_idx/scales tensor passes assert_quant_invariants, unchanged.
// (b) The device's maxMemAllocSize, and whether the allocations spec 4 §3.4
//     plans can be made AND coexist: KV K and V at max_len 16384
//     (48 x 8 x 128 x 16384 x 2 B = 1,610,612,736 B each) and the largest
//     per-layer expert buffer (fused gate||up weights, 100 x 1,966,080 B).
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common/json.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: k2_stage0_probe <snapshot-or-repo>\n");
    return 2;
  }
  const std::string snap = loader::resolve_snapshot(argv[1]);
  std::printf("snapshot %s\n", snap.c_str());

  // (a) quant config + invariants
  std::ifstream cf(snap + "config.json");
  if (!cf) { std::fprintf(stderr, "cannot read %sconfig.json\n", snap.c_str()); return 1; }
  std::stringstream cs;
  cs << cf.rdbuf();
  const loader::QuantConfig qc = loader::QuantConfig::parse(common::json::parse(cs.str()));
  std::printf("quant: bits %u group %u sym %d desc_act %d (declared %d) method '%s' "
              "packing '%s' dynamic rules %zu\n",
              qc.bits, qc.group_size, int(qc.sym), int(qc.desc_act), int(qc.desc_act_declared),
              qc.quant_method.c_str(), qc.packing_format.c_str(), qc.dynamic_rule_count);
  loader::SafetensorsSet set(snap);
  const loader::QuantScan scan = loader::assert_quant_invariants(set);
  std::printf("invariants: OK over %zu tensors; %zu subnormal scales; %zu g_idx tensors\n",
              set.tensors().size(), scan.subnormal_scales, scan.g_idx_tensors);

  // (b) allocations
  l0::Context ctx(0);
  std::printf("device '%s' maxMemAllocSize %llu B\n", ctx.name().c_str(),
              (unsigned long long)ctx.props().maxMemAllocSize);
  const size_t kKv = size_t(48) * 8 * 128 * 16384 * 2;   // 1,610,612,736
  const size_t kGu = size_t(100) * 1966080;              // 196,608,000
  const std::vector<size_t> want = {kKv, kKv, kGu};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  std::vector<l0::Mem> held;
  held.reserve(want.size());
  std::vector<uint8_t> pat(4096), back(4096);
  for (size_t i = 0; i < want.size(); ++i) {
    held.emplace_back(ctx, l0::MemKind::Device, want[i]);
    for (size_t j = 0; j < pat.size(); ++j) pat[j] = uint8_t((i * 131 + j) & 0xFF);
    // Touch both ends: an allocation that "succeeds" but cannot be written at
    // its far end is the failure this step exists to catch.
    for (size_t off : {size_t(0), want[i] - pat.size()}) {
      imm.copy(static_cast<uint8_t*>(held.back().ptr()) + off, pat.data(), pat.size());
      std::memset(back.data(), 0, back.size());
      imm.copy(back.data(), static_cast<uint8_t*>(held.back().ptr()) + off, back.size());
      if (std::memcmp(pat.data(), back.data(), pat.size()) != 0) {
        std::fprintf(stderr, "alloc %zu (%zu B): readback mismatch at offset %zu\n", i, want[i], off);
        return 1;
      }
    }
    std::printf("alloc %zu: %zu B OK (both ends written and read back; %zu held)\n", i, want[i],
                held.size());
  }
  std::puts("k2_stage0_probe OK");
  return 0;
}
```

- [ ] **Step 2: Register it** - append to `tools/probe/CMakeLists.txt`:

```cmake
# Spec 4 stage 0 (plan 8a Task 1): quant checks on the K2 checkpoint and the
# device allocations spec 4 §3.4 plans. Host + plain Level Zero; no kernels.
add_executable(k2_stage0_probe k2_stage0_probe.cc)
target_include_directories(k2_stage0_probe PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(k2_stage0_probe PRIVATE b70_loader b70_l0)
```

- [ ] **Step 3: Build** - `tools/box.sh build 2>&1 | tail -3`. Expected: the build ends `Built target k2_stage0_probe` with no warnings (the tree builds with `-Wall -Wextra -Werror`).

- [ ] **Step 4: Run it (short; foreground is fine)** - `tools/box.sh run 'ZE_AFFINITY_MASK=1 ./build/tools/probe/k2_stage0_probe /home/user/.cache/huggingface/hub/models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ/snapshots/c0fd9971ad6d3bb4caedf726e798a02b8df9bcb5'`.
  Expected: `quant: bits 4 group 64 sym 1 desc_act 0 (declared 1) method 'gptq' ... dynamic rules 45`, `invariants: OK`, a `maxMemAllocSize` line, three `alloc ... OK` lines, `k2_stage0_probe OK`.
  **If `QuantConfig::parse` throws** (for example on a key it does not know): stop, record the exact message in the facts doc - plan 8b then has to handle it and must not edit `src/loader/quant.cc`. **If an allocation fails:** record which one and the device's `maxMemAllocSize`; plan 8b then splits the KV per layer. Do not work around it here.

- [ ] **Step 5: Start the facts doc** with the verbatim output:

```markdown
# K2-Horizon stage-0 facts - 2026-09-14

Plan 8a. Every line is **measured** on `user@box`, device 1 (`ZE_AFFINITY_MASK=1`),
checkpoint snapshot `c0fd997`, unless labelled otherwise. Plans 8b-8e cite this file.

## 1. Quant checks (k2_stage0_probe, fact a)

<paste the quant: and invariants: lines verbatim>

Verdict: <"QuantConfig::parse and assert_quant_invariants accept the checkpoint unchanged" OR the exact exception>.

## 2. Allocations (k2_stage0_probe, fact b)

<paste the device and alloc lines verbatim>

Verdict: <"two 1.61 GB KV buffers and a 196.6 MB expert buffer coexist; plan 8b allocates KV as one K and one V buffer" OR "allocation <n> failed at maxMemAllocSize <x>; plan 8b allocates KV per layer">.
```

- [ ] **Step 6: Commit**

```bash
git add tools/probe/k2_stage0_probe.cc tools/probe/CMakeLists.txt docs/k2-stage0-facts-2026-09-14.md
git commit -m "probe(k2): stage-0 quant checks and allocation limits on the K2 checkpoint

Claude-Session: "
```

---

### Task 2: The fixed cost of one small launch

**Files:**
- Create: `tools/probe/k2_launch_probe.cc`
- Modify: `tools/probe/CMakeLists.txt` (append)
- Modify: `docs/k2-stage0-facts-2026-09-14.md` (§3)

**Interfaces:**
- Consumes: `src/kernels/gemv.cl` (entry point `gemv(w, scales, x, out)`, grid `(N/64, S)`, work-group 64), `common::Int4Gptq::random(K, N, seed)` and its `qweight`/`scales` vectors (`src/common/int4.h`), `l0::Queue`, `l0::Fence`, `l0::CmdList::regular`, `l0::Module`, `l0::Kernel`.
- Produces: µs per launch of an int4 GEMV at 2560 → 1536 (a MoE expert gate‖up) as launches per list grow - the fixed per-launch cost spec 4 §8 risk 1 names. Information for the tuning spec, **not a bar**.

- [ ] **Step 1: Write the probe**

```cpp
// tools/probe/k2_launch_probe.cc - spec 4 stage 0, fact (d): what one small
// launch costs. A K2 MoE expert's fused gate||up is an int4 GEMV at
// K = 2560 -> N = 1536: ~3.9 M int4 multiply-adds, tiny next to the 27B's
// shapes. With ~2,070 launches per K2 token, a fixed per-launch cost F
// dominates if F x 2,070 is a large share of the step. This measures F by
// timing ONE regular list holding L launches of that GEMV, for several L, all
// against the same weights (so the data is L2-warm and the slope is the launch,
// not the bandwidth).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "common/bf16.h"
#include "common/int4.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

int main() {
  constexpr uint32_t K = 2560, N = 1536;
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  const common::Int4Gptq w = common::Int4Gptq::random(K, N, 20260914);
  l0::Mem wbuf(ctx, l0::MemKind::Device, w.qweight.size() * 4);
  imm.copy(wbuf.ptr(), w.qweight.data(), w.qweight.size() * 4);
  l0::Mem sbuf(ctx, l0::MemKind::Device, w.scales.size() * 2);
  imm.copy(sbuf.ptr(), w.scales.data(), w.scales.size() * 2);
  std::vector<uint16_t> x(K);
  for (uint32_t k = 0; k < K; ++k) x[k] = common::f32_to_bf16(float(int(k % 17) - 8) * 0.0625f);
  l0::Mem xbuf(ctx, l0::MemKind::Device, x.size() * 2);
  imm.copy(xbuf.ptr(), x.data(), x.size() * 2);
  l0::Mem obuf(ctx, l0::MemKind::Device, size_t(N) * 4);

  l0::Module mod(ctx, kernels::path("probe_k2_gemv_M1_K2560_N1536_S1_L0"));
  l0::Kernel k = mod.kernel("gemv");
  k.group_size(64);
  k.arg_ptr(0, wbuf.ptr());
  k.arg_ptr(1, sbuf.ptr());
  k.arg_ptr(2, xbuf.ptr());
  k.arg_ptr(3, obuf.ptr());

  std::printf("| launches in list | us per list (median of 5) | us per launch |\n|---:|---:|---:|\n");
  for (int L : {1, 16, 128, 512, 2048}) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    for (int i = 0; i < L; ++i) list.launch(k, N / 64, 1);
    list.close();
    std::vector<double> us;
    for (int rep = 0; rep < 8; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      q.execute(list, &fence);
      fence.wait();
      const double t =
          std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
      if (rep >= 3) us.push_back(t);   // drop 3 warm-up replays
    }
    std::sort(us.begin(), us.end());
    std::printf("| %d | %.1f | %.3f |\n", L, us[2], us[2] / L);
  }
  std::puts("k2_launch_probe OK");
  return 0;
}
```

- [ ] **Step 2: Register it** - append to `tools/probe/CMakeLists.txt`. The variant is compiled under a probe-only name so it cannot collide with plan 8c's production `gemv_M1_K2560_N1536_S1_L0`:

```cmake
# Spec 4 stage 0 (plan 8a Task 2): fixed cost per launch of a K2 expert-sized
# int4 GEMV. gemv.cl at K2's expert gate||up shape, under a probe-only name.
add_ocloc_kernel(probe_k2_gemv_M1_K2560_N1536_S1_L0 SOURCE ${CMAKE_SOURCE_DIR}/src/kernels/gemv.cl
                 DEFINES M=1 K=2560 N=1536 S=1 LAYOUT=0)
add_executable(k2_launch_probe k2_launch_probe.cc)
target_include_directories(k2_launch_probe PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(k2_launch_probe PRIVATE b70_l0)
b70_target_kernel_dir(k2_launch_probe)
add_dependencies(k2_launch_probe kernel_probe_k2_gemv_M1_K2560_N1536_S1_L0)
```

- [ ] **Step 3: Build** - `tools/box.sh build 2>&1 | tail -3`. Expected: `Built target k2_launch_probe`.

- [ ] **Step 4: Prove idleness, then run** - idleness first, verbatim into the doc:
  `ssh -o BatchMode=yes user@box 'docker ps -q | wc -l; for f in /proc/*/fdinfo/*; do grep -l drm-driver "$f" >/dev/null 2>&1 || continue; pid=$(echo "$f"|cut -d/ -f3); echo "$pid $(cat /proc/$pid/comm)"; done | sort -u; uptime'`
  Then `tools/box.sh run 'ZE_AFFINITY_MASK=1 ./build/tools/probe/k2_launch_probe'`. Expected: a five-row table and `k2_launch_probe OK`. Grade the rows **record** only if the idleness check showed zero containers and zero DRM holders, otherwise **iterate**.

- [ ] **Step 5: Record** - add to the facts doc:

```markdown
## 3. Fixed cost per launch (k2_launch_probe, fact d) - <RECORD|ITERATE> grade

<idleness check output, verbatim>
<the table, verbatim>

Derived: the per-launch slope between the 128- and 2048-launch rows is <s> µs. At ~2,067 launches
per K2 decode token (plan 8e's count), launch overhead is ≈ <s × 2067 / 1000> ms/token, **derived**.
This is information for a later tuning spec, not a spec-4 bar.
```

- [ ] **Step 6: Commit**

```bash
git add tools/probe/k2_launch_probe.cc tools/probe/CMakeLists.txt docs/k2-stage0-facts-2026-09-14.md
git commit -m "probe(k2): fixed cost per launch of an expert-sized int4 GEMV

Claude-Session: "
```

---

### Task 3: K2 prompt ids and the BOS decision

**Files:**
- Create: `tools/oracle/tokenize_k2.py`
- Create: `tests/golden/k2/prompts/prose.ids`, `code.ids`, `cjk.ids`
- Modify: `docs/k2-stage0-facts-2026-09-14.md` (§4)

**Interfaces:**
- Consumes: the three committed texts `tests/golden/prompts/{prose,code,cjk}.txt` (unchanged), the K2 snapshot's `tokenizer.json` / `tokenizer_config.json`.
- Produces: `tests/golden/k2/prompts/<p>.ids` - whitespace-separated uint ids, one line, the format `tests/golden/golden_common.h::read_ids` and `tools/oracle/dump.py --prompt` already read. Plans 8d and 8e consume these files.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""K2-Horizon ids for the golden prompts (spec 4 §4 "Prompts").

The texts are the SAME files the 27B's golden prompts use
(tests/golden/prompts/*.txt), read exactly as tools/oracle/tokenize.py reads
them: trailing newlines stripped, interior newlines kept. K2 has a BOS token
(<|ifm|begin_of_text|> = 0) that Qwen3.8 does not. The rule fixed by spec 4:
the ids are what K2's tokenizer produces BY DEFAULT (`encode(text)`), and whether
that starts with BOS is recorded, not chosen here.

Usage, on the box, in the venv that has transformers 5.14.1:
    ~/auto-round/.venv/bin/python tools/oracle/tokenize_k2.py <snapshot> <out_dir>
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

from transformers import AutoTokenizer  # noqa: E402

PROMPTS = ("prose", "code", "cjk")
BOS_ID = 0


def main() -> None:
    snapshot, out_dir = sys.argv[1], sys.argv[2]
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    tok = AutoTokenizer.from_pretrained(snapshot)
    os.makedirs(out_dir, exist_ok=True)
    print(f"tokenizer class {type(tok).__name__}, bos_token {tok.bos_token!r} id {tok.bos_token_id}, "
          f"eos_token {tok.eos_token!r} id {tok.eos_token_id}")
    for p in PROMPTS:
        with open(os.path.join(repo, "tests", "golden", "prompts", f"{p}.txt"), encoding="utf-8") as f:
            text = f.read().rstrip("\n")
        ids = tok.encode(text)                              # the default - the rule
        bare = tok.encode(text, add_special_tokens=False)
        starts_bos = len(ids) > 0 and ids[0] == BOS_ID
        in_window = 24 <= len(ids) <= 64
        print(f"{p}: {len(ids)} ids (bare {len(bare)}), starts_with_bos={starts_bos}, "
              f"24-64 window={'yes' if in_window else 'NO'}")
        if tok.decode(bare) != text:
            print(f"  WARNING: decode(encode(text, add_special_tokens=False)) != text for {p}")
        with open(os.path.join(out_dir, f"{p}.ids"), "w", encoding="utf-8") as f:
            f.write(" ".join(str(i) for i in ids) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on the box** (seconds; no GPU, no container):
  `tools/box.sh sync && tools/box.sh run '~/auto-round/.venv/bin/python tools/oracle/tokenize_k2.py /home/user/.cache/huggingface/hub/models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ/snapshots/c0fd9971ad6d3bb4caedf726e798a02b8df9bcb5 tests/golden/k2/prompts'`
  Expected: a tokenizer line and three prompt lines. **If `AutoTokenizer` cannot load the tokenizer** (`TokenizersBackend` unknown to this transformers), stop and record the exception verbatim - do not substitute another tokenizer.

- [ ] **Step 3: Pull the ids back** - `for p in prose code cjk; do tools/box.sh pull tests/golden/k2/prompts/$p.ids; done`. Check each file is one line of integers: `wc -w tests/golden/k2/prompts/*.ids`.

- [ ] **Step 4: Record** - add to the facts doc:

```markdown
## 4. K2 prompt ids (tokenize_k2.py) - measured

<the script's output, verbatim>

**BOS decision:** the ids are K2's default `encode(text)`; they <do|do not> start with BOS (id 0).
**Window:** <all three inside 24-64 | <p> is <n> ids, outside the window - recorded, not edited>.
The 27B's Qwen ids in `tests/golden/prompts/*.ids` are untouched.
```

- [ ] **Step 5: Commit**

```bash
git add tools/oracle/tokenize_k2.py tests/golden/k2/prompts docs/k2-stage0-facts-2026-09-14.md
git commit -m "test(k2): golden prompt ids from K2's tokenizer, BOS decision recorded

Claude-Session: "
```

---

### Task 4: Oracle feasibility in the reference image - OPERATOR-GATED

**This task starts a container. Ask the operator for an explicit go before Step 3, and run it only when the box has no other heavy job.** If the go is not given, commit Steps 1-2 and record "not run, awaiting the operator" in the facts doc.

**Files:**
- Create: `tools/oracle/k2_import_check.py`
- Modify: `docs/k2-stage0-facts-2026-09-14.md` (§5)

**Interfaces:**
- Consumes: `tools/oracle/run_in_container.sh` (runs a command in `vllm-xpu-env-next-p314-t215-vxkp0` with `$SNAP` set; with `ORACLE_MODEL=<hub dir name>` the whole HF cache is mounted so snapshot symlinks resolve).
- Produces: whether the image's transformers imports `modeling_k2_horizon.py` via `trust_remote_code`, the model's parameter count built on the meta device, and the derived bf16 state-dict size plan 8d budgets memory against.

- [ ] **Step 1: Write the check** (no weights are loaded - meta device only, so it needs seconds and little memory):

```python
#!/usr/bin/env python3
"""Spec 4 stage 0, fact (c'): can the reference image run K2-Horizon's remote code?

Builds K2HorizonForCausalLM on the META device (no weights read) and prints its
parameter count and the bf16 bytes a dequantised state dict would take - the
number tools/oracle/dump_k2.py (plan 8d) must fit in the box's 121 GB.
Run inside the reference image: tools/oracle/run_in_container.sh 'python3 tools/oracle/k2_import_check.py "$SNAP"'
"""
import sys

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM


def main() -> None:
    snap = sys.argv[1]
    print(f"transformers {transformers.__version__}, torch {torch.__version__}")
    cfg = AutoConfig.from_pretrained(snap, trust_remote_code=True)
    print(f"config class {type(cfg).__name__}, model_type {cfg.model_type}, layers {cfg.num_hidden_layers}")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
    n = sum(p.numel() for p in model.parameters())
    print(f"model class {type(model).__name__}, parameters {n:,}, bf16 state dict {n * 2 / 1e9:.3f} GB (derived)")
    names = [k for k, _ in model.named_parameters()][:5]
    print("first parameter names:", names)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit the script** (before the container run, so the run is reproducible from a commit):

```bash
git add tools/oracle/k2_import_check.py
git commit -m "tools(oracle): K2 remote-code import check for the reference image

Claude-Session: "
```

- [ ] **Step 3: On the operator's go, run it** (short, but detached out of habit):
  `tools/box.sh sync && ssh -o BatchMode=yes user@box 'cd ~/b70-inference-server && ORACLE_MODEL=models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ setsid nohup tools/oracle/run_in_container.sh "python3 tools/oracle/k2_import_check.py \"\$SNAP\"" > $HOME/k2-import-check.log 2>&1 < /dev/null &'`
  Poll: `ssh -o BatchMode=yes user@box 'tail -8 $HOME/k2-import-check.log'` until a `parameters` line or a traceback appears.
  Expected: `model class K2HorizonForCausalLM, parameters <~36.2e9 + embed/lm_head> ...`. **A traceback is a finding** (spec 4 §8 risk 4): record it verbatim and stop.

- [ ] **Step 4: Record** - add to the facts doc:

```markdown
## 5. Oracle feasibility (k2_import_check.py, fact c) - <measured | not run: awaiting the operator>

<the log, verbatim>

Derived for plan 8d: bf16 state dict <x> GB; with the ~12 GB of overhead the 27B's dump showed
(61.4 GiB peak on a 50.1 GiB state dict), peak ≈ <x + 12> GB against the box's 121 GB - **estimated**.
```

- [ ] **Step 5: Commit** - `git add docs/k2-stage0-facts-2026-09-14.md && git commit -m "docs(k2): stage-0 oracle feasibility" -m "Claude-Session: "`.

---

### Task 5: How torch rounds the ops K2 adds - OPERATOR-GATED, same container go as Task 4

**Why:** the 27B's kernels mirror torch's per-op bf16 rounding, and the 27B's golden gate proved that mirror. K2 adds five ops the 27B never had, and for each one the plan-8c kernel has to pick a spelling. Each spelling is a compile-time define with a pre-registered default (below). This probe measures which spelling torch actually uses, on this image, in seconds, before a gate run has to discover it the slow way. **It changes no bar:** the golden gate still arbitrates, and a define that disagrees with the probe is a one-line CMake change in plan 8c Task 1.

**Files:**
- Create: `tools/oracle/k2_semantics_probe.py`
- Modify: `docs/k2-stage0-facts-2026-09-14.md` (§6)

**Interfaces:**
- Consumes: nothing but torch in the reference image.
- Produces: one line per question, `Q<n> <name>: <candidate>=<fraction equal> ...`, and the facts-doc table plan 8c Task 1 reads its defines from:

| define (plan 8c) | question | pre-registered default |
|---|---|---|
| `K2_SCORE_SCALE` | Q2: `bf16 * (128 ** -0.5)` multiplies by the fp32 scalar or by its bf16 rounding | fp32 scalar `0.08838834764831844f` (bf16 alternative `0.08837890625f`) |
| `K2_TOPK_SUM_T4` | Q5 k=4: `fp32[.,4].sum(-1)` association | `0` SEQ (left to right, top-k order); `1` PAIR |
| `K2_TOPK_SUM_T8` | Q5 k=8: `fp32[.,8].sum(-1)` association | `0` SEQ; `1` PAIR; `2` LANES4 |
| `K2_TOPK_TIE_LOW` | Q6: `torch.topk` on equal values returns the lower index first | `1` |
| (none - information) | Q1 bf16 linear, Q3 softplus (the device has no fp64, so the kernel's fp32 formula is fixed), Q4 SiLU, Q7 bf16 matmul, Q8 fp32 sigmoid | recorded only |

- [ ] **Step 1: Write the probe**

```python
#!/usr/bin/env python3
"""Spec 4 stage 0: which rounding torch uses for the ops K2 adds. CPU, seconds, no weights.

Run inside the reference image:
    tools/oracle/run_in_container.sh 'python3 tools/oracle/k2_semantics_probe.py'
Every answer is a FRACTION of elements equal, bitwise, to a candidate spelling. 1.000000 means
the spelling reproduces torch on every element tried; anything else is not that spelling.
"""
import math

import torch
import torch.nn.functional as F

torch.manual_seed(20260914)
B, F32, F64 = torch.bfloat16, torch.float32, torch.float64


def eq(a: torch.Tensor, b: torch.Tensor) -> str:
    return f"{(a.reshape(-1) == b.reshape(-1)).double().mean().item():.6f}"


def rne(t: torch.Tensor) -> torch.Tensor:          # fp32 -> bf16, round-to-nearest-even
    return t.to(F32).to(B)


def main() -> None:
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}")

    # Q1: F.linear in bf16 == one rounding of an fp32 or fp64 accumulation?
    x = (torch.randn(64, 2560) * 2).to(B)
    w = (torch.randn(1024, 2560) * 0.02).to(B)
    y = F.linear(x, w)
    print("Q1 bf16 linear:", "rne(f32 matmul)=" + eq(y, rne(x.float() @ w.float().t())),
          "rne(f64 matmul)=" + eq(y, rne((x.double() @ w.double().t()).float())))

    # Q2: bf16 * python float, the attention score scale 128 ** -0.5
    s = 128 ** -0.5
    a = (torch.randn(1 << 16) * 30).to(B)
    r = a * s
    s_b = torch.tensor(s, dtype=F32).to(B).float()
    print("Q2 score scale:", "rne(f32(a)*s_f32)=" + eq(r, rne(a.float() * torch.tensor(s, dtype=F32))),
          "rne(f32(a)*f32(bf16(s)))=" + eq(r, rne(a.float() * s_b)),
          "rne(f64(a)*s)=" + eq(r, rne((a.double() * s).float())),
          f"bf16(s)={s_b.item():.12f}")

    # Q3: softplus on bf16, beta = ln 2, threshold 20 (both branches covered)
    g = torch.linspace(-60, 60, 1 << 16).to(B)
    sp = F.softplus(g, beta=math.log(2))
    beta32 = torch.tensor(math.log(2), dtype=F32)
    gx = g.float()
    bx = gx * beta32
    cand32 = torch.where(bx > 20, gx, torch.log1p(torch.exp(bx)) / beta32)
    gd = g.double()
    bxd = gd * math.log(2)
    cand64 = torch.where(bxd > 20, gd, torch.log1p(torch.exp(bxd)) / math.log(2)).float()
    print("Q3 softplus:", "rne(f32 formula)=" + eq(sp, rne(cand32)),
          "rne(f64 formula)=" + eq(sp, rne(cand64)),
          "threshold-branch elems=" + str(int((bx > 20).sum())))

    # Q4: SiLU on bf16 == rne(fp32 silu)?
    v = (torch.randn(1 << 16) * 4).to(B)
    print("Q4 silu:", "rne(f32 silu)=" + eq(F.silu(v), rne(F.silu(v.float()))))

    # Q5: fp32 sum over the last dim of k = 4 and k = 8 - which association?
    for k in (4, 8):
        t = torch.rand(1 << 16, k, dtype=F32) * torch.logspace(-3, 3, k, dtype=F32)
        ref = t.sum(-1)
        seq = t[:, 0].clone()
        for i in range(1, k):
            seq = seq + t[:, i]
        pair = t.clone()
        while pair.shape[1] > 1:
            pair = pair[:, 0::2] + pair[:, 1::2]
        lanes4 = None
        if k == 8:
            lanes4 = (t[:, 0] + t[:, 4]) + (t[:, 1] + t[:, 5]) + (t[:, 2] + t[:, 6]) + (t[:, 3] + t[:, 7])
        out = [f"k={k}", "SEQ=" + eq(ref, seq), "PAIR=" + eq(ref, pair[:, 0])]
        if lanes4 is not None:
            out.append("LANES4=" + eq(ref, lanes4))
        print("Q5 topk sum:", *out)

    # Q6: torch.topk tie order
    tie = torch.tensor([[0.5, 0.7, 0.7, 0.1, 0.7, 0.2]], dtype=F32)
    print("Q6 topk ties: indices", torch.topk(tie, 3, dim=-1).indices.tolist(),
          "(lower-first would be [[1, 2, 4]])")

    # Q7: the attention score matmul in bf16
    q = (torch.randn(1, 32, 1, 128)).to(B)
    kk = (torch.randn(1, 32, 300, 128)).to(B)
    m = torch.matmul(q, kk.transpose(2, 3))
    print("Q7 bf16 matmul:", "rne(f32)=" + eq(m, rne(q.float() @ kk.float().transpose(2, 3))),
          "rne(f64)=" + eq(m, rne((q.double() @ kk.double().transpose(2, 3)).float())))

    # Q8: fp32 sigmoid == 1 / (1 + exp(-x))?
    z = torch.randn(1 << 16, dtype=F32) * 8
    print("Q8 sigmoid:", "1/(1+exp(-x))=" + eq(torch.sigmoid(z), 1.0 / (1.0 + torch.exp(-z))))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit the script before running it**

```bash
git add tools/oracle/k2_semantics_probe.py
git commit -m "tools(oracle): K2 torch rounding-semantics probe (pre-registered defaults in plan 8a)

Claude-Session: "
```

- [ ] **Step 3: On the operator's go, run it** (seconds; no model, no HF cache needed):
  `tools/box.sh sync && ssh -o BatchMode=yes user@box 'cd ~/b70-inference-server && setsid nohup tools/oracle/run_in_container.sh "python3 tools/oracle/k2_semantics_probe.py" > $HOME/k2-semantics.log 2>&1 < /dev/null &'`, then poll `tail -12 $HOME/k2-semantics.log` until the `Q8` line appears.

- [ ] **Step 4: Record** - add §6 to the facts doc: the log verbatim, then the define table above with a `measured` column. For every define: the candidate whose fraction is `1.000000` is the value; if no candidate reaches 1.000000, record the best fraction and **keep the default** (the gate arbitrates). Commit: `git add docs/k2-stage0-facts-2026-09-14.md && git commit -m "docs(k2): torch rounding semantics for the ops K2 adds" -m "Claude-Session: "`.

---

## Self-review

**Spec coverage:** §5 stage 0 (a) allocation limit → Task 1; (b) import + oracle memory → Task 4 (meta-device parameter count; the full one-token forward is deferred to plan 8d's first dump, where the real `/usr/bin/time -v` peak is recorded - a deliberate narrowing, because a full forward needs the ~75 GB dequant and is the dump itself); (c) re-tokenised prompts + BOS → Task 3; (d) launch cost → Task 2. §8 risks 1, 3, 4, 5 each have a measurement here. Task 5 is an addition the spec did not list: it front-loads spec §3.2's "rounding points to mirror" for the five ops the 27B never had (score scale at head_dim 128, softplus, top-k weight sum, tie order), so plan 8c's defines start from torch's measured behaviour rather than a guess; §8 risk 2 (top-k tie order) is Q6. The quant-config check (§1, "the int4 layout is layout 0 unchanged") → Task 1.
**Placeholders:** the `<...>` markers are measurement outputs pasted at execution, not design gaps.
**Types:** probes use only `src/l0` and `src/loader` public headers as they exist at `a6ce30e`+; no new runtime types are introduced.
