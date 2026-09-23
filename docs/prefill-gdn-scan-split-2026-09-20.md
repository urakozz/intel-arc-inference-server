# Split-BF16 DPAS GDN scan: bounded experiment record

Status: **pre-registered; not a production adoption.**  The default remains
the vector `pf_gdn_scan`.  This record is deliberately narrower than the D3
proposal in `prefill-gdn-scan-dpas-2026-09-05.md`: it records one opt-in build
and its gates, rather than proposing any new architecture.

## Question and fixed arithmetic

D2 was fast but failed a real-prompt state-cosine gate (Vishva `code`,
`0.998499499 < 0.999`) and one RTN determined token (`92/93`).  D3 tests only
whether preserving the low residual of the *state path* repairs those failures.

`B70_PREFILL_GDN_SCAN` is read as a **process-lifetime** selector before any
GDN launch: unset or `vector` binds the existing vector entry; `dpas_split`
binds only the new experimental `pf_gdn_scan_dpas_split`; any other nonempty
value throws before a launch or buffer mutation.  Changing the environment
inside an engine process is unsupported because replay capture may retain its
first kernel selection.  Selector comparisons are separate processes.

The experimental entry retains FP32 register state and FP32 global master
state.  At each 64-position sub-chunk it forms, with RNE BF16 conversion,
`S_hi = rne(S)` and `S_lo = rne(S - bf16(S_hi))`.  Both `W.S` and `Q.S` use
independent FP32-accumulating DPAS chains over the two terms and combine the
two chain results once.  `vn` remains FP32 after that combined subtraction.
For the update, `D = vn * exp(gl-gc)` is split identically; the high DPAS
chain starts from the FP32-decayed master state and the low chain starts at
zero, and they are combined once into that FP32 state.  `A2*vn` deliberately
remains D2's single-BF16 controlled variable.  This is approximate split-BF16
arithmetic, **not** an FP32 equivalence claim.  The expected resource price is
about 35,328 B SLM and 50.5 DPAS/work-item/sub-chunk.

## Gates and decision

No numerical threshold is loosened.  Before a speed judgment the split
selector must pass: the scan structural/chunk checks (including zero and
nonzero state, a residual-sensitive fixture, C=100 ragged 64+36 boundary,
finite output), `prefill_determinism_test`'s 9 cases x 3 runs bitwise, both
model token gates with every printed `gdn_state` cosine strictly greater than
`0.999`, and both-model prefill consistency.  Exit status from a gate alone is
not sufficient: cosine logs are inspected independently.

**User acceptance decision (2026-09-21):** the unavailable RTN/W4G64 gate is
retired for this bounded experiment.  The current acceptance checkpoint is
`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`, with the registered
primary-model gates above and the matched benchmark below.  This is an
explicit scope change by the user, not a relaxed numerical threshold or a
claim that the absent RTN gate passed.

Any band, structural, finite, determinism, token, cosine, or consistency
failure is a failed experiment: retain evidence, report the first tensor/row,
and do not weaken a gate or benchmark it as an improvement.  Only if every
gate holds may a matched warm 4096 replay benchmark compare separate
`vector` and `dpas_split` processes.  The vector default remains unchanged
regardless of the experiment result.

## Planned commands

All GPU work is serialized on GPU 0, with no `ONEAPI_DEVICE_SELECTOR`:

```sh
# First inspect GPU users; do not stop user work.
ssh -o BatchMode=yes -o ConnectTimeout=10 user@box \
  'ps -eo pid=,stat=,etime=,command= | grep -E "b70|ctest|prefill|vllm" || true'

# Transfer only task files, build the existing configurations, then targeted RED/GREEN.
rsync -az --relative ./src/kernels/prefill/pf_gdn_scan.cl ./src/runtime/prefill/gdn.cc \
  ./src/runtime/prefill/gdn.h ./tests/prefill/gdn_scan_split_test.cc \
  ./tests/prefill/gdn_chunk_test.cc ./tests/CMakeLists.txt \
  ./docs/prefill-gdn-scan-split-2026-09-20.md \
  user@box:/home/user/b70-inference-server/
ssh -o BatchMode=yes -o ConnectTimeout=10 user@box \
  'cd /home/user/b70-inference-server && cmake --build build-nosycl -j2 && cmake --build build -j2'
ssh -o BatchMode=yes -o ConnectTimeout=10 user@box \
  'cd /home/user/b70-inference-server && env -u ONEAPI_DEVICE_SELECTOR ZE_AFFINITY_MASK=0 ctest --test-dir build-nosycl -R "gdn_scan_split_test" -V -j1'

# After the targeted structural gate passes, run determinism, both token/cosine
# gates and consistency in a split-selector process.  Preserve CTest's status
# independently of the parser, then require both to pass.
ssh -o BatchMode=yes -o ConnectTimeout=10 user@box \
  'cd /home/user/b70-inference-server; log=/home/user/gdn-split-prefill-gates.log; set +e; env -u ONEAPI_DEVICE_SELECTOR ZE_AFFINITY_MASK=0 B70_PREFILL_GDN_SCAN=dpas_split ctest --test-dir build -R "prefill_(determinism|gate.*|consistency.*)_test" -V -j1 > "$log" 2>&1; ctest_rc=$?; python3 /tmp/gdn_split_cosine_gate.py "$log" --expected-test prefill_gate_l0_test --expected-test prefill_gate_test; parser_rc=$?; printf "ctest_rc=%s parser_rc=%s\\n" "$ctest_rc" "$parser_rc"; test "$ctest_rc" -eq 0 && test "$parser_rc" -eq 0'

The command's exit status is checked directly.  Then run
`python3 /tmp/gdn_split_cosine_gate.py /home/user/gdn-split-prefill-gates.log`
with every registered *token/cosine* gate name supplied as `--expected-test
NAME`; CTest's independent status covers determinism and consistency too.  The
parser rejects a missing/truncated/nonfinite table, a non-passing token/cosine
test, or a per-prompt `worst` reduction not strictly greater than `0.999`.  A printed
`worst: Lx` is the minimum over all 48 GDN layer rows; normal output therefore
has three such reductions (prose/code/cjk) plus 24 selected diagnostic rows
per test, not 48 raw rows.  Record the expected test names and worst rows for
both checkpoints below.

# Only after the gates, compare matched warm replay in separate vector/split processes.
ssh -o BatchMode=yes -o ConnectTimeout=10 user@box \
  'cd /home/user/b70-inference-server && env -u ONEAPI_DEVICE_SELECTOR ZE_AFFINITY_MASK=0 B70_PREFILL_GDN_SCAN=vector ./build-nosycl/tests/prefill_replay_test urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --bench'
ssh -o BatchMode=yes -o ConnectTimeout=10 user@box \
  'cd /home/user/b70-inference-server && env -u ONEAPI_DEVICE_SELECTOR ZE_AFFINITY_MASK=0 B70_PREFILL_GDN_SCAN=dpas_split ./build-nosycl/tests/prefill_replay_test urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --bench'
```

## TDD evidence and results

**RED, before production changes:** after a no-SYCL cache regeneration,
`ZE_AFFINITY_MASK=0 ./build-nosycl/tests/gdn_scan_split_test` exited 134.  The
new direct, hand-derived residual test reached the real module and failed only
because the requested experimental entry did not exist:

```
prefill::KernelCache: variant 'pf_gdn_scan' has no entry point
'pf_gdn_scan_dpas_split': ... ZE_RESULT_ERROR_INVALID_KERNEL_NAME
```

Remote log: `/home/user/gdn-split-red.log`.  This is the intended RED:
there was no production selector or split kernel at that point.  The initial
registration was corrected from the SYCL-only block to the direct L0 suite so
this test runs in `build-nosycl`; output is poisoned with NaNs before launch so
the C=100 all-zero assertion also detects missing writes.

No speed claim is permitted until the decision gates above are green.

**Initial focused no-SYCL GREEN:** after implementation, remote
`cmake --build build-nosycl --target gdn_scan_split_test gdn_scan_selector_test -j2`
and `ctest --test-dir build-nosycl -R 'gdn_scan_(split|selector_.*)_test' -V -j1`
passed **5/5** at `ZE_AFFINITY_MASK=0` (`/home/user/gdn-split-green-target.log`).
The direct kernel test covers the hand-derived S residual, D residual,
nonzero and zero initial states, C=100's 64+36 boundary, finite values and
repeat bitwise determinism.  Separate-process selector cases confirm unset and
`vector` bind `pf_gdn_scan`, `dpas_split` binds `pf_gdn_scan_dpas_split`, and
an invalid value rejects before any `Context` launch.

**Boundary-test refinement after review:** the resolver-only test was removed;
the selector is private to `gdn.cc`.  New separate-process `gdn_chunk` tests
exercise its real boundary.  An invalid environment calls it with valid
arguments, requires a `runtime_error`, zero `Context` launches, and bytewise
unchanged caller-owned state/ring sentinels.  For unset, `vector`, and
`dpas_split`, a nonzero-state C=64 fixture compares the actual `gdn_chunk`
state and `gdn_o` to the independently launched selected scan entry on the
same saved pre-state, and requires it differ from the opposite entry.  This is
not resolver-string introspection or a public test-only API.

The characterization run passed **5/5** in 29.69 s:
`/home/user/gdn-split-boundary-characterization.log`.  Its intentional
mapping mutation (`dpas_split` hard-wired to the vector entry) then failed the
split boundary test at `tests/prefill/gdn_chunk_test.cc:303`, where actual
`gdn_chunk` state no longer matched the direct split entry:
`/home/user/gdn-split-boundary-red.log`.  That RED demonstrates this test
catches the selector regression it names.  The proper split mapping was
restored before fresh verification.  The restored full-build command,
`env -u ONEAPI_DEVICE_SELECTOR ZE_AFFINITY_MASK=0 ctest --test-dir build -R
'gdn_(chunk|scan_(invalid_selector|default_dispatch|vector_dispatch|split_dispatch|split))_test'
-V -j1`, passed **6/6** in 32.63 s at
`/home/user/gdn-split-boundary-green.log`: the original structural
`gdn_chunk_test`, real invalid-selector protection, unset/vector/split actual
dispatch, and the direct split residual fixture.  The fresh no-SYCL command
`cmake -S . -B build-nosycl && cmake --build build-nosycl --target
gdn_scan_split_test -j2 && env -u ONEAPI_DEVICE_SELECTOR ZE_AFFINITY_MASK=0
ctest --test-dir build-nosycl -R gdn_scan_split_test -V -j1` passed **1/1**;
local capture: `/tmp/gdn-split-nosycl-green.log`.

**RTN availability:** `build/CMakeCache.txt` configures
`B70_RTN_SNAPSHOT=/home/user/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64`,
whose parent is absent.  Fresh CMake reports that `prefill_gate_rtn_test` and
`prefill_gate_long_test` are not registered.  A read-only search of the remote
models tree, Hugging Face cache and repository found no compatible relocated
RTN/W4G64 checkpoint; `oracle-out-rtn` alone remains and cannot run a gate.
Only the registered Vishva/L0 gate paths may run now.  The exact RTN checkpoint
must be restored or an already-compatible path supplied and CMake reconfigured
to run the historical RTN registration.  On 2026-09-21 the user explicitly
retired that unavailable gate and authorized acceptance/benchmarking against
the current AutoRound GPTQ checkpoint instead; no RTN result is inferred.

**Available Vishva gates (split selector):** the serial GPU0 command in this
record completed its five registered tests in 276.25 s at
`/home/user/gdn-split-prefill-gates.log`; the driver recorded
`ctest_rc=0 parser_rc=0` at
`/home/user/gdn-split-prefill-gates-driver.log`.  The two token/cosine gates
(`prefill_gate_l0_test` and `prefill_gate_test`) each had the same strict
48-layer reductions: prose `0.999903435` (L33), code `0.999189068` (L60), and
cjk `0.999903208` (L60), all strictly greater than `0.999`.  Both token gates
were 93/93 determined rows exact; both consistency registrations passed, and
`prefill_determinism_test` passed its 9 prompt/chunk cases x 3 reset runs
bitwise.  This is a primary-model-only result, not completion of the
pre-registered both-model contract.

**Fresh regressions after the boundary revision:** serialized GPU0 full CTest
runs used the unchanged `vector` selector for general-regression coverage;
the split-specific structural and model gates above remain separate evidence.
The freshly rebuilt no-SYCL suite passed **66/66** in 454.11 s at
`/home/user/gdn-split-full-nosycl-ctest.log`.  The freshly rebuilt SYCL
suite passed **81/81** in 782.32 s at
`/home/user/gdn-split-full-sycl-ctest.log`.  The detached driver at
`/home/user/gdn-split-full-regression-driver.log` recorded
`build_nosycl_rc=0 ctest_nosycl_rc=0 build_sycl_rc=0 ctest_sycl_rc=0`.

**ABBA matched replay benchmark (2026-09-22):** after the user retirement of
the RTN requirement, the idle GPU0 benchmark used four independent Level Zero
processes in `vector, dpas_split, dpas_split, vector` order.  Each process
used the configured snapshot
`84575a18f209992ef96d819b31f924b489e3d55d`, 4096 IDs made by repeating
`tests/golden/prompts/prose.ids`, and the uint32-LE SHA256
`342eace92c4214eadd1ccfda421d3a576ae753f9be4ee61abbca167316237e67`.
The executable asserted `pos == 4096`, two 2048-token chunks and the expected
launch count; every raw run returned first token **383** and every process
ended `prefill_replay_test OK`.  Each listed sample is an unprofiled
`Engine::prefill()` wall time; run 0 is first-use and excluded from medians.

| Process / selector | Immediate warm ms (runs 1-3) | Immediate process median | Recorded warm ms (runs 1-3) | Recorded process median |
|---|---|---:|---|---:|
| vector_a / vector | 2732.695, 2741.614, 2737.732 | 2737.732 ms / 1496.129 t/s | 2732.018, 2732.128, 2733.433 | 2732.128 ms / 1499.198 t/s |
| split_a / dpas_split | 2576.224, 2574.195, 2584.543 | 2576.224 ms / 1589.924 t/s | 2577.281, 2579.346, 2579.590 | 2579.346 ms / 1587.999 t/s |
| split_b / dpas_split | 2574.290, 2591.013, 2586.333 | 2586.333 ms / 1583.710 t/s | 2582.022, 2577.515, 2582.791 | 2582.022 ms / 1586.354 t/s |
| vector_b / vector | 2755.651, 2757.968, 2758.210 | 2757.968 ms / 1485.151 t/s | 2746.171, 2741.451, 2743.357 | 2743.357 ms / 1493.061 t/s |

The selector aggregate is the statistical median of all six warm samples for
each selector/mode (the middle pair averaged): vector immediate **2748.632 ms
/ 1490.196 t/s**, vector recorded **2737.442 ms / 1496.287 t/s**; split
immediate **2580.383 ms / 1587.361 t/s**, split recorded **2579.468 ms /
1587.924 t/s**.  Thus split recorded improves over vector recorded by
157.974 ms and 91.637 t/s, but remains 35.425 ms and 22.116 t/s behind the
matched current vLLM result (2544.043 ms / 1610.04 t/s).  Raw logs are
`/home/user/gdn-split-abba-20260921-{driver,vector_a,split_a,split_b,vector_b}.log`.

**One split profile diagnostic (not throughput):** after ABBA, one separate
idle-GPU0 process used `B70_PREFILL_PROFILE=1` with
`B70_PREFILL_GDN_SCAN=dpas_split`; the private process selector maps that
value to `pf_gdn_scan_dpas_split`, and the already-green real-boundary test
independently verifies that mapping.  The instrumented run reported 4096
positions, 17,383 launches, 2609.7 ms call wall and 2493.6 ms L0 timestamp
sum.  Same-definition narrow GDN core (`gate+A+solve+wu+A2+scan`) was
527.6 ms in the old vector diagnostic and 351.8 ms here; all named `gdn_*`
rows were 606.8 and 431.0 ms, respectively.  The scan row was 304.7 to
127.2 ms (96 launches), a 177.5 ms reduction, while the other five narrow
core rows were 222.9 to 224.6 ms (+1.7 ms).  Excluding named `gdn_*` rows,
the rest rose from 2044.6 to 2062.6 ms (+18.0 ms), chiefly `linear_l0`
1620.1 to 1635.8 ms (+15.7 ms).  Total L0 GPU time fell 2651.4 to 2493.6 ms
(-157.8 ms).  These are single instrumented runs with profiling coupling and
run variation; they establish the measured scan-row change, not a causal
whole-pipeline or whole-GDN saving.  They are never inputs to the ABBA
throughput or vLLM comparison.  Log:
`/home/user/gdn-split-profile-20260922.log`.

**Experiment decision and self-review:** the opt-in split implementation,
residual-sensitive scan tests, real `gdn_chunk` selector boundary tests,
registered primary-model gates, and the authorized current-checkpoint speed
comparison are green.  The split selector is faster than vector but does not
exceed the matched vLLM result.  It remains opt-in: the production default is
still vector and no vendor adoption or default switch is proposed.  The
remaining arithmetic concern is intentional scope: `A2*vn` is still
single-BF16 and could affect later-layer outputs; it is not claimed harmless
beyond the same-layer state path.

## Contract amendment - 2026-09-22 (operator ruling)

The both-model contract pre-registered in "Gates and decision" is amended: the
only checkpoint this project gates on is
`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`.  RTN/W4G64 is **retired** - not
restored, not built, not used as a performance reference, and never grounds for
relaxing a gate.  Ruling A31 deleted that checkpoint from the box on 2026-09-09;
the contract, not the evidence, was the thing out of date.

**What this changes about the result above: nothing that was measured.**  Every
gate recorded in this file ran on the primary checkpoint, and that set is now
the complete contract, so the experiment's correctness evidence is **closed
rather than externally blocked**.  The "RTN availability" section and the
"primary-model-only result, not completion of the pre-registered both-model
contract" qualifier are superseded by this amendment; they are kept as the
record of what was true when written.

**What it costs, and is not yet decided.**  `prefill_gate_rtn_test` and
`prefill_gate_long_test` are both registered only inside
`if(EXISTS "${B70_RTN_SNAPSHOT}/config.json")` (`tests/CMakeLists.txt`), so
retiring RTN also retires the only registration path for the **multi-chunk long
gate** (spec 2 §6.2's 2820-id prompt).  Neither is registered today, because the
snapshot is gone.  The long gate therefore needs either an AutoRound
`oracle-out-long` set of its own or an explicit decision to drop it; this
amendment does not decide that, and no gate elsewhere covers a >2048-id prompt
against a CPU oracle.

The production default is unchanged by this amendment: `vector` remains the
shipped scan.  Whether the now-fully-gated `dpas_split` selector is promoted is
a separate decision with its own record.

## Amendment - 2026-09-23 (the "Available Vishva gates" section is void, and the kernel is fixed)

Two corrections, neither of which changes the vector default.

**1. This record's green gates are not evidence.** Its cosines are bit-identical
to the vector kernel's from three days earlier; that run was executing
`pf_gdn_scan`, not the split entry
(`docs/superpowers/specs/2026-09-22-prefill-parity-program-design.md` §11). Run
with its dispatch proven, the split entry as shipped on 2026-09-20 **fails**
`prefill_gate_l0_test`: 92/93, code L60 `gdn_state` cosine 0.996344994.

**2. The cause was `A2`, and it is fixed.** "`A2*vn` deliberately remains D2's
single-BF16 controlled variable" was the whole failure - a 2x2 on the golden gate
put all of it on `A2`'s rounding and none on `vn`'s. `A2` now carries hi/lo BF16
limbs like `S` and `D`; the entry passes at 93/93 with every cosine over 0.999
and keeps a measured 2.33x over the vector scan on the `gdn_scan` profile row.
See `docs/prefill-gdn-scan-split-fix-2026-09-23.md`. The ABBA throughput table
above was measured through `prefill_replay_test` run directly, where the two
selectors did produce different wall times, so it stands as a selector
comparison - but it was measured on the pre-fix kernel.
