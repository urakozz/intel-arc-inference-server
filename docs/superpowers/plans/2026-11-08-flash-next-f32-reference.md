# Flash-Next's f32 reference for the prefill gates (operator 2026-11-08, B; before Bolt's)

**Goal:** judge Flash-Next's prefill on Intel's checkpoint (`qwen4exp_prefill_intel_pp_test` = `r33.pp`, two cards,
`layers:37`; the one-card `layers:18` twin) against an f32 CPU run of the same truncated model on the same prompt,
instead of prefill against decode alone. Until it exists A' applies (2026-11-08: step 4's state bars and step 6's last
logits within 2x the chunk-16 rounding control's cosine distance; routing, selections and tokens printed).

**Why:** on the truncated model the same engine moves as far between chunks of 16 and 2048 (rounding only) as between
prefill and decode (2026-11-07: first QSA layer worst 0.9939 both, median 0.99951 / 0.99949, p01 0.9796 / 0.9806,
logits cosine 0.949 for the control; tokens 16 / 32 agree prefill vs decode, the worst layer L35 0.22). A model without
its last 11 layers has near-flat logits and its top-512 block selections flip with rounding, so "prefill equals decode"
cannot be the bar. An f32 reference says which side, if either, is wrong.

## The rule

Per layer and per state kind (QSA K / V rows, compressed keys, GDN state and conv ring, PLE rings), the expert sets
and block selections on the last chunk's rows, and the last logits: **prefill's distance to f32 <= 2x decode's distance
to f32** (the 2026-11-01 slack), both printed. Decode is the path the golden gates (`r32.*`) validate. Tokens by the
f32 judge (one bf16 ulp near-tie, teacher-forced on the f32 run's tokens), printed beside. No f32 file: the real-
checkpoint tests keep A'.

## Tasks

1. [x] **`qwen4exp_ref.py run --dtype fp32 --layers N --state-dump`** (`tools/oracle/qwen4exp_ref.py`): the int4 checkpoint
   dequantised to f32 (the box Xeon has no bf16 compute: f32 is the fast path), the prompt exactly as the test builds it
   (`q4exp_4k.ids` cycled to 4500 positions - a new `--cycle 4500`), dumping after the prompt every layer's state in the
   engine's layout keys (`state.L<l>.{k,v,ckeys,gdn,conv,ple}`), the last 404 rows' routes and selections, the last
   logits, and 32 greedy tokens. Mac test: `test_qwen4exp_ref.py` on the tiny model - fp32 vs bf16 agree to bf16
   rounding, the dump keys complete.
2. [x] **`qwen4exp_oracle.sh intel-f32 <N>`** -> `$DATA/oracle-out-q4exp-f32-L<N>/q4exp_4k.state.safetensors` for N = 37
   and 18; resumable; under the 64 GB CPU cap (layer-streamed); a data key `oracle_q4exp_f32` in
   `tools/box_validate/data.sh`. Measure N = 18 first (time, peak RAM), then queue N = 37 as a CPU chain line.
3. [x] **`qwen4exp_prefill_test ... judge:<file>`**: reads the dump and applies the rule in steps 4-6 (A' stays the fallback
   without it); `r33.pp` and its one-card twin pass the file (CMake, like the Ornith gates). The 2x comparison helper
   unit-tested on the Mac.
4. [x] **Box:** `r30.host` (the Python tests), then `r33.pp` and the one-card twin against f32; record both distances per
   layer in open-work and close the row.

## Order and cost

Before Bolt's f32 reference (`2026-11-08-bolt-f32-reference.md`), after 3a (one agent slot). CPU: hours for N = 37 at
f32 (ESTIMATED; Task 2 measures N = 18 first). No engine change. The Flash-Next golden-set remake (`r30.golden*`, the
CPU list) uses the same f32 path once Task 1 exists.

## Execution record (2026-11-10)

Implementation: branch `q4exp-f32-ref`, signed commits `f67d4e6b`, `df13329e`, `24346fb2`, `89bd5ed2`.
The registered Intel runs use the engine-format int8 output head and int8 PLE table; the wrapper pins the
checkpoint at `refs/main`. Dumps carry prompt, compute/head/PLE/vocabulary identity and reject invalid sets.
Task 3 is implemented and reviewed; the full box build (JOBS16) and initial `r30.host` pass. Tiny dtype/head/state tests, the host distance
helper, wrapper dry-run checks, shell syntax and Mac Python syntax pass. Detached CPU measurements and GPU
judgments are pending. Reconfigure CMake after linking the completed references so `EXISTS` enables `judge:`.

The first real CPU attempt stopped before model execution: the engine's PLE sibling is an absolute symlink
outside the HF cache mount. `89bd5ed2` adds a read-only physical PLE mount (`ORACLE_PLE`, oracle tooling only),
covered by an external-symlink dry run and a Docker-argument test of the mount and 64g/no-swap cap.
The same L18-first chain was retried; time/RAM and GPU results remain pending.

## Initial measured judgment (2026-11-10; not an attribution)

Both CPU dumps and both registered judged runs completed. L18:561.9s, peak process RSS44.1GiB;
L37:1013.7s, peak process RSS67.7GiB. Both used the configured64g/no-swap Docker limit; no cgroup
memory.peak measurement was retained. The scratch chain returned1 after L37 because its RSS guard also
applied to the final run (its “L18” error label is misleading); both completed dumps were already published.

The one-card gate FAILed (328.69s): conv L2 distance9.37984366e-5/3.07412299e-5 (prefill/decode,3.0512x),
L14 distance3.27466847e-4/1.63637745e-4 (2.0012x). The two-card real case of r33.pp FAILed (611.34s):
13conv kinds,2GDN kinds and last logits; worst conv L28 is0.0191467019/0.00387283859 (4.9438x),
last logits0.0696688533/0.0336527804 (2.0702x). Full per-layer/kind numbers are recorded in
`docs/probes/q4exp-f32-initial-judgment-2026-11-10.csv` (logits use a pseudo-layer equal to the layer count).

Tokens accepted by the one-ulp judge: L18 decode24/32, prefill27/32; L37 both20/32. Thus the decode
baseline also disagrees with the reference. Keep Waiting21 open and the branch unmerged pending the
operator-requested decode-only audit of PLE identity/dequantization, int4 weights, and earliest states.
Task4's checkbox records execution and measurements; it does not claim a passing gate or a closed row.
