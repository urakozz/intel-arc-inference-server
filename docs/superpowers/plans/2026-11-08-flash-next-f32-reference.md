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

1. **`qwen4exp_ref.py run --dtype fp32 --layers N --state-dump`** (`tools/oracle/qwen4exp_ref.py`): the int4 checkpoint
   dequantised to f32 (the box Xeon has no bf16 compute: f32 is the fast path), the prompt exactly as the test builds it
   (`q4exp_4k.ids` cycled to 4500 positions - a new `--cycle 4500`), dumping after the prompt every layer's state in the
   engine's layout keys (`state.L<l>.{k,v,ckeys,gdn,conv,ple}`), the last 404 rows' routes and selections, the last
   logits, and 32 greedy tokens. Mac test: `test_qwen4exp_ref.py` on the tiny model - fp32 vs bf16 agree to bf16
   rounding, the dump keys complete.
2. **`qwen4exp_oracle.sh intel-f32 <N>`** -> `$DATA/oracle-out-q4exp-f32-L<N>/q4exp_4k.state.safetensors` for N = 37
   and 18; resumable; under the 64 GB CPU cap (layer-streamed); a data key `oracle_q4exp_f32` in
   `tools/box_validate/data.sh`. Measure N = 18 first (time, peak RAM), then queue N = 37 as a CPU chain line.
3. **`qwen4exp_prefill_test ... judge:<file>`**: reads the dump and applies the rule in steps 4-6 (A' stays the fallback
   without it); `r33.pp` and its one-card twin pass the file (CMake, like the Ornith gates). The 2x comparison helper
   unit-tested on the Mac.
4. **Box:** `r30.host` (the Python tests), then `r33.pp` and the one-card twin against f32; record both distances per
   layer in open-work and close the row.

## Order and cost

Before Bolt's f32 reference (`2026-11-08-bolt-f32-reference.md`), after 3a (one agent slot). CPU: hours for N = 37 at
f32 (ESTIMATED; Task 2 measures N = 18 first). No engine change. The Flash-Next golden-set remake (`r30.golden*`, the
CPU list) uses the same f32 path once Task 1 exists.
