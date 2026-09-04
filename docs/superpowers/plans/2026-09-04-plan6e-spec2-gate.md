# Spec 2 Gate Implementation Plan (plan 6e)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the record-grade prefill rows at the spec's final sha, prove decode is untouched, decide the gate against the bar ruled at Stage 0, and write the records (win path) or the memo (short path).

**Architecture:** No engine code changes. One harness change (`tools/bench_decode.sh --pp`), then measurement, arithmetic, and documents. Task 1 may be pulled forward - plans 6c/6d want it for their attribution runs.

**Tech Stack:** bash (`tools/bench_decode.sh`, `tools/box.sh`), the `b70-decode --bench --pp` CLI from plan 6b, markdown.

**Spec:** `docs/superpowers/specs/2026-09-04-spec2-prefill-design.md` §2 (bar), §6 (bars), §7 (gate), §9 (records). Interfaces: `.superpowers/sdd/2026-09-04-plan6-spec2-prefill/interfaces.md`.

## Global Constraints (spec §8, verbatim)

- Box workflow `tools/box.sh`; JOBS 44; the **system** oneAPI toolchain (`/opt/intel/oneapi/compiler/2026.1/bin/icpx` via full path or `setvars.sh`, `docs/10-the-box.md:42`) - the container's SYCL is the operator's and stays read-only; `sycl-tla` built once as a static dependency, version pinned in the build.
- Probes run on card 1 (`ZE_AFFINITY_MASK=1`) while a server holds card 0; record rows only on a provably idle box (zero DRM holders).
- Every number labelled measured vs derived vs estimated/external, with grade and conditions; no two values for one quantity without a reconciliation sentence.
- Nothing pushed; tags local; work on the branch the operator names.
- Design must not preclude prefix caching: chunk boundaries at multiples of 1024 align with the queued block-snapshot scheme (`docs/04-architecture.md`, "Follow-on"); no work on it here.

Plus, for every task here: never kill a process or touch docker; commits end with `Claude-Session: `.

---

### Task 1: `tools/bench_decode.sh --pp N`

**Files:**
- Modify: `tools/bench_decode.sh` (flag parsing block ~lines 55-62; remote invocation ~line 95; awk summary ~lines 100-120)
- Modify: `.superpowers/sdd/2026-09-04-plan6-spec2-prefill/interfaces.md` (CLI section - the stdout row contract below is the amendment)

**Interfaces:**
- Consumes (plan 6b): `b70-decode <ckpt> --bench --pp N --tg T` prefills N ids of `kBenchPrompt` cycled via `Engine::prefill()`, then runs the tg measurement. **stdout row contract (this task fixes it; 6b implements it):** with `--pp` the CLI prints the existing tg row unchanged - `| b70-decode <sha> | <depth=N> | <tg> | <t/s> | <ms/token> |` - followed by a second row `| b70-decode <sha> | pp | <N> | <ms total> | <t/s> |`, where ms total spans the first prefill launch to the first generated id present in `cur_token` (loader excluded, first-token-inclusive). The human line `pp: N ids in X ms (Y t/s)` goes to stderr.
- Produces: `tools/bench_decode.sh --pp N [--tg T] [--runs R] [--model M]` printing both rows per run and two median lines: the existing tg summary and `pp median <t/s> (<ms>) over R run(s); min, max, spread`.

- [ ] **Step 1: Write the failing harness test** - a shell test that runs the script against a fake `b70-decode` (a stub script echoing two canned rows) and asserts both median lines appear.

Create `tests/tools/bench_decode_pp_test.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
tmp=$(mktemp -d)
# Stub box.sh: `run` prints canned rows; `sync`/`build` no-op.
cat > "$tmp/box.sh" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  sync|build) exit 0 ;;
  run) echo "| b70-decode abc1234 | 4096 | 256 | 32.20 | 31.06 |"
       echo "| b70-decode abc1234 | pp | 4096 | 2100.0 | 1950.48 |" ;;
esac
EOF
chmod +x "$tmp/box.sh"
out=$(BENCH_BOX_SH="$tmp/box.sh" tools/bench_decode.sh --pp 4096 --runs 3 --no-build 2>&1 >/dev/null)
grep -q "^median 32.20 t/s" <<<"$out"
grep -q "^pp median 1950.48 t/s (2100.0 ms) over 3 run(s)" <<<"$out"
echo OK
```

- [ ] **Step 2: Run it to verify it fails**

Run: `bash tests/tools/bench_decode_pp_test.sh`
Expected: FAIL - `bench_decode.sh: unknown argument '--pp'`.

- [ ] **Step 3: Implement** - in `tools/bench_decode.sh`:

(a) add `PP="${PP:-}"` next to `TG=`; (b) in the `case` add `--pp) need_value "$@"; PP="$2"; shift 2 ;;`; (c) add a `BOX_SH="${BENCH_BOX_SH:-tools/box.sh}"` variable and use `"$BOX_SH"` for the three `tools/box.sh` calls (this is what lets the test stub it); (d) build the remote flags: `PPFLAG=""; if [ -n "$PP" ]; then PPFLAG="--pp $PP"; fi` and change the remote line to `... --bench --depth $DEPTH --tg $TG $PPFLAG"` (with `--pp`, `--depth` is ignored by the CLI - plan 6b defines `--pp` as replacing ingest; keep passing it so the tg row's depth column stays N); (e) capture BOTH rows: `row="$("$BOX_SH" run ...)"` now holds two lines - split with `tg_row=$(grep -v '| pp |' <<<"$row")`, `pp_row=$(grep '| pp |' <<<"$row" || true)`; push `tg_row` to `rows` and `pp_row` to a new `pprows` array; (f) after the existing awk, add:
```bash
if [ -n "$PP" ]; then
printf '%s\n' "${pprows[@]}" | awk -F'|' '
  { ms[NR] = $5 + 0; ts[NR] = $6 + 0 }
  END {
    if (NR < 1) exit 0
    for (i = 1; i <= NR; i++) for (j = i + 1; j <= NR; j++)
      if (ts[j] < ts[i]) { t = ts[i]; ts[i] = ts[j]; ts[j] = t; t = ms[i]; ms[i] = ms[j]; ms[j] = t }
    mid = int((NR + 1) / 2)
    pct = (ts[mid] > 0) ? (100 * (ts[NR] - ts[1]) / ts[mid]) : 0
    printf "pp median %.2f t/s (%.1f ms) over %d run(s); min %.2f, max %.2f, spread %.2f (%.2f%%)\n",
           ts[mid], ms[mid], NR, ts[1], ts[NR], ts[NR] - ts[1], pct
  }' >&2
fi
```
Update the header comment's usage block with `tools/bench_decode.sh --pp 4096   # prefill 4096 ids, then tg 256`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `bash tests/tools/bench_decode_pp_test.sh`
Expected: `OK`.

- [ ] **Step 5: Amend `interfaces.md`** CLI section with the two-row stdout contract from the Interfaces block above (replace the single-line `pp: …` sentence; keep the stderr human line).

- [ ] **Step 6: Commit**

```bash
git add tools/bench_decode.sh tests/tools/bench_decode_pp_test.sh
git add -f .superpowers/sdd/2026-09-04-plan6-spec2-prefill/interfaces.md
git commit -m "bench: --pp threads prefill rows through bench_decode.sh

Claude-Session: "
```

---

### Task 2: Pre-gate verification at the final sha

**Files:** none modified; produces `.superpowers/sdd/2026-09-04-plan6-spec2-prefill/gate-preflight.md`.

**Interfaces:** Consumes plans 6b/6c/6d landed on the branch; the tests they registered (`prefill_gate_test`, `prefill_consistency_test`, `prefill_determinism_test`, `attn_chunk_test`, `gdn_chunk_test`, `gemm_test`, `dequant_test`).

- [ ] **Step 1: Confirm the tree is the final sha and clean** - `git status --short` empty; `git log --oneline -1` recorded as `<SHA>` in `gate-preflight.md`.
- [ ] **Step 2: Full suite from a bare invocation** - `tools/box.sh test` (no env aliases). Expected: all tests pass (count = 40 + the prefill tests registered by 6b/6c/6d; record the exact number). A first-invocation `attn_test` transient has been seen twice before; if any test fails once, re-run the full suite and record both results.
- [ ] **Step 3: Decode invariants** - `tools/box.sh run "ZE_AFFINITY_MASK=1 ./build/tests/replay_determinism_test"`; expected output contains `774 kernels` and `19 modules` and three bitwise-identical replays. Record verbatim.
- [ ] **Step 4: Write `gate-preflight.md`** with the sha, the suite tally, and the invariant lines. No commit (the SDD workspace is ledger material; the controller commits the ledger).

---

### Task 3: Decode control rows (spec §6 bar 5)

**Files:** produces rows appended to `gate-preflight.md`.

**Interfaces:** Consumes `tools/bench_decode.sh` (unchanged path, no `--pp`).

- [ ] **Step 1: Prove the box idle** - `ssh user@box 'docker ps -q | wc -l; for f in /proc/*/fdinfo/*; do grep -l drm-driver "$f" >/dev/null 2>&1 || continue; pid=$(echo "$f"|cut -d/ -f3); echo "$pid $(cat /proc/$pid/comm)"; done | sort -u; uptime'`. Expected: `0`, an empty DRM list, load < 1. If any DRM holder is present, STOP: do not kill it; report the pid/comm to the controller and wait - the rows below are record-grade or they are not taken.
- [ ] **Step 2: RTN triple** - `MODEL=/home/user/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64 tools/bench_decode.sh --runs 3 --depth 4096 --tg 256`. Record all three rows and the median line.
- [ ] **Step 3: Vishva triple** - `MODEL=/home/user/.cache/huggingface/hub/models--Vishva007--Qwen3.8-27B-W4A16-AutoRound-GPTQ/snapshots/2a9077667e28aa53e61d91bdee5d7962e8674668 tools/bench_decode.sh --runs 3 --depth 4096 --tg 256 --no-build`.
- [ ] **Step 4: Re-verify idle after** (Step 1's command again) and record.
- [ ] **Step 5: The drift rule** - compare medians to `2a7df0b`'s **32.22 t/s / 31.03 ms** (RTN) and **29.33 / 34.09** (Vishva). Inside ≤ 0.09% (the measured same-checkpoint day drift, docs/BENCHMARKS "day-scale drift") → "decode untouched" holds. Outside → the difference is a finding: name the commit range and what in it could touch decode (the buffer split is the only candidate by design); the gate does not proceed until it is explained.

---

### Task 4: Record-grade prefill rows

**Files:** produces rows in `gate-preflight.md`; the bench logs kept under the SDD workspace.

**Interfaces:** Consumes Task 1's `--pp`; plan 6b's `Engine::prefill` with `PrefillScratch::kC` (4096 or 2048 as plans 6c ruled).

- [ ] **Step 1: Idle proof** (Task 3 Step 1, verbatim; same STOP rule).
- [ ] **Step 2: The gate row** - `MODEL=/home/user/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64 tools/bench_decode.sh --pp 4096 --tg 256 --runs 3`. Record three `pp` rows + three tg rows + both median lines. This is the row the gate decides on: **device-side pp, first-token-inclusive, loader-excluded, chunk = kC, RTN, record grade**.
- [ ] **Step 3: Chunk-width sensitivity (labelled, not the gate)** - the same command with the CLI's chunk override if plan 6b exposed one (`--chunk 2048`, `--chunk 1024`); one run each; recorded as iterate-grade context for the memo.
- [ ] **Step 4: The byte-matched row** - Vishva snapshot, `--pp 4096 --tg 256 --runs 3 --no-build`. It carries the same bytes vLLM reads (bf16 `lm_head`), so it is the row quoted beside vLLM's pp.
- [ ] **Step 5: Idle re-verification** and record.

---

### Task 5: Gate arithmetic

**Files:** produces `.superpowers/sdd/2026-09-04-plan6-spec2-prefill/gate-arithmetic.md`.

**Interfaces:** Consumes the **ruled bar** from plan 6a T6 (the operator's ruling recorded in the ledger - quote it with its date) and the composed ceiling it was derived from.

- [ ] **Step 1: Bar check** - `pp_median_RTN ≥ bar` → WIN; else SHORT. State both numbers and the margin in t/s and %.
- [ ] **Step 2: Efficiency** - `pp_median / composed_ceiling` (the Stage-0 sum at the same chunk width); per-term comparison where the `--profile`-style attribution from 6c/6d exists: which term missed its Stage-0 price and by how much.
- [ ] **Step 3: Against vLLM** - RTN and Vishva pp medians beside **1973 t/s**, with the three labels written out each time: device-side vs HTTP-inclusive; our chunk width vs vLLM's 2 × 2048; checkpoint bytes (13.673 vs 15.540 GB). State plainly whether the byte-matched row leads or trails and by how much. Do not derive an "HTTP-equivalent" number - spec 3 measures that.
- [ ] **Step 4: Seconds for a 4096-token prompt** - `121 s` before (bench log `2a7df0b`) vs now; the factor, stated once.

---

### Task 6: Records (win path) - or Task 7 (short path)

**Files (win):**
- Modify: `docs/BENCHMARKS.md` - new section "The spec-2 gate rows - `<SHA>`, <date>" beside the spec-1.7 gate rows: the pp table (RTN gate row, Vishva byte-matched row, chunk-sensitivity rows labelled iterate), the tg rows re-taken here, box conditions (DRM-clean, load), the three labels paragraph, the vLLM comparison paragraph.
- Modify: `README.md` - phase-1 row: "pp target met - device-side <N> t/s at `<SHA>` (vLLM 1973 HTTP-inclusive; spec 3 takes the HTTP row)"; the "Measured state" paragraph gains one sentence.
- Modify: `docs/05-perf-model.md` - a superseding block under "Phase 1, measured": the prefill verdict with the same labels; the prefill roofline arithmetic (48.97 GFLOP/token, the measured XMX rate from 6a T6).
- Modify: `docs/15-step-anatomy.md` - the prefill anatomy table finalised with the record-run attribution.
- Modify: `docs/superpowers/specs/2026-09-04-spec2-prefill-design.md` - status line: "Closed <date> at `<SHA>`: <pp> t/s vs bar <bar> - MET".

- [ ] **Step 1: Write the BENCHMARKS section** (numbers from Tasks 3-5 only; every cell labelled).
- [ ] **Step 2: README + docs/05 + docs/15 + spec status.**
- [ ] **Step 3: Two-values sweep** - `grep -n "121 s\|29.5 ms\|1973\|pp4096" README.md docs/*.md` and reconcile every hit with the new rows (supersede in place; never delete a historical number).
- [ ] **Step 4: Commit and tag**

```bash
git add docs/BENCHMARKS.md README.md docs/05-perf-model.md docs/15-step-anatomy.md docs/superpowers/specs/2026-09-04-spec2-prefill-design.md
git commit -m "docs(spec2): record the gate - <pp> t/s pp4096 device-side, bar <bar> MET

Claude-Session: "
git tag -a spec2-done -m "spec 2 gate: <pp> t/s pp4096 (device-side) vs bar <bar>"
```

### Task 7: The memo (short path)

**Files:** Create `docs/superpowers/specs/<date>-spec2-gate-memo.md` (the spec-1.7 memo `docs/superpowers/specs/2026-09-04-spec1.7-gate-memo.md` is the template: verdict table, what the lead is, per-stage yields, falsified predictions, what is left and why nothing is queued, owed items, ruling).

- [ ] **Step 1: Write it** from Task 5's arithmetic: which term missed its Stage-0 price; whether the composed ceiling itself sits under vLLM (the "silicon" answer) or the implementation does (the "skill" answer); the redesign menu priced from Stage 0.
- [ ] **Step 2: Records that hold regardless** - the BENCHMARKS rows and the docs/05 block from Task 6 Steps 1-2 are written on the short path too (the numbers are real either way); README says "short" with the number.
- [ ] **Step 3: Commit** (no tag); STOP for the operator's ruling.

---

## Self-review

**Spec coverage:** §2 bar → Task 5; §6 bars 1-4 → Task 2 (the tests plans 6b-6d registered); bar 5 → Task 3; bar 6 → Task 2 Steps 2-3; §7 gate both paths → Tasks 6/7; §9 records → Task 6; §8 constraints → Global Constraints + Task 3/4 idle proofs. Gap: none.
**Placeholder scan:** `<SHA>`, `<pp>`, `<bar>`, `<N>`, `<date>` are measurement outputs substituted at execution, not design gaps. No "TBD"/"similar to".
**Type consistency:** the stdout row contract in Task 1 matches the amended `interfaces.md` CLI section; checkpoint paths match plans 6b-6d; the drift figure (≤ 0.09%) and prior rows (32.22/31.03, 29.33/34.09 at `2a7df0b`) match docs/BENCHMARKS.
