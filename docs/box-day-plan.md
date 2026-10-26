# Box day: the run plan for the validation queue

Written 2026-10-07 on the Mac, without the box. This is the order in which to run the 28 rows
of [box-validation-queue.md](superpowers/plans/box-validation-queue.md) through
`tools/box_validate.sh` (stages: `tools/box_validate/stages.sh`) once the box is back. Every
stage id below comes from `tools/box_validate.sh --list` / `--dry-run [--with optin]` on main
`97a519b` (180 default stages including `pre`, 43 opt-in, 12 manual; row 29's opt-in
`r29.cost`, plan 19a Task 4's probe, makes 44). **Every time in this file is an estimate**:
nothing here has been timed on the current tree. Where an estimate comes from is stated next
to it.

## At a glance

| # | session | cards | invocation (from the frozen worktree, §1.1) | stages | wall (est.) |
|---|---|---|---|---|---|
| 0 | preflight, downloads, data push | - | `tools/box_validate.sh --only pre` + §1.3 / §1.4 | 1 | 0.5-1 h (downloads go on in the background) |
| 1 | **G0, stop the line** | device 0 | `tools/box_validate.sh --only g0` | 4 | 2.5-4.5 h |
| 2 | one-card gates: Agnes and the Qwen3.8 feature rows | device 0 | `tools/box_validate.sh --only r1,r2,r4,r5,r6,r7,r8,r9,r11,r12,r17,r18 --skip r1.sweep,r1.speed,r2.cost,r2.auto_rows,r6.cost,r8.cost,r8.rows,r11.speed` | 40 | 8-13 h |
| 3 | Ornith's real checkpoint loads (row 19) | device 0 | `tools/box_validate.sh --only r19` | 3 | ~0.25 h |
| 4 | Ornith decode, prefill, serving (rows 10, 13, 16) | device 0 (beside 5 on device 1: §1.8) | `tools/box_validate.sh --only r10,r13,r16 --skip r16.cost` | 21 | 3-5 h |
| 5 | K2-Horizon (rows 14, 15, 21, 25) | device 0, or device 1 beside 4 (`--device 1`, §1.8) | `tools/box_validate.sh --only r14,r15,r21,r25 --with r14.oracle` | 35 | 4.5-7.5 h |
| 6 | timed one-card rows, **box idle (RECORD)** | device 0, both idle | `tools/box_validate.sh --only r1.sweep,r1.speed,r2.cost,r2.auto_rows,r3,r6.cost,r8.cost,r8.rows,r11.speed,r16.cost` | 10 | 7-11 h |
| 6b | DFlash verify cost at M = 5..8 and one block's draft cost (row 29, plan 19a Task 4) | device 0, both idle | `tools/box_validate.sh --only r29.cost` | 1 (opt-in) | 0.25-0.5 h (est.) |
| 6c | llama-benchy comparison matrix: b70-serve and vLLM, one client and set of flags (row 41) | device 0, both idle | `tools/box_validate.sh --only r41.ours,r41.vllm,r41.vllm_mtp` | 3 (opt-in) | 2-3.5 h (est.) |
| 6d | the batching probe (row 66, plan 13a): the row builds' check, the step table, the interleave chunk | device 0, both idle for the timed stages | `tools/box_validate.sh --only r66`, then `--only r66.cost,r66.chunk` | 3 + 2 (opt-in) | 0.5 h + 1-2 h (est.) |
| 7 | pipeline decode, then prefill (rows 22, 23) | **both** | `tools/box_validate.sh --only r22,r23` | 22 | 4.5-7 h |
| 8 | pipeline integration (row 27) | **both** | `tools/box_validate.sh --only r27` | 14 | 4.5-7 h |
| 9 | Kolibri-1 on the synthetic checkpoints (rows 24, 26, 28) | **both** | `tools/box_validate.sh --only r24,r26,r28 --with r24.oracle_synth` | 31 | 4.5-7 h |
| 9b | Qwen3.8-Flash-Next on the synthetic checkpoints (rows 30-34, spec 21) | **both** | `tools/box_validate.sh --only r30,r31,r32,r33,r34 --with r32.ple_rate,r32.bench_attn` | 32 + 2 opt-in | 3.5-6 h (est.) |
| 9c | spec 22's expert-offload probe (row 36, plan 22a): links, copy / zero-copy rates, capacity, the curves, the projection | **both**, idle | `tools/box_validate.sh --only r36` (+ CPU lane: `--with r36.prune,r36.prune_a4`) | 15 (+ 2 opt-in) | 3-5 h GPU (est.); the CPU opt-ins hours to days |
| 10 | the rest of the suite | device 0 | `tools/box_validate.sh --only x` | 1 | 0.5-1 h |
| 11 | Bolt on SergiioB's g128 checkpoint (row 57): the CPU reference, golden sets, the g128 A4 reference, the taps | box CPU, no card | `tools/box_validate.sh --only r57 --with r57.ppl,r57.hfcheck,r57.golden,r57.a4_ref,r57.taps` | 1 + 5 opt-in | hours, then days for `r57.a4_ref` (est.) |
| 12 | **the Bolt BF16 window** (rows 57, 58): the BF16 original downloaded, our g64 quantisation, the BF16 references, gate Q and F6, then the BF16 original removed | **both** for `r58.tune`, else box CPU | §2, Session 12 | row 58's + 3 opt-in | ~3-6 days (est.) |

**Total, the default set: ~40-63 h of box time, ~51 h central (est.)** - two to two and a half
days if the sessions run back to back. Sessions 1-10 together run each of the 179 default
stages exactly once (checked against the default `--dry-run`: no stage missing, none twice);
sessions 5 and 9 add the two opt-in oracle stages the K2 and Kolibri gates need. The short
path (§3) is ~36 h. The opt-in stages (§3.3) are extra: tier 1 alone 7-10 h, every tier
30-45 h or more.

**Added 2026-10-13: Session 9b** for rows 30-34 (spec 21, Qwen3.8-Flash-Next), merged after this
plan was written - their 32 default stages were in no session (checked against `--list`: every
other default stage is). **Session 9c** (row 36, spec 22's expert-offload probe, plan 22a) follows it:
it needs 9b's data and binaries. With 9b and 9c, sessions 1-10 run each of today's 221 default stages
once; 9b adds ~3.5-6 h (est.) plus its CPU-lane data (§1.6) and, optionally, Intel's real checkpoint
(§1.3); 9c ~3-5 h of GPU time (est.) plus its opt-in CPU stages.

**Added 2026-10-22: Sessions 11 and 12** for Nemotron 3.5 Bolt (spec 25). Session 11 needs only SergiioB's g128
checkpoint (41.9 GB). The BF16 original (149.8 GB) is downloaded **on the box only**, for Session 12, and
removed after it: everything that reads it (row 58's quantisation, the BF16 golden sets, the BF16 A4
reference, gate Q, F6) runs in that one window.

Row 20 (Kolibri-1 spec 20a, the quantisation) has no runbook stage and needs spec 20
decision 1; it is not part of the box day. Everything on the real Kolibri int4 checkpoint
(spec 20b, which does not exist yet) is recorded SKIP "missing data" in session 9 -
expected, not a failure.

## How the runbook constrains the plan

- **One orchestrator per tree and card; the whole box or one card per stage.** `remote.sh` runs
  the selected stages strictly in registry order (row order). A `gpu` stage holds the whole box
  (`tools/gpu_lock.sh box`: `~/b70-gpu.lock` exclusive), a `card` stage only its orchestrator's
  card (`tools/gpu_lock.sh card $DEVICE`). One orchestrator never overlaps two GPU stages, but a
  second one on device 1 (`--device 1 --only <stages>`, the same state) runs its `card` stages
  beside the first one's - two correctness streams at once (§1.8). Timed, two-card and
  device-0-only stages stay `gpu`: they wait for both streams' running card stages, and new card
  stages wait behind them. CPU work (downloads, the CPU oracles) still runs beside non-timed GPU
  stages (§1.6).
- **State = commit x baseline.** Results live in
  `~/b70-validate/b70-inference-server-validate/<HEAD sha12>-vs-<baseline sha12>/`. Every
  session must run from the **same commit** with the **same baseline** (default `b32aaaf`), or
  G0 is not PASS in the new state and every later stage is SKIP "blocked". Hence the frozen
  worktree in §1.1. A fix pushed during the day is a new commit: either G0 again (~3 h) in the
  new state, or `--state <old name>` to keep adding to the old one (only for fixes that do not
  touch device code or the paths G0 compares).
- **`--only` never adds G0.** It runs `pre` plus the named stages; G0 must already be PASS in
  the state. A row prefix (`r22`) names the row's default stages only; opt-in stages need their
  id or `--with`. `after` lists and the recaps' `need_pass` are enforced against this state:
  a stage whose prerequisite is not PASS is SKIP "needs PASS first", so the session order below
  is also the dependency order (the hidden ones: `r5.r0`, `r10.r0` and `r13.r0` `need_pass`
  `r1.gates`, so row 1 goes before rows 5, 10 and 13).
- **Re-running the same command** re-attaches to a live run; after the run has finished it
  launches again and re-runs every stage that is not PASS. To fetch without running anything,
  use `--summary`; to look, `--status`. Long sessions: add `--no-wait` and re-attach later. A
  WiFi drop costs nothing: the orchestrator is detached on the box.

## 1. Preconditions

### 1.1 Freeze the tree (Mac)

- [ ] Pick the commit (main after this plan, or later) and give it its own worktree, so main
      can move during the day without changing the state:
      ```sh
      git worktree add --detach ../b70-boxday main
      cp tools/box.env ../b70-boxday/tools/box.env      # untracked; never commit it, never print it
      cd ../b70-boxday
      tools/box_validate.sh --dry-run | head -6          # tree <sha12>, baseline b32aaaf, state <sha12>-vs-b32aaafbc7e2
      ```
      `git diff --quiet HEAD` must hold (a `-dirty` suffix in the state name means it did not).
      `LOCAL_DATA` (the `oracle-out-*` the driver can push) resolves to the main checkout
      through the git common dir, so the worktree needs no copies of the data.
- [ ] Read `tools/box_validate.sh --dry-run --only g0` and `--dry-run --with optin` once.
- [ ] No other agent's GPU work on the box for the whole day: their jobs would take the GPU
      lock between stages, break the timed rows' grade and occupy the second card.

### 1.2 The box: reachable, synced, preflight

- [ ] `tools/box_validate.sh --only pre` - syncs the tree to `~/b70-inference-server-validate`,
      extracts the baseline into `~/b70-inference-server-g0-b32aaafbc7e2`, and runs only the
      preflight (minutes). Its `pre.log` shows `uptime`, `nproc`, `free -g`, `df -h ~`,
      `/dev/dri`, `ocloc`, `xpu-smi discovery`, the idle grade, and `have.env` - the
      `HAVE_<key>=0|1` list every stage's `needs` is checked against. Fetch it with
      `tools/box_validate.sh --summary` (it fetches the whole state) or read it on the box.
- [ ] Present before session 1 (from `have.env`): `qwen` (the gate snapshot `84575a18`,
      `SNAP_QWEN`), `oracle_qwen` (`oracle-out-primary/`, box-made), `oracle_image` (the
      oracle container image), `tok_python` (`~/auto-round/.venv/bin/python` with
      `tokenizers`, for `r1.parity`). Nice to have: `xpu_smi` (`r7.serve` records device memory
      with it), `uvx` (only the opt-in llama-benchy rows).
- [ ] Both cards enumerate (`/dev/dri` shows the two B70 render nodes) and the P2P kernel patch
      is still in place (it is what `r22.devices` checks first; peer access absent fails every
      `peer` arm).

### 1.3 Downloads on the box (start right after 1.2, detached)

| checkpoint | repo | size | needed by | key |
|---|---|---|---|---|
| Agnes 3.0 Flash int4 | `urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ` | 22.9 GB | row 1, `r7.plan_agnes`, `r7.lines_agnes`, `r7.serve_agnes`, `r22.agnes`, `r23.agnes`, `r27.agnes` | `agnes` |
| Ornith 1.5 35B-A3B int4 | `urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ` | 23.0 GB (22.99 GB of shards, measured on the Mac's copy) | rows 10, 13, 16, 19, `r22.ornith`, `r23.ornith`, `r27.ornith` | `ornith` |
| K2-Horizon int4 | `urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ` | 21.8 GB | rows 14, 15, 21, 25 | `k2` |
| Qwen3.8 int4 g64 gate snapshot | `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ` `--revision 84575a18f209992ef96d819b31f924b489e3d55d` | 20.0 GB | everything (G0) | `qwen` - **only if `HAVE_qwen=0`**: the box has used this snapshot since August |
| Kolibri-1 int4 | does not exist (spec 20b) | ~42.5 GB when it does | `r24.partial`, `r24.pp_real`, `r26.real`, `r28.real` | `kolibri` |
| tiny qwen4_exp model | `qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next` | 124 MB | `r30.host` (session 9b) | `q4exp_tiny` |
| Qwen3.8-Flash-Next original, **small files only** | `Qwen/Qwen3.8-Flash-Next` `config.json tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja` | a few MB (never the 360 GB) | `r31.synth` (the synthetic checkpoints), `r34.a4_ref` | `q4exp_orig_small` |
| Nemotron 3.5 Bolt int4 g128 (SergiioB) | `SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym` `--revision b9f381d20cbd4cede9f729c2647d2542c4a35997` | 41.9 GB | row 57 (Session 11), Session 12, rows 59-65 | `bolt_g128` (`bolt_small`: its small files alone, `--exclude '*.safetensors'`) |
| Nemotron 3.5 Bolt BF16 original (**Session 12 only**, removed after it) | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16` `--revision 567021037ff88364143e24ea05e6abd0f72086ff` | 149.8 GB | `r57.golden_bf16`, `r57.a4_bf16`, `r57.gateq`, row 58 (`r58.rtn`, `r58.tune`, `r58.eval`) | `bolt_bf16` |
| Intel's Qwen3.8-Flash-Next W4A16 (**optional**, 9b part 2) | `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` | 181.17 GB (102.4 GB of it the PLE shards) | `r31.load_real`, `r32.partial`, `r33.golden_real` (SKIP "missing data" without it), the opt-in real-weight stages of rows 30-34 | `q4exp_intel` |

```sh
export REMOTE_DIR=b70-inference-server-validate
tools/box.sh run 'tools/probe/detach.sh $HOME/dl-models.log bash -c "for r in urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ; do uvx --from huggingface_hub hf download \$r || exit 1; done"'
tools/box.sh run 'tail -3 $HOME/dl-models.log'      # poll; the log ends in ALLDONE rc=0
```

For session 9b add the two small spec 21 downloads (the tiny model, the original's small files), and
Intel's 181 GB only if part 2 of 9b runs (disk first, §1.5):

```sh
tools/box.sh run 'tools/probe/detach.sh $HOME/dl-q4exp.log bash -c "uvx --from huggingface_hub hf download qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next && uvx --from huggingface_hub hf download Qwen/Qwen3.8-Flash-Next config.json tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja"'
# optional, 9b part 2 (hours):
tools/box.sh run 'tools/probe/detach.sh $HOME/dl-q4exp-intel.log uvx --from huggingface_hub hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound'
```

~67.7 GB in all (without spec 21's); at an unmeasured 30-100 MB/s that is 10-40 min (est.), well inside session
1. (`uvx` may be only at `~/.local/bin/uvx` in a non-interactive shell.) The Agnes checkpoint
**is not in the Mac's HF cache any more**, so it is downloaded here rather than copied (see
§1.4 for why that matters).

### 1.4 Data from the Mac (`--push-data`, or by hand)

`--push-data` skips the Agnes checkpoint when the Mac no longer has it (it is gone from the
Mac's cache, checked 2026-10-07 - download it on the box, §1.3) and pushes every golden set that
exists, `oracle-out-kolibri-a4` included (fixed 2026-10-07; before, the Agnes rsync failed under
`set -e` and aborted before any golden set). By hand, into the box's data tree (`DATA_DIR`,
`~/b70-inference-server`; `pre` links every `oracle-out*` there into the validate and baseline
trees):

```sh
cd <main checkout>                       # where oracle-out-* live
set -a; . tools/box.env; set +a          # BOX; never echo it
rsync -a --info=progress2 oracle-out-agnes oracle-out-agnes-mtp oracle-out-ornith oracle-out-ornith-mtp "$BOX:b70-inference-server/"
rsync -a ~/.cache/huggingface/hub/models--Aleph-Alpha--Kolibri-1-BF16 "$BOX:.cache/huggingface/hub/"
```

| set (Mac) | size | read by | state |
|---|---|---|---|
| `oracle-out-agnes` | 1.1 GB | `r1.gates` (`oracle_agnes`) | ready |
| `oracle-out-agnes-mtp` | 91 MB | `r1.mtp_head` (`oracle_agnes_mtp`) | ready |
| `oracle-out-ornith` | 481 MB | `r10.gates`, `r13.gates` (`oracle_ornith`) | ready |
| `oracle-out-ornith-mtp` | 91 MB | `r16.m1` (`oracle_ornith_mtp`) | ready |
| Kolibri-1-BF16 release, tokenizer and configs only | ~10 MB (tokenizer.json 9.5 MB; no weights) | `r24.oracle_synth` (make_synth copies them into each synthetic checkpoint; b70-serve needs them for `r28.*`), `r28.host`'s `kolibri_tokenizer_test` | ready |
| `oracle-out-12a-qwen38` | 1.0 GB | no stage (the 12b tolerances are already in the tree, `6ee1d59`) | optional, record only |
| `oracle-out-19a` | 1.3 GB | no stage | done (2026-10-08); optional, record only |
| `oracle-out-ornith-a4`, `oracle-out-k2-a4`, `oracle-out-kolibri-a4` | - | `r16.a4`, `r25.toolcall` / `r25.a4`, `r28.a4` | **not made**: Ornith's being made on the Mac (2026-10-08, §5), K2's on the box CPU (`r25.a4_ref`, hours: ~1-1.5 days sequential, ~0.6-1 day with `a4_ref.sh`'s default `BATCH=auto RESIDENT=auto` - ESTIMATED), Kolibri's wherever 20b runs |

Note `r25.toolcall` is a **default** stage that needs `k2_a4_set` (`oracle-out-k2-a4/set/manifest.json`):
it is SKIP "missing data" in session 5 unless `r25.a4_ref` (opt-in, hours: see the table above) has run, or the set
alone is made on the box before session 5 - one container run of make_set.py, no reference
(est. minutes), written into the data tree so `pre` links it:
`REMOTE_DIR=b70-inference-server-validate tools/box.sh run 'OUT=$HOME/b70-inference-server/oracle-out-k2-a4 tools/toolcall/a4_ref.sh k2 set'`
(needs the K2 download; read the script's header first).

### 1.5 Disk and RAM

- [ ] **Disk** (`df -h ~` in `pre.log`): ~68 GB of new checkpoints, ~12.4 GB of Kolibri
      synthetic checkpoints (`r24.oracle_synth`: two 5-layer real-width checkpoints, ~6.2 GB
      each), ~2 GB of golden sets, and three build directories (the validate tree, the
      baseline tree, `build-agnes-sweep`; unmeasured, budget 10-20 GB). **Want >= 120 GB free**;
      +20 GB if the Qwen3.8 snapshot must be fetched; +42.5 GB later for Kolibri int4; +20 GB
      for the manual `r18.ct` checkpoint (`RedHatAI/Qwen3.8-27B-INT4`), which is not on the day's
      list. **Session 9b:** +~27 GB for the two Qwen3.8-Flash-Next synthetic checkpoints
      (`r31.synth`, ~13.5 / ~13.7 GB); part 2 adds **~233 GB** (Intel's 181.17 GB + the 51.8 GB
      int8 PLE file `r31.ple_convert` writes beside it) - only if `df -h ~` has it.
      **Session 11:** +41.9 GB (Bolt g128), plus its golden sets and A4 outputs (unmeasured, budget 5 GB).
      **Session 12:** +149.8 GB for the BF16 original **while the window lasts**, plus row 58's two g64
      exports (RTN and ours, ~40.5 GB each - spec 25's 40.46 GB of device bytes; the files ESTIMATED near
      it) and its calibration data (unmeasured): **want >= 240 GB free when it starts**. After it the
      BF16 original goes and the two exports stay (~81 GB).
- [ ] **RAM** (121 GB on the box): one CPU oracle at a time; 9b part 2's `r31.load_real` pins the
      51.8 GB PLE table (MemAvailable - 16 GiB rule: `free -g` >= 70 GB, nothing else running), the
      real-weight references SKIP below MemAvailable 64 GB (`Q4_REF_MIN_GB`); `r14.oracle` needs
      MemAvailable >= 32 GB (`K2_REF_MIN_GB`), `r24.oracle_synth` >= 24 GB (`KOL_REF_MIN_GB`);
      never during `g0.build` (a compile slows a CPU oracle by ~28 %, docs/10, and both want RAM).
      Bolt (`BOLT_REF_MIN_GB`): the g128 stages want MemAvailable >= 48 GB, the BF16 ones >= 96 GB;
      `r58.tune` refuses below 64 GB free (ESTIMATED) - so in Session 12 a BF16 CPU stage and `tune`
      never overlap (121 GB).

### 1.6 Build, and the optional CPU lane

- [ ] The builds are stages, not a separate step: `g0.build` builds the tree under test **and**
      the baseline (Release, `-j44`: `JOBS` defaults to 44 on the box), `r1.sweep_build` a third
      tree with `-DB70_AGNES_SWEEP=ON`. `g0.build` is the first ocloc compile of ~250 kernel
      binaries written blind (rows 8, 10, 11, 13-16, 19, 21, 22, 24, 26) - the most likely
      first-day stop (§5).
- [ ] **Optional, saves ~1.5 h of serial time:** once `g0.build` is PASS and while sessions 1-2
      run (no timed stage in either), bake the two CPU oracles on the box, one after the other,
      capped so the GPU tests keep their host threads:
      ```sh
      export REMOTE_DIR=b70-inference-server-validate
      tools/box.sh run 'ORACLE_THREADS=22 tools/probe/detach.sh $HOME/oracle-k2.log tools/box_validate/k2_oracle.sh $HOME/b70-inference-server'            # ~20 min (est.), needs the K2 download
      tools/box.sh run 'ORACLE_THREADS=22 tools/probe/detach.sh $HOME/oracle-kol.log tools/box_validate/kolibri_oracle.sh $HOME/b70-inference-server synth'  # ~1-1.5 h (est.)
      ```
      Both are resumable and skip what exists, so `r14.oracle` / `r24.oracle_synth` in sessions
      5 / 9 then only re-link the sets and record the gap distributions.
- [ ] **For session 9b, on the CPU lane before it** (opt-in CPU stages, one at a time, once G0
      is PASS - `--only` needs it - and never during sessions 6-9): the synthetic Qwen3.8-Flash-Next checkpoints from the
      original's small files, their golden sets, and the MTP head's M1 data. Without them every
      synthetic GPU stage of rows 31-34 SKIPs "missing data":
      ```sh
      tools/box_validate.sh --only r31.synth            # ~27 GB, make_synth.py + ple_int8.py (est. < 1 h)
      tools/box_validate.sh --only r32.oracle_synth     # the golden sets, ~0.5-1 h each form (est.)
      tools/box_validate.sh --only r34.fixture          # M1's data, ~0.5 h (est.)
      ```
      (The Mac has `oracle-out-q4exp-synth/{ours,intel}` - 26 GB, checkpoints only, no golden sets -
      but over the box's flaky WiFi making them on the box is the safer path; by hand:
      `rsync -a oracle-out-q4exp-synth "$BOX:b70-inference-server/"`.)
      **Part 2 (optional, real weights, after Intel's download):** `--only r31.ple_convert` (the int8
      PLE file, ~51.8 GB) and `--only r30.golden_layers` (21c's `--layers 4 / 18` sets, HOURS) - they
      turn `r31.load_real`, `r32.partial`, `r33.golden_real` from SKIP into real runs; `r30.traces`
      (spec 22 P0.6 / P0.8's routing traces) and `r34.accept` (decision 4's MTP acceptance) are the
      other real-weight CPU stages worth the time, both HOURS.
- [ ] **Also for the CPU lane - queue row 35, the W4A4 probe** (plan
      `docs/superpowers/plans/2026-10-09-w4a4-probe.md` Task 2; Task 1's tool is on the tree):
      `tools/box_validate/w4a4_grid.sh` runs `eval_quantised.py sim <Qwen3.8 bf16> <gate checkpoint>
      --prompt <p> --variants none,w4a16,h8,h4,h4p2 --group 256 --vocab-used 248077` on cjk / code /
      heldout (478) / long512 in the oracle container, `free -g` printed and MemAvailable >= 70 GB
      first (else SKIP), resumable, into `~/b70-inference-server/oracle-out-w4a4/`; needs the bf16
      `Qwen/Qwen3.8-27B` (54 GB) in the HF cache. Detached, capped like the oracles above:
      ```sh
      export REMOTE_DIR=b70-inference-server-validate
      tools/box.sh run 'ORACLE_THREADS=22 tools/probe/detach.sh $HOME/w4a4-grid.log tools/box_validate/w4a4_grid.sh $HOME/b70-inference-server grid'        # hours (est.)
      tools/box.sh run 'ORACLE_THREADS=22 tools/probe/detach.sh $HOME/w4a4-class.log tools/box_validate/w4a4_grid.sh $HOME/b70-inference-server per-class'  # hours (est.), after grid
      tools/box.sh run 'tools/box_validate/w4a4_grid.sh $HOME/b70-inference-server summary'   # the tables and the stopping rule
      ```
      or as runbook stages: `tools/box_validate.sh --only r35 --with r35.grid` (then
      `r35.per_class`; both opt-in, CPU, and G0 must have passed in that state). §15.4's 478 held-out
      ids are not in the tree: `W4A4_HELDOUT=<file>` if the box still has them, otherwise the script
      re-makes a 478-id held-out prompt from this tree's texts (read h4 against the same run's
      none / w4a16 / h8). **Never during sessions 6-9**: a running container makes every timed row
      ITERATE.

### 1.7 Idle protocol (before sessions 6, 7, 8, 9)

- [ ] `REMOTE_DIR=b70-inference-server-validate tools/box.sh run tools/box_validate/idle.sh now`
      right before launching - an idle reading expires within minutes. It prints
      `grade=RECORD|ITERATE`, the container count and every DRM holder (pid, comm).
- [ ] Holders are the operator's to end: GNOME `baobab` / `ptyxis` service daemons ignore
      SIGTERM and need `kill -9`; nautilus, chrome, gnome-control-center and resources open both
      B70s too; killing Xwayland takes the desktop down - ask. **We never kill anything on the
      box**: report pid / comm and the command.
- [ ] Zero containers (no CPU oracle, no vLLM). Timed stages print their own `IDLE before` /
      `IDLE after` lines; quote the grade they print, never upgrade it.
- [ ] Rows 22, 23, 24, 26, 27, 28 need **both cards free of other DRM holders** (their GPU
      commands set `ZE_AFFINITY_MASK=0,1` themselves; no other row does). Series rows stay on
      device 0 (`DEVICE` defaults to 0): device 1 is ~3.4 % slower on prefill (docs/10).
- [ ] No second stream (§1.8) during an idle session: `tools/box_validate.sh --status` lists
      both orchestrators; `tools/gpu_lock.sh status` on the box shows every GPU lock (`box free`,
      both cards `free`, the gate `free`).

### 1.8 Two streams, one per card

The lock protocol (`tools/gpu_lock.sh`'s header has it in full; flock(2) on files in `~`):

| job | takes, in this order | runs beside |
|---|---|---|
| card N: a `card` stage of the device-N orchestrator, or `tools/gpu_lock.sh card N -- <cmd>` | `~/b70-gpu<N>.lock` exclusive, the gate `~/b70-gpu.wait` for an instant, `~/b70-gpu.lock` **shared**; `ZE_AFFINITY_MASK=N` | a card job on the other card |
| whole box: a `gpu` stage, or `tools/gpu_lock.sh box -- <cmd>` | the gate exclusive while it waits, then `~/b70-gpu.lock` **exclusive** (the gate dropped) | nothing |
| whole box, the old way: `flock ~/b70-gpu.lock <cmd>`, `gpu-self` scripts | `~/b70-gpu.lock` exclusive | nothing |

- **The rule:** a card job starts when its card is free and no whole-box job runs **or waits**; a
  whole-box job starts when no job holds either card. Once a whole-box job waits, new card jobs
  stop at the gate: the running ones drain and it runs next, so two overlapping card streams
  cannot starve a timed stage. flock(2) alone would - Linux grants a shared lock whenever no
  exclusive one is *held*, a waiting one does not stop it. The old `flock ~/b70-gpu.lock` form
  still works and excludes everything, but has no place at the gate: while card streams run,
  take the box by hand with `tools/gpu_lock.sh box -- <cmd>` (also under `detach.sh`), and run
  the opt-in `gpu-self` rows (their scripts lock the old way) only when no card stream runs.
  Card -> gate -> box is the one order: no deadlock. Never nest (a job holding a lock must not
  take another: flock on a second open of the same file waits for itself).
- **Two orchestrators, one tree, one state.** `~/b70-validate/<tree>/.run-<N>.lock` per device
  (`.current-<N>` names its state; an orchestrator from before this, holding `.run.lock`, blocks
  both). Each stage is claimed (`<id>.claim`) while it runs: the other orchestrator leaves a
  claimed stage ("RUNNING under the other orchestrator - left to it"), and a RUNNING status whose
  claim is free is a dead orchestrator's - it runs again. G0 runs alone (DEFERRED while the other
  is active); a stage that builds holds `.build.lock`; cpu stages of the two run one at a time
  (`.cpu.lock`: the CPU oracles' RAM, the host tests' fixed `/tmp` directories). Servers listen on
  8013 (device 0) / 8113 (device 1). `progress.log` lines say `[d0]` / `[d1]`, each status line
  and the summary's `device` column the device and lock a stage ran with; `--dry-run` and
  `--list` print the lock each stage will take.
- **Launching the second stream** - from the same frozen worktree (§1.1), so the same commit and
  state, after the first one started:
  ```sh
  tools/box_validate.sh --device 1 --only <stages> --no-wait   # the driver sees device 0's run: no sync, no push
  tools/box_validate.sh --status                               # both orchestrators, every stage with its device
  ```
  The driver refuses another state or commit beside a running orchestrator, and `--push-data`
  (push before the first launch). Device 1 runs correctness only: a timed or series row belongs
  on device 0 - the device-1 dry-run names every whole-box stage in its selection.
- **Split by row family, not inside a chain.** `after` / `need_pass` read the shared state: a
  stage whose prerequisite is RUNNING in the other stream at that moment is SKIP "needs PASS
  first" (relaunch it later). A `gpu` stage inside a stream still waits for the other stream's
  running card stage and then holds the whole box.
- **Host RAM:** two engines load at once (each checkpoint read through the page cache) and each
  b70-serve's auto prefix cache pins up to 32 GiB, sized from MemAvailable at its start. Stages
  that check a host-RAM budget (`free -g`: Qwen3.8-Flash-Next's 51.8 GB
  PLE table) and the passkey probes (fixed `/tmp/*passkey` directories) stay `gpu`.

Which sessions split (`tools/box_validate.sh --list --only <rows>` shows each stage's lock):

| sessions / rows | how | notes |
|---|---|---|
| 3, 4 and 5 | device 0: session 3, then session 4; device 1: session 5 from the start | rows 14, 15, 21 are `card` + `cpu` only; `r13.cli` (timing rows) and `r25.passkey` take the whole box; `r14.oracle` (cpu, ~20 min) holds the cpu lock, so session 4's host stages wait for it |
| 6, 6b, 6c, 7, 8, 9c | no split | timed / idle-graded / two-card |
| 9, 9b | no split by default | the synthetic one-card gates are `card`, but `r24.pp` / `r26.pp` / `r28.*` / `r3x.pp` need both cards; a card-only device-1 stream of row 26 or 33 beside device 0's other rows is possible once row 24 / 32 has passed |
| later rows (no session yet) | pairs of one-card families: `--only r54,r55,r56` (DFlash) on device 0 with `--device 1 --only r60,r61` (Bolt synthetic) | `--skip` the timed stages (`r60.p0`, `r60.fuse` - in the device-1 stream) and run those later on device 0 with the box idle; each row's two-card stages (`*.pp`, `*.synth_pp`, `r60.golden`, `r61.golden`, `r56.pp_serve`) take the whole box when they come |

```sh
# sessions 3-5 on two cards (the frozen worktree, G0 PASS in this state)
tools/box_validate.sh --only r19                                                # device 0, ~15 min
tools/box_validate.sh --device 1 --only r14,r15,r21,r25 --with r14.oracle --no-wait   # device 1: session 5
tools/box_validate.sh --only r10,r13,r16 --skip r16.cost --no-wait             # device 0: session 4
tools/box_validate.sh --status
```

## 2. The sessions

Each session is one launch; start the next only after reading the previous one's
`progress.log` / summary. Per-session estimates come from these anchors (all estimates unless
marked measured): the whole suite at 103 tests took 29 min on 2026-09-27 (measured,
docs/probe-mtp-2026-09-27.md); Qwen3.8 loads in 13.6 s on a warm page cache (measured,
docs/10); pp4096 at 2125 t/s and pp130816 in 175 s (746.6 t/s), decode ~30 t/s, ~20 t/s at
128k (measured, BENCHMARKS); the stage titles' and rownotes' own figures (`r14.load` ~5 min a
test, `r14.oracle` ~20 min, `r24.oracle_synth` ~1 h, `r8.m3_small` ~1.5 h per size, `r22.s1_32k`
~20 min a `--pp 2` arm, `r23.s2_128k` ~2.5 min a one-card run, `r27.p3` ~1 h a placement);
everything else is ~10-30 min a checkpoint test stage.

### Session 1 - G0 (stop the line)

```sh
tools/box_validate.sh --only g0
```

Stages: `g0.build`, `g0.sha`, `g0.bitwise`, `g0.suite` (+ `pre`). Device 0.
Estimate 2.5-4.5 h: two full builds 0.5-1 h (unmeasured), `g0.bitwise` (11 gate tests on
both builds) ~1 h, `g0.suite` (the baseline's whole suite minus the agnes / ornith / kv8 labels,
on the new build) 1-2 h - the 103-test suite took 29 min, the baseline registers more,
including the long prefill gates.

**Go / no-go:** all four PASS, or the day stops here. `g0.sha` lists the added binaries and any
pre-existing one that changed (a changed one is a finding, not noise); `g0.bitwise.diff` is the
line-by-line difference of the gate outputs. A FAIL means a fix on a branch, a new commit and
session 1 again. If `g0.build` fails on one family's kernels, `CMAKE_ARGS='-DB70_KOLIBRI=OFF'`
(or `B70_K2` / `B70_ORNITH` / `B70_KV8`) builds the rest so G0 can still be judged - that is
triage, not validation: the family's rows then wait for the fix.

### Session 2 - one-card gates: Agnes and the Qwen3.8 feature rows

```sh
tools/box_validate.sh --only r1,r2,r4,r5,r6,r7,r8,r9,r11,r12,r17,r18 \
  --skip r1.sweep,r1.speed,r2.cost,r2.auto_rows,r6.cost,r8.cost,r8.rows,r11.speed --no-wait
```

40 stages, device 0, in this order: row 1 (`r1.host` ... `r1.passkey`, incl. `r1.sweep_build`),
`r2.suite`, `r4.g64`, `r5.host`, `r5.r0`, row 6 (`r6.memory`, `r6.mtp_long`,
`r6.greedy_long`), row 7 (`r7.plan` ... `r7.serve_agnes`), row 8 (`r8.kernels`, `r8.m3`,
`r8.off`, `r8.memory`, `r8.serve`), `r9.auto`, row 11 (`r11.bf16`, `r11.kernels`, `r11.q2`,
`r11.q5`, `r11.memory`), row 12 (`r12.suite`, `r12.d2`, `r12.a4`), `r17.host`, `r17.k0`,
`r18.host`, `r18.g64`. The timed stages of these rows wait for session 6.
Needs: `agnes`, `oracle_agnes`, `oracle_agnes_mtp`, `tok_python`, `oracle_image`.
Estimate 8-13 h (an overnight run): row 1 ~2-2.5 h (the sweep build ~0.3-0.6 h, 72-layer
gates, features, passkey at 60k), row 6 ~1.5-2 h (`r6.mtp_long` at 32k and 128k), `r8.m3`
~1.5 h (128k, both heads; `r8.m3_small` is ~1.5 h per size), row 7 ~1 h, row 11 ~1.5-2 h,
the rest ~2-3 h.

**Check:** `r1.gates` and `r1.features` PASS (`r5.r0` needs both, `r10.r0` and `r13.r0` need
`r1.gates`);
`r11.kernels` PASS (a failure stops row 11 by design); `r8.kernels` PASS (the 9 draft-vocab
binaries' first run).

### Session 3 - Ornith's real checkpoint (row 19)

```sh
tools/box_validate.sh --only r19
```

`r19.host`, `r19.r0`, `r19.load`, device 0, ~15 min (est.). Run it on its own and first: the
int4 a||b load (64 real columns, the 15.7 MB prefill copy, the W check against doc_w
2.345 GB) is what every Ornith gate stands on. **Check:** `r19.load` PASS before session 4.

### Session 4 - Ornith decode, prefill, serving (rows 10, 13, 16)

```sh
tools/box_validate.sh --only r10,r13,r16 --skip r16.cost
```

21 stages, device 0: `r10.r0`, `r10.nockpt`, `r10.load`, `r10.gates`, `r10.refusals`,
`r13.r0`, `r13.host`, `r13.kernels`, `r13.prefill`, `r13.gates`, `r13.split`, `r13.cli`,
`r16.r0`, `r16.host`, `r16.kernels`, `r16.mtp`, `r16.m1`, `r16.serve`, `r16.golden_server`,
`r16.prefix`, `r16.tokdiff`. Needs `ornith`, `oracle_ornith`, `oracle_ornith_mtp`.
Estimate 3-5 h (row 10 ~1 h, row 13 ~1.5 h, row 16 ~1.5-2 h).
**Check:** `r10.nockpt` (moe.cl's sub-group path and the 288-lane `moe_down` on the card for
the first time), `r13.kernels` (grouped == dense bitwise) and `r16.kernels` PASS.

**The f32 judge (operator ruling 2026-10-23).** Ornith's token gates (`golden_gate_ornith*` in
`r10.gates`, `prefill_gate_ornith_*` in `r13.gates`) judge the engine against the f32 reference
teacher-forced on the bf16 golden's tokens, `oracle-out-ornith-f32` (data key
`oracle_ornith_f32`); the bf16 golden is printed beside it with its disagreements marked. Why:
on the box's first run the bf16 golden's own determined rows disagreed with exact arithmetic
(Ornith prose 18, cjk 14, code 3) and the engine had taken exact arithmetic's token where it
"failed". Make it once, on the box CPU, before the session (or `--with r10.oracle_f32`):
`tools/box_validate/f32_oracle.sh ~/b70-inference-server ornith` - ~1.5 min a prompt, peak RSS
18-21 GiB, 22 threads under `/tmp/b70-cpu-model.lock`, SKIPs below MemAvailable 95 GB (70 GB kept
for the other jobs + the run) - then `tools/box_validate/data.sh link ~/b70-inference-server
<tree>`. Qwen3.8's and Agnes's gates take no judge.

**The quantisation-aware judges (operator ruling 2026-10-23, the third).** The int8 variants are
judged against f32 math that applies the same int8 roundings (`tools/oracle/int8_sim.py`, its codes
bitwise the engine's host references - `test_int8_sim.py`): `golden_gate_ornith_i8head` against
`oracle-out-ornith-f32-i8head` (the per-row int8 head of `loader/lm_head_int8.cc`; key
`oracle_ornith_f32_i8head`), `prefill_gate_ornith_l0-int8*` against `oracle-out-ornith-f32-pfi8`
(the prompt forward through spec 5's h8 linears: rotation, per-token and per-column int8; key
`oracle_ornith_f32_pfi8`). Made with the f32 judge: `f32_oracle.sh ~/b70-inference-server
ornith-i8head` and `... ornith-pfi8` (`r10.oracle_f32` runs all three; each ~5 min, the same lock
and 95 GB rule). A miss against them is not the variant's quantisation.

**Known divergences (operator ruling 2026-10-23, debug Monday 2026-10-26 evening).** Accepted per
registration with `--accept=<prompt>:<step>,...` (tests/golden/golden_common.h): the row prints
"ACCEPTED (known divergence, 2026-10-23, debug 2026-10-26)" with its judge gap and does not fail;
any other failing row still fails; an accepted row that passes prints "accepted row now passes -
remove it". The rows, and what is known (card 0, 2026-10-23):
- `golden_gate_ornith_i8head` code 3 (judge gap 0.138, ulp 0.125), cjk 15 (0.125, ulp 0.0625).
- `prefill_gate_ornith_l0-int8` prose 22 (0.469), cjk 15 (0.118). The same kernels at chunk 16
  pass both; the GDN state is as far from the pfi8 judge at chunk 2048 as at 16 (0.99476 / 0.99484
  prose), and the failing run's row-22 logits are the closer to the judge (cos 0.99709 / 0.99534).
- `prefill_gate_ornith_l0_c16` (bf16) prose 18 (0.447), cjk 15: chunk widths 2048 / 64 / 32 / 16 / 8
  leave the state's distance to f32 flat (0.996 prose) and flip the rows non-monotonically
  (prose 18 wrong at 32 and 16, right at 8 and 2048) - chunk-boundary rounding of the GDN scan.
- `k2_golden_i8head` prose 1 (0.136): gemv_i8w recomputed on the host from the device's own
  hidden and int8 rows agrees to 4e-6 (`B70_K2_HEAD_CHECK`) - the divergence is the hidden state;
  the bf16 head on it picks the judge's 1961.
cjk 15 is a 3-4-way f32 near tie (four ids within 0.31) that different variants resolve
differently.

**Bars still to set for Ornith (runs written down, not made - 2026-10-23).**
- `r16.m1` (`mtp_head_ornith_test`, per-row cosine >= 0.999): every reference argmax is in the
  engine's top 2; only the cosine bar fails (worst 0.9878) and the bf16 golden's own main logits
  are 0.9928 from f32 on the same rows. Needed: an f32 MTP-head reference - `ornith_ref.py run
  --dtype f32 --force <oracle-out-ornith's tokens> --mtp-out <dir>` per prompt (the head on the
  f32 run's hidden; the head's experts RTN as now). The code path needs a small extension first:
  `mtp_head()` builds its experts in bf16 and would have to follow `--dtype f32`. Cost as the
  f32 judge plus the head: ~2 min a prompt, ~22 GiB peak - fits beside the Bolt job under the
  95 GB rule. The bar is then the bf16 M1 reference's own cosine to it.
- `prefill_split_ornith_*` (Qwen3.8's provisional bars; fa0-3 rows 0.85-0.97 at rows 986 / 1104,
  GDN L0 state 0.999998, multiples of 64 bitwise - Agnes's pattern): needed, as for Agnes, a CPU
  reference over `long32k.ids[0, 2600)` for `prefill_split_test --oracle` - `dump.py`-style
  logits for the listed rows, i.e. `ornith_ref.py` with a prompt of 2600 ids (`--max-prompt 2600
  --gen 1`). Not measured for Ornith; K2's comparable 2600-id bf16 run took 66 min at 22 threads
  (the run is mostly single-threaded in the expert loop) with 28.8 GiB peak; expect Ornith
  similar (~1-1.5 h, ~30 GiB) - fits under the 95 GB rule, but not beside a second CPU job.

### Session 5 - K2-Horizon (rows 14, 15, 21, 25)

```sh
tools/box_validate.sh --only r14,r15,r21,r25 --with r14.oracle
```

35 stages, device 0: row 14 (`r14.k0`, `r14.host`, `r14.k1`, `r14.load`, `r14.k3`,
`r14.oracle`, `r14.golden`, `r14.golden_eager`, `r14.cli`), row 15 (`r15.k0`, `r15.host`,
`r15.k1`, `r15.prefill`, `r15.split`, `r15.golden`, `r15.cli`), row 21 (`r21.k0` ...
`r21.cli`), row 25 (`r25.k0`, `r25.host`, `r25.reject`, `r25.snapshot`, `r25.snapshot_kv8`,
`r25.serve`, `r25.chat`, `r25.passkey`, `r25.toolcall`). The order inside the launch is the
dependency order: 14 before 15 (`r15.prefill` after `r15.k1`), both before 21 (every `r21.*`
GPU stage waits for its bf16 sibling), all before 25.
Needs `k2`, `oracle_image`; `oracle_k2` is made by `r14.oracle` within the launch (or by the
CPU lane, §1.6).
Estimate 4.5-7.5 h (`r14.load` ~10 min, `r14.oracle` ~20 min, row 15 ~1.5 h, row 21 ~1.5-2 h,
row 25 ~1.5 h with `r25.passkey` ~0.5 h).
**Check:** `r14.k1` and `r15.k1` (the 45 K2 binaries' first run) PASS; then read `r14.oracle`'s
gap distribution before trusting `r14.golden` at the proposed 1e-3 (§4). `r25.toolcall` SKIPs
without `oracle-out-k2-a4/set` (§1.4).

**Since the operator ruling of 2026-10-23:** K2's token gates (`r14.golden`, `r15.golden`,
`r21.golden*`) judge against the f32 teacher-forced reference `oracle-out-k2-f32` (data key
`oracle_k2_f32`; make it with `--with r14.oracle_f32` or by hand:
`tools/box_validate/f32_oracle.sh ~/b70-inference-server k2`, ~2.5 min a prompt, ~17 GiB peak,
the same lock and 95 GB rule as Ornith's, session 4), the bf16 golden printed beside it; the
routing gate is a rate (the bf16 reference's own rate against f32: MoE 0.2010, MoVA 0.1096).
Flash stays the default (spec 18 §10.1, amended); the eager stages (`r14.golden_eager`,
`r15.golden_eager`, `r21.golden_eager`, and `k2_prefill_eager*` inside `r15.prefill` /
`r21.prefill`) are DIAGNOSTIC: ctest label `diagnostic`, their routing / set / dense-worst bars
printed, not failed on.
The `*_i8head` K2 gates judge against `oracle-out-k2-f32-i8head` (k2_ref.py `--lm-head int8`; key
`oracle_k2_f32_i8head`; `f32_oracle.sh ~/b70-inference-server k2-i8head`, made by `r14.oracle_f32`).
Known divergence: `k2_golden_i8head` prose 1 is accepted until 2026-10-26 (session 4's "Known divergences").

### Session 6 - timed one-card rows, box idle

Idle protocol (§1.7) first: both cards free, zero containers, the CPU lane stopped.

```sh
tools/box_validate.sh --only r1.sweep,r1.speed,r2.cost,r2.auto_rows,r3,r6.cost,r8.cost,r8.rows,r11.speed,r16.cost
```

10 stages, device 0: `r1.sweep`, `r1.speed`, `r2.cost`, `r2.auto_rows`, `r3.readme`,
`r6.cost`, `r8.cost`, `r8.rows`, `r11.speed`, `r16.cost`. Prerequisites from earlier sessions:
`r1.sweep_build`, `r6.memory`, `r11.q2` (session 2), `r16.mtp` (session 4).
Estimate 7-11 h: `r1.speed`, `r3.readme`, `r11.speed` ~1-1.5 h each (depth rows: every arm
re-ingests to 32k / 64k / 128k, 175 s per 130816-token prefill, median of 3 after a warm-up),
`r2.auto_rows` and `r8.rows` ~1.5-2 h each (4 server arms x 3 rounds; 8 bench arms x 3),
the cost tables ~0.3-1 h each.
**Check:** every row's `IDLE before` / `IDLE after` grade; a row taken ITERATE is still a
reading, not a record.

**Added 2026-10-23 - row 67, the prefill bandwidth profile** (opt-in, ~20-30 min, est.): hw-smi showed 600+ GB/s of
the card's 638 GB/s during `r1.features`' Agnes `prefill_split_test` (l0-int8, long32k) - the highest bandwidth any of
our kernels has reached; decode runs ~445 GB/s effective. Append `,r67.profile` to the launch above (or run it alone,
idle): `B70_PREFILL_PROFILE=1` per-phase GPU time at 32k on Agnes and Qwen3.8, l0 and l0-int8, then the four arms
unprofiled. Watch hw-smi by hand during it; the per-phase bytes and GB/s are derived afterwards on the Mac
(docs/BENCHMARKS.md), to find which kernel streams at ~94 % and what the decode kernels can take from it.

### Session 6b - DFlash verify cost at M = 5..8 (plan 19a Task 4), box idle

The one number DFlash's go / no-go still needs. P0 (Tasks 2-3, `docs/probe-dflash-2026-10-08.md`)
measured the acceptance on the Mac: 7.21 tokens per verify at K = 7 on the A4 tool calls, 6.15 on
code, ~2.2 on prose; its projection against `--mtp auto` (+54 % / +65 % / +17 %) **extrapolates**
verify(M) past spec 8's measured M = 2..4 (1.17 / 1.52 / 1.74 plain steps) to M = 8 = 2.62. If
verify(8) comes out near 3.5, A4's margin shrinks to ~+17 %.

**The probe build** (plan 19a Task 4, branch `spec19a-task4-probe`, written blind on the Mac):
`B70_VERIFY_M8` (default ON, so G0's `build` has it; `g0.sha` lists the binaries as added) builds
Qwen3.8's verify list at M = 5..8 - the five int4 GEMVs, the int8 head, a||b, the norms, embed,
argmax, `attn_prep` / attention v2 at M rows (attn_v2.cl now allows M <= 8), the MTP head's KV fill,
and `gdn_step_slots_M<M>_N8` (gdn_step.cl's slot count became overridable; 8 slots) - plus
DFlash2's drafter GEMVs at M = 8 in int8 / bf16 / int4 g64 and the int8 head at 7 rows (full and
32k). No engine binds them. `probe_mtp_steps` takes an eighth argument, `max_m` (default 4: every
existing command line, r2.cost / r6.cost / r8.cost / r16.cost, is unchanged): at 5..8 it builds the
M = 5..8 lists over a probe-owned 8-slot `MtpBuffers` (+1.06 GB) beside the engine's own M = 1..4,
rewinds pos to the depth before every arm, and adds a table of **interleaved pairs** (verify M = 1
and verify M back to back, the order alternating by round, median of 3). `probe_draft_cost` times
one DFlash2 block's GEMVs on random weights of the checkpoint's shapes (5 layers x {attn conv
proj, q||k||v, o_proj, mlp conv proj, gate||up, down, the commit's ctx k||v} + fc + the selector
projection; 37 launches) and the int8 head's 7 rows; `--plain-ms` prints shares of a plain step.

Run on an idle box (§1.7), device 0, after session 6 (same idle conditions, shares its warm-up):

```sh
tools/box_validate.sh --only r29.cost
```

which runs (stage `r29.cost`, `tools/box_validate/stages.sh` row 29):

```sh
# int8 head, verify M = 1..8 and draft k = 1..3, rotated rounds + interleaved pairs, median of 3
build/tools/probe/probe_mtp_steps $SNAP_QWEN 4096 32 3 int8 off 16384 8
build/tools/probe/probe_mtp_steps $SNAP_QWEN 32768 32 3 int8 off 65536 8
# one DFlash2 block's GEMVs (int8 / bf16 / int4 g64 at M = 8, the int8 head's 7 rows at V 248320
# and 32768), random weights; --plain-ms = the 4k run's verify M=1 ms
build/tools/probe/probe_draft_cost --calls 20 --rounds 3 --plain-ms <verify M=1 ms at 4k>
```

Read: the `| pair M=<M> |` lines (verify(M) / verify(1) as interleaved pairs; the rotated table
above them is the cross-check), the `| block <format>, head V <V> |` lines (the draft cost in
plain steps), and the per-linear rows (GB/s - where an M = 8 GEMV leaves the bandwidth regime).
The verify rows include the MTP head's 10-launch KV fill, as spec 8's M = 1..4 rows do (+1.3 % at
M = 1); a DFlash verify does not run it, so they are an upper bound by about that much.
Estimate 0.25-0.5 h (est., from assumed call times: two loads and prefills, then per depth ~11
arms x 104 calls and 7 x 3 pairs of 32 calls at 35-90 ms a call, ~2-4 min; the draft probe fills
~8 GB of random weights, then runs for seconds).

Then plan 19a Task 5 (a Mac edit): re-run `docs/probe-dflash-2026-10-08.md` §3's table with
the measured costs and record the verdict in spec 19 (`≥ 10 %` over `--mtp auto` on the coding
and agentic sets = go for 19b).

### Session 6c - the llama-benchy comparison matrix (row 41), box idle

So our numbers read the same way as vLLM's and as tables others publish: one client (llama-benchy
pinned at 0.4.0), one set of flags, one card, the same Qwen3.8 int4 g64 checkpoint for every server.
Run on an idle box (§1.7), device 0, right after sessions 6 / 6b (the same idle conditions):

```sh
tools/box_validate.sh --only r41.ours,r41.vllm,r41.vllm_mtp
```

Each stage serves one engine at a time (never two on the card) and runs three matrices: **A** =
`--pp 512 4096 --tg 128 --concurrency 1 2 4 8 --runs 2 --format md` verbatim (the usual published
form; a repeated prompt may hit a prefix cache), **B** = A with `--no-cache` (the fair
cross-engine reading), **D** = `--pp 512 --tg 128 --depth 4096 32768 --concurrency 1 --runs 2
--no-cache` (agentic depth). Arms: `b70-serve --mtp 0` and `--mtp auto` (`r41.ours`), vLLM plain
(`r41.vllm`), vLLM's MTP-4 + fp8-KV recipe (`r41.vllm_mtp`; if the XPU image refuses or crashes on
it, `~/benchy-vllm.log` is the reading - no patches).

Read: the `| pp... | tg... |` table lines per arm, the `client:` line (version and server flags) and
`vllm version` (never the image tag). b70-serve's concurrency 2 / 4 / 8 rows are requests served one
after another until spec 13b / 13c's batched decode - recorded as the baseline batching must beat.
Estimate 2-3.5 h (est.: 4 server arms x 3 matrices; vLLM's startup with graph capture ~5-15 min a
launch, 6 launches; each A / B matrix 16 cells x 2 runs).
Then (a Mac edit) the rows go into docs/BENCHMARKS.md "llama-benchy comparisons".

### Session 6d - the batching probe (row 66, plan 13a), box idle

Added 2026-10-23. Rows of DIFFERENT sequences in one decode step (spec 13 §9): the numbers 13b's three decisions wait
on - the slots per model, the 8-row budget's allocation and the interleave chunk (docs/probe-batching-2026-10-23.md
§7). The probe build is its own tree's second build dir, `build-batch-probe` (`-DB70_BATCH_PROBE=ON`, OFF in G0's
build): the 80 row builds' first ocloc compile, ~10-20 min at -j44 (est.).

```sh
tools/box_validate.sh --only r66                                # k0, the build, the check (one card, ~20-40 min est.)
# then, on an idle box (§1.7: both cards free of DRM holders, zero containers, the CPU lane stopped), device 0:
tools/box_validate.sh --only r66.cost,r66.chunk
```

`r66.check` runs `probe_batch_step --no-time --no-chunks` five times (Qwen3.8 int8 / bf16 KV, Agnes int8 / bf16 with
MTP at 2 slots, Agnes int8 `--mtp off` at 4): every arm's rows bitwise their slot's run alone, isolation, the production
tie - a `check: FAIL` stops 13b (report the CHECK lines first). `r66.cost` times every arm against `1x1@4k` as
interleaved pairs (median of 3) for Qwen3.8 at both KV forms and Agnes's two layouts; `r66.chunk` one prefill chunk of
512 / 1024 / 2048 at 4k and 32k beside the 1-row steps. Each run prefills 32k + 3 x 4k (or 32k + 4k) first (~30-60 s,
est.). Estimate for the timed pair: 1-2 h (est.: ~16 arms x (3 rounds x 2 x 32 replays + warm-up) at 35-100 ms a
replay per configuration, four configurations, plus the chunk rounds). **Check:** the `IDLE before` / `after` grades.

### Session 7 - pipeline parallel decode, then prefill (rows 22, 23)

Idle protocol (§1.7): **both cards free of DRM holders.**

```sh
tools/box_validate.sh --only r22,r23
```

22 stages, both cards: `r22.host`, `r22.r0`, `r22.devices`, `r22.p1`, `r22.p1_kv8`, `r22.p4`,
`r22.cli`, `r22.agnes`, `r22.ornith`, `r22.s1`, then `r23.host`, `r23.r0`, `r23.reject`,
`r23.p1`, `r23.p1_kv8`, `r23.p4`, `r23.cli`, `r23.agnes`, `r23.ornith`, `r23.s2_4k`,
`r23.s2_32k`, `r23.s2_64k`. Row 22 strictly before row 23: `r23.p1` is `after r22.p1`,
`r23.reject` / `r23.p4` after `r22.devices`.
Estimate 4.5-7 h: `r22.p1` and `r23.p1` ~0.75-1 h each (copy and peer x auto split and the
cuts at 5 and 62, plus restores; every run a two-card load), the `_kv8` twins ~0.3 h each,
`r22.s1` ~0.75 h (3 arms x 4 runs at depth 4096, ingested through the decode lists), the S2
stages ~1-1.25 h together.
**Check:** `r22.devices` prints `devices: 0 ..., 1 ...` with peer access; `r22.p1` and `r23.p1`
PASS bitwise - rows 24, 26 and 27 wait on them.

### Session 8 - pipeline integration (row 27)

```sh
tools/box_validate.sh --only r27
```

14 stages, both cards: `r27.host`, `r27.k0`, `r27.reject`, `r27.mtp`, `r27.mtp_kv8`,
`r27.prefix`, `r27.prefix_kv8`, `r27.cli_mtp`, `r27.serve`, `r27.golden`, `r27.agnes`,
`r27.ornith`, `r27.s3`, `r27.p3`. Needs `r22.p1`, `r23.p1` PASS.
Estimate 4.5-7 h, of which `r27.p3` (passkey at ~250k on two cards, three placements, ~1 h
each derived) is ~3 h; the rest ~2-3.5 h.
**Check:** `r27.mtp` and `r27.prefix` PASS bitwise; `r27.serve` / `r27.s3`'s memory lines show
device 1's head + embedding replica (2.54 GB on Qwen3.8) and `max_len: auto -> 262144`.

### Session 9 - Kolibri-1 on the synthetic checkpoints (rows 24, 26, 28)

```sh
tools/box_validate.sh --only r24,r26,r28 --with r24.oracle_synth
```

31 stages, both cards for `r24.reject`, `r24.pp`, `r24.cli`, `r26.pp`, `r26.cli` and every
`r28.*` GPU stage; the one-card ones (`r24.k1`, `r24.load`, `r24.k3`, `r24.golden`,
`r26.k1`, `r26.prefill`, ...) run on device 0. Order: row 24 (`r24.k0`, `r24.host`, `r24.k1`,
`r24.reject`, `r24.oracle_synth`, `r24.load`, `r24.k3`, `r24.golden`, `r24.golden_eager`,
`r24.pp`, `r24.cli`, `r24.partial`, `r24.pp_real`), row 26 (`r26.k0` ... `r26.real`), row 28
(`r28.k0`, `r28.host`, `r28.reject`, `r28.snapshot`, `r28.serve`, `r28.chat`, `r28.real`).
Needs `oracle_image` and the Kolibri-1-BF16 tokenizer files (§1.4); `oracle_kolibri_synth` is
made by `r24.oracle_synth` (or the CPU lane). `r24.partial`, `r24.pp_real`, `r26.real`,
`r28.real` SKIP "missing data" (spec 20b).
Estimate 4.5-7 h: `r24.oracle_synth` ~1-1.5 h (0 if baked in the CPU lane), the synthetic
5-layer checkpoints load in seconds, ~2.5-4 h for the rest.
**Check:** `r24.k1` and `r26.k1` (the 40 Kolibri binaries' first run) PASS; `r24.pp` /
`r26.pp` bitwise; `r28.snapshot` (restores bitwise) PASS.

### Session 9b - Qwen3.8-Flash-Next on the synthetic checkpoints (rows 30-34, spec 21)

```sh
tools/box_validate.sh --only r30,r31,r32,r33,r34 --with r32.ple_rate,r32.bench_attn
```

32 default stages plus two cheap opt-in GPU probes, both cards for the `.pp` / `.cli` stages:
`r30.host` (CPU, the tiny model in the oracle image), the K0 recaps `r31.k0` .. `r34.k0`, the host
stages, `r31.plan`, `r31.load`, then per row the first ocloc compile of each blind kernel block
(`r32.k1` 33 binaries, `r33.k1` 30, `r34.k1` 122) and its tests, the synthetic gates (`r32.k3`,
`r32.golden`, `r33.prefill`, `r33.golden`, `r34.snapshot`, `r34.mtp`, `r34.serve`), the refusals, and
the two-card twins (`r32.pp`, `r32.cli`, `r33.pp`, `r33.cli`, `r34.pp` - each `after r22.p1`, so after
session 7). `r32.ple_rate` is spec 22 P0's first host-USM rate (no checkpoint; `free -g` >= 70 GB for
its 51.8 GB arm) and `r32.bench_attn` decision 10's sparse-vs-dense attention evidence.
Needs: the CPU-lane data of §1.6 (`oracle_q4exp_synth`, `oracle_q4exp_synth_golden`, M1's fixture),
the tiny model and the original's small files (§1.3). `r31.load_real`, `r32.partial` and
`r33.golden_real` SKIP "missing data" unless part 2's real checkpoint, PLE file and `--layers` sets
exist - expected, not a failure; re-run them with the same command once they do (a re-run repeats
only what is not PASS).
Estimate 3.5-6 h (est., unmeasured: three kbins blocks, 185 binaries in all; 4-layer real-width
synthetic checkpoints, ~13.5 GB each, load in about a minute; the `.pp` stages are two-card loads).
**Check:** `r32.k0` / `r33.k0` / `r34.k0` PASS (prep.cl, pf_gated_head.cl, q4_qsa.cl, q4_ple.cl
changed by defines only - every existing binary unchanged); `r32.k1`, `r33.k1`, `r34.k1` (first
runs); `r32.golden`, `r33.golden` (token gates); `r34.mtp`'s M2 bitwise; the `.pp` stages bitwise.
Hand back: `r32.ple_rate`'s rate lines and `r34.mtp`'s verify-cost lines (spec 22's inputs).
### Session 9c - spec 22's expert-offload probe (row 36, plan 22a), both cards idle

After spec 21's rows 30-34 (their data: row 30's traces and golden sets, `r32.k1`'s `q4_moe` binary, row 34's
A4 reference) and after Session 7's `r22.p1` (two cards, one context). Idle protocol (§1.7): **both cards free
of DRM holders**, RECORD grade for the timed stages, **no container and no CPU oracle up** (the pinned arms:
`r36.pin` SKIPs "host RAM" below 110 GB available or with any container running).

```sh
tools/box_validate.sh --only r36                                  # the default stages
tools/box_validate.sh --only r36 --with r36.prune,r36.prune_a4    # the CPU lane, later, alone
```

Order: `r36.k0`, `r36.host` (the budget lines -> `$DATA/oracle-out-q4exp-offload/budget.json`), `r36.build`
(the probe's two kernels, first ocloc compile), **`r36.links`** (P0.1 idle: a card below Gen3 x8 is a STOP -
spec 22 §3 says stop and report; an UNKNOWN line means `lspci -vv` needed root: run
`sudo lspci -vv -s 04:00.0` / `08:00.0` by hand for the record), `r36.links_load` (P0.1 under `probe_offload
hold`), `r36.copy` (P0.2), `r36.zc` (P0.3: the arms and the mixed-launch law), `r36.replay` (the visibility
hazard, 10^4 replays), `r36.hostbw` (P0.4), `r36.pin` / `r36.alias` (P0.5), then the CPU stages `r36.curve`
(P0.6, uses `r36.copy`'s measured expert copy), `r36.reap` (P0.8's masks) and `r36.project` (P0.7: the STOP /
GO-CANDIDATE line). A mismatch in `r36.replay` or `r36.alias` is reported first, with its replay / page.
Estimate 3-5 h of GPU time (est.: `copy` ~20 min a card set, `zc` ~30-60 min - 13 arms x 4 P x interleaved
pairs, 3 device sets -, `replay` ~20 min, `hostbw` ~10 min, `pin` ~30-60 min of pinning up to ~100 GiB, `alias`
~10 min) plus `r36.curve` up to ~1 h (the swap simulation at S 1). The opt-in CPU lane (`r36.prune` hours to
days, `r36.prune_a4` days, ESTIMATED from row 30's walls) runs one oracle at a time with `free -g` >= 70 GB,
never during the GPU stages. Then (a Mac edit) the record `docs/probe-offload-<date>.md` and spec 22's
decisions section (plan 22a Task 5 Steps 6-7); the operator's go / no-go follows.
**Check:** `r36.links` prints `links: dev0 Gen3 x16, dev1 Gen3 x<w>` without STOP; `r36.zc` prints `exact=1` on
every arm and a `law_summary`; `r36.replay` / `r36.alias` `result=exact` / `clean`; `r36.project` ends in
`STOP` or `GO-CANDIDATE`.

### Session 10 - the rest, and the summary

```sh
tools/box_validate.sh --only x
tools/box_validate.sh --summary          # box-validation-<date>.md: every stage of the state
```

`x.rest`: ctest over every registered test no stage has run (new host tests, the routed tests'
twins), ~0.5-1 h (est.). The summary covers the whole state, so the last one written is the
day's record.

### Session 11 - Bolt on SergiioB's g128 checkpoint (row 57), box CPU

```sh
tools/box.sh run 'tools/probe/detach.sh $HOME/dl-bolt.log uvx --from huggingface_hub hf download SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym --revision b9f381d20cbd4cede9f729c2647d2542c4a35997'
tools/box_validate.sh --only r57 --with r57.ppl,r57.hfcheck,r57.golden,r57.a4_ref,r57.taps
```

No card and no GPU lock: it can run beside a GPU session, one CPU oracle at a time and never
during a build. `r57.host` needs only the small files; the rest SKIP "missing data" until the
download is complete. `r57.golden` (hours) feeds 25b-25d's gates;
`r57.a4_ref` (days, resumable; est.) is the tool-call reference 25e's `r62.a4` scores the engine
against (`bolt_a4_int4`). `r57.golden_bf16`, `r57.a4_bf16` and `r57.gateq` SKIP here (no BF16):
they belong to Session 12.

### Session 12 - the Bolt BF16 window (rows 57, 58)

The only time the BF16 original is on the box. Start it when (1) plan 25q is merged (row 58's
script and stages exist), (2) Session 11's `r57.golden` and `r57.a4_ref` are done, so the window
holds only work that needs BF16, (3) `df -h ~` has >= 240 GB free (§1.5), and (4) both cards can be
given to `r58.tune` for up to two days. Everything runs in this order, **serially** (RAM, §1.5):

```sh
# 1. the download (149.8 GB; 25 min - 1.5 h at an unmeasured 30-100 MB/s)
tools/box.sh run 'tools/probe/detach.sh $HOME/dl-bolt-bf16.log uvx --from huggingface_hub hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 --revision 567021037ff88364143e24ea05e6abd0f72086ff'
# 2. RTN g64 from the BF16 original (the native-g64 development checkpoint), checked
tools/box_validate.sh --only r58.host,r58.rtn,r58.rtn_ref
# 3. the operator starts the tune: calib, coverage, tune on both cards (~16-40 h, the GPU lock), check
#    (plan 25q Task 4; no other GPU row meanwhile, no BF16 CPU stage either - RAM)
# 4. the BF16 references and gate Q, box CPU
tools/box_validate.sh --only r57.golden_bf16,r57.a4_bf16,r57.gateq
# 5. F6 (KL / top-1 per set and A4 for RTN, SergiioB's g128 and ours against the BF16 original), our export's golden sets
tools/box_validate.sh --only r58.eval,r58.ours_ref
```

Then (a Mac edit) gate Q's table into spec 25 (Decision 3: recorded), F6 into plan 25q's card, and
the operator's call on which g64 export replaces SergiioB's g128 as the served form. Only after
that does the operator remove the BF16 original from the box's HF cache (nothing after this
session reads it; to re-run any of these stages, download it again at the same revision).

**Estimate (all ESTIMATED):** download ~1 h, RTN hours, `tune` 16-40 h (plan 25q; 800 iterations
~32-40 h), `r57.golden_bf16` hours (~1.5x the g128 run, `bolt_oracle.sh`), `r57.a4_bf16` days (36
scenarios x 2048 new ids on the CPU), gate Q and F6 hours - **~3-6 days** with the BF16 original on
disk. **The short path** drops the BF16 A4 reference: gate Q and F6 then report KL / top-1 only
(no A4 column) and the window shrinks by those days; `r58.eval` and `r57.gateq` need
`bolt_a4_ref` (both forms' A4) as written, so taking the short path means running them without
that key, a change to make on the Mac before the window.

## 3. When time is short

### 3.1 Never skip

- **G0:** `g0.build`, `g0.sha`, `g0.bitwise`, `g0.suite`.
- **The K0 / R0 recaps** (each costs seconds; they turn G0 into each row's "nothing moved"):
  `r5.r0`, `r10.r0`, `r11.bf16`, `r13.r0`, `r14.k0`, `r15.k0`, `r16.r0`, `r17.k0`, `r19.r0`,
  `r21.k0`, `r22.r0`, `r23.r0`, `r24.k0`, `r25.k0`, `r26.k0`, `r27.k0`, `r28.k0`.
- **The bitwise P1 gates and their safety twins:** `r22.p1`, `r22.p1_kv8`, `r23.p1`,
  `r23.p1_kv8`, `r27.mtp`, `r27.mtp_kv8`, `r27.prefix`, `r27.prefix_kv8`, `r24.pp`, `r26.pp`,
  `r28.snapshot`; and P4, the "never hangs" proofs, `r22.p4`, `r23.p4`.
- **First runs of new kernels** (a blind kernel either works here or nowhere): `r8.kernels`,
  `r10.nockpt`, `r11.kernels`, `r13.kernels`, `r14.k1`, `r15.k1`, `r16.kernels`, `r21.k1`,
  `r24.k1`, `r26.k1`, `r32.k1`, `r33.k1`, `r34.k1`; and `r19.load`, `r22.devices`.
- **The token gates:** `r1.gates`, `r10.gates`, `r13.gates`, `r14.golden`, `r15.golden`,
  `r21.golden`, `r24.golden`, `r26.golden`, `r32.golden`, `r33.golden`.

### 3.2 Defer first (default stages; the short path, ~36 h instead of ~51 h)

| what | stages | saves (est.) | why it can wait |
|---|---|---|---|
| the whole timed block | session 6 | 7-11 h | records and cost tables; no gate depends on them. Run it in the next idle window |
| P3 at 262144 | `--skip r27.p3` in session 8 | ~3 h | passkey at ~250k; S3's memory lines (`r27.s3`) still prove the length fits |
| draft-vocab M3 at 128k | `--skip r8.m3` in session 2 | ~1.5 h | `--draft-vocab` is off by default; `r8.kernels` / `r8.off` / `r8.memory` still run |
| S2 at pp65536 | `--skip r23.s2_64k` in session 7 | ~0.5 h | pp32768 already tests the >= 1.7x bar |
| the passkeys inside default rows | `--skip r1.passkey,r25.passkey` | ~1 h | long-context recall; the golden gates cover correctness |

Skip with the session's own command plus `--skip <ids>`; the skipped stages stay "not run" in
the state and run later with `--only <ids>`.

### 3.3 Opt-in, in the order to spend spare box time

| tier | invocation | est. | what it buys |
|---|---|---|---|
| conditional | `--only r15.golden_eager` | ~0.5 h | only when `r15.golden` fails determined rows (spec 18 §11) |
| 1: decisions | `--only r10.p0,r13.p0,r15.p0,r13.speed,r14.speed,r15.speed,r21.speed,r16.mtp_rows,r8.auto_rows,r23.split_sweep` (idle box, both cards free for `r23.split_sweep`) | 7-10 h | the P0 profiles (incl. the int4 a\|\|b cell's share), K2's flash vs eager decode cost at 4k-32k (§10.1's ~3 % rule), K2's int8 vs bf16 KV speed, Ornith's MTP K, `--mtp auto` with a draft vocab, the time-balanced split |
| 2: long context | `--only r6.passkey120k,r7.passkey,r11.passkey120k,r11.passkey262k,r16.passkey,r21.passkey,r22.s1_32k,r23.s2_128k` | 6-10 h | recall at 120k-262k, S1 at depth 32k, S2 at pp131072 |
| 3: hours | `--only r6.d1,r7.depth,r8.rows_full,r8.m3_small,r10.speed,r27.p3_mtp` | 12-20 h | A12's depth rows, decode at the auto depth, the full §11 matrix, M3 at 32k / 64k, Ornith at 250k, P3 with MTP |
| 3: K2 A4 | `--only r25.a4_ref,r25.a4` | ~0.6-1 day (CPU; ESTIMATED, batched + resident weights - `BATCH=1 RESIDENT=none` is the old ~1-1.5 days) + ~1 h | K4's reference on the box CPU (blocks the idle grade while it runs), then the run; no bar |
| 3: comparison rows | `--only r1.prefix_benchy,r16.benchy,r25.benchy,r27.benchy` (needs `uvx`) | 3-5 h | llama-benchy rows for BENCHMARKS / README |
| 3: comparison matrix | Session 6c: `--only r41.ours,r41.vllm,r41.vllm_mtp` (needs `uvx`, the vLLM image; idle box) | 2-3.5 h | b70-serve vs vLLM through one pinned client and flag set (row 41) |
| blocked | `r2.opencode`, `r12.d5` (no opencode recording), `r11.a4` (`A4_REF_DIR`: set it if Qwen3.8's A4 bf16 reference dir is on the box), `r16.a4` (`oracle-out-ornith-a4`), `r24.oracle_real`, `r24.speed`, `r26.speed`, `r28.passkey`, `r28.a4`, `r28.benchy` (spec 20b) | - | data that does not exist yet |

Manual stages (commands in `--dry-run --with ...` and in the summary, never run by the
orchestrator): `r1.a4`, `r1.vllm`, `r4.g128`, `r10.p0_sweeps`, `r11.near_ties`, `r13.p0_arms`,
`r14.sweeps`, `r15.p0_arms`, `r16.a4_ref`, `r18.ct`, `r24.sweeps`, `r26.p0`.

Optional second G0: `r21.k0` lists `k2_moe.cl`'s pre-18e binaries as "not compared" because
`b32aaaf` predates K2. `--baseline 6ee1d59` (main just before 18e) compares them, but it is a
new state: another G0 (~3 h) before `--only r21.k0` can run in it. Defer.

## 4. What the results decide

Each is an edit on a branch after the run (none is made by the runbook), from the output of the
named stage.

| decision | read from | the rule |
|---|---|---|
| 16b decode hand-off default (`--pipeline-handoff copy\|peer`) | `r22.s1` (+ `r22.p1` under peer) | keep `copy` unless `peer` wins S1 **and** P1 holds bitwise under peer; S1's bar: `--pp 2` within 2 % of one card |
| `pp_recv`'s `spin_limit` | `r22.p4` | the peer throw's time / 2^24 is the first measured per-load time; if a clean step ever times out, raise the default |
| 16c prefill hand-off default, multi-group peer copy | `r23.s2_4k`, `r23.s2_32k`, `r23.s2_64k` (copy vs peer arms) | `copy` stays the default; if `peer`'s single work-group 20 MB copy costs more than a few % of a chunk, note it (a multi-group variant is the fix) |
| byte-balanced vs time-balanced split | the `pp: device N busy` lines of the S2 stages; opt-in `r23.split_sweep` | if device 1's ~3 % slower prefill unbalances the busy times, a time-balanced cut (30 / 32 / 34 swept) |
| S2 bars | `r23.s2_4k` (>= 1.2x), `r23.s2_32k` / `r23.s2_64k` (>= 1.7x) | estimates to confirm or re-derive |
| MTP cost tables (`MtpCost` defaults) | `r2.cost` (int8 head, verify M = 1..4, draft k = 1..3); at depth `r6.cost` (32k, 120k) | replace the defaults with the measured table |
| draft-vocab default and draft shares | `r8.rows` (medians); `r8.cost` -> `kInt8DraftHeadShare` / `kBf16DraftHeadShare` | the default leaves `off` only if a size beats it by >= 2 % on A4 and loses nowhere (operator's ruling) |
| Ornith `--mtp-cost` default and default K | `r16.cost` | an Ornith table beside Qwen3.8's |
| int8 KV: tolerances and default | `r11.kernels` (gated flash rows, PROVISIONAL >= 0.9999 - the card's first run sets them; `flash_long_kv8_test` at 2e-3 per depth / 5e-4 averaged, set in `6ee1d59`), `r11.q2` (`kv8_vs_oracle_test` <= 1e-4), `r11.speed` (1.1x / 1.15x / 1.2x at 32k / 64k / 128k, prefill <= 3 % worse) | int8 the default only after every row-11 gate passes (plan 12b Task 4 Step 3) |
| K2 near-tie tolerance `B70_K2_TIE_TOL` | `r14.oracle`'s selection-gap distribution (MoE 8th / 9th, MoVA 4th / 5th) | 1e-3 proposed; the driver forwards `B70_K2_TIE_TOL` to `r14.golden`, `r15.golden`, `r15.cli` and the `r21.golden`, `r21.golden_eager`, `r21.golden_prefill` twins; re-run those with `--redo` once set |
| K2 attention default (`kDefaultK2Attn`) | `r14.golden` vs `r14.golden_eager` (and `r21.golden` vs `r21.golden_eager` on int8 KV); eager's cost from opt-in `r15.speed`'s eager pairs | spec 18 §10.1: `eager` if flash fails determined rows that eager passes, or both pass and eager's routing diagnostic is strictly closer; otherwise `flash`; if eager wins and costs > ~3 % at 32k, a faster eager next |
| K2 prefill bars (PROPOSED: KV rows >= 0.999, median >= 0.9998, p01 >= 0.99; routing margin 2e-2) | `r15.prefill`'s printed distributions | set from the distribution |
| K2 int8 kernel bars (PROPOSED) | `r21.k1` runs at the proposals | the real numbers come from `tools/oracle/kv8_k2_repeat.sh` (4-8 h CPU, not a stage) |
| Kolibri near-tie tolerance `B70_KOL_TIE_TOL` | `r24.oracle_synth`'s gap distribution (MoE 6th / 7th); `r24.oracle_real` after 20b | 1e-2 proposed - override with `B70_KOL_TIE_TOL=<tol> tools/box_validate.sh ...` (forwarded to r24 / r26's golden stages since 2026-10-07) |
| Kolibri attention default | `r24.golden` vs `r24.golden_eager`, `r26.golden` vs `r26.golden_eager` | spec 18 §10.1's rule, as for K2 |
| Kolibri prefill bars (PROPOSED: row >= 0.999 on layer 0, median >= 0.9998, p01 >= 0.99) | `r26.prefill` | set from the distribution |
| Ornith prefill consistency (`kConsistTieRel`, `kConsistWeightAbs`) and the non-64 split bars | `r13.prefill`, `r13.split` | calibrate from the printed distribution; the split bars are Qwen3.8's (PROVISIONAL) until then |
| Ornith int4 a\|\|b cell (PROVISIONAL S1 L1) | opt-in `r10.p0` (its share of the 526 launches) | an S > 1 form needs gdn_step to sum slices (a P0 arm, not built) |
| Agnes GEMV cells P1-P3 | `r1.sweep` | replace `make_agnes()`'s {S, L} picks and `add_gemv_variant` cells, rebuild, **G0 again** |
| Agnes W (P4) and speeds (P7) | `r1.load` (W within 2 %), `r1.speed` | replace the PENDING numbers |
| `--mem-reserve-gb` 1.5 | `r7.serve` (device memory via xpu-smi while a fresh server is up) | confirm, or tune after a long operator session |
| `--prefix-cache-gb auto` | `r9.auto` (startup line and its /proc/meminfo inputs) | confirm the chosen size |
| MTP embedding on device 1 | `r27.serve`, `r27.s3` memory lines | already **decided: replicate** (operator, 2026-10-07); the box only confirms it fits |

## 5. Known risks

- **The first ocloc compile.** ~250 kernel binaries written blind (rows 8, 10, 11, 13-16, 19,
  21, 22, 24, 26) meet ocloc in `g0.build`; one failure stops the build and G0. Triage with a
  family switched off (§2, session 1), fix on a branch, new commit, G0 again.
- **`pp_send` / `pp_recv`: one work-group moving 20 MB** per prefill chunk on the peer path -
  unmeasured (spec 16 §9). It shows in the peer arms of `r23.s2_4k`, `r23.s2_32k` and `r23.s2_64k`; `copy` stays the default.
  The same kernels' bounded spin (`spin_limit`, 2^24 loads under a 120 s fence bound) has never
  run: `r22.p4` is its first timing.
- **Nothing multi-device has ever run** (rows 22-24, 26-28): one Level Zero context over both
  cards, cross-device events, peer access, load-then-place (device 0 briefly holds the whole
  model). A P1 difference is a bug - read the first differing byte the test prints. Needs both
  cards free and the P2P patch still in the kernel.
- **The DFlash P0 finished on the Mac** (2026-10-08 02:54, `docs/probe-dflash-2026-10-08.md`;
  provisional go on code and agentic output). Its formal verdict waits for Session 6b
  (`r29.cost`; the probe build is on branch `spec19a-task4-probe`, blind: its M = 5..8 binaries
  first compile in G0's build - an ocloc failure there names the variant). 19a also wants data **from** the box: spec 8
  P0's 256-id golden continuations and the A4 bf16 ids (`GOLDEN_CONT_DIR` / `EXTRA_SOURCES`) -
  pull them on the box day (no GPU) to firm up the prose number (52 anchors today).
- **Ornith's A4 reference is being made on the Mac** (started 2026-10-08, container
  `a4-ref-ornith`, `oracle-out-ornith-a4/ref.log`; MEASURED ~30 min a scenario sequential, ~18 h
  the set; resumed batched + resident (`a4_ref.sh`'s defaults since branch `a4-ref-batched`,
  bitwise the sequential ids) ~1.25x faster - ESTIMATED: Ornith's step is mostly per-sequence
  compute, tools/toolcall/a4_ref.sh's header): push `oracle-out-ornith-a4` when its `status`
  shows 36/36, or `r16.a4` records SKIP.
- **PROPOSED / PROVISIONAL bars everywhere** (kv8 gated flash rows, K2 prefill and int8 bars,
  Kolibri tie and partial-forward bars, Ornith consistency and split bars): a FAIL against one
  is first a reading - look at the printed distribution before calling it a bug.
- **`r25.toolcall` is default but its data is not made** (`oracle-out-k2-a4/set`): SKIP unless
  the set is made first (§1.4).
- **A stale rownote:** row 11's note still says the int8-KV constants are provisional until the
  Qwen3.8 repeat; they were replaced in `6ee1d59` (docs/probe-int8-kv-qwen38-2026-10-06.md).
  Only the gated flash rows of `kv8_kernels_test` remain PROVISIONAL.
- **The idle grade is fragile:** a GNOME daemon or another agent's container appearing
  mid-session makes the timed rows ITERATE; the `IDLE after` line catches it, a re-run in a
  clean window fixes it (`--redo <id>`).
- **Memory: CPU oracles need RAM.** 121 GB on the box; `r14.oracle` refuses below 32 GB
  available, `r24.oracle_synth` below 24 GB. Run one at a time, never under `g0.build`.
