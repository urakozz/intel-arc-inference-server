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
| 4 | Ornith decode, prefill, serving (rows 10, 13, 16) | device 0 | `tools/box_validate.sh --only r10,r13,r16 --skip r16.cost` | 21 | 3-5 h |
| 5 | K2-Horizon (rows 14, 15, 21, 25) | device 0 | `tools/box_validate.sh --only r14,r15,r21,r25 --with r14.oracle` | 35 | 4.5-7.5 h |
| 6 | timed one-card rows, **box idle (RECORD)** | device 0, both idle | `tools/box_validate.sh --only r1.sweep,r1.speed,r2.cost,r2.auto_rows,r3,r6.cost,r8.cost,r8.rows,r11.speed,r16.cost` | 10 | 7-11 h |
| 6b | DFlash verify cost at M = 5..8 and one block's draft cost (row 29, plan 19a Task 4) | device 0, both idle | `tools/box_validate.sh --only r29.cost` | 1 (opt-in) | 0.25-0.5 h (est.) |
| 7 | pipeline decode, then prefill (rows 22, 23) | **both** | `tools/box_validate.sh --only r22,r23` | 22 | 4.5-7 h |
| 8 | pipeline integration (row 27) | **both** | `tools/box_validate.sh --only r27` | 14 | 4.5-7 h |
| 9 | Kolibri-1 on the synthetic checkpoints (rows 24, 26, 28) | **both** | `tools/box_validate.sh --only r24,r26,r28 --with r24.oracle_synth` | 31 | 4.5-7 h |
| 10 | the rest of the suite | device 0 | `tools/box_validate.sh --only x` | 1 | 0.5-1 h |

**Total, the default set: ~40-63 h of box time, ~51 h central (est.)** - two to two and a half
days if the sessions run back to back. Sessions 1-10 together run each of the 179 default
stages exactly once (checked against the default `--dry-run`: no stage missing, none twice);
sessions 5 and 9 add the two opt-in oracle stages the K2 and Kolibri gates need. The short
path (§3) is ~36 h. The opt-in stages (§3.3) are extra: tier 1 alone 7-10 h, every tier
30-45 h or more.

Row 20 (Kolibri-1 spec 20a, the quantisation) has no runbook stage and needs spec 20
decision 1; it is not part of the box day. Everything on the real Kolibri int4 checkpoint
(spec 20b, which does not exist yet) is recorded SKIP "missing data" in session 9 -
expected, not a failure.

## How the runbook constrains the plan

- **One orchestrator per tree, one GPU lock per box.** `remote.sh` runs the selected stages
  strictly in registry order (row order), and every `gpu` stage holds `~/b70-gpu.lock` - one
  lock for both cards. So no two GPU stages ever overlap, whichever card they use: the
  single-card sessions leave device 1 idle, and a second orchestrator on `DEVICE=1` would only
  queue on the same lock. The only real parallelism is CPU work (downloads, the CPU oracles)
  beside non-timed GPU stages (§1.6).
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

```sh
export REMOTE_DIR=b70-inference-server-validate
tools/box.sh run 'tools/probe/detach.sh $HOME/dl-models.log bash -c "for r in urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ; do uvx --from huggingface_hub hf download \$r || exit 1; done"'
tools/box.sh run 'tail -3 $HOME/dl-models.log'      # poll; the log ends in ALLDONE rc=0
```

~67.7 GB in all; at an unmeasured 30-100 MB/s that is 10-40 min (est.), well inside session
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
      list.
- [ ] **RAM** (121 GB on the box): one CPU oracle at a time; `r14.oracle` needs
      MemAvailable >= 32 GB (`K2_REF_MIN_GB`), `r24.oracle_synth` >= 24 GB (`KOL_REF_MIN_GB`);
      never during `g0.build` (a compile slows a CPU oracle by ~28 %, docs/10, and both want RAM).

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
      5 / 9 then only re-link the sets and record the gap distributions. **Never during
      sessions 6-9**: a running container makes every timed row ITERATE.

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

### Session 10 - the rest, and the summary

```sh
tools/box_validate.sh --only x
tools/box_validate.sh --summary          # box-validation-<date>.md: every stage of the state
```

`x.rest`: ctest over every registered test no stage has run (new host tests, the routed tests'
twins), ~0.5-1 h (est.). The summary covers the whole state, so the last one written is the
day's record.

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
  `r24.k1`, `r26.k1`; and `r19.load`, `r22.devices`.
- **The token gates:** `r1.gates`, `r10.gates`, `r13.gates`, `r14.golden`, `r15.golden`,
  `r21.golden`, `r24.golden`, `r26.golden`.

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
