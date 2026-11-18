# The box

The engine is developed on a laptop and built, tested and benchmarked on a separate Linux machine with the cards in
it. These docs call that machine "the box", and commands name it `$BOX`. This document describes what the box has
to be, how a remote build and the validation runs work, how GPU access is shared between jobs, and the traps that
have cost time. It is for anyone setting up a similar machine or running jobs on this one. Nothing in the engine
requires the laptop-and-box arrangement, but the tools assume it.

The laptop never compiles C++: every build and test runs on the box, `tools/box_check.sh` for a branch and
`tools/box_validate.sh` for the full list.

## What the box has to be

The box is a Linux workstation, a Dell T5810-class Broadwell-EP host, with two cards on PCIe 3.0 x16.

| | |
|---|---|
| OS | Linux. Ubuntu is what this was developed against; no Windows or macOS runtime exists |
| CPU / RAM | a many-core CPU and, on this box, 121 GB of RAM: the CPU oracle wants about 70 GB free for a layer-streamed 27B forward. The engine's own build is comfortable at full width; the reference vLLM container's torch build is not ([09-vllm-patch-postmortem.md](09-vllm-patch-postmortem.md)) |
| GPU | 2 x Intel Arc Pro B70 (Battlemage / Xe2), 32 GB each. One card is enough for everything except the two-card modes (tensor parallel, replica, pipeline parallel; [BENCHMARKS.md](BENCHMARKS.md)); the second is also useful as an idle control |
| Driver | `libze_intel_gpu`, IGC and `intel-ocloc`. The B70 is `bmg-g31`, so kernels build with `ocloc -device bmg-g31` |
| Level Zero | headers and loader from the distribution packages |
| Host compilers | GCC 16 (`/usr/local/bin/g++-16.2`, the default the build picks; C++26 with static reflection, `-freflection`) and CMake 4.2 or later, plus `ccache`. GCC 16's `libstdc++` is newer than the system's: the binaries find it through their rpath (`/usr/local/lib64`), so no `LD_LIBRARY_PATH` is needed. The system `g++` 15.2 stays for other projects |
| Rust | needed for the pinned `tokenizers` crate ([11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md)). A user-local `rustup` with `--profile minimal` is enough; the build looks in `~/.cargo/bin` first and never changes `PATH`. The crate builds `oniguruma` from source through `cc`, so a working `/usr/bin/cc` is required |
| oneAPI | not used by the engine build. Only the stand-alone oneDNN reference probe (`tools/probe/onednn/CMakeLists.txt`, oneDNN found through `B70_ONEDNN_ROOT`) needs `icpx`, which is usually not on `PATH` in a non-interactive shell, so pass its full path |

The C++ side builds natively on the host. No container is needed for anything except the Python CPU oracle and the
reference vLLM stack.

### Versions on the box

| | version |
|---|---|
| kernel (running) | Linux 7.4.0-rc2 (mainline), which supports peer-to-peer access natively (earlier kernels needed a local patch) |
| IGC (Intel Graphics Compiler, which turns OpenCL C into GPU code) | 2.42.6, from the GitHub release debs |
| compute runtime (`libze-intel-gpu1`, `intel-opencl-icd`, `intel-ocloc`) | 26.40.40196.8, from the GitHub release debs (held, so an apt upgrade does not move it) |
| Level Zero loader | 1.34 |

This stack was installed on 2026-11-04 and changed both the compiler and the runtime, so speed rows in BENCHMARKS
taken before that date come from the older stack (compute runtime 26.35, IGC 2.41.5) until they are re-measured on a
quiet box ([open-work.md](open-work.md)).

Kernel binaries cannot go stale silently. `cmake/ocloc_stamp.cmake` records `ocloc --version` and the real paths of
the `libigc` / `libigdfcl` libraries the loader resolves, and rewrites a stamp file only when they change. Every
kernel target depends on that stamp, so a new compiler recompiles every kernel. The engine build uses no oneAPI
compiler at all, so the oneAPI and oneDNN versions on the box matter only to the stand-alone oneDNN probe.

### Peer access between the cards

Peer-to-peer access (P2P) is one card reading or writing the other card's memory directly over PCIe. Tensor parallel
exchanges partial results this way every layer, and the hardware check refuses two visible cards that cannot both reach
each other's memory, with the advice to run on one card with `ZE_AFFINITY_MASK=0`. Peer access needs a kernel that supports it (the mainline 7.4.0-rc2 above does) and both cards
under one PCIe root complex.

The BAR is the window through which the host or the other card reaches a card's memory. The firmware's resizable-BAR
setting (ReBarState 14) caps it at 16 GB (15 does not boot); the `xe` driver resizes both cards' BARs to the full
32 GB when it loads.

The driver can answer EBUSY ("busy") to a request, and the runtime retries that forever. With no retry limit, an
EBUSY that never clears is a hang, not an error. It happened with one large pinned host allocation that the second
card maps through a shared import (the prefix cache). The rule since then is that any host allocation both cards use
stays at or below 4 GiB, which is why the prefix cache is made of 2 GiB pieces.

## Where things live on the box

New things on the box go under `~/b70`, and nothing new sits at the top level of the home directory:

| path | what |
|---|---|
| `~/b70/trees/<name>` | a branch's tree (every branch but main; `tools/box_dir.sh` picks the name) |
| `~/b70/logs/<topic>/` | logs: box_check, probes (`tools/probe/detach.sh` makes the directory), benchmarks, CPU jobs |
| `~/b70/scripts/` | the owner's scripts and the job chains |
| `~/b70/scratch/`, `~/b70/models/` | probe captures; local model folders that are not in the HF cache |
| `~/b70/hdd`, `~/b70/hdd3` | the cold and archive disks ("Model storage" below) |
| `~/b70/trash/<date>/` | what is no longer needed: `mv` it here, and the box's owner runs `rm` |

A few older paths stay at the top level, because every validation run uses them:

- the base tree `~/b70-inference-server`, which is main's tree and holds the golden sets `oracle-out*` that every
  other tree links to;
- the validation tree `~/b70-inference-server-validate`;
- its baseline builds `~/b70-inference-server-g0-<sha12>`, which the identical-binaries check compares against;
- the validation results, `~/b70-validate/<tree>/<state>/`.

The GPU lock files `~/b70-gpu*.lock` and `~/b70-gpu.wait` stay in `$HOME` for good: every script opens them there,
and a lock moved under a running holder no longer excludes anything.

## CPU jobs and host memory

CPU jobs are memory-capped, with no swap. Every CPU reference container on the box
(`tools/oracle/run_in_container.sh`) runs with `--memory X --memory-swap X`, where X is `ORACLE_MEMORY`, 64g by
default. When a job reaches its cap, the kernel kills it inside its container and nothing else notices. The other
roughly 65 GiB of the 121 GiB are for one GPU job's host side (measured at about 52 GiB: the pinned prefix cache, up
to 32 GiB, plus weight staging), the OS, and the page cache of the GPU job's weight reads. A job that needs more says
so (`ORACLE_MEMORY=96g`, at most MemTotal - 9 GiB) and runs only while the cards are idle: it holds the lock on both
cards for its run and is refused (exit 75) while a GPU job runs.

The reason is an incident: an uncapped reference job (whose `MEM=80g` sized only its own plan) plus a GPU job's
roughly 63 GB of driver-held host memory once filled RAM and swap. The kernel killed the reference, and the next
job's swapping beside a GPU job's weight reads took the NVMe off the bus.

Only one CPU model job runs at a time. Every CPU reference script (the f32 judges
`tools/box_validate/f32_oracle.sh` and `f32_refs.sh`, the Kolibri oracles) holds `/tmp/b70-cpu-model.lock` for its
whole run, and first checks MemAvailable (the kernel's estimate of RAM free for a new job) against its own floor
(`F32_REF_MIN_GB`, `KOL_REF_MIN_GB`, ...). Below the floor the script exits 77, which a stage records as SKIP, not
FAIL. These jobs run beside untimed GPU work with `ORACLE_THREADS=22` (half the cores, so the GPU tests keep their
host threads), but never during a build (a compile slows a CPU reference by about 28%, see "Why an idle box matters")
and never during a timed row. A finished reference lands in the data tree (`~/b70-inference-server/oracle-out-*`);
`tools/box_validate/data.sh link ~/b70-inference-server <tree>` links it into another tree.

## Driving a remote build

`tools/box.sh` syncs the tree, configures, builds and runs `ctest` over ssh. Set `BOX=user@host` before using it, or
put it in the untracked `tools/box.env`; there is no default address in the repo. `REMOTE_DIR`, `JOBS`, `BUILD_DIR`
and `CMAKE_ARGS` override the rest. Each branch or worktree gets its own remote tree (`tools/box_dir.sh`), so
parallel lines of work do not overwrite one another.

```sh
tools/box.sh sync                 # rsync the tree
tools/box.sh build                # sync + cmake configure + build
tools/box.sh test [regex]         # sync + build + ctest, optionally -R regex
tools/box.sh run <cmd...>         # run a command in the remote tree
tools/box.sh pull <path>          # copy a generated file back
```

Two properties of that script matter more than they look:

- It syncs without `.git`, so the remote tree cannot name the commit it is building. Anything that wants to record a
  revision has to have it substituted by the local shell before the command crosses the ssh boundary.
- The build directory is a `BUILD_DIR` rather than a hardcoded path, so a second build directory can live beside the
  first and one tree serves every configuration.

Long jobs must survive the ssh session. Detach them with
`tools/box.sh run 'tools/probe/detach.sh ~/b70/logs/<topic>/<name>.log <cmd...>'`, which wraps `setsid nohup`; the
log ends in `ALLDONE rc=<status>`. Poll the log at most every 30 minutes. This is not fastidiousness: BuildKit
discards all completed work, including finished layers, if the docker client dies, and a dropped connection can
silently lose hours.

The branch check is `tools/box_check.sh`. It syncs this checkout, does the full build, compares every kernel's `ocloc`
command line against a base (`tools/kernel_cmdlines`: an existing variant's command line must not change or vanish),
and runs `ctest -LE checkpoint` (every test that needs no model files) on a free card. Passed, Failed and Skipped are
counted separately, because a skipped test is not a pass. Logs go under `~/b70/logs`. It does not gate merges ("When
each run happens" below).

GPU access is serialised. Every job that touches a card goes through `tools/gpu_lock.sh`: `box` takes both cards
(timed rows, two-card work, anything that wants an idle grade), and `card N` takes one card and runs the command with
`ZE_AFFINITY_MASK=N`, so a second job can use the other card. A both-cards job waiting at the gate stops new card
jobs from starting, so timed work cannot starve. The details, and how two validation streams share the cards, are in
"Two streams, one per card" below.

Our own jobs are stopped with `tools/box_stop.sh`. It stops only the kinds of job listed here (TERM, then KILL after
10 s) and refuses anything else:

| `tools/box_stop.sh ...` | what it stops |
|---|---|
| `list` | nothing: shows what is running |
| `cpu-chain` | the loop that runs the CPU reference queue; its current job keeps running |
| `cpu-job <queue name>` | one CPU reference job and its container |
| `validation` | the newest validation run; starting it again with the same state resumes |
| `tree <branch>` | every job working in that branch's tree |
| `job <branch> <pid>`, `main-job <pid>` | one process tree, checked to be working in that branch or in the main tree |
| `hf-tier`, `dflash-quant`, `q4exp-count-wait` | the model-storage copy, the drafter quantisation, the Flash-Next count driver |

To see what the box is doing, run `tools/box_status.sh` (read-only; `--watch N` refreshes every N seconds). It shows
the GPU locks and who holds them, the jobs waiting for a card, the CPU reference job and the logs written in the last
half hour. `tools/box_tidy.sh --dry-run` lists stale branch trees that it would move to `~/b70/trash`.

## Validation runs

Validating work in bulk is one command, `tools/box_validate.sh`; run it with `--dry-run` first to read what it will
do. It runs the rows and stages of `tools/box_validate/stages.sh` (`tools/box_validate.sh --list` prints them) in one
detached orchestrator on the box, under the GPU lock. The first stage is the identical-binaries check, G0: kernel
binaries and gate outputs must be identical to a baseline build, and the run stops if they are not. The run can be
resumed after an ssh drop or a reboot, and when it is done it writes `box-validation-<date>.md`. Its results are on
the box under `~/b70-validate/`, and the latest result of every stage is in `~/b70/logs/stage-latest.txt`.

### When each run happens

Merges do not wait for a box run. The full validation run is not per merge and not on a timer: it runs on demand, on
the latest main, and covers every merge since the last run together. The command for it is
`tools/box_validate.sh --nightly`. It launches nothing when only documentation has changed since the newest finished
run (`tools/box_validate/changed_since.sh`: everything but `docs/` and `*.md` counts), and it runs every test
(`--no-reuse`) when the newest finished full run is older than 7 days, as a weekly safety net. Runs in between reuse
every unchanged PASS.

The merge check, `tools/box_validate.sh --merge-check`, is meant to take about 20-30 minutes; a fixed DFlash model
sample, `merge.dflash`, runs outside the changed tests' time cap. It runs:

- the build of the tree under test (the baseline's build is kept between runs and reused while its commit, build
  settings, compiler, ocloc and kernel series are the same: `tools/box_validate/baseline.py`);
- the kernel-binary comparison;
- the host stages that need no model;
- only the tests whose inputs changed since their last recorded PASS, shortest first, up to 480 s of their recorded
  times. A changed test over that budget is not run; the summary says so and lists it for the quick run.

It keeps its own state (`<sha>-merge-vs-<baseline>`) and does not stand for a quick or a full run.

The quick run, `tools/box_validate.sh --quick`, is for every change. It was meant to take an hour; on a box shared
with other jobs it once took about 3 hours, most of it G0's gate outputs and the suite under the whole-box lock. Since
then the baseline build is reused, and G0's gates, the suite and the rest of the suite (`x.rest`) hold one card, so
the other card stays free. Only the tests labelled `pp` or `tp` (`g0.suite_2card`, `x.rest_2card`) and the stages
that really use two cards hold the whole box. How much that saves is not measured yet.

The quick run is G0 as in the full run, except that the suite runs only the tests that load no model (no
`checkpoint` label), followed by a fixed sample of each model's kernel, decode, prefill and serving stages, a few with
a smaller input (the 32k MTP test instead of 128k; one server round over 2 golden and 3 tool-call prompts). Every
other stage is recorded SKIP "full only": the speed rows, the CPU reference builds, the long-context and many-prompt
runs. The gates and bars are the full run's; only the input is smaller. `--list` shows each stage's quick form (the
QUICK column), `--dry-run --quick` shows what it runs, and the block at the end of `tools/box_validate/stages.sh` is
the list. A quick run keeps its own state (`<sha>-quick-vs-<baseline>`), so its PASSes never stand for a full run's
(though a test's own reuse is shared: the same test with the same inputs). `--status` and `--summary` with `--quick`
read it, and its summary is headed "(quick)". It does not tidy the box at launch. Run it from a branch's own tree
(`REMOTE_DIR=b70/trees/<branch>`) when the validation tree is busy.

### A test whose inputs did not change keeps its PASS

Every test runs through `tools/test_launcher.sh`. Under box_check and box_validate it runs under `strace`, which
records every file the test, and every program it starts, opened, ran, checked or found missing, plus the environment
it saw. `tools/box_validate/reuse.py` stores each PASS with the state of those inputs
(`~/b70/test-results/<test>.json`): files of the tree by content, model snapshots and reference outputs by real path,
size and mtime, missing files as missing, and the number of visible cards. Before ctest runs, a test whose every input
is still identical takes that PASS (`<stage>.reused.junit.xml`, its old output behind a `REUSED:` line); everything
else runs.

A device binary must match byte for byte, because a new driver compiler changes binaries with no source change. A host
program from another tree, built from the same sources with the same settings, counts as the same, so the validation
run takes box_check's PASSes for the half of the suite that needs no model files. Only PASS is kept: a failure or a
skip always runs again. `tools/box_validate.sh --no-reuse` runs everything.

### Running stages by hand

The orchestrator is the one detached `tools/box_validate/remote.sh` process on the box that runs the stages in order.
These are the rules it enforces, and what follows from them.

A state is a commit paired with a baseline. Results live in `~/b70-validate/<tree>/<HEAD sha12>-vs-<baseline sha12>/`
(default baseline `3769e2a5`; `--baseline main` when validating a branch before its merge). Every launch that should
add to one state must come from the same commit with the same baseline; otherwise G0 is not PASS there and every
later stage is SKIP "blocked". So when main may move under a long run, launch from a detached worktree
(`git worktree add --detach ../b70-boxday main`, with the untracked `tools/box.env` copied in, never committed). A
`-dirty` suffix in the state name means the tree had uncommitted changes. A fix is a new commit and so a new state,
and G0 runs again; `--state <old name>` keeps adding to the old state, and is only for fixes that touch no device code
and nothing G0 compares.

Selecting stages:

- `--list` and `--dry-run` print every stage with the lock it will take; read them first.
- `--only <ids or row prefixes>` runs the preflight `pre` plus those stages and never adds G0, which must already be
  PASS in the state. A row prefix (`r22`) means the row's default stages; opt-in stages need their id or `--with <id>`.
- `--skip <ids>` leaves stages out (they stay "not run" for a later `--only`), and `--redo <ids>` runs a PASS again.
- Manual stages are never run by the orchestrator; their commands are in `--dry-run --with` and in the summary.
- A stage whose prerequisite is not PASS in this state is SKIP "needs PASS first", so the launch order is also the
  dependency order.

The preflight, `--only pre`, syncs the tree, extracts the baseline, and writes `pre.log` (uptime, cores, `free -g`,
`df -h ~`, `/dev/dri`, `ocloc`, the idle grade) and `have.env`, the `HAVE_<key>=0|1` list of data present
(`tools/box_validate/data.sh have`) against which every stage's needs are checked. A missing checkpoint or golden set
is SKIP "missing data", which is expected rather than a failure. `have.env` is taken once per launch, so data that
arrives later (a download, a remounted disk) needs a fresh launch. `--push-data` pushes the laptop's golden sets first,
from the main checkout found through the git common directory, so a worktree needs no copies.

Re-running the same command re-attaches to a live run; after the run has finished, it launches again and re-runs
every stage that is not PASS. `--no-wait` launches and returns, `--status` looks without changing anything, and
`--summary` fetches the results and writes `box-validation-<date>.md` without running anything. A dropped ssh
connection costs nothing, because the orchestrator is detached on the box.

A timed stage prints `IDLE before` / `IDLE after` lines. Quote that grade, and re-run an ITERATE row in a clean window
with `--redo <id>`. A FAIL against a bar marked PROPOSED or PROVISIONAL is a reading first: look at the printed
distribution before calling it a bug.

## Two streams, one per card

Two orchestrators can run at once, one per card, so two lines of correctness work can share the box. The locks are
`flock(2)` locks (a lock on a file, released when the holder exits) on files in `$HOME`; the header of
`tools/gpu_lock.sh` has the protocol in full.

| job | takes, in this order | runs beside |
|---|---|---|
| card N: a `card` stage of the device-N orchestrator, or `tools/gpu_lock.sh card N -- <cmd>` | `~/b70-gpu<N>.lock` exclusive, the gate `~/b70-gpu.wait` for an instant, `~/b70-gpu.lock` shared; `ZE_AFFINITY_MASK=N` | a card job on the other card |
| both cards: a `gpu` stage, or `tools/gpu_lock.sh box -- <cmd>` | the gate exclusive while it waits, then `~/b70-gpu.lock` exclusive (the gate dropped) | nothing |
| both cards, the old way: `flock ~/b70-gpu.lock <cmd>`, `gpu-self` stages (their scripts lock themselves) | `~/b70-gpu.lock` exclusive | nothing |

A card job starts when its card is free and no both-cards job runs or waits; a both-cards job starts when no job holds
either card. The gate exists because Linux flock has no queue: a shared lock is granted whenever no exclusive one is
held, so two overlapping card streams would otherwise starve a waiting timed job forever. The old
`flock ~/b70-gpu.lock` form still excludes everything but skips the gate, so while card streams run, take the box with
`tools/gpu_lock.sh box -- <cmd>`, and run `gpu-self` stages only when no card stream runs. Card, then gate, then box
is the one order, so there is no deadlock. Never nest: a job holding one of these locks must not take another (flock on
a second open of the same file waits for itself). `tools/gpu_lock.sh status` shows what is held right now.

Two orchestrators can share one tree and one state. Each device has its own run lock
(`~/b70-validate/<tree>/.run-<N>.lock`). A running stage is claimed (`<id>.claim`) and the other orchestrator leaves
it alone; a RUNNING stage whose claim is free belonged to a dead orchestrator and runs again. G0 runs alone, a stage
that builds holds `.build.lock`, and the `cpu` stages of the two orchestrators run one at a time (`.cpu.lock`, because
of the CPU references' RAM and the host tests' fixed `/tmp` directories). Servers listen on 8013 (device 0) and 8113
(device 1), and `progress.log` marks lines `[d0]` / `[d1]`.

To launch the second stream, from the same tree and commit after the first one started, run
`tools/box_validate.sh --device 1 --only <stages> --no-wait`, then `--status` to see both. Beside a running
orchestrator the driver refuses another state or commit, and refuses `--push-data` (push before the first launch).

Device 1 runs correctness only. The tooling keeps timed and series rows on device 0 ("Device selection" below), and
the device-1 dry-run names every both-cards stage in its selection. Two-card stages (`*.pp`, the pipeline-parallel
rows) take both cards when they come.

Split the work between the streams by row family, not inside a chain. A stage whose prerequisite is RUNNING in the
other stream at that moment is SKIP "needs PASS first" and has to be relaunched later. Pair families that are mostly
`card` stages; `tools/box_validate.sh --list --only <rows>` shows each stage's lock.

Mind the host RAM. Two engines load at once, and each `b70-serve` pins up to 32 GiB of prefix cache, sized from
MemAvailable at its start. Stages with a host-RAM budget (Qwen3.8-Flash-Next's 51.8 GB PLE table) and the passkey
probes (fixed `/tmp/*passkey` directories) therefore take both cards.

## Device selection, and the gotcha

A box with two cards will not hand you the same one twice unless you say which. Where one card also drives a display,
the render nodes do not map to selector indices the way you would guess, and the display card is not the one you want
under a benchmark.

The contract is in [04-architecture.md](04-architecture.md), and it is worth repeating here because getting it wrong
silently invalidates a measurement: the CLIs have no device flag; `ZE_AFFINITY_MASK` decides which cards the driver
enumerates; one card is device 0 of that view, and two are devices 0 and 1 (`ZE_AFFINITY_MASK=1` runs on the second
card alone).

The two cards may not be interchangeable. In one session on 2026-09-05, with the same binary and checkpoint, prefill
lost 3.2% to 3.4% on the second card while decode lost only 0.2% to 0.4%: a difference in sustained compute rather
than bandwidth. The likeliest cause found since is a hardware monitoring tool left open during that measurement, but
it has not been re-measured ([07-open-questions.md](07-open-questions.md)). So the validation tooling keeps series rows
on device 0, and every series row in [BENCHMARKS.md](BENCHMARKS.md) names its card.

## Why an idle box matters

A benchmark row counts as a record only when the box is provably idle: zero containers and zero processes holding a
DRM file descriptor on any card, checked before and after the run set. The harness measures that condition itself and
prints the grade the run earned. A grade is quoted as printed and never upgraded by hand. The grade ladder is in
[BENCHMARKS.md](BENCHMARKS.md).

Idleness matters unevenly, and both halves are measured:

- A CPU compile does not contend with decode. Under a 12-core compile the decode step read 36.27 ms per token against
  an idle median of 36.32, 0.14% apart, because the step is 99.7% inside the GPU fence. That is why a loaded box
  downgrades a row's grade without invalidating it.
- A CPU compile does contend with the CPU oracle. Regenerating one golden set took 25 min 14 s under a 12-core compile
  against 18 min 16 s on an idle box, 28% slower. Anything that runs on the CPU pays full price.

Desktop session daemons often hold a DRM file descriptor without submitting any work. They still cost a row its record
grade, because "submitted no work" is supporting evidence, not proof.

### The idle protocol, before timed or two-card work

1. Read the grade right before the launch, because a reading expires within minutes:
   `tools/box.sh run tools/box_validate/idle.sh now` prints `IDLE now grade=RECORD|ITERATE`, the container count, the
   load average, and every DRM holder (a process with an open file on a card's `/dev/dri` node) by pid and name.
2. Holders are for the box's owner to end. Never kill anything on the box that is not ours (ours stop through
   `tools/box_stop.sh`); report the pid, the name and the command. Known holders: the GNOME `baobab` / `ptyxis` service
   daemons, which ignore SIGTERM and need `kill -9`; nautilus, chrome, gnome-control-center and resources also open
   both cards; killing Xwayland takes the desktop down, so ask first.
3. Zero containers: no CPU reference job and no vLLM. Stop the CPU jobs for the timed window.
4. No second stream: `tools/box_validate.sh --status` lists both orchestrators, and `tools/gpu_lock.sh status` on the
   box should show the box lock, both cards and the gate free.
5. Two-card rows need both cards free of other holders. Their commands set `ZE_AFFINITY_MASK=0,1` themselves; series
   rows stay on device 0.

The grade is fragile: a desktop daemon or another job's container appearing mid-run turns a row ITERATE. The stage's
own `IDLE after` line catches it, and `--redo <id>` in a clean window fixes it.

## The reference stack

The vLLM container that produces the baseline numbers, the exact serve command and the exact bench command are all in
[BENCHMARKS.md](BENCHMARKS.md). Two things about running it are worth knowing first.

- Read the vLLM version string the server prints, never the image tag. A tag gets rebuilt: one tag in this project's
  history carried two builds 149 commits apart under the same name, and a number attributed to the wrong one cannot be
  recovered later.
- `--compilation-config '{"inductor_compile_config":{"pre_grad_fusion_options":{}}}'` is required, not optional.
  torch 2.14 defaults that option to an XPU-only `batch_linear_lhs` fusion that costs about 30% of decode on this
  workload.

A golden set belongs to its container as well as to its checkpoint. The oracle is `transformers` running inside
whichever image made the dump, so two sets dumped under different images are not interchangeable evidence, even for
the same weights. Record the image beside each set.

## Gotchas that cost time

- `docker stop` does not immediately free the container name. A following `docker run --name X` can fail with exit
  125; run `docker rm -f X` first.
- Never run `docker volume prune` on a box that also runs anything else. It takes unrelated named volumes with it.
- A non-interactive ssh shell has a short `PATH`. `uvx` may be only at `~/.local/bin/uvx`, and `icpx` is not on it at
  all, so give full paths in `tools/box.sh run` commands.
- Loading a 27B model takes 2 to 4 minutes in the reference stack, and graph capture plus compilation adds another 2
  to 4, so budget about 8 minutes from launch to a servable endpoint. Our own loader takes 13.6 s on a warm page cache.
- A consumer NVMe drive can drop out under APST (autonomous power state transitions).
  `nvme_core.default_ps_max_latency_us=0` on the kernel command line prevents it. If the box loses its disk after a
  kernel update, check that first.

## Model storage: hot, cold, archive

Three disks hold the Hugging Face cache, in tiers:

| tier | where | holds |
|---|---|---|
| hot | NVMe, `~/.cache/huggingface/hub` | the working set: the models the next runs read, as real folders |
| cold | 8 TB WD Red, `~/b70/hdd/hf-hub` | every model, always (the bf16 originals, our int4 exports, everything hot) |
| archive | 3 TB, `~/b70/hdd3/hf-hub` | parked models, instead of cold |

Every `~/.cache/huggingface/hub/models--X` is either a real folder (hot) or a symlink to `models--X` on the 8 TB or
3 TB disk, so every tool keeps finding every model at its usual path and only the read speed differs. The 8 TB disk
reads about 200 MB/s; a model that fits in RAM is read once and then served from the page cache. `tools/hf_tier.sh`
(on the box, from any tree) moves a model between tiers:

```
tools/hf_tier.sh status                 # every model: tier, size, which disks hold a copy; free space per disk
tools/hf_tier.sh hot  Org/Name          # copy to the NVMe, verify, the entry becomes a real folder (keeps 40 GB free)
tools/hf_tier.sh cold Org/Name          # make the 8 TB copy complete, verify, the entry becomes a link to it
tools/hf_tier.sh archive Org/Name       # the same on the 3 TB disk; the 8 TB copy goes to the trash
tools/hf_tier.sh sync Org/Name          # only complete the 8 TB copy of a hot model
```

The script copies first, verifies (sizes; sha256 against the blob's name for LFS files; a byte compare for the rest),
then swaps the entry. It refuses a model that a running process has open. Nothing is deleted: the folder a swap
replaces goes to that disk's trash (`~/b70/trash`, `~/b70/hdd/trash`, `~/b70/hdd3/trash`, under `<date>/hf-tier/`),
and the NVMe space comes back only when the box's owner empties it.

huggingface_hub 2.x keeps a cache-wide shared blob store (`hub/blobs/<2 hex>/<xet hash>`, with per-repo blobs linking
into it). The script copies those as real files, so every cold or archive copy is self-contained, and it moves shared
payloads that nothing on the NVMe links to any more into the trash with their folder. Copies take hours, so run them
detached (`setsid nohup ... > ~/b70/logs/hf-tier/<name>.log`). The logic and rules are in `tools/hf_tier.py`, and
`python3 tools/test_hf_tier.py` tests them on a fake hub (laptop or box).

A model too big for RAM whose CPU reference streams its weights stays cold and goes hot only for that job:
`hf_tier.sh hot` before it, `hf_tier.sh cold` after. The 8 TB copy stays, so `cold` only re-verifies it and moves the
NVMe folder to the trash.

Flash-Next's int8 PLE file (`ple_int8.py`, 52 GB) is not a cache model. Its NVMe copy is the main tree's
`oracle-out-q4exp-ple-int8/`, a real folder with an `oracle-out*` name so that `tools/box.sh sync --delete` leaves it
alone; its cold copy is `~/b70/hdd/b70/qwen3.8-flash-next-ple-int8/`.

## After a reboot

The 8 TB and 3 TB disks are mounted from `/etc/fstab` by UUID with `nofail`, so a missing disk does not stop the
boot; the cold links just dangle until it is back. Check both after a reboot, before any validation run:
`findmnt ~/b70/hdd ~/b70/hdd3 && tools/hf_tier.sh status`. A cold model whose disk is missing shows `broken`. A stage
run checks its data once, at launch (`have.env`), so a run launched before the disk was back needs a fresh launch.
Device names move between boots, so mount by UUID, never by `/dev/sdX`.
