# The box

This engine is developed on a laptop and built, tested and benchmarked on a
separate Linux machine with the cards in it - "the box" throughout these docs,
and `$BOX` in every command. Nothing here requires that arrangement, but
everything here assumes it, and a few of the consequences are worth writing
down.

Since 2026-11-03 the box is also *the check*: there is no Mac-side build or test run any
more (`tools/mac_check.sh` was retired). New things on the box go under `~/b70`, and nothing new
sits at the top level of the home directory:

| path | what |
|---|---|
| `~/b70/trees/<name>` | a branch's tree (every branch but main, `tools/box_dir.sh`) |
| `~/b70/logs/<topic>/` | logs: box_check, probes (`tools/probe/detach.sh` makes the directory), benchmarks, CPU jobs |
| `~/b70/scripts/` | the operator's and the chains' scripts |
| `~/b70/scratch/`, `~/b70/models/` | probe captures; local model folders that are not in the HF cache |
| `~/b70/hdd`, `~/b70/hdd3` | the cold and archive disks ("Model storage" below) |
| `~/b70/trash/<date>/` | what is no longer needed: `mv` it here, the operator runs `rm` |

Still at the top level until one box session moves them (they are in use by every
validation run): the base tree `~/b70-inference-server` (main's tree, and the golden sets
`oracle-out*` that every other tree links to), the validation tree
`~/b70-inference-server-validate`, its G0 baselines `~/b70-inference-server-g0-<sha12>` and
its results `~/b70-validate/<tree>/<state>/`. The GPU lock files `~/b70-gpu*.lock` /
`~/b70-gpu.wait` stay in `$HOME` for good: every script and habit opens them there, and a
lock moved under a running holder no longer excludes anything.

The box is a Linux workstation with **two cards**, and the layout extends to four
(spec 16's pipeline parallel is written for two and keeps the device count a
parameter).

## What the box has to be

| | |
|---|---|
| OS | Linux. Ubuntu is what this was developed against; no Windows or macOS runtime exists |
| CPU / RAM | a many-core CPU and about 128 GB of RAM: the CPU oracle wants ~70 GB free for a layer-streamed 27B forward. The engine's own build is comfortable at full width; the reference vLLM container's torch build is not (doc 09) |
| GPU | **2 x Intel Arc Pro B70**, 32 GB each, extendable to 4. One is enough for everything except pipeline parallel (spec 16); a second is also useful as an idle control |
| Driver | `libze_intel_gpu`, IGC and `intel-ocloc`. The B70 is `bmg-g31`, so kernels build with `ocloc -device bmg-g31` |
| Level Zero | headers and loader from the distribution packages |
| Host compilers | `g++` 15.2 and CMake 4.2 or later, plus `ccache` |
| Rust | needed for the pinned `tokenizers` crate (doc 11). A user-local `rustup` with `--profile minimal` is enough; the build looks in `~/.cargo/bin` first and never changes `PATH`. The crate builds `oniguruma` from source through `cc`, so a working `/usr/bin/cc` is required |
| oneAPI | not used by the engine build. Only the stand-alone oneDNN reference probe (`tools/probe/onednn/CMakeLists.txt`) needs `icpx`, which is usually **not** on `PATH` in a non-interactive shell, so pass its full path |

The C++ side builds **natively on the host**. No container is needed for
anything except the Python CPU oracle and the reference vLLM stack.

**CPU jobs are memory-capped, with no swap.** Every CPU reference container on the box
(`tools/oracle/run_in_container.sh`) runs with `--memory X --memory-swap X`, X = 56g by default:
at its cap the kernel kills the job inside its container and nothing else notices. The other
~65 GiB of the 121 GiB are for one GPU job's host side (measured ~52 GiB: the pinned prefix cache,
up to 32 GiB, plus weight staging), the OS and the page cache of the GPU job's weight reads. A job
that needs more says so (`ORACLE_MEMORY=96g`) and runs only with the cards idle: it holds the
whole-box GPU lock for its run, and is refused (exit 75) while a GPU job runs. Why: on 2026-11-06 an
uncapped reference (its `MEM=80g` only sized its own plan) plus a GPU job's ~63 GB of driver-held
host memory filled RAM and swap; the kernel killed the reference, and the next job's swapping
beside a GPU job's weight reads took the NVMe off the bus.

`tools/box.sh` and the benchmark helpers take the machine from
`BOX=user@host`; there is no default address worth writing down.

### Versions on the box (since 2026-11-04)

| | version |
|---|---|
| kernel (running) | Linux 7.4.0-rc1, with peer-to-peer (P2P, one card writing straight into the other's memory over PCIe) supported natively; before 2026-11-04 it was 7.3.0-rc4 with a local P2P patch |
| IGC (Intel Graphics Compiler, which turns OpenCL C into GPU code) | 2.42.6, from the GitHub release debs |
| compute runtime (`libze-intel-gpu1`, `intel-opencl-icd`, `intel-ocloc`) | 26.40.40196.8, from the GitHub release debs |

The 2026-11-04 upgrade changed the compiler and the runtime, so **every speed row in
BENCHMARKS taken before it comes from the older stack** (compute runtime 26.35, IGC 2.41.5)
until it is re-measured on a quiet box (open-work, "After the table is empty"). Kernel
binaries cannot go stale silently: `cmake/ocloc_stamp.cmake` records `ocloc --version` and
the real paths of the `libigc` / `libigdfcl` libraries the loader resolves, rewrites a stamp
file only when they change, and every kernel target depends on it, so a new compiler
recompiles every kernel. The engine build uses no oneAPI compiler at all (the sycl-tla prefill
backend was removed on 2026-10-27), so the oneAPI and oneDNN versions on the box matter only
to the stand-alone oneDNN probe.

## Driving a remote build

`tools/box.sh` syncs the tree, configures, builds and runs `ctest` over ssh.
Set `BOX=user@host` before using it (or put it in the untracked `tools/box.env`);
`REMOTE_DIR`, `JOBS`, `BUILD_DIR` and `CMAKE_ARGS` override the rest. Each branch or
worktree gets its own remote tree (`tools/box_dir.sh`), so parallel lines of work do not
overwrite one another.

```sh
tools/box.sh sync                 # rsync the tree
tools/box.sh build                # sync + cmake configure + build
tools/box.sh test [regex]         # sync + build + ctest, optionally -R regex
tools/box.sh run <cmd...>         # run a command in the remote tree
tools/box.sh pull <path>          # copy a generated file back
```

Two properties of that script are load-bearing rather than incidental.

- **It syncs without `.git`.** So the remote tree cannot name the commit it is
  building. Anything that wants to record a revision has to be substituted by
  the *local* shell before the command crosses the ssh boundary.
- **A second build directory can live beside the first.** It is a `BUILD_DIR`
  rather than a hardcoded path, so one tree serves every configuration.

**Long jobs must survive the ssh session.** Detach them
(`setsid nohup <script> >/dev/null 2>&1 </dev/null &`) and poll the log. This is
not fastidiousness: BuildKit discards *all* completed work, including finished
layers, if the docker client dies, and a dropped connection can silently lose
hours.

**The merge check** is `tools/box_check.sh`, run before every merge: it syncs this checkout,
does the full build, compares every kernel's `ocloc` command line against a base
(`tools/kernel_cmdlines`: an existing variant's command line must not change or vanish), and
runs `ctest -LE checkpoint` (every test that needs no model files) on a free card. Passed,
Failed and Skipped are counted apart, because a Skipped test is not a pass. Logs go under
`~/b70/logs`.

**GPU access is serialised.** Every job that touches a card goes through
`tools/gpu_lock.sh`: `box` takes the whole machine (timed rows, two-card work, anything that
wants an idle grade), `card N` takes one card and runs the command with `ZE_AFFINITY_MASK=N`
so a second job can use the other card. A whole-box job waiting at the gate stops new card
jobs, so timed work cannot starve.

**Stopping our own jobs** is `tools/box_stop.sh list | cpu-chain | cpu-job <queue name> | hf-tier |
validation` - it stops only those (TERM, then KILL after 10 s) and refuses anything else.

**Validating work in bulk** is one command, `tools/box_validate.sh`
(`--dry-run` first to read what it will do). It runs the rows of
[box-validation-queue.md](superpowers/plans/box-validation-queue.md) as stages in one
detached orchestrator on the box, under the GPU lock, G0 (kernel binaries and gate outputs
identical to a baseline build) first and stop-the-line on a G0 failure, resumable after an ssh
drop or a reboot, and writes `box-validation-<date>.md` when it is done. Its results are on the box under `~/b70-validate/`, the latest result of every stage in
`~/b70/logs/stage-latest.txt`.

**When each runs.** Merges go in after the Mac checks (operator, 2026-11-07); the validation run is not per
merge and not every night - it runs ON DEMAND, when the operator asks (2026-11-07), on the latest main,
`tools/box_validate.sh --nightly`, and covers every merge since the last run together. `--nightly` launches nothing when only documentation changed since the newest finished run
(`tools/box_validate/changed_since.sh`: everything but `docs/` and `*.md` counts), and runs every test
(`--no-reuse`) when the newest finished full run is older than 7 days - the weekly safety net; the nights between
reuse every unchanged PASS.

**A test whose inputs did not change keeps its PASS.** Every test runs through
`tools/test_launcher.sh`; under box_check and box_validate it runs under `strace`, which records every
file the test - and every program it starts - opened, ran, checked or found missing, and the
environment it saw. `tools/box_validate/reuse.py` stores each PASS with the state of those inputs
(`~/b70/test-results/<test>.json`): files of the tree by content, model snapshots and reference outputs
by real path, size and mtime, missing files as missing, the number of visible cards. Before ctest runs,
a test whose every input is still identical takes that PASS (`<stage>.reused.junit.xml`, its old output
behind a `REUSED:` line); everything else runs. A device binary must match byte for byte (a new driver
compiler changes them with no source changed); a host program from another tree built from the same
sources with the same settings counts as the same, so the validation run takes box_check's PASSes
for the half of the suite that needs no model files. Only PASS is kept: a failure or a skip always runs
again. `tools/box_validate.sh --no-reuse` runs everything.

## Device selection, and the gotcha

A box with two cards will not hand you the same one twice unless you say which.
Where one card also drives a display, the render nodes do not map to selector
indices the way you would guess, and the display card is not the one you want
under a benchmark.

The contract is in [04-architecture.md](04-architecture.md) and it is worth
repeating here because it is the thing that silently invalidates a measurement:
the CLIs have no device flag; `ZE_AFFINITY_MASK` decides which cards the driver
enumerates, one card is device 0 of that view, two are devices 0 and 1
(`ZE_AFFINITY_MASK=1` runs on the second card alone).

**The two cards are not interchangeable.** Same binary, same checkpoint, same
ten minutes: prefill loses 3.2% to 3.4% on the second card while decode loses
only 0.2% to 0.4%. That is a difference in sustained compute rather than in
bandwidth, and it is unattributed (doc 07). Every series row in
[BENCHMARKS.md](BENCHMARKS.md) is therefore taken on one named card and says
which.

## Why an idle box matters

A benchmark row only counts as a record when the box is provably idle: zero
containers and zero processes holding a DRM file descriptor on any card,
verified before *and* after the run set. The harness measures that condition
itself and prints the grade it earned; a grade is quoted as printed and never
upgraded by hand. The grade ladder is in [BENCHMARKS.md](BENCHMARKS.md).

Idleness matters **unevenly**, and both halves are measured:

- **A CPU compile does not contend with decode.** Under a 12-core compile the
  decode step read 36.27 ms/token against an idle median of 36.32, 0.14% apart,
  because the step is 99.7% inside the GPU fence. That is why a loaded box
  downgrades a row's grade without invalidating it.
- **A CPU compile absolutely contends with the CPU oracle.** Regenerating one
  golden set took 25 min 14 s under a 12-core compile against 18 min 16 s on an
  idle box, 28% slower. Anything that runs on the CPU pays full price.

Desktop session daemons routinely hold a DRM file descriptor without submitting
any work. They still cost the row its record grade, because "submitted no work"
is a corroboration and not a proof.

## The reference stack

The vLLM container that produces the baseline numbers, the exact serve command
and the exact bench command all live in [BENCHMARKS.md](BENCHMARKS.md). Two
things about running it are worth knowing before you do.

- **Read the vLLM version string the server prints, never the image tag.** A tag
  gets rebuilt. One tag in this project's history carried two builds 149 commits
  apart under the same name, and a number attributed to the wrong one is
  unrecoverable later.
- **`--compilation-config '{"inductor_compile_config":{"pre_grad_fusion_options":{}}}'`
  is required**, not optional. torch 2.14 defaults that option to an XPU-only
  `batch_linear_lhs` fusion that costs about 30% of decode on this workload.

A golden set belongs to its container as well as to its checkpoint: the oracle
is `transformers` running inside whichever image made the dump, so two sets
dumped under different images are not interchangeable evidence even for the same
weights. Record the image beside each set.

## Gotchas that cost time

- **`docker stop` does not immediately free the container name.** A follow-up
  `docker run --name X` can fail with exit 125; `docker rm -f X` first.
- **Never `docker volume prune`** on a box that also runs anything else. It
  takes unrelated named volumes with it.
- **Model load for a 27B is 2 to 4 minutes in the reference stack**, and graph
  capture plus compile adds another 2 to 4. Budget about 8 minutes from launch
  to a servable endpoint. Our own loader takes 13.6 s on a warm page cache.
- **A consumer NVMe can drop out under APST** (autonomous power state
  transitions). `nvme_core.default_ps_max_latency_us=0` on the kernel command line
  prevents it. If the box loses its disk after a kernel update, check that first.

## Model storage: hot, cold, archive

Three disks hold the Hugging Face cache, in tiers (since 2026-11-06; the USB disk and its bind mounts are gone):

| tier | where | holds |
|---|---|---|
| hot | NVMe, `~/.cache/huggingface/hub` | the working set: the models the next runs read, as real folders |
| cold | 8 TB WD Red, `~/b70/hdd/hf-hub` | every model, always (the bf16 originals, our int4 exports, everything hot) |
| archive | 3 TB, `~/b70/hdd3/hf-hub` | parked models (Nanbeige, Flint), instead of cold |

Every `~/.cache/huggingface/hub/models--X` is either a real folder (hot) or a symlink to `models--X` on the 8 TB or
3 TB disk, so every tool keeps finding every model at its usual path - only the read speed differs (the 8 TB disk
reads ~200 MB/s; a model that fits in RAM is read once and then served from the page cache). `tools/hf_tier.sh`
(on the box, from any tree) moves a model between tiers:

```
tools/hf_tier.sh status                 # every model: tier, size, which disks hold a copy; free space per disk
tools/hf_tier.sh hot  Org/Name          # copy to the NVMe, verify, the entry becomes a real folder (keeps 40 GB free)
tools/hf_tier.sh cold Org/Name          # make the 8 TB copy complete, verify, the entry becomes a link to it
tools/hf_tier.sh archive Org/Name       # the same on the 3 TB disk; the 8 TB copy goes to the trash
tools/hf_tier.sh sync Org/Name          # only complete the 8 TB copy of a hot model
```

It copies first, verifies (sizes; sha256 against the blob's name for LFS files; byte compare for the rest), then
swaps the entry; it refuses a model a running process has open. Nothing is deleted: the folder a swap replaces goes to
that disk's trash (`~/b70/trash`, `~/b70/hdd/trash`, `~/b70/hdd3/trash`, under `<date>/hf-tier/`) and the NVMe
space comes back only when the operator empties it. huggingface_hub 2.x keeps a cache-wide shared blob store
(`hub/blobs/<2 hex>/<xet hash>`, per-repo blobs are links into it); the script copies those as real files, so every
cold or archive copy is self-contained, and moves the shared payloads nothing on the NVMe links to any more into the
trash with their folder. Copies take hours: run them detached (`setsid nohup ... > ~/b70/logs/hf-tier/<name>.log`).
Logic and rules: `tools/hf_tier.py`; tests on a fake hub: `python3 tools/test_hf_tier.py` (Mac or box).

A model too big for RAM whose CPU reference streams its weights (Bolt's bf16 original, 150 GB) stays cold and goes
hot only for that job: `hf_tier.sh hot` before it, `hf_tier.sh cold` after (the 8 TB copy stays, so `cold` only
re-verifies it and moves the NVMe folder to the trash).

Flash-Next's int8 PLE file (`ple_int8.py`, 52 GB) is not a cache model: its NVMe copy is the main tree's
`oracle-out-q4exp-ple-int8/` (a real folder - an `oracle-out*` name, so `tools/box.sh sync --delete` leaves it alone),
its cold copy `~/b70/hdd/b70/qwen3.8-flash-next-ple-int8/`.

## After a reboot

The 8 TB and 3 TB disks are mounted from `/etc/fstab` by UUID (`nofail`: a missing disk does not stop the boot, the
cold links just dangle until it is back). Check both after a reboot, before any validation run:
`findmnt ~/b70/hdd ~/b70/hdd3 && tools/hf_tier.sh status` - a cold model whose disk is missing shows `broken`.
A stage run checks its data once, at launch (`have.env`), so a run launched before the disk was back needs a fresh
launch. Device names move between boots, so mount by UUID, never by `/dev/sdX`.
