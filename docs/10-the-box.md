# The box

This engine is developed on a laptop and built, tested and benchmarked on a
separate Linux machine with the cards in it - "the box" throughout these docs,
and `$BOX` in every command. Nothing here requires that arrangement, but
everything here assumes it, and a few of the consequences are worth writing
down.

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
| oneAPI | only for the optional sycl-tla reference backend. `icpx` is usually **not** on `PATH` in a non-interactive shell, so source the environment script or use the full path |

The C++ side builds **natively on the host**. No container is needed for
anything except the Python CPU oracle and the reference vLLM stack.

`tools/box.sh` and the benchmark helpers take the machine from
`BOX=user@host`; there is no default address worth writing down.

## Driving a remote build

`tools/box.sh` syncs the tree, configures, builds and runs `ctest` over ssh.
Set `BOX=user@host` before using it; `REMOTE_DIR`, `JOBS`, `BUILD_DIR` and
`CMAKE_ARGS` override the rest.

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
- **A second build directory lives beside the first.** The no-SYCL
  configuration (`cmake -DB70_PREFILL=OFF`) is a `BUILD_DIR` rather than a
  hardcoded path, so one tree serves every configuration.

**Long jobs must survive the ssh session.** Detach them
(`setsid nohup <script> >/dev/null 2>&1 </dev/null &`) and poll the log. This is
not fastidiousness: BuildKit discards *all* completed work, including finished
layers, if the docker client dies, and a dropped connection can silently lose
hours.

## Device selection, and the gotcha

A box with two cards will not hand you the same one twice unless you say which.
Where one card also drives a display, the render nodes do not map to selector
indices the way you would guess, and the display card is not the one you want
under a benchmark.

The contract is in [04-architecture.md](04-architecture.md) and it is worth
repeating here because it is the thing that silently invalidates a measurement:

1. `--device N` wins, and it governs **both** execution paths, so a SYCL
   backend binds the same physical card by Level Zero handle rather than by a
   second selector agreeing.
2. `ONEAPI_DEVICE_SELECTOR=level_zero:N` is the fallback. The SYCL runtime
   honours it natively; the raw Level Zero path parses it itself, because Level
   Zero does not read it.
3. `ZE_AFFINITY_MASK` sits underneath both as a driver-level filter on which
   devices are enumerated at all. Under a mask, every index above refers to the
   masked view.

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
