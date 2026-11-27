# Environment variables

This page lists what `b70-serve`, `b70-decode`, the tests and the scripts in `tools/` read from the environment. It is
a reference for people who run experiments, tests or the scripts.

You do not need any of these to run the engine. The engine configures itself, and a command-line flag always wins
over a variable. Leave the experiment switches unset to run what the tests ran.

The page has four parts:

1. the two variables a user may meet;
2. switches for experiments;
3. the scripts' settings;
4. test-only variables.

## The two a user may meet

| variable | default | what it does |
|---|---|---|
| `HF_HOME` | `~/.cache/huggingface` | where a model's repo id is looked up: `$HF_HOME/hub/models--<org>--<name>`, at the revision `refs/main` names. The same place `hf download` writes to |
| `ZE_AFFINITY_MASK` | unset: every card visible | read by the Level Zero driver, not by the engine. It hides cards from the process: `ZE_AFFINITY_MASK=1` leaves only the second card, which the engine then sees as device 0. |

## Switches for experiments: the engine

| variable | read by | default | what it does |
|---|---|---|---|
| `B70_SPEC_LOOKUP_ROWS` | one card, MTP with prompt lookup | `64` | `16`, `32` or `64`: the widest verify list a long copied run may use. Any other value is refused |
| `B70_MTP_ADAPTIVE` | `b70-serve` with the MTP head | on | `0` drafts a fixed K = `--spec-max` instead of choosing K from each request's acceptance. For A/B runs and for measuring the cost table |
| `B70_MTP_STEP_K` | `b70-serve` with the MTP head | on | `0` goes back to choosing the number of MTP guesses once per request. By default the server decides at every step, from the head's own confidence, whether to guess again |
| `B70_KV_CACHE` | both CLIs | `bf16` | the default of `--kv-cache` (`bf16` or `int8`); any other value is refused. The tests use it to run the same binaries over the int8 cache |
| `B70_PREFILL_PROFILE` | the prompt pass | off | `1` times every phase of the prompt pass, and `b70-decode --bench --prefill-length` prints the table on stderr; `2` also names the kernel of every launch. It adds a host wait per phase, so never use it for a recorded speed |
| `B70_PF_GDN_DUMP` | the prompt pass | unset | a directory: the inputs of three delta-net calls are written there for debugging. It waits on the host, so never use it in a timed run |
| `B70_GIT_SHA` | `b70-decode --bench` | `unknown` | the commit printed in the bench row. A tree copied without `.git` cannot know it, so whoever runs the bench sets it |

## Switches for experiments: single models

Each of these selects a measured alternative form of one model's kernels. The default is what the tests ran. Any
value not listed is refused.

| variable | model | default | what it does |
|---|---|---|---|
| `B70_KOLIBRI_ATTN` | Kolibri-1 | `flash` | `eager` runs the attention the way the reference implementation does, as a chain of bf16 steps: 856 kernel launches a token instead of 756, and a score row of 4 B x 48 heads x context length. Decode and prefill alike |
| `B70_Q4_RESIDENT` | Qwen3.8-Flash-Next (any load) | unset | a test placement of the routed experts: `all`, `none`, `random:<f>:<seed>` or `adversarial` (the profile's most-routed experts in host memory). Unset, the planner fills VRAM from the routing profile. Placement never changes the output (tested byte for byte) |
| `B70_Q4_EXPERTS` | Qwen3.8-Flash-Next (any load) | `table` | `direct` binds the expert kernels without the address table, only when every expert is in VRAM. For A/B timing |
| `B70_Q4_LOAD_HASH` | Qwen3.8-Flash-Next (any load) | unset | `1` prints one line of hashes of every loaded buffer (each card's weights read back, the host experts, the PLE table) after the load, for comparing two loader versions byte for byte. Adds about a minute |
| `B70_Q4_ATTN` | Qwen3.8-Flash-Next | `flash` | `eager` is the reference's bf16 chain for its sparse attention layers: 767 launches a token at 48 layers instead of 779 in decode, and the eager sparse attention in prefill |
| `B70_INK_GLUE` | Inkling 2 Mini (the prompt pass) | `f32` | `bf16` keeps the rows passed between the prompt pass's glue kernels (the cross-pass messages, the controllers' mixed rows, the expert GEMMs' outputs) in bf16 instead of fp32; the sums inside each kernel stay fp32 and the residual stream stays fp32. Faster, not bit-identical to `f32`. For A/B timing and quality checks |
| `B70_Q4_HC` | Qwen3.8-Flash-Next (any load) | `bf16` | `int8` converts the hyper-connection mixers to int8 at load (one fp16 scale per 32 weights), so a decoded token reads about 0.6 GB less; not bit-identical to `bf16`. For A/B timing and quality checks |
| `B70_Q4_MTP_NORM` | Qwen3.8-Flash-Next (`b70-serve` with MTP) | `single` | how the MTP head normalises its input: `single` is one RMS norm over the 4 streams together (what vLLM does), `per_stream` is one per stream |
| `B70_CPU_EXPERTS_AVX2` | Qwen3.8-Flash-Next (host-RAM experts computed on the CPU) | unset | `1` uses the AVX2 kernel even on a CPU with AVX-512; the startup line then says `AVX2 (forced)`. The AVX-512 kernel has not yet been tested; if output looks wrong (garbage, wrong tokens), please report it and set `B70_CPU_EXPERTS_AVX2=1` to use the AVX2 kernel |
| `B70_Q4_PROFILE` | Qwen3.8-Flash-Next (`b70-decode --bench`) | unset | `R`: after the bench, print each launch's mean device time over R profiled replays. For probes |

## The scripts in `tools/`

| variable | read by | default | what it does |
|---|---|---|---|
| `B70_DATA` | tests (at configure time), `tools/ocloc_cache.py`, `tools/quantize/*.sh`, `tools/oracle`, `tools/toolcall` | unset | the root of the reference data some tests and tools read: the CPU reference sets under `oracle-out/<model>/<set>/` (`tools/oracle/README.md`), captures under `logs/` and `scratch/`, quantize output and logs. Unset, those tests skip (exit 77) and the quantize scripts stop and ask for it. It also places the ocloc cache at `$B70_DATA/cache/ocloc`, else `~/.cache/b70-ocloc` |
| `ORACLE_MODEL` / `ORACLE_SNAP`, `ORACLE_THREADS`, `ORACLE_IMAGE`, `ORACLE_MEMORY`, `HF_CACHE`, `REPO_DIR` | `tools/oracle/run_in_container.sh` and the scripts that call it (`golden.sh`, `check.sh`) | Qwen3.8 in the HF cache, all threads, the script's reference image, `64g`, `~/.cache/huggingface`, this tree | the CPU reference's checkpoint (a repo directory in the cache, or an absolute snapshot path, which wins), its CPU threads, the container, its memory cap, and the two mounts. `golden.sh` adds `OUT_DIR` (`$B70_DATA/oracle-out/qwen38/golden`), `IDS_DIR` (`tests/golden/prompts`) and `PROMPTS` (`prose code cjk`) |
| `B70_GPU_LOCK` | `tools/oracle/run_in_container.sh` | unset | a lock file the machine's GPU jobs hold: a reference run above the 64g CPU cap takes it exclusively (refused while a GPU job holds it). Unset, no lock is taken |
| `B70_CPU_LOCK` | `tools/quantize/*.sh` (`DEVICE=cpu`) | unset | a lock file for CPU model jobs, so one runs at a time; unset, no lock is taken |
| `B70_CARD_RUN` | `tools/quantize/*.sh` (`DEVICE=xpu`) | unset | a wrapper the card jobs run through, called as `<wrapper> <card> -- <cmd...>` (a machine's GPU lock); unset, the job runs with `ZE_AFFINITY_MASK=<card>` |

## Test-only

The checkpoint paths the tests load are CMake options (`-DB70_TEST_SNAPSHOT`, `-DB70_ORNITH_SNAPSHOT`), not
environment variables. A near-tie tolerance below is how close two candidates' scores may be before a test calls the
choice between them undetermined.

| variable | read by | default | what it does |
|---|---|---|---|
| `B70_LONGCTX_TESTS` | the `longctx` tests | unset: SKIP | `1` runs the long-context MTP variants (`mtp_verify_{32k,128k}_test`, `mtp_gpu_{32k,128k}_test`) |
| `B70_KOL_TIE_TOL` | `kolibri_golden_test`, `kolibri_partial_test` | `1e-2` | the near-tie tolerance of Kolibri-1's expert routing check |
| `B70_Q4_TIE_TOL`, `B70_Q4_SEL_TOL` | `qwen4exp_golden_test`, `qwen4exp_partial_test` | `1e-3` | the same for Qwen3.8-Flash-Next: expert routing (the 10th against the 11th expert) and the sparse attention's position selection (the 512th against the 513th); a gap of exactly 0 is undetermined |
| `B70_Q4_EXPERT_CHECK` | the Qwen3.8-Flash-Next engine (`qwen4exp_tier_synth_test`, `qwen4exp_tier_synth_pp_test` set it) | on | `off` skips the load-time check that reads every expert block back through the address tables. Only for the residency tests, which must run nothing on the card before the decode lists |
| `B70_SNAPSHOT_DIR` | `template_test`, `prompt_order_test` | Qwen3.8's snapshot under `$HF_HOME` | the snapshot whose chat template it renders |
| `B70_Q4EXP_TOKENIZER_JSON`, `B70_KOLIBRI_TOKENIZER_JSON` | the Flash-Next and Kolibri-1 tokenizer tests | the snapshot's file | a `tokenizer.json` to test instead |
| `B70_DUMP_FORCED`, `B70_FORCED_JUDGE`, `B70_FORCED_BF16` | the f32 judge (`tests/golden/forced_judge.h`, `tests/server/spec_judge.h`) | unset | directories: where to dump forced-token runs, the f32 judge's references, and the bf16 twin's references ([14-golden-gate.md](14-golden-gate.md)) |
| `B70_MTP_JUDGE` | `mtp_head_test` | unset | the directory of the MTP head's f32 references |
| `B70_SERVE_ARGS` | `golden_server_test` | unset | extra `b70-serve` arguments for the test's server |
| `B70_SPLIT_VERBOSE`, `B70_SPLIT_REFS` | `prefill_split_test` | unset | `B70_SPLIT_VERBOSE` set: also print the worst KV row per attention layer of each non-bitwise split; `B70_SPLIT_REFS=<f32>,<bf16>`: the f32 and bf16 reference files the f32 judge compares the splits against |
| `B70_COPY_PINNED_GB`, `B70_COPY_STRESS`, ... | `copy_test` | unset | probes of large pinned host allocations and copy-engine stress, run by hand |
