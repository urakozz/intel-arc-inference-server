# Environment variables

What `b70-serve`, `b70-decode`, the tests and the scripts in `tools/` read from the environment.

**You do not need any of these to run the engine.** The engine configures itself; a command-line flag always
wins over a variable. The page has four parts: the two variables a user may meet, the switches for experiments,
the scripts' settings, and test-only variables. Leave the experiment switches unset to run what the tests ran.

## The two a user may meet

| variable | default | what it does |
|---|---|---|
| `HF_HOME` | `~/.cache/huggingface` | where a model's repo id is looked up: `$HF_HOME/hub/models--<org>--<name>`, at the revision `refs/main` names. The same place `hf download` writes to. |
| `ZE_AFFINITY_MASK` | unset: every card visible | read by the Level Zero driver, not by the engine. It hides cards from the process: `ZE_AFFINITY_MASK=1` leaves only the second card, which the engine then sees as device 0. On the box, never set it by hand: `tools/gpu_lock.sh` sets it together with the card's lock. |

## Switches for experiments (the engine)

| variable | read by | default | what it does |
|---|---|---|---|
| `B70_SPEC_LOOKUP` | `b70-serve` with `--spec auto`, `mtp` or `dflash` | on | `0` turns prompt lookup off, so the server runs plain MTP or plain DFlash. Prompt lookup is the copying of earlier text as a guess (see [18-running-models.md](18-running-models.md)); this switch is the comparison arm against it. Only the exact value `0` counts. `--spec dflash-lookup` ignores it. |
| `B70_MTP_STEP_K` | `b70-serve` with the MTP head | on | `0` goes back to choosing the number of MTP guesses once per request. By default the server decides at every step, from the head's own confidence, whether to guess again. |
| `B70_KV_CACHE` | both CLIs | `bf16` | the default of `--kv-cache` (`bf16` or `int8`); any other value is refused. The tests use it to run the same binaries over the int8 cache. |
| `B70_PREFILL_PROFILE` | the prompt pass | off | `1` times every phase of the prompt pass, and `b70-decode --bench --prefill-length` prints the table on stderr; `2` also names the kernel of every launch. It adds a host wait per phase, so never use it for a recorded speed. |
| `B70_PF_GDN_DUMP` | the prompt pass | unset | a directory: the inputs of three delta-net calls are written there for debugging. It waits on the host, so never in a timed run. |
| `B70_GIT_SHA` | `b70-decode --bench` | `unknown` | the commit printed in the bench row. The box's tree has no `.git`, so `tools/bench_decode.sh` sets it. |

## Switches for experiments (single models)

Each of these selects a measured alternative form of one model's kernels. The default is what the tests ran.
Any value not listed is refused.

| variable | model | default | what it does |
|---|---|---|---|
| `B70_KOLIBRI_ATTN` | Kolibri-1 (`b70-decode`) | `flash` | `eager` runs the attention the way the reference implementation does, as a chain of bf16 steps: 856 kernel launches a token instead of 756, and a score row of 4 B x 48 heads x context length. |
| `B70_Q4_ATTN` | Qwen3.8-Flash-Next (`b70-decode`) | `flash` | `eager` is the reference's bf16 chain for its sparse attention layers: 767 launches a token at 48 layers instead of 779. Not yet run on a card. |
| `B70_Q4_MTP_NORM` | Qwen3.8-Flash-Next (`b70-serve --spec mtp`) | `single` | how the MTP head normalises its input: `single` is one RMS norm over the 4 streams together (what vLLM does), `per_stream` is one per stream. |
| `B70_BOLT_ATTN` | Nemotron 3.5 Bolt | `flash` | `eager` is the reference's bf16 attention chain in one launch: 645 launches a token instead of 653. Not yet run on a card. |
| `B70_BOLT_FUSE` | Nemotron 3.5 Bolt | `none` | fused variants of Bolt's decode step, each bitwise the same as the default and each saving 40 launches a token: `router` (the router inside the MoE block's norm launch), `route` (the expert ranking recomputed in every expert kernel), `conv` (the convolution inside the state-space kernel), `all` (533 launches), or a comma list. Not yet run on a card. |
| `B70_BOLT_SSM` | Nemotron 3.5 Bolt | `fp32` | the precision of the Mamba state: `bf16` halves its bytes and rounds after every update. |

## The scripts in `tools/`

| variable | read by | default | what it does |
|---|---|---|---|
| `BOX` | `box.sh`, `bench_decode.sh`, `serve_bench.sh`, `box_validate.sh` | none, required | ssh target (`user@host`) of the machine with the GPU. Set it in the environment or in the untracked `tools/box.env` (copy `tools/box.env.example`); the environment wins. |
| `REMOTE_DIR` | the same | from the branch | the tree on the box, relative to its `$HOME`: `b70-inference-server` for main, `b70/trees/spec7a` for a `spec7a-*` branch, `b70/trees/research` for `research-*`, `b70/trees/<branch>` for any other (`tools/box_dir.sh`); `box_validate.sh` uses `b70-inference-server-validate`. |
| `BOX_SUFFIX` | the same | from the branch | overrides that suffix; `none` forces the base tree. |
| `JOBS` | `box.sh` (and every script that builds through it) | `44` | parallelism of the remote build |
| `BUILD_DIR`, `CMAKE_ARGS` | `box.sh` | `build`, empty | build directory on the box and extra `-D` flags for the configure step |
| `MODEL`, `DEPTH`, `TG`, `RUNS`, `PP`, `PP_CHUNK`, `PP_BACKEND`, `MAX_LEN` | `bench_decode.sh` | Qwen3.8's repo id, `4096`, `256`, `3`, the rest unset | the same as its flags (`--model`, `--depth`, ...). Unset adds nothing to the command, so it stays the one every earlier row used. It also forwards `ZE_AFFINITY_MASK`. |
| `MODEL`, `PORT`, `SERVED_NAME`, `RUNS` | `serve_bench.sh` | Qwen3.8's snapshot in the box's HF cache, `8000`, `b70`, `3` | the served checkpoint (plain: `--spec off`), its port, the model name the requests carry, and the number of llama-benchy invocations |
| `SNAP_QWEN`, `SNAP_ORNITH`, `SNAP_K2`, `SNAP_KOLIBRI`, `DEVICE`, `PORT`, ... | `box_validate.sh` | the four checkpoints, `0`, `8013 + 100 x DEVICE` | the checkpoints, card and port the validation run uses; the full list (rounds, baseline, data directories) is in `tools/box_validate.sh --help` |
| `ORACLE_MODEL` / `ORACLE_SNAP`, `ORACLE_THREADS`, `ORACLE_IMAGE`, `HF_CACHE`, `REPO_DIR` | `tools/oracle/run_in_container.sh` and the scripts that call it (`golden.sh`, `check.sh`) | Qwen3.8 in the HF cache, all threads, the script's reference image, `~/.cache/huggingface`, this tree | the CPU reference's checkpoint (a repo directory in the cache, or an absolute snapshot path, which wins), its CPU threads, the container, and the two mounts. `golden.sh` adds `OUT_DIR` (`oracle-out`), `IDS_DIR` (`tests/golden/prompts`) and `PROMPTS` (`prose code cjk`). |

| `B70_BOLT_A_EXP` | `tools/oracle/bolt_ref.py`, `bolt_mamba_ops.py`, `bolt_fixture.py` | `torch` | `engine`: Bolt's CPU reference computes `A = -exp(A_log)` with a float32 port of the engine's own `exp` instead of `torch.exp`, so that the kernels can be checked against it bit for bit; `bolt_ref.py run --a-exp` sets it. |

## Test-only

The checkpoint paths the tests load are CMake options (`-DB70_TEST_SNAPSHOT`, `-DB70_ORNITH_SNAPSHOT`,
`-DB70_K2_SNAPSHOT`), not environment variables. A *near-tie tolerance* below is how close two candidates'
scores may be before a test calls the choice between them undetermined.

| variable | read by | default | what it does |
|---|---|---|---|
| `B70_LONGCTX_TESTS` | the `longctx` tests | unset: SKIP | `1` runs the long-context MTP variants (`mtp_verify_{32k,128k}_test`, `mtp_gpu_{32k,128k}_test`) |
| `B70_K2_TIE_TOL` | `k2_golden_test` | `1e-3` | the near-tie tolerance of K2-Horizon's expert routing check |
| `B70_KOL_TIE_TOL` | `kolibri_golden_test`, `kolibri_partial_test` | `1e-2` | the same for Kolibri-1 |
| `B70_BOLT_TIE_TOL` | `bolt_golden_test`, `bolt_prefill_test` | `1e-3` | the same for Bolt's expert routing (the 9th against the 10th expert); a gap of exactly 0 is undetermined |
| `B70_BOLT_SNAPSHOT` | `template_bolt_test`, `bolt_tokenizer_test`, `bolt_request_test` | our Bolt export in the HF cache | the snapshot whose chat template and tokenizer files the tests use; without it the tests SKIP |
| `B70_Q4_TIE_TOL`, `B70_Q4_SEL_TOL` | `qwen4exp_golden_test`, `qwen4exp_partial_test` | `1e-3` | the same for Qwen3.8-Flash-Next: expert routing (the 10th against the 11th expert) and the sparse attention's position selection (the 512th against the 513th); a gap of exactly 0 is undetermined |
| `B70_SNAPSHOT_DIR` | `template_test`, `prompt_order_test` | Qwen3.8's snapshot under `$HF_HOME` | the snapshot whose chat template it renders |
| `B70_K2_TOKENIZER_JSON` | `k2_tokenizer_test` | the CMake option of that name | K2's `tokenizer.json`; absent, the test is skipped |
| `B70_SPLIT_VERBOSE` | `prefill_split_test` | unset | set: also print the worst KV row per attention layer of each non-bitwise split |
