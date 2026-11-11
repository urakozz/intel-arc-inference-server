# Environment variables

What `b70-serve`, `b70-decode`, the tests and the scripts in `tools/` read from the environment. None of them is
needed to run the engine.

Read by `b70-serve` and `b70-decode`. A command-line flag always wins over the variable
that sets its default. The switches below are for experiments and A/B timing; leave them unset
to run what the gates ran.

| variable | read by | default | what it does |
|---|---|---|---|
| `HF_HOME` | both CLIs | `~/.cache/huggingface` | where a repo id is looked up: `$HF_HOME/hub/models--<org>--<name>`, revision from `refs/main`. The same place `hf download` writes to. |
| `ZE_AFFINITY_MASK` | the Level Zero driver | unset: every card visible | hides cards from the process: `ZE_AFFINITY_MASK=1` leaves only the second card, which the engine then sees as device 0. `tools/box.sh` and `tools/bench_decode.sh` forward it to the box. |
| `B70_KV_CACHE` | both CLIs | `bf16` | the default of `--kv-cache` (`bf16` or `int8`); any other value is refused. This is how the tests run the same binaries over the int8 cache. |
| `B70_PREFILL_PROFILE` | prefill | off | `1` times every prefill phase; `b70-decode --bench --prefill-length` prints the table on stderr (Qwen-family models; not yet for K2). It adds a host wait per phase, so never use it for a recorded row. |
| `B70_KOLIBRI_ATTN` | `b70-decode`, Kolibri-1 | `flash` | Kolibri-1's decode and prefill attention. `eager` is the reference's bf16 chain (spec 18 §10.1's switch, spec 20 §11): 856 launches per token instead of 756, plus a score row of 4 B x 48 heads x max_len; prefill binds the `_EAGER` flash builds (the same 2252 launches per chunk, spec 20 §12). Other values are refused. |
| `B70_Q4_MTP_NORM` | `b70-serve --spec mtp`, Qwen3.8-Flash-Next | `single` | the MTP head's `pre_fc_norm_hidden` (spec 21 decision 4): `single` is vLLM's one RMS over the 4 streams (10240), `per_stream` one per stream; read at construction, other values are refused |
| `B70_Q4_ATTN` | `b70-decode`, Qwen3.8-Flash-Next | `flash` | Qwen3.8-Flash-Next's QSA decode and prefill attention (spec 21c / 21d): `eager` is the reference's bf16 chain over the selected positions (one launch for flash's two in each QSA layer: 767 launches a token at 48 layers instead of 779); prefill binds the `_EAGER` sparse attention for the rows past position 2050 (the same launches a chunk). Other values are refused. Never run on a B70. |
| `B70_BOLT_ATTN` | `b70-decode` and `b70-serve`, Nemotron 3.5 Bolt | `flash` | Nemotron 3.5 Bolt's decode attention (spec 25c): `eager` binds the one-launch `mg_attn_eager` (the reference's bf16 chain, Decision 15): 645 launches a token instead of 653, plus a score row of 4 B x 32 heads x max_len. Other values are refused. Never run on a B70. |
| `B70_BOLT_FUSE` | `b70-decode` and `b70-serve`, Nemotron 3.5 Bolt | `none` | the fused arms of Bolt's decode list (spec 25 §6.1, Decision 16): `router` (the router rows inside the MoE block's norm launch, -40 launches), `route` (the rank recomputed in every `nb_moe_up` work-group, -40), `conv` (the conv inside `nb_ssm`, -40), `all` (533 launches), or a comma list. Each arm is bitwise the as-built list; the default stays `none` until the box's A/B (queue row 60). Other values are refused. Never run on a B70. |
| `B70_BOLT_SSM` | `b70-decode` and `b70-serve`, Nemotron 3.5 Bolt | `fp32` | the Mamba SSM state's precision (Decision 2): `bf16` halves its bytes (rounded after every update) - a measured arm; the default changes only on the operator's ruling with row 60's numbers. Other values are refused. |
| `B70_BOLT_TIE_TOL` | `bolt_golden_test` | `1e-3` | the near-tie tolerance of Bolt's routing diagnostic (the 9th / 10th sel; spec 25 Decision 11), proposed until row 57's gap distribution on the real weights; a gap of exactly 0 is undetermined |
| `B70_BOLT_A_EXP` | `tools/oracle/bolt_ref.py`, `bolt_mamba_ops.py`, `bolt_fixture.py` | `torch` | `engine`: the reference computes `A = -exp(A_log)` with a float32 port of the engine's `exp_torch` (Sleef expf u10) instead of `torch.exp` - the kernels' bitwise fixture uses it (spec 25 §15); `bolt_ref.py run --a-exp` sets it |
| `B70_BOLT_SNAPSHOT` | `template_bolt_test`, `bolt_tokenizer_test`, `bolt_request_test` | the HF cache's urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ at 8932aa8d (our export, the served checkpoint) | the snapshot whose `chat_template.jinja`, `tokenizer.json` and `tokenizer_config.json` the tests render and tokenise with (spec 25e; nothing of the model is vendored but its `config.json` and `generation_config.json`); the tests SKIP (77) without it. |
| `B70_GIT_SHA` | `b70-decode --bench` | `unknown` | the commit printed in the bench row. The box's tree has no `.git`, so `tools/bench_decode.sh` sets it. |

The scripts in `tools/`:

| variable | read by | default | what it does |
|---|---|---|---|
| `BOX` | `box.sh`, `bench_decode.sh`, `serve_bench.sh`, `box_validate.sh` | none, required | ssh target (`user@host`) of the machine with the GPU. Set it in the environment or in the untracked `tools/box.env` (copy `tools/box.env.example`); the environment wins. |
| `REMOTE_DIR` | the same | from the branch | the tree on the box, relative to its `$HOME`: `b70-inference-server` for main, `b70/trees/spec7a` for a `spec7a-*` branch, `b70/trees/research` for `research-*`, `b70/trees/<branch>` for any other (`tools/box_dir.sh`); `box_validate.sh` uses `b70-inference-server-validate`. |
| `BOX_SUFFIX` | the same | from the branch | overrides that suffix; `none` forces the base tree. |
| `JOBS` | `box.sh` (and every script that builds through it) | `44` | parallelism of the remote build |
| `BUILD_DIR`, `CMAKE_ARGS` | `box.sh` | `build`, empty | build directory on the box and extra `-D` flags for the configure step |
| `MODEL`, `DEPTH`, `TG`, `RUNS`, `PP`, `PP_CHUNK`, `PP_BACKEND`, `MAX_LEN` | `bench_decode.sh` | Qwen3.8's repo id, `4096`, `256`, `3`, the rest unset | the same as its flags (`--model`, `--depth`, ...). Unset adds nothing to the command, so it stays the one every earlier row used. It also forwards `ZE_AFFINITY_MASK`. |
| `MODEL`, `PORT`, `SERVED_NAME`, `RUNS` | `serve_bench.sh` | Qwen3.8's snapshot in the box's HF cache, `8000`, `b70`, `3` | the served checkpoint (plain: `--spec off`), its port, the model name the requests carry, and the number of llama-benchy invocations |
| `SNAP_QWEN`, `SNAP_ORNITH`, `SNAP_K2`, `SNAP_KOLIBRI`, `DEVICE`, `PORT`, ... | `box_validate.sh` | the four checkpoints, `0`, `8013` | the checkpoints, card and port the validation run uses; the full list (rounds, baseline, data directories) is in `tools/box_validate.sh --help` |
| `ORACLE_MODEL` / `ORACLE_SNAP`, `ORACLE_THREADS`, `ORACLE_IMAGE`, `HF_CACHE`, `REPO_DIR` | `tools/oracle/run_in_container.sh` and the scripts that call it (`golden.sh`, `check.sh`) | Qwen3.8 in the HF cache, all threads, the script's reference image, `~/.cache/huggingface`, this tree | the CPU reference's checkpoint (a repo directory in the cache, or an absolute snapshot path, which wins), its CPU threads, the container, and the two mounts. `golden.sh` adds `OUT_DIR` (`oracle-out`), `IDS_DIR` (`tests/golden/prompts`) and `PROMPTS` (`prose code cjk`). |

Test-only (the checkpoint paths the tests load are CMake options, e.g. `-DB70_TEST_SNAPSHOT`,
`-DB70_ORNITH_SNAPSHOT`, `-DB70_K2_SNAPSHOT`, not environment variables):

| variable | read by | default | what it does |
|---|---|---|---|
| `B70_LONGCTX_TESTS` | the `longctx` tests | unset: SKIP | `1` runs the long-context MTP variants (`mtp_verify_{32k,128k}_test`, `mtp_gpu_{32k,128k}_test`) |
| `B70_K2_TIE_TOL` | `k2_golden_test` | `1e-3` | the near-tie tolerance of K2's routing diagnostic, proposed until the reference run measures it |
| `B70_KOL_TIE_TOL` | `kolibri_golden_test`, `kolibri_partial_test` | `1e-2` | the same for Kolibri-1 (spec 20c), proposed until `kolibri_oracle.sh` prints the gap distribution |
| `B70_Q4_TIE_TOL`, `B70_Q4_SEL_TOL` | `qwen4exp_golden_test`, `qwen4exp_partial_test` | `1e-3` | Qwen3.8-Flash-Next (spec 21c): the routing diagnostic's 10th / 11th and gate S's 512th / 513th near-tie tolerances, proposed until row 30 prints the gap distributions; a gap of exactly 0 is undetermined |
| `B70_SNAPSHOT_DIR` | `template_test` | Qwen3.8's snapshot under `$HF_HOME` | the snapshot whose chat template it renders |
| `B70_K2_TOKENIZER_JSON` | `k2_tokenizer_test` | the CMake option of that name | K2's `tokenizer.json`; absent, the test is skipped |
| `B70_SPLIT_VERBOSE` | `prefill_split_test` | unset | set: also print the worst KV row per attention layer of each non-bitwise split |
