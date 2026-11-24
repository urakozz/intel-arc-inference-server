# tools/prefix - request logs and the prefix reuse they allow (spec 7 P0)

Record a session:

    build/src/cli/b70-serve <snapshot> --max-len 131072 --log-requests $B70_DATA/logs/prefix/oc-log-1 --port 8000

Each served request becomes `$B70_DATA/logs/prefix/oc-log-1/NNNNNN.json` (counted from 1 per server
start): `t_start`, `t_first_token`, `t_end` (steady-clock seconds), `endpoint`,
`request` (the body as received), `prompt_ids`, `out_ids`, `response_text` (every
generated piece, before tool-call parsing).

Analyse it:

    python3 tools/prefix/analyze_log.py $B70_DATA/logs/prefix/oc-log-1

prints, per request, the prompt length, the longest common prefix with the previous
request's ids (prompt + generated) and with any earlier request's, where the best
match ends (an earlier prompt's end, inside earlier generated ids, or elsewhere),
and the prompt tokens a perfect prefix cache could reuse; then the total fraction.

Tests: `python3 tools/prefix/test_analyze_log.py`.

## DFlash's final speculation replay (spec 19 v2 Task 7b)

Run after the final cost and combined acceptance gates. One idle GPU is sufficient. The owner permits the
other card to be busy when more than 16 of 44 CPU threads are available; check that before a card-only run:

```bash
tools/probe/detach.sh $B70_DATA/logs/dflash-v2/card1.log \
  python3 tools/prefix/replay_spec.py $B70_DATA/logs/opencode-session-int8 $B70_DATA/logs/dflash-v2/card1/session-int8 --card 1
```

With `--card N` the script runs on card N only; without it, it takes both cards.
Card-only runs carry a busy-machine grade; record CPU availability and concurrent
work with the results. It forces one card with
`--parallel 1` and replays `--spec off`, `--spec mtp`, and `--spec dflash`, in three rotated rounds.
Every arm keeps the prefix cache on (32 GiB), context 131072, bf16 KV, int8 target head, greedy sampling and the
recorded output cap. The MTP arm must report `adaptive per step`; an inherited `B70_MTP_STEP_K=0` is cleared.
Servers are stopped and waited for before the next arm starts. No device affinity is
set by the harness.

Use a new output directory for each recording; repeat on `opencode-session` and `opencode-session-2` as checks.
`table.txt` uses the server's decode spans and reports median ratios, including DFlash/MTP. Per-arm
`requests.jsonl` retains prompt/output ids for `serve_judge.py`; unique request tags pair the logs even if their
filenames reflect a different completion order. `run.json` records the binary hash, policy, shape, order and
judge result, selected card and resource grade. Missing f32 references leave forced contexts and return **77**, not PASS. This replay alone does
not establish Task 8's other-corpus or prefill regression bars.

After a pause, rerun the same command with `--resume`. The saved `config.json` must match the binary,
recording hash, models, serving settings and the complete content manifest of compiled `.bin` kernels.
The kernel directory defaults to the server build's `kernels/`; `--kernel-dir` names it for a custom layout. Each complete arm is rebuilt from its reply/server logs and checked
against the saved requests before reuse; an incomplete arm must be moved outside the arm directories first.
`arm.json` and `run.json` retain each arm's card when continuing on a different card. Completed results are never
overwritten by a new server run. Legacy configurations without the kernel manifest are refused for resume;
their existing requests remain usable as archived measurements.

Host-only stand-in tests: `python3 tools/prefix/test_replay_spec.py`.
