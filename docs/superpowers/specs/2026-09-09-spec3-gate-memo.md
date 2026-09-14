# Spec 3 - integration-gate attempt: blocked at the exact-ID bar

Date: 2026-09-09. Integration commits `0933e2f..983e852` on
`spec1.7-codex-exp`. Authority: the amended spec-3 design, plan 7d, and the
controller rulings for checkpoint `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`
snapshot `84575a18f209992ef96d819b31f924b489e3d55d` (bf16 `lm_head`).

**Verdict: BLOCKED. No `spec3-done` tag.** The plan's Task 2 rule requires a
mismatch to print both lists and stop. The server process owned by the test was
terminated with SIGTERM and reaped; no server was left running. Tasks 3-5 were
not run after that hard-bar failure, so this memo deliberately contains no
HTTP, CLI-control, or host-sampling measurements.

**Update 2026-09-14: bar 3 is now met - see §4.** The comparator question this
memo raised in §2 was resolved by putting both sides on the same engine path
(`--prefill`), not by relaxing the server test. §1-§3 below are kept as the
historical record of the 2026-09-09 finding.

## 1. Bar score

| bar | result | evidence |
|---|---|---|
| 1. tokenizer parity + streamer | carried forward: met | T1/T2's committed host suite: 10,240 parity/streamer cases, zero mismatches |
| 2. template parity | carried forward: met | T2's three byte-identical checkpoint-template vectors |
| 3. golden through server | **SHORT** | `golden_server_test`: `prose` 32/32 exact; `code` 32/32 exact; `cjk` exact prompt IDs but generation diverges at position 23 |
| 4. llama-benchy coherence and rows | not run | Task 2 stopping rule |
| 5. HTTP tg within 2% of CLI | not run | Task 2 stopping rule |
| 6. full suite / no protected decode changes | not completed | the targeted build passed; protected directories are empty over `0801d20..983e852`, but the full suite and decode re-gate were not run after bar 3 failed |

`spec2-done` is not a tag in this checkout, so the plan's literal
`git diff --stat spec2-done..HEAD` control cannot run. The directly relevant
integration-range control is empty for `src/runtime`, `src/kernels`, `src/l0`,
`src/loader`, and `src/model`.

## 2. The failing comparison

The gate starts one `b70-serve` using the real engine and `Engine::prefill`,
then, for each raw prompt, checks the server's tokenizer IDs against the
committed `.ids` and compares generation to `b70-decode --ids`, which uses
`Engine::ingest`.

| prompt | prompt ids | generated IDs |
|---|---:|---:|
| prose | exact | 32 / 32 exact |
| code | exact | 32 / 32 exact |
| cjk | exact | 22-token prefix exact; token 23 onward diverges |

For `cjk`, server IDs were

```
[29545, 271, 95815, 108553, 97663, 108447, 96494, 3709, 98844, 95895, 97771, 95726, 114183, 112338, 100700, 96986, 1710, 271, 550, 220, 99737, 96863, 271, 99737, 96863, 95761, 108762, 96151, 108767, 97164, 96026, 97663]
```

and `b70-decode` IDs were

```
[29545, 271, 95815, 108553, 97663, 108447, 96494, 3709, 98844, 95895, 97771, 95726, 114183, 112338, 100700, 96986, 1710, 271, 550, 220, 99737, 96863, 4960, 96147, 123676, 95726, 127305, 96760, 271, 108549, 97799, 97204]
```

This does not license an engine change or a looser server test. The two paths
are not the same replay sequence: prefill is algebraically equivalent to
per-token ingest but intentionally independently rounded, and the existing
prefill-consistency gate documents a tie-aware, teacher-forced comparison for
that reason. Plan 7d's premise that this direct comparison had “no tie question”
is therefore falsified for this prompt. The next action needs an operator ruling
on the intended contract: make serving use ingest for this correctness gate,
provide a prefill-equivalent decode reference, or define a principled tie-aware
server criterion.

## 3. Smoke and owed work

Task 1's device-1 smoke loaded the gate checkpoint successfully. Its
non-streaming answer to “What is the capital of France?” ended: “The capital of
France is Paris.” Streaming produced a role frame followed by individual content
frames and `[DONE]`. The launched PID `444748` was stopped by that exact PID.

No performance claim follows from this smoke. The box's idle window was not
used for Task 3's rows because Task 2 had already reached the plan's stopping
condition. Host sampling was not implemented or measured; Task 5's planned
`<= 0.62 ms/token` pre-registration was not reached, and greedy remains the
only serving path.

Requested ruling: choose the correctness comparator/path before any
llama-benchy, HTTP-cost, or host-sampling work resumes.

## 4. 2026-09-14 - bar 3 passed with `--prefill`

Commit `0ce35c4` (`feat(cli): add --prefill to ids replay`) answered the
ruling request in §3 by giving the CLI reference a way onto the server's own
path instead of loosening the server test: `b70-decode --ids <file> --n N
--prefill [--pp-chunk C]` now routes the prompt through `Engine::prefill`
instead of `Engine::ingest`. `tests/server/golden_server_test.cc` passes
`--prefill` to the CLI reference, so `b70-serve` (which prefills) and the CLI
reference run the **same engine path** - the tie question in §2 no longer
applies, and exact equality is the legitimate bar.

A same-day attempt at bar 3 on this branch failed at startup with
`ZE_RESULT_ERROR_DEVICE_LOST` on the first Level Zero memory copy, shortly
after the box had lost its network and rebooted; one identical retry passed
bar 3 in 115.52 s with exact prompt-id and 32/32 generated-id equality on all
three golden prompts. That run was never committed - it exists only in a log.

This is the committed confirmation, run after another box reboot (uptime
~4h22m at run start, no other GPU work observed, load ~16 from an unrelated
`clang` build left alone). Build: `tools/box.sh build`, 100% clean. Full
suite: `ZE_AFFINITY_MASK=1 ctest --test-dir build --output-on-failure`, 66
tests, **100% passed, 0 failed** (`golden_gate_test` and `prefill_gate_test`
skipped - no CPU oracle dump for this checkpoint, a known missing input, not
a regression). No `ZE_RESULT_ERROR_DEVICE_LOST` on this run. `golden_server_test`
passed in 126.93 s:

| prompt | prompt ids | generated ids |
|---|---|---|
| prose | exact | 32/32 exact |
| code  | exact | 32/32 exact |
| cjk   | exact | 32/32 exact |

From `build/Testing/Temporary/LastTest.log`:

```
prose: prompt ids identical, 32/32 generated ids identical
code: prompt ids identical, 32/32 generated ids identical
cjk: prompt ids identical, 32/32 generated ids identical
```

**Bar 3 is met**, on both the transient-retry run and this clean post-reboot
run, via the same engine path on both sides (`--prefill`). Bars 4-5
(llama-benchy coherence/rows; HTTP tg within 2% of CLI) and host sampling
remain deferred by operator ruling - not attempted here, per the task scope
that produced this addendum. No `spec3-done` tag is created by this run.

## 5. Task 5 pre-registration - host sampling cost bar (2026-09-14, committed before measurement)

Recorded here, in its own commit, before `sample_into_control` is implemented
or measured, per plan 7d Task 5 Step 1's rule that the bar precedes the
number.

- **Bar:** host sampling (temperature/top-k/top-p, read-back + partial sort +
  `discrete_distribution`) must cost **≤ 2% of the 31.0 ms/token step, i.e.
  ≤ 0.62 ms/token**, measured as `sampled ms/token − greedy ms/token` on
  `b70-serve`, same box, same checkpoint (`urakozz` `84575a1`), 3 requests each
  side, median.
- **Derived estimate (pre-registered, not yet measured): 0.2-0.4 ms/token** -
  993 KB logits readback over PCIe at ~10 GB/s ≈ 0.1 ms, plus a 248,320-element
  top-k partial sort ≈ 0.1-0.3 ms.
- **Ship rule:** if the measured cost is ≤ 0.62 ms/token, sampling ships on by
  default (a request may still ask for greedy). If it measures over the bar,
  the code stays but the server's own default remains greedy-only for now
  (`--sampling off` default), and the number is recorded, not hidden. Greedy
  itself is never slowed either way - the readback only runs when
  `!sampling.greedy`.
- Measurement to follow in this same document once taken.
