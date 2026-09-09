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
