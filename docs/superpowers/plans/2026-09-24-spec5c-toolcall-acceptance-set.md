# Spec 5c - the tool-call acceptance set (gate A4), implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fixed set of 36 agentic-coding prompts whose next assistant turn is a tool call, with greedy baselines from the bf16 model and from the engine, and a scorer that decides spec 5's gate A4.

**Architecture:** A generator builds multi-turn conversations (system, a coding task, earlier tool calls and their results drawn from this repository's real files) with an opencode-like tool set. The checkpoint's own chat template renders them to ids. The bf16 baseline is generated on CPU with transformers in the reference container. The engine's are generated with `b70-decode --ids --n --prefill --pp-backend`. The scorer extracts the first Qwen XML `<tool_call>` from each output and compares tool name and parameters.

**Tech Stack:** Python 3 (transformers, tokenizers) in the oracle container, `b70-decode`, the box.

**Spec:** `docs/superpowers/specs/2026-09-24-spec5-int8-prefill-linears-design.md` (stage T0, bar A4). **Independent of plans 5a and 5b**; run it alongside them.

## Global Constraints

- The format is the checkpoint's: tools rendered by its `chat_template.jinja`, calls as `<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>\n</tool_call>`.
- `enable_thinking=False` for every prompt (the next turn should open with the call, and the outputs stay short).
- Greedy decoding, 192 new tokens, for bf16 and engine alike.
- Prompt length 800 to 3000 ids (the CPU bf16 baseline costs about 10 minutes per prompt at this size).
- A4, from the spec: `l0-int8` matches bf16 (same tool name, same parameters) on **at least as many** prompts as `l0` does.

## Review Focus

- **Parameters with multi-line values** (an `edit` whose old or new string spans lines). The scorer must compare them exactly after trimming a single leading and trailing newline, never after collapsing whitespace. Pinned in Task 3 with a fixture.
- **An output with no tool call**, or with text before the call. Scored as "no call", not a crash; text before the call is allowed (the template permits reasoning before it). Pinned in Task 3.
- **Two calls in one turn.** Only the first is compared, and the count is reported. Pinned in Task 3.
- **Truncated output** (192 tokens ending inside `<parameter>`). Scored as "incomplete", distinct from a mismatch. Pinned in Task 3.
- **Ids drift.** The three runs must see byte-identical id files. Pinned in Task 2 by a SHA-256 manifest the runners check.

---

### Task 1: the scenario generator

**Files:**
- Create: `tools/toolcall/make_set.py`, `tools/toolcall/tools.json`

**Interfaces:**
- Produces: `tests/golden/toolcall/<name>.json` (the messages and tools), `<name>.ids` (rendered ids), and `manifest.json` (name, id count, SHA-256 of the ids file) for 36 scenarios.

- [ ] **Step 1: The tool set**

`tools/toolcall/tools.json`: six OpenAI-style function tools, as an agent coding client sends them:

```json
[
  {"type": "function", "function": {"name": "read", "description": "Read a file from the local filesystem.",
    "parameters": {"type": "object", "properties": {
      "filePath": {"type": "string", "description": "Absolute path to the file"},
      "offset": {"type": "integer", "description": "Line to start from"},
      "limit": {"type": "integer", "description": "Number of lines to read"}}, "required": ["filePath"]}}},
  {"type": "function", "function": {"name": "grep", "description": "Search file contents with a regular expression.",
    "parameters": {"type": "object", "properties": {
      "pattern": {"type": "string"}, "path": {"type": "string"}, "include": {"type": "string"}},
      "required": ["pattern"]}}},
  {"type": "function", "function": {"name": "glob", "description": "Find files by glob pattern.",
    "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
      "required": ["pattern"]}}},
  {"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {
      "command": {"type": "string"}, "description": {"type": "string"}}, "required": ["command", "description"]}}},
  {"type": "function", "function": {"name": "edit", "description": "Replace an exact string in a file.",
    "parameters": {"type": "object", "properties": {
      "filePath": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"}},
      "required": ["filePath", "oldString", "newString"]}}},
  {"type": "function", "function": {"name": "write", "description": "Write a file.",
    "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}},
      "required": ["filePath", "content"]}}}
]
```

- [ ] **Step 2: The scenarios**

`make_set.py` builds 36 conversations from six templates times six real files of this repository (`src/runtime/prefill/linear_l0.cc`, `src/runtime/prefill/step.cc`, `src/loader/loader.cc`, `src/server/openai.cc`, `tools/rotate/rotation.py`, `tools/box.sh`). The repository root is `/ws` in the container, `/work` in the prompts. Each conversation is `system` (a short agent instruction), `user` (a task naming the file's real content), then 1 to 3 completed assistant tool calls with their `tool` results (the real file text, truncated to keep the prompt in the 800 to 3000 id band), ending where the assistant must act. The templates:

1. "Find where `<symbol>` is defined." History: none. Expected next: `grep`.
2. "Explain `<function>` in `<file>`." History: `glob` giving the path. Expected next: `read`.
3. "Rename `<local>` to `<new>` in `<file>`." History: `read` of the file. Expected next: `edit` with a single-line old string.
4. "Fix the comment above `<function>`." History: `read`. Expected next: `edit` with a multi-line old string.
5. "Run the tests for this file." History: `read`. Expected next: `bash`.
6. "Where is `<symbol>` used?" History: `grep` giving three hits. Expected next: `read` of the first hit with `offset`.

Symbols, functions and locals are chosen per file by a regex over its real text (the first `void \w+\(` / `def \w+\(`, the first local `const \w+ =`), so every task names something that exists. Render with `AutoTokenizer.from_pretrained(<snapshot>).apply_chat_template(messages, tools=tools, add_generation_prompt=True, enable_thinking=False, tokenize=True)`. Write `<name>.json`, `<name>.ids` (space-separated) and `manifest.json`. Refuse to write a scenario outside the 800 to 3000 id band; shorten the tool result instead.

- [ ] **Step 3: Run it**

Run (on the box):

```bash
tools/oracle/run_in_container.sh 'python3 tools/toolcall/make_set.py "$SNAP" tests/golden/toolcall'
```

Expected: 36 scenarios written, and a length table with min ≥ 800 and max ≤ 3000.

- [ ] **Step 4: Commit**

```bash
git add tools/toolcall/make_set.py tools/toolcall/tools.json tests/golden/toolcall/
git commit -m "tests: a 36-prompt agentic tool-call set rendered by the checkpoint's template (spec 5 T0)"
```

---

### Task 2: the baselines

**Files:**
- Create: `tools/toolcall/oracle_generate.py`, `tools/toolcall/engine_generate.sh`

**Interfaces:**
- Consumes: Task 1's `tests/golden/toolcall/`.
- Produces: `<out>/<name>.<run>.txt` (the decoded continuation) and `<name>.<run>.ids`, where `run` is `bf16`, `l0` or `l0-int8`.

- [ ] **Step 1: The bf16 runner**

`oracle_generate.py <bf16 snapshot> <set dir> <out dir>`: build `Qwen3_5ForCausalLM` the way `tools/rotate/check_rotation.py` does (`text_config`, `load_sd` in bf16, eager attention), then for each scenario, after verifying the ids file's SHA-256 against `manifest.json`, run `model.generate(input_ids, max_new_tokens=192, do_sample=False)`. Write the new ids and the text decoded with the snapshot's tokenizer (`skip_special_tokens=False`). It resumes where it stopped: a scenario whose output exists is skipped, because the whole set takes hours.

- [ ] **Step 2: The engine runner**

`engine_generate.sh <snapshot> <set dir> <out dir> <backend>`: for each scenario, verify the SHA-256, run `build/src/cli/b70-decode <snapshot> --ids <name>.ids --n 192 --prefill --pp-backend <backend> --max-len 16384`, save stdout as `<name>.<backend>.ids`, and decode with the same tokenizer (a two-line `tokenizers` call) into `<name>.<backend>.txt`.

- [ ] **Step 3: Run the baselines**

```bash
# bf16, CPU, detached (hours)
(setsid nohup tools/oracle/run_in_container.sh \
  'python3 tools/toolcall/oracle_generate.py $(ls -d /hf/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ | head -1) tests/golden/toolcall /scratch/toolcall-out' \
  > ~/models/toolcall-bf16.log 2>&1 < /dev/null &)
# engine W4A16, GPU, minutes
ZE_AFFINITY_MASK=0 tools/toolcall/engine_generate.sh "$SNAP" tests/golden/toolcall /tmp/toolcall-out l0
```

`run_in_container.sh` is started with `ORACLE_MODEL=models--Qwen--Qwen3.8-27B`. The `l0-int8` run waits for plan 5b; the script takes the backend as its argument.
Expected: 36 `.bf16.txt` and 36 `.l0.txt` files.

- [ ] **Step 4: Commit**

```bash
git add tools/toolcall/oracle_generate.py tools/toolcall/engine_generate.sh
git commit -m "tools: bf16 and engine greedy runners for the tool-call set (spec 5 T0)"
```

---

### Task 3: the scorer

**Files:**
- Create: `tools/toolcall/score.py`, `tools/toolcall/test_score.py`

**Interfaces:**
- Produces: `score.py <out dir> <reference run> <candidate run>...` prints a table per scenario (reference call, each candidate's verdict: `match`, `mismatch`, `no call`, `incomplete`) and the match counts, and exits 0.

- [ ] **Step 1: Write the failing test**

`tools/toolcall/test_score.py`, one fixture per Review Focus line:

```python
from score import parse_first_call

def test_multiline_value_exact():
    t = "<tool_call>\n<function=edit>\n<parameter=oldString>\na\n  b\n</parameter>\n</function>\n</tool_call>"
    assert parse_first_call(t) == ("call", "edit", {"oldString": "a\n  b"}, 1)

def test_text_before_call_is_allowed():
    t = "Let me look.\n<tool_call>\n<function=read>\n<parameter=filePath>\n/work/x.cc\n</parameter>\n</function>\n</tool_call>"
    assert parse_first_call(t)[:3] == ("call", "read", {"filePath": "/work/x.cc"})

def test_no_call():
    assert parse_first_call("The function returns early.")[0] == "no call"

def test_two_calls_first_wins():
    one = "<tool_call>\n<function=grep>\n<parameter=pattern>\nfoo\n</parameter>\n</function>\n</tool_call>"
    two = one.replace("grep", "glob")
    kind, name, params, count = parse_first_call(one + "\n" + two)
    assert (kind, name, count) == ("call", "grep", 2)

def test_truncated_is_incomplete():
    t = "<tool_call>\n<function=edit>\n<parameter=oldString>\nhalf a str"
    assert parse_first_call(t)[0] == "incomplete"
```

- [ ] **Step 2: Run it to see it fail**

Run: `cd tools/toolcall && python3 -m pytest test_score.py -q`
Expected: FAIL, `No module named 'score'`.

- [ ] **Step 3: Implement**

`score.py`:

```python
"""Score tool calls: the first Qwen XML <tool_call> of each output against a reference run.

    score.py <out dir> <reference run> <candidate run>...
"""
import os
import re
import sys

CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FUNC = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.S)


def _trim(v: str) -> str:
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def parse_first_call(text: str):
    """-> (kind, name, params, n_calls); kind is 'call', 'no call' or 'incomplete'."""
    calls = CALL.findall(text)
    if not calls:
        return ("incomplete" if "<tool_call>" in text else "no call", None, {}, 0)
    f = FUNC.search(calls[0])
    if not f:
        return ("incomplete", None, {}, len(calls))
    params = {k.strip(): _trim(v) for k, v in PARAM.findall(f.group(2))}
    return ("call", f.group(1).strip(), params, len(calls))


def main() -> None:
    out, ref, cands = sys.argv[1], sys.argv[2], sys.argv[3:]
    names = sorted(n[: -len(f".{ref}.txt")] for n in os.listdir(out) if n.endswith(f".{ref}.txt"))
    matches = {c: 0 for c in cands}
    print(f"| scenario | {ref} | " + " | ".join(cands) + " |")
    print("|---|---|" + "---|" * len(cands))
    for n in names:
        r = parse_first_call(open(os.path.join(out, f"{n}.{ref}.txt"), encoding="utf-8").read())
        cells = []
        for c in cands:
            p = os.path.join(out, f"{n}.{c}.txt")
            if not os.path.exists(p):
                cells.append("missing")
                continue
            got = parse_first_call(open(p, encoding="utf-8").read())
            if got[0] != "call":
                cells.append(got[0])
            elif r[0] == "call" and got[1:3] == r[1:3]:
                cells.append("match")
                matches[c] += 1
            else:
                cells.append("mismatch")
        ref_cell = f"{r[1]}" if r[0] == "call" else r[0]
        print(f"| {n} | {ref_cell} | " + " | ".join(cells) + " |")
    print()
    for c in cands:
        print(f"{c}: {matches[c]} / {len(names)} match {ref}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `cd tools/toolcall && python3 -m pytest test_score.py -q`
Expected: 5 passed.

- [ ] **Step 5: Score the baseline**

Run: `python3 tools/toolcall/score.py /tmp/toolcall-out bf16 l0` (copy the bf16 outputs next to the engine's first).
Expected: a table of 36 rows and `l0: N / 36 match bf16`. **N is the number A4 compares `l0-int8` against.** Record it in the spec's amendment section with the date.

- [ ] **Step 6: Commit**

```bash
git add tools/toolcall/score.py tools/toolcall/test_score.py
git commit -m "tools: the tool-call scorer and the W4A16 baseline for gate A4 (spec 5 T0)"
```

**Done when:** the set, the runners and the scorer are committed, and `l0`'s match count against bf16 is recorded. Plan 5b Task 4 then runs `engine_generate.sh ... l0-int8` and `score.py ... bf16 l0 l0-int8`.
