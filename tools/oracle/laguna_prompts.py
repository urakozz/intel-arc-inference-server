#!/usr/bin/env python3
"""Plan 30a Task 4: the trace sources rendered through Laguna S 2.1's own template and tokenizer.

    laguna_prompts.py <snapshot> --out DIR [--answers DIR]

<snapshot> needs only the original's small files (tokenizer.json, tokenizer_config.json, chat_template.jinja). The
sources are Flash-Next's traces' (offload_curve.py's classes), each written as DIR/<name>.ids with its first answer
position in DIR/sources.tsv (`NAME<TAB>FILE<TAB>ANSWER_FROM`, the list `laguna_ref.py trace --list` reads):

  the 36 tool-call scenarios  tests/golden/toolcall/<name>.json, manifest.json's names (messages, tools, enable_thinking) rendered with
                              add_generation_prompt; with --answers, the recorded answer's FIRST tool call
                              (<answers>/<name>.bf16.txt, Qwen3.8's A4 reference outputs, tools/toolcall/a4_ref.sh qwen
                              ref; score.py's parser) appended as the assistant turn, so the rows from answer_from on
                              are that call in Laguna's syntax; without one the scenario is prompt only
  q4exp_agentic               tests/golden/prompts/q4exp_agentic.json without its trailing tool result: the rows from
                              answer_from on are its last assistant turn (the name keeps offload_curve.py's class)
  code, prose                 tests/golden/prompts/{code,prose}.txt after BOS (2): prompt only

The render is text (transformers' apply_chat_template), tokenised by tokenizer.json itself (the engine's binding),
without added special tokens (the template writes BOS). Every answer is checked to extend its prompt's ids exactly.
Prints rows per source and the total against Flash-Next's 80,604 (Task 4's within-20 % check).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402

REPO = os.path.dirname(os.path.dirname(_HERE))
TOOLCALL = os.path.join(REPO, "tests", "golden", "toolcall")
PROMPTS = os.path.join(REPO, "tests", "golden", "prompts")
FLASH_NEXT_ROWS = 80604
BOS = 2


def _score():
    spec = importlib.util.spec_from_file_location("laguna_score", os.path.join(REPO, "tools", "toolcall", "score.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Renderer:
    def __init__(self, snapshot: str):
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer
        self.tmpl = AutoTokenizer.from_pretrained(snapshot)
        self.tok = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))

    def ids(self, messages, tools, enable_thinking, gen: bool) -> list[int]:
        text = self.tmpl.apply_chat_template(messages, tools=tools, enable_thinking=enable_thinking,
                                             add_generation_prompt=gen, tokenize=False)
        return self.tok.encode(text, add_special_tokens=False).ids

    def answered(self, messages, tools, enable_thinking, answer) -> tuple[list[int], int]:
        """(ids of messages + answer, the answer's first row) - the prompt's ids must be a prefix."""
        p = self.ids(messages, tools, enable_thinking, True)
        if answer is None:
            return p, len(p)
        full = self.ids(messages + [answer], tools, enable_thinking, False)
        if full[: len(p)] != p:
            raise SystemExit(f"laguna_prompts: the answered render does not extend the prompt's ids "
                             f"(first difference at {next(i for i, (x, y) in enumerate(zip(p, full)) if x != y)})")
        return full, len(p)


def sources(r: Renderer, answers: str | None) -> list[tuple[str, list[int], int]]:
    out = []
    score = _score() if answers else None
    for m in json.load(open(os.path.join(TOOLCALL, "manifest.json"), encoding="utf-8")):
        d = json.load(open(os.path.join(TOOLCALL, f"{m['name']}.json"), encoding="utf-8"))
        answer = None
        txt = os.path.join(answers, f"{d['name']}.bf16.txt") if answers else None
        if txt and os.path.exists(txt):
            kind, name, params, _ = score.parse_first_call(open(txt, encoding="utf-8").read())
            if kind == "call":
                answer = {"role": "assistant", "content": "",
                          "tool_calls": [{"type": "function", "function": {"name": name, "arguments": params}}]}
        if answers and answer is None:
            print(f"{d['name']}: no recorded call in {txt} - prompt only", flush=True)
        ids, af = r.answered(d["messages"], d["tools"], d["enable_thinking"], answer)
        out.append((d["name"], ids, af))
    a = json.load(open(os.path.join(PROMPTS, "q4exp_agentic.json"), encoding="utf-8"))
    msgs = list(a["messages"])
    while msgs[-1]["role"] != "assistant":
        msgs.pop()
    ids, af = r.answered(msgs[:-1], a["tools"], a["enable_thinking"], msgs[-1])
    out.append(("q4exp_agentic", ids, af))
    for name in ("code", "prose"):
        text = open(os.path.join(PROMPTS, f"{name}.txt"), encoding="utf-8").read().rstrip("\n")
        ids = [BOS] + r.tok.encode(text, add_special_tokens=False).ids
        out.append((name, ids, len(ids)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--answers", help="Qwen3.8's A4 reference outputs (<name>.bf16.txt)")
    a = ap.parse_args()
    r = Renderer(a.snapshot)
    srcs = sources(r, a.answers)
    os.makedirs(a.out, exist_ok=True)
    lines = []
    for name, ids, af in srcs:
        with open(os.path.join(a.out, f"{name}.ids"), "w", encoding="utf-8") as f:
            f.write(" ".join(map(str, ids)) + "\n")
        lines.append(f"{name}\t{name}.ids\t{af}")
        print(f"{name}: {len(ids)} rows, answer from {af} ({len(ids) - af} answer rows)")
    with open(os.path.join(a.out, "sources.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    total = sum(len(ids) for _, ids, _ in srcs)
    print(f"{len(srcs)} sources, {total} rows ({total / FLASH_NEXT_ROWS:.2f} x Flash-Next's {FLASH_NEXT_ROWS}), "
          f"{sum(len(ids) - af for _, ids, af in srcs)} answer rows; sample: {r.tok.decode(srcs[0][1][:3], skip_special_tokens=False)!r}")


if __name__ == "__main__":
    main()
