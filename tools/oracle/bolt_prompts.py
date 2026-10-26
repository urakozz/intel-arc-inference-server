#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 25a Task 5: the Nemotron 3.5 Bolt golden prompts - short, 4k, 32k and an agentic session.

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/bolt_prompts.py <Bolt snapshot> [--out tests/golden/prompts] [--check]

<snapshot> needs only the small files (tokenizer.json, tokenizer_config.json, chat_template.jinja, config.json).
Bolt's template writes no BOS and its post-processor adds none (spec 25 §7; docs/probe-bolt-2026-10-21.md §4), so
no raw prompt starts with one; the text is tools/oracle/tokenize.py's rule (no template, trailing newlines stripped,
encoded without special tokens):

    bolt_short.ids    prose.txt + code.txt (each encoded alone, concatenated - as the other models' short sets)
    bolt_short_prose.ids / bolt_short_code.ids   each alone (gate Q's prose / code texts)
    bolt_4k.ids       the first 4096 ids of long.txt repeated (joined by blank lines until it is long enough -
    bolt_32k.ids      the first 32768                         long32k.ids' recipe)
    bolt_agentic.ids  bolt_agentic.json (an original session on this repo: reasoning_content and structured
                      tool_calls, which the template renders as <think>...</think> and Qwen3-Coder XML calls)
                      through the checkpoint's own template (apply_chat_template, its tools, add_generation_prompt,
                      enable_thinking true: the prompt ends in `<|im_start|>assistant\\n<think>\\n`)

Prints each length; refuses a 32k set shorter than 32768. --check compares with the committed files (exit 1 on a
difference) instead of writing.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import json  # noqa: E402

REPO = os.path.dirname(os.path.dirname(_HERE))
PROMPTS = os.path.join(REPO, "tests", "golden", "prompts")
IM_START, IM_END, BOS = 10, 11, 1


def text(name: str) -> str:
    return open(os.path.join(PROMPTS, f"{name}.txt"), encoding="utf-8").read().rstrip("\n")


def build(snapshot: str) -> dict[str, list[int]]:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    tok = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
    if tok.token_to_id("<|im_start|>") != IM_START or tok.token_to_id("<|im_end|>") != IM_END:
        raise SystemExit(f"{snapshot}: <|im_start|> / <|im_end|> are not {IM_START} / {IM_END}")

    def enc(t):
        return tok.encode(t, add_special_tokens=False).ids
    sets = {"bolt_short": enc(text("prose")) + enc(text("code")), "bolt_short_prose": enc(text("prose")),
            "bolt_short_code": enc(text("code"))}
    long_t = text("long")
    reps = [long_t]
    while len(enc("\n\n".join(reps))) < 32768:
        reps.append(long_t)
    long_ids = enc("\n\n".join(reps))
    sets["bolt_4k"], sets["bolt_32k"] = long_ids[:4096], long_ids[:32768]
    sess = json.load(open(os.path.join(PROMPTS, "bolt_agentic.json"), encoding="utf-8"))
    ht = AutoTokenizer.from_pretrained(snapshot)
    kw = {"enable_thinking": True}
    r = ht.apply_chat_template(sess["messages"], tools=sess["tools"], add_generation_prompt=True, tokenize=True, **kw)
    sets["bolt_agentic"] = [int(x) for x in (r["input_ids"] if hasattr(r, "keys") else r)]
    rendered = ht.apply_chat_template(sess["messages"], tools=sess["tools"], add_generation_prompt=True, tokenize=False,
                                      **kw)
    if sets["bolt_agentic"] != ht(rendered, add_special_tokens=False)["input_ids"]:
        raise SystemExit("bolt_agentic: apply_chat_template's ids != tokenizer(its text)")
    if not rendered.endswith("<|im_start|>assistant\n<think>\n") or "<tool_call>\n<function=" not in rendered:
        raise SystemExit("bolt_agentic: the render lacks the generation prompt or a Qwen3-Coder XML call block")
    if BOS in sets["bolt_agentic"] or any(BOS in v for v in sets.values()):
        raise SystemExit("a BOS (1) in a Bolt prompt: the template writes none")
    return sets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--out", default=PROMPTS)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sets = build(a.snapshot)
    for k, v in sets.items():
        print(f"{k}: {len(v)} ids (first {v[:4]}, last {v[-3:]})")
    if len(sets["bolt_32k"]) < 32768:
        print(f"FAIL: bolt_32k has {len(sets['bolt_32k'])} ids, fewer than 32768")
        return 1
    rc = 0
    for k, v in sets.items():
        p = os.path.join(a.out, f"{k}.ids")
        line = " ".join(map(str, v)) + "\n"
        if a.check:
            same = os.path.exists(p) and open(p, encoding="utf-8").read() == line
            print(f"{p}: {'same' if same else 'DIFFERS'}")
            rc |= 0 if same else 1
        else:
            with open(p, "w", encoding="utf-8") as f:
                f.write(line)
    return rc


if __name__ == "__main__":
    sys.exit(main())
