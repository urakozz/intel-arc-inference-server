#!/usr/bin/env python3
"""Spec 24a Task 5: the LFM2.5-24B-A2B golden prompts - short, 4k, 32k and an agentic session.

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/lfm2moe_prompts.py <LFM2.5 snapshot> [--out tests/golden/prompts]
                                                                    [--check]

<snapshot> needs only the small files (tokenizer.json, tokenizer_config.json, chat_template.jinja, config.json).
Every raw prompt starts with LFM2's BOS (<|startoftext|>, 124894: the template writes it first, so the model was
trained with it at position 0); the text is tools/oracle/tokenize.py's rule (no template, trailing newlines
stripped, encoded without special tokens):

    lfm2moe_short.ids    BOS + prose.txt + code.txt (each encoded alone, concatenated - as Qwen3.8's short set)
    lfm2moe_4k.ids       the first 4096 ids of BOS + long.txt repeated (the text joined by blank lines until it is
    lfm2moe_32k.ids      the first 32768                                long enough - long32k.ids' recipe)
    lfm2moe_agentic.ids  lfm2moe_agentic.json (an original session on this repo: reasoning and pythonic tool calls
                         given as structured tool_calls) through the checkpoint's own template
                         (apply_chat_template, its tools, add_generation_prompt: the prompt ends in <think>)

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
BOS = 124894


def text(name: str) -> str:
    return open(os.path.join(PROMPTS, f"{name}.txt"), encoding="utf-8").read().rstrip("\n")


def build(snapshot: str) -> dict[str, list[int]]:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    tok = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
    if tok.token_to_id("<|startoftext|>") != BOS:
        raise SystemExit(f"{snapshot}: <|startoftext|> is {tok.token_to_id('<|startoftext|>')}, expected {BOS}")

    def enc(t):
        return tok.encode(t, add_special_tokens=False).ids
    sets = {"lfm2moe_short": [BOS] + enc(text("prose")) + enc(text("code"))}
    long_t = text("long")
    reps = [long_t]
    while len(enc("\n\n".join(reps))) < 32768:
        reps.append(long_t)
    long_ids = [BOS] + enc("\n\n".join(reps))
    sets["lfm2moe_4k"], sets["lfm2moe_32k"] = long_ids[:4096], long_ids[:32768]
    sess = json.load(open(os.path.join(PROMPTS, "lfm2moe_agentic.json"), encoding="utf-8"))
    ht = AutoTokenizer.from_pretrained(snapshot)
    r = ht.apply_chat_template(sess["messages"], tools=sess["tools"], add_generation_prompt=True, tokenize=True)
    sets["lfm2moe_agentic"] = [int(x) for x in (r["input_ids"] if hasattr(r, "keys") else r)]
    rendered = ht.apply_chat_template(sess["messages"], tools=sess["tools"], add_generation_prompt=True, tokenize=False)
    if sets["lfm2moe_agentic"] != ht(rendered, add_special_tokens=False)["input_ids"]:
        raise SystemExit("lfm2moe_agentic: apply_chat_template's ids != tokenizer(its text)")
    if not rendered.endswith("<|im_start|>assistant\n<think>") or "<|tool_call_start|>[" not in rendered:
        raise SystemExit("lfm2moe_agentic: the render lacks the generation prompt or a pythonic call block")
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
    if len(sets["lfm2moe_32k"]) < 32768:
        print(f"FAIL: lfm2moe_32k has {len(sets['lfm2moe_32k'])} ids, fewer than 32768")
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
