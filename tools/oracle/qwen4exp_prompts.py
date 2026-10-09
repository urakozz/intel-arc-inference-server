#!/usr/bin/env python3
"""Spec 21a Task 5: the Qwen3.8-Flash-Next golden prompts - short (QSA dense) and four past the QSA cut.

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/qwen4exp_prompts.py <tokenizer dir> [--out tests/golden/prompts]
                                                                       [--check]

<tokenizer dir> holds the ORIGINAL checkpoint's tokenizer.json, tokenizer_config.json and chat_template.jinja
(Qwen/Qwen3.8-Flash-Next's small files). Its tokenizer has Qwen3.8's vocabulary, merges and added tokens but
NOT its pre-tokenizer: the split regex adds \\p{M} (combining marks) to letter runs (docs/probe-qwen4exp-*.md),
so Qwen3.8's committed .ids are re-checked here, not assumed: each text prompt is re-encoded and compared,
and long32k.ids (no .txt) is decoded (the vocabularies are the same) and re-encoded.

    q4exp_short.ids    prose.ids + code.ids                  < 2052 positions: QSA selects everything (dense gate)
    q4exp_4k/8k/32k.ids the first 4096 / 8192 / 32768 ids of long32k.ids
    q4exp_agentic.ids  q4exp_agentic.json through the checkpoint's own template (apply_chat_template,
                       add_generation_prompt, enable_thinking=True, its tools) - an original session on this repo
Refuses a set where 4k / 8k / 32k / agentic are not all past position 2051 (spec 21a's Global Constraints).
--check: compare with the committed files instead of writing (exit 1 on a difference).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import json  # noqa: E402

REPO = os.path.dirname(os.path.dirname(_HERE))
PROMPTS = os.path.join(REPO, "tests", "golden", "prompts")
QSA_CUT = 2051


def read_ids(p):
    return [int(x) for x in open(p, encoding="utf-8").read().split()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tokenizer_dir")
    ap.add_argument("--out", default=PROMPTS)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    tok = Tokenizer.from_file(os.path.join(a.tokenizer_dir, "tokenizer.json"))

    def enc(text):
        return tok.encode(text, add_special_tokens=False).ids
    out: dict[str, list[int]] = {}
    for name in ("prose", "code", "cjk"):        # long.ids is ids concatenated (make_long_prompt.sh), not long.txt's
        text = open(os.path.join(PROMPTS, f"{name}.txt"), encoding="utf-8").read().rstrip("\n")
        old, new = read_ids(os.path.join(PROMPTS, f"{name}.ids")), enc(text)
        print(f"{name}.txt: re-encoded {len(new)} ids, Qwen3.8's {len(old)}: equal {new == old}")
        out[name] = new
    l32 = read_ids(os.path.join(PROMPTS, "long32k.ids"))
    text = tok.decode(l32, skip_special_tokens=False)
    re_ids = enc(text)
    n_eq = next((i for i, (x, y) in enumerate(zip(l32, re_ids)) if x != y), min(len(l32), len(re_ids)))
    print(f"long32k.ids: decoded and re-encoded {len(re_ids)} ids; equal to Qwen3.8's for the first {n_eq} of {len(l32)}")
    base = l32 if n_eq >= len(l32) - 8 else re_ids        # the last few ids may re-split at the cut
    sets = {"q4exp_short": out["prose"] + out["code"],
            "q4exp_4k": base[:4096], "q4exp_8k": base[:8192], "q4exp_32k": base[:32768]}
    sess = json.load(open(os.path.join(PROMPTS, "q4exp_agentic.json"), encoding="utf-8"))
    ht = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    r = ht.apply_chat_template(sess["messages"], tools=sess["tools"], add_generation_prompt=True,
                               enable_thinking=sess.get("enable_thinking", True), tokenize=True)
    sets["q4exp_agentic"] = list(r["input_ids"] if isinstance(r, dict) or hasattr(r, "keys") else r)
    for k, v in sets.items():
        print(f"{k}: {len(v)} ids")
    short = len(sets["q4exp_short"])
    if short > QSA_CUT:
        print(f"FAIL: q4exp_short has {short} ids, must stay <= {QSA_CUT} (QSA dense)")
        return 1
    bad = [k for k in ("q4exp_4k", "q4exp_8k", "q4exp_32k", "q4exp_agentic") if len(sets[k]) <= QSA_CUT]
    if bad:
        print(f"FAIL: {bad} do not reach past position {QSA_CUT}: the selection would never act")
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
