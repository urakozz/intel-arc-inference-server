#!/usr/bin/env python3
"""Spec 23a Task 4: the Muse-Glimmer golden prompts - short (every window covers the prompt) and four past it.

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/muse_prompts.py <bf16 snapshot> [--qwen38-tokenizer FILE]
                                                                  [--out tests/golden/prompts] [--check]

<bf16 snapshot> holds the ORIGINAL's tokenizer.json and chat_template.jinja (meta-models/Muse-Glimmer-30B's small
files; the GPTQ repo's tokenizer.json is byte-identical, its template is older - decision 1).

    muse_short_prose.ids / muse_short_code.ids  prose.txt / code.txt by Muse's tokenizer, no special tokens (< 2048)
    muse_4k / 8k / 32k.ids  the first 4096 / 8192 / 32768 ids of the long text: long32k.ids is Qwen3.8's ids, decoded
                            with Qwen3.8's tokenizer (tok::default_tokenizer_json()'s file: $B70_TOKENIZER_JSON, else
                            the HF cache's urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ@84575a18) and re-encoded by
                            Muse's; Muse's BPE needs fewer ids for the same text, so long.txt is appended until 32768
                            ids exist (the char count used is printed)
    muse_agentic.ids        muse_agentic.json through the original's template (apply_chat_template, its tools,
                            reasoning_strength / current_date from the json, add_generation_prompt) - ONE BOS
Refuses a set where 4k / 8k / 32k / agentic are not all past 2048 positions, or a short one is not below it.
--check: compare with the committed files instead of writing (exit 1 on a difference).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402

REPO = os.path.dirname(os.path.dirname(_HERE))
PROMPTS = os.path.join(REPO, "tests", "golden", "prompts")
WINDOW = 2048
BOS = 200000
QWEN38_REV = "84575a18f209992ef96d819b31f924b489e3d55d"


def read_ids(p):
    return [int(x) for x in open(p, encoding="utf-8").read().split()]


def qwen38_tokenizer() -> str | None:
    if os.environ.get("B70_TOKENIZER_JSON"):
        return os.environ["B70_TOKENIZER_JSON"]
    for hub in (os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"), "/hf/hub"):
        p = os.path.join(hub, "models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ", "snapshots", QWEN38_REV,
                         "tokenizer.json")
        if os.path.exists(p):
            return p
    return None


def build(snapshot: str, q38: str) -> dict[str, list[int]]:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    tok = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))

    def enc(text):
        return tok.encode(text, add_special_tokens=False).ids
    sets = {}
    for name in ("prose", "code"):
        text = open(os.path.join(PROMPTS, f"{name}.txt"), encoding="utf-8").read().rstrip("\n")
        sets[f"muse_short_{name}"] = enc(text)
    qt = Tokenizer.from_file(q38)
    l32 = read_ids(os.path.join(PROMPTS, "long32k.ids"))
    text = qt.decode(l32, skip_special_tokens=False)
    extra = open(os.path.join(PROMPTS, "long.txt"), encoding="utf-8").read()
    used, n_extra = text, 0
    ids = enc(used)
    while len(ids) < 32768:
        used += "\n\n" + extra
        n_extra += 1
        ids = enc(used)
    print(f"long text: long32k.ids ({len(l32)} Qwen3.8 ids) decoded to {len(text)} chars"
          + (f" + long.txt x {n_extra}" if n_extra else "") + f" = {len(used)} chars -> {len(ids)} Muse ids")
    for n, k in ((4096, "muse_4k"), (8192, "muse_8k"), (32768, "muse_32k")):
        sets[k] = ids[:n]
    sess = json.load(open(os.path.join(PROMPTS, "muse_agentic.json"), encoding="utf-8"))
    ht = AutoTokenizer.from_pretrained(snapshot)
    with open(os.path.join(snapshot, "chat_template.jinja"), encoding="utf-8") as f:
        template = f.read()
    r = ht.apply_chat_template(sess["messages"], tools=sess["tools"], chat_template=template, add_generation_prompt=True,
                               tokenize=True, reasoning_strength=sess["reasoning_strength"],
                               current_date=sess["current_date"], return_dict=False)
    sets["muse_agentic"] = list(r)
    return sets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--qwen38-tokenizer", default=None)
    ap.add_argument("--out", default=PROMPTS)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    q38 = a.qwen38_tokenizer or qwen38_tokenizer()
    if not q38:
        print("FAIL: Qwen3.8's tokenizer.json is needed to decode long32k.ids (--qwen38-tokenizer, B70_TOKENIZER_JSON, "
              "or hf download urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ tokenizer.json --revision " + QWEN38_REV + ")")
        return 1
    sets = build(a.snapshot, q38)
    for k, v in sets.items():
        print(f"{k}: {len(v)} ids")
    print(f"muse_agentic: BOS count {sets['muse_agentic'].count(BOS)} (must be 1), first id {sets['muse_agentic'][0]}")
    if sets["muse_agentic"].count(BOS) != 1 or sets["muse_agentic"][0] != BOS:
        print("FAIL: the agentic render must hold exactly one BOS, first")
        return 1
    bad = [k for k in ("muse_short_prose", "muse_short_code") if len(sets[k]) >= WINDOW]
    bad += [k for k in ("muse_4k", "muse_8k", "muse_32k", "muse_agentic") if len(sets[k]) <= WINDOW]
    if bad:
        print(f"FAIL: {bad}: short sets must stay below {WINDOW} positions, long ones past it (the window and the ring)")
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
