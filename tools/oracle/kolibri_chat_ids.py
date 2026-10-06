#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The ids of a chat-formatted Kolibri-1 golden prompt (spec 20c Task 1 Step 6).

    kolibri_chat_ids.py <snapshot> <prompt.json>    -> ids, whitespace-separated, on stdout

<prompt.json> is {"messages": [...], "tools": [...], "reasoning_effort": "..."} (tests/golden/prompts/
de_chat.json). The ids are the checkpoint's own chat template through transformers:
apply_chat_template(messages, tools=..., add_generation_prompt=True, reasoning_effort=...), tokenized
without special tokens added (Kolibri has no BOS; the template writes every special token itself).
Run it in the oracle container (tools/box_validate/kolibri_oracle.sh does).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import json  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("prompt")
    a = ap.parse_args()
    with open(a.prompt, encoding="utf-8") as f:
        p = json.load(f)
    tok = AutoTokenizer.from_pretrained(a.snapshot)
    if tok.chat_template is None:
        raise SystemExit(f"{a.snapshot}: the tokenizer has no chat template")
    kw = {}
    if p.get("reasoning_effort"):
        kw["reasoning_effort"] = p["reasoning_effort"]
    text = tok.apply_chat_template(p["messages"], tools=p.get("tools"), add_generation_prompt=True,
                                   tokenize=False, **kw)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if tok.bos_token_id is not None and ids and ids[0] == tok.bos_token_id:
        raise SystemExit("the template rendered a BOS; Kolibri-1 has none (docs/probe-kolibri-2026-10-05.md)")
    print(f"{a.prompt}: {len(ids)} ids", file=sys.stderr)
    print(" ".join(str(i) for i in ids))


if __name__ == "__main__":
    main()
