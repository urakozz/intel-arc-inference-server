#!/usr/bin/env python3
"""Reference renders of Agnes 3.0 Flash's chat template (spec 14, Review Focus 5).

    dump_agnes_template.py <agnes-snapshot-dir>

Renders every case of tests/tokenizer/agnes_template_cases.json through
transformers' own Jinja path - `apply_chat_template` when the snapshot has a
tokenizer, else `render_jinja_template` with the checkpoint's chat_template.jinja
and its tokenizer_config.json special tokens (the same call apply_chat_template
makes; it needs no tokenizer files) - into tests/tokenizer/agnes_template_<name>.txt.
tests/tokenizer/template_test.cc compares the engine's renderer against them.

Note (2026-10-03): Agnes's chat_template.jinja is byte-identical to Qwen3.8's
(sha256 c3cf9e34...), so the engine takes the same minja fallback for it.
"""
import hashlib
import json
import os
import sys

import transformers

snapshot = sys.argv[1]
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "tokenizer")
cases = json.load(open(os.path.join(here, "agnes_template_cases.json"), encoding="utf-8"))
source = open(os.path.join(snapshot, "chat_template.jinja"), encoding="utf-8").read()
cfg = json.load(open(os.path.join(snapshot, "tokenizer_config.json"), encoding="utf-8"))
print("chat_template.jinja sha256", hashlib.sha256(source.encode("utf-8")).hexdigest())

tokenizer = None
if os.path.exists(os.path.join(snapshot, "tokenizer.json")):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot)


def render(case):
    # The engine holds a request as nlohmann::json, whose objects are KEY-SORTED, so
    # the tool schemas reach its template sorted (tests/tokenizer/template_tools.json
    # is written sorted for the same reason). Render HF's side from the same sorted
    # objects, or the comparison grades JSON key order rather than the template.
    case = json.loads(json.dumps(case, sort_keys=True))
    kw = dict(add_generation_prompt=True, enable_thinking=case["think"])
    if tokenizer is not None:
        return tokenizer.apply_chat_template(case["messages"], tools=case["tools"], tokenize=False, **kw)
    from transformers.utils.chat_template_utils import render_jinja_template
    special = {k: cfg.get(k) for k in ("bos_token", "eos_token", "pad_token", "unk_token")}
    out, _ = render_jinja_template(conversations=[case["messages"]], tools=case["tools"],
                                   chat_template=source, **kw, **special)
    return out[0]


for case in cases:
    text = render(case)
    name = f"agnes_template_{case['name']}.txt"
    open(os.path.join(here, name), "w", encoding="utf-8", newline="").write(text)
    print(f"{name}: {len(text.encode('utf-8'))} bytes")
print("transformers", transformers.__version__, "via", "apply_chat_template" if tokenizer else "render_jinja_template")
