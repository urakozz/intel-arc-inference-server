#!/usr/bin/env python3
"""Reference renders for tests/tokenizer/template_test.

Run on the box: ~/auto-round/.venv/bin/python tools/tokenizer/dump_template.py <snapshot_dir>
"""

import json
import sys

import transformers
from transformers import AutoTokenizer


snapshot = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(snapshot)
messages = json.load(open("tests/tokenizer/template_messages.json", encoding="utf-8"))
tools = json.load(open("tests/tokenizer/template_tools.json", encoding="utf-8"))


def dump(name, **kwargs):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    open(f"tests/tokenizer/{name}", "w", encoding="utf-8", newline="").write(text)
    print(f"{name}: {len(text.encode('utf-8'))} bytes")


dump("template_think_on.txt", enable_thinking=True)
dump("template_think_off.txt", enable_thinking=False)
dump("template_tools.txt", tools=tools, enable_thinking=False)
print("transformers", transformers.__version__)
