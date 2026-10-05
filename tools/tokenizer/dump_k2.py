#!/usr/bin/env python3
"""Reference renders and ids of K2-Horizon's tokenizer and chat template (spec 18d).

    dump_k2.py <k2-snapshot-dir>

<k2-snapshot-dir> holds the served (int4) repo's small files: tokenizer.json,
tokenizer_config.json, chat_template.jinja (and config.json). Writes, under tests/tokenizer/:

  k2_template_<name>.txt  every case of k2_template_cases.json through transformers'
                          apply_chat_template (add_generation_prompt, the case's kwargs as
                          chat_template_kwargs) - template_test compares the engine's render;
  k2_tokenizer.json       the tokenizer facts k2_tokenizer_test checks: sha256s, vocab size,
                          BOS / EOS / tag ids, a varied text set (ids without and with special
                          tokens, decode), digests of corpus.txt's encode / decode in 10 chunks,
                          and the ids of three apply_chat_template(tokenize=True) renders.

Run in the oracle image (transformers 5.15, tokenizers 0.22.2 = the Rust crate's pin):
  docker run --rm --memory 28g -v "$PWD:/ws" -v <dir>:/k2:ro -w /ws --entrypoint python3 \
      agnes-ref-img:latest tools/tokenizer/dump_k2.py /k2
"""
import hashlib
import json
import os
import sys

import tokenizers
import transformers
from transformers import AutoTokenizer

snapshot = sys.argv[1]
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "tokenizer")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


spec = json.load(open(os.path.join(here, "k2_template_cases.json"), encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(snapshot)
raw = tokenizers.Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))


def case_inputs(case):
    tools = None if case["tools"] is None else [spec["tools"][n] for n in case["tools"]]
    # The engine holds a request as nlohmann::json, whose objects are KEY-SORTED: render
    # HF's side from the same sorted objects (as dump_agnes_template.py does), or the
    # comparison grades JSON key order rather than the template.
    msgs, tools = json.loads(json.dumps([case["messages"], tools], sort_keys=True))
    return msgs, tools


def render(case, tokenize=False):
    msgs, tools = case_inputs(case)
    return tok.apply_chat_template(msgs, tools=tools, tokenize=tokenize, add_generation_prompt=True,
                                   **case["kwargs"])


for case in spec["cases"]:
    text = render(case)
    name = f"k2_template_{case['name']}.txt"
    open(os.path.join(here, name), "w", encoding="utf-8", newline="").write(text)
    print(f"{name}: {len(text.encode('utf-8'))} bytes")

# --- the tokenizer -------------------------------------------------------------------------
texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces, trailing spaces   ",
    "tabs\tand\nnewlines\r\nand CRLF\r\n",
    "I'M SURE they'RE fine; we'll see, it's John's.",
    "Numbers 1234567 3.14159 -42 0xB70 1,000,000 ١٢٣ １２３",
    "int main(int argc, char** argv) {\n    return argc > 1 ? 0 : 1;\n}\n",
    "def f(x):\n\treturn [i ** 2 for i in range(x)]  # squares\n",
    "Größe, Straße, naïve café, Ærøskøbing",
    "Café vs Café (NFD vs NFC)",
    "Привет, мир! Как дела?",
    "今天模型读取新的提示。東京で静かな川を見た。한국어 토큰을 정확히 읽는다.",
    "مرحبا بالعالم",
    "emoji 😀 👩🏽‍💻 🇩🇪 🏳️‍🌈 👨‍👩‍👧‍👦",
    "zero​width‌joiner‍",
    "<|ifm|begin_of_text|>raw text after BOS",
    "<|ifm|im_start|>user\nHi<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think>\nOk.</ifm|think>Hello!<|ifm|im_end|>",
    "<ifm|tool_calls>\n<ifm|tool_call>read\n<ifm|arg_key>filePath</ifm|arg_key>\n<ifm|arg_value>/work/a.cc</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>",
    "<ifm|tool_call>{\"name\": \"read\", \"arguments\": {\"limit\": 40}}</ifm|tool_call>",
    "<ifm|arg_type>integer</ifm|arg_type><ifm|think_fast>\n</ifm|think_fast><ifm|think_faster>\n</ifm|think_faster>",
    "<|ifm|endoftext|>",
    "text <ifm|think> inline </ifm|think> text",
    "a" * 300,
    " ".join(["word"] * 50),
    "\n\n\n   \n\t\t  ",
    "",
]
tags = ["<|ifm|begin_of_text|>", "<|ifm|endoftext|>", "<|ifm|im_start|>", "<|ifm|im_end|>",
        "<ifm|think>", "</ifm|think>", "<ifm|think_fast>", "</ifm|think_fast>",
        "<ifm|think_faster>", "</ifm|think_faster>", "<ifm|tool_calls>", "</ifm|tool_calls>",
        "<ifm|tool_call>", "</ifm|tool_call>", "<ifm|arg_key>", "</ifm|arg_key>",
        "<ifm|arg_value>", "</ifm|arg_value>", "<ifm|arg_type>", "</ifm|arg_type>",
        "<ifm|tools>", "</ifm|tools>"]
cases = []
for t in texts:
    ids = raw.encode(t, add_special_tokens=False).ids
    cases.append({"text": t, "ids": ids, "ids_special": raw.encode(t, add_special_tokens=True).ids,
                  "decoded": raw.decode(ids, skip_special_tokens=False),
                  "decoded_skip": raw.decode(ids, skip_special_tokens=True)})
    assert tok.encode(t, add_special_tokens=False) == ids, t


def fnv1a64(data, h=0xcbf29ce484222325):
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


corpus = open(os.path.join(here, "corpus.txt"), encoding="utf-8", newline="").read().split("\n")
if corpus and corpus[-1] == "":
    corpus.pop()
chunks = []
step = (len(corpus) + 9) // 10
for b in range(0, len(corpus), step):
    he, hd, n = 0xcbf29ce484222325, 0xcbf29ce484222325, 0
    for line in corpus[b:b + step]:
        ids = raw.encode(line, add_special_tokens=False).ids
        n += len(ids)
        # Each line: its ids as little-endian u32, then 0xFFFFFFFF; the decode's UTF-8, then 0x00.
        he = fnv1a64(b"".join(i.to_bytes(4, "little") for i in ids) + b"\xff\xff\xff\xff", he)
        hd = fnv1a64(raw.decode(ids, skip_special_tokens=False).encode("utf-8") + b"\x00", hd)
    chunks.append({"first": b, "lines": len(corpus[b:b + step]), "ids": n,
                   "encode_fnv1a64": f"{he:016x}", "decode_fnv1a64": f"{hd:016x}"})

renders = []
for name in ("plain", "tool_call_history", "calls_xml_typed"):
    case = next(c for c in spec["cases"] if c["name"] == name)
    out = render(case, tokenize=True)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    renders.append({"case": name, "ids": list(ids)})

facts = {
    "comment": "tools/tokenizer/dump_k2.py; k2_tokenizer_test checks the engine's tokenizer against it",
    "tokenizer_json_sha256": sha256(os.path.join(snapshot, "tokenizer.json")),
    "chat_template_sha256": sha256(os.path.join(snapshot, "chat_template.jinja")),
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "vocab_size_with_added": raw.get_vocab_size(with_added_tokens=True),
    "bos_id": tok.bos_token_id,
    "eos_id": tok.eos_token_id,
    "tag_ids": {t: raw.token_to_id(t) for t in tags},
    "cases": cases,
    "corpus_chunks": chunks,
    "renders": renders,
}
open(os.path.join(here, "k2_tokenizer.json"), "w", encoding="utf-8", newline="\n").write(
    json.dumps(facts, ensure_ascii=False, indent=1) + "\n")
print(f"k2_tokenizer.json: {len(cases)} cases, corpus {len(corpus)} lines in {len(chunks)} chunks, "
      f"vocab {facts['vocab_size_with_added']}, bos {facts['bos_id']}, eos {facts['eos_id']}")
print("transformers", transformers.__version__, "tokenizers", tokenizers.__version__)
