#!/usr/bin/env python3
"""Reference renders and ids of Kolibri-1's tokenizer and chat template (spec 20e).

    dump_kolibri.py <kolibri-snapshot-dir>

<kolibri-snapshot-dir> holds the checkpoint's small files: tokenizer.json, tokenizer_config.json
(its "chat_template" is the template - the release ships no chat_template.jinja) and
generation_config.json. Aleph-Alpha/Kolibri-1-BF16 at 8c8b3489 is the one the references were
written from; 20b's int4 export copies these files unchanged. Writes, under tests/tokenizer/:

  kolibri_template_<name>.txt  every case of kolibri_template_cases.json through transformers'
                               apply_chat_template (add_generation_prompt, the case's kwargs as
                               chat_template_kwargs) - template_kolibri_test compares the engine's
                               render;
  kolibri_tokenizer.json       the facts kolibri_tokenizer_test checks: sha256s, the id count and
                               the ids the head has but the tokenizer does not define, EOS / tag ids,
                               a varied text set (German prose, code, digits, emoji, the tags
                               inline, empty: ids without and with special tokens - no BOS is ever
                               added - and both decodes), digests of corpus.txt's encode / decode in
                               10 chunks, and the ids of three apply_chat_template(tokenize=True)
                               renders.

Run in the oracle image (transformers 5.15, tokenizers 0.22.2 = the Rust crate's pin), small
files only:
  docker run --rm --memory 8g --cpus 2 -v "$PWD:/ws" -v <snapshot>:/kol:ro -w /ws \
      --entrypoint python3 agnes-ref-img:latest tools/tokenizer/dump_kolibri.py /kol
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


spec = json.load(open(os.path.join(here, "kolibri_template_cases.json"), encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(snapshot)
raw = tokenizers.Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
config = json.load(open(os.path.join(snapshot, "tokenizer_config.json"), encoding="utf-8"))


def case_inputs(case):
    tools = None if case["tools"] is None else [spec["tools"][n] for n in case["tools"]]
    # The engine holds a request as nlohmann::json, whose objects are KEY-SORTED: render HF's side
    # from the same sorted objects (dump_k2.py's rule), or the comparison grades key order.
    msgs, tools = json.loads(json.dumps([case["messages"], tools], sort_keys=True))
    return msgs, tools


def render(case, tokenize=False):
    msgs, tools = case_inputs(case)
    return tok.apply_chat_template(msgs, tools=tools, tokenize=tokenize, add_generation_prompt=True,
                                   **case["kwargs"])


for case in spec["cases"]:
    text = render(case)
    name = f"kolibri_template_{case['name']}.txt"
    open(os.path.join(here, name), "w", encoding="utf-8", newline="").write(text)
    print(f"{name}: {len(text.encode('utf-8'))} bytes")

# --- the tokenizer -------------------------------------------------------------------------
texts = [
    "Hello, world!",
    "Guten Morgen! Wie geht es Ihnen heute?",
    "Die Größe der Straße beträgt 3,5 m; der Fußweg ist schmaler.",
    "Er sagte: „Das ist nicht mein Problem.“ – und ging.",
    "Donaudampfschifffahrtsgesellschaftskapitänsmütze und Rindfleischetikettierungsüberwachungsaufgabe",
    "ÄÖÜ äöü ß ẞ é è à ç ñ",
    "Zahlen: 1234567 3,14159 -42 0xB70 1.000.000 2026-10-06 12:30 Uhr",
    "int main(int argc, char** argv) {\n    return argc > 1 ? 0 : 1;\n}\n",
    "def f(x):\n\treturn [i ** 2 for i in range(x)]  # Quadrate\n",
    "  leading spaces, trailing spaces   ",
    "tabs\tand\nnewlines\r\nand CRLF\r\n",
    "The quick brown fox jumps over the lazy dog.",
    "Café vs Café (NFD vs NFC)",
    "emoji 😀 👩🏽‍💻 🇩🇪 🏳️‍🌈",
    "Привет, мир! 今天 東京",
    "<|im_start|>user\nHallo<|im_end|>\n<|im_start|>assistant\n<think>\nKurz.\n</think>\n\nHallo!<|im_end|>",
    "<tool_call>\n{\"name\": \"grep\", \"arguments\": {\"pattern\": \"Schlüssel\"}}\n</tool_call>",
    "<tool_response>\nok\n</tool_response>",
    "<|endoftext|><|pad|><|text|><|chat|><|/role|>",
    "<|pii-email-address|> und <|pii-iban|> sind Platzhalter",
    "text <think> inline </think> text",
    "a" * 300,
    "\n\n\n   \n\t\t  ",
    "",
]
specials = list(range(127900, 127907)) + list(range(127913, 127923))
plain_tags = list(range(127907, 127913))
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
for name in ("plain", "tool_call_history", "german"):
    case = next(c for c in spec["cases"] if c["name"] == name)
    out = render(case, tokenize=True)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    renders.append({"case": name, "ids": list(ids)})

vocab = raw.get_vocab(with_added_tokens=True)
defined = sorted(vocab.values())
head_rows = 128000   # config.json vocab_size: the head's rows
facts = {
    "comment": "tools/tokenizer/dump_kolibri.py; kolibri_tokenizer_test checks the engine's tokenizer against it",
    "tokenizer_json_sha256": sha256(os.path.join(snapshot, "tokenizer.json")),
    "chat_template_sha256": hashlib.sha256(config["chat_template"].encode("utf-8")).hexdigest(),
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "vocab_size_with_added": raw.get_vocab_size(with_added_tokens=True),
    "len_tok": len(tok),
    "head_rows": head_rows,
    "undefined_ids": [i for i in range(head_rows) if raw.id_to_token(i) is None],
    "bos_id": tok.bos_token_id,
    "eos_id": tok.eos_token_id,
    "tag_ids": {raw.id_to_token(i): i for i in specials + plain_tags},
    "special_ids": specials,
    "plain_tag_ids": plain_tags,
    "cases": cases,
    "corpus_chunks": chunks,
    "renders": renders,
}
assert facts["vocab_size_with_added"] == len(defined) == defined[-1] + 1, "the defined ids are not 0..n-1"
open(os.path.join(here, "kolibri_tokenizer.json"), "w", encoding="utf-8", newline="\n").write(
    json.dumps(facts, ensure_ascii=False, indent=1) + "\n")
print(f"kolibri_tokenizer.json: {len(cases)} cases, corpus {len(corpus)} lines in {len(chunks)} chunks, "
      f"{facts['vocab_size_with_added']} ids, undefined {facts['undefined_ids']}, bos {facts['bos_id']}, "
      f"eos {facts['eos_id']}")
print("transformers", transformers.__version__, "tokenizers", tokenizers.__version__)
