#!/usr/bin/env python3
"""Reference renders and ids of Inkling 2 Mini's tokenizer and chat template (spec 41f Task 1).

    dump_inkling2.py <inkling-snapshot-dir>

<inkling-snapshot-dir> holds the checkpoint's small files: tokenizer.json, tokenizer_config.json and
chat_template.jinja (thinkingmachines/Inkling-2-Mini at b85c3cfa; the three template files are vendored at
tests/tokenizer/inkling2/). Writes, under tests/tokenizer/:

  inkling2_template_<name>.txt  every case of inkling2_template_cases.json without "error" through transformers'
                                apply_chat_template (the case's kwargs as chat_template_kwargs) - template_inkling2_test
                                compares the engine's render; a case with "error" must make the template raise that
                                text (checked here too);
  inkling2_tokenizer.json       what inkling2_tokenizer_test checks: sha256s, the id count, the tag ids, a text set
                                (code, digits, emoji, the tags inline), whole vocabulary words (the tokenizer sets
                                ignore_merges: a word that is one vocabulary entry is that one id), digests of
                                corpus.txt's encode / decode in 10 chunks, and every render's ids
                                (apply_chat_template(tokenize=True)).

Messages and tools are rendered from KEY-SORTED objects: b70-serve holds a request in nlohmann::json, whose objects are
sorted. Run on the Linux machine with the interpreter that has transformers and tokenizers (no weights are read):
  $HOME/auto-round/.venv/bin/python tools/tokenizer/dump_inkling2.py <snapshot>
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


spec = json.load(open(os.path.join(here, "inkling2_template_cases.json"), encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(snapshot)
raw = tokenizers.Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))


def case_inputs(case):
    tools = None if case["tools"] is None else [spec["tools"][n] for n in case["tools"]]
    msgs, tools = json.loads(json.dumps([case["messages"], tools], sort_keys=True))
    return msgs, tools


def render(case, tokenize=False):
    msgs, tools = case_inputs(case)
    return tok.apply_chat_template(msgs, tools=tools, tokenize=tokenize, add_generation_prompt=True, **case["kwargs"])


renders = []
for case in spec["cases"]:
    if "error" in case:
        try:
            render(case)
        except Exception as e:   # the template's raise_exception
            assert case["error"] in str(e), (case["name"], str(e))
            print(f"{case['name']}: refused ({e})")
            continue
        raise SystemExit(f"{case['name']}: the template rendered; it should refuse with {case['error']!r}")
    text = render(case)
    name = f"inkling2_template_{case['name']}.txt"
    open(os.path.join(here, name), "w", encoding="utf-8", newline="").write(text)
    out = render(case, tokenize=True)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    assert list(ids) == raw.encode(text, add_special_tokens=False).ids, case["name"]
    renders.append({"case": case["name"], "ids": list(ids)})
    print(f"{name}: {len(text.encode('utf-8'))} bytes, {len(ids)} ids")

texts = [
    "Hello, world!",
    "int main(int argc, char** argv) {\n    return argc > 1 ? 0 : 1;\n}\n",
    "def f(x):\n\treturn [i ** 2 for i in range(x)]  # squares\n",
    "  leading spaces, trailing spaces   ",
    "tabs\tand\nnewlines\r\nand CRLF\r\n",
    "Zahlen: 1234567 3,14159 -42 0xB70 1.000.000 2026-11-26 12:30",
    "Die Größe der Straße beträgt 3,5 m. ÄÖÜ äöü ß é è à ç ñ",
    "emoji 😀 👩🏽‍💻 🇩🇪 🏳️‍🌈",
    "Привет, мир! 今天 東京 안녕하세요 مرحبا",
    "<|message_user|><|content_text|>Hi<|end_message|><|message_model|><|content_thinking|>ok<|end_message|>"
    "<|message_model|>bash<|content_invoke_tool_json|>{\"name\":\"bash\",\"args\":{\"command\":\"ls\"}}<|end_message|>"
    "<|content_model_end_sampling|>",
    "text <|content_text|> inline <|end_message|> text",
    "a" * 300,
    "\n\n\n   \n\t\t  ",
    "",
]
cases = []
for t in texts:
    ids = raw.encode(t, add_special_tokens=False).ids
    cases.append({"text": t, "ids": ids, "ids_special": raw.encode(t, add_special_tokens=True).ids,
                  "decoded": raw.decode(ids, skip_special_tokens=False),
                  "decoded_skip": raw.decode(ids, skip_special_tokens=True)})
    assert tok.encode(t, add_special_tokens=False) == ids, t

# ignore_merges: words that are whole vocabulary entries (byte-level: "Ġ" is a space), a spread of them in id order.
vocab = raw.get_vocab(with_added_tokens=False)
byte_space = "Ġ"
whole = sorted((i, s) for s, i in vocab.items() if len(s) >= 7 and s.isascii() and
               (s[1:] if s.startswith(byte_space) else s).isalpha())
step = max(1, len(whole) // 400)
words = []
for i, s in whole[::step]:
    text = " " + s[1:] if s.startswith(byte_space) else s
    ids = raw.encode(text, add_special_tokens=False).ids
    words.append({"text": text, "ids": ids})
single = sum(1 for w in words if len(w["ids"]) == 1)


def fnv1a64(data, h=0xcbf29ce484222325):
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


corpus = open(os.path.join(here, "corpus.txt"), encoding="utf-8", newline="").read().split("\n")
if corpus and corpus[-1] == "":
    corpus.pop()
chunks = []
cstep = (len(corpus) + 9) // 10
for b in range(0, len(corpus), cstep):
    he, hd, n = 0xcbf29ce484222325, 0xcbf29ce484222325, 0
    for line in corpus[b:b + cstep]:
        ids = raw.encode(line, add_special_tokens=False).ids
        n += len(ids)
        he = fnv1a64(b"".join(i.to_bytes(4, "little") for i in ids) + b"\xff\xff\xff\xff", he)
        hd = fnv1a64(raw.decode(ids, skip_special_tokens=False).encode("utf-8") + b"\x00", hd)
    chunks.append({"first": b, "lines": len(corpus[b:b + cstep]), "ids": n,
                   "encode_fnv1a64": f"{he:016x}", "decode_fnv1a64": f"{hd:016x}"})

tags = ["<|endoftext|>", "<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>",
        "<|content_text|>", "<|content_image|>", "<|content_model_end_sampling|>", "<|content_thinking|>",
        "<|end_message|>", "<|content_tool_error|>", "<|content_xml|>", "<|begin_of_text|>",
        "<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>"]
facts = {
    "comment": "tools/tokenizer/dump_inkling2.py; inkling2_tokenizer_test checks the engine's tokenizer against it",
    "tokenizer_json_sha256": sha256(os.path.join(snapshot, "tokenizer.json")),
    "chat_template_sha256": sha256(os.path.join(snapshot, "chat_template.jinja")),
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "vocab_size_with_added": raw.get_vocab_size(with_added_tokens=True),
    "eos_id": tok.eos_token_id,
    "tag_ids": {t: raw.token_to_id(t) for t in tags},
    "cases": cases,
    "words": words,
    "corpus_chunks": chunks,
    "renders": renders,
}
open(os.path.join(here, "inkling2_tokenizer.json"), "w", encoding="utf-8", newline="\n").write(
    json.dumps(facts, ensure_ascii=False, indent=1) + "\n")
print(f"inkling2_tokenizer.json: {len(cases)} cases, {len(words)} whole-vocabulary words ({single} one id), corpus "
      f"{len(corpus)} lines in {len(chunks)} chunks, {len(renders)} renders, {facts['vocab_size_with_added']} ids, "
      f"eos {facts['eos_id']}")
print("transformers", transformers.__version__, "tokenizers", tokenizers.__version__)
