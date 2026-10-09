#!/usr/bin/env python3
"""Reference ids of Qwen3.8-Flash-Next's tokenizer and chat template (spec 21e Task 1).

    dump_qwen4exp.py <qwen4exp-snapshot-dir> [<qwen3.8 tokenizer.json>]

<qwen4exp-snapshot-dir> holds the ORIGINAL checkpoint's small files (Qwen/Qwen3.8-Flash-Next):
tokenizer.json (sha256 0997f410...), tokenizer_config.json, chat_template.jinja (sha256 c3cf9e34... =
Qwen3.8's / Agnes's) and generation_config.json - 21b's synthetic checkpoints copy them unchanged
(oracle-out-q4exp-synth/{ours,intel}/ckpt). 21a found the tokenizer NOT byte-identical to Qwen3.8's:
same vocabulary, merges and added tokens, but the pre-tokenizer's split regex adds \\p{M} (combining
marks) - so this writes a fixture rather than asserting Qwen3.8's file. Intel's checkpoint ships
Qwen3.8's tokenizer.json; the engine serves the original's (plan 21e Task 1).

Writes tests/tokenizer/qwen4exp_tokenizer.json, the facts qwen4exp_tokenizer_test checks:
  sha256s (tokenizer.json, chat_template.jinja), the id count, every added token's id (248044..248076),
  EOS ids, a 24-text set (prose, code, digits, emoji, CJK, and the scripts whose combining marks the
  \\p{M} split keeps with their letters - Devanagari, Thai, Hebrew points, Arabic harakat, Zalgo, a
  combining macron the NFC normalizer cannot compose): ids without and with special tokens and both
  decodes; with the optional Qwen3.8 tokenizer.json, which of those texts its file encodes differently
  (the evidence that the gate tells the two files apart); digests of tests/tokenizer/corpus.txt's
  encode / decode in 10 chunks; three chat renders' ids (agnes_template_cases.json's plain, thinking,
  tool_call_history - the template is Agnes's, byte for byte) through transformers' apply_chat_template.

Run in the oracle image (tokenizers 0.22.2 = the Rust crate's pin), small files only:
  docker run --rm --memory 8g --cpus 4 -v "$PWD:/ws" -v <snapshot>:/q4:ro -w /ws \\
      --entrypoint python3 agnes-ref-img:latest tools/tokenizer/dump_qwen4exp.py /q4 [/ws/<q38 tokenizer.json>]
"""
import hashlib
import json
import os
import sys

import tokenizers
import transformers
from transformers import AutoTokenizer

snapshot = sys.argv[1]
q38_path = sys.argv[2] if len(sys.argv) > 2 else None
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "tokenizer")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


tok = AutoTokenizer.from_pretrained(snapshot)
raw = tokenizers.Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
q38 = tokenizers.Tokenizer.from_file(q38_path) if q38_path else None
spec = json.load(open(os.path.join(here, "agnes_template_cases.json"), encoding="utf-8"))

texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "int main(int argc, char** argv) {\n    return argc > 1 ? 0 : 1;\n}\n",
    "def f(x):\n\treturn [i ** 2 for i in range(x)]  # squares\n",
    "Numbers: 1234567 3.14159 -42 0xB70 1,000,000 2026-10-09 12:30",
    "  leading spaces, trailing spaces   ",
    "tabs\tand\nnewlines\r\nand CRLF\r\n",
    "emoji 😀 👩🏽‍💻 🇩🇪 🏳️‍🌈",
    "Привет, мир! 今天天气很好。東京タワー 서울",
    "नमस्ते दुनिया, यह एक परीक्षण है।",
    "สวัสดีครับ ภาษาไทยมีสระและวรรณยุกต์",
    "שָׁלוֹם עוֹלָם",
    "مَرْحَبًا بِالْعَالَمِ",
    "Z̤͔ͧ̑̓ä͖̭̈̇lͮ̒ͫǧ̗͚̚o̙̔ͮ̇͐̇",
    "ā b̲ c⃝ x͡y",
    "Café vs Café (NFC vs NFD)",
    "Tiếng Việt có dấu: Hà Nội, Đà Nẵng",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nShort.\n</think>\n\nHi!<|im_end|>",
    "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.cc\n</parameter>\n</function>\n</tool_call>",
    "<tool_response>\nok\n</tool_response>",
    "<|endoftext|><|vision_start|><|image_pad|><|vision_end|><|fim_prefix|>",
    "a" * 300,
    "\n\n\n   \n\t\t  ",
    "",
]
cases = []
differ = []
for t in texts:
    ids = raw.encode(t, add_special_tokens=False).ids
    cases.append({"text": t, "ids": ids, "ids_special": raw.encode(t, add_special_tokens=True).ids,
                  "decoded": raw.decode(ids, skip_special_tokens=False),
                  "decoded_skip": raw.decode(ids, skip_special_tokens=True)})
    assert tok.encode(t, add_special_tokens=False) == ids, t
    if q38 is not None and q38.encode(t, add_special_tokens=False).ids != ids:
        differ.append(len(cases) - 1)


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


def render(case, tokenize):
    # The engine holds a request as nlohmann::json, whose objects are KEY-SORTED: render HF's side from
    # the same sorted objects (dump_k2.py's rule), or the comparison grades key order.
    msgs, tools = json.loads(json.dumps([case["messages"], case["tools"]], sort_keys=True))
    return tok.apply_chat_template(msgs, tools=tools, tokenize=tokenize, add_generation_prompt=True,
                                   enable_thinking=case["think"])


renders = []
for name in ("plain", "thinking", "tool_call_history"):
    case = next(c for c in spec if c["name"] == name)
    text = render(case, False)
    want = open(os.path.join(here, f"agnes_template_{name}.txt"), encoding="utf-8", newline="").read()
    out = render(case, True)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    renders.append({"case": name, "ids": list(ids), "text_is_agnes_render": text == want})

added = list(range(248044, 248077))
facts = {
    "comment": "tools/tokenizer/dump_qwen4exp.py; qwen4exp_tokenizer_test checks the engine's tokenizer against it",
    "tokenizer_json_sha256": sha256(os.path.join(snapshot, "tokenizer.json")),
    "qwen38_tokenizer_json_sha256": sha256(q38_path) if q38_path else None,
    "chat_template_sha256": sha256(os.path.join(snapshot, "chat_template.jinja")),
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "vocab_size_with_added": raw.get_vocab_size(with_added_tokens=True),
    "len_tok": len(tok),
    "bos_id": tok.bos_token_id,
    "eos_id": tok.eos_token_id,
    "generation_eos": json.load(open(os.path.join(snapshot, "generation_config.json")))["eos_token_id"],
    "tag_ids": {raw.id_to_token(i): i for i in added},
    "cases": cases,
    "cases_qwen38_differs": differ,
    "corpus_chunks": chunks,
    "renders": renders,
}
assert facts["vocab_size_with_added"] == 248077, facts["vocab_size_with_added"]
assert all(r["text_is_agnes_render"] for r in renders), "the template's renders are not Agnes's"
open(os.path.join(here, "qwen4exp_tokenizer.json"), "w", encoding="utf-8", newline="\n").write(
    json.dumps(facts, ensure_ascii=False, indent=1) + "\n")
print(f"qwen4exp_tokenizer.json: {len(cases)} cases ({len(differ)} encode differently with Qwen3.8's file: "
      f"{differ}), corpus {len(corpus)} lines in {len(chunks)} chunks, {facts['vocab_size_with_added']} ids, "
      f"eos {facts['eos_id']}, generation eos {facts['generation_eos']}")
print("transformers", transformers.__version__, "tokenizers", tokenizers.__version__)
