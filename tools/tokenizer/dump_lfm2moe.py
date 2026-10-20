#!/usr/bin/env python3
"""Reference renders and ids of LFM2.5-24B-A2B's tokenizer and chat template (spec 24e, plan 24e Task 1).

    dump_lfm2moe.py <LFM2.5-24B-A2B snapshot dir> [<tests/golden/prompts/lfm2moe_agentic.json>]

<snapshot dir> holds the checkpoint's small files (tokenizer.json, tokenizer_config.json, chat_template.jinja);
LiquidAI/LFM2.5-24B-A2B at 1769516a is the one the references were written from. Writes, under tests/tokenizer/:

  lfm2moe_template_<name>.txt   every case of lfm2moe_template_cases.json (plus `agentic`, the frozen coding session of
                                tests/golden/prompts/lfm2moe_agentic.json) through transformers' apply_chat_template
                                (add_generation_prompt, the case's kwargs as chat_template_kwargs) - lfm2moe_template_test
                                compares chat::Template's render byte for byte;
  lfm2moe_tokenizer.json        the facts lfm2moe_tokenizer_test checks: sha256s, the id count and the head rows the
                                tokenizer does not define, every added token's id and `special` flag, a varied text set
                                (code, CJK, digits, emoji, the tags inline, pythonic calls, empty: ids without and with
                                special tokens and both decodes), digests of corpus.txt's encode / decode in 10 chunks,
                                and the ids of every template case (tokenizer(text) and apply_chat_template(tokenize=True)).

The engine holds a request as nlohmann::json, whose objects are KEY-SORTED: HF's side renders from the same sorted
objects (dump_k2.py's rule), or the comparison would grade key order - the known limitation of spec 24 section 4.6.

Run in the oracle image with transformers 5.19.0 from its site directory (tools/oracle/qwen4exp_env.sh), small files
only; the Mac holds /tmp/b70-cpu-model.lock around it:
  docker run --rm --memory 8g --cpus 2 -e PYTHONPATH=/site -e HF_HUB_OFFLINE=1 -v "$PWD:/ws" -v <site>:/site:ro \
      -v <snapshot>:/lfm:ro -w /ws --entrypoint python3 agnes-ref-img:latest \
      tools/tokenizer/dump_lfm2moe.py /lfm tests/golden/prompts/lfm2moe_agentic.json
"""
import hashlib
import json
import os
import sys

import tokenizers
import transformers
from transformers import AutoTokenizer

snapshot = sys.argv[1]
agentic_path = sys.argv[2] if len(sys.argv) > 2 else None
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "tokenizer")

if transformers.__version__ != "5.19.0":
    raise SystemExit(f"transformers {transformers.__version__}: the references are 5.19.0's (qwen4exp_env.sh)")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


spec = json.load(open(os.path.join(here, "lfm2moe_template_cases.json"), encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(snapshot)
raw = tokenizers.Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))


def case_inputs(case):
    tools = case.get("tools_inline")
    if tools is None:
        tools = None if case["tools"] is None else [spec["tools"][n] for n in case["tools"]]
    msgs, tools = json.loads(json.dumps([case["messages"], tools], sort_keys=True))
    return msgs, tools


def render(case, tokenize=False):
    msgs, tools = case_inputs(case)
    return tok.apply_chat_template(msgs, tools=tools, tokenize=tokenize, add_generation_prompt=True, **case["kwargs"])


cases = list(spec["cases"])
if agentic_path:
    sess = json.load(open(agentic_path, encoding="utf-8"))
    # The session carries its tools inline (lfm2moe_template_test reads the same file for this case).
    cases.append({"name": "agentic", "note": "tests/golden/prompts/lfm2moe_agentic.json (24a): six tools, 14 messages",
                  "messages": sess["messages"], "tools": None, "tools_inline": sess["tools"], "kwargs": {}})

texts = {}
ids_of_case = {}
for case in cases:
    text = render(case)
    texts[case["name"]] = text
    name = f"lfm2moe_template_{case['name']}.txt"
    open(os.path.join(here, name), "w", encoding="utf-8", newline="").write(text)
    out = render(case, tokenize=True)
    hf_ids = list(out["input_ids"] if hasattr(out, "keys") else out)
    ids = raw.encode(text, add_special_tokens=False).ids
    assert hf_ids == ids == tok(text)["input_ids"] == tok.encode(text), case["name"]
    ids_of_case[case["name"]] = ids
    print(f"{name}: {len(text.encode('utf-8'))} bytes, {len(ids)} ids")

if agentic_path:
    golden = [int(x) for x in open(agentic_path.replace(".json", ".ids")).read().split()]
    print("agentic == 24a's golden ids:", ids_of_case["agentic"] == golden)

# --- the tokenizer -------------------------------------------------------------------------
tc = "<|tool_call_start|>"
te = "<|tool_call_end|>"
text_set = [
    "Hello, world!",
    "int main(int argc, char** argv) {\n    return argc > 1 ? 0 : 1;\n}\n",
    "def f(x):\n\treturn [i ** 2 for i in range(x)]  # squares\n",
    "Numbers: 1234567 3.14159 -42 0xB70 1,000,000 2026-10-20 12:30",
    "  leading spaces, trailing spaces   ",
    "tabs\tand\nnewlines\r\nand CRLF\r\n",
    "The quick brown fox jumps over the lazy dog.",
    "Größe und Straße: ÄÖÜ äöü ß ẞ é è à ç ñ",
    "Café vs Café (NFC vs NFD)",
    "emoji \U0001F600 \U0001F469\U0001F3FD‍\U0001F4BB \U0001F1E9\U0001F1EA",
    "Привет, мир! 今天 東京 こんにちは",
    "it's we've they'll I'm you'd don't SHE'S",
    "<|startoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>",
    "<think>reasoning</think>answer<|im_end|>",
    tc + "[read_file(path='/a.cc', limit=50)]" + te,
    tc + "[grep(pattern='def foo\\\\(', flags=[\"-i\", \"it's\"]), bash(command='ls\\n')]" + te,
    "<|tool_list_start|>[]<|tool_list_end|> <|pad|> <|reserved_5|> <|image|>",
    "text <think> inline </think> text " + tc + te,
    "a" * 300,
    "\n\n\n   \n\t\t  ",
    "",
]
added = sorted(raw.get_added_tokens_decoder().items())
cases_out = []
for t in text_set:
    ids = raw.encode(t, add_special_tokens=False).ids
    cases_out.append({"text": t, "ids": ids, "ids_special": raw.encode(t, add_special_tokens=True).ids,
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
        he = fnv1a64(b"".join(i.to_bytes(4, "little") for i in ids) + b"\xff\xff\xff\xff", he)
        hd = fnv1a64(raw.decode(ids, skip_special_tokens=False).encode("utf-8") + b"\x00", hd)
    chunks.append({"first": b, "lines": len(corpus[b:b + step]), "ids": n,
                   "encode_fnv1a64": f"{he:016x}", "decode_fnv1a64": f"{hd:016x}"})

vocab = raw.get_vocab(with_added_tokens=True)
defined = sorted(vocab.values())
head_rows = 128000   # config.json vocab_size: the head's rows
facts = {
    "comment": "tools/tokenizer/dump_lfm2moe.py; lfm2moe_tokenizer_test checks the engine's tokenizer against it",
    "tokenizer_json_sha256": sha256(os.path.join(snapshot, "tokenizer.json")),
    "chat_template_sha256": sha256(os.path.join(snapshot, "chat_template.jinja")),
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "vocab_size_with_added": raw.get_vocab_size(with_added_tokens=True),
    "len_tok": len(tok),
    "head_rows": head_rows,
    "undefined_from": min(i for i in range(head_rows) if raw.id_to_token(i) is None),   # rows [from, head_rows) have no token
    "bos_id": tok.bos_token_id,
    "eos_id": tok.eos_token_id,
    "pad_id": tok.pad_token_id,
    "added_tokens": {a.content: {"id": i, "special": bool(a.special)} for i, a in added},
    "cases": cases_out,
    "corpus_chunks": chunks,
    "template_ids": ids_of_case,
}
assert all(raw.id_to_token(i) is None for i in range(facts["undefined_from"], head_rows))
assert facts["undefined_from"] == facts["vocab_size_with_added"]
assert facts["vocab_size_with_added"] == len(defined) == defined[-1] + 1, "the defined ids are not 0..n-1"
open(os.path.join(here, "lfm2moe_tokenizer.json"), "w", encoding="utf-8", newline="\n").write(
    json.dumps(facts, ensure_ascii=False, indent=1) + "\n")
print(f"lfm2moe_tokenizer.json: {len(cases_out)} texts, corpus {len(corpus)} lines in {len(chunks)} chunks, "
      f"{facts['vocab_size_with_added']} ids, undefined from {facts['undefined_from']}, bos {facts['bos_id']}, "
      f"eos {facts['eos_id']}, {len(added)} added tokens")
print("transformers", transformers.__version__, "tokenizers", tokenizers.__version__)
