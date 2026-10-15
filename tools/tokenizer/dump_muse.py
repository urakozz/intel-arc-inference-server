#!/usr/bin/env python3
"""Reference renders, ids and parses of Muse-Glimmer-30B's tokenizer, chat template and output format (spec 23e).

    dump_muse.py tokenizer|templates|parse <bf16-snapshot> <out.json>

<bf16-snapshot> is meta-models/Muse-Glimmer-30B @ a4e59da5 (the ORIGINAL's small files: tokenizer.json sha256
c9dbee66..., tokenizer_config.json with its `response_template`, chat_template.jinja sha256 cfc67e5f... - spec 23
decision 1: the GPTQ / -AutoRound repos ship an older 7167-byte template, 114f55eb...; generation_config.json).
Only small files are read. Every mode is transformers 5.19.0 (spec 21a's site beside the oracle image):

  tokenizer  tests/tokenizer/muse_tokenizer.json - the sha256s, the id count, the named added tokens and the
             drafter's mask id both ways, EOS / BOS / pad, 24 texts (prose, code, digits, emoji, CJK, combining
             marks, every special tag inline, ATEM markup, a tool output, empty): ids without and with special
             tokens (the post-processor ADDS BOS) and both decodes; FNV-1a digests of tests/tokenizer/corpus.txt's
             10240 lines in 10 chunks; the ids of every template case through apply_chat_template(tokenize=True)
             (the one-BOS rule: the server tokenises the render WITHOUT special tokens).
  templates  tests/tokenizer/muse_templates.json - the message lists (>= 20) and their renders through
             apply_chat_template(tokenize=False) with the case's kwargs (current_date fixed, so no render reads the
             clock) and add_generation_prompt; template_muse_test compares chat::Template's render byte for byte.
  parse      tests/server/muse_parse_cases.json - complete assistant turns (the assistant parts of renders, and
             model-like outputs) through tokenizer.parse_response(text, prefix=prompt, tools=...): the reference of
             toolcall_muse_test. Where our parser differs BY DECISION the case says so ("differs") and carries our
             value: spec 23 decision 2 (several to=self blocks are joined with "\\n"; transformers keeps the last),
             decision 15 (argument values typed by the tool's schema first; transformers JSON-decodes first and
             strips), and the ns.ns name collapse (spec 23 §6.5; transformers keeps the emitted name).

Every request object is KEY-SORTED before rendering (the server holds requests in nlohmann::json, whose objects
are sorted - dump_k2.py's rule), so the comparison never grades key order.

Run in the oracle image with the 5.19.0 site first on PYTHONPATH (small files only):
  docker run --rm --memory 8g --cpus 4 -v "$PWD:/ws" -v <oracle-out-q4exp/site>:/site:ro \\
      -v <hf-cache>/hub:/hub:ro -w /ws -e PYTHONPATH=/site \\
      --entrypoint python3 agnes-ref-img:latest tools/tokenizer/dump_muse.py templates \\
      /hub/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68 \\
      tests/tokenizer/muse_templates.json
(the whole hub directory: the snapshot's links reach blobs outside the model's own directory)
"""
import hashlib
import json
import os
import sys

import tokenizers
import transformers
from transformers import AutoTokenizer

DATE = "2026-10-15"
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "tokenizer")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sort(x):
    return json.loads(json.dumps(x, sort_keys=True))


# ---- the tools (OpenAI form) ----------------------------------------------------------------------------------
def fn(name, description, props, required=()):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": list(required)}}}


TOOLS = {
    "fs.read_file": fn("fs.read_file", "Read a file from the workspace.",
                       {"path": {"type": "string", "description": "Path relative to the root"},
                        "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path"]),
    "fs.write_file": fn("fs.write_file", "Write a file.",
                        {"path": {"type": "string"}, "content": {"type": "string"},
                         "append": {"type": "boolean"}}, ["path", "content"]),
    "shell.run": fn("shell.run", "Run a shell command and return its output.",
                    {"command": {"type": "string"}, "timeout": {"type": "number"},
                     "env": {"type": "object"}, "args": {"type": "array", "items": {"type": "string"}},
                     "cwd": {"type": ["string", "null"]}}, ["command"]),
    "read_file": fn("read_file", "Read a file (bare name).",
                    {"path": {"type": "string"}, "offset": {"type": "integer"}}, ["path"]),
    "bash": fn("bash", "Run bash.", {"command": {"type": "string"}, "verbose": {"type": "boolean"}}, ["command"]),
    "calendar.get": fn("calendar.get", "Get a calendar entry.", {"id": {"type": "string"}}, ["id"]),
    "typed.probe": fn("typed.probe", "Every schema type, for the value-typing cases.",
                      {"s": {"type": "string"}, "i": {"type": "integer"}, "n": {"type": "number"},
                       "b": {"type": "boolean"}, "o": {"type": "object"}, "a": {"type": "array"},
                       "sn": {"type": ["string", "null"]}}, []),
}


def tools_of(names):
    return None if names is None else [TOOLS[n] for n in names]


def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


ALL_VALUES = {"flag": True, "off": False, "none": None, "count": 42, "ratio": 0.5,
              "text": "line one\nline \"two\" with 'quotes' and <tags>", "items": [1, "two", None, True],
              "obj": {"k": "v", "n": [1, 2]}}

SYS = "You are a coding agent working in a git repository."
U1 = {"role": "user", "content": "Read src/main.cc and tell me what it does."}

# name, messages, tools, kwargs (current_date added unless the case says "no_date"), add_generation_prompt
TEMPLATE_CASES = [
    ("no_system", [U1], None, {}, True),
    ("system_plain", [{"role": "system", "content": SYS}, U1], None, {}, True),
    ("system_strength", [{"role": "system", "content": SYS + "\nReasoning strength: low."}, U1], None, {}, True),
    ("effort_title", [{"role": "system", "content": SYS + "\nReasoning effort: medium."}, U1], None, {}, True),
    ("effort_capital", [{"role": "system", "content": SYS + "\nReasoning Effort: medium."}, U1], None, {}, True),
    ("effort_lower", [{"role": "system", "content": "reasoning effort: medium. " + SYS}, U1], None, {}, True),
    ("effort_upper", [{"role": "system", "content": SYS + "\nREASONING EFFORT: MEDIUM"}, U1], None, {}, True),
    ("strength_low", [U1], None, {"reasoning_strength": "low"}, True),
    ("strength_medium", [U1], None, {"reasoning_strength": "medium"}, True),
    ("strength_high", [{"role": "system", "content": SYS}, U1], None, {"reasoning_strength": "high"}, True),
    ("strength_xhigh", [{"role": "system", "content": SYS}, U1], None, {"reasoning_strength": "xhigh"}, True),
    ("knowledge_cutoff", [U1], None, {"knowledge_cutoff": "2025-12-31"}, True),
    ("tools_dotted", [U1], ["fs.read_file", "fs.write_file", "shell.run"], {}, True),
    ("tools_bare_system", [{"role": "system", "content": SYS}, U1], ["read_file", "bash"], {}, True),
    ("tools_ns_descriptions", [U1], ["fs.read_file", "shell.run", "bash"],
     {"tool_namespace_descriptions": {"fs": "The workspace's files.", "shell": "A POSIX shell, \"sandboxed\"."}},
     True),
    ("call_history_one", [U1, {"role": "assistant", "content": None,
                               "tool_calls": [call("c1", "fs.read_file", {"path": "src/main.cc", "offset": 1})]},
                          {"role": "tool", "name": "fs.read_file", "tool_call_id": "c1",
                           "content": "int main() { return 0; }"},
                          {"role": "assistant", "content": "It returns 0."},
                          {"role": "user", "content": "Thanks. Now run the tests."}],
     ["fs.read_file", "shell.run"], {}, True),
    ("call_history_parallel", [U1, {"role": "assistant", "content": None,
                                    "reasoning_content": "Two things to do: read the file and run the probe.",
                                    "tool_calls": [call("c1", "fs.read_file", {"path": "src/main.cc"}),
                                                   call("c2", "shell.run", dict(ALL_VALUES, command="make -j4"))]},
                               {"role": "tool", "tool_call_id": "c1", "content": "int main() {}"},
                               {"role": "tool", "tool_call_id": "c2", "content": "ok\n"},
                               {"role": "assistant", "content": "Both done."}],
     ["fs.read_file", "shell.run"], {}, True),
    ("tool_result_by_name", [U1, {"role": "assistant", "content": "",
                                  "tool_calls": [call("x9", "bash", {"command": "ls -la", "verbose": False})]},
                             {"role": "tool", "name": "bash", "content": "total 0"}],
     ["bash"], {}, True),
    ("reasoning_answer", [U1, {"role": "assistant", "reasoning_content": "The user wants a summary.",
                               "content": "It prints hello."},
                          {"role": "user", "content": "And then?"}], None, {}, True),
    ("recipient_override", [U1, {"role": "assistant", "recipient": "fs", "content": "status"},
                            {"role": "assistant", "content": "Status checked."}], ["fs.read_file"], {}, True),
    ("end_turn_override", [U1, {"role": "assistant", "content": "Partial answer.", "end_turn": False},
                           {"role": "assistant", "content": "Rest of the answer.", "end_turn": True}], None, {}, True),
    ("consecutive_assistant", [U1, {"role": "assistant", "content": None,
                                    "tool_calls": [call("c1", "read_file", {"path": "a.txt"})]},
                               {"role": "assistant", "content": "I read it."}], ["read_file"], {}, True),
    ("no_generation_prompt", [U1, {"role": "assistant", "content": "Done."}], None, {}, False),
    ("list_content", [{"role": "user", "content": [{"type": "text", "text": "Part one. "},
                                                    {"type": "text", "text": "Part two."}]}], None, {}, True),
    ("unicode", [{"role": "user", "content": "Grüße aus Köln – 東京 😀 \u0301e"}], None, {}, True),
    ("all_values_call", [U1, {"role": "assistant", "content": None,
                              "tool_calls": [call("c7", "typed.probe", {"s": "007", "i": 7, "n": 1.25, "b": True,
                                                                         "o": {"a": 1}, "a": ["x"], "sn": None})]}],
     ["typed.probe"], {}, False),
]


def template_inputs(case):
    name, messages, tools, kwargs, gen = case
    kw = dict(kwargs)
    kw.setdefault("current_date", DATE)
    msgs, tls = sort([messages, tools_of(tools)])
    return name, msgs, tls, kw, gen


def render(tok, msgs, tools, kw, gen, tokenize=False):
    out = tok.apply_chat_template(msgs, tools=tools, tokenize=tokenize, add_generation_prompt=gen, **kw)
    if tokenize:
        return list(out["input_ids"] if hasattr(out, "keys") else out)
    return out


def mode_templates(tok, snapshot, out_path):
    cases = []
    for case in TEMPLATE_CASES:
        name, msgs, tools, kw, gen = template_inputs(case)
        cases.append({"name": name, "messages": msgs, "tools": tools, "kwargs": kw, "add_generation_prompt": gen,
                      "render": render(tok, msgs, tools, kw, gen)})
    # A tool call whose arguments are a JSON STRING: the template raises (T:28-30) - the server parses it first.
    bad = sort([[U1, {"role": "assistant", "content": None,
                      "tool_calls": [call("c1", "fs.read_file", "{\"path\": \"a\"}")]}], tools_of(["fs.read_file"])])
    try:
        render(tok, bad[0], bad[1], {"current_date": DATE}, True)
        raised = None
    except Exception as e:  # jinja2's TemplateError via raise_exception
        raised = str(e)
    assert raised and "mapping" in raised, raised
    facts = {"comment": "tools/tokenizer/dump_muse.py templates; template_muse_test renders each case through "
                        "chat::Template and compares byte for byte",
             "transformers": transformers.__version__,
             "chat_template_sha256": sha256(os.path.join(snapshot, "chat_template.jinja")),
             "bos_token": tok.bos_token, "eos_token": tok.eos_token, "current_date": DATE,
             "string_arguments_case": {"messages": bad[0], "tools": bad[1], "error": raised},
             "cases": cases}
    write(out_path, facts)
    print(f"{out_path}: {len(cases)} cases; string arguments raise: {raised[:80]}")


def fnv1a64(data, h=0xcbf29ce484222325):
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "int main(int argc, char** argv) {\n    return argc > 1 ? 0 : 1;\n}\n",
    "def f(x):\n\treturn [i ** 2 for i in range(x)]  # squares\n",
    "Numbers: 1234567 3.14159 -42 0xB70 1,000,000 2026-10-15 12:30",
    "  leading spaces, trailing spaces   ",
    "tabs\tand\nnewlines\r\nand CRLF\r\n",
    "emoji 😀 👩🏽‍💻 🇩🇪 🏳️‍🌈",
    "Привет, мир! 今天天气很好。東京タワー 서울",
    "नमस्ते दुनिया, यह एक परीक्षण है।",
    "สวัสดีครับ ภาษาไทยมีสระและวรรณยุกต์",
    "مَرْحَبًا بِالْعَالَمِ  שָׁלוֹם עוֹלָם",
    "Z̤͔ͧ̑̓ä͖̭̈̇lͮ̒ͫǧ̗͚̚o̙̔ͮ̇͐̇  ā b̲ c⃝ x͡y",
    "Grüße aus Köln: „Straße“ – Café vs Café (NFC vs NFD)",
    "<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.<|eot|>"
    "<|start|>user<|message|>Hi<|eot|><|start|>assistant to=self<|message|>Think.<|eom|>"
    "<|start|>assistant to=user<|message|>Hello!<|eot|><|end_of_text|>",
    "<|finetune_right_pad|><|patch|><|video|><|image|><|image_start|><|image_end|><|reserved_special_token_1818|>",
    " to=self<|message|>reasoning<|eom|><|start|>assistant to=fs.read_file<|message|>",
    "<atem:function_calls>\n<atem:invoke name=\"fs.read_file\">\n<atem:parameter name=\"path\">src/main.cc"
    "</atem:parameter>\n<atem:parameter name=\"offset\">10</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
    "<|start|>tool fs.read_file<|message|><tool_output name=\"fs.read_file\">\nint main() {}\n</tool_output><|eot|>",
    "# Valid recipients: \"self\", \"fs.*\", \"user\".",
    "{\"name\": \"fs\", \"description\": \"\"}\n// Function schemas",
    "a" * 300,
    "\n\n\n   \n\t\t  ",
    "",
]

NAMED = ["<|begin_of_text|>", "<|end_of_text|>", "<|eom|>", "<|eot|>", "<|finetune_right_pad|>", "<|start|>",
         "<|message|>", "<|image_start|>", "<|image_end|>", "<|vid_start|>", "<|vid_end|>",
         "<|vid_frame_separator|>", "<|image|>", "<|video|>", "<|patch|>", "<|reserved_special_token_1818|>"]


def mode_tokenizer(tok, snapshot, out_path):
    raw = tokenizers.Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
    cases = []
    for t in TEXTS:
        ids = raw.encode(t, add_special_tokens=False).ids
        assert tok.encode(t, add_special_tokens=False) == ids, t
        cases.append({"text": t, "ids": ids, "ids_special": raw.encode(t, add_special_tokens=True).ids,
                      "decoded": raw.decode(ids, skip_special_tokens=False),
                      "decoded_skip": raw.decode(ids, skip_special_tokens=True)})
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
    for case in TEMPLATE_CASES:
        name, msgs, tools, kw, gen = template_inputs(case)
        text = render(tok, msgs, tools, kw, gen)
        ids = render(tok, msgs, tools, kw, gen, tokenize=True)
        assert ids == raw.encode(text, add_special_tokens=False).ids, name
        assert ids.count(200000) == 1 and ids[0] == 200000, name   # one BOS (the template's)
        renders.append({"case": name, "ids": ids})
    gen_cfg = json.load(open(os.path.join(snapshot, "generation_config.json")))
    facts = {
        "comment": "tools/tokenizer/dump_muse.py tokenizer; muse_tokenizer_test checks the engine's tokenizer "
                   "against it",
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "tokenizer_json_sha256": sha256(os.path.join(snapshot, "tokenizer.json")),
        "chat_template_sha256": sha256(os.path.join(snapshot, "chat_template.jinja")),
        "vocab_size_with_added": raw.get_vocab_size(with_added_tokens=True),
        "len_tok": len(tok),
        "bos_id": tok.bos_token_id,
        "eos_id": tok.eos_token_id,
        "pad_id": tok.pad_token_id,
        "generation_eos": gen_cfg["eos_token_id"],
        "named_ids": {t: raw.token_to_id(t) for t in NAMED},
        "cases": cases,
        "corpus_chunks": chunks,
        "renders": renders,
    }
    assert facts["vocab_size_with_added"] == 202048, facts["vocab_size_with_added"]
    write(out_path, facts)
    print(f"{out_path}: {len(cases)} texts, corpus {len(corpus)} lines in {len(chunks)} chunks, "
          f"{len(renders)} renders' ids, {facts['vocab_size_with_added']} ids")
    print("transformers", transformers.__version__, "tokenizers", tokenizers.__version__)


# ---- parse: complete assistant turns ----------------------------------------------------------------------------
EOT = "<|eot|>"


def turn_of(tok, prefix_msgs, assistant_msgs, tools, kw):
    """The assistant part of a render: render(prefix + assistant, no generation prompt) after the prompt
    render(prefix, generation prompt) - exactly what the model would have written after `<|start|>assistant`."""
    msgs, tls = sort([prefix_msgs + assistant_msgs, tools_of(tools)])
    pmsgs = sort(prefix_msgs)
    k = dict(kw)
    k.setdefault("current_date", DATE)
    prompt = render(tok, pmsgs, tls, k, True)
    full = render(tok, msgs, tls, k, False)
    assert full.startswith(prompt), "the prompt is not a prefix of the full render"
    return prompt, full[len(prompt):]


def default_prompt(tok, tools):
    k = {"current_date": DATE}
    return render(tok, sort([U1]), sort(tools_of(tools)), k, True)


PARSE_RENDERED = [
    # name, assistant messages, tools
    ("answer_only", [{"role": "assistant", "content": "It prints hello."}], None),
    ("reasoning_answer", [{"role": "assistant", "reasoning_content": "The user wants a summary.",
                           "content": "It prints hello."}], None),
    ("reasoning_call", [{"role": "assistant", "content": None, "reasoning_content": "I should read the file first.",
                         "tool_calls": [call("c1", "fs.read_file", {"path": "src/main.cc", "offset": 10})]}],
     ["fs.read_file", "shell.run"]),
    ("parallel_calls", [{"role": "assistant", "content": None, "reasoning_content": "Two calls.",
                         "tool_calls": [call("c1", "fs.read_file", {"path": "src/main.cc"}),
                                        call("c2", "shell.run", {"command": "make -j4", "timeout": 30.5,
                                                                 "env": {"CC": "clang", "N": [1, 2]},
                                                                 "args": ["-k", "all"], "cwd": None})]}],
     ["fs.read_file", "shell.run"]),
    ("bare_name_call", [{"role": "assistant", "content": None,
                         "tool_calls": [call("c1", "read_file", {"path": "a.txt", "offset": 3})]}], ["read_file"]),
    ("call_then_answer", [{"role": "assistant", "content": None,
                           "tool_calls": [call("c1", "bash", {"command": "ls", "verbose": True})]},
                          {"role": "assistant", "content": "Listed."}], ["bash"]),
    ("multiline_value", [{"role": "assistant", "content": None,
                          "tool_calls": [call("c1", "fs.write_file", {"path": "x.py", "append": False,
                                                                      "content": "def f():\n    return \"<b>\"\n"})]}],
     ["fs.write_file"],
     # transformers strips the string value (allow_non_json's text); ours keeps it exactly (decision 15)
     {"differs": ["typing"], "arguments": [{"append": False, "content": "def f():\n    return \"<b>\"\n",
                                            "path": "x.py"}]}),
]

# Model-like outputs (the model writes the recipient after the prompt's `<|start|>assistant`). Each:
# name, output text (EOS included where the turn ends with one), tools, our differences: a dict with
# "differs" (the decisions) and the fields our parser yields there.
INV = "<atem:function_calls>\n<atem:invoke name=\"{name}\">\n{params}</atem:invoke>\n</atem:function_calls>"


def atem(name, params):
    return INV.format(name=name, params="".join(f"<atem:parameter name=\"{k}\">{v}</atem:parameter>\n"
                                                for k, v in params))


PARSE_HAND = [
    ("two_reasoning_blocks",
     " to=self<|message|>First thought.<|eom|><|start|>assistant to=self<|message|>Second thought.<|eom|>"
     "<|start|>assistant to=user<|message|>Answer." + EOT, None,
     {"differs": ["reasoning_join"], "reasoning": "First thought.\nSecond thought."}),
    ("reasoning_then_call_eom",
     " to=self<|message|>Plan: read it.<|eom|><|start|>assistant to=fs.read_file<|message|>" +
     atem("fs.read_file", [("path", "src/a.cc"), ("limit", "20")]) + EOT, ["fs.read_file"], {}),
    ("invoke_in_reasoning",
     " to=self<|message|>I could write " + atem("fs.read_file", [("path", "x")]) + " but no.<|eom|>"
     "<|start|>assistant to=user<|message|>No call needed." + EOT, ["fs.read_file"], {}),
    ("invoke_in_content",
     " to=user<|message|>Use this markup: " + atem("fs.read_file", [("path", "x")]) + EOT, ["fs.read_file"], {}),
    ("recipient_namespace_only",
     " to=fs<|message|>" + atem("fs.read_file", [("path", "b.txt")]) + EOT, ["fs.read_file"], {}),
    ("typing_string_schema",
     " to=typed.probe<|message|>" + atem("typed.probe", [("s", "007"), ("sn", "null"), ("i", "007"), ("n", "1.50"),
                                                          ("b", "true"), ("o", "{\"a\": [1]}"), ("a", "[1, \"x\"]")])
     + EOT, ["typed.probe"], {"differs": ["typing"], "arguments": [{"s": "007", "sn": None, "i": 7, "n": 1.5,
                                                                     "b": True, "o": {"a": [1]}, "a": [1, "x"]}]}),
    ("typing_spaces",
     " to=typed.probe<|message|>" + atem("typed.probe", [("s", " x "), ("i", " 12 "), ("b", " false ")]) + EOT,
     ["typed.probe"], {"differs": ["typing"], "arguments": [{"s": " x ", "i": 12, "b": False}]}),
    ("typing_true_string",
     " to=typed.probe<|message|>" + atem("typed.probe", [("s", "true"), ("sn", "\"quoted\"")]) + EOT,
     ["typed.probe"], {"differs": ["typing"], "arguments": [{"s": "true", "sn": "\"quoted\""}]}),
    ("typing_no_schema",
     " to=fs.read_file<|message|>" + atem("fs.read_file", [("path", "p"), ("extra", "{\"k\": 1}"), ("word", "hi")])
     + EOT, ["fs.read_file"], {}),
    ("doubled_name",
     " to=read_file.read_file<|message|>" + atem("read_file.read_file", [("path", "a.txt")]) + EOT, ["read_file"],
     {"differs": ["name_collapse"], "names": ["read_file"]}),
    ("unknown_leaf",
     " to=weather.get<|message|>" + atem("weather.get", [("id", "1")]) + EOT, ["calendar.get"], {}),
    ("bare_recipient_dotted_invoke",
     " to=read_file<|message|>" + atem("read_file", [("path", "c.txt")]) + EOT, ["read_file"], {}),
    ("two_invokes_one_channel",
     " to=fs.read_file<|message|><atem:function_calls>\n<atem:invoke name=\"fs.read_file\">\n<atem:parameter "
     "name=\"path\">a</atem:parameter>\n</atem:invoke>\n<atem:invoke name=\"fs.read_file\">\n<atem:parameter "
     "name=\"path\">b</atem:parameter>\n</atem:invoke>\n</atem:function_calls>" + EOT, ["fs.read_file"], {}),
    ("answer_whitespace",
     " to=user<|message|>\n  Indented answer with trailing space.  \n" + EOT, None, {}),
    ("unicode_answer",
     " to=self<|message|>Denke nach … 東京<|eom|><|start|>assistant to=user<|message|>Grüße 😀" + EOT, None, {}),
]


def tf_message(parsed):
    out = {"reasoning": parsed.get("reasoning_content", ""), "content": parsed.get("content", ""),
           "tool_calls": [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                          for c in parsed.get("tool_calls", [])]}
    return out


def mode_parse(tok, snapshot, out_path):
    cases = []
    for entry in PARSE_RENDERED:
        name, assistant, tools = entry[:3]
        prompt, text = turn_of(tok, [U1], assistant, tools, {})
        tls = sort(tools_of(tools))
        parsed = tok.parse_response(text, prefix=prompt, tools=tls)
        case = {"name": name, "source": "render", "prompt_tail": prompt[-40:], "text": text, "tools": tls,
                "transformers": tf_message(parsed), "messages": sort(assistant)}
        if len(entry) > 3:
            case["ours"] = entry[3]
        cases.append(case)
    for name, text, tools, ours in PARSE_HAND:
        tls = sort(tools_of(tools))
        prompt = default_prompt(tok, tools)
        parsed = tok.parse_response(text, prefix=prompt, tools=tls)
        case = {"name": name, "source": "model-like", "prompt_tail": prompt[-40:], "text": text, "tools": tls,
                "transformers": tf_message(parsed)}
        if ours:
            case["ours"] = ours
        cases.append(case)
    facts = {"comment": "tools/tokenizer/dump_muse.py parse; toolcall_muse_test holds server::parse_muse_output to "
                        "transformers' parse_response on these complete turns, except the listed differences "
                        "(spec 23 decisions 2 and 15, the ns.ns collapse of §6.5) where `ours` is the expected value",
             "transformers": transformers.__version__,
             "response_template": tok.response_template if hasattr(tok, "response_template") else None,
             "cases": cases}
    write(out_path, facts)
    print(f"{out_path}: {len(cases)} cases ({sum(1 for c in cases if 'ours' in c)} with a documented difference)")


def write(path, facts):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(facts, ensure_ascii=False, indent=1) + "\n")


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("tokenizer", "templates", "parse"):
        sys.exit("usage: dump_muse.py tokenizer|templates|parse <bf16-snapshot> <out.json>")
    mode, snapshot, out_path = sys.argv[1:]
    assert transformers.__version__ == "5.19.0", transformers.__version__
    assert sha256(os.path.join(snapshot, "chat_template.jinja")).startswith("cfc67e5f"), "not the original's template"
    tok = AutoTokenizer.from_pretrained(snapshot)
    {"tokenizer": mode_tokenizer, "templates": mode_templates, "parse": mode_parse}[mode](tok, snapshot, out_path)


if __name__ == "__main__":
    main()
