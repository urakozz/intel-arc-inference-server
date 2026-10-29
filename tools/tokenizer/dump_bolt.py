#!/usr/bin/env python3
"""Reference renders, ids and parse cases of Nemotron 3.5 Bolt's tokenizer, chat template and output format (spec 25e).

    dump_bolt.py tokenizer|templates|parse <snapshot> <out.json> [<tests/golden/prompts/bolt_agentic.json>]

<snapshot> is our export urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ @ 8932aa8d (or NVIDIA's repos:
the template, tokenizer.json and ultra_v3_reasoning_parser.py are byte-identical in all of them - spec 25 section 7). Only small files
are read: tokenizer.json (sha256 d0432ad8...), tokenizer_config.json, chat_template.jinja (sha256 6521afc7..., 12,514 B),
generation_config.json, ultra_v3_reasoning_parser.py. Every mode is transformers 5.19.0 (spec 21a's site beside the oracle
image). The fixtures are GENERATED OUTPUTS (renders, ids, digests), not copies of the model's files (spec 25 Decision 20: the
operator allowed vendoring, generated fixtures stay the default): the tests read the template and tokenizer.json from the
snapshot at run time.

  tokenizer  tests/tokenizer/bolt_tokenizer.json - the sha256s, the id count, the added tokens 0-21 and 990 by name, id and
             `special` flag both ways (<think> .. </tool_response> are NOT special: a decode that skips special tokens keeps
             them), EOS / BOS / pad, 30 texts (prose, code, digits, emoji, CJK, combining marks, every tag inline, Qwen XML
             calls, empty): ids without and with special tokens (the post-processor adds NONE) and both decodes; FNV-1a
             digests of the decode of EACH of the 131072 single ids and of its re-encode; FNV-1a digests of
             tests/tokenizer/corpus.txt's 10240 lines in 10 chunks; the ids of every template case through
             apply_chat_template(tokenize=True) (no BOS anywhere: the template writes none).
  templates  tests/tokenizer/bolt_templates.json - the message lists (>= 50) and their renders through
             apply_chat_template(tokenize=False) with the case's kwargs and add_generation_prompt; template_bolt_test compares
             chat::Template's render byte for byte. The frozen agentic session (tests/golden/prompts/bolt_agentic.json, the
             4th argument) is one of the cases.
  parse      tests/server/bolt_parse_cases.json - model outputs (what follows the prompt's generation prompt) with their
             expected split into reasoning / content / tool calls, WRITTEN BY HAND from ultra_v3_reasoning_parser.py's rules
             (DeepSeek-R1's parser plus: enable_thinking false -> everything is content, a stray </think> included; one
             leading newline of the reasoning dropped when the output carries a <think> itself; force_nonempty_content turns
             reasoning with no content into content) and the qwen3_coder format (tools/toolcall/score.py `qwen-think`). Every
             expectation is cross-checked HERE against a restatement of the plugin's three rules (`ultra_v3`, below) and, for the
             first call, against score.py's reader; each case names the rule it tests. Where our server differs BY DECISION
             (spec 25e: streaming reasoning is verbatim and held while force_nonempty_content is set; a call cut by the end of
             generation is dropped) the case says so.

Every request object is KEY-SORTED before rendering (the server holds requests in nlohmann::json, whose objects are sorted -
dump_k2.py's rule), so the comparison never grades key order.

Run in the oracle image with the 5.19.0 site first on PYTHONPATH (small files only; the Mac holds /tmp/b70-cpu-model.lock):
  docker run --rm --memory 8g --cpus 4 -v "$PWD:/ws" -v <oracle-out-q4exp/site>:/site:ro \\
      -v <hf-cache>/hub:/hub:ro -w /ws -e PYTHONPATH=/site -e HF_HUB_OFFLINE=1 \\
      --entrypoint python3 agnes-ref-img:latest tools/tokenizer/dump_bolt.py templates \\
      /hub/models--urakozz--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ/snapshots/8932aa8d1844f91056c5b9433448075a842b9156 \\
      tests/tokenizer/bolt_templates.json tests/golden/prompts/bolt_agentic.json
(the whole hub directory: the snapshot's links reach blobs outside the model's own directory)
"""
import hashlib
import json
import os
import re
import sys

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "tokenizer")
tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "toolcall")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sort(x):
    return json.loads(json.dumps(x, sort_keys=True))


def write(path, facts):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(facts, ensure_ascii=False, indent=1) + "\n")


# ---- the tools (OpenAI form) ----------------------------------------------------------------------------------
def fn(name, description, props, required=(), extra_params=None, extra_fn=None, wrap=True):
    params = {"type": "object", "properties": props}
    if required is not None:
        params["required"] = list(required)
    params.update(extra_params or {})
    f = {"name": name, "description": description, "parameters": params}
    f.update(extra_fn or {})
    return {"type": "function", "function": f} if wrap else f


TOOLS = {
    "read": fn("read", "Read a file from the local filesystem.",
               {"filePath": {"type": "string", "description": "Absolute path to the file"},
                "offset": {"type": "integer", "description": "Line to start from"},
                "limit": {"type": "integer", "description": "Number of lines to read"}}, ["filePath"]),
    "shell": fn("shell", "  Run a shell command.\n  Returns its output.  ",
                {"command": {"type": "string", "description": "  The command line  "},
                 "timeout": {"type": "number", "description": "Seconds", "default": 30.5, "minimum": 0},
                 "env": {"type": "object", "description": "Environment",
                         "properties": {"CC": {"type": "string"}, "N": {"type": "array", "items": {"type": "integer"}}},
                         "additionalProperties": False},
                 "mode": {"type": "string", "enum": ["fast", "safe", None, 3, True]},
                 "args": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                 "cwd": {"type": ["string", "null"], "nullable": True, "x-note": "Grüße <b>&</b>"}},
                ["command", "mode"], extra_params={"additionalProperties": False, "$defs": {"d": {"type": "string"}}},
                extra_fn={"strict": True, "x-meta": {"k": [1, 2, {"z": None}]}}),
    "no_params": fn("ping", "No parameters at all.", {}, None),
    "no_required": fn("noop", "Properties without required.", {"a": {"type": "string"}}, None),
    "no_description": {"type": "function", "function": {"name": "bare", "parameters": {
        "type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}},
    "unwrapped": fn("unwrapped", "Given without the function wrapper.", {"q": {"type": "string"}}, ["q"], wrap=False),
    "props_not_mapping": {"type": "function", "function": {"name": "odd", "description": "properties is a list",
                                                         "parameters": {"type": "object", "properties": ["a", "b"]}}},
    "edit": fn("edit", "Replace text in a file.",
               {"filePath": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"},
                "replaceAll": {"type": "boolean"}}, ["filePath", "oldString", "newString"]),
    "typed": fn("typed.probe", "Every schema type.",
                {"s": {"type": "string"}, "i": {"type": "integer"}, "n": {"type": "number"}, "b": {"type": "boolean"},
                 "o": {"type": "object"}, "a": {"type": "array"}, "sn": {"type": ["string", "null"]}}, []),
}


def tools_of(names):
    return None if names is None else [TOOLS[n] for n in names]


def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


ALL_VALUES = {"flag": True, "off": False, "none": None, "count": 42, "ratio": 0.5, "tiny": 1e-05, "whole": 100.0,
              "text": "line one\nline \"two\" with 'quotes' and <tags>", "items": [1, "two", None, True, 2.5],
              "obj": {"k": "v", "n": [1, 2], "z": None, "t": True}, "uni": "Grüße 東京 😀", "empty": "", "neg": -7}

SYS = "You are a coding agent working in a git repository."
U1 = {"role": "user", "content": "Read src/main.cc and tell me what it does."}
R1 = "The user wants a summary.\nI should read the file first.\n"


def A(content=None, reasoning=None, calls=None):
    m = {"role": "assistant", "content": content}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    if calls is not None:
        m["tool_calls"] = calls
    return m


def T(cid, content, name=None):
    m = {"role": "tool", "tool_call_id": cid, "content": content}
    if name:
        m["name"] = name
    return m


# name, messages, tool names, kwargs, add_generation_prompt
TEMPLATE_CASES = [
    ("no_system", [U1], None, {}, True),
    ("system_plain", [{"role": "system", "content": SYS}, U1], None, {}, True),
    ("system_empty_string", [{"role": "system", "content": ""}, U1], None, {}, True),
    ("system_null_content", [{"role": "system", "content": None}, U1], None, {}, True),
    ("system_list_parts", [{"role": "system", "content": [{"type": "text", "text": "Part A. "},
                                                          {"type": "text", "text": "Part B."}]}, U1], None, {}, True),
    ("no_generation_prompt", [U1, A("Done.")], None, {}, False),
    ("generation_prompt_thinking_off", [{"role": "system", "content": SYS}, U1], None, {"enable_thinking": False}, True),
    ("thinking_off_no_system", [U1], None, {"enable_thinking": False}, True),
    ("thinking_on_explicit", [U1], None, {"enable_thinking": True}, True),
    ("medium_effort", [{"role": "system", "content": SYS}, U1], None, {"medium_effort": True}, True),
    ("medium_effort_two_users", [U1, A("It prints hello."), {"role": "user", "content": "And then?"}], None,
     {"medium_effort": True}, True),
    ("medium_effort_list_content", [{"role": "user", "content": [{"type": "text", "text": "a"},
                                                                  {"type": "text", "text": "b"}]}], None,
     {"medium_effort": True}, True),
    ("medium_effort_thinking_off", [U1], None, {"medium_effort": True, "enable_thinking": False}, True),
    ("medium_effort_last_is_tool", [U1, A(None, None, [call("c1", "read", {"filePath": "/a"})]), T("c1", "x")],
     ["read"], {"medium_effort": True}, True),
    ("truncate_false_no_history", [U1], None, {"truncate_history_thinking": False}, True),
    ("unknown_kwarg_ignored", [U1], None, {"some_other_flag": "x", "force_nonempty_content": True}, True),
    ("tools_one", [U1], ["read"], {}, True),
    ("tools_system", [{"role": "system", "content": SYS}, U1], ["read", "edit"], {}, True),
    ("tools_empty_system", [{"role": "system", "content": ""}, U1], ["read"], {}, True),
    ("tools_nested_enum_extra_keys", [{"role": "system", "content": SYS}, U1], ["shell"], {}, True),
    ("tools_no_params", [U1], ["no_params"], {}, True),
    ("tools_no_required", [U1], ["no_required"], {}, True),
    ("tools_no_description", [U1], ["no_description"], {}, True),
    ("tools_unwrapped", [U1], ["unwrapped"], {}, True),
    ("tools_properties_not_mapping", [U1], ["props_not_mapping"], {}, True),
    ("tools_many", [{"role": "system", "content": SYS}, U1], ["read", "shell", "no_params", "edit", "typed"], {}, True),
    ("tools_thinking_off", [{"role": "system", "content": SYS}, U1], ["read"], {"enable_thinking": False}, True),
    ("tools_empty_list", [U1], [], {}, True),
    ("call_history_one", [U1, A(None, None, [call("c1", "read", {"filePath": "src/main.cc", "offset": 1})]),
                          T("c1", "int main() { return 0; }"), A("It returns 0."),
                          {"role": "user", "content": "Thanks. Now run the tests."}], ["read", "shell"], {}, True),
    ("call_all_value_types", [U1, A(None, None, [call("c1", "shell", dict(ALL_VALUES, command="make -j4"))]),
                              T("c1", "ok")], ["shell"], {}, True),
    ("call_string_numbers_alike", [U1, A(None, None, [call("c1", "typed.probe", {
        "s": "007", "i": 7, "n": 1.25, "b": True, "o": {"a": 1}, "a": ["x"], "sn": None})]), T("c1", "ok")],
     ["typed"], {}, True),
    ("call_content_then_call", [U1, A("Let me look.", None, [call("c1", "read", {"filePath": "a"})]), T("c1", "r")],
     ["read"], {}, True),
    ("call_empty_content_string", [U1, A("", None, [call("c1", "read", {"filePath": "a"})]), T("c1", "r")],
     ["read"], {}, True),
    ("call_whitespace_content", [U1, A("  \n ", None, [call("c1", "read", {"filePath": "a"})]), T("c1", "r")],
     ["read"], {}, True),
    ("call_no_arguments", [U1, A(None, None, [call("c1", "ping", {})]), T("c1", "pong")], ["no_params"], {}, True),
    ("call_bare_function_dict", [U1, A(None, None, [{"name": "read", "arguments": {"filePath": "x"}}]), T("c1", "r")],
     ["read"], {}, True),
    ("call_parallel_two", [U1, A(None, None, [call("c1", "read", {"filePath": "a"}), call("c2", "read",
                                                                                          {"filePath": "b"})]),
                           T("c1", "A"), T("c2", "B"), A("Both read.")], ["read"], {}, True),
    ("call_three_results_grouped", [U1, A(None, R1, [call("c1", "read", {"filePath": "a"}),
                                                     call("c2", "read", {"filePath": "b"}),
                                                     call("c3", "read", {"filePath": "c"})]),
                                    T("c1", "A"), T("c2", "B"), T("c3", "C"), A("Three read.")], ["read"], {}, True),
    ("tool_result_list_parts", [U1, A(None, None, [call("c1", "read", {"filePath": "a"})]),
                                T("c1", [{"type": "text", "text": "one "}, {"type": "text", "text": "two"}])],
     ["read"], {}, True),
    ("tool_result_empty_and_null", [U1, A(None, None, [call("c1", "read", {"filePath": "a"}),
                                                       call("c2", "read", {"filePath": "b"})]),
                                    T("c1", ""), T("c2", None)], ["read"], {}, True),
    ("tool_result_multiline", [U1, A(None, None, [call("c1", "read", {"filePath": "a"})]),
                               T("c1", "<file>\n00001| #pragma once\n00002| int x;\n</file>\n")], ["read"], {}, True),
    ("reasoning_before_last_user_dropped", [U1, A("It prints hello.", R1), {"role": "user", "content": "And then?"}],
     None, {}, True),
    ("reasoning_before_last_user_kept", [U1, A("It prints hello.", R1), {"role": "user", "content": "And then?"}],
     None, {"truncate_history_thinking": False}, True),
    ("reasoning_after_last_user_kept", [U1, A("It prints hello.", R1)], None, {}, True),
    ("reasoning_call_before_last_user", [U1, A(None, R1, [call("c1", "read", {"filePath": "a"})]), T("c1", "r"),
                                         A("Done.", "Now answer."), {"role": "user", "content": "Next."}],
     ["read"], {}, True),
    ("reasoning_call_before_last_user_kept", [U1, A("Looking.", R1, [call("c1", "read", {"filePath": "a"})]), T("c1", "r"),
                                              A("Done.", "Now answer."), {"role": "user", "content": "Next."}],
     ["read"], {"truncate_history_thinking": False}, True),
    ("reasoning_call_after_last_user", [U1, A("Looking.", R1, [call("c1", "read", {"filePath": "a"})]), T("c1", "r"),
                                        A(None, "Second step.", [call("c2", "read", {"filePath": "b"})]), T("c2", "r2")],
     ["read"], {}, True),
    ("reasoning_whitespace_only", [U1, A("Answer.", "  \n"), {"role": "user", "content": "Again."}], None, {}, True),
    ("reasoning_empty_string", [U1, A("Answer.", ""), {"role": "user", "content": "Again."}], None, {}, True),
    ("reasoning_not_string", [U1, A("Answer.", 7), {"role": "user", "content": "Again."}], None, {}, True),
    ("content_with_think_tags_dropped", [U1, A("<think>\nhidden\n</think>Visible."), {"role": "user", "content": "x"}],
     None, {}, True),
    ("content_with_think_tags_kept", [U1, A("<think>\nhidden\n</think>Visible.")], None, {}, True),
    ("content_with_close_tag_only", [U1, A("rest</think>Visible."), {"role": "user", "content": "x"}], None, {}, True),
    ("content_with_open_tag_only", [U1, A("Visible<think>cut"), {"role": "user", "content": "x"}], None, {}, True),
    ("call_content_think_tags_dropped", [U1, A("<think>\nh\n</think>Seen.", None, [call("c1", "read", {"filePath": "a"})]),
                                         T("c1", "r"), {"role": "user", "content": "x"}], ["read"], {}, True),
    ("call_content_open_tag_dropped", [U1, A("Seen<think>cut", None, [call("c1", "read", {"filePath": "a"})]), T("c1", "r"),
                                       {"role": "user", "content": "x"}], ["read"], {}, True),
    ("empty_assistant_before_last_user", [U1, A(""), {"role": "user", "content": "x"}], None, {}, True),
    ("empty_assistant_after_last_user", [U1, A("")], None, {}, True),
    ("assistant_null_content_no_calls", [U1, A(None), {"role": "user", "content": "x"}], None, {}, True),
    ("assistant_list_parts", [U1, A([{"type": "text", "text": "Hel"}, {"type": "input_text", "text": "lo"}]),
                              {"role": "user", "content": "x"}], None, {}, True),
    ("list_content", [{"role": "user", "content": [{"type": "text", "text": "Part one. "},
                                                    {"type": "text", "text": "Part two."}]}], None, {}, True),
    ("list_content_odd_parts", [{"role": "user", "content": [{"type": "text"}, {"type": "text", "text": None},
                                                              {"type": "text", "text": "ok"},
                                                              {"type": "tool_use", "text": "dropped"}]}], None, {}, True),
    ("list_content_images", [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": "what is this?"},
        {"type": "image", "image": "y"}]}], None, {}, True),
    ("list_content_one_image", [{"role": "user", "content": [{"type": "text", "text": "see"},
                                                              {"type": "input_image", "image_url": "z"}]}], None, {}, True),
    ("list_content_image_tag_in_text", [{"role": "user", "content": [{"type": "text", "text": "an <image> tag"},
                                                                      {"type": "image_url"}]}], None, {}, True),
    ("list_content_video", [{"role": "user", "content": [{"type": "video", "video": "v"},
                                                          {"type": "text", "text": "t"}]}], None, {}, True),
    ("user_null_content", [{"role": "user", "content": None}], None, {}, True),
    ("user_empty_content", [{"role": "user", "content": ""}], None, {}, True),
    ("developer_role", [{"role": "system", "content": SYS}, {"role": "developer", "content": "Be brief."}, U1],
     None, {}, True),
    ("consecutive_users", [U1, {"role": "user", "content": "Also this."}], None, {}, True),
    ("consecutive_assistants", [U1, A("One."), A("Two."), {"role": "user", "content": "x"}], None, {}, True),
    ("assistant_first", [A("Hi, I am ready."), U1], None, {}, True),
    ("system_not_first", [U1, {"role": "system", "content": "late"}, A("ok")], None, {}, True),
    ("tool_first_after_user", [U1, T("c0", "stray result")], None, {}, True),
    ("tool_last_no_generation", [U1, A(None, None, [call("c1", "read", {"filePath": "a"})]), T("c1", "r")], ["read"], {},
     False),
    ("unicode", [{"role": "user", "content": "Grüße aus Köln – 東京 😀 \u0301e"}], None, {}, True),
    ("unicode_tool_args", [U1, A(None, None, [call("c1", "read", {"filePath": "/tmp/ünï/日本語.txt"})]), T("c1", "ok")],
     ["read"], {}, True),
    ("multiline_argument", [U1, A(None, None, [call("c1", "edit", {
        "filePath": "x.py", "oldString": "def f():\n    return 1\n", "newString": "def f():\n    return \"<b>\"\n",
        "replaceAll": False})]), T("c1", "Edit applied successfully.")], ["edit"], {}, True),
    ("medium_truncate_false_thinking_off_tools", [{"role": "system", "content": SYS}, U1,
                                                  A("A.", R1, [call("c1", "read", {"filePath": "a"})]), T("c1", "r"),
                                                  {"role": "user", "content": "B?"}], ["read"],
     {"medium_effort": True, "truncate_history_thinking": False, "enable_thinking": False}, True),
]


def template_inputs(case):
    name, messages, tools, kwargs, gen = case
    msgs, tls = sort([messages, tools_of(tools)])
    return name, msgs, tls, dict(kwargs), gen


def agentic_case(path):
    s = json.load(open(path, encoding="utf-8"))
    msgs, tls = sort([s["messages"], s["tools"]])
    return "agentic_session", msgs, tls, {}, True


def render(tok, msgs, tools, kw, gen, tokenize=False):
    out = tok.apply_chat_template(msgs, tools=tools, tokenize=tokenize, add_generation_prompt=gen, **kw)
    if tokenize:
        return list(out["input_ids"] if hasattr(out, "keys") else out)
    return out


def all_cases(agentic):
    cases = [template_inputs(c) for c in TEMPLATE_CASES]
    if agentic:
        cases.append(agentic_case(agentic))
    return cases


def mode_templates(tok, snapshot, out_path, agentic):
    cases = []
    for name, msgs, tools, kw, gen in all_cases(agentic):
        cases.append({"name": name, "messages": msgs, "tools": tools, "kwargs": kw, "add_generation_prompt": gen,
                      "render": render(tok, msgs, tools, kw, gen)})
    # Reviewing the corpus the way the plan asks: the always-there system block and the generation prompts.
    for c in cases:
        if c["add_generation_prompt"] and c["messages"] and c["messages"][0]["role"] != "system":
            assert c["render"].startswith("<|im_start|>system\n"), c["name"]
    facts = {"comment": "tools/tokenizer/dump_bolt.py templates; template_bolt_test renders each case through "
                        "chat::Template and compares byte for byte",
             "transformers": transformers.__version__,
             "chat_template_sha256": sha256(os.path.join(snapshot, "chat_template.jinja")),
             "bos_token": tok.bos_token, "eos_token": tok.eos_token,
             "cases": cases}
    write(out_path, facts)
    print(f"{out_path}: {len(cases)} cases, {sum(len(c['render']) for c in cases)} bytes of renders")


def fnv1a64(data, h=0xcbf29ce484222325):
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "int main(int argc, char** argv) {\n    return argc > 1 ? 0 : 1;\n}\n",
    "def f(x):\n\treturn [i ** 2 for i in range(x)]  # squares\n",
    "Numbers: 1234567 3.14159 -42 0xB70 1,000,000 2026-10-22 12:30",
    "  leading spaces, trailing spaces   ",
    "tabs\tand\nnewlines\r\nand CRLF\r\n",
    "emoji 😀 👩🏽‍💻 🇩🇪 🏳️‍🌈",
    "Привет, мир! 今天天气很好。東京タワー 서울",
    "नमस्ते दुनिया, यह एक परीक्षण है।",
    "สวัสดีครับ ภาษาไทยมีสระและวรรณยุกต์",
    "مَرْحَبًا بِالْعَالَمِ  שָׁלוֹם עוֹלָם",
    "Z̤͔ͧ̑̓ä͖̭̈̇lͮ̒ͫǧ̗͚̚o̙̔ͮ̇͐̇  ā b̲ c⃝ x͡y",
    "Grüße aus Köln: „Straße“ – Café vs Café (NFC vs NFD)",
    "<s><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
    "<|im_start|>assistant\n<think>\nThink.\n</think>Hello!<|im_end|></s>",
    "<think></think><think>\n</think>\n\n<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n"
    "</tool_call><tool_response>\nok\n</tool_response>",
    "<unk><image><img></img><quad>[INST][/INST][AVAILABLE_TOOLS][/AVAILABLE_TOOLS][TOOL_RESULTS][/TOOL_RESULTS][TOOL_CALLS]",
    "<SPECIAL_990><SPECIAL_991> <SPECIAL_999>x<SPECIAL_1000>",
    "<tool_call>\n<function=read>\n<parameter=filePath>\n/work/src/runtime/pipeline_plan.cc\n</parameter>\n"
    "<parameter=offset>\n10\n</parameter>\n</function>\n</tool_call>",
    "# Tools\n\nYou have access to the following functions:\n\n<tools>\n<function>\n<name>read</name>\n</function>\n</tools>",
    "{\"name\": \"read\", \"arguments\": {\"filePath\": \"a\"}}\n// Function schemas",
    "<think>\nA thought with a stray </think> inside and <think> twice.\n</think>",
    "reasoning effort: efficient}\n\n{reasoning effort: efficient}",
    "a" * 300,
    "\n\n\n   \n\t\t  ",
    "ÄÖÜ äöü ß ǅ ǈ ǋ ǲ ϒ ᾼ ᾈ İ ı",
    "https://example.com/a?b=c&d=e#frag user@example.org C:\\Users\\x\\file.txt /usr/local/bin/python3",
    "```python\nprint('hi')\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
    "\u00a0nbsp\u2003em space\u200bzero width\u2028line sep\ufeffBOM",
    "",
]

NAMED_IDS = list(range(0, 22)) + [990]


def mode_tokenizer(tok, snapshot, out_path, agentic):
    raw = tokenizers.Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
    file_json = json.load(open(os.path.join(snapshot, "tokenizer.json"), encoding="utf-8"))
    added = {a["content"]: a for a in file_json["added_tokens"]}
    named = {}
    for i in NAMED_IDS:
        tag = raw.id_to_token(i)
        assert raw.token_to_id(tag) == i, (i, tag)
        assert tok.convert_ids_to_tokens(i) == tag and tok.convert_tokens_to_ids(tag) == i, (i, tag)
        named[tag] = {"id": i, "special": added[tag]["special"]}
    assert named["<think>"]["id"] == 12 and named["</think>"]["id"] == 13 and named["<tool_call>"]["id"] == 14
    assert not named["<think>"]["special"] and named["<|im_end|>"]["special"] and named["</s>"]["special"]
    cases = []
    for t in TEXTS:
        ids = raw.encode(t, add_special_tokens=False).ids
        assert tok.encode(t, add_special_tokens=False) == ids, t
        assert tok.encode(t, add_special_tokens=True) == ids, "the post-processor adds something: " + t
        cases.append({"text": t, "ids": ids, "ids_special": raw.encode(t, add_special_tokens=True).ids,
                      "decoded": raw.decode(ids, skip_special_tokens=False),
                      "decoded_skip": raw.decode(ids, skip_special_tokens=True)})
    # Every id's single-id decode, and the ids that string encodes back to: the byte-level partial tokens decode to U+FFFD,
    # so the round trip is not the identity everywhere - the digests pin what it IS.
    n = raw.get_vocab_size(with_added_tokens=True)
    hd, he, exact = 0xcbf29ce484222325, 0xcbf29ce484222325, 0
    for i in range(n):
        s = raw.decode([i], skip_special_tokens=False)
        back = raw.encode(s, add_special_tokens=False).ids
        exact += back == [i]
        hd = fnv1a64(i.to_bytes(4, "little") + s.encode("utf-8") + b"\x00", hd)
        he = fnv1a64(b"".join(b.to_bytes(4, "little") for b in back) + b"\xff\xff\xff\xff", he)
    all_ids = {"count": n, "exact_round_trips": exact, "decode_fnv1a64": f"{hd:016x}", "reencode_fnv1a64": f"{he:016x}"}
    corpus = open(os.path.join(here, "corpus.txt"), encoding="utf-8", newline="").read().split("\n")
    if corpus and corpus[-1] == "":
        corpus.pop()
    chunks = []
    step = (len(corpus) + 9) // 10
    for b in range(0, len(corpus), step):
        h1, h2, cnt = 0xcbf29ce484222325, 0xcbf29ce484222325, 0
        for line in corpus[b:b + step]:
            ids = raw.encode(line, add_special_tokens=False).ids
            cnt += len(ids)
            # Each line: its ids as little-endian u32, then 0xFFFFFFFF; the decode's UTF-8, then 0x00.
            h1 = fnv1a64(b"".join(i.to_bytes(4, "little") for i in ids) + b"\xff\xff\xff\xff", h1)
            h2 = fnv1a64(raw.decode(ids, skip_special_tokens=False).encode("utf-8") + b"\x00", h2)
        chunks.append({"first": b, "lines": len(corpus[b:b + step]), "ids": cnt,
                       "encode_fnv1a64": f"{h1:016x}", "decode_fnv1a64": f"{h2:016x}"})
    renders = []
    for name, msgs, tools, kw, gen in all_cases(agentic):
        text = render(tok, msgs, tools, kw, gen)
        ids = render(tok, msgs, tools, kw, gen, tokenize=True)
        assert ids == raw.encode(text, add_special_tokens=False).ids, name
        assert 1 not in ids, name   # no BOS: the template writes none, the post-processor adds none
        renders.append({"case": name, "ids": ids})
    gen_cfg = json.load(open(os.path.join(snapshot, "generation_config.json")))
    facts = {
        "comment": "tools/tokenizer/dump_bolt.py tokenizer; bolt_tokenizer_test checks the engine's tokenizer against it",
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "tokenizer_json_sha256": sha256(os.path.join(snapshot, "tokenizer.json")),
        "chat_template_sha256": sha256(os.path.join(snapshot, "chat_template.jinja")),
        "vocab_size_with_added": n,
        "len_tok": len(tok),
        "bos_id": tok.bos_token_id,
        "eos_id": tok.eos_token_id,
        "pad_id": tok.pad_token_id,
        "generation_eos": gen_cfg["eos_token_id"],
        "added_token_count": len(file_json["added_tokens"]),
        "named": named,
        "all_ids": all_ids,
        "cases": cases,
        "corpus_chunks": chunks,
        "renders": renders,
    }
    assert n == 131072 and len(tok) == 131072, (n, len(tok))
    write(out_path, facts)
    print(f"{out_path}: {len(cases)} texts, {n} single-id round trips ({exact} exact), corpus {len(corpus)} lines in "
          f"{len(chunks)} chunks, {len(renders)} renders' ids")
    print("transformers", transformers.__version__, "tokenizers", tokenizers.__version__)


# ---- parse ------------------------------------------------------------------------------------------------------------
def ultra_v3(model_output, enable_thinking_false=False, force_nonempty=False):
    """A restatement of ultra_v3_reasoning_parser.py (+ basic_parsers.BaseThinkingReasoningParser.extract_reasoning, the
    version the operator's vLLM checkout holds): (reasoning, content), None for absent. Not ours: the cross-check of the hand
    written expectations below."""
    start, end = "<think>", "</think>"
    if enable_thinking_false:
        return None, model_output or None
    parts = model_output.partition(start)
    out = parts[2] if parts[1] else parts[0]
    if end not in out:
        reasoning, content = out, None
    else:
        reasoning, _, content = out.partition(end)
        content = content or None
    if reasoning and reasoning.startswith("\n") and start in model_output:
        reasoning = reasoning[1:]
    if content is None and force_nonempty:
        reasoning, content = None, reasoning
    return reasoning, content


def xml_call(name, params):
    return "<tool_call>\n<function=" + name + ">\n" + "".join(
        "<parameter=" + k + ">\n" + v + "\n</parameter>\n" for k, v in params) + "</function>\n</tool_call>"


# Each case: name, rule (what it tests), text (what the model writes after the prompt), thinking (the prompt ends in
# "<think>\n"), force (force_nonempty_content), tools (names, schema-first typing) and `expect` - what the SERVER yields:
# reasoning, content, tool_calls [{"name", "arguments"}]. `differs` (a note) marks where the server deliberately differs from
# the reference reading (ultra_v3 for the split, score.py's `qwen-think` for the first call); the cross-check below skips what
# the case's `skip` list names (a call's surrounding whitespace, the raw-string reader on typed values ...).
def case(name, rule, text, thinking=True, force=False, tools=None, calls=None, reasoning="", content="", differs=None,
         skip=()):
    c = {"name": name, "rule": rule, "text": text, "thinking": thinking, "force_nonempty_content": force,
         "tools": tools_of(tools), "expect": {"reasoning": reasoning, "content": content, "tool_calls": calls or []}}
    if differs:
        c["differs"] = differs
    c["skip"] = list(skip)
    return c


CALL_READ = xml_call("read", [("filePath", "/work/a.cc"), ("offset", "10")])
READ_A = {"name": "read", "arguments": {"filePath": "/work/a.cc", "offset": 10}}
PARSE_CASES = [
    case("answer_after_reasoning", "thinking on: the output opens inside reasoning up to </think>; the rest is content",
         "The user wants a summary.\n</think>It prints hello.", reasoning="The user wants a summary.\n",
         content="It prints hello."),
    case("reasoning_kept_verbatim", "ultra_v3 does not strip the reasoning: its whitespace is the model's (the template "
         "re-adds <think>\\n ... </think> around it)", "  A thought  \n\n</think>Answer.", reasoning="  A thought  \n\n",
         content="Answer."),
    case("content_kept_after_close", "content is everything after </think>, its leading whitespace included "
         "(DeepSeek-R1's partition)", "r</think>\n\nAnswer.\n", reasoning="r", content="\n\nAnswer.\n"),
    case("empty_reasoning", "an immediate </think>: empty reasoning, content follows", "</think>Answer.", reasoning="",
         content="Answer."),
    case("no_close_is_reasoning", "no </think>: everything is reasoning, content null (force_nonempty_content off)",
         "Still thinking when max_tokens hit.", reasoning="Still thinking when max_tokens hit."),
    case("no_close_forced_content", "force_nonempty_content: reasoning that never closed becomes content, reasoning null",
         "Still thinking when max_tokens hit.", force=True, content="Still thinking when max_tokens hit."),
    case("close_empty_forced", "force_nonempty_content: </think> then nothing -> final content is None -> the reasoning "
         "becomes content", "Thought only.</think>", force=True, content="Thought only."),
    case("close_empty_not_forced", "</think> then nothing, force off: reasoning stays, content null",
         "Thought only.</think>", reasoning="Thought only."),
    case("forced_but_content_present", "force_nonempty_content with real content changes nothing",
         "T</think>C", force=True, reasoning="T", content="C"),
    case("forced_close_whitespace_content", "force_nonempty_content: content that is only whitespace is NOT None "
         "(`content or None`): the reasoning stays", "T</think>\n", force=True, reasoning="T", content="\n"),
    case("forced_empty_output", "force_nonempty_content on an empty output: nothing at all", "", force=True),
    case("thinking_off_everything_content", "enable_thinking false: the prompt ends <think></think>; everything is content",
         "Just an answer.", thinking=False, content="Just an answer."),
    case("thinking_off_stray_close", "enable_thinking false: a stray </think> the model echoes is content",
         "Answer</think> tail", thinking=False, content="Answer</think> tail"),
    case("thinking_off_stray_open_close", "enable_thinking false: even a whole <think>..</think> is content",
         "<think>x</think>Answer", thinking=False, content="<think>x</think>Answer"),
    case("thinking_off_force_ignored", "enable_thinking false wins over force_nonempty_content (ultra_v3 returns first)",
         "Answer", thinking=False, force=True, content="Answer"),
    case("second_close_is_content", "only the FIRST </think> splits; a second one is content", "r</think>a</think>b",
         reasoning="r", content="a</think>b"),
    case("call_inside_reasoning_is_reasoning", "a <tool_call> inside reasoning is reasoning text, never a call",
         "I could " + CALL_READ + " but let me think.\n</think>No call.", tools=["read"],
         reasoning="I could " + CALL_READ + " but let me think.\n", content="No call.", skip=["scorer"]),
    case("call_after_reasoning", "a call after </think>: the Qwen XML parser's", "Read it.\n</think>" + CALL_READ,
         tools=["read"], reasoning="Read it.\n", calls=[READ_A], skip=["content"]),
    case("call_right_after_close_with_newlines", "whitespace between </think> and the call is dropped",
         "Read it.\n</think>\n\n" + CALL_READ, tools=["read"], reasoning="Read it.\n", calls=[READ_A], skip=["content"]),
    case("content_then_call", "content before the call is content, verbatim (its trailing whitespace too, as vLLM's "
         "qwen3_coder returns the text before <tool_call>)", "R</think>Let me look.\n\n" + CALL_READ, tools=["read"],
         reasoning="R", content="Let me look.\n\n", calls=[READ_A], skip=["content"]),
    case("two_calls", "two calls, in order", "R</think>" + CALL_READ + "\n" + xml_call("read", [("filePath", "/b")]),
         tools=["read"], reasoning="R", calls=[READ_A, {"name": "read", "arguments": {"filePath": "/b"}}],
         skip=["content"]),
    case("schema_first_typing", "schema-first typing: a string-typed parameter keeps its text, the others JSON-decode",
         "R</think>" + xml_call("typed.probe", [("s", "007"), ("sn", "null"), ("i", "42"), ("n", "1.50"), ("b", "true"),
                                                ("o", "{\"a\": [1]}"), ("a", "[1, \"x\"]")]), tools=["typed"],
         reasoning="R", calls=[{"name": "typed.probe", "arguments": {
             "s": "007", "sn": "null", "i": 42, "n": 1.5, "b": True, "o": {"a": [1]}, "a": [1, "x"]}}],
         skip=["content", "scorer"], differs="a string|null parameter written null stays the text 'null' (the Qwen parser's "
         "schema-first rule, shared with Qwen3.8; vLLM's qwen3_coder returns None); the scorer reads raw strings"),
    case("no_schema_json_else_text", "no schema for the key: JSON-decoded when it parses, else the text",
         "R</think>" + xml_call("mystery", [("a", "{\"k\": 1}"), ("b", "hello"), ("c", "42"), ("d", "true")]),
         reasoning="R", calls=[{"name": "mystery", "arguments": {"a": {"k": 1}, "b": "hello", "c": 42, "d": True}}],
         skip=["content", "scorer"]),
    case("multiline_value", "a multi-line value is kept whole; one newline each side of the value is the format's",
         "R</think>" + xml_call("edit", [("filePath", "x.py"), ("oldString", "def f():\n    return 1\n"),
                                         ("newString", "def f():\n    return 2\n\n")]), tools=["edit"], reasoning="R",
         calls=[{"name": "edit", "arguments": {"filePath": "x.py", "oldString": "def f():\n    return 1\n",
                                               "newString": "def f():\n    return 2\n\n"}}], skip=["content"]),
    case("value_with_close_parameter_text", "a </parameter> inside a value that is not followed by the next parameter / "
         "</function> is part of the value (the Qwen parser's rule)",
         "R</think><tool_call>\n<function=edit>\n<parameter=oldString>\na </parameter> b\n</parameter>\n<parameter=newString>\n"
         "x\n</parameter>\n</function>\n</tool_call>", tools=["edit"], reasoning="R",
         calls=[{"name": "edit", "arguments": {"oldString": "a </parameter> b", "newString": "x"}}], skip=["content", "scorer"],
         differs="score.py's qwen-think reader (and vLLM's qwen3_coder) cut the value at the first </parameter>; the "
         "server's parser keeps the whole value"),
    case("call_cut_by_max_tokens_dropped", "a call cut by max_tokens is dropped AND stripped (spec 23 decision 16's rule: an "
         "agent must never execute a truncated edit)", "R</think>Let me edit.\n<tool_call>\n<function=edit>\n"
         "<parameter=filePath>\nx.py\n</parameter>\n<parameter=oldString>\nabc", tools=["edit"], reasoning="R",
         content="Let me edit.\n", skip=["scorer"], differs="the Qwen / Agnes stream emits a call whose function tag is "
         "complete even when a parameter is cut; Bolt drops it"),
    case("call_cut_inside_tag_dropped", "generation ends inside the opening tag of a call: nothing of it is content",
         "R</think>Ok.<tool_call>\n<func", reasoning="R", content="Ok.", skip=["scorer"]),
    case("call_cut_before_function_end_dropped", "all parameters closed but </function> missing: still cut -> dropped",
         "R</think><tool_call>\n<function=read>\n<parameter=filePath>\n/a\n</parameter>\n", tools=["read"], reasoning="R",
         skip=["scorer"]),
    case("call_without_close_tag_kept", "</function> present, </tool_call> missing (EOS after the function): the call is "
         "complete and kept", "R</think><tool_call>\n<function=read>\n<parameter=filePath>\n/a\n</parameter>\n</function>",
         tools=["read"], reasoning="R", calls=[{"name": "read", "arguments": {"filePath": "/a"}}], skip=["scorer"]),
    case("malformed_call_is_content", "a closed block without <function=...> is content, not a call",
         "R</think><tool_call>\nnot a call\n</tool_call>", reasoning="R", content="<tool_call>\nnot a call\n</tool_call>",
         skip=["scorer", "content"]),
    case("content_after_call", "content after a call is content", "R</think>" + CALL_READ + "\nTrailing words.",
         tools=["read"], reasoning="R", content="Trailing words.", calls=[READ_A], skip=["content"]),
    case("thinking_off_call", "thinking off: a call is parsed the same way, with no reasoning channel",
         "Doing it.\n" + CALL_READ, thinking=False, tools=["read"], content="Doing it.\n", calls=[READ_A], skip=["content"]),
    case("thinking_off_call_cut", "thinking off: a cut call is dropped too", "Doing it.\n<tool_call>\n<function=read>\n"
         "<parameter=filePath>\n/a", thinking=False, tools=["read"], content="Doing it.\n", skip=["content", "scorer"]),
    case("thinking_off_think_tags_before_call", "thinking off: a </think> then a call - all content, the call parsed",
         "x</think>" + CALL_READ, thinking=False, tools=["read"], content="x</think>", calls=[READ_A], skip=["content"]),
    case("force_moves_reasoning_as_text", "force_nonempty_content moves reasoning to content AS TEXT: a call written in "
         "unfinished reasoning is never executed", "Maybe " + CALL_READ, force=True, tools=["read"],
         content="Maybe " + CALL_READ, skip=["scorer", "content"],
         differs="vLLM would run its tool parser over the moved text and return the call; the server keeps it text"),
    case("unicode_split_safe", "reasoning, content and arguments carry multi-byte text across any split",
         "Denke nach … 東京 😀\n</think>Grüße 😀" + xml_call("read", [("filePath", "/tmp/ünï/日本語.txt")]), tools=["read"],
         reasoning="Denke nach … 東京 😀\n", content="Grüße 😀",
         calls=[{"name": "read", "arguments": {"filePath": "/tmp/ünï/日本語.txt"}}], skip=["content"]),
    case("tag_prefix_in_reasoning", "text that merely starts like </think> is reasoning",
         "a </thin and </thinking> and <thi\n</think>done", reasoning="a </thin and </thinking> and <thi\n", content="done"),
    case("start_token_in_output", "a <think> the model writes itself: ultra_v3 (DeepSeek-R1's partition) drops everything up "
         "to and including it and then ONE leading newline of what is left",
         "junk<think>\nthe real thought\n</think>ans", reasoning="junk<think>\nthe real thought\n", content="ans",
         skip=["reasoning", "scorer"], differs="live output never carries the start token (the prompt wrote it): the server "
         "does NOT drop text before a <think> or a newline - it is reasoning text, verbatim"),
]


def mode_parse(out_path):
    sys.path.insert(0, tools_dir)
    import score  # noqa: E402  (tools/toolcall/score.py: the first-call reader of A4)

    cases = []
    for c in PARSE_CASES:
        e, text, skip = c["expect"], c["text"], c["skip"]
        thinking, force = c["thinking"], c["force_nonempty_content"]
        # 1. the split against a restatement of ultra_v3 (reasoning exactly; content exactly when no call is in play).
        r, k = ultra_v3(text, not thinking, force)
        moved = thinking and force and r is None and k is not None
        if "reasoning" not in skip:
            assert (r or "") == e["reasoning"], (c["name"], r, e["reasoning"])
        if "content" not in skip and ("<tool_call>" not in text or moved):
            assert (k or "") == e["content"], (c["name"], k, e["content"])
        # 2. the first call as score.py's `qwen-think` reads it (the A4 scorer): raw strings.
        if thinking and "scorer" not in skip:
            kind, name, params, n = score.parse_first_call(text, "qwen-think")
            assert (kind == "call") == bool(e["tool_calls"]), (c["name"], kind)
            if kind == "call":
                assert name == e["tool_calls"][0]["name"], c["name"]
                c["scorer"] = {"kind": kind, "name": name, "params": params, "calls": n}
        cases.append(c)
    facts = {"comment": "tools/tokenizer/dump_bolt.py parse; bolt_stream_test holds server::BoltOutputStream to these "
                        "expectations (the whole text, byte at a time, every two-way split, random pieces); `differs` says "
                        "where the server deliberately departs from the reference reading",
             "ultra_v3_sha256_prefix": "14e03de5", "cases": cases}
    write(out_path, facts)
    print(f"{out_path}: {len(cases)} cases ({sum(1 for c in cases if 'differs' in c)} with a documented difference, "
          f"{sum(1 for c in cases if 'scorer' in c)} with the scorer's first call)")


def main():
    if len(sys.argv) < 4 or sys.argv[1] not in ("tokenizer", "templates", "parse"):
        sys.exit("usage: dump_bolt.py tokenizer|templates|parse <snapshot> <out.json> [<bolt_agentic.json>]")
    mode, snapshot, out_path = sys.argv[1:4]
    agentic = sys.argv[4] if len(sys.argv) > 4 else None
    if mode == "parse":
        assert sha256(os.path.join(snapshot, "ultra_v3_reasoning_parser.py")).startswith("14e03de5"), "not ultra_v3"
        mode_parse(out_path)
        return
    assert transformers.__version__ == "5.19.0", transformers.__version__
    assert sha256(os.path.join(snapshot, "chat_template.jinja")).startswith("6521afc7"), "not Bolt's template"
    tok = AutoTokenizer.from_pretrained(snapshot)
    {"tokenizer": mode_tokenizer, "templates": mode_templates}[mode](tok, snapshot, out_path, agentic)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "parse":
        import tokenizers
        import transformers
        from transformers import AutoTokenizer
    main()
