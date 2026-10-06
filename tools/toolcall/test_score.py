"""score.py's readers: Qwen XML, and K2-Horizon's three formats behind its reasoning (spec 18d).

    python3 tools/toolcall/test_score.py     (pytest also collects it, where installed)

The K2 cases include every assistant turn with tool calls of the recorded HF renders
(tests/tokenizer/k2_template_*.txt, transformers' apply_chat_template of
tests/tokenizer/k2_template_cases.json - spec 18 §12): the first call read back must be the
message's first call, in xml, xml_typed and json.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import is_k2, k2_split_reasoning, parse_first_call  # noqa: E402

TOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "tokenizer")


def test_multiline_value_exact():
    t = "<tool_call>\n<function=edit>\n<parameter=oldString>\na\n  b\n</parameter>\n</function>\n</tool_call>"
    assert parse_first_call(t) == ("call", "edit", {"oldString": "a\n  b"}, 1)


def test_text_before_call_is_allowed():
    t = "Let me look.\n<tool_call>\n<function=read>\n<parameter=filePath>\n/work/x.cc\n</parameter>\n</function>\n</tool_call>"
    assert parse_first_call(t)[:3] == ("call", "read", {"filePath": "/work/x.cc"})


def test_no_call():
    assert parse_first_call("The function returns early.")[0] == "no call"


def test_two_calls_first_wins():
    one = "<tool_call>\n<function=grep>\n<parameter=pattern>\nfoo\n</parameter>\n</function>\n</tool_call>"
    two = one.replace("grep", "glob")
    kind, name, params, count = parse_first_call(one + "\n" + two)
    assert (kind, name, count) == ("call", "grep", 2)


def test_truncated_is_incomplete():
    t = "<tool_call>\n<function=edit>\n<parameter=oldString>\nhalf a str"
    assert parse_first_call(t)[0] == "incomplete"


# ---- K2-Horizon (spec 18d) ------------------------------------------------------------------

K2_XML = ("I should read it.</ifm|think_faster>Reading.\n<ifm|tool_calls>\n<ifm|tool_call>read\n"
          "<ifm|arg_key>filePath</ifm|arg_key>\n<ifm|arg_value>/w/a.cc</ifm|arg_value>\n"
          "<ifm|arg_key>offset</ifm|arg_key>\n<ifm|arg_value>12</ifm|arg_value>\n</ifm|tool_call>\n"
          "<ifm|tool_call>glob\n<ifm|arg_key>pattern</ifm|arg_key>\n<ifm|arg_value>**/*.cc</ifm|arg_value>\n"
          "</ifm|tool_call>\n</ifm|tool_calls><|ifm|im_end|>")


def test_k2_detected_qwen_not():
    assert is_k2(K2_XML)
    assert not is_k2("<tool_call>\n<function=read>\n</function>\n</tool_call>")


def test_k2_xml_first_call():
    assert parse_first_call(K2_XML) == ("call", "read", {"filePath": "/w/a.cc", "offset": "12"}, 2)


def test_k2_xml_typed():
    t = ("short</ifm|think><ifm|tool_calls>\n<ifm|tool_call>read\n<ifm|arg_key>limit</ifm|arg_key>\n"
         "<ifm|arg_type>integer</ifm|arg_type>\n<ifm|arg_value>40</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>")
    assert parse_first_call(t) == ("call", "read", {"limit": "40"}, 1)


def test_k2_json_and_string_arguments():
    t = ('x</ifm|think_fast><ifm|tool_calls>\n<ifm|tool_call>{"name": "edit", "arguments": {"filePath": "/b", '
         '"replaceAll": true, "n": [1, 2]}}</ifm|tool_call>\n</ifm|tool_calls>')
    assert parse_first_call(t) == ("call", "edit", {"filePath": "/b", "replaceAll": "true", "n": "[1, 2]"}, 1)
    s = ('</ifm|think><ifm|tool_calls>\n<ifm|tool_call>{"name": "read", "arguments": "{\\"filePath\\": \\"/c\\"}"}'
         '</ifm|tool_call>\n</ifm|tool_calls>')
    assert parse_first_call(s)[:3] == ("call", "read", {"filePath": "/c"})


def test_k2_reasoning_ends_implicitly_at_the_block():
    t = "no close tag<ifm|tool_calls>\n<ifm|tool_call>ls\n</ifm|tool_call>\n</ifm|tool_calls>"
    assert k2_split_reasoning(t)[0] == "no close tag"
    assert parse_first_call(t) == ("call", "ls", {}, 1)


def test_k2_cut_inside_reasoning():
    assert parse_first_call("still thinking about <ifm|tool_call>read")[0] == "reasoning"
    assert parse_first_call("<ifm|think_faster>\nhmm")[0] == "reasoning"


def test_k2_no_call_and_incomplete():
    assert parse_first_call("done</ifm|think>The answer is 4.<|ifm|im_end|>")[0] == "no call"
    cut = "ok</ifm|think><ifm|tool_calls>\n<ifm|tool_call>read\n<ifm|arg_key>filePath</ifm|arg_key>\n<ifm|arg_val"
    assert parse_first_call(cut)[0] == "incomplete"
    bad = "ok</ifm|think><ifm|tool_calls>\n<ifm|tool_call>{\"name\": \"read\", \"arguments\": {</ifm|tool_call>"
    assert parse_first_call(bad)[0] == "incomplete"
    noname = "ok</ifm|think><ifm|tool_calls>\n<ifm|tool_call>\n<ifm|arg_key>a</ifm|arg_key>\n<ifm|arg_value>1</ifm|arg_value>\n</ifm|tool_call>"
    assert parse_first_call(noname)[0] == "incomplete"


def _same_value(raw: str, want) -> bool:
    return raw == want if isinstance(want, str) else json.loads(raw) == want


def test_k2_hf_renders_round_trip():
    """Every assistant turn with tool calls in the recorded HF renders reads back as the message's
    first call (name and every argument), with the message's call count."""
    with open(os.path.join(TOK, "k2_template_cases.json"), encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    seen = set()
    for case in cases:
        msgs = [m for m in case["messages"] if m["role"] == "assistant"]
        if not any(m.get("tool_calls") for m in msgs):
            continue
        with open(os.path.join(TOK, f"k2_template_{case['name']}.txt"), encoding="utf-8") as f:
            render = f.read()
        turns = re.findall(r"<\|ifm\|im_start\|>assistant\n(.*?)<\|ifm\|im_end\|>", render, re.S)
        assert len(turns) == len(msgs), case["name"]
        for m, text in zip(msgs, turns):
            if not m.get("tool_calls"):
                continue
            kind, name, params, n = parse_first_call(text)
            first = m["tool_calls"][0]["function"]
            assert (kind, name, n) == ("call", first["name"], len(m["tool_calls"])), (case["name"], kind, name, n)
            assert set(params) == set(first["arguments"]), (case["name"], params)
            for k, v in first["arguments"].items():
                assert _same_value(params[k], v), (case["name"], k, params[k], v)
            seen.add(case["kwargs"].get("tool_call_format", "xml"))
    assert seen == {"xml", "xml_typed", "json"}, seen


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"ok  {n}")
    print(f"{len(tests)} tests passed")
