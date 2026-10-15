"""score.py's readers: Qwen XML, K2-Horizon's three formats behind its reasoning (spec 18d), and
Kolibri-1's hermes JSON calls behind the reasoning the model opens (spec 20e, `--format hermes`).

    python3 tools/toolcall/test_score.py     (pytest also collects it, where installed)

The K2 cases include every assistant turn with tool calls of the recorded HF renders
(tests/tokenizer/k2_template_*.txt, transformers' apply_chat_template of
tests/tokenizer/k2_template_cases.json - spec 18 §12): the first call read back must be the
message's first call, in xml, xml_typed and json. The hermes cases likewise read back every
assistant turn with calls of tests/tokenizer/kolibri_template_*.txt (spec 20e Task 1).
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


# ---- Kolibri-1: hermes JSON (spec 20e) --------------------------------------------------------

def test_hermes_first_call():
    t = '<tool_call>\n{"name": "grep", "arguments": {"pattern": "Schlüssel"}}\n</tool_call>'
    assert parse_first_call(t, "hermes") == ("call", "grep", {"pattern": "Schlüssel"}, 1)


def test_hermes_broken_and_none():
    broken = '<tool_call>\n{"name": "f", "arguments": {"a": 1}\n</tool_call>'   # a missing brace
    assert parse_first_call(broken, "hermes")[0] == "incomplete"
    assert parse_first_call('<tool_call>\n{"name": "read", "argu', "hermes")[0] == "incomplete"
    assert parse_first_call('<tool_call>{"name": "a b"}</tool_call>', "hermes")[0] == "incomplete"
    assert parse_first_call("Die Antwort ist 4.", "hermes")[0] == "no call"


def test_hermes_reasoning_string_arguments_values():
    t = ('<think>\nIch lese sie.\n</think>\n\nOk.\n<tool_call>\n{"name": "read", "arguments": '
         '"{\\"filePath\\": \\"/w/a.cc\\", \\"limit\\": 40}"}\n</tool_call>\n<tool_call>\n'
         '{"name": "grep", "arguments": {}}\n</tool_call>')
    assert parse_first_call(t, "hermes") == ("call", "read", {"filePath": "/w/a.cc", "limit": 40}, 2)
    assert parse_first_call("<think>\nnoch nicht fertig <tool_call>", "hermes")[0] == "reasoning"
    inside = '<think>\n<tool_call>{"name": "x"}</tool_call>\n</think>\n\nkeine'
    assert parse_first_call(inside, "hermes")[0] == "no call"   # a call inside reasoning is not one


def test_qwen_default_unchanged_on_hermes_text():
    t = '<tool_call>\n{"name": "grep", "arguments": {"pattern": "x"}}\n</tool_call>'
    assert parse_first_call(t)[0] == "incomplete"   # the default reads Qwen XML, as before


def test_hermes_hf_renders_round_trip():
    """Every Kolibri assistant turn with calls in the HF renders reads back as its first call."""
    with open(os.path.join(TOK, "kolibri_template_cases.json"), encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    n_turns = 0
    for case in cases:
        msgs = [m for m in case["messages"] if m["role"] == "assistant"]
        if not any(m.get("tool_calls") for m in msgs):
            continue
        with open(os.path.join(TOK, f"kolibri_template_{case['name']}.txt"), encoding="utf-8") as f:
            render = f.read()
        turns = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", render, re.S)
        assert len(turns) == len(msgs), case["name"]
        for m, text in zip(msgs, turns):
            if not m.get("tool_calls"):
                continue
            first = m["tool_calls"][0]["function"]
            assert parse_first_call(text, "hermes") == ("call", first["name"], first["arguments"],
                                                        len(m["tool_calls"])), case["name"]
            n_turns += 1
    assert n_turns == 4, n_turns


# ---- LFM2.5-24B-A2B (spec 24a Task 6): pythonic calls behind the reasoning the prompt opens --------------------

S, E = "<|tool_call_start|>", "<|tool_call_end|>"
BS = chr(92)


def lfm(block: str, before: str = "", after: str = "<|im_end|>") -> str:
    return "I should look at the file first.</think>" + before + S + block + E + after


def test_pythonic_literal_types():
    t = lfm("[f(s='x', i=-3, fl=2.5e-1, d=.5, t=True, n=None, xs=[1, 'a', [2]], m={\"k\": \"v\", \"n\": {\"x\": 1}}, "
            "tu=(1, 2))]")
    assert parse_first_call(t, "pythonic") == ("call", "f", {
        "s": "x", "i": -3, "fl": 0.25, "d": 0.5, "t": True, "n": None, "xs": [1, "a", [2]],
        "m": {"k": "v", "n": {"x": 1}}, "tu": [1, 2]}, 1)


def test_pythonic_json_words_equal_python_words():
    a = parse_first_call(lfm("[f(a=True, b=None, c=False)]"), "pythonic")
    b = parse_first_call(lfm("[f(a=true, b=null, c=false)]"), "pythonic")
    assert a == b and a[2] == {"a": True, "b": None, "c": False}


def test_pythonic_quotes_and_escapes():
    # a top-level string: the template's escapes (backslash, its own quote, n, r) decode; others stay verbatim
    t = lfm("[read(filePath='/w/it" + BS + "'s" + BS + BS + "a" + BS + "nb" + BS + "rc', p='C:" + BS + "Users',"
            " re='" + BS + "d+" + BS + "t', dq=\"say \\\"hi\\\"\")]")
    assert parse_first_call(t, "pythonic")[2] == {"filePath": "/w/it's" + BS + "a\nb\rc", "p": "C:" + BS + "Users",
                                                  "re": BS + "d+" + BS + "t", "dq": 'say "hi"'}
    # inside a list / dict (tojson): JSON's escapes, \t and \u included
    t = lfm('[f(xs=["a' + BS + 'tb", "' + BS + 'u00e9"], m={"q": "x' + BS + '"y"})]')
    assert parse_first_call(t, "pythonic")[2] == {"xs": ["a\tb", "é"], "m": {"q": 'x"y'}}
    # an unescaped quote that is not followed by a closer stays in the string
    assert parse_first_call(lfm("[f(s='it's a test', k=1)]"), "pythonic")[2] == {"s": "it's a test", "k": 1}
    # delimiters inside strings are content
    assert parse_first_call(lfm("[f(s='a, b) ] { = >= c')]"), "pythonic")[2] == {"s": "a, b) ] { = >= c"}
    # triple quotes; double quotes at top level
    assert parse_first_call(lfm("[f(s='''multi\nline''', d=\"x\")]"), "pythonic")[2] == {"s": "multi\nline", "d": "x"}


def test_pythonic_two_calls_first_wins_and_text_around():
    t = lfm("[read(filePath='/w/a'), grep(pattern='x')]", before="Let me read it.\n", after="\nThen grep.<|im_end|>")
    assert parse_first_call(t, "pythonic") == ("call", "read", {"filePath": "/w/a"}, 2)
    two = lfm("[a(x=1)]") + "\n" + S + "[b(y=2)]" + E
    assert parse_first_call(two, "pythonic")[:2] == ("call", "a")


def test_pythonic_names_bare_call_and_empty():
    assert parse_first_call(lfm("f(a=1)"), "pythonic") == ("call", "f", {"a": 1}, 1)          # no brackets
    assert parse_first_call(lfm("[\"f\"(\"path\"='x')]"), "pythonic")[1:3] == ("f", {"path": "x"})
    assert parse_first_call(lfm("[f()]"), "pythonic")[2] == {}
    assert parse_first_call(lfm("[f(name='',)]"), "pythonic")[2] == {"name": ""}             # trailing comma
    assert parse_first_call(lfm("[f(\n  a=1,\n  b=2\n)]"), "pythonic")[2] == {"a": 1, "b": 2}
    assert parse_first_call(lfm("[f({'a': 1})]"), "pythonic")[2] == {"a": 1}                  # a lone positional dict


def test_pythonic_incomplete():
    assert parse_first_call(lfm("[f(1)]"), "pythonic")[0] == "incomplete"                    # positional
    assert parse_first_call(lfm("[f(a)]"), "pythonic")[0] == "incomplete"                    # a name, no value
    assert parse_first_call(lfm("[f(a=)]"), "pythonic")[0] == "incomplete"                   # = without a value
    t = "done.</think>" + S + "[read(filePath='/w/a"                                         # unterminated block
    assert parse_first_call(t, "pythonic")[0] == "incomplete"
    assert parse_first_call(lfm("[f(a=1"), "pythonic")[0] == "incomplete"


def test_pythonic_reasoning_and_no_call():
    # the prompt ends in <think>: an output without </think> ended inside the reasoning
    assert parse_first_call("still thinking " + S + "[f(a=1)]" + E, "pythonic")[0] == "reasoning"
    assert parse_first_call("thought.</think>The function returns early.<|im_end|>", "pythonic")[0] == "no call"
    # a block inside the reasoning is reasoning text, not a call
    inside = "maybe " + S + "[f(a=1)]" + E + " no.</think>Nothing to do.<|im_end|>"
    assert parse_first_call(inside, "pythonic")[0] == "no call"


def test_pythonic_quote_recovery_backtracks():
    """Spec 24 §4.6: a quote followed by a closer that leaves the rest unparseable is kept in the string (code inside
    a single-quoted argument)."""
    t = lfm("[write(path='/w/m.py', content='if __name__ == '__main__':\\n    main()')]")
    assert parse_first_call(t, "pythonic") == ("call", "write", {
        "path": "/w/m.py", "content": "if __name__ == '__main__':\n    main()"}, 1)
    t = lfm("[bash(command='print('hi', x)', description='run')]")
    assert parse_first_call(t, "pythonic")[2] == {"command": "print('hi', x)", "description": "run"}


def test_pythonic_cut_block_keeps_closed_calls():
    """Spec 24 §4.6: generation ending inside a block emits every call whose ) closed; an unfinished first call is
    incomplete."""
    t = "ok.</think>" + S + "[read(filePath='/w/a'), edit(filePath='/w/a', oldString='half"
    assert parse_first_call(t, "pythonic") == ("call", "read", {"filePath": "/w/a"}, 1)
    assert parse_first_call("ok.</think>" + S + "[read(filePath='/w/a'", "pythonic")[0] == "incomplete"


def test_pythonic_template_render_round_trip():
    """chat_template.jinja's format_arg_value, restated: a string argument single-quoted with backslash, quote,
    newline and CR escaped; lists / dicts through tojson; everything else str(). Reading it back gives the value."""
    def fmt(v):
        if isinstance(v, str):
            return "'" + v.replace(BS, BS + BS).replace("'", BS + "'").replace("\n", BS + "n").replace("\r", BS + "r") + "'"
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)
    args = {"s": "it's C:" + BS + "x\nline\r", "n": 3, "f": 1.5, "b": False, "z": None,
            "xs": ["a\"b", 1, {"k": "t" + BS + "u"}], "m": {"q": "x\ty"}}
    block = "[f(" + ", ".join(f"{k}={fmt(v)}" for k, v in args.items()) + ")]"
    assert parse_first_call(lfm(block), "pythonic") == ("call", "f", args, 1)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"ok  {n}")
    print(f"{len(tests)} tests passed")
