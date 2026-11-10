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
from score import is_k2, k2_split_reasoning, load_tree, main, parse_first_call, validity  # noqa: E402

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


# ---- Nemotron 3.5 Bolt (spec 25a Task 6): Qwen3-Coder XML behind the reasoning the prompt opens -------------------

def bolt(after_think: str, reasoning: str = "Let me read the file first.\n") -> str:
    """An output continuing Bolt's prompt (`...<|im_start|>assistant\\n<think>\\n`): reasoning, </think>, the rest."""
    return reasoning + "</think>" + after_think


BOLT_CALL = ("\n<tool_call>\n<function=read>\n<parameter=filePath>\n/work/src/a.cc\n</parameter>\n"
             "<parameter=limit>\n40\n</parameter>\n</function>\n</tool_call>\n<|im_end|>")


def test_qwen_think_reasoning_then_call():
    assert parse_first_call(bolt("I will read it." + BOLT_CALL), "qwen-think") == (
        "call", "read", {"filePath": "/work/src/a.cc", "limit": "40"}, 1)
    two = bolt(BOLT_CALL.replace("<|im_end|>", "") + "<tool_call>\n<function=bash>\n</function>\n</tool_call>")
    assert parse_first_call(two, "qwen-think")[:2] == ("call", "read") and parse_first_call(two, "qwen-think")[3] == 2


def test_qwen_think_call_inside_reasoning_is_reasoning():
    inside = "I could call <tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>\n"
    assert parse_first_call(inside, "qwen-think")[0] == "reasoning"            # never closed: all reasoning
    assert parse_first_call(bolt("Done, nothing to call.<|im_end|>", inside), "qwen-think")[0] == "no call"
    assert parse_first_call(inside + "</think>", "qwen")[:2] == ("call", "bash")   # the plain Qwen rule would take it
    assert parse_first_call("<think>\n" + bolt(BOLT_CALL), "qwen-think")[:2] == ("call", "read")   # a leading <think>


def test_qwen_think_multiline_and_parameter_like_text():
    body = "line one\n    line two\n\nline four"
    t = bolt(f"\n<tool_call>\n<function=write>\n<parameter=content>\n{body}\n</parameter>\n</function>\n</tool_call>")
    assert parse_first_call(t, "qwen-think")[2] == {"content": body}
    odd = "see </parameterized> and </parameters> here"
    t = bolt(f"\n<tool_call>\n<function=write>\n<parameter=content>\n{odd}\n</parameter>\n</function>\n</tool_call>")
    assert parse_first_call(t, "qwen-think")[2] == {"content": odd}
    lit = "a </parameter> b"
    t = bolt(f"\n<tool_call>\n<function=write>\n<parameter=content>\n{lit}\n</parameter>\n</function>\n</tool_call>")
    assert parse_first_call(t, "qwen-think")[2] == {"content": "a "}               # cut, as vLLM's parser cuts
    cut = bolt("\n<tool_call>\n<function=read>\n<parameter=filePath>\n/w")
    assert parse_first_call(cut, "qwen-think")[0] == "incomplete"


def test_qwen_think_bolt_template_round_trip():
    """The assistant spans Bolt's own template rendered (tests/server/bolt_template_cases.json, apply_chat_template in
    transformers 5.19.0) parse back to the call they render: the name and every argument as the template wrote it
    (`| string` for scalars - Python's True / None -, tojson for lists and dicts)."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "..", "tests", "server", "bolt_template_cases.json"), encoding="utf-8") as f:
        cases = {c["name"]: c for c in json.load(f)["cases"]}
    want = {"content before the call, reasoning": ("read", {"filePath": "/w/x"}),
            "a call: string / number / bool / null / list / dict arguments":
                ("f", {"s": "a\nb", "n": "3", "x": "2.5", "b": "True", "z": "None", "l": '[1, "two"]', "d": '{"k": "v"}'})}
    for name, (fn, args) in want.items():
        text = cases[name]["text"]
        span = text[text.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n"):]
        span = span[: span.index("<|im_end|>") + len("<|im_end|>")]
        got = parse_first_call(span, "qwen-think")
        assert got[:3] == ("call", fn, args), (name, got)
    print(f"  {len(want)} spans of Bolt's own renders parse back to their calls")


# ---- validity against the scenario's files (open-work 3o) ----------------------------------

TREE = load_tree()
LL = "/work/src/runtime/prefill/linear_l0.cc"


def edit_call(old: str, new: str, path: str = LL) -> str:
    return (f"<tool_call>\n<function=edit>\n<parameter=filePath>\n{path}\n</parameter>\n<parameter=oldString>\n"
            f"{old}\n</parameter>\n<parameter=newString>\n{new}\n</parameter>\n</function>\n</tool_call><|im_end|>")


# r11.a4's t3_rename-linear_l0 under l0-int8 prefill + int8 KV on 2026-11-10 (uint32_t copied as uint3_t),
# and the bf16 reference's call: the same edit with the right digit.
T3_OLD = ("  const void* scales = sh.layout == 0 ? w.scales->ptr() : nullptr;\n  float* out = s.partials.as<float>();\n"
          "  for (uint32_t n0 = 0; n0 < sh.N; n0 += kNs) {\n    cx.launch(dq, kNs / 16, sh.K / 64, 1,\n"
          "              {PtrArg(w.mem.ptr()), PtrArg(scales), PtrArg(slab.ptr()), arg_val(n0)});")
T3_NEW = T3_OLD.replace("scales =", "scales_val =").replace("PtrArg(scales)", "PtrArg(scales_val)")


def v(text: str) -> str:
    return validity(parse_first_call(text), TREE)


def test_tree_is_what_the_prompts_show():
    """Every read result of the 36 conversations is the snapshot's text, line for line."""
    d = os.path.join(TOK, "..", "golden", "toolcall")
    names = [m["name"] for m in json.load(open(os.path.join(d, "manifest.json")))]
    seen = 0
    for n in names:
        msgs = json.load(open(os.path.join(d, n + ".json")))["messages"]
        for i, m in enumerate(msgs):
            if m["role"] == "tool" and m["content"].startswith("<file>"):
                a = msgs[i - 1]["tool_calls"][0]["function"]["arguments"]
                a = json.loads(a) if isinstance(a, str) else a
                lines = TREE[2][a["filePath"][len("/work/"):]].split("\n")
                for k, ln in re.findall(r"^(\d{5})\| (.*)$", m["content"], re.M):
                    assert lines[int(k) - 1] == ln, (n, k)
                    seen += 1
    assert len(names) == 36 and seen > 1000


def test_negative_control_uint3_t_fails_its_twin_passes():
    bad = edit_call(T3_OLD.replace("uint32_t", "uint3_t"), T3_NEW.replace("uint32_t", "uint3_t"))
    assert v(bad) == "edit's oldString occurs 0 times in src/runtime/prefill/linear_l0.cc"
    assert v(edit_call(T3_OLD, T3_NEW)) == ""


def test_edit_rules():
    narrower = T3_OLD.split("\n")[:2]
    assert v(edit_call("\n".join(narrower), "\n".join(T3_NEW.split("\n")[:2]))) == ""      # a narrower oldString
    assert "occurs 2 times" in v(edit_call(narrower[0], "x"))                                   # linear_l0_silu has it too
    assert "occurs 3 times" in v(edit_call("  const model::GemvShape& sh = w.shape;", "x"))
    assert "equals" in v(edit_call(T3_OLD, T3_OLD))
    assert "not a file the conversation read" in v(edit_call("x", "y", "/work/src/runtime/prefill/linear_l0.h"))
    assert "occurs 0 times" in v(edit_call(T3_OLD.replace("\n    cx.launch", "\n  cx.launch"), T3_NEW))  # indentation counts


def call(name: str, **kw) -> str:
    ps = "".join(f"<parameter={k}>\n{x}\n</parameter>\n" for k, x in kw.items())
    return f"<tool_call>\n<function={name}>\n{ps}</function>\n</tool_call>"


def test_read_grep_glob_rules():
    assert v(call("read", filePath="/work/tools/box.sh", offset="15", limit="20")) == ""
    assert v(call("read", filePath="src/runtime/prefill/linear_l0.h")) == ""                 # relative to /work
    assert "not a file of the tree" in v(call("read", filePath="/work/src/runtime/prefill/linear.cc"))
    assert "not an integer" in v(call("read", filePath=LL, offset="ten"))
    assert "past the end" in v(call("read", filePath=LL, offset="5000"))
    assert v(call("grep", pattern=r"\bsilu_fused\b", path="/work")) == ""
    assert v(call("grep", pattern="kv_stride", path="/work/src/runtime/prefill/step.cc")) == ""   # narrower
    assert v(call("grep", pattern="step", path="/work/tests", include="CMakeLists.txt")) == ""
    assert v(call("grep", pattern="nothing_has_this_name", path="/work")) == ""                  # no hit is an answer
    assert "does not compile" in v(call("grep", pattern="invalid(", path="/work"))
    assert "not in the tree" in v(call("grep", pattern="x", path="/work/src/nowhere"))
    assert v(call("glob", pattern="tests/**/*linear*", path="/work")) == ""                    # the reference's, no hit
    assert v(call("glob", pattern="tools/**", path="/work")) == ""                             # broader
    assert "not a directory" in v(call("glob", pattern="*", path=LL))
    assert v(call("bash", command="ctest -R x", description="y")) == ""
    assert v("<tool_call>\n<function=edit>\n<parameter=oldString>\nhalf") == "the call does not parse"
    assert v("I am done.") == ""


def test_bar_fails_on_an_invalid_call():
    import io
    import tempfile
    from contextlib import redirect_stdout
    with tempfile.TemporaryDirectory() as d:
        for n in ("a", "b"):
            open(os.path.join(d, f"{n}.bf16.txt"), "w").write(edit_call(T3_OLD, T3_NEW))
            open(os.path.join(d, f"{n}.good.txt"), "w").write(edit_call(T3_OLD, T3_NEW))
        open(os.path.join(d, "a.bad.txt"), "w").write(edit_call(T3_OLD.replace("uint32_t", "uint3_t"), T3_NEW))
        open(os.path.join(d, "b.bad.txt"), "w").write(edit_call(T3_OLD, T3_NEW))
        for cands, rc in ((["good"], None), (["bad"], 1)):
            sys.argv = ["score.py", "--min", "1", d, "bf16"] + cands
            out = io.StringIO()
            try:
                with redirect_stdout(out):
                    main()
                got = None
            except SystemExit as e:
                got = e.code
            assert got == rc, (cands, out.getvalue())
        assert "| a | edit | invalid |" in out.getvalue() and "bad: 1 / 2 match bf16" in out.getvalue()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"ok  {n}")
    print(f"{len(tests)} tests passed")
