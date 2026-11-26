"""score.py's readers: Qwen XML,
Kolibri-1's hermes JSON calls behind the reasoning the model opens (spec 20e, `--format hermes`) and
Inkling 2 Mini's <|content_invoke_tool_json|> blocks (`--format inkling`).

    python3 tools/toolcall/test_score.py     (pytest also collects it, where installed)

The hermes cases read back every
assistant turn with calls of tests/tokenizer/kolibri_template_*.txt (spec 20e Task 1).
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import load_tree, main, parse_first_call, validity  # noqa: E402

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


# ---- Inkling 2 Mini: <|content_invoke_tool_json|> blocks ------------------------------------

def ink_call(name: str, args: dict, json_name: str | None = None) -> str:
    """A call block as the checkpoint's template writes it: sorted keys, no spaces, HTML-safe escapes (tojson)."""
    body = json.dumps(args, sort_keys=True, separators=(",", ":"))
    for c, e in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), ("'", "\\u0027")):
        body = body.replace(c, e)
    return (f"<|message_model|>{name}<|content_invoke_tool_json|>"
            f'{{"name":{json.dumps(json_name or name)},"args":{body}}}<|end_message|>')


END_TURN = "<|content_model_end_sampling|>"


def test_inkling_first_call():
    t = ("<|message_model|><|content_text|>Let me look.<|end_message|>"
         + ink_call("read", {"filePath": "/work/a.cc", "limit": 40}) + ink_call("grep", {"pattern": "x"}) + END_TURN)
    assert parse_first_call(t, "inkling") == ("call", "read", {"filePath": "/work/a.cc", "limit": 40}, 2)
    headless = t[len("<|message_model|><|content_text|>Let me look.<|end_message|><|message_model|>"):]
    assert parse_first_call(headless, "inkling")[:3] == ("call", "read", {"filePath": "/work/a.cc", "limit": 40})


def test_inkling_escapes_and_multiline_values():
    old = "if (a < b && c > 'd') {\n  return;\n}"
    assert parse_first_call(ink_call("edit", {"oldString": old, "newString": "x"}), "inkling")[2]["oldString"] == old


def test_inkling_thinking_none_and_broken():
    think = "<|message_model|><|content_thinking|>I should read it" + ink_call("read", {"filePath": "a"})
    assert parse_first_call(think, "inkling")[0] == "call"            # a closed-over block: the call follows it
    assert parse_first_call("<|message_model|><|content_thinking|>still thinking", "inkling")[0] == "reasoning"
    inside = "<|message_model|><|content_thinking|>read<|content_invoke_tool_json|>{}<|end_message|>"
    assert parse_first_call(inside + "<|message_model|><|content_text|>no<|end_message|>", "inkling")[0] == "no call"
    assert parse_first_call("<|message_model|><|content_text|>Done.<|end_message|>" + END_TURN, "inkling")[0] \
        == "no call"
    cut = ink_call("edit", {"oldString": "a"})[:-len("}<|end_message|>")]
    assert parse_first_call(cut, "inkling")[0] == "incomplete"                     # cut off mid-way
    assert parse_first_call(ink_call("edit", {"a": 1})[:-len("<|end_message|>")], "inkling")[0] == "incomplete"
    assert parse_first_call(ink_call("read", {"a": 1}, "grep"), "inkling")[0] == "incomplete"   # names differ
    bad_args = '<|message_model|>read<|content_invoke_tool_json|>{"name":"read","args":"x"}<|end_message|>'
    assert parse_first_call(bad_args, "inkling")[0] == "incomplete"
    assert parse_first_call(ink_call("read", {"a": 1}), "qwen")[0] == "no call"   # the default reads Qwen XML


def test_inkling_round_trips_the_36_conversations():
    """Every assistant call of the A4 conversations, rendered as Inkling's template writes it, reads back."""
    d = os.path.join(TOK, "..", "golden", "toolcall")
    n_calls = 0
    for m in json.load(open(os.path.join(d, "manifest.json"))):
        for msg in json.load(open(os.path.join(d, m["name"] + ".json")))["messages"]:
            for c in msg.get("tool_calls") or []:
                a = c["function"]["arguments"]
                a = json.loads(a) if isinstance(a, str) else a
                got = parse_first_call(ink_call(c["function"]["name"], a) + END_TURN, "inkling")
                assert got == ("call", c["function"]["name"], a, 1), m["name"]
                n_calls += 1
    assert n_calls == 30, n_calls


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


def test_inkling_scores_row_by_row():
    """--format inkling: the tree's validity rules apply (Inkling's set re-renders the same 36 conversations)."""
    import io
    import tempfile
    from contextlib import redirect_stdout
    good = ink_call("edit", {"filePath": LL, "oldString": T3_OLD, "newString": T3_NEW}) + END_TURN
    typo = ink_call("edit", {"filePath": LL, "oldString": T3_OLD.replace("uint32_t", "uint3_t"),
                             "newString": T3_NEW}) + END_TURN
    other = ink_call("read", {"filePath": LL, "offset": 10}) + END_TURN
    with tempfile.TemporaryDirectory() as d:
        for n, cand in (("a", good), ("b", typo), ("c", other)):
            open(os.path.join(d, f"{n}.ref.txt"), "w").write(good)
            open(os.path.join(d, f"{n}.kv8.txt"), "w").write(cand)
        sys.argv = ["score.py", "--format", "inkling", "--min", "1", d, "ref", "kv8"]
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                main()
            rc = None
        except SystemExit as e:
            rc = e.code
        o = out.getvalue()
        assert rc == 1, o
        for row in ("| a | edit | match |", "| b | edit | invalid |", "| c | edit | mismatch |",
                    "kv8: 1 / 3 match ref", "kv8: 1 invalid call(s)", "ref: 0 parse failure(s)"):
            assert row in o, (row, o)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"ok  {n}")
    print(f"{len(tests)} tests passed")
