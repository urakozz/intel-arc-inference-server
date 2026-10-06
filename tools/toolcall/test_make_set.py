"""make_set.py --from: the re-tokenised set's template variables and key order (spec 18d).

    python3 tools/toolcall/test_make_set.py     (pytest also collects it, where installed)

A fake tokenizer stands in for the checkpoint's (no transformers on the Mac): it records what
apply_chat_template was called with and returns 900 ids derived from it.
"""
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_set  # noqa: E402


class FakeTok:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, msgs, tools=None, add_generation_prompt=False, enable_thinking=None,
                            tokenize=False, **kwargs):
        assert add_generation_prompt and tokenize
        self.calls.append({"msgs": msgs, "tools": tools, "enable_thinking": enable_thinking, "kwargs": kwargs})
        seed = int(hashlib.sha256(json.dumps([msgs, tools, kwargs]).encode()).hexdigest()[:8], 16)
        return {"input_ids": [(seed + i) % 250000 for i in range(900)]}


def make_src(d):
    """Two scenarios whose objects are NOT key-sorted (tools.json's order)."""
    tools = [{"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {"offset": {"type": "integer"}, "filePath": {"type": "string"}}}}}]
    manifest = []
    for n in ("t1_a", "t2_b"):
        sc = {"name": n, "expect": "read", "enable_thinking": False,
              "messages": [{"role": "user", "content": f"open {n}"},
                           {"role": "assistant", "content": "",
                            "tool_calls": [{"id": "c1", "type": "function",
                                            "function": {"name": "read", "arguments": {"offset": 3, "filePath": "/w"}}}]},
                           {"role": "tool", "tool_call_id": "c1", "content": "ok"}],
              "tools": tools}
        with open(os.path.join(d, f"{n}.json"), "w", encoding="utf-8") as f:
            json.dump(sc, f)
        with open(os.path.join(d, f"{n}.ids"), "w", encoding="utf-8") as f:
            f.write("1 2 3\n")
        manifest.append({"name": n, "ids": 3, "sha256": "x"})
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)


_TMP = []   # TemporaryDirectory objects, cleaned up at exit


def run(mtype, kwargs=None):
    src, out = tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()
    _TMP.extend([src, out])
    make_src(src.name)
    tok = FakeTok()
    make_set.retokenise("unused", src.name, out.name, kwargs, tok=tok, mtype=mtype)
    return tok, out.name


def keys_sorted(x):
    if isinstance(x, dict):
        return list(x) == sorted(x) and all(keys_sorted(v) for v in x.values())
    if isinstance(x, list):
        return all(keys_sorted(v) for v in x)
    return True


def test_k2_defaults_sorted_and_recorded():
    tok, out = run("k2_horizon")
    assert len(tok.calls) == 2
    for c in tok.calls:
        assert c["kwargs"] == {"tool_call_format": "xml", "reasoning_effort": "low"}
        assert keys_sorted(c["msgs"]) and keys_sorted(c["tools"])
    for n in ("t1_a", "t2_b"):
        sc = json.load(open(os.path.join(out, f"{n}.json"), encoding="utf-8"))
        assert sc["chat_template_kwargs"] == {"tool_call_format": "xml", "reasoning_effort": "low"}
    man = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    for e in man:
        raw = open(os.path.join(out, f"{e['name']}.ids"), "rb").read()
        assert e["ids"] == 900 and e["sha256"] == hashlib.sha256(raw).hexdigest()


def test_k2_explicit_kwargs_win():
    kw = {"tool_call_format": "json", "reasoning_effort": "medium"}
    tok, out = run("k2_horizon", dict(kw))
    assert all(c["kwargs"] == kw for c in tok.calls)
    assert json.load(open(os.path.join(out, "t1_a.json"), encoding="utf-8"))["chat_template_kwargs"] == kw


def test_qwen_path_unchanged():
    tok, out = run("qwen3_5_moe")
    for c in tok.calls:
        assert c["kwargs"] == {} and c["enable_thinking"] is False
        assert not keys_sorted(c["tools"])   # rendered as the set holds them, unsorted
    assert "chat_template_kwargs" not in json.load(open(os.path.join(out, "t1_a.json"), encoding="utf-8"))


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"ok  {n}")
    print(f"{len(tests)} tests passed")
