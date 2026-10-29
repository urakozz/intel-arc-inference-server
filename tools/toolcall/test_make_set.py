"""make_set.py --from: the re-tokenised set's template variables and key order (spec 18d); --lang de:
Kolibri-1's German set (spec 20e, KL4) - the German table, key-sorted renders, thinking off.

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


def test_german_set_for_kolibri():
    corpus = make_set.scan_repo()
    out = tempfile.TemporaryDirectory()
    _TMP.append(out)
    tok = FakeTok()
    man = make_set.build_set("unused", out.name, "de", tok=tok, mtype="kolibri1", corpus=corpus)
    assert len(man) == 36 and len(tok.calls) == 36
    for c in tok.calls:
        assert c["enable_thinking"] is False and c["kwargs"] == {}
        assert keys_sorted(c["msgs"]) and keys_sorted(c["tools"])   # b70-serve's render order
        assert c["msgs"][0]["content"] == make_set.SYSTEM_DE
    # Every user turn is the German table's, none the English one's.
    de = [v for k, v in make_set.TEXT["de"].items() if k != "system"]
    en = [v for k, v in make_set.TEXT["en"].items() if k != "system"]
    stems = lambda table: [t.split("{")[0] for t in table]   # noqa: E731
    users = [m["content"] for c in tok.calls for m in c["msgs"] if m["role"] == "user"]
    assert all(any(u.startswith(st) for st in stems(de)) for u in users)
    assert not any(u.startswith(st) for u in users for st in stems(en) if len(st) > 3)
    sc = json.load(open(os.path.join(out.name, "t1_define-linear_l0.json"), encoding="utf-8"))
    assert sc["lang"] == "de" and sc["enable_thinking"] is False
    assert sc["messages"][1]["content"].startswith("Finde heraus, wo `")
    assert set(make_set.TEXT["de"]) == set(make_set.TEXT["en"])


def test_english_set_unchanged_by_the_table():
    """The English table renders the original set's strings (Qwen's path: no key sorting, no lang)."""
    corpus = make_set.scan_repo()
    out = tempfile.TemporaryDirectory()
    _TMP.append(out)
    tok = FakeTok()
    make_set.build_set("unused", out.name, "en", tok=tok, mtype="qwen3_5", corpus=corpus)
    assert tok.calls[0]["msgs"][0]["content"] == make_set.SYSTEM
    assert not keys_sorted(tok.calls[0]["tools"])
    sc = json.load(open(os.path.join(out.name, "t5_test-step.json"), encoding="utf-8"))
    assert "lang" not in sc
    assert sc["messages"][1]["content"] == "I just changed /work/src/runtime/prefill/step.cc. Run the tests for this file."


def test_bolt_defaults_sorted_thinking_on():
    """Spec 25a Task 6: Bolt (nemotron_h_omni) renders key-sorted with enable_thinking TRUE and medium_effort, both
    recorded per scenario (the set's own scenarios say enable_thinking False: Qwen3.8's render)."""
    tok, out = run("nemotron_h_omni")
    assert len(tok.calls) == 2
    for c in tok.calls:
        assert c["kwargs"] == make_set.BOLT_KWARGS == {"medium_effort": True} and c["enable_thinking"] is True
        assert keys_sorted(c["msgs"]) and keys_sorted(c["tools"])
    sc = json.load(open(os.path.join(out, "t1_a.json"), encoding="utf-8"))
    assert sc["chat_template_kwargs"] == {"medium_effort": True} and sc["enable_thinking"] is True


def _bolt_snapshot():
    import glob
    for hub in (os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"), "/hf/hub"):
        for c in sorted(glob.glob(os.path.join(
                hub, "models--urakozz--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ", "snapshots", "*"))):
            if os.path.exists(os.path.join(c, "chat_template.jinja")):
                return c
    return None


def test_bolt_render_is_apply_chat_template():
    """The committed set tests/golden/toolcall-bolt: every scenario's ids are apply_chat_template's on its key-sorted
    messages with the recorded kwargs (re-rendered here byte for byte), the text ends in the reasoning-open generation
    prompt, holds no BOS, and its history calls are Qwen3-Coder XML. SKIPs without transformers 5.19.0 or the pinned
    snapshot's small files (the container with the qwen4exp site has both)."""
    snap = _bolt_snapshot()
    try:
        import transformers
        ok = transformers.__version__ == "5.19.0"
    except ImportError:
        ok = False
    if not (snap and ok):
        print("  skip test_bolt_render_is_apply_chat_template: needs transformers 5.19.0 and the Bolt snapshot")
        return
    from transformers import AutoTokenizer
    make_set.check_bolt_template(snap)
    tok = AutoTokenizer.from_pretrained(snap)
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(here, "..", "..", "tests", "golden", "toolcall-bolt")
    man = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    assert len(man) == 36
    for e in man:
        n = e["name"]
        raw = open(os.path.join(d, f"{n}.ids"), "rb").read()
        assert hashlib.sha256(raw).hexdigest() == e["sha256"], n
        sc = json.load(open(os.path.join(d, f"{n}.json"), encoding="utf-8"))
        assert sc["enable_thinking"] is True and sc["chat_template_kwargs"] == {"medium_effort": True}, n
        msgs, tools = json.loads(json.dumps([sc["messages"], sc["tools"]], sort_keys=True))
        text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=True, **sc["chat_template_kwargs"])
        got = [int(x) for x in raw.split()]
        assert got == tok(text, add_special_tokens=False)["input_ids"], n
        assert tok.decode(got) == text and 1 not in got, n
        assert text.startswith("<|im_start|>system\n") and text.endswith("<|im_start|>assistant\n<think>\n"), n
        assert "{reasoning effort: efficient}<|im_end|>" in text, n
        if any(m.get("tool_calls") for m in sc["messages"]):
            assert "<tool_call>\n<function=" in text, n
    print(f"  bolt: the committed 36 scenarios = apply_chat_template byte for byte (thinking on, medium effort)")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"ok  {n}")
    print(f"{len(tests)} tests passed")
