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


def test_lfm2moe_sorted_no_kwargs():
    """Spec 24a Task 6: LFM2.5 (model_type lfm2_moe) renders from KEY-SORTED messages and tools - b70-serve's
    nlohmann::json order (spec 24 §4.6) - with no template kwargs (its template always reasons)."""
    tok, out = run("lfm2_moe")
    assert len(tok.calls) == 2
    for c in tok.calls:
        assert c["kwargs"] == {} and keys_sorted(c["msgs"]) and keys_sorted(c["tools"])
    assert "chat_template_kwargs" not in json.load(open(os.path.join(out, "t1_a.json"), encoding="utf-8"))


def _lfm_snapshot():
    import glob
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    for root in (hf, "/hf", os.path.expanduser("~/.cache/huggingface")):
        for c in sorted(glob.glob(os.path.join(root, "hub", "models--LiquidAI--LFM2.5-24B-A2B", "snapshots", "*"))):
            if os.path.exists(os.path.join(c, "chat_template.jinja")):
                return c
    return None


def test_muse_defaults_sorted_and_recorded():
    """Muse-Glimmer (spec 23a Task 5): reasoning_strength low + current_date, key-sorted, recorded per scenario."""
    tok, out = run("muse_glimmer")
    assert len(tok.calls) == 2
    for c in tok.calls:
        assert c["kwargs"] == make_set.MUSE_KWARGS == {"reasoning_strength": "low", "current_date": "2026-10-15"}
        assert keys_sorted(c["msgs"]) and keys_sorted(c["tools"])
        call = [m for m in c["msgs"] if m.get("tool_calls")][0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(call, dict)                 # objects: the template raises on a JSON string (T:28-30)
    sc = json.load(open(os.path.join(out, "t1_a.json"), encoding="utf-8"))
    assert sc["chat_template_kwargs"] == make_set.MUSE_KWARGS


def _muse_snapshot():
    import glob
    for hub in (os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"), "/hf/hub"):
        for c in sorted(glob.glob(os.path.join(hub, "models--meta-models--Muse-Glimmer-30B", "snapshots", "*"))):
            if os.path.exists(os.path.join(c, "chat_template.jinja")):
                return c
    return None


def test_lfm2moe_render_is_apply_chat_template():
    """A scenario re-rendered with LFM2's own template: the set's ids are apply_chat_template's, which are
    tokenizer(text)'s for its text (byte for byte); the history call is pythonic. SKIPs without transformers 5.19.0
    or the LFM2.5 snapshot's small files (the container with the qwen4exp site has both)."""
    snap = _lfm_snapshot()
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError:
        print("  SKIP: no transformers")
        return
    if snap is None or transformers.__version__ != "5.19.0":
        print(f"  SKIP: LFM2.5 snapshot {snap}, transformers {transformers.__version__}")
        return
    tok = AutoTokenizer.from_pretrained(snap)
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "golden", "toolcall")
    out = tempfile.TemporaryDirectory()
    _TMP.append(out)
    make_set.retokenise(snap, src, out.name, tok=tok, mtype="lfm2_moe")
    for name in ("t3_rename-linear_l0", "t6_usage-box"):
        sc = json.load(open(os.path.join(out.name, f"{name}.json"), encoding="utf-8"))
        msgs, tools = json.loads(json.dumps([sc["messages"], sc["tools"]], sort_keys=True))
        text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
        ids = [int(x) for x in open(os.path.join(out.name, f"{name}.ids"), encoding="utf-8").read().split()]
        assert ids == tok(text, add_special_tokens=False)["input_ids"], name
        assert text.startswith("<|startoftext|><|im_start|>system\n") and text.endswith("<|im_start|>assistant\n<think>")
        assert "<|tool_call_start|>[" in text and "List of tools: [" in text
        assert tok.decode(ids) == text
    print(f"  lfm2moe: 36 scenarios re-rendered; two checked byte for byte against apply_chat_template")


def test_muse_render():
    """With transformers 5.19.0 and the original's small files (the oracle image + the site; skipped on a bare host): two
    scenarios of the real A4 set re-rendered for Muse are byte-identical to apply_chat_template on their key-sorted
    messages with the recorded kwargs, hold ONE BOS (first) and no <|patch|>; the GPTQ repo's older template is refused."""
    snap = _muse_snapshot()
    try:
        import transformers
        ok = transformers.__version__ == "5.19.0"
    except ImportError:
        ok = False
    if not (snap and ok):
        print("  skip test_muse_render: needs transformers 5.19.0 and meta-models/Muse-Glimmer-30B's small files")
        return
    from transformers import AutoTokenizer
    here = os.path.dirname(os.path.abspath(__file__))
    src_set = os.path.join(here, "..", "..", "tests", "golden", "toolcall")
    src, out = tempfile.TemporaryDirectory(), tempfile.TemporaryDirectory()
    _TMP.extend([src, out])
    names = ["t1_define-linear_l0", "t4_comment-box"]
    for n in names:
        for ext in ("json", "ids"):
            with open(os.path.join(src_set, f"{n}.{ext}"), encoding="utf-8") as f:
                open(os.path.join(src.name, f"{n}.{ext}"), "w", encoding="utf-8").write(f.read())
    json.dump([{"name": n} for n in names], open(os.path.join(src.name, "manifest.json"), "w"))
    make_set.retokenise(snap, src.name, out.name)
    tok = AutoTokenizer.from_pretrained(snap)
    for n in names:
        sc = json.load(open(os.path.join(out.name, f"{n}.json"), encoding="utf-8"))
        msgs, tools = json.loads(json.dumps([sc["messages"], sc["tools"]], sort_keys=True))
        want = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True, return_dict=False,
                                       **sc["chat_template_kwargs"])
        got = [int(x) for x in open(os.path.join(out.name, f"{n}.ids"), encoding="utf-8").read().split()]
        assert got == list(want), n
        assert got.count(200000) == 1 and got[0] == 200000 and 200092 not in got, n
        text = tok.decode(got, skip_special_tokens=False)
        assert "<atem:invoke name=" in text and text.endswith("<|start|>assistant"), n
    gq = snap.replace("models--meta-models--Muse-Glimmer-30B", "models--Vishva007--Muse-Glimmer-30B-W4A16-AutoRound-GPTQ")
    import glob
    gq = sorted(glob.glob(os.path.join(os.path.dirname(gq), "*")))
    if gq and os.path.exists(os.path.join(gq[0], "chat_template.jinja")):
        try:
            make_set.check_muse_template(gq[0])
            raise AssertionError("the GPTQ repo's older template was accepted")
        except SystemExit as e:
            assert "cfc67e5f" in str(e)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"ok  {n}")
    print(f"{len(tests)} tests passed")
