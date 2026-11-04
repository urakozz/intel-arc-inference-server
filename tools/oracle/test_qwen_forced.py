#!/usr/bin/env python3
"""Checks for qwen_forced.py. CPU, test_dump_taps.py's tiny random Qwen3.5 checkpoint, seconds.

    test_qwen_forced.py [test_name ...]

The trie forward against transformers' own forward of each context in float32 (the same weights upcast): contexts
that share prefixes, branch inside a GDN chunk and inside an attention query block, two roots, and one context twice
under another key; then `run` end to end through the layer-streamed build - the files' shape and rows against the
reference's logits through the bf16 head and the int8 head, the hard link, the key check and the resume.
"""
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import copy  # noqa: E402
import importlib.util  # noqa: E402

import torch  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TD = _load("test_dump_taps")
QF = _load("qwen_forced")
QF.ROW_BLOCK, QF.Q_BLOCK, QF.HEAD_ROWS = 37, 16, 50     # every blocking path, on a small trie


def contexts():
    """{name: (prompt, force)}: a 150-id base continued four ways, a short prompt, a second root, a duplicate."""
    base = TD.seq(150, 11)
    return {
        "mtp_gpu_bf16_a": (base[:100], base[100:150]),
        "mtp_gpu_bf16_b": (base[:100], base[100:103] + TD.seq(20, 12)),      # branches 3 into the forced ids
        "dflash_gpu_int8_c": (base[:70], base[70:140]),                       # branches inside the prompt
        "dflash_gpu_int8_d": (base[:100], TD.seq(9, 13)),
        "mtp_verify_e": (TD.seq(30, 14), TD.seq(5, 15)),                      # another root
        "dflash_lookup_gpu_int8_c": (base[:70], base[70:140]),                # = c's ids under another key
        "mtp_gpu_bf16_f": (base[:1], base[1:6]),                              # a one-id prompt
        "dflash_gpu_int8_g": (base[:60], base[60:140]),                       # c's ids split 10 earlier
    }


def keyed(ctx):
    return {f"{n}_{QF.fnv_key_hash(p, f)}": (p, f) for n, (p, f) in ctx.items()}


def ref_model():
    _, model = TD.tiny_checkpoint()
    m = copy.deepcopy(model).float().eval()
    m.model.rotary_emb = type(m.model.rotary_emb)(m.config)
    return m


def ref_hidden(m, ids):
    with torch.no_grad():
        return m.model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0]


def test_trie():
    ctx = keyed(contexts())
    trie, jobs = QF.plan(ctx)
    seqs = {tuple(p + f) for p, f in ctx.values()}
    # every distinct prefix is one node
    prefixes = {s[:i + 1] for s in seqs for i in range(len(s))}
    assert len(trie.tok) == len(prefixes), (len(trie.tok), len(prefixes))
    for k, (p, f) in ctx.items():
        path = trie.path(p + f)
        assert [trie.tok[n] for n in path] == p + f
        assert [trie.depth[n] for n in path] == list(range(len(p + f)))
        assert all(trie.parent[b] == a for a, b in zip(path, path[1:]))
        assert jobs[k]["rows"] == path[len(p) - 1:]
        assert jobs[k]["form"] == ("int8" if k.startswith("dflash") else "bf16")
    for s, ps in enumerate(trie.seg_parent):
        a, L = trie.seg_start[s], trie.seg_len[s]
        assert all(trie.parent[i] == i - 1 for i in range(a + 1, a + L)), "a segment is a chain"
        if ps >= 0:
            assert trie.parent[a] == trie.seg_start[ps] + trie.seg_len[ps] - 1 and ps < s
    print(f"trie: {len(trie.tok)} positions, {len(trie.seg_start)} segments, paths / depths / rows consistent")


def test_forward_matches_transformers():
    ctx = keyed(contexts())
    trie, _ = QF.plan(ctx)
    m = ref_model()
    with torch.no_grad():
        final = QF.trie_forward(m, trie, log=lambda s: None)
    worst = 0.0
    for k, (p, f) in ctx.items():
        want = ref_hidden(m, p + f)
        got = final[torch.tensor(trie.path(p + f))]
        err = ((got - want).abs().max() / want.abs().max()).item()
        worst = max(worst, err)
        assert err < 2e-5, f"{k}: relative max error {err:.2e}"
    print(f"trie forward == transformers' f32 forward of each context (worst relative error {worst:.1e})")


def test_run_end_to_end():
    snap, _ = TD.tiny_checkpoint()
    m = ref_model()
    ctx = keyed(contexts())
    root = tempfile.mkdtemp(prefix="qwen_forced_test_")
    cdir, out, out8, work = (os.path.join(root, d) for d in ("ctx", "out", "out8", "work"))
    os.makedirs(cdir)
    for k, (p, f) in ctx.items():
        for ext, v in ((".prompt.ids", p), (".force.ids", f)):
            with open(os.path.join(cdir, k + ext), "w") as fh:
                fh.write(" ".join(map(str, v)) + "\n")
    argv = ["run", snap, "--ctx", cdir, "--out", out, "--out-i8head", out8, "--work", work, "--vocab-used",
            str(TD.VOCAB_USED)]
    sys.argv = ["qwen_forced.py"] + argv
    QF.main()
    from safetensors import safe_open
    i8 = _load("int8_sim")
    w = m.lm_head.weight.detach()
    heads = {"bf16": w.float(), "int8": i8.int8_head_weight(w)}
    for k, (p, f) in ctx.items():
        form = "int8" if k.startswith("dflash") else "bf16"
        path = os.path.join(out8 if form == "int8" else out, k + ".golden.safetensors")
        with safe_open(path, framework="pt") as fh:
            lg, tk, md = fh.get_tensor("logits"), fh.get_tensor("tokens"), fh.metadata()
        assert lg.shape == (len(f) + 1, TD.VOCAB) and lg.dtype == torch.float32
        assert tk.tolist() == f and md["lm_head"] == form and md["gen"] == str(len(f))
        want = ref_hidden(m, p + f)[len(p) - 1:] @ heads[form].t()
        err = ((lg - want).abs().max() / want.abs().max()).item()
        assert err < 2e-5, f"{k}: logits relative error {err:.2e}"
    c1 = os.path.join(out8, [k for k in ctx if k.startswith("dflash_gpu_int8_c")][0] + ".golden.safetensors")
    c2 = os.path.join(out8, [k for k in ctx if k.startswith("dflash_lookup")][0] + ".golden.safetensors")
    assert os.stat(c1).st_ino == os.stat(c2).st_ino, "one context's ids under two keys: one file"
    c3 = os.path.join(out8, [k for k in ctx if k.startswith("dflash_gpu_int8_g")][0] + ".golden.safetensors")
    assert os.stat(c1).st_ino != os.stat(c3).st_ino, "the same ids at another prompt length: another file"
    assert not os.path.exists(os.path.join(work, "hidden.pt")), "the checkpoint is removed once done"
    # a second run finds every judge and does nothing
    QF.main()
    # a key whose ids do not hash to it is refused
    bad = os.path.join(cdir, "mtp_gpu_bf16_x_0123456789abcdef")
    for ext in (".prompt.ids", ".force.ids"):
        with open(bad + ext, "w") as fh:
            fh.write("1 2 3\n")
    try:
        QF.load_contexts([cdir])
        raise AssertionError("a mis-keyed context was accepted")
    except SystemExit as e:
        assert "hash" in str(e)
    print("run: every judge's rows == the reference's logits through its head (bf16 / int8), the duplicate a hard "
          "link, a re-run a no-op, a mis-keyed context refused")


def test_resume():
    """A checkpoint from a killed run (layer 5 of 8 done) gives the same final hidden state."""
    ctx = keyed(contexts())
    trie, _ = QF.plan(ctx)
    m = ref_model()
    with torch.no_grad():
        full = QF.trie_forward(m, trie, log=lambda s: None)
    work = tempfile.mkdtemp(prefix="qwen_forced_resume_")
    layers = m.model.layers
    m.model.layers = layers[:5]          # "killed" after layer 4: the checkpoint says next_layer 5
    with torch.no_grad():
        QF.trie_forward(m, trie, work=work, log=lambda s: None)
    m.model.layers = layers
    msgs = []
    with torch.no_grad():
        again = QF.trie_forward(m, trie, work=work, log=msgs.append)
    assert any("resumed at layer 5" in s for s in msgs), msgs[:2]
    assert torch.equal(full, again)
    print("resume: a checkpoint at layer 5 continues to the same final hidden state (bitwise)")


TESTS = [test_trie, test_forward_matches_transformers, test_run_end_to_end, test_resume]

if __name__ == "__main__":
    want = set(sys.argv[1:])
    torch.manual_seed(0)
    for t in TESTS:
        if not want or t.__name__ in want:
            t()
    print("test_qwen_forced: OK")
