#!/usr/bin/env python3
"""Checks for qwen_forced.py. CPU, test_dump_taps.py's tiny random Qwen3.5 checkpoint, seconds.

    test_qwen_forced.py [test_name ...]

The trie forward against transformers' own forward of each context in float32 (the same weights upcast): contexts
that share prefixes, branch inside a GDN chunk and inside an attention query block, two roots, and one context twice
under another key; then `run` end to end through the layer-streamed build - the files' shape and rows against the
reference's logits through the bf16 head and the int8 head, the hard link, the key check and the resume. The l0-int8
prompt pass (--pfi8-re) on a 1024-wide tiny model (int8_sim's rotation is 1024-k blocks): the trie forward against a
per-context forward whose linears take int8_sim.fq_x / fq_w on the test prompt's rows only (ornith_ref.py
--sim-int8-prefill's style), with contexts where one position is prompt in one key and decode in another.
"""
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import glob  # noqa: E402
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
    assert not glob.glob(os.path.join(work, "**", "hidden.pt"), recursive=True), "the checkpoints are removed"
    # a second run finds every judge and does nothing
    QF.main()
    # Force several memory groups and chunked singletons through the real streamed run/output path.
    cap = QF.estimate_peak(214, 150, TD.TEXT, chunk_rows=64, output_rows=81)
    groups = QF.memory_plan(ctx, TD.TEXT, cap, chunk_rows=64)
    assert len(groups) > 1 and any(g[2] for g in groups)
    out_group, out8_group = out + "-group", out8 + "-group"
    sys.argv = ["qwen_forced.py", "run", snap, "--ctx", cdir, "--out", out_group, "--out-i8head", out8_group,
                "--work", work, "--vocab-used", str(TD.VOCAB_USED), "--chunk-rows", "64",
                "--memory-gib", str(cap / 2**30)]
    QF.main()
    for k in ctx:
        old, new = (out8, out8_group) if k.startswith("dflash") else (out, out_group)
        with safe_open(os.path.join(old, k + ".golden.safetensors"), framework="pt") as f:
            want = f.get_tensor("logits")
        with safe_open(os.path.join(new, k + ".golden.safetensors"), framework="pt") as f:
            got = f.get_tensor("logits")
        err = ((got - want).abs().max() / want.abs().max()).item()
        assert err < 2e-5, (k, err)
    assert not glob.glob(os.path.join(work, "**", "hidden.pt"), recursive=True)
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


# ------------------------------------------------------------------------------------- the l0-int8 prompt pass

TEXT8 = dict(TD.TEXT, hidden_size=1024, intermediate_size=1024, num_hidden_layers=4, num_attention_heads=4,
             num_key_value_heads=2, head_dim=256, linear_num_key_heads=2, linear_num_value_heads=4,
             linear_key_head_dim=64, linear_value_head_dim=256, layer_types=["linear_attention"] * 3 +
             ["full_attention"],
             rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                              "mrope_section": [16, 8, 8], "mrope_interleaved": True})


def wide_model():
    """A 4-layer (3 GDN + 1 FA) Qwen3.5 whose every h8 linear has K a multiple of 1024, f32, random weights."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    tc = Qwen3_5TextConfig(**TEXT8)
    tc._attn_implementation = "eager"
    torch.manual_seed(1)
    m = Qwen3_5ForCausalLM(tc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 2 and "embed" not in n:
                p.copy_(torch.randn_like(p) / p.shape[1] ** 0.5)
    return m


H8 = {"linear_attn": ("in_proj_qkv", "in_proj_z", "out_proj"), "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
      "mlp": ("gate_proj", "up_proj", "down_proj")}


def ref_hidden_pfi8(m, ids, P, i8):
    """The per-context reference: transformers' f32 forward of `ids` with every h8 linear patched - rows t < P
    through i8.fq_x(x) . i8.fq_w(w), the rest the plain linear (ornith_ref.sim_int8_prefill's rule, by row)."""
    import torch.nn.functional as F
    saved = []
    for lay in m.model.layers:
        for part, names in H8.items():
            if not hasattr(lay, part):
                continue
            for n in names:
                mod = getattr(getattr(lay, part), n)
                orig = mod.forward

                def fwd(x, mod=mod, orig=orig):
                    out = orig(x).clone()
                    if P:
                        out[0, :P] = F.linear(i8.fq_x(x[0, :P]), i8.fq_w(mod.weight), mod.bias)
                    return out
                mod.forward = fwd
                saved.append((mod, orig))
    try:
        return ref_hidden(m, ids)
    finally:
        for mod, orig in saved:
            mod.forward = orig


def pfi8_case():
    """A temp tree with the tests' prompt files, the contexts, and the plan: one position is prompt in one key and
    decode in another (pb's 80..119 against t1_pa's), one key has no int8 pass (dflash), one is another root."""
    base = TD.seq(150, 21)
    root = tempfile.mkdtemp(prefix="qwen_forced_pfi8_")
    os.makedirs(os.path.join(root, "tests/golden/toolcall"))
    os.makedirs(os.path.join(root, "tests/golden/prompts"))
    for d, name, ids in (("toolcall", "t1_pa", base[:80]), ("prompts", "pb", base[:120]),
                         ("toolcall", "t2_pc", TD.seq(40, 22))):
        with open(os.path.join(root, "tests/golden", d, name + ".ids"), "w") as fh:
            fh.write(" ".join(map(str, ids)) + "\n")
    ctx = keyed({
        "mtp_gpu_bf16_t1_pa": (base[:100], base[100:130]),               # 80 prompt-pass ids, 20 decoded, then forced
        "mtp_gpu_int8_t1_pa": (base[:90], base[90:100] + TD.seq(12, 23)),  # the same prompt pass, branches at 100
        "mtp_gpu_rf1_bf16_pb": (base[:125], base[125:140]),              # 80..119: prompt here, decode in t1_pa's
        "dflash_gpu_int8_c": (base[:70], base[70:130]),                  # t1_pa's ids, no int8 pass at all
        "mtp_gpu_bf16_t2_pc": (TD.seq(40, 22) + TD.seq(6, 24), TD.seq(8, 25)),   # another root
    })
    trie, jobs = QF.plan(ctx, pfi8_re=QF.PFI8_RE, root=root)
    return base, ctx, trie, jobs


def pfi8_errors(m, ctx, trie, jobs, i8):
    """Per key: the trie forward's relative max error against its per-context reference, and (keys with the pass)
    the smaller of its distances to the same context without the pass and with it over every position."""
    with torch.no_grad():
        final = QF.trie_forward(m, trie, log=lambda s: None)
    out = {}
    for k, (p, f) in ctx.items():
        P = jobs[k]["pfi8"]
        got = final[torch.tensor(trie.path(p + f, P))]
        rel = []
        for pp in ([P, 0, len(p + f)] if P else [P]):
            want = ref_hidden_pfi8(m, p + f, pp, i8)
            rel.append(((got - want).abs().max() / want.abs().max()).item())
        out[k] = (rel[0], min(rel[1:]) if P else None)
    return out


def test_pfi8():
    base, ctx, trie, jobs = pfi8_case()
    want_p = {"t1_pa": 80, "pb": 120, "dflash": 0, "t2_pc": 40}
    for k, j in jobs.items():
        assert j["pfi8"] == next(v for n, v in want_p.items() if n in k), (k, j["pfi8"])
    a = trie.path(ctx[[k for k in ctx if "bf16_t1_pa" in k][0]][0], 80)
    b = trie.path(base[:125], 120)
    c = trie.path(base[:70], 0)
    assert a[:80] == b[:80] and a[80] != b[80], "prompt in one key, decode in another: split at the first such position"
    assert not set(a) & set(c), "a key without the int8 pass shares no position with one that has it"
    assert sum(trie.i8) == len(set(b[:120]) | set(trie.path(TD.seq(40, 22), 40)))
    m = wide_model()
    i8 = QF._load("int8_sim")
    # 1. The plumbing, exactly: a smooth stand-in for fq_x / fq_w (row-local and weight-dependent like them, but with
    # no rounding boundary, so the forwards' f32 summation order cannot flip a code) - every key at f32 accuracy, far
    # from both wrong prompt passes.
    fq = (i8.fq_x, i8.fq_w)
    i8.fq_x = lambda x, moe=False: x + 0.25 * torch.tanh(3 * x)
    i8.fq_w = lambda w, moe=False: w * (1 + 0.25 * torch.cos(40 * w))
    try:
        err = pfi8_errors(m, ctx, trie, jobs, i8)
    finally:
        i8.fq_x, i8.fq_w = fq
    worst = max(e for e, _ in err.values())
    near = min(w for _, w in err.values() if w is not None)
    assert worst < 5e-5, err
    assert near > 1e-2, err
    # 2. int8_sim's own fq_x / fq_w: the same, up to the codes that the two forwards' f32 summation orders put on
    # opposite sides of a rounding boundary (one such code moves its row by ~1/127 of its largest entry).
    err8 = pfi8_errors(m, ctx, trie, jobs, i8)
    for k, (e, w) in err8.items():
        print(f"  {k}: int8 prompt pass {jobs[k]['pfi8']}: {e:.1e} from its reference" +
              (f", {w:.1e} from the nearest wrong prompt pass" if w is not None else ""))
    worst8 = max(e for e, _ in err8.values())
    assert worst8 < 5e-2, err8
    print(f"the l0-int8 prompt pass: trie forward == the per-context forward with fq_x / fq_w on the test prompt's rows "
          f"(smooth stand-in: worst {worst:.1e}, the wrong prompt passes >= {near:.1e} away; int8_sim's own: worst "
          f"{worst8:.1e}, boundary codes); a position is shared only by keys that computed it the same way")


def test_memory_plan():
    ctx = keyed(contexts())
    cap = QF.estimate_peak(160, 160, TD.TEXT)
    groups = QF.memory_plan(ctx, TD.TEXT, cap, chunk_rows=64)
    assert len(groups) > 1
    assert {k for g in groups for k in g[1]} == set(ctx)
    assert all(g[3] <= cap for g in groups)
    long = keyed({"mtp_verify_long": (TD.seq(31742, 31), [1, 2]),
                  "mtp_verify_longer": (TD.seq(130046, 32), [3, 4])})
    groups = QF.memory_plan(long, QF.DEFAULT_CONFIG, 64 * 2**30, chunk_rows=256)
    assert len(groups) == 2 and all(len(g[1]) == 1 for g in groups)
    assert all(g[3] <= 64 * 2**30 for g in groups)
    assert any(g[2] for g in groups), "the 130k context needs chunking"
    try:
        QF.memory_plan(long, TD.TEXT, 1, chunk_rows=64)
        raise AssertionError("an impossible memory cap was accepted")
    except ValueError as e:
        assert "memory cap" in str(e)


def test_branch_state_memory():
    # Deepest path first: DFS descends the spine before visiting its side leaves, so GDN
    # must retain every unfinished ancestor's recurrent state at once.
    ctx = {f"mtp_verify_comb_{d:02}": ([1] * (12 - d) + [2], [3]) for d in range(12)}
    trie, _ = QF.plan(ctx)
    cap = QF.estimate_peak(len(trie.tok), 14, TD.TEXT)
    groups = QF.memory_plan(ctx, TD.TEXT, cap)
    assert len(groups) > 1, "a comb's retained recurrent states must count against the cap"
    assert all(g[3] <= cap for g in groups)
    assert {k for g in groups for k in g[1]} == set(ctx)
    print("memory plan: unfinished GDN branch states are included in each capped group")


def test_chunked_forward():
    ctx = keyed({"mtp_verify_a": (TD.seq(150, 11), TD.seq(7, 12))})
    trie, _ = QF.plan(ctx)
    m = ref_model()
    with torch.no_grad():
        full = QF.trie_forward(m, trie, log=lambda s: None)
        for chunk in (64, 73):
            got = QF.trie_forward(m, trie, chunk_rows=chunk, log=lambda s: None)
            err = ((got - full).abs().max() / full.abs().max()).item()
            assert err < 2e-5, (chunk, err)
    # The stream/checkpoint path must use the same chunked computation on resume.
    work = tempfile.mkdtemp(prefix="qwen_forced_chunk_resume_")
    layers = m.model.layers
    m.model.layers = layers[:5]
    with torch.no_grad():
        QF.trie_forward(m, trie, chunk_rows=64, work=work, log=lambda s: None)
        m.model.layers = layers
        got = QF.trie_forward(m, trie, chunk_rows=64, work=work, log=lambda s: None)
    err = ((got - full).abs().max() / full.abs().max()).item()
    assert err < 2e-5, err
    print("chunked: f32 hidden states match one-shot, including resume and non-aligned chunks")


TESTS = [test_trie, test_forward_matches_transformers, test_run_end_to_end, test_resume, test_pfi8, test_memory_plan, test_branch_state_memory, test_chunked_forward]

if __name__ == "__main__":
    want = set(sys.argv[1:])
    torch.manual_seed(0)
    for t in TESTS:
        if not want or t.__name__ in want:
            t()
    print("test_qwen_forced: OK")
