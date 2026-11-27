#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""Checks for dflash_accept.py (spec 19a Task 3). CPU, tiny random drafters, seconds.

    test_dflash_accept.py [test_name ...]

- the batched drafter (B blocks at once, the head over all mask rows) equals
  dflash_ref.Drafter.draft_block anchor by anchor - ids, and the hidden rows - for DFlash 2 and
  DFlash (v1, incl. a full-attention and a causal layer), K = 1..7, with anchors before and past
  the sliding window and context positions >= the anchor present (Review Focus 2-4);
- top_k_fast == top_k_stable, ties across the k-th value included;
- the int8 RTN (loader/lm_head_int8.cc's rule) and the int8 head;
- the acceptance accounting: lengths, censoring where the text leaves the greedy path, the
  per-depth counts and E_K;
- a run over tap dumps written by dump_taps.save_dump, resumed.
"""
import argparse
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load("test_dflash_ref")          # its synthetic drafters (and dflash_ref as T.ref)
ref = T.ref
DA = _load("dflash_accept")
DA._load_deps()
DT = DA.dt


def _ctx(d, n, seed=1):
    taps = T.random_taps(d, n, seed)
    return d.context_kv(taps, torch.arange(n)), taps


def _check_batch_equals_reference(d, n_ctx, anchors_pos, label):
    ctx, _ = _ctx(d, n_ctx)
    g = torch.Generator().manual_seed(3)
    anchor_ids = torch.randint(0, T.VOCAB - 4, (len(anchors_pos),), generator=g).tolist()
    head = DA.Head(d.lm_head)
    worst = 0.0
    for K in range(1, d.cfg.max_k + 1):
        ps = torch.tensor(anchors_pos)
        h = DA.forward_blocks(d, anchor_ids, ps, K, ctx)
        ids = DA.draft_batch(d, head, anchor_ids, ps, K, ctx, {"full": None})["full"]
        for b, p in enumerate(anchors_pos):
            o = d.draft_block(anchor_ids[b], p, K, ctx)
            assert torch.equal(ids[b], o.ids), (label, K, p, ids[b].tolist(), o.ids.tolist())
            worst = max(worst, (h[b] - o.hidden).abs().max().item())
    assert worst < 1e-4, (label, worst)
    return worst


def test_batch_equals_draft_block():
    d2 = T.make(T.v2_dict(window=6))                               # 2 sliding non-causal layers
    w2 = _check_batch_equals_reference(d2, 20, [1, 2, 5, 9, 13, 16], "v2")
    d1 = T.make(T.v1_dict(window=4, is_causal=None))                # Ornith-like: causal sliding + full
    w1 = _check_batch_equals_reference(d1, 20, [0, 3, 7, 12, 15], "v1")
    print(f"batched drafts == draft_block per anchor (v2, v1 causal+full), K = 1..max; hidden within "
          f"{max(w1, w2):.1e}")


def test_top_k_fast():
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 6, (50, 40), generator=g).float()           # many ties
    x[3] = 1.0                                                       # all tied
    x[4, :] = float("-inf")
    x[4, 7] = 2.0
    for k in (1, 4, 16):
        i1, v1 = DA.top_k_fast(x, k)
        i2, v2 = ref.top_k_stable(x, k)
        assert torch.equal(i1, i2) and torch.equal(v1, v2), k
    print("top_k_fast == top_k_stable (ties across the boundary, -inf rows)")


def test_int8():
    g = torch.Generator().manual_seed(1)
    w = (torch.randn(37, 70, generator=g) * torch.rand(37, 1, generator=g) * 3).to(torch.bfloat16)
    w[5] = 0
    q, s = DA.rtn_int8_rows(w)
    assert q.dtype == torch.int8 and q.abs().max() <= 127 and s[5] == 0 and (q[5] == 0).all()
    err = (q.float() * s[:, None] - w.float()).abs()
    assert (err <= s[:, None] / 2 + 1e-6).all()
    assert (q.abs().amax(1)[s > 0] == 127).all()
    d = T.make(T.v2_dict())
    before = {n: v.clone() for n, v in d.w.items()}
    n = DA.int8_simulate(d)
    lin = [k for k in before if k.endswith(DA.LINEAR_SUFFIXES)]
    assert n == len(lin) and n == 1 + 2 * (4 + 3 + 2) + 1, n
    for k, v in d.w.items():
        if k in lin:
            assert not torch.equal(v, before[k]) and len(torch.unique(v[0])) <= 255
        else:
            assert torch.equal(v, before[k]), k
    h = torch.randn(3, T.HID, generator=g)
    hb, hq = DA.Head(d.lm_head), DA.Head(d.lm_head, int8=True, chunk=64)
    q, s = DA.rtn_int8_rows(d.lm_head)
    assert torch.allclose(hq.logits(h), h @ (q.float() * s[:, None]).t(), atol=1e-4)
    cos = torch.nn.functional.cosine_similarity(hq.logits(h), hb.logits(h), dim=-1).min()
    assert cos > 0.999, cos
    print(f"int8 RTN per row (s = amax/127, |err| <= s/2); {n} drafter linears quantised, the rest untouched; "
          f"int8 head cos {float(cos):.5f}")


def test_accept_accounting():
    #           0  1  2  3  4  5  6  7  8  9
    x = torch.tensor([9, 9, 1, 2, 3, 4, 5, 6, 7, 8])                   # text
    g = torch.tensor([9, 1, 2, 3, 4, 0, 6, 7, 8, 5])                   # greedy after t (g_from 0)
    # text agrees with greedy except x[6]=5 vs g[5]=0
    ps = torch.tensor([1, 2, 4, 5])
    drafts = torch.tensor([[2, 3, 4, 9],      # p=1: d0=2 vs g[1]=1, a miss at depth 0
                           [2, 3, 4, 0],      # p=2: g[2..5] = 2,3,4,0 -> all 4 accepted
                           [4, 0, 6, 7],      # p=4: g[4]=4 ok, g[5]=0 ok, then x[6]=5 != g[5]=0 -> censored
                           [0, 6, 7, 8]])     # p=5: g[5]=0 ok; depth 1 needs x[6]==g[5] -> censored
    length, cens, counted, accepted = DA.accept_rows(drafts, ps, x, g, 0)
    assert length.tolist() == [0, 4, 2, 1] and cens.tolist() == [False, False, True, True], (length, cens)
    assert counted == [4, 2, 1, 1] and accepted == [3, 2, 1, 1], (counted, accepted)
    # g_from offsets the greedy array
    l2, c2, _, _ = DA.accept_rows(drafts, ps, x, g[1:], 1)
    l3, c3, _, _ = DA.accept_rows(drafts, ps, x, g, 0)
    assert torch.equal(l2, l3) and torch.equal(c2, c3)
    e = DA.pool([{"4": {"rows": 4, "censored": 2, "hist": [1, 0, 0, 0, 1], "counted": counted,
                        "accepted": accepted}}], [4])["4"]
    assert abs(e["E"] - (1 + 0.75 + 0.75 + 0.75 + 0.75)) < 1e-9 and e["mean_len_plus1"] == 3.0
    assert DA.anchors_of(10, 20) == [10, 11, 12, 13]
    print("acceptance: lengths, censoring off the greedy path, per-depth counts, E_K, anchors")


def _fake_dump(path, name, d, n_prompt, n, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, T.VOCAB - 4, (n,), generator=g)
    taps = T.random_taps(d, n, seed)
    greedy = torch.cat([ids[1:], ids[:1]])                     # text = greedy everywhere
    res = {"taps": {i: t.to(torch.bfloat16) for i, t in taps.items()}, "greedy": greedy,
           "top_ids": greedy[:, None].repeat(1, 4), "top_logits": torch.zeros(n, 4), "lse": torch.zeros(n)}
    DT.save_dump(path, name, ids.tolist(), n_prompt, 0, res, {"vocab_used": 300})


def test_run_over_dumps():
    cd = T.v2_dict(window=6)
    cfg = ref.config_from_dict(cd)
    t, e, h = T.random_tensors(cfg, 2)
    with tempfile.TemporaryDirectory() as root:
        dd, td = T._write_checkpoint(root, cd, t, e, h)
        d = ref.load_drafter(dd, td)
        dumps = os.path.join(root, "dumps")
        os.makedirs(dumps)
        _fake_dump(os.path.join(dumps, "prose__a.taps.safetensors"), "prose/a", d, 8, 20, 1)
        _fake_dump(os.path.join(dumps, "code__b.taps.safetensors"), "code/b", d, 5, 15, 2)
        out = os.path.join(root, "out")
        ns = argparse.Namespace(dumps=dumps, out_dir=out, arms="bf16,int8", vocab_used=300,
                                k="1-7", batch=4, drafter=dd,
                                w4a16=None, target=td, sources=None, dry_run=False, candidates="", costs=None,
                                draft_cost=0.2)
        DA.cmd_run(ns)
        r = json.load(open(os.path.join(out, "accept.bf16.json")))
        assert sorted(r["variants"]) == ["bf16"] and sorted(r["variants"]["bf16"]) == ["code/b", "prose/a"]
        st = r["variants"]["bf16"]["prose/a"]
        assert st["7"]["rows"] == len(DA.anchors_of(8, 20)) == 6 and st["7"]["censored"] == 0
        assert sum(st["3"]["hist"]) == 6 and len(st["3"]["hist"]) == 4
        assert r["agree"]["prose/a"] == [12, 12]
        assert st["3"]["counted"][0] == 6
        mt = os.path.getmtime(os.path.join(out, "accept.bf16.json"))
        DA.cmd_run(ns)                                   # resumed: nothing left to do
        assert os.path.getmtime(os.path.join(out, "accept.bf16.json")) == mt
        assert os.path.isfile(os.path.join(out, "summary.md"))
        summ = DA.summarize([os.path.join(out, "accept.bf16.json"), os.path.join(out, "accept.int8.json")])
        assert set(summ) == {"bf16", "int8"} and "all" in summ["bf16"]
        assert summ["bf16"]["all"]["sources"] == 2
        ns.dry_run, ns.out_dir = True, os.path.join(root, "dry")
        DA.cmd_run(ns)
        assert not os.listdir(ns.out_dir)
    print("run: two dumps x bf16 / int8, JSON per arm, resumed, summary, dry run writes nothing")


def test_positive_alignment():
    """End to end with a known answer: a dump whose greedy ids ARE draft_block's K = 1 and K = 2
    drafts (taps from position 3, so the greedy array is offset) accepts every depth-0 draft
    (E_1 = 2) and, where the text follows the greedy path, K = 2 accepts both."""
    d = T.make(T.v2_dict(window=6))
    n, n_prompt, f = 22, 9, 3
    g = torch.Generator().manual_seed(4)
    ids = torch.randint(0, T.VOCAB - 4, (n,), generator=g)
    taps = {i: t[f:] for i, t in T.random_taps(d, n, 5).items()}
    ctx = d.context_kv(taps, torch.arange(f, n))
    greedy = torch.randint(0, T.VOCAB - 4, (n - f,), generator=g)
    for p in DA.anchors_of(n_prompt, n):
        greedy[p - f] = d.draft_block(int(ids[p]), p, 1, ctx).ids[0]
    res = {"taps": {i: t.to(torch.bfloat16) for i, t in taps.items()}, "greedy": greedy,
           "top_ids": greedy[:, None], "top_logits": torch.zeros(n - f, 1), "lse": torch.zeros(n - f)}
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "x__y.taps.safetensors")
        # bf16 taps in the file: the drafts above came from the float32 taps, so recompute
        DT.save_dump(path, "x/y", ids.tolist(), n_prompt, f, res, {})
        dump = DT.load_dump(path)
        ctx = d.context_kv({i: dump["taps"][i] for i in d.cfg.target_layer_ids}, torch.arange(f, n))
        for p in DA.anchors_of(n_prompt, n):
            dump["greedy"][p - f] = d.draft_block(int(ids[p]), p, 1, ctx).ids[0]
        head = DA.Head(d.lm_head)
        r, _ = DA.run_source(d, head, dump, [1], {"a": None}, batch=4)
        e = DA.pool([r["a"]], [1])["1"]
        assert e["E"] == 2.0 and e["alpha"] == [1.0] and e["hist"] == [0, e["rows"]], e
        # K = 2 on a text that IS the greedy path: ids[p + 1] = greedy[p] and greedy[p + 1] = d_1
        x = dump["ids"].clone()
        p = DA.anchors_of(n_prompt, n)[0]
        o = d.draft_block(int(x[p]), p, 2, ctx)
        x[p + 1] = o.ids[0]
        dump["ids"] = x
        dump["greedy"][p - f], dump["greedy"][p + 1 - f] = o.ids[0], o.ids[1]
        r, _ = DA.run_source(d, head, dump, [2], {"a": None}, batch=4)
        st = r["a"]["2"]
        assert st["accepted"][1] >= 1 and st["hist"][2] >= 1, st
    print("alignment: greedy[p] = draft_block's d_0 -> E_1 = 2 exactly (taps from 3); a greedy-path text accepts K = 2")


def test_inkling_whole_block():
    """Inkling's form in the batch: one K = 7 block an anchor equals Drafter.draft_block (embed_norm, the softcap
    after the correction, the live rows), the 2-bit candidate variant drafts live ids, and sampled_and_gtau's
    arithmetic on a target that puts all its mass on the drafts."""
    cd = T.v2_dict(window=6)
    cd["dflash_config"] = dict(cd["dflash_config"], final_logit_softcapping=3.0,
                               selector_softcap_mode="after_correction")
    cfg = ref.config_from_dict(cd)
    t, e, h = T.random_tensors(cfg, 3)
    live = T.VOCAB - 40
    w = torch.ones(T.HID).to(torch.bfloat16) * 1.5
    d = ref.Drafter(cfg, t, e, h, embed_norm=(w, 1e-6), valid_vocab=live)
    ctx, _ = _ctx(d, 20)
    anchors_pos = [1, 8, 15]
    anchor_ids = [5, 77, 140]
    ps = torch.tensor(anchors_pos)
    head = DA.Head(d.lm_head, int8=True)
    out = DA.draft_batch(d, head, anchor_ids, ps, 7, ctx, {"full": None, "c": 64}, walk=True)
    ids, real, cand, chosen = out["full"]
    for b, p in enumerate(anchors_pos):
        hd = ref.Drafter(cfg, t, e, (head.q.float() * head.s[:, None]), embed_norm=(w, 1e-6), valid_vocab=live)
        o = hd.draft_block(anchor_ids[b], p, 7, ctx)
        assert torch.equal(ids[b], o.ids) and torch.equal(cand[b], o.cand_ids), (p, ids[b].tolist(), o.ids.tolist())
    assert int(out["c"][0].max()) < live
    # a target certain of every draft: point mass E_K = 1 + K; exact E from the selector's q of the drafts
    A = len(anchors_pos)
    n = 24
    f = 0
    rows = n - f
    top_ids = torch.full((rows, 1), -1, dtype=torch.long)
    for b, p in enumerate(anchors_pos):
        for j in range(7):
            if p + j < rows:
                top_ids[p + j - f, 0] = ids[b, j]
    dump = {"tap_from": f, "top_ids": top_ids, "top_logits": torch.zeros(rows, 1), "lse": torch.zeros(rows)}
    length = torch.full((A,), 7)
    x = DA.sampled_and_gtau(dump, (ids, real, cand, chosen), ps, length, torch.zeros(A, dtype=torch.bool),
                            [0.0] + [1.0 + 0.1 * k for k in range(8)], 0.2)
    assert [round(v / A, 6) for v in x["pm"]] == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], x["pm"]
    q = torch.softmax(real, -1).gather(-1, chosen[..., None])[..., 0]
    assert abs(x["exact"][-1] - float((1 + torch.cumprod(q, 1).sum(1)).sum())) < 1e-6
    an = x["anchors"]
    assert an["length"] == [7] * A and an["censored"] == [False] * A
    assert all(abs(a - b) < 1e-4 for r, t in zip(an["q"], q.tolist()) for a, b in zip(r, t)), "per-anchor q"
    g = x["gtau"]["0.5"]
    assert g["ids"] <= 8 * A and g["rows"] <= 8 * A and g["given_up"] >= 0
    print("Inkling: one K = 7 block an anchor == draft_block, 2-bit candidates live, T 1.0 / GTAU arithmetic")


TESTS = [test_top_k_fast, test_int8, test_accept_accounting,
         test_batch_equals_draft_block, test_positive_alignment, test_run_over_dumps,
         test_inkling_whole_block]


def main() -> None:
    torch.set_grad_enabled(False)
    only = sys.argv[1:]
    for t in TESTS:
        if not only or t.__name__ in only:
            t()
    print("ok")


if __name__ == "__main__":
    main()
