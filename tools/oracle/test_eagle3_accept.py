#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for eagle3_accept.py (the EAGLE3 K2 P0). CPU, tiny random drafters, seconds.

    test_eagle3_accept.py [test_name ...]

- run_source's drafts == eagle3_ref.Drafter.propose anchor by anchor, and its counts == a
  brute-force re-count of the protocol (accepted prefix, censoring where the text leaves the
  greedy path);
- the alignment, positively: a text that IS the drafter's own chain at an anchor accepts all
  K = 5 there (row p = token x[p] + the aux of p - 1; d_j is compared with g[p + j]);
- the prompt-lookup replay == tools/spec/lookup_accept.py's precompute on a recorded = greedy
  text (same matcher, same accepted counts), and its drafted-count histogram;
- the end to end CLI over k2_taps-format dumps: run (two arms), resume, lookup, eagle3_cost.py
  run, summary (E_K, a_j, S_K tables).
"""
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402


def _load(name, path=None):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path or os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TE = _load("test_eagle3_ref")          # tiny drafters
E3 = TE.E3
EA = _load("eagle3_accept")
torch_, _, KT, DA = EA.deps()
LA = EA.lookup_mod()


def synth_dump(c, n, n_prompt, seed=0, keep_ctx=8, ids=None, greedy=None, name="golden/x"):
    """A k2_taps.load_dump-shaped dict with random aux / routes (no K2 needed)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, TE.VOCAB, (n,), generator=g) if ids is None else torch.as_tensor(ids)
    aux_from = max(0, n_prompt - keep_ctx - 1)
    hf = n_prompt - 1
    if greedy is None:
        greedy = torch.randint(0, TE.VOCAB, (n - hf,), generator=g)
        greedy[:-1] = torch.where(torch.rand(n - hf - 1, generator=g) < 0.7, ids[hf + 1:], greedy[:-1])
    R = n - hf
    return {"name": name, "n_prompt": n_prompt, "n": n, "aux_from": aux_from, "head_from": hf,
            "aux_ids": c.aux_ids, "ids": ids.long(),
            "aux": {a: (2 * torch.randn(n - aux_from, c.target_hidden, generator=g)).to(torch.bfloat16) for a in c.aux_ids},
            "greedy": torch.as_tensor(greedy).long(), "greedy_flatnorm": torch.as_tensor(greedy).long().clone(),
            "final": torch.zeros(R, c.target_hidden, dtype=torch.bfloat16),
            "top_ids": torch.zeros(R, 4, dtype=torch.long), "top_logits": torch.zeros(R, 4), "lse": torch.zeros(R),
            "moe_ids": torch.stack([torch.randperm(100, generator=g)[:8].sort().values for _ in range(R * 45)]).view(R, 45, 8).to(torch.uint8),
            "mova_ids": torch.stack([torch.randperm(64, generator=g)[:4].sort().values for _ in range(R * 45)]).view(R, 45, 4).to(torch.uint8)}


def brute_count(drafts, ps, x, g, hf, K):
    """The protocol, re-counted plainly: per anchor the accepted prefix and the censor point."""
    counted, accepted, hist, cens = [0] * K, [0] * K, [0] * (K + 1), 0
    N = len(x)
    for a, p in enumerate(ps):
        n, censored = 0, False
        for j in range(K):
            if p + j > N - 1 or (j > 0 and int(x[p + j]) != int(g[p + j - 1 - hf])):
                censored = True
                break
            counted[j] += 1
            if int(drafts[a][j]) != int(g[p + j - hf]):
                break
            accepted[j] += 1
            n += 1
        if censored:
            cens += 1
        else:
            hist[n] += 1
    return counted, accepted, hist, cens


def test_run_source_equals_reference():
    _, c, t = TE.make(seed=31, window=6)
    d = E3.Drafter(c, t)
    dump = synth_dump(c, 30, 14, seed=31)
    ctx = d.context(dump["ids"], KT.aux_cat(dump, c.aux_ids), dump["aux_from"])
    anchors = EA.anchors_of(14, 30)
    drafts = torch.stack([d.propose(ctx, p, 5) for p in anchors])
    # make K2's greedy take some drafts: d_0 / d_1 accepted at every third anchor, the text not
    # following (censored at depth 1 or 2 there)
    hf = dump["head_from"]
    for i in range(0, len(anchors), 3):
        p = anchors[i]
        dump["greedy"][p - hf] = drafts[i, 0]
        if i % 2 == 0:
            dump["ids"][p + 1] = drafts[i, 0]
            dump["greedy"][p + 1 - hf] = drafts[i, 1]
    res = EA.run_source(d, dump, [1, 2, 3, 4, 5], batch=3, window="query", DA=DA)
    ctx = d.context(dump["ids"], KT.aux_cat(dump, c.aux_ids), dump["aux_from"])
    drafts = torch.stack([d.propose(ctx, p, 5) for p in anchors])
    for K in range(1, 6):
        cnt, acc, hist, cens = brute_count(drafts[:, :K].tolist(), anchors, dump["ids"], dump["greedy"], 13, K)
        r = res[str(K)]
        assert (r["counted"], r["accepted"], r["hist"], r["censored"]) == (cnt, acc, hist, cens), (K, r, cnt, acc, hist, cens)
    assert res["5"]["rows"] == len(anchors) and res["5"]["censored"] > 0 and res["2"]["accepted"][0] > 0, res["5"]
    print(f"  run_source == propose per anchor; counts == the brute-force protocol (K 1..5, {len(anchors)} anchors, "
          f"{res['5']['censored']} censored at K = 5)")


def test_positive_alignment():
    _, c, t = TE.make(seed=32, window=6)
    d = E3.Drafter(c, t)
    n_prompt, n = 12, 12 + 6
    dump = synth_dump(c, n, n_prompt, seed=32)
    p = n_prompt
    ctx = d.context(dump["ids"], KT.aux_cat(dump, c.aux_ids), dump["aux_from"])
    chain = d.propose(ctx, p, 5)
    x, g = dump["ids"].clone(), dump["greedy"].clone()
    hf = dump["head_from"]
    x[p + 1: p + 6] = chain                       # the text continues with the drafts ...
    g[p - hf: p + 5 - hf] = chain                 # ... and K2's greedy is that text
    g[p - 1 - hf] = x[p]
    dump["ids"], dump["greedy"] = x, g
    res = EA.run_source(d, dump, [1, 2, 3, 4, 5], batch=4, window="query", DA=DA)
    pooled = DA.pool([res], [5])["5"]
    assert res["5"]["accepted"][4] >= 1 and res["5"]["hist"][5] >= 1, res["5"]
    # off by one the other way (g[p + j] = d_{j+1}) accepts nothing at p
    g2 = g.clone()
    g2[p - hf: p + 4 - hf] = chain[1:]
    dump["greedy"] = g2
    r2 = EA.run_source(d, dump, [5], batch=4, window="query", DA=DA)
    assert r2["5"]["hist"][5] == 0
    print(f"  a text that IS the chain at p accepts 5 of 5 there (E_5 over the {res['5']['rows']} anchors "
          f"{pooled['E']:.2f}); shifted by one: no full accept")


def test_lookup_matches_lookup_accept():
    _, c, _ = TE.make(seed=33)
    base = [5, 6, 7, 8, 9, 10, 11]
    ids = [1, 2] + base + [3, 4] + base + [12] + base[:5] + [13, 14, 15, 16, 17, 18]
    n_prompt = 9
    x = torch.tensor(ids)
    g = torch.cat([x[n_prompt:], torch.tensor([0])])           # recorded == greedy, last one free
    dump = synth_dump(c, len(ids), n_prompt, seed=33, ids=x, greedy=g)
    res = EA.lookup_source(dump, [1, 2, 3, 4, 5], min_match=2, max_match=64, DA=DA)
    seqd = {"prompt_ids": ids[:n_prompt], "out_ids": ids[n_prompt:]}
    pre = LA.precompute(seqd, 64, 5)
    anchors = EA.anchors_of(n_prompt, len(ids))
    for K in (1, 3, 5):
        want = [0] * (K + 1)
        for p in anchors:
            _, a_cnt, avail = pre[p - n_prompt]
            want[min(a_cnt, avail, K)] += 1
        r = res[str(K)]
        assert r["censored"] == 0 and r["hist"] == want, (K, r["hist"], want)
        assert sum(r["drafted_hist"]) == len(anchors)
    nd = [min(pre[p - n_prompt][2], 5) for p in anchors]
    assert res["5"]["drafted_hist"] == [nd.count(k) for k in range(6)]
    print(f"  lookup replay == lookup_accept.precompute's accepted counts (K 1/3/5); drafted hist {res['5']['drafted_hist']}")


def _write_drafter(dirname, d, t):
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in t.items()}, os.path.join(dirname, "model.safetensors"))
    with open(os.path.join(dirname, "config.json"), "w") as f:
        json.dump(d, f)


def test_cli_end_to_end():
    dd, c, t = TE.make(seed=34, window=6)
    with tempfile.TemporaryDirectory() as tmp:
        snap = os.path.join(tmp, "drafter")
        dumps = os.path.join(tmp, "dumps")
        out = os.path.join(tmp, "accept")
        os.makedirs(snap)
        os.makedirs(dumps)
        _write_drafter(snap, dd, t)
        for i, (name, n, n_p) in enumerate((("golden/prose", 26, 10), ("a4/t1", 30, 16), ("a4/t2", 22, 9))):
            dm = synth_dump(c, n, n_p, seed=40 + i, name=name)
            tens = {"ids": dm["ids"].to(torch.int32), "greedy": dm["greedy"].to(torch.int32),
                    "greedy_flatnorm": dm["greedy_flatnorm"].to(torch.int32), "final": dm["final"],
                    "top_ids": dm["top_ids"].to(torch.int32), "top_logits": dm["top_logits"], "lse": dm["lse"],
                    "moe_ids": dm["moe_ids"], "mova_ids": dm["mova_ids"]}
            tens.update({f"aux.{a}": v for a, v in dm["aux"].items()})
            KT.save_dump(os.path.join(dumps, KT.file_name(name)), name, tens,
                         {"n_prompt": n_p, "n": n, "aux_from": dm["aux_from"], "head_from": dm["head_from"],
                          "aux_ids": ",".join(map(str, c.aux_ids)), "sparse_layers": "x"})
        py = [sys.executable]
        acc = py + [os.path.join(_HERE, "eagle3_accept.py")]
        r = subprocess.run(acc + ["run", "--dumps", dumps, "--out-dir", out, "--arms", "bf16,int8h", "--drafter", snap,
                                  "--batch", "5"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        st = json.load(open(os.path.join(out, "accept.int8h.json")))
        assert sorted(st["sources"]) == ["a4/t1", "a4/t2", "golden/prose"] and st["meta"]["quantised_linears"] == 9
        again = subprocess.run(acc + ["run", "--dumps", dumps, "--out-dir", out, "--arms", "bf16,int8h", "--drafter", snap],
                               capture_output=True, text=True)
        assert again.returncode == 0 and "[bf16] done" in again.stdout and "[int8h] done" in again.stdout, again.stdout
        # --embed-from: a K2-shaped index whose embedding differs -> arms <arm>-k2emb
        from safetensors.torch import save_file
        k2 = os.path.join(tmp, "k2")
        os.makedirs(k2)
        save_file({"model.embed_tokens.weight": (t["embed_tokens.weight"].float() * 1.5).to(torch.bfloat16)},
                  os.path.join(k2, "s.safetensors"))
        with open(os.path.join(k2, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {"model.embed_tokens.weight": "s.safetensors"}}, f)
        ke = subprocess.run(acc + ["run", "--dumps", dumps, "--out-dir", out, "--arms", "bf16", "--drafter", snap,
                                   "--embed-from", k2], capture_output=True, text=True)
        assert ke.returncode == 0 and os.path.isfile(os.path.join(out, "accept.bf16-k2emb.json")), ke.stdout + ke.stderr
        lk = subprocess.run(acc + ["lookup", "--dumps", dumps, "--out-dir", out], capture_output=True, text=True)
        assert lk.returncode == 0 and os.path.isfile(os.path.join(out, "lookup.json")), lk.stdout + lk.stderr
        cost = subprocess.run(py + [os.path.join(_HERE, "eagle3_cost.py"), "run", "--dumps", dumps, "--out",
                                    os.path.join(out, "cost.json"), "--drafter-config", os.path.join(snap, "config.json")],
                              capture_output=True, text=True)
        assert cost.returncode == 0, cost.stdout + cost.stderr
        md = os.path.join(out, "summary.md")
        s = subprocess.run(acc + ["summary", "--out-dir", out, "--cost", os.path.join(out, "cost.json"), "--md", md],
                           capture_output=True, text=True)
        assert s.returncode == 0, s.stdout + s.stderr
        text = open(md).read()
        if os.environ.get("SHOW_MD"):
            print(text)
        for want in ("## a4", "## golden", "## all", "| bf16 |", "| int8h |", "| lookup (match at", "S_5 @ 4096",
                     "ratio int8 head", "### S_K at every depth"):
            assert want in text, (want, text)
        k2row = [ln for ln in text.splitlines() if ln.startswith("| bf16-k2emb | 31 |")]
        cells = [x.strip() for x in k2row[0].split("|")] if k2row else []
        assert cells and all(x not in ("", "-") for x in cells[10:15]), k2row   # S_K with the bf16 drafter's bytes
        dry = subprocess.run(acc + ["run", "--dumps", dumps, "--out-dir", out, "--dry-run"], capture_output=True, text=True)
        assert dry.returncode == 0 and "anchors" in dry.stdout
    print("  CLI: run (bf16, int8h) -> resume -> lookup -> eagle3_cost run -> summary.md with every table; --dry-run")


TESTS = [test_run_source_equals_reference, test_positive_alignment, test_lookup_matches_lookup_accept, test_cli_end_to_end]


def main() -> None:
    torch.set_grad_enabled(False)
    only = sys.argv[1:]
    for t in TESTS:
        if not only or t.__name__ in only:
            print(t.__name__)
            t()
    print("ok")


if __name__ == "__main__":
    main()
