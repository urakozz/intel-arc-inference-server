#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced EAGLE-3 acceptance on K2 tap dumps, the prompt-lookup comparison, the summary
(the EAGLE3 K2 P0).

    eagle3_accept.py run --dumps DIR --out-dir OUT [--arms bf16,int8,int8h,int4h] [--k 1-5]
                     [--batch 64] [--drafter SNAP] [--window query|anchor] [--sources GLOB]
                     [--dry-run]
    eagle3_accept.py lookup --dumps DIR --out-dir OUT [--k 1-5] [--min-match 3] [--max-match 64]
    eagle3_accept.py summary --out-dir OUT [--cost OUT/cost.json] [--depth 4096] [--md FILE]

**Protocol** (dflash_accept.py's, with EAGLE-3's chain). A dump (k2_taps.py) holds ids
x[0..N-1] = prompt + K2's recorded greedy continuation, the aux taps, and g[t] = K2's argmax
after position t (t >= n_prompt - 1). For every anchor p with n_prompt <= p <= N - 5:
  - the drafter's context is step 0 over every row (eagle3_ref.Drafter.context: row s = token
    x[s] + the aux of s - 1, from the TARGET's taps - what vLLM rebuilds each round);
  - the anchor is row p (x[p] is the pending token; its aux is position p - 1's); d_0 is its
    greedy id, d_1..d_4 the chain (eagle3_ref.propose_batch), drafts for positions p + 1 ..;
  - K = 1..5 take the first K (the chain is autoregressive: K = 3's drafts ARE K = 5's first 3);
  - the verify keeps the longest prefix with d_j == g[p + j]; depth j is known only while the
    text agrees with the drafts before it (x[p + 1 + i] == g[p + i], i < j), else the row is
    CENSORED at j (dflash_accept.accept_rows, reused).
Per corpus (the name before '/'), arm and K: alpha_j (conditional on depths < j kept), E_K =
1 + sum_i prod_{j<=i} alpha_j tokens per verify (the target's own token included), a_j =
prod_{i<=j} alpha_i (the UNconditional per-position acceptance: vLLM's / the model card's
0.44 / 0.17 metric), the length histogram. Greedy only.

**Arms** eagle3_ref.ARMS: bf16 (as shipped), int8 (body int8 RTN, head bf16), int8h (+ head
int8), int4h (body int4 g64 RTN as src/loader/rtn.h + head int8). The embedding: the drafter's
own (embed_requires_grad false - a frozen copy of the verifier's) - what vLLM uses either way:
it swaps in the target's only when byte-identical (llm_base_proposer.py:1449-1516).
`eagle3_ref.py facts --k2` compares the bytes; `--embed-from <K2 snapshot>` drafts with K2's
instead (arms `<arm>-k2emb`: what an engine sharing K2's table would get if they differ).

**Prompt lookup** (`lookup`): spec 19e's proposer exactly as tools/spec/lookup_accept.py replays
it (src/server/prompt_lookup.h's matcher, no candidate cap): at anchor p the matcher holds
x[0..p]; with a match >= --min-match (b70-serve's lookup_min_match, 3) its continuation drafts up
to K ids, else none (a plain step). Scored by the same accept_rows against g; an anchor with
fewer drafts than K counts the missing depths as misses (E per ITERATION), and the summary
costs each anchor at verify(1 + drafts) or 1 (no match), the lookup's draft cost ~0 (host).

**Summary**: the tables above per corpus, eagle3_cost.py's verify / draft bytes, and the
DERIVED speed-up S(K) = E_K / (V(K + 1) + D(K)) with V the verify-bytes ratio and D the
drafter's bytes for K drafts, both over one plain step's (int8 head, int8 KV at --depth).

Resumable: OUT/accept.<arm>.json / OUT/lookup.json are rewritten after every source; a re-run
skips what they hold. --dry-run prints the planned anchors, FLOP and memory; loads nothing.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import fnmatch  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

KMAX = 5
ASSUMED_GFLOPS = 300.0
ARMS = ("bf16", "int8", "int8h", "int4h")


def _load(name, path=None):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path or os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def deps():
    """(torch, eagle3_ref, k2_taps, dflash_accept with its deps loaded)."""
    import torch
    E3 = _load("eagle3_ref")
    KT = _load("k2_taps")
    DA = _load("dflash_accept")
    DA._load_deps()
    return torch, E3, KT, DA


def lookup_mod():
    return _load("lookup_accept", os.path.join(_HERE, "..", "spec", "lookup_accept.py"))


def anchors_of(n_prompt: int, n: int):
    return list(range(n_prompt, n - KMAX + 1))


def dump_files(dumps_dir, pattern=None):
    fs = sorted(glob.glob(os.path.join(dumps_dir, "*.k2taps.safetensors")))
    if pattern:
        fs = [f for f in fs if fnmatch.fnmatch(os.path.basename(f), pattern)]
    return fs


def _ks(spec: str):
    lo, _, hi = spec.partition("-")
    ks = list(range(int(lo), int(hi or lo) + 1))
    if not (1 <= ks[0] and ks[-1] <= KMAX):
        raise SystemExit(f"--k within 1-{KMAX}")
    return ks


def _save(path, state):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(path + ".tmp", path)


def _load_state(path, init):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return init


def score(DA, drafts, ps, x, g, g_from, ks, n_drafts=None) -> dict:
    """Per K the accept_rows stats of drafts[:, :K] (+ how many anchors had 0..K drafts)."""
    import torch
    res = {}
    for K in ks:
        length, cens, counted, accepted = DA.accept_rows(drafts[:, :K], ps, x, g, g_from)
        r = {"rows": int(ps.numel()), "censored": int(cens.sum()),
             "hist": torch.bincount(length[~cens], minlength=K + 1).tolist(), "counted": counted, "accepted": accepted}
        if n_drafts is not None:
            r["drafted_hist"] = torch.bincount(n_drafts.clamp(max=K), minlength=K + 1).tolist()
        res[str(K)] = r
    return res


def text_stats(dump) -> dict:
    """How often the recorded continuation is the teacher-forced greedy, and how often the
    one-group final norm (speculators' verifier_norm) gives K2's greedy - over the continuation."""
    n_p, n, hf = dump["n_prompt"], dump["n"], dump["head_from"]
    x, g = dump["ids"], dump["greedy"]
    agree = int((x[n_p:] == g[n_p - 1 - hf: n - 1 - hf]).sum())
    flat = int((dump["greedy_flatnorm"][n_p - 1 - hf: n - 1 - hf] == g[n_p - 1 - hf: n - 1 - hf]).sum())
    return {"agree": agree, "flat_agree": flat, "cont": n - n_p}


def run_source(d, dump, ks, batch: int, window: str, DA) -> dict:
    import torch
    c = d.cfg
    if not set(c.aux_ids) <= set(dump["aux_ids"]):
        raise SystemExit(f"{dump['name']}: the dump's aux {dump['aux_ids']} lack the drafter's {c.aux_ids}")
    KT = _load("k2_taps")
    ctx = d.context(dump["ids"], KT.aux_cat(dump, c.aux_ids), dump["aux_from"], head_from=dump["n_prompt"])
    anchors = anchors_of(dump["n_prompt"], dump["n"])
    if not anchors:
        return {}
    ps = torch.tensor(anchors)
    drafts = torch.cat([d.propose_batch(ctx, anchors[b:b + batch], KMAX, window) for b in range(0, len(anchors), batch)])
    return score(DA, drafts, ps, dump["ids"], dump["greedy"], dump["head_from"], ks)


def lookup_source(dump, ks, min_match: int, max_match: int, DA) -> dict:
    import torch
    la = lookup_mod()
    x = dump["ids"].tolist()
    n_p, n = dump["n_prompt"], dump["n"]
    anchors = anchors_of(n_p, n)
    if not anchors:
        return {}
    want = set(anchors)
    lk = la.Lookup(max_match)
    lk.extend(x[:n_p])
    rows, nd = [], []
    for p in range(n_p, n):
        lk.append(x[p])                                # the matcher holds x[0..p]
        if p in want:
            dr = lk.propose(KMAX, min_match)[0][:KMAX]
            nd.append(len(dr))
            rows.append(dr + [-1] * (KMAX - len(dr)))
    drafts = torch.tensor(rows, dtype=torch.long)
    return score(DA, drafts, torch.tensor(anchors), dump["ids"], dump["greedy"], dump["head_from"], ks,
                 n_drafts=torch.tensor(nd, dtype=torch.long))


# ------------------------------------------------------------------------------ commands


def read_k2_embed(snapshot: str):
    """K2's model.embed_tokens.weight (bf16 as stored) from its shard index."""
    from safetensors import safe_open
    with open(os.path.join(snapshot, "model.safetensors.index.json"), encoding="utf-8") as f:
        fn = json.load(f)["weight_map"]["model.embed_tokens.weight"]
    with safe_open(os.path.join(snapshot, fn), framework="pt", device="cpu") as h:
        return h.get_tensor("model.embed_tokens.weight")


def plan(dumps, pattern):
    """[(name, n_prompt, n)] from the dump headers, or sources.json before any dump exists
    (no torch: the dry run's)."""
    fs = dump_files(dumps, pattern)
    if fs:
        from safetensors import safe_open
        out = []
        for f in fs:
            with safe_open(f, framework="pt", device="cpu") as h:
                md = h.metadata()
            out.append((md["name"], int(md["n_prompt"]), int(md["n"])))
        return fs, out
    sj = os.path.join(dumps, "sources.json")
    if not os.path.isfile(sj):
        raise SystemExit(f"no dumps and no sources.json in {dumps}")
    with open(sj, encoding="utf-8") as f:
        srcs = [(s["name"], s["n_prompt"], s["n"]) for s in json.load(f)]
    if pattern:
        srcs = [s for s in srcs if fnmatch.fnmatch(s[0].replace("/", "__") + ".k2taps.safetensors", pattern)]
    return [], srcs


def cmd_run(a) -> None:
    arms = [s for s in a.arms.split(",") if s]
    bad = [s for s in arms if s not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}")
    ks = _ks(a.k)
    files, srcs = plan(a.dumps, a.sources)
    n_anchor = sum(len(anchors_of(p, n)) for _, p, n in srcs)
    rows = sum(n - max(0, p - 2049) for _, p, n in srcs)
    # step 0: ~0.22 GFLOP a row (fc + layer) + the 32k head on rows >= n_prompt - 1; steps 1..4:
    # ~0.36 GFLOP per anchor and step (layer + head) - derived from the shapes
    gflop = rows * 2 * 109.5e6 / 1e9 + sum(n - p + 1 for _, p, n in srcs) * 2 * 83.9e6 / 1e9 \
        + n_anchor * (KMAX - 1) * 2 * 182e6 / 1e9
    print(f"{len(srcs)} sources, {n_anchor} anchors, {rows} context rows; K {ks[0]}..{ks[-1]} (one chain of {KMAX}); "
          f"arms {arms}; window {a.window}")
    print(f"per arm ~{gflop / 1e3:.1f} TFLOP, ~{gflop / ASSUMED_GFLOPS / 60:.0f} min at {ASSUMED_GFLOPS:.0f} GFLOP/s "
          f"(assumed); memory (estimated): drafter float32 0.8 GB + embedding bf16 1.3 GB, a source's context "
          f"~12 KB a row -> ~3-4 GiB peak")
    if a.dry_run:
        return
    torch, E3, KT, DA = deps()
    torch.set_grad_enabled(False)
    os.makedirs(a.out_dir, exist_ok=True)
    snap = a.drafter or E3.find_snapshot(E3.REPO_ID, ("config.json", "model.safetensors"))
    if snap is None:
        raise SystemExit(f"{E3.REPO_ID} (config.json + model.safetensors) not in {E3.hf_hub_dir()}")
    t0 = time.time()
    tensors = E3.read_tensors(snap)
    suffix = ""
    if a.embed_from:                                   # K2's embedding instead of the drafter's own
        tensors = dict(tensors)
        tensors["embed_tokens.weight"] = read_k2_embed(a.embed_from)
        suffix = "-k2emb"
    for arm in arms:
        name = arm + suffix
        path = os.path.join(a.out_dir, f"accept.{name}.json")
        state = _load_state(path, {"arm": name, "sources": {}, "text": {}, "meta": {}})
        todo = [f for f in files if KT.dump_header(f)["name"] not in state["sources"]]
        if not todo:
            print(f"[{arm}] done", flush=True)
            continue
        d = E3.Drafter(E3.read_config(snap), tensors)
        nq = d.quantize(arm)
        state["meta"] = {"drafter": snap, "arm": name, "quantised_linears": nq, "embed": a.embed_from or "the drafter's", "window": a.window, "ks": ks,
                         "batch": a.batch, "threads": torch.get_num_threads(), "extra_tensors": d.extra}
        print(f"[{arm}] drafter {snap} ({nq} linears quantised; window {a.window})", flush=True)
        for fn in todo:
            dump = KT.load_dump(fn)
            ts = time.time()
            state["sources"][dump["name"]] = run_source(d, dump, ks, a.batch, a.window, DA)
            st = text_stats(dump)
            st["in_draft_vocab"] = int(d.t2d[dump["ids"][dump["n_prompt"]:]].sum())   # draftable at all
            state["text"][dump["name"]] = st
            _save(path, state)
            e = DA.pool([state["sources"][dump["name"]]], ks)
            print(f"[{arm}] {dump['name']}: {len(anchors_of(dump['n_prompt'], dump['n']))} anchors, E_K "
                  + " ".join("%.2f" % e[str(K)]["E"] for K in ks if str(K) in e) + f"; {time.time() - ts:.0f} s", flush=True)
        del d
    print(f"wall {time.time() - t0:.0f} s", flush=True)


def cmd_lookup(a) -> None:
    ks = _ks(a.k)
    files, srcs = plan(a.dumps, a.sources)
    print(f"lookup: {len(srcs)} sources, {sum(len(anchors_of(p, n)) for _, p, n in srcs)} anchors, min match "
          f"{a.min_match}, max match {a.max_match}")
    if a.dry_run:
        return
    torch, E3, KT, DA = deps()
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "lookup.json")
    state = _load_state(path, {"sources": {}, "meta": {}})
    state["meta"] = {"min_match": a.min_match, "max_match": a.max_match, "ks": ks,
                     "matcher": "tools/spec/lookup_accept.py Lookup (src/server/prompt_lookup.h semantics, no candidate cap)"}
    for fn in files:
        dump = KT.load_dump(fn)
        if dump["name"] in state["sources"]:
            continue
        state["sources"][dump["name"]] = lookup_source(dump, ks, a.min_match, a.max_match, DA)
        _save(path, state)
        e = DA.pool([state["sources"][dump["name"]]], ks)
        print(f"[lookup] {dump['name']}: E_K " + " ".join("%.2f" % e[str(K)]["E"] for K in ks if str(K) in e), flush=True)
    _save(path, state)


# ------------------------------------------------------------------------------ summary


def pool_with_drafted(DA, entries, ks):
    out = DA.pool(entries, ks)
    for K in ks:
        sk = str(K)
        dh = [e[sk]["drafted_hist"] for e in entries if sk in e and "drafted_hist" in e[sk]]
        if sk in out and dh:
            out[sk]["drafted_hist"] = [sum(h[j] for h in dh) for j in range(K + 1)]
        if sk in out:
            prod, a = 1.0, []
            for al in out[sk]["alpha"]:
                prod *= (al or 0.0)
                a.append(prod)
            out[sk]["a_uncond"] = a
    return out


def speedup(cost, E, K, depth, arm=None, drafted_hist=None, scope="all") -> float | None:
    C = _load("eagle3_cost")
    try:
        if drafted_hist is not None:                       # the lookup: per-anchor verify size
            n = sum(drafted_hist)
            c = sum(h * (1.0 if j == 0 else C.verify_ratio(cost, j + 1, depth, scope)) for j, h in enumerate(drafted_hist))
            return E / (c / n) if n else None
        return E / (C.verify_ratio(cost, K + 1, depth, scope) + C.draft_ratio(cost, arm.split("-")[0], K, depth))
    except KeyError:
        return None


def summarize(out_dir: str, cost_path: str | None, depth: int, md_path: str | None) -> str:
    _, _, _, DA = deps()
    accept = {}
    text = {}
    for fn in sorted(glob.glob(os.path.join(out_dir, "accept.*.json"))):
        with open(fn, encoding="utf-8") as f:
            s = json.load(f)
        accept[s["arm"]] = s["sources"]
        text.update(s.get("text", {}))
    lookup = None
    if os.path.isfile(os.path.join(out_dir, "lookup.json")):
        with open(os.path.join(out_dir, "lookup.json"), encoding="utf-8") as f:
            lookup = json.load(f)["sources"]
    cost = None
    if cost_path and os.path.isfile(cost_path):
        with open(cost_path, encoding="utf-8") as f:
            cost = json.load(f)
    ks = list(range(1, KMAX + 1))
    props = {arm: srcs for arm, srcs in sorted(accept.items(), key=lambda kv: ARMS.index(kv[0]) if kv[0] in ARMS else 9)}
    if lookup:
        props["lookup"] = lookup
    names = sorted({n for s in props.values() for n in s})
    corpora = sorted({n.split("/")[0] for n in names}) + ["all"]
    L = ["# EAGLE3 K2 P0 - summary", "",
         "Acceptance is MEASURED (teacher-forced, greedy, against K2's own argmax); verify / draft costs and "
         "speed-ups are DERIVED (bytes only, eagle3_cost.py; the engine bandwidth-bound). "
         f"Sources {len(names)}: " + ", ".join(f"{c} {sum(1 for n in names if n.split('/')[0] == c)}" for c in corpora[:-1]) + "."]
    for corp in corpora:
        sel = [n for n in names if corp == "all" or n.split("/")[0] == corp]
        tx = [text[n] for n in sel if n in text]
        L += ["", f"## {corp}", ""]
        if tx:
            cont = sum(t["cont"] for t in tx)
            L.append(f"Recorded continuation = teacher-forced K2 greedy at {sum(t['agree'] for t in tx) / max(1, cont):.3f} "
                     f"of {cont} ids; K2's argmax with the final norm as ONE group (speculators' LlamaRMSNorm "
                     f"verifier_norm - the training targets' likely head) = K2's at "
                     f"{sum(t['flat_agree'] for t in tx) / max(1, cont):.3f}.")
            iv = [t["in_draft_vocab"] for t in tx if "in_draft_vocab" in t]
            if iv:
                L.append(f"The 32k draft vocabulary holds {sum(iv) / max(1, cont):.3f} of the continuation's ids (an "
                         f"id outside it is never drafted: a ceiling on a_1).")
            L.append("")
        L.append("| proposer | anchors | " + " | ".join(f"E_{K}" for K in ks) + " | a_1..a_5 (per position) | "
                 "alpha_1..alpha_5 (conditional) | " + " | ".join(f"S_{K} @ {depth}" for K in ks) + " | best K |")
        L.append("|---|---:|" + "---:|" * len(ks) + "---|---|" + "---:|" * len(ks) + "---:|")
        for prop, srcs in props.items():
            entries = [srcs[n] for n in sel if n in srcs]
            if not entries:
                continue
            p = pool_with_drafted(DA, entries, ks)
            if str(KMAX) not in p:
                continue
            es = [p[str(K)]["E"] for K in ks]
            sp = []
            for K in ks:
                if cost is None:
                    sp.append(None)
                elif prop == "lookup":
                    sp.append(speedup(cost, es[K - 1], K, depth, drafted_hist=p[str(K)]["drafted_hist"]))
                else:
                    sp.append(speedup(cost, es[K - 1], K, depth, arm=prop))
            best = max(((s, -K) for s, K in zip(sp, ks) if s is not None), default=(None, None))
            best = (best[0], "plain" if best[0] is not None and best[0] <= 1.0 else (-best[1] if best[1] else None))
            a5, al5 = p[str(KMAX)]["a_uncond"], p[str(KMAX)]["alpha"]
            cov = ""
            if prop == "lookup":
                dh = p[str(KMAX)]["drafted_hist"]
                cov = f" (match at {1 - dh[0] / max(1, sum(dh)):.2f})"
            L.append(f"| {prop}{cov} | {p['1']['rows']} | " + " | ".join(f"{e:.3f}" for e in es) + " | "
                     + " ".join(f"{x:.2f}" for x in a5) + " | " + " ".join(f"{x:.2f}" if x is not None else "-" for x in al5)
                     + " | " + " | ".join(f"{s:.3f}" if s is not None else "-" for s in sp)
                     + f" | {best[1] if best[1] else '-'} |")
    if cost is not None:
        C = _load("eagle3_cost")
        L += ["", "## Verify and draft cost (derived from the recorded routes)", "", C.format_md(cost), "",
              f"S_K = E_K / (V(K + 1) + D(K)): V = the verify's bytes over a plain step's, D = the drafter's bytes "
              f"for K drafts (fc once + K steps) over a plain step's; int8 head, int8 KV at depth {depth}. The lookup "
              f"row costs each anchor at V(1 + its drafts), or 1 with no match. A bound from bytes alone: launch "
              f"overhead, the drafter's serial steps and compute at M rows are not in it."]
        L += ["", f"### S_K at every depth (all sources)", "",
              "| proposer | depth | " + " | ".join(f"S_{K}" for K in ks) + " |", "|---|---:|" + "---:|" * len(ks)]
        for prop, srcs in props.items():
            p = pool_with_drafted(DA, list(srcs.values()), ks)
            if str(KMAX) not in p:
                continue
            for dpt in cost["depths"]:
                row = []
                for K in ks:
                    e = p[str(K)]["E"]
                    s = speedup(cost, e, K, dpt, drafted_hist=p[str(K)]["drafted_hist"]) if prop == "lookup" \
                        else speedup(cost, e, K, dpt, arm=prop)
                    row.append(f"{s:.3f}" if s is not None else "-")
                L.append(f"| {prop} | {dpt} | " + " | ".join(row) + " |")
    text_md = "\n".join(L) + "\n"
    print(text_md)
    if md_path:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text_md)
    return text_md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--batch", type=int, default=64)
    r.add_argument("--drafter")
    r.add_argument("--window", choices=("query", "anchor"), default="query")
    r.add_argument("--embed-from", help="a K2 snapshot: draft with ITS embedding (arms named <arm>-k2emb)")
    lk = sub.add_parser("lookup")
    lk.add_argument("--min-match", type=int, default=3)
    lk.add_argument("--max-match", type=int, default=64)
    for p in (r, lk):
        p.add_argument("--dumps", required=True)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--k", default=f"1-{KMAX}")
        p.add_argument("--sources", help="fnmatch over the dump file names")
        p.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--cost")
    s.add_argument("--depth", type=int, default=4096)
    s.add_argument("--md")
    a = ap.parse_args()
    if a.cmd == "summary":
        summarize(a.out_dir, a.cost, a.depth, a.md)
    elif a.cmd == "lookup":
        cmd_lookup(a)
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()
