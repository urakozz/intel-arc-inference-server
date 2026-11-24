#!/usr/bin/env python3
"""Plan 21-QFN-v2a Task 7: the quality of our Qwen3.8-Flash-Next export, on the CPU with qwen4exp_ref.py's layer-streamed
model in f32 (the test machine's CPU has no bf16 units: f32 is its fast path).

  qwen4exp_quality.py forced <snapshot> --ctx DIR --out DIR [--ple int8:/ple] [--batch 8] [--chunk 2048] [--match RE]
      every DIR/<key>.prompt.ids with its <key>.force.ids (serve_judge.py's dump; the key's FNV-1a hash is checked) ->
      OUT/<key>.golden.safetensors: `logits` f32 [1 + gen][V] (row j: the decision after the prompt + force[:j]),
      `tokens` (the forced ids), `prompt.ids` - what serve_judge.py --judge reads. f32 arithmetic, the engine's int8
      head (int8_sim) and the given PLE table. The contexts share prefixes (the tool-call prompts' system text and
      tools): they run as a radix trie of their ids, layer-major (qwen4exp_ref.layer_major), a node's cache copied into
      each child, so a shared prefix is computed once. The rows are teacher-forced prefill rows, not decode steps.
  qwen4exp_quality.py kl --ref <bf16 original> --cand <export> --source SET/NAME:IDS ... --work DIR --out JSON
                         [--ple-ref bf16] [--ple-cand int8:/ple] [--max-tokens 4096]
      KL(P_ref || P_cand) in nats and the top-1 agreement on every row, teacher-forced, per source and per set (the
      name's part before '/', token-weighted), with both models' perplexity. Each model is ONE layer-major pass over
      all sources (whole sequences up to --max-tokens, so a layer is read once); its final mixer's rows and its head
      are kept in WORK (the reference's pass is reused when there). The candidate is scored with the engine's int8
      head and with its own bf16 head.
  qwen4exp_quality.py force <f32 state> <out ids>      the f32 run's `tokens` as an ids file (run --force)
  qwen4exp_quality.py twin-trace <bf16 golden> <prompt ids> <out>
      the bf16 reference's routes (qwen4exp_ref.py run --force, every row) in qwen4exp_route_capture's form, so an f32
      companion (run --engine-routes) judges them exactly as it judges the engine's
  qwen4exp_quality.py twin-count <bf16 golden> <f32 companion> [--json FILE]
      spec 21-QFN-v2 §10's twin count, the reference's side: the companion's rejected routes over every prompt and
      continuation row, and the bf16 reference's own greedy id on each continuation row judged by the companion's
      logits (golden_common.h judge_row's one-bf16-ulp rule) - the lines qwen4exp_prefill_test prints for the engine.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import copy  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402


def _load(name: str, path: str | None = None):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path or os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


Q = _load("qwen4exp_ref")


def judge_module():
    return _load("serve_judge", os.path.join(_HERE, "..", "box_validate", "serve_judge.py"))


def peak() -> str:
    return f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB"


def build(snapshot: str, ple: str, experts: str):
    tc = Q.text_config(snapshot, None, experts, dtype=torch.float32)
    Q.require_cacheable(tc)
    model, pf, lazy = Q.build_streamed(snapshot, tc, Q.PleTable(Q.ple_source(snapshot, ple), tc))
    return tc, model, pf, lazy


def int8_head(w: torch.Tensor) -> torch.Tensor:
    return Q._load("int8_sim").int8_head_weight(w)


# ------------------------------------------------------------------------------------------------
# forced: the f32 judges of serve_judge.py's dumped contexts

def trie(seqs, group, start):
    """A radix node over seqs[k] for k in group, all sharing seqs[k][:start]: its segment [start, end) is the group's
    common part, its children the groups by the next id (a sequence ending at `end` has no child)."""
    end = min(len(seqs[k]) for k in group)
    first = seqs[group[0]]
    e = start
    while e < end and all(seqs[k][e] == first[e] for k in group):
        e += 1
    by: dict[int, list[int]] = {}
    for k in group:
        if len(seqs[k]) > e:
            by.setdefault(seqs[k][e], []).append(k)
    return {"start": start, "end": e, "group": group, "children": [trie(seqs, g, e) for g in by.values()]}


def cmd_forced(a) -> None:
    from safetensors.torch import save_file
    t0 = time.time()
    sj = judge_module()
    ctx = []
    for p in sorted(glob.glob(os.path.join(a.ctx, "*.prompt.ids"))):
        key = os.path.basename(p)[: -len(".prompt.ids")]
        if a.match and not re.search(a.match, key):
            continue
        if os.path.exists(os.path.join(a.out, key + ".golden.safetensors")):
            continue
        prompt, force = Q.read_ids(p), Q.read_ids(p[: -len(".prompt.ids")] + ".force.ids")
        if not prompt or not force or key[-16:] != sj.fnv_key_hash(prompt, force):
            Q.die(f"{key}: empty ids or the key's hash is not its prompt / force ids")
        ctx.append((key, prompt, force))
    os.makedirs(a.out, exist_ok=True)
    if not ctx:
        print(f"forced: nothing to do in {a.ctx}")
        return
    seqs = [p + f for _, p, f in ctx]
    root = trie(seqs, list(range(len(ctx))), 0)
    nodes, stack = 0, [root]
    work = 0
    while stack:
        n = stack.pop()
        nodes += 1
        work += n["end"] - n["start"]
        stack += n["children"]
    print(f"forced: {len(ctx)} contexts, {sum(map(len, seqs))} ids, {work} trie positions in {nodes} nodes; "
          f"batch {a.batch}, chunk {a.chunk}", flush=True)
    tc, model, pf, lazy = build(a.snapshot, a.ple, a.experts)
    with torch.no_grad():
        model.lm_head.weight = torch.nn.Parameter(int8_head(model.lm_head.weight), requires_grad=False)
    step = Q.layer_major(model, pf)
    first_row = [len(p) - 1 for _, p, _ in ctx]
    rows = [[None] * (len(f) + 1) for _, _, f in ctx]
    pending, done = [(root, {})], 0
    while pending:
        batch = []
        while pending and len(batch) < a.batch:
            batch.append(pending.pop())
        t1 = time.time()
        longest = max(n["end"] - n["start"] for n, _ in batch)
        for c0 in range(0, longest, a.chunk):
            items, owner = [], []
            for n, st in batch:
                s0 = n["start"] + c0
                if s0 < n["end"]:
                    items.append((seqs[n["group"][0]][s0:min(n["end"], s0 + a.chunk)], s0, st))
                    owner.append(n)
            for n, (ids, s0, _), R in zip(owner, items, step(items)):
                s1 = s0 + len(ids)
                want = sorted({r for k in n["group"] for r in range(max(s0, first_row[k]), min(s1, len(seqs[k])))})
                if not want:
                    continue
                with torch.no_grad():
                    lg = Q.final_logits(model, R[torch.tensor(want) - s0]).float()
                at = {r: i for i, r in enumerate(want)}
                for k in n["group"]:
                    for r in range(max(s0, first_row[k]), min(s1, len(seqs[k]))):
                        rows[k][r - first_row[k]] = lg[at[r]].clone()
        for n, st in batch:
            kids = n["children"]
            states = [copy.deepcopy(st) for _ in kids[:-1]] + ([st] if kids else [])
            pending += list(zip(kids, states))
            for k in n["group"]:
                if len(seqs[k]) != n["end"]:
                    continue
                key, prompt, force = ctx[k]
                if any(r is None for r in rows[k]):
                    raise RuntimeError(f"{key}: rows missing at its end")
                t = {"logits": torch.stack(rows[k]).contiguous(), "tokens": torch.tensor(force, dtype=torch.int32),
                     "prompt.ids": torch.tensor(prompt, dtype=torch.int32)}
                save_file(t, os.path.join(a.out, key + ".golden.safetensors"),
                          metadata={"model": "Qwen3.8-Flash-Next", "checkpoint": os.path.abspath(a.snapshot),
                                    "dtype": "fp32", "lm_head": "int8", "ple": a.ple, "n_prompt": str(len(prompt)),
                                    "gen": str(len(force)), "rows": "teacher-forced prefill rows (radix trie)",
                                    "transformers": Q.PINNED_TRANSFORMERS, "torch": torch.__version__})
                rows[k] = None
                done += 1
        print(f"  {len(batch)} nodes ({sum(n['end'] - n['start'] for n, _ in batch)} positions) {time.time() - t1:.0f}s; "
              f"{done}/{len(ctx)} contexts written, {len(pending)} nodes pending, {peak()}", flush=True)
    print(f"forced: wall {time.time() - t0:.0f}s, {peak()}, prefetch wait {pf.seconds:.1f}s")


# ------------------------------------------------------------------------------------------------
# kl: ours against the bf16 original

def final_rows(snapshot: str, ple: str, experts: str, sources, max_tokens: int):
    """({name: the final mixer's rows f32 [T][H]}, the head f32 [V][H]) - one layer-major pass over all sources."""
    tc, model, pf, lazy = build(snapshot, ple, experts)
    step = Q.layer_major(model, pf)
    t1 = time.time()
    items = [(ids[:max_tokens], 0, {}) for _, ids in sources]
    out = {}
    with torch.no_grad():
        for (name, _), R in zip(sources, step(items)):
            out[name] = model.model.hyper_connection_mixer(R[None])[0].float().contiguous()
    head = model.lm_head.weight.detach().float().clone()
    print(f"kl: {snapshot}: {sum(len(i[0]) for i in items)} rows of {len(items)} sources in {time.time() - t1:.0f}s "
          f"({lazy.filled} expert loads, prefetch wait {pf.seconds:.1f}s), {peak()}", flush=True)
    return out, head


def cmd_kl(a) -> None:
    from safetensors.torch import load_file, save_file
    t0 = time.time()
    sources = []
    for s in a.source:
        name, _, path = s.partition(":")
        sources.append((name, Q.read_ids(path)))
    if not sources:
        Q.die("kl: no --source")
    os.makedirs(a.work, exist_ok=True)
    ref_file = os.path.join(a.work, "ref.rows.safetensors")
    names = [n for n, _ in sources]
    if os.path.exists(ref_file):
        t = load_file(ref_file)
        if any(n not in t for n in names):
            Q.die(f"{ref_file} lacks a source: remove it (ids differ)")
        ref, ref_head = {n: t[n] for n in names}, t["head"]
        print(f"kl: the reference's rows reused from {ref_file}")
    else:
        ref, ref_head = final_rows(a.ref, a.ple_ref, a.experts, sources, a.max_tokens)
        save_file(dict(ref, head=ref_head), ref_file, metadata={"snapshot": os.path.abspath(a.ref), "ple": a.ple_ref})
    import gc
    gc.collect()
    cand, cand_head = final_rows(a.cand, a.ple_cand, a.experts, sources, a.max_tokens)
    heads = {"int8": int8_head(cand_head), "bf16": cand_head}
    res = {"sources": {}, "sets": {}, "ref": os.path.abspath(a.ref), "cand": os.path.abspath(a.cand),
           "ple_ref": a.ple_ref, "ple_cand": a.ple_cand}
    for name, ids in sources:
        hr, hc = ref[name], cand[name]
        T = hr.shape[0]
        tgt = torch.tensor(ids[:T])
        r = {"rows": T}
        for hn, w in heads.items():
            kl, top1, nll_p, nll_q, kls = 0.0, 0, 0.0, 0.0, []
            for b0 in range(0, T, 256):
                b1 = min(T, b0 + 256)
                lp = torch.log_softmax(hr[b0:b1] @ ref_head.t(), -1)
                lq = torch.log_softmax(hc[b0:b1] @ w.t(), -1)
                k = (lp.exp() * (lp - lq)).sum(-1)
                kls.append(k)
                kl += float(k.double().sum())
                top1 += int((lp.argmax(-1) == lq.argmax(-1)).sum())
                n1 = min(b1, T - 1)
                if n1 > b0:
                    nx = tgt[b0 + 1:n1 + 1, None]
                    nll_p -= float(lp[: n1 - b0].gather(1, nx).double().sum())
                    nll_q -= float(lq[: n1 - b0].gather(1, nx).double().sum())
            ks = torch.cat(kls).double()
            r[hn] = {"kl_sum": kl, "top1": top1, "nll_ref": nll_p, "nll_cand": nll_q, "kl_mean": kl / T,
                     "top1_rate": top1 / T, "kl_p50": float(ks.quantile(0.5)), "kl_p99": float(ks.quantile(0.99)),
                     "kl_max": float(ks.max())}
        res["sources"][name] = r
    for hn in heads:
        for name, r in res["sources"].items():
            sname = name.split("/", 1)[0]
            s = res["sets"].setdefault(hn, {}).setdefault(sname, {"rows": 0, "kl_sum": 0.0, "top1": 0, "nll_ref": 0.0,
                                                                    "nll_cand": 0.0, "pred": 0})
            s["rows"] += r["rows"]
            s["pred"] += r["rows"] - 1
            for f in ("kl_sum", "top1", "nll_ref", "nll_cand"):
                s[f] += r[hn][f]
    print(f"\n| set | source | rows | KL mean, int8 head | top-1, int8 head | KL p99 | KL max | KL mean, bf16 head | "
          f"top-1, bf16 head |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in res["sources"].items():
        i, b = r["int8"], r["bf16"]
        print(f"| {name.split('/', 1)[0]} | {name} | {r['rows']} | {i['kl_mean']:.5f} | {i['top1_rate']:.4f} | "
              f"{i['kl_p99']:.4f} | {i['kl_max']:.3f} | {b['kl_mean']:.5f} | {b['top1_rate']:.4f} |")
    print("\n| set | head | rows | KL mean | top-1 | ppl original | ppl ours |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for hn, sets in res["sets"].items():
        for sname, s in sets.items():
            s.update(kl_mean=s["kl_sum"] / s["rows"], top1_rate=s["top1"] / s["rows"],
                     ppl_ref=math.exp(s["nll_ref"] / max(s["pred"], 1)), ppl_cand=math.exp(s["nll_cand"] / max(s["pred"], 1)))
            print(f"| {sname} | {hn} | {s['rows']} | {s['kl_mean']:.5f} | {s['top1_rate']:.4f} | {s['ppl_ref']:.4f} | "
                  f"{s['ppl_cand']:.4f} |")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nkl: wrote {a.out}; wall {time.time() - t0:.0f}s, {peak()}")


# ------------------------------------------------------------------------------------------------
# the twin count's reference side

def cmd_force(a) -> None:
    from safetensors.torch import load_file
    tok = load_file(a.state)["tokens"].tolist()
    with open(a.out, "w") as f:
        f.write(" ".join(map(str, tok)) + "\n")
    print(f"force: {len(tok)} ids -> {a.out}")


def cmd_twin_trace(a) -> None:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    g = load_file(a.golden)
    with safe_open(a.golden, "pt") as f:
        meta = f.metadata()
    T, gen, L = int(meta["n_prompt"]), int(meta["gen"]), int(meta["layers"])
    if meta.get("chunk") != "2048" or not meta.get("force"):
        Q.die(f"{a.golden}: not a --chunk 2048 --force run")
    src = Q.read_ids(a.prompt)
    prompt = [src[i % len(src)] for i in range(T)]
    V = g["logits"].shape[1]
    ple = int(meta["ple"] == "int8")   # the companion must read the same table (validate_trace)
    out = {"trace.spec": torch.tensor([1, L, T, gen, 2048, 8192, 1, ple, min(248077, V), V], dtype=torch.int32),
           "trace.path": torch.tensor([1], dtype=torch.int32),
           "prompt.ids": torch.tensor(prompt, dtype=torch.int64), "tokens": g["tokens"].to(torch.int64)}
    for l in range(L):
        r = g[f"route.ids.L{l}"]
        if r.shape[0] != T + gen:
            Q.die(f"route.ids.L{l}: {r.shape[0]} rows, expected {T + gen}")
        out[f"route.ids.L{l}"] = r.to(torch.int32).contiguous()
    save_file(out, a.out)
    print(f"twin-trace: {L} layers x {T + gen} rows -> {a.out}")


def cmd_twin_count(a) -> None:
    from safetensors import safe_open
    from safetensors.torch import load_file
    sj = judge_module()
    g, c = load_file(a.golden), load_file(a.companion)
    with safe_open(a.companion, "pt") as f:
        meta = f.metadata()
    used = int(meta["vocab_used"])
    gen = g["tokens"].numel()
    if not torch.equal(g["tokens"].long(), c["tokens"].long()):
        Q.die("the bf16 reference and its companion force different continuations")
    if g["logits"].shape[0] != gen + 1 or c["logits"].shape[0] != gen + 1:
        Q.die("expected the last prompt row + one row a continuation step in both files (--logits-tail 1)")
    L = int(c["state.spec"][2])
    v = torch.cat([c[f"route.judge.L{l}"] for l in range(L)])
    exact, ties, bad = (int((v == i).sum()) for i in range(3))
    accepted, same = 0, 0
    lines = []
    for j in range(gen):
        row = c["logits"][j].numpy()
        i = int(torch.argmax(g["logits"][j][:used]))
        top, s, ulp = sj.judge_row(row, used)
        ok = i == top if len(s) == 1 else i in s
        accepted += ok
        same += i == top
        if not ok:
            lines.append(f"   f32 bf16-reference row {j} rejects id {i}, top {top}, gap "
                         f"{(float(row[top]) - float(row[i])) / ulp:.4g} bf16 ulp")
    print("\n".join(lines))
    print(f"f32 bf16-reference route alignment: {exact} exact, {ties} accepted near-ties, {bad} rejected "
          f"({L} layers x {v.numel() // L} rows)")
    print(f"f32 tokens bf16 reference: {accepted} / {gen} accepted (one bf16 ulp), {same} exact; teacher-forced on f32")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"exact": exact, "near_ties": ties, "rejected": bad, "tokens_accepted": accepted,
                       "tokens_exact": same, "gen": gen}, f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forced")
    f.add_argument("snapshot")
    f.add_argument("--ctx", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--ple", default="int8:/ple")
    f.add_argument("--batch", type=int, default=8, help="trie nodes in flight (each its own cache)")
    f.add_argument("--chunk", type=int, default=2048)
    f.add_argument("--match", default="", help="only the keys this regex finds")
    f.add_argument("--experts", default=Q.DEFAULT_EXPERTS, choices=["grouped_mm", "eager"])
    k = sub.add_parser("kl")
    k.add_argument("--ref", required=True)
    k.add_argument("--cand", required=True)
    k.add_argument("--ple-ref", default="bf16")
    k.add_argument("--ple-cand", default="int8:/ple")
    k.add_argument("--source", action="append", default=[], help="SET/NAME:IDS_FILE")
    k.add_argument("--work", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--max-tokens", type=int, default=4096)
    k.add_argument("--experts", default=Q.DEFAULT_EXPERTS, choices=["grouped_mm", "eager"])
    fo = sub.add_parser("force")
    fo.add_argument("state")
    fo.add_argument("out")
    tt = sub.add_parser("twin-trace")
    tt.add_argument("golden")
    tt.add_argument("prompt")
    tt.add_argument("out")
    tc = sub.add_parser("twin-count")
    tc.add_argument("golden")
    tc.add_argument("companion")
    tc.add_argument("--json", default="")
    a = ap.parse_args()
    {"forced": cmd_forced, "kl": cmd_kl, "force": cmd_force, "twin-trace": cmd_twin_trace,
     "twin-count": cmd_twin_count}[a.cmd](a)


if __name__ == "__main__":
    main()
