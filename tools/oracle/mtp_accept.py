#!/usr/bin/env python3
"""MTP draft acceptance by depth, greedy and sampled (spec 8 P0, plan 8a Task 2).

    mtp_accept.py <snapshot> --out results.json
        [--source NAME:CTX_IDS:CONT_IDS ...]     id files (whitespace-separated)
        [--opencode DIR]                         a recorded server log (spec 7 plan 7a):
                                                 DIR/NNNNNN.json with prompt_ids, out_ids,
                                                 request; the longest turns up to --max-out
                                                 output tokens are taken
        [--ctx-max 2048] [--max-out 2000] [--controls]
        [--temperature T --top-p P --top-k K]    default: the snapshot's generation_config.json,
                                                 or the opencode requests' own values
    mtp_accept.py --summarize A.json [B.json ...] --out merged.json
                                                 pool per-source results (also .partial files)
                                                 without the model; no container needed

Rerun on the recorded opencode log (spec 7 plan 7a) in one command, on the box:
    tools/oracle/run_in_container.sh 'python3 tools/oracle/mtp_accept.py "$SNAP" \
        --opencode /ws/tests/golden/opencode/session1 --out /ws/oracle-out-spec8a/accept_opencode.json'

One teacher-forced CPU forward of the main model (mtp_ref.build_main, the same
dequantised model as the golden oracle) over ctx[-ctx_max:] + cont gives, per
position, the main logits and the main hidden (post final norm, the reference
wiring). The head (mtp_ref.MtpHead) then runs a depth-3 chain for every row t:
step 1 on (h_t, x[t+1]) at position t, attending over the step-1 KV of rows
0..t (what prefill and earlier commits leave in the head's cache); step i+1 on
(the head's own step-i hidden, the step-(i+1) token) at position t+i, attending
over the same rows plus its own earlier chained keys (Review Focus 5).

Greedy, row t (continuation rows only, t >= len(ctx)-1): with g[i] = argmax of
the main logits at i, depth 1 accepts when argmax q1[t] == g[t+1]. Depth d > 1
is conditional: counted only on rows where depths 1..d-1 were accepted and the
text agrees with them (x[t+1+j] == g[t+j]); the chain is fed x[t+1+j], which on
those rows IS the accepted draft. alpha_d = accepted / counted.

Sampled: after the same temperature / top-k / top-p filter on p (main, row
t+d) and q (head, depth d, row t), alpha_d = mean over rows of
sum_x min(p(x), q(x)) - the exact per-position acceptance probability of the
Leviathan rule, no sampling noise. Depth d > 1 is conditioned on the previous
draft having been accepted AS THE TEXT'S TOKEN x[t+d] (the chain is fed it):
the main distribution is only known along the teacher-forced text. This stands
in for "draft sampled from q"; it is the conditional the verify step sees
whenever the draft matches the text.

Derived: expected tokens per iteration at K drafts, E_K = 1 + sum_{i<=K} prod_{j<=i} alpha_j.
Run inside the reference container (tools/oracle/run_in_container.sh).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

torch = ref = None   # loaded by _load(): --summarize runs on a host with no torch


def _load():
    global torch, ref
    import torch as _torch
    torch = _torch
    spec = importlib.util.spec_from_file_location("mtp_ref", os.path.join(_HERE, "mtp_ref.py"))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

DEPTH = 3


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]


def opencode_sources(d, max_out, ctx_max):
    turns = []
    for fn in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(fn, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("out_ids") and r.get("prompt_ids"):
            turns.append((len(r["out_ids"]), os.path.basename(fn), r))
    turns.sort(key=lambda x: -x[0])
    out, total, params = [], 0, {}
    for n, name, r in turns:
        if total >= max_out:
            break
        cont = r["out_ids"][: max_out - total]
        total += len(cont)
        out.append((f"opencode/{name}", r["prompt_ids"][-ctx_max:], cont))
        body = r.get("request") or {}
        for k in ("temperature", "top_p", "top_k"):
            if k in body and k not in params:
                params[k] = body[k]
    return out, params


def filt(logits, temp, top_k, top_p):
    """[R, V] -> (ids [R, k], probs [R, k]) after temperature, top-k, top-p (vLLM order)."""
    v, i = torch.topk(logits.float() / temp, top_k, dim=-1)       # sorted descending
    p = torch.softmax(v, -1)
    cum = p.cumsum(-1)
    keep = (cum - p) < top_p                                      # the smallest set reaching top_p
    p = p * keep
    return i, p / p.sum(-1, keepdim=True)


def overlap(pi, pp, qi, qp):
    m = pi[:, :, None] == qi[:, None, :]
    return (torch.minimum(pp[:, :, None], qp[:, None, :]) * m).sum((1, 2))


def expected_tokens(alphas):
    e, prod, out = 1.0, 1.0, []
    for a in alphas:
        prod *= a
        e += prod
        out.append(e)
    return out


def run_source(model, head, name, ctx, cont, sp, controls):
    t0 = time.time()
    x = ctx + cont
    T, c0 = len(x), len(ctx)
    logits, pre, post = ref.main_forward(model, x)
    t_main = time.time() - t0
    g = logits.float().argmax(-1)                     # g[i]: main's greedy prediction of x[i+1]
    xt = torch.tensor(x)
    R = T - 1                                         # head rows t = 0..T-2
    rows = torch.arange(R)

    def shifted(k):                                   # x[t+k] for rows t, 0 past the end
        s = torch.zeros(R, dtype=torch.long)
        n = max(0, T - k)
        s[:n] = xt[k:k + n]
        return s

    feeds = [shifted(d + 1) for d in range(1, DEPTH)]
    chain = head.chain(post[:R], shifted(1), rows, DEPTH, feed_ids=feeds)
    res = {"n_ctx": c0, "n_cont": len(cont), "t_main_s": round(t_main, 1)}

    # greedy
    gr = []
    cond = rows >= c0 - 1
    for d in range(1, DEPTH + 1):
        valid = cond & (rows + d <= T - 1)
        tgt = torch.zeros(R, dtype=torch.long)
        n = int((rows + d <= T - 1).sum())
        tgt[:n] = g[d:d + n]
        hit = (chain[d - 1][0].float().argmax(-1) == tgt) & valid
        a, c = int(hit.sum()), int(valid.sum())
        gr.append({"accepted": a, "counted": c, "alpha": a / c if c else None})
        # next depth: this one accepted, and the text agrees with the accepted draft
        m = min(n, T - d - 1)
        agree = torch.zeros(R, dtype=torch.bool)
        agree[:m] = xt[d + 1:d + 1 + m] == g[d:d + m]
        cond = hit & agree
    res["greedy"] = gr

    # sampled
    sa = []
    base = (rows >= c0 - 1)
    for d in range(1, DEPTH + 1):
        valid = base & (rows + d <= T - 1)
        idx = rows[valid]
        if len(idx) == 0:
            sa.append({"counted": 0, "alpha": None})
            continue
        pi, pp = filt(logits[idx + d], **sp)
        qi, qp = filt(chain[d - 1][0][idx], **sp)
        ov = overlap(pi, pp, qi, qp)
        sa.append({"counted": len(idx), "sum": float(ov.sum()), "alpha": float(ov.mean())})
    res["sampled"] = sa

    if controls:
        res["controls"] = {}
        for w, cfg in ref.WIRINGS.items():
            h = pre if cfg["hidden"] == "pre" else post
            (lg, _), = head.chain(h[:R], shifted(1), rows + cfg["pos_shift"], 1,
                                  hidden_first=cfg["order"] == "hidden_first")
            valid = rows >= c0 - 1
            hit = (lg.float().argmax(-1) == g[1:T]) & valid
            res["controls"][w] = {"accepted": int(hit.sum()), "counted": int(valid.sum()),
                                  "alpha": int(hit.sum()) / int(valid.sum())}
    res["t_total_s"] = round(time.time() - t0, 1)
    ga = [r["alpha"] for r in gr]
    print(f"[{name}] ctx {c0} cont {len(cont)}: greedy a1..3 "
          + " ".join(f"{a:.3f}" if a is not None else "-" for a in ga)
          + " | sampled " + " ".join(f"{r['alpha']:.3f}" if r["alpha"] is not None else "-" for r in sa)
          + (" | controls " + " ".join(f"{w}={v['alpha']:.3f}" for w, v in res.get("controls", {}).items()))
          + f" | main fwd {t_main:.0f}s total {res['t_total_s']:.0f}s", flush=True)
    return res


def pooled(results):
    g, s = [], []
    for d in range(DEPTH):
        a = sum(r["greedy"][d]["accepted"] for r in results)
        c = sum(r["greedy"][d]["counted"] for r in results)
        g.append(a / c if c else None)
        n = sum(r["sampled"][d]["counted"] for r in results)
        t = sum(r["sampled"][d].get("sum", 0.0) for r in results)
        s.append(t / n if n else None)
    return {"greedy_alpha": g, "sampled_alpha": s,
            "greedy_E_K": expected_tokens([a or 0 for a in g]),
            "sampled_E_K": expected_tokens([a or 0 for a in s]),
            "n_cont": sum(r["n_cont"] for r in results)}


def summarize(files, out):
    per = {}
    for fn in files:
        with open(fn, encoding="utf-8") as f:
            per.update(json.load(f)["per_source"])
    groups = {}
    for name in per:
        groups.setdefault(name.split("/")[0], []).append(per[name])
    summary = {k: pooled(v) for k, v in groups.items()}
    for k, v in summary.items():
        print(f"== {k}: {len(groups[k])} sources, {v['n_cont']} cont tokens; greedy a {v['greedy_alpha']} "
              f"E_K {v['greedy_E_K']}; sampled a {v['sampled_alpha']} E_K {v['sampled_E_K']}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_source": per}, f, indent=1)


def main() -> None:
    if "--summarize" in sys.argv:
        i, o = sys.argv.index("--summarize"), sys.argv.index("--out")
        return summarize(sys.argv[i + 1:o], sys.argv[o + 1])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--opencode")
    ap.add_argument("--ctx-max", type=int, default=2048)
    ap.add_argument("--max-out", type=int, default=2000)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--top-k", type=int)
    args = ap.parse_args()
    t0 = time.time()
    _load()
    torch.set_grad_enabled(False)
    print(f"torch threads {torch.get_num_threads()}", flush=True)

    with open(os.path.join(args.snapshot, "generation_config.json"), encoding="utf-8") as f:
        gc = json.load(f)
    sp = {"temp": gc.get("temperature", 1.0), "top_k": gc.get("top_k", 20), "top_p": gc.get("top_p", 1.0)}
    sp_from = "generation_config.json"
    sources = []
    for s in args.source:
        name, cf, xf = s.split(":")
        sources.append((name, read_ids(cf)[-args.ctx_max:], read_ids(xf)))
    if args.opencode:
        oc, params = opencode_sources(args.opencode, args.max_out, args.ctx_max)
        sources += oc
        if params:
            sp.update({"temp": params.get("temperature", sp["temp"]), "top_k": params.get("top_k", sp["top_k"]),
                       "top_p": params.get("top_p", sp["top_p"])})
            sp_from = "opencode requests"
    if args.temperature is not None:
        sp["temp"], sp_from = args.temperature, "command line"
    if args.top_p is not None:
        sp["top_p"] = args.top_p
    if args.top_k is not None:
        sp["top_k"] = args.top_k
    if not sp["top_k"] or sp["top_k"] <= 0:
        sp["top_k"] = 1000                                   # "off": wide enough for the overlap sum
    print(f"sampling: {sp} (from {sp_from}); {len(sources)} sources", flush=True)

    model, tc = ref.build_main(args.snapshot)
    head = ref.MtpHead.from_snapshot(args.snapshot, tc, model.model.embed_tokens.weight, model.lm_head.weight)
    print(f"loaded {time.time() - t0:.0f}s", flush=True)

    per = {}
    for name, ctx, cont in sources:
        per[name] = run_source(model, head, name, ctx, cont, sp, args.controls)
        with open(args.out + ".partial", "w", encoding="utf-8") as f:
            json.dump({"per_source": per}, f, indent=1)
    groups = {}
    for name in per:
        groups.setdefault(name.split("/")[0], []).append(per[name])
    summary = {k: pooled(v) for k, v in groups.items()}
    for k, v in summary.items():
        print(f"== {k}: {v['n_cont']} cont tokens; greedy a {v['greedy_alpha']} E_K {v['greedy_E_K']}; "
              f"sampled a {v['sampled_alpha']} E_K {v['sampled_E_K']}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"sampling": sp, "sampling_from": sp_from, "summary": summary, "per_source": per}, f, indent=1)
    print(f"wrote {args.out}; wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
