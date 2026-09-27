#!/usr/bin/env python3
"""int8 per-row `lm_head` against the bf16 head, on the CPU (spec 9 P0, plan 9a).

    lm_head_probe.py dump <snapshot> --out H.safetensors --source NAME:CTX_IDS:CONT_IDS ...
        One teacher-forced forward of the main model (mtp_ref.build_main, the
        dequantised W4A16 model the golden oracle uses) per source over ctx +
        cont. Keeps the hidden AFTER the final norm (model.norm's output, the
        input to lm_head) on the decision rows len(ctx)-1 .. len(ctx)+len(cont)-1
        (every row that predicts a continuation token, plus the last one), bf16.
        Review Focus 1 check, per source: the bf16 head applied here to the kept
        hidden against the model's own logits on the same rows, cosine.
        Rewrites OUT after every source, so a killed run keeps what it finished.

    lm_head_probe.py analyze <snapshot> --hiddens H.safetensors --out R.json [--vocab-used 248077]
        No model: loads `lm_head.weight` (bf16) from the snapshot, times the
        int8 quantisation of the full [248320, 5120] tensor, and per decision row
        compares the int8 head's logits with the bf16 head's on the same hidden.

The two heads, both W?A16 with fp32 accumulation (what the engine's kernels do):
    bf16:  logits = fp32(h) @ fp32(W)^T
    int8:  logits = (fp32(h) @ fp32(q)^T) * s,  q = rne(W / s) in [-127, 127],
           s = max|W_row| / 127 in fp32 (symmetric, one scale per row; the
           hidden is NOT quantised). An all-zero row gets s = 0, q = 0.

Per row, over ids < vocab_used only (the engine never looks past it): logits
cosine, argmax equality (a mismatch is a near-tie when the int8 argmax is in the
bf16-rounded reference's top-1 set: the golden gate's tie rule), top-20 set
equality, and KL(p_bf16 || p_int8) after the engine's sample() filter
(src/cli/serve_adapters.h: top-k by logit, softmax at temperature, top-p cut
after the first prefix whose mass >= top_p), at generation_config.json's values
and at temperature 0.6. KL is +inf where p keeps an id q does not ("support
mismatch"); the unfiltered KL over all vocab_used ids at the same temperature is
reported beside it.

Run inside the reference container (tools/oracle/run_in_container.sh).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

VOCAB_USED = 248077          # model::Qwen35::kVocabUsed (src/model/qwen35.h)
LM_HEAD = "lm_head.weight"
BOUND = 0.5 + 127 * 2.0 ** -23   # |w - q*s| / s: half a step plus fp32's scale and quotient rounding
TOPN = 20                    # the checkpoint's sampling top_k: the L1 top-20 set


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"oracle_{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the two heads ------------------------------------------------------------

def quantise_rows(w: torch.Tensor, chunk: int = 16384):
    """bf16 [V, H] -> (int8 [V, H], fp32 scale [V]); symmetric, RNE, s = max|row| / 127."""
    V = w.shape[0]
    q = torch.empty(w.shape, dtype=torch.int8)
    s = torch.empty(V, dtype=torch.float32)
    for a in range(0, V, chunk):
        wf = w[a:a + chunk].float()
        sc = wf.abs().amax(1) / 127
        safe = torch.where(sc > 0, sc, torch.ones_like(sc))
        q[a:a + chunk] = torch.round(wf / safe[:, None]).clamp_(-127, 127).to(torch.int8)
        s[a:a + chunk] = sc
    return q, s


def logits_ref(h: torch.Tensor, w: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    hf = h.float()
    out = torch.empty(h.shape[0], w.shape[0], dtype=torch.float32)
    for a in range(0, w.shape[0], chunk):
        out[:, a:a + chunk] = hf @ w[a:a + chunk].float().t()
    return out


def logits_int8(h: torch.Tensor, q: torch.Tensor, s: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    hf = h.float()
    out = torch.empty(h.shape[0], q.shape[0], dtype=torch.float32)
    for a in range(0, q.shape[0], chunk):
        out[:, a:a + chunk] = (hf @ q[a:a + chunk].float().t()) * s[a:a + chunk]
    return out


# --- the sampling filter and the metrics ---------------------------------------

def filt(x: torch.Tensor, vocab_used: int, temp: float, top_k: int, top_p: float):
    """[R, V] -> (ids [R, k], probs [R, k] fp64, 0 past the top-p cut): serve_adapters.h sample()."""
    k = min(top_k if top_k > 0 else vocab_used, vocab_used)
    v, i = torch.topk(x[:, :vocab_used].float(), k, dim=-1)       # sorted descending
    p = torch.softmax(v.double() / (temp if temp > 0 else 1.0), -1)
    cum = p.cumsum(-1)
    keep = (cum - p) < min(max(top_p, 0.0), 1.0)                  # cut after the first prefix >= top_p
    keep[:, 0] = True
    p = p * keep
    return i, p / p.sum(-1, keepdim=True)


def kl_filtered(pi, pp, qi, qp):
    """KL(p || q) per row over p's kept ids; (+inf, True) where q lacks one of them."""
    m = pi[:, :, None] == qi[:, None, :]
    qv = (m * qp[:, None, :]).sum(-1)                              # q's prob of each p id
    live = pp > 0
    bad = (live & (qv <= 0)).any(1)
    term = torch.where(live, pp * (torch.log(pp.clamp_min(1e-300)) - torch.log(qv.clamp_min(1e-300))),
                       torch.zeros_like(pp))
    kl = term.sum(1)
    kl[bad] = float("inf")
    return kl, bad


def kl_full(ref, qnt, vocab_used, temp, rows=32):
    out = []
    for a in range(0, ref.shape[0], rows):
        lp = torch.log_softmax(ref[a:a + rows, :vocab_used].double() / temp, -1)
        lq = torch.log_softmax(qnt[a:a + rows, :vocab_used].double() / temp, -1)
        out.append((lp.exp() * (lp - lq)).sum(-1))
    return torch.cat(out)


def row_metrics(ref: torch.Tensor, qnt: torch.Tensor, vocab_used: int, samplings: dict) -> dict:
    r, q = ref[:, :vocab_used], qnt[:, :vocab_used]
    rd, qd = r.double(), q.double()
    cos = (rd * qd).sum(1) / (rd.norm(dim=1) * qd.norm(dim=1))
    am_r, am_q = r.argmax(1), q.argmax(1)                          # lower index on exact ties, like argmax.cl
    rb = r.to(torch.bfloat16).float()                              # the golden tie rule's bf16 values
    near_tie = rb.gather(1, am_q[:, None])[:, 0] == rb.max(1).values
    gap = r.gather(1, am_r[:, None])[:, 0] - r.gather(1, am_q[:, None])[:, 0]
    t_r = torch.topk(r, TOPN, dim=1).indices.sort(1).values
    t_q = torch.topk(q, TOPN, dim=1).indices.sort(1).values
    out = {"cos": cos, "argmax_ref": am_r, "argmax_q": am_q, "near_tie": near_tie & (am_r != am_q),
           "gap": gap, "top20_equal": (t_r == t_q).all(1)}
    for name, sp in samplings.items():
        pi, pp = filt(ref, vocab_used, **sp)
        qi, qp = filt(qnt, vocab_used, **sp)
        out[f"kl.{name}"], out[f"bad.{name}"] = kl_filtered(pi, pp, qi, qp)
        out[f"klfull.{name}"] = kl_full(ref, qnt, vocab_used, sp["temp"])
    return out


def summarise(m: dict, samplings: dict) -> dict:
    n = int(m["cos"].numel())
    mis = m["argmax_ref"] != m["argmax_q"]
    d = {"positions": n,
         "cos_min": float(m["cos"].min()), "cos_mean": float(m["cos"].mean()),
         "argmax_mismatch": int(mis.sum()), "argmax_mismatch_near_tie": int(m["near_tie"].sum()),
         "argmax_mismatch_gaps": [float(g) for g in m["gap"][mis]],
         "top20_equal": int(m["top20_equal"].sum()), "top20_equal_pct": 100.0 * float(m["top20_equal"].float().mean())}
    for name in samplings:
        kl, bad, kf = m[f"kl.{name}"], m[f"bad.{name}"], m[f"klfull.{name}"]
        d[name] = {"kl_mean": float(kl.mean()), "kl_p99": float(torch.quantile(kl.clamp_max(1e300), 0.99)),
                   "kl_max": float(kl.max()), "support_mismatch": int(bad.sum()),
                   "kl_mean_finite": float(kl[~bad].mean()) if bool((~bad).any()) else None,
                   "klfull_mean": float(kf.mean()), "klfull_p99": float(torch.quantile(kf, 0.99)),
                   "klfull_max": float(kf.max())}
    return d


# --- the two commands ------------------------------------------------------------

def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]


def cmd_dump(args):
    from safetensors.torch import save_file
    ref = _load_module("mtp_ref")
    torch.set_grad_enabled(False)
    t0 = time.time()
    print(f"torch threads {torch.get_num_threads()}", flush=True)
    try:
        from transformers import AutoTokenizer
        print(f"tokenizer len {len(AutoTokenizer.from_pretrained(args.snapshot))} (vocab_used {VOCAB_USED})", flush=True)
    except Exception as exc:  # the check is informational
        print(f"tokenizer check skipped: {exc}", flush=True)
    model, _tc = ref.build_main(args.snapshot)
    w = model.lm_head.weight.detach()
    print(f"loaded {time.time() - t0:.0f}s; lm_head {tuple(w.shape)} {w.dtype}", flush=True)
    tensors, meta = {}, {}
    side = args.out + ".check.json"
    for s in args.source:
        name, cf, xf = s.split(":")
        ctx, cont = read_ids(cf), read_ids(xf)
        t1 = time.time()
        logits, _pre, post = ref.main_forward(model, ctx + cont)
        a = len(ctx) - 1
        h = post[a:].to(torch.bfloat16).contiguous()                 # len(cont) + 1 decision rows
        own = logits[a:].float()
        mine = logits_ref(h, w)
        cos = torch.nn.functional.cosine_similarity(mine.double(), own.double(), dim=1)
        am = own[:-1, :VOCAB_USED].argmax(1)
        match = int((am == torch.tensor(cont)).sum())
        tensors[f"h/{name}"] = h
        meta[name] = {"ctx": len(ctx), "cont": len(cont), "rows": int(h.shape[0]),
                      "self_cos_min": float(cos.min()), "self_cos_mean": float(cos.mean()),
                      "own_argmax_eq_cont": match, "forward_s": round(time.time() - t1, 1)}
        print(f"{name}: {meta[name]}", flush=True)
        save_file(tensors, args.out, metadata={"sources": json.dumps(meta)})
        with open(side, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
    print(f"wrote {args.out}; wall {time.time() - t0:.0f}s", flush=True)


def cmd_analyze(args):
    from safetensors import safe_open
    ref = _load_module("mtp_ref")
    torch.set_grad_enabled(False)
    with open(os.path.join(args.snapshot, "generation_config.json"), encoding="utf-8") as f:
        gc = json.load(f)
    base = {"temp": float(gc["temperature"]), "top_k": int(gc["top_k"]), "top_p": float(gc["top_p"])}
    samplings = {f"t{base['temp']:g}": base, "t0.6": dict(base, temp=0.6)}
    print(f"samplings {samplings} (generation_config.json + temperature 0.6)", flush=True)
    w = ref.read_tensors(args.snapshot, [LM_HEAD])[LM_HEAD]
    assert w.dtype == torch.bfloat16 and tuple(w.shape) == (248320, 5120), (w.dtype, w.shape)

    nt = torch.get_num_threads()
    timing = {}
    for threads in (nt, 1):
        torch.set_num_threads(threads)
        t = time.perf_counter()
        q, s = quantise_rows(w)
        timing[f"quantise_s_{threads}thr"] = round(time.perf_counter() - t, 2)
    torch.set_num_threads(nt)
    zero_rows = int((s == 0).sum())
    bound_ok = True
    for a in range(0, w.shape[0], 16384):                           # fp64, like the unit test
        sd = s[a:a + 16384].double()[:, None]
        rt = (w[a:a + 16384].double() - q[a:a + 16384].double() * sd).abs()
        bound_ok &= bool((rt <= sd * BOUND).all())
    del rt
    print(f"quantisation {timing}; zero-scale rows {zero_rows}; round-trip bound {'ok' if bound_ok else 'FAILED'}; "
          f"bytes int8 {q.numel()} + scales {s.numel() * 4}", flush=True)

    with safe_open(args.hiddens, framework="pt", device="cpu") as fh:
        meta = json.loads(fh.metadata()["sources"])
        hid = {k[2:]: fh.get_tensor(k) for k in fh.keys()}
    per, groups = {}, {}
    for name, h in hid.items():
        m = row_metrics(logits_ref(h, w), logits_int8(h, q, s), args.vocab_used, samplings)
        per[name] = summarise(m, samplings)
        per[name]["dump"] = meta.get(name)
        groups.setdefault(name.split("/")[0], []).append(m)
        print(f"{name}: {json.dumps({k: v for k, v in per[name].items() if k != 'argmax_mismatch_gaps'})}", flush=True)
    summary = {g: summarise({k: torch.cat([m[k] for m in ms]) for k in ms[0]}, samplings) for g, ms in groups.items()}
    summary["all"] = summarise({k: torch.cat([m[k] for ms in groups.values() for m in ms]) for k in
                                next(iter(groups.values()))[0]}, samplings)
    for g, d in summary.items():
        print(f"== {g}: {json.dumps(d)}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"samplings": samplings, "vocab_used": args.vocab_used, "timing": timing,
                   "threads": nt, "zero_scale_rows": zero_rows, "round_trip_bound_ok": bound_ok,
                   "summary": summary, "per_source": per}, f, indent=1)
    print(f"wrote {args.out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("snapshot")
    d.add_argument("--out", required=True)
    d.add_argument("--source", action="append", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("snapshot")
    a.add_argument("--hiddens", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--vocab-used", type=int, default=VOCAB_USED)
    args = ap.parse_args()
    (cmd_dump if args.cmd == "dump" else cmd_analyze)(args)


if __name__ == "__main__":
    main()
