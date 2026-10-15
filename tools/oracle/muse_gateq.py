#!/usr/bin/env python3
"""Spec 23a Task 7: gate Q - the GPTQ checkpoint against the bf16 original (spec 23 §3.3, decision 5's bars).

    muse_gateq.py kl <bf16 run dir> <gptq snapshot> --set <dir> --out <json> [--bf16 <bf16 snapshot>]
                     [--text NAME=IDS ...] [--only a,b] [--chunk C]
    muse_gateq.py verdict <a4 dir> <kl json>

kl (Q-KL): teacher-forced on the bf16 reference's continuations - for each scenario of the set (<name>.ids, checked
against manifest.json) the prompt + <run>/<name>.bf16.ids, and for each --text source its own ids (row 0 predicts
id 1) - the capped logits rows that predict every continuation id, per row KL(p_bf16 || p_gptq) of softmax(capped
row) at T 1.0 (fp32, from the bf16 rows the models output) and the top-1 agreement (argmax of the capped rows: the
lowest id among equals, as generate's greedy picks). The bf16 rows are cached as <run>/<name>.bf16.rows.safetensors
(made with --bf16 <snapshot> when absent - one model in memory at a time: the bf16 pass first, then the GPTQ pass);
both models are muse_ref.py's (layer-streamed, eager, the GPTQ form dequantised). Writes JSON: "a4" pooled over every
scenario row (rows, kl_mean / p50 / p99 / max, top1), "text" per --text source (recorded, not gated), "scenarios".

verdict (Q-A4 + Q-KL): reads <a4 dir>/<name>.bf16.txt and <name>.gptq.txt (oracle_generate.py --model muse, both
forms) with score.py's ATEM reader - parse failures and first-call agreement - and the kl json, then prints each bar
and `gate Q: WITHIN the proposed bars` or `gate Q: OUTSIDE the proposed bars on ...` - a RECORDED MEASUREMENT of the
checkpoint's distance from bf16 (operator ruling 2026-10-15: Vishva007's quant is kept; exit 0 either way). The bars
(PROPOSED until 23a prints the distributions):
the GPTQ form has no more parse failures than bf16's; first-call agreement >= 25 / 36 (Qwen3.8's measured int4-vs-bf16
A4 agreement); mean KL <= 0.02 and top-1 >= 0.95 on the A4 rows.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

BARS = {"agree_min": 25, "kl_mean_max": 0.02, "top1_min": 0.95}


def _load(name: str, path: str | None = None):
    spec = importlib.util.spec_from_file_location(f"gq_{name}", path or os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_ids(p: str) -> list[int]:
    return [int(x) for x in open(p, encoding="utf-8").read().split()]


def sources(set_dir: str, run: str, texts: list[str], only=None) -> list[tuple[str, str, list[int], list[int]]]:
    """[(name, kind "a4" | "text", prompt ids, continuation ids)]."""
    out = []
    with open(os.path.join(set_dir, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    for e in man:
        if only and e["name"] not in only:
            continue
        raw = open(os.path.join(set_dir, f"{e['name']}.ids"), "rb").read()
        if hashlib.sha256(raw).hexdigest() != e["sha256"]:
            sys.exit(f"FATAL: {e['name']}.ids does not match {set_dir}/manifest.json")
        cont = os.path.join(run, f"{e['name']}.bf16.ids")
        if not os.path.exists(cont):
            sys.exit(f"FATAL: {cont} is absent - the bf16 reference's continuation (a4_ref.sh muse ref)")
        out.append((e["name"], "a4", [int(x) for x in raw.split()], read_ids(cont)))
    for t in texts:
        name, _, path = t.partition("=")
        ids = read_ids(path)
        out.append((name, "text", ids[:1], ids[1:]))
    return out


def teacher_rows(ref, model, tc, prompt: list[int], cont: list[int], chunk: int):
    """The capped logits rows (bf16, as the model outputs them) predicting cont[0..C-1] after prompt: absolute rows
    T-1 .. T+C-2 of one teacher-forced pass (muse_ref.forward_chunks, the run's prefill path)."""
    import torch
    T, C = len(prompt), len(cont)
    seq = prompt + cont
    keep = []

    def on_rows(r0, _raw, cap):
        a, b = max(r0, T - 1), min(r0 + cap.shape[0], T + C - 1)
        if b > a:
            keep.append(cap[a - r0:b - r0].clone())
    rec = ref.Recorder(model, tc, acts=False)
    try:
        ref.forward_chunks(model, seq, ref.auto_chunk(len(seq), chunk), rec, on_rows=on_rows)
    finally:
        rec.remove()
    rows = torch.cat(keep, 0)
    if rows.shape[0] != C:
        raise RuntimeError(f"{rows.shape[0]} rows, expected {C}")
    return rows


def kl_rows(p_cap, q_cap):
    """Per row KL(p || q) of softmax over the capped rows at T 1.0 (fp32) and the argmax agreement."""
    import torch
    lp = torch.log_softmax(p_cap.float(), -1)
    lq = torch.log_softmax(q_cap.float(), -1)
    kl = (lp.exp() * (lp - lq)).sum(-1).clamp_min(0.0)
    return kl, p_cap.argmax(-1) == q_cap.argmax(-1)


def stats(kl, agree) -> dict:
    import torch
    k = kl.double()
    return {"rows": int(k.numel()), "kl_mean": float(k.mean()), "kl_p50": float(torch.quantile(k, 0.5)),
            "kl_p99": float(torch.quantile(k, 0.99)), "kl_max": float(k.max()), "top1": float(agree.double().mean())}


def build(ref, snap: str, layers):
    tc = ref.text_config(snap, layers)
    model, _pf = ref.build_streamed(snap, tc)
    return model, tc


def cmd_kl(a) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    ref = _load("muse_ref")
    t0 = time.time()
    srcs = sources(a.set, a.run, a.text, set(a.only.split(",")) if a.only else None)
    cache = {n: os.path.join(a.run, f"{n}.bf16.rows.safetensors") for n, *_ in srcs}

    def ids_sha(prompt, cont) -> str:      # what a cached row set was computed over (and how)
        key = f"{prompt}|{cont}|layers={a.layers}|chunk={a.chunk}"
        return hashlib.sha256(key.encode()).hexdigest()
    for name, _k, prompt, cont in srcs:   # a continuation regenerated under the cache: refuse, never compare stale rows
        if os.path.exists(cache[name]):
            with safe_open(cache[name], framework="pt") as f:
                got = (f.metadata() or {}).get("ids_sha256")
            if got != ids_sha(prompt, cont):
                sys.exit(f"FATAL: {cache[name]} was computed over other ids (stale: the prompt / continuation, --layers "
                         f"or --chunk changed) - move it aside and re-run")
    todo = [s for s in srcs if not os.path.exists(cache[s[0]])]
    if todo:
        if not a.bf16:
            sys.exit(f"FATAL: {len(todo)} sources lack cached bf16 rows ({todo[0][0]}...) - pass --bf16 <snapshot>")
        model, tc = build(ref, a.bf16, a.layers)
        for name, _k, prompt, cont in todo:
            t1 = time.time()
            rows = teacher_rows(ref, model, tc, prompt, cont, a.chunk)
            save_file({"cap": rows.contiguous()}, cache[name] + ".tmp",
                      metadata={"snapshot": os.path.abspath(a.bf16), "n_prompt": str(len(prompt)), "n_cont": str(len(cont)),
                                "ids_sha256": ids_sha(prompt, cont)})
            os.replace(cache[name] + ".tmp", cache[name])
            print(f"bf16 rows {name}: {len(prompt)} + {len(cont)} ids, {time.time() - t1:.0f}s", flush=True)
        del model
    model, tc = build(ref, a.gptq, a.layers)
    per, pool = {}, {"a4": ([], [])}
    for name, kind, prompt, cont in srcs:
        t1 = time.time()
        q = teacher_rows(ref, model, tc, prompt, cont, a.chunk)
        with safe_open(cache[name], framework="pt") as f:
            p = f.get_tensor("cap")
        if p.shape != q.shape:
            sys.exit(f"FATAL: {name}: cached bf16 rows {list(p.shape)} vs {list(q.shape)} (another continuation?)")
        kl, agree = kl_rows(p, q)
        per[name] = dict(stats(kl, agree), kind=kind)
        if kind == "a4":
            pool["a4"][0].append(kl)
            pool["a4"][1].append(agree)
        print(f"kl {name} ({kind}): rows {per[name]['rows']} kl_mean {per[name]['kl_mean']:.5g} p99 "
              f"{per[name]['kl_p99']:.5g} top1 {per[name]['top1']:.4f} ({time.time() - t1:.0f}s)", flush=True)
    out = {"a4": stats(torch.cat(pool["a4"][0]), torch.cat(pool["a4"][1])) if pool["a4"][0] else {},
           "text": {n: s for n, s in per.items() if s["kind"] == "text"},
           "scenarios": {n: s for n, s in per.items() if s["kind"] == "a4"},
           "gptq": os.path.abspath(a.gptq), "bf16": os.path.abspath(a.bf16) if a.bf16 else "cached",
           "bars": BARS, "transformers": ref.PINNED_TRANSFORMERS}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(a.out + ".tmp", a.out)
    if out["a4"]:
        print(f"A4 pooled: rows {out['a4']['rows']} kl_mean {out['a4']['kl_mean']:.5g} p50 {out['a4']['kl_p50']:.5g} "
              f"p99 {out['a4']['kl_p99']:.5g} top1 {out['a4']['top1']:.4f}")
    print(f"wrote {a.out}; wall {time.time() - t0:.0f}s")


def a4_table(a4_dir: str, ref_run: str = "bf16", cand: str = "gptq") -> dict:
    """score.py's ATEM reader over <a4 dir>/<name>.{bf16,gptq}.txt: scenarios, parse failures, first-call agreement."""
    score = _load("score", os.path.join(_HERE, "..", "toolcall", "score.py"))
    names = sorted(n[: -len(f".{ref_run}.txt")] for n in os.listdir(a4_dir) if n.endswith(f".{ref_run}.txt"))
    rf = cf = agree = missing = 0
    for n in names:
        r = score.parse_first_call(open(os.path.join(a4_dir, f"{n}.{ref_run}.txt"), encoding="utf-8").read(), "atem")
        p = os.path.join(a4_dir, f"{n}.{cand}.txt")
        if not os.path.exists(p):
            missing += 1
            continue
        c = score.parse_first_call(open(p, encoding="utf-8").read(), "atem")
        rf += r[0] == "incomplete"
        cf += c[0] == "incomplete"
        agree += r[0] == "call" and c[0] == "call" and r[1:3] == c[1:3]
    return {"n": len(names), "ref_fail": rf, "cand_fail": cf, "agree": agree, "missing": missing}


def verdict(a4_dir: str, kl_json: str, bars: dict = BARS) -> tuple[bool, list[str]]:
    t = a4_table(a4_dir)
    with open(kl_json, encoding="utf-8") as f:
        kl = json.load(f)["a4"]
    checks = [
        (t["cand_fail"] <= t["ref_fail"], f"Q-A4 parse failures: gptq {t['cand_fail']} <= bf16 {t['ref_fail']}"),
        (t["agree"] >= bars["agree_min"] and not t["missing"],
         f"Q-A4 first-call agreement: {t['agree']} / {t['n']} >= {bars['agree_min']}"
         + (f" ({t['missing']} gptq outputs missing)" if t["missing"] else "")),
        (kl["kl_mean"] <= bars["kl_mean_max"], f"Q-KL mean KL(bf16 || gptq): {kl['kl_mean']:.5g} <= {bars['kl_mean_max']} "
                                               f"(p50 {kl['kl_p50']:.4g}, p99 {kl['kl_p99']:.4g}, {kl['rows']} rows)"),
        (kl["top1"] >= bars["top1_min"], f"Q-KL top-1 agreement: {kl['top1']:.4f} >= {bars['top1_min']}"),
    ]
    lines = [("PASS  " if ok else "FAIL  ") + s for ok, s in checks]
    ok = all(c for c, _ in checks)
    bad = [s.split(":")[0] for c, s in checks if not c]
    # Operator ruling 2026-10-15: the checkpoint is KEPT; gate Q is a recorded measurement of its distance from bf16.
    lines.append("gate Q: WITHIN the proposed bars (recorded; the checkpoint is kept)" if ok else
                 f"gate Q: OUTSIDE the proposed bars on {'; '.join(bad)} (recorded in spec 23 §3.3; the checkpoint is kept)")
    return ok, lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kl")
    k.add_argument("run")
    k.add_argument("gptq")
    k.add_argument("--set", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--bf16")
    k.add_argument("--text", action="append", default=[], help="NAME=IDS_FILE, teacher-forced over itself")
    k.add_argument("--only")
    k.add_argument("--chunk", type=int, default=-1)
    k.add_argument("--layers", type=int)
    v = sub.add_parser("verdict")
    v.add_argument("a4")
    v.add_argument("kl")
    a = ap.parse_args(argv)
    if a.cmd == "kl":
        cmd_kl(a)
        return
    _ok, lines = verdict(a.a4, a.kl)
    print("\n".join(lines))          # a measurement: exit 0 either way (the operator's ruling, 2026-10-15)


if __name__ == "__main__":
    main()
