#!/usr/bin/env python3
"""The tensor-parallel twin count (docs/probes/tp-twin-count-2026-11-10.md): every FIRST divergence of each TP path
from its one-card twin and of each pair of one card's own legitimate paths, judged by the f32 judge.

  twin_count.py --dir D --dump DIR --judge DIR [--json FILE] [--vocab-used 248077]

D is tools/probe/tp_twin_count.sh's output: <D>/<kv>/decode/<arm>.r1/requests.jsonl (b70-decode arms, through
serve_client.py record) and <D>/<kv>/serve/<arm>.r1/requests.jsonl (b70-serve arms), kv in bf16 / int8.

A pair (A, B) of arms on one context and KV form: identical, or the first index d where the streams differ. That
divergence is one forced context - the prompt + A's ids [0, d), one forced id A[d] - key tp_twin_<set>_<name>_<FNV-1a
64> exactly as golden::ForcedJudge::key, dumped to --dump (<key>.prompt.ids / .force.ids: what
tools/box_validate/f32_refs.sh forced-qwen with QWEN_KEYS_RE='^tp_twin_' turns into <key>.golden.safetensors in
--judge, the int8-head judges). Its row 0 is the f32 decision at d; each side's id is accepted or rejected by
judge_row's one-bf16-ulp rule (serve_judge.py), a rejected id printed with its gap below the f32 top in bf16 ulp.

The count: TP's rejected ids (the TP side of the TP twin pairs) against one card's own-path rejected ids (either side
of one card's own pairs); TP passes when its count is no more. Printed per path and KV form with the pairs, the
divergences and the rates per divergence and per pair; one-card-side rejections inside the TP pairs are printed, not
counted. A divergence without its reference counts on neither side: the verdict is AWAITING until none is left.
Exit 0 PASS, 1 FAIL, 77 awaiting references.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import serve_client  # noqa: E402
from serve_judge import VOCAB_USED, Judge, dump, forced_key, judge_row  # noqa: E402

TAG = "tp_twin"
KVS = ("bf16", "int8")
# (class, path, source, reference arm, other arm): "tp" pairs count the other (TP) side, "own" pairs either side.
PAIRS = (
    ("tp", "TP prefill", "decode", "one-pf", "tp-pf"),
    ("tp", "TP plain decode", "decode", "one-in", "tp-in"),
    ("tp", "TP MTP", "decode", "one-mtp", "tp-mtp"),
    ("tp", "served TP --spec off", "serve", "one", "tp"),
    ("tp", "served TP --spec auto", "serve", "one-auto", "tp-auto"),
    ("own", "prefill vs token by token", "decode", "one-in", "one-pf"),
    ("own", "MTP vs plain", "decode", "one-pf", "one-mtp"),
    ("own", "served --spec off vs --spec auto", "serve", "one", "one-auto"),
)


def load(d):
    """{(kv, source, arm): {(set, name): record}}"""
    out = {}
    for kv in KVS:
        for src in ("decode", "serve"):
            for r in serve_client.records(os.path.join(d, kv, src)):
                out.setdefault((kv, src, r["label"]), {})[(r["set"], r["name"])] = r
    return out


def verdict_of(a, prompt, x, y, d):
    """One first divergence judged: dict with the key, both ids, accepted flags and gaps (None: awaiting)."""
    ctx, force = list(prompt) + list(x[:d]), [x[d]]
    k = forced_key(f"{TAG}_{a['set']}_{a['name']}", ctx, force)
    dump(a["dump"], k, ctx, force)
    out = {"key": k, "at": d, "ref_id": x[d], "got_id": y[d], "prompt": len(prompt)}
    path = os.path.join(a["judge"], k + ".golden.safetensors")
    if not os.path.exists(path):
        out["awaiting"] = True
        return out
    j = Judge(path, force)
    row = j.row(0)
    top, s, ulp = judge_row(row, min(a["used"], j.V))
    acc = (lambda i: i == top) if len(s) == 1 else (lambda i: i in s)
    gap = lambda i: (float(row[top]) - float(row[i])) / ulp  # noqa: E731
    out.update(awaiting=False, top=int(top), ref_ok=bool(acc(x[d])), got_ok=bool(acc(y[d])),
               ref_gap=gap(x[d]), got_gap=gap(y[d]), tie=len(s) > 1)
    return out


def run(a):
    recs = load(a.dir)
    opt = {"dump": a.dump, "judge": a.judge, "used": a.vocab_used}
    rows = []
    for kv in KVS:
        for cls, path, src, ra, rb in PAIRS:
            A, B = recs.get((kv, src, ra)), recs.get((kv, src, rb))
            if not A or not B:
                print(f"MISSING {kv} {path}: arm {ra if not A else rb} has no records")
                rows.append({"kv": kv, "class": cls, "path": path, "missing": True})
                continue
            for key in sorted(A):
                r0, r1 = A[key], B.get(key)
                base = {"kv": kv, "class": cls, "path": path, "ctx": f"{key[0]}/{key[1]}", "ref": ra, "got": rb}
                if r1 is None:
                    print(f"MISSING {kv} {path} {key[0]}/{key[1]} in {rb}")
                    rows.append(dict(base, missing=True))
                    continue
                p0, p1 = serve_client_prompt(r0), serve_client_prompt(r1)
                if p0 is None or p0 != p1:
                    print(f"PROMPTS {kv} {path} {key[0]}/{key[1]}: the two arms' prompt ids differ or are unknown")
                    rows.append(dict(base, missing=True))
                    continue
                x, y = r0["out_ids"], r1["out_ids"]
                n = min(len(x), len(y))
                d = next((i for i in range(n) if x[i] != y[i]), n)
                if d == n:
                    rows.append(dict(base, identical=len(x) == len(y), length=len(x) != len(y)))
                    if len(x) != len(y):
                        print(f"LENGTH {kv} {path} {key[0]}/{key[1]}: {len(x)} vs {len(y)} ids, one a prefix")
                    continue
                v = verdict_of(dict(opt, set=key[0], name=key[1]), p0, x, y, d)
                rows.append(dict(base, **v))
                if v["awaiting"]:
                    print(f"AWAIT {kv} | {path} | {key[0]}/{key[1]} at {d}: {ra} {x[d]} / {rb} {y[d]} - {v['key']}")
                    continue
                sides = []
                for who, i, ok, g in ((ra, x[d], v["ref_ok"], v["ref_gap"]), (rb, y[d], v["got_ok"], v["got_gap"])):
                    sides.append(f"{who} {i} " + ("accepted" if ok else f"REJECTED ({g:.2f} ulp below)"))
                print(f"DIV {kv} | {path} | {key[0]}/{key[1]} at {d} (prompt {len(p0)}): f32 top {v['top']}"
                      f"{' (a 1-ulp tie)' if v['tie'] else ''}; " + "; ".join(sides))
    return report(rows, a)


def serve_client_prompt(r):
    if r.get("prompt_ids"):
        return list(r["prompt_ids"])
    return None


def report(rows, a):
    def agg(sel):
        s = [r for r in rows if sel(r) and not r.get("missing")]
        div = [r for r in s if "at" in r]
        judged = [r for r in div if not r.get("awaiting")]
        return {
            "pairs": len(s), "identical": sum(1 for r in s if r.get("identical")),
            "divergences": len(div), "awaiting": len(div) - len(judged),
            "both_ok": sum(1 for r in judged if r["ref_ok"] and r["got_ok"]),
            "got_rej": sum(1 for r in judged if not r["got_ok"]),
            "ref_rej": sum(1 for r in judged if not r["ref_ok"]),
            "missing": sum(1 for r in rows if sel(r) and r.get("missing")),
        }

    def counted(c, cls):   # TP pairs: the TP (other) side; own pairs: either side
        return c["got_rej"] if cls == "tp" else c["got_rej"] + c["ref_rej"]

    print("\n| class | path | KV | pairs | identical | first divergences | both accepted | rejected (counted) |"
          " other side rejected | awaiting |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    table = []
    for cls in ("tp", "own"):
        for _, path, _, _, _ in [p for p in PAIRS if p[0] == cls]:
            for kv in KVS:
                c = agg(lambda r, kv=kv, path=path: r["kv"] == kv and r["path"] == path)
                other = c["ref_rej"] if cls == "tp" else "-"
                print(f"| {cls} | {path} | {kv} | {c['pairs']} | {c['identical']} | {c['divergences']} | "
                      f"{c['both_ok']} | {counted(c, cls)} | {other} | {c['awaiting']} |")
                table.append(dict(c, cls=cls, path=path, kv=kv, counted=counted(c, cls)))
    tot = {}
    for cls in ("tp", "own"):
        for kv in KVS + ("both",):
            c = agg(lambda r, cls=cls, kv=kv: r["class"] == cls and (kv == "both" or r["kv"] == kv))
            tot[(cls, kv)] = dict(c, counted=counted(c, cls))
    print("\n| class | KV | pairs | first divergences | rejected (counted) | per divergence | per pair | awaiting |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for (cls, kv), c in tot.items():
        jd = c["divergences"] - c["awaiting"]
        pd = f"{c['counted'] / jd:.3f}" if jd else "-"
        pp = f"{c['counted'] / c['pairs']:.4f}" if c["pairs"] else "-"
        print(f"| {cls} | {kv} | {c['pairs']} | {c['divergences']} | {c['counted']} | {pd} | {pp} | {c['awaiting']} |")
    t, o = tot[("tp", "both")], tot[("own", "both")]
    missing = t["missing"] + o["missing"]
    if t["awaiting"] or o["awaiting"] or missing:
        verdict, rc = "AWAITING", 77
        why = f"{t['awaiting'] + o['awaiting']} divergence(s) await their f32 reference, {missing} pair(s) missing"
    else:
        ok = t["counted"] <= o["counted"]
        verdict, rc = ("PASS", 0) if ok else ("FAIL", 1)
        why = f"TP rejected {t['counted']} <= one card's own {o['counted']}" if ok else \
            f"TP rejected {t['counted']} > one card's own {o['counted']}"
        jt, jo = t["divergences"], o["divergences"]
        if jt and jo:
            rt, ro = t["counted"] / jt, o["counted"] / jo
            agree = (rt <= ro) == ok
            why += (f"; per divergence {rt:.3f} vs {ro:.3f}, per pair {t['counted'] / t['pairs']:.4f} vs "
                    f"{o['counted'] / o['pairs']:.4f}" + ("" if agree else " - the normalised rate DISAGREES with "
                                                                         "the raw count: to the operator"))
    print(f"\nTWIN COUNT {verdict}: {why}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"rows": rows, "table": table, "verdict": verdict, "why": why}, f, indent=1)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="tools/probe/tp_twin_count.sh's output")
    ap.add_argument("--dump", required=True, help="where the forced contexts are written")
    ap.add_argument("--judge", required=True, help="the int8-head f32 references (<key>.golden.safetensors)")
    ap.add_argument("--json", default="", help="the rows, the table and the verdict as JSON")
    ap.add_argument("--vocab-used", type=int, default=VOCAB_USED)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
