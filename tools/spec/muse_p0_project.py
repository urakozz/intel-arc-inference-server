#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 23f Task 5: the P0 projection for Muse-Glimmer's DFlash drafter - tokens per plain decode step for DFlash and
for prompt lookup, per corpus and drafter arm, and spec 23 §8.3's stopping rule (recorded, not applied: the operator
ruled 2026-10-15 that the engine half is built whatever P0 shows).

    muse_p0_project.py --accept DIR [--dumps DIR] [--probe LOG] [--verify-cost 1,1.17,...] [--draft-cost bf16=0.34,...]
                       [--sampled raw|capped] [--json OUT] [--md OUT]

Inputs:
  --accept DIR      dflash_accept.py run --model muse's accept.<arm>.json (greedy E_K, K = 1..15; with --sampled the
                    expected-acceptance E_K of that q form instead)
  --dumps DIR       the same tap dumps: prompt lookup's tokens per step on the SAME continuations
                    (tools/spec/lookup_accept.py's simulate, n = 3, the AdaptiveK policy over the same verify costs with a
                    free draft - spec 19e's lookup), per corpus
  --probe LOG       probe_draft_cost --shapes muse's output: the verify's "x M = 1" column at the first --depth it ran
                    (the attention estimate included) and the drafter blocks' "x the M = 1 verify's GEMVs" ratios -
                    replaces the DERIVED table below
  --verify-cost     verify M = 1..16 in plain steps (overrides --probe's)
  --draft-cost      one block's cost in plain steps per drafter format (overrides --probe's)

**The DERIVED cost table** (until queue row 47's r47.cost measures it - spec 23 §2 / §8.2): verify M = 1..4 Qwen3.8's
measured int8-head rows 1.00 / 1.18 / 1.56 / 1.79 (spec 8 §10, the same GEMV kernels at a 5 % larger model), then +0.305
a row (lookup_accept.py's extension) to M = 16; the draft row flat in K: the drafter's weights read once (bf16 5.11 GB,
int8 2.56, int4 g64 1.36) + the target's int8 head over 15 rows (1.35 GB), at the plain step's 14.86 GB a token -
bf16 0.43, int8 0.26, int4 g64 0.18 (the "h" arms read the same).

**The projection**: DFlash at K = E_K / (verify(K + 1) + draft); its best K; lookup = ids / cost of its replay; plain = 1.
**The stopping rule** (spec 23 §8.3, superseded as a gate): "go" only if DFlash beats plain AND lookup by >= 10 % on the
coding (code) and agentic (a4 + agentic) corpora.
"""
import argparse
import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import lookup_accept as la  # noqa: E402

KMAX = 15
DERIVED_VERIFY = la.extend_verify(la.VERIFY_INT8, KMAX)          # M = 1..16
PLAIN_GB = 14.86                                                 # spec 23 §2: read per token at depth 4096
HEAD15_GB = 1.346                                                # the int8 head (spec 23 §2)
DRAFTER_GB = {"bf16": 5.112, "int8": 2.556, "int4g64": 1.358}    # spec 23 §8.2
DERIVED_DRAFT = {f: round((gb + HEAD15_GB) / PLAIN_GB, 3) for f, gb in DRAFTER_GB.items()}
CODING = ("code",)
AGENTIC = ("a4", "agentic")


def arm_format(arm: str) -> str:
    base = arm[:-1] if arm.endswith("h") else arm
    return base if base in DRAFTER_GB else "bf16"


def read_accept(accept_dir: str, sampled: str | None):
    """{arm: {corpus: [E_1 .. E_15]}} pooled per corpus (dflash_accept's pool / pool_sampled)."""
    da_spec = __import__("importlib.util").util.spec_from_file_location(
        "dflash_accept", os.path.join(_HERE, "..", "oracle", "dflash_accept.py"))
    da = __import__("importlib.util").util.module_from_spec(da_spec)
    da_spec.loader.exec_module(da)
    out = {}
    for fn in sorted(glob.glob(os.path.join(accept_dir, "accept.*.json"))):
        with open(fn, encoding="utf-8") as f:
            r = json.load(f)
        for arm, srcs in r["variants"].items():
            groups = {}
            for name, e in srcs.items():
                groups.setdefault(name.split("/")[0], []).append(e)
            groups["all"] = list(srcs.values())
            per = {}
            for corp, es in groups.items():
                if sampled:
                    sm = da.pool_sampled(es, KMAX).get(sampled)
                    if sm:
                        per[corp] = sm["E"]
                else:
                    p = da.pool(es, range(1, KMAX + 1))
                    per[corp] = [p[str(K)]["E"] if str(K) in p else None for K in range(1, KMAX + 1)]
            out[arm] = per
    return out


def parse_probe(path: str):
    """(verify [16] in plain steps, {format: draft cost}) from probe_draft_cost --shapes muse's log: the single-pass
    verify rows' first "x M = 1" (the verify total with the attention estimate at the first depth), the better of the
    one- and two-pass forms at M >= 9; the drafter rows' ratio to the M = 1 GEMV pass, scaled to the M = 1 verify's
    total at that depth."""
    verify = [None] * 16
    two = [None] * 16
    draft = {}
    one_gemv = one_total = None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\| verify M (\d+)( \(two passes[^)]*\))? \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|",
                     line)
        if m:
            M = int(m.group(1))
            ratio_total = float(m.group(7))
            (two if m.group(2) else verify)[M - 1] = ratio_total
            if M == 1 and not m.group(2):
                one_gemv, one_total = float(m.group(3)), float(m.group(6))
            continue
        m = re.match(r"\| drafter block (bf16|int8|int4-g64) \+ int8 head x15 \| ([\d.]+) \| ([\d.]+) \|", line)
        if m and one_gemv:
            fmt = {"int4-g64": "int4g64"}.get(m.group(1), m.group(1))
            draft[fmt] = float(m.group(2)) / one_total
    if verify[0] is None:
        raise ValueError(f"{path}: no 'verify M 1' row (probe_draft_cost --shapes muse's table)")
    verify = [min(v, t) if t is not None else v for v, t in zip(verify, two)]
    if any(v is None for v in verify):
        raise ValueError(f"{path}: verify rows missing: {[i + 1 for i, v in enumerate(verify) if v is None]}")
    return verify, draft


def dflash_rate(E, verify, draft):
    """[(K, E_K, cost, rate)] and the best (K, rate)."""
    rows = []
    for K in range(1, KMAX + 1):
        e = E[K - 1]
        if e is None:
            continue
        cost = verify[K] + draft
        rows.append((K, e, cost, e / cost))
    best = max(rows, key=lambda r: r[3]) if rows else None
    return rows, best


def lookup_rate(seqs, verify, n=3):
    """AdaptiveK lookup (free drafts) over the same verify costs, K <= 15: ids per plain-step cost."""
    tot = la.run_all(seqs, KMAX, n, verify, mode="auto", max_k=KMAX).row()
    return tot["speedup"], tot


def verdict(table):
    """spec 23 §8.3's rule over {corpus: {"dflash": rate, "lookup": rate}}."""
    def ok(corps):
        rs = [table[c] for c in corps if c in table]
        if not rs:
            return None
        return all(r["dflash"] >= 1.10 and r["dflash"] >= 1.10 * r["lookup"] for r in rs)
    c, a = ok(CODING), ok(AGENTIC)
    if c is None or a is None:
        return "UNDECIDED (no " + ("coding" if c is None else "agentic") + " corpus measured)"
    return "GO" if (c and a) else "NO GO"


def project(acc, seqs_by_corpus, verify, draft_cost, arm=None):
    out = {}
    for a, per in acc.items():
        if arm and a != arm:
            continue
        d = draft_cost.get(arm_format(a), DERIVED_DRAFT[arm_format(a)])
        res = {}
        for corp, E in per.items():
            rows, best = dflash_rate(E, verify, d)
            lk = lookup_rate(seqs_by_corpus[corp], verify)[0] if corp in seqs_by_corpus else None
            res[corp] = {"best_k": best[0] if best else None, "E": best[1] if best else None,
                         "dflash": best[3] if best else 0.0, "lookup": lk if lk is not None else 0.0,
                         "draft_cost": d, "rows": rows}
        out[a] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--accept", required=True)
    ap.add_argument("--dumps")
    ap.add_argument("--probe")
    ap.add_argument("--verify-cost", type=la.parse_floats)
    ap.add_argument("--draft-cost", help="fmt=cost,... (bf16, int8, int4g64)")
    ap.add_argument("--sampled", choices=("raw", "capped"))
    ap.add_argument("--json")
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    verify, draft, src = list(DERIVED_VERIFY), dict(DERIVED_DRAFT), "DERIVED (spec 23 §2 / §8.2; r47.cost replaces it)"
    if a.probe:
        verify, pd = parse_probe(a.probe)
        draft.update(pd)
        src = f"measured: {a.probe}"
    if a.verify_cost:
        if len(a.verify_cost) < KMAX + 1:
            ap.error(f"--verify-cost needs M = 1..{KMAX + 1}")
        verify, src = a.verify_cost, "--verify-cost"
    if a.draft_cost:
        for kv in a.draft_cost.split(","):
            k, v = kv.split("=")
            draft[k] = float(v)
    acc = read_accept(a.accept, a.sampled)
    seqs = {}
    if a.dumps:
        allseqs = la.load_dumps(a.dumps)
        for s in allseqs:
            seqs.setdefault(s["group"], []).append(s)
        seqs["all"] = allseqs
    res = project(acc, seqs, verify, draft)
    lines = [f"verify M = 1..16 ({src}): " + " / ".join(f"{v:.3g}" for v in verify),
             "draft (one block, plain steps): " + ", ".join(f"{k} {v:.3g}" for k, v in sorted(draft.items())),
             f"E_K: {'sampled, q ' + a.sampled if a.sampled else 'greedy'}", "",
             "| arm | corpus | best K | E_K | DFlash x plain | lookup x plain | DFlash / lookup |",
             "|---|---|---:|---:|---:|---:|---:|"]
    verdicts = {}
    for arm in sorted(res):
        table = {}
        for corp in sorted(res[arm], key=lambda c: (c == "all", c)):
            r = res[arm][corp]
            table[corp] = r
            ratio = r["dflash"] / r["lookup"] if r["lookup"] else float("nan")
            lines.append(f"| {arm} | {corp} | {r['best_k']} | {r['E']:.2f} | {r['dflash']:.3f} | "
                         f"{r['lookup']:.3f} | {ratio:.3f} |")
        verdicts[arm] = verdict(table)
    lines.append("")
    for arm, v in sorted(verdicts.items()):
        lines.append(f"stopping rule (spec 23 §8.3, recorded - the engine half is built regardless): {arm}: {v}")
    text = "\n".join(lines)
    print(text)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"verify": verify, "draft": draft, "source": src, "results": res, "verdicts": verdicts}, f,
                      indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
