#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 26a Task 5: the P0 projection for the LiLiCorr drafter on our Bolt - tokens per plain decode step for LiLiCorr
and for prompt lookup, per corpus and drafter arm, and the DEFAULTS P0 records (not a gate: the operator's 2026-10-15
ruling builds 26b / 26c whatever this shows).

    lilicorr_p0_project.py --accept DIR [--dumps DIR] [--probe LOG [--depth D] [--plain-ms MS]] [--union JSON]
                           [--sampled lattice|lattice+lp] [--json OUT] [--md OUT]

Inputs:
  --accept DIR   dflash_accept.py run --model bolt-lilicorr's accept.<arm>.json (greedy E_K, K = 1..7; with --sampled the
                 expected-acceptance E_K of that q form)
  --dumps DIR    the same tap dumps: prompt lookup on the SAME continuations (lookup_accept.py's AdaptiveK, n = 3, K <= 7,
                 the same verify costs, a free draft)
  --probe LOG    probe_draft_cost --shapes bolt-lilicorr's output: the verify per M at --depth (default the first it ran;
                 GEMVs + plain experts + head + the attention estimate + the state's slot traffic), the time per expert
                 block, the state's slot / replay terms and the three LiLiCorr blocks - replaces the DERIVED table
  --plain-ms MS  the measured plain decode step at that depth (row 60's r60.speed / r60.p0): the probe times the GEMV
                 and state part only, so with it verify(M) = 1 + (probe(M) - probe(1)) / MS and a block = ms / MS; without
                 it the probe's own M = 1 total stands for the plain step (the ratios overstate the verify's growth)
  --union JSON   tools/spec/expert_union.py --json: u per M (p50 / p90); the union-aware MoE's verify = the plain form's
                 minus 40 x (9 M - u) blocks (Decision 10) - at the probe's measured time per block, else DERIVED bytes

**The DERIVED cost table** (spec 26 §4, until row 63's r63.cost measures it): a plain step reads 5.49 GB at depth 4096
(spec 25 §3: the non-expert weights 4.255 GB with the int8 head, 9 experts x 40 blocks of 2.500 MB average = 0.900 GB,
the SSM state read + written 40 x 8 MiB = 0.336 GB, KV 8,192 B a position); verify(M) reads the non-expert weights once,
9 M expert blocks a MoE block (the plain M-row form) and (1 + M) x 4 MiB of state a Mamba block (19c's slots, Decision 4);
one block reads the drafter (exact 2.152 GB / int8 1.077 / int4 g64 0.572) + the LiLiCorr head 0.074 + the int8 head over
7 rows 0.537 GB.

**The projection**: LiLiCorr at K = E_K / (verify(K + 1) + block); its best K; lookup the same verify costs, its draft
free; plain = 1. SPEED-Bench's published acceptance (24GB target, K 7, T 1.0 / top-p 0.95) is printed beside ours.
**The defaults recorded** (spec 26 Decisions 2, 4, 5, 6, 10): the drafter format (int8 per row unless int4 g64's E_K is
within 0.05 of exact's at the chosen K), K_max, the q form, the replay lever (built only if the slot writes exceed 10 %
of a verify at the chosen K), the union MoE's saving, and the cost row 26c's AdaptiveK starts from.
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

KMAX = 7
GB = 1e9
NONEXPERT_GB = 4.255                       # spec 25 §14 (bolt_layout.h's bytes, the int8 head included)
BLOCK_MB = 2.500                           # an expert block's average over the 40 MoE blocks (expert_union.py)
MOE_BLOCKS, MAMBA_BLOCKS, TOPK = 40, 40, 9
STATE_MB = 4.194304                        # 4 MiB fp32 a Mamba block
KV_BYTES_POS = 8192                        # 8 attention blocks x 2 KV heads x 128 x {K, V} x bf16
DRAFTER_GB = {"exact": 2.152, "int8": 1.077, "int4g64": 0.572}
LIL_HEAD_GB, HEAD7_GB = 0.074, 0.537
SPEED_BENCH = {"coding": 5.26, "math": 4.59, "RAG": 5.13, "multilingual": 5.06, "reasoning": 4.47, "writing": 3.41,
               "overall": 4.30}                                 # published, the card (24GB target)
CORPUS_TO_BENCH = {"code": "coding", "a4": "coding", "agentic": "coding", "prose": "writing", "all": "overall"}
AGENTIC = ("a4", "agentic")


def plain_gb(depth: int = 4096) -> float:
    return NONEXPERT_GB + MOE_BLOCKS * TOPK * BLOCK_MB / 1e3 + MAMBA_BLOCKS * 2 * STATE_MB / 1e3 + KV_BYTES_POS * depth / GB


def derived_verify(depth: int = 4096, u=None) -> list:
    """verify(M) M = 1..8 in plain steps (DERIVED bytes); u: {M: distinct experts a MoE block} = the union-aware form."""
    out = []
    for M in range(1, KMAX + 2):
        ex = (u[M] if u and M in u and M > 1 else TOPK * M) * MOE_BLOCKS * BLOCK_MB / 1e3
        st = MAMBA_BLOCKS * ((1 + M) if M > 1 else 2) * STATE_MB / 1e3
        out.append((NONEXPERT_GB + ex + st + KV_BYTES_POS * depth / GB) / plain_gb(depth))
    return out


def derived_block() -> dict:
    return {f: (gb + LIL_HEAD_GB + HEAD7_GB) / plain_gb() for f, gb in DRAFTER_GB.items()}


def arm_format(arm: str) -> str:
    base = arm[:-1] if arm.endswith("h") else arm
    return base if base in DRAFTER_GB else "exact"


def _da():
    import importlib.util
    spec = importlib.util.spec_from_file_location("dflash_accept", os.path.join(_HERE, "..", "oracle", "dflash_accept.py"))
    da = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(da)
    return da


def read_accept(accept_dir: str, sampled: str | None):
    """{arm: {corpus: [E_1 .. E_7]}} pooled per corpus (dflash_accept's pool / pool_sampled)."""
    da = _da()
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


def parse_probe(path: str, depth: int | None = None, plain_ms: float | None = None) -> dict:
    """probe_draft_cost --shapes bolt-lilicorr's log -> {"verify": [8] plain steps, "block": {fmt: plain steps},
    "block_ms", "slots_share": [8], "replay_ms", "slots_ms", "t_block_ms", "plain_ms", "depth"}."""
    text = open(path, encoding="utf-8").read()
    head = re.search(r"\| M \| GEMVs \+ plain experts \+ head ms \|.*\n", text)
    if not head:
        raise ValueError(f"{path}: no verify table (probe_draft_cost --shapes bolt-lilicorr's)")
    depths = [int(x) for x in re.findall(r"verify at (\d+) ms", head.group(0))]
    if not depths:
        raise ValueError(f"{path}: the verify table has no depth columns")
    di = depths.index(depth) if depth in depths else 0
    rows = {}
    table = text[head.end():]
    nxt = re.search(r"^\| (M|arm) \|", table, re.M)        # the next table (by part / union / blocks) ends this one
    table = table[: nxt.start()] if nxt else table
    for m in re.finditer(r"^\| (\d) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|((?: [\d.]+ \| [\d.]+ \| [\d.]+ \|)+)",
                         table, re.M):
        M = int(m.group(1))
        cols = [float(x) for x in re.findall(r"[\d.]+", m.group(6))]
        rows[M] = {"gemv_ms": float(m.group(2)), "slots_ms": float(m.group(4)), "replay_ms": float(m.group(5)),
                   "attn_ms": cols[3 * di], "total_ms": cols[3 * di + 1], "ratio": cols[3 * di + 2]}
    if sorted(rows) != list(range(1, KMAX + 2)):
        raise ValueError(f"{path}: verify rows {sorted(rows)}, want M = 1..8")
    base = plain_ms if plain_ms else rows[1]["total_ms"]
    verify = [1.0 + (rows[M]["total_ms"] - rows[1]["total_ms"]) / base for M in range(1, KMAX + 2)]
    tb = re.search(r": ([\d.]+) ms a 2\.507 MB block", text)
    blocks = {}
    for m in re.finditer(r"^\| LiLiCorr block, drafter (int4-g64|int8|bf16) [^|]*\| ([\d.]+) \|", text, re.M):
        fmt = {"int4-g64": "int4g64", "bf16": "exact"}.get(m.group(1), m.group(1))
        blocks[fmt] = float(m.group(2)) / base
    return {"verify": verify, "block": blocks, "rows": rows, "t_block_ms": float(tb.group(1)) if tb else None,
            "plain_ms": base, "plain_measured": bool(plain_ms), "depth": depths[di]}


def read_union(path: str) -> dict:
    """{M: (p50, p90)} from expert_union.py --json."""
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    return {int(M): (m["u_p50"], m["u_p90"]) for M, m in r["M"].items()}


def union_verify(verify: list, union: dict, which: int, probe: dict | None, depth: int = 4096) -> list:
    """The union-aware MoE's verify: the plain form's minus 40 x (9 M - u) expert blocks (u = p50 or p90)."""
    out = list(verify)
    for M in range(2, KMAX + 2):
        if M not in union:
            continue
        saved = MOE_BLOCKS * (TOPK * M - union[M][which])
        if probe and probe.get("t_block_ms"):
            out[M - 1] -= saved * probe["t_block_ms"] / probe["plain_ms"]
        else:
            out[M - 1] -= saved * BLOCK_MB / 1e3 / plain_gb(depth)
    return out


def rate(E, verify, block):
    """[(K, E_K, cost, rate)] and the best."""
    rows = []
    for K in range(1, KMAX + 1):
        e = E[K - 1] if K - 1 < len(E) else None
        if e is None:
            continue
        cost = verify[K] + block
        rows.append((K, e, cost, e / cost))
    return rows, (max(rows, key=lambda r: r[3]) if rows else None)


def lookup_rate(seqs, verify, n=3):
    return la.run_all(seqs, KMAX, n, verify, mode="auto", max_k=KMAX).row()["speedup"]


def project(acc, seqs, verify, block):
    out = {}
    for arm, per in acc.items():
        b = block[arm_format(arm)]
        res = {}
        for corp, E in per.items():
            rows, best = rate(E, verify, b)
            lk = lookup_rate(seqs[corp], verify) if corp in seqs else None
            res[corp] = {"best_k": best[0] if best else None, "E": best[1] if best else None,
                         "lilicorr": best[3] if best else 0.0, "lookup": lk, "block": b, "rows": rows}
        out[arm] = res
    return out


def pick_arm(res: dict, base: str) -> str | None:
    """The served form of a format: with the int8 head (Decision 18) when measured, else as shipped."""
    for a in (base + "h", base):
        if a in res:
            return a
    return None


def defaults(res: dict, greedy: dict, sampled: dict, verify: list, probe: dict | None, union_gain=None) -> list[str]:
    """The lines spec 26 §"P0 verdict" records."""
    L = []
    corp = "all"
    i8, ex, i4 = pick_arm(res, "int8"), pick_arm(res, "exact"), pick_arm(res, "int4g64")
    K = None
    if i8 and corp in res[i8]:
        K = res[i8][corp]["best_k"]
    if K is None:
        return ["defaults: UNDECIDED (no int8 arm over the corpora)"]
    fmt = "int8"
    if ex and i4 and corp in greedy.get(ex, {}) and corp in greedy.get(i4, {}):
        e_ex, e_i4 = greedy[ex][corp][K - 1], greedy[i4][corp][K - 1]
        fmt = "int4g64" if e_ex is not None and e_i4 is not None and e_i4 >= e_ex - 0.05 else "int8"
        L.append(f"drafter format (Decision 2): {fmt} - at K {K} (int8's best over all) E_K exact {e_ex:.3f}, int4 g64 "
                 f"{e_i4:.3f} ({'within' if fmt == 'int4g64' else 'not within'} 0.05)")
    else:
        L.append(f"drafter format (Decision 2): int8 (the default; exact / int4 g64 not both measured)")
    best_ks = sorted({r["best_k"] for r in res[pick_arm(res, fmt) or i8].values() if r["best_k"]})
    L.append(f"K_max (Decision 5): 7 (block 8); the best K per corpus {best_ks} - AdaptiveK picks per request")
    qa = pick_arm(sampled.get("lattice", {}), fmt) if sampled else None
    if qa and "lattice+lp" in sampled and corp in sampled["lattice"].get(qa, {}):
        el = sampled["lattice"][qa][corp][K - 1]
        elp = sampled["lattice+lp"].get(qa, {}).get(corp, [None] * KMAX)[K - 1]
        q = "lattice" if elp is None or el >= elp else "lattice+lp"
        L.append(f"q form (Decision 6): {q} - sampled E_{K} lattice {el:.3f} vs lattice+lp "
                 f"{elp if elp is None else round(elp, 3)} (both lossless through accept_sparse)")
    else:
        L.append("q form (Decision 6): lattice (vLLM's; the sampled estimate not measured)")
    M = K + 1
    if probe:
        r = probe["rows"][M]
        writes = r["slots_ms"] * M / (1 + M)
        share = writes / r["total_ms"]
        L.append(f"replay lever (Decision 4): slot writes {writes:.3f} ms of a {r['total_ms']:.3f} ms verify at M {M} = "
                 f"{100 * share:.1f} % -> {'BUILD the read-only verify + replay (above 10 %)' if share > 0.10 else 'not built (<= 10 %)'}")
    else:
        bytes_w = MAMBA_BLOCKS * M * STATE_MB / 1e3
        share = bytes_w / (verify[M - 1] * plain_gb())
        L.append(f"replay lever (Decision 4): slot writes {bytes_w:.3f} GB of a {verify[M - 1] * plain_gb():.3f} GB verify "
                 f"at M {M} = {100 * share:.1f} % (DERIVED bytes) -> "
                 f"{'BUILD (above 10 %)' if share > 0.10 else 'not built (<= 10 %)'}")
    if union_gain:
        L.append(f"union MoE (Decision 10): verify at M {M} {union_gain[0]:.3f} -> {union_gain[1]:.3f} plain steps (u "
                 f"p50); built in 26c regardless")
    else:
        L.append("union MoE (Decision 10): built in 26c regardless; its saving PENDING the route records (r63.union)")
    L.append("AdaptiveK's cost row (26c, MtpCost::bolt_lilicorr): verify M = 1..8 " +
             " / ".join(f"{v:.3f}" for v in verify) + f"; the block ({fmt}) {res[pick_arm(res, fmt) or i8][corp]['block']:.3f}")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--accept", required=True)
    ap.add_argument("--dumps")
    ap.add_argument("--probe")
    ap.add_argument("--depth", type=int)
    ap.add_argument("--plain-ms", type=float)
    ap.add_argument("--union")
    ap.add_argument("--sampled", choices=("lattice", "lattice+lp"))
    ap.add_argument("--json")
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    depth = a.depth or 4096
    verify, block, src = derived_verify(depth), derived_block(), f"DERIVED (spec 26 §4, depth {depth}; r63.cost replaces it)"
    probe = None
    if a.probe:
        probe = parse_probe(a.probe, a.depth, a.plain_ms)
        verify = probe["verify"]
        block.update(probe["block"])
        src = (f"measured: {a.probe} at depth {probe['depth']} ("
               + ("the plain step " + f"{probe['plain_ms']:.3f} ms" if probe["plain_measured"] else
                  "the probe's M = 1 total as the plain step - ratios overstate the verify's growth") + ")")
    union = read_union(a.union) if a.union else None
    greedy = read_accept(a.accept, None)
    sampled = {q: read_accept(a.accept, q) for q in ("lattice", "lattice+lp")}
    acc = read_accept(a.accept, a.sampled) if a.sampled else greedy
    seqs = {}
    if a.dumps:
        allseqs = la.load_dumps(a.dumps)
        for s in allseqs:
            seqs.setdefault(s["group"], []).append(s)
        seqs["all"] = allseqs
    res = project(acc, seqs, verify, block)
    res_u = project(acc, seqs, union_verify(verify, union, 0, probe, depth), block) if union else None
    lines = [f"verify M = 1..8 in plain steps ({src}): " + " / ".join(f"{v:.3f}" for v in verify),
             "one block in plain steps: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(block.items())),
             f"E_K: {'sampled, q ' + a.sampled if a.sampled else 'greedy'}", "",
             "| arm | corpus | best K | E_K | LiLiCorr x plain | lookup x plain | LiLiCorr / lookup |"
             + (" with the union MoE (u p50) |" if res_u else ""),
             "|---|---|---:|---:|---:|---:|---:|" + ("---:|" if res_u else "")]
    for arm in sorted(res):
        for corp in sorted(res[arm], key=lambda c: (c == "all", c)):
            r = res[arm][corp]
            lk = r["lookup"]
            lines.append(f"| {arm} | {corp} | {r['best_k']} | {r['E']:.2f} | {r['lilicorr']:.3f} | "
                         + (f"{lk:.3f} | {r['lilicorr'] / lk:.3f} |" if lk else "- | - |")
                         + (f" {res_u[arm][corp]['lilicorr']:.3f} |" if res_u else ""))
    lines += ["", "Sampled acceptance (T 1.0, top-p 0.95, E_7 tokens a verify) beside SPEED-Bench's published (the 24GB "
              "target, K 7) - Decision 9's gap; the corpus-to-category map is ours (PROVISIONAL):", "",
              "| arm | corpus | ours q lattice | ours q lattice+lp | published (category) |", "|---|---|---:|---:|---:|"]
    for arm in sorted(sampled["lattice"]):
        for corp, E in sorted(sampled["lattice"][arm].items(), key=lambda kv: (kv[0] == "all", kv[0])):
            e2 = sampled["lattice+lp"].get(arm, {}).get(corp, [None] * KMAX)[KMAX - 1]
            cat = CORPUS_TO_BENCH.get(corp)
            lines.append(f"| {arm} | {corp} | {E[KMAX - 1]:.2f} | {'-' if e2 is None else f'{e2:.2f}'} | "
                         + (f"{SPEED_BENCH[cat]:.2f} ({cat}) |" if cat else "- |"))
    gain = None
    if res_u:
        i8 = pick_arm(res, "int8")
        K = res[i8]["all"]["best_k"] if i8 and "all" in res[i8] else None
        if K:
            gain = (verify[K], union_verify(verify, union, 0, probe, depth)[K])
    lines += ["", "Defaults (recorded, not a gate - spec 26 §'P0 verdict'):"]
    lines += ["- " + x for x in defaults(res, greedy, sampled, verify, probe, gain)]
    text = "\n".join(lines)
    print(text)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"verify": verify, "block": block, "source": src, "results": res, "union_results": res_u}, f,
                      indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
