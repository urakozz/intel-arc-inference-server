#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The expert union of M consecutive rows - what a verify's MoE reads (spec 26 Decision 10, plan 26a Task 3).

    expert_union.py GOLDEN... [--config tests/model/bolt/config.json] [--mmax 8] [--rows all|gen] [--json OUT]

GOLDEN: bolt_ref.py `run` outputs (`*.golden.safetensors`, or directories of them): `route.ids.L<l>` int32 [rows][9],
the ids each row routes to in MoE block l (the set, ascending). For every run of M = 2..mmax consecutive rows (every
start; `--rows gen` only rows at or past the golden's n_prompt - the decode rows) and every MoE block l, u = the number
of DISTINCT experts the M rows route to. Printed per M: u's mean / p50 / p90 over (run, block), the same per E bucket
(the block's n_routed_experts: 240 .. 496), and the routed bytes a verify reads:
  union-aware    sum_l u_l x block(I_l) - each distinct expert's block read once (spec 26 Decision 10's kernel)
  plain M-row    sum_l 9 M x block(I_l) - every row reads its own 9 blocks (the M-row form without the union)
with block(I) = 2 x I x 544 B (src/loader/bolt_layout.h: up + down at latent 1024, int4 g64 layout 1). M = 1 is the
plain decode step's 9 blocks a layer. The union's mean / p50 / p90 bytes are over the runs (each run's sum over the 40
blocks). Dependency-free Python 3: the safetensors header and the int32 tensors are read directly.
"""
import argparse
import glob
import json
import os
import struct
import sys
from array import array

TOPK = 9
LATENT = 1024
TILE = 544                        # a layout-1 tile's bytes: 64 k x 16 n nibbles + 16 f16 scales (bolt_layout.h)


def block_bytes(I: int, latent: int = LATENT) -> int:
    """bolt_layout.h's expert_block_bytes: up (K latent, N I) + down (K I, N latent), layout 1 = 2 x I x 544 at 1024."""
    return (I // 16) * (latent // 64) * TILE + (latent // 16) * (I // 64) * TILE


def read_routes(path: str) -> tuple[dict[int, list[tuple[int, ...]]], dict]:
    """{block: [row's ids]} from a golden's route.ids.L<l> tensors, and the metadata."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        meta = hdr.pop("__metadata__", {}) or {}
        out = {}
        for name, info in hdr.items():
            if not name.startswith("route.ids.L"):
                continue
            if info["dtype"] != "I32" or len(info["shape"]) != 2:
                raise ValueError(f"{path}: {name} is {info['dtype']} {info['shape']}, want I32 [rows][k]")
            rows, k = info["shape"]
            a, b = info["data_offsets"]
            f.seek(base + a)
            arr = array("i")
            arr.frombytes(f.read(b - a))
            if sys.byteorder != "little":
                arr.byteswap()
            out[int(name[len("route.ids.L"):])] = [tuple(arr[r * k:(r + 1) * k]) for r in range(rows)]
    return out, meta


def moe_latent(config_path: str) -> int:
    """moe_latent_size (1024 on Bolt) - the experts' K / N beside I."""
    with open(config_path, encoding="utf-8") as f:
        c = json.load(f)
    return int(c.get("llm_config", c).get("moe_latent_size") or LATENT)


def moe_blocks(config_path: str) -> dict[int, tuple[int, int]]:
    """{block: (E, I)} for every MoE block of a Bolt config (the omni wrapper's llm_config or the bare one)."""
    with open(config_path, encoding="utf-8") as f:
        c = json.load(f)
    lc = c.get("llm_config", c)
    out = {}
    for i, b in enumerate(lc["block_configs"]):
        if b.get("block_type") == "moe":
            out[i] = (int(b["n_routed_experts"]), int(b["moe_intermediate_size"]))
    return out


def pct(v: list, q: float):
    if not v:
        return None
    s = sorted(v)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def union_counts(routes: dict[int, list[tuple[int, ...]]], first: int, mmax: int):
    """{M: {block: [u for every start r >= first with r + M <= rows]}} - distinct ids of rows r .. r + M - 1."""
    out = {M: {} for M in range(1, mmax + 1)}
    for l, rows in routes.items():
        n = len(rows)
        for M in out:
            out[M][l] = []
        for r in range(first, n):
            s = set()
            for m in range(min(mmax, n - r)):
                s.update(rows[r + m])
                out[m + 1][l].append(len(s))
    return out


def analyse(files: list[str], blocks: dict[int, tuple[int, int]], mmax: int = 8, rows: str = "all",
            latent: int = LATENT) -> dict:
    per_M = {M: {"u": [], "by_E": {}, "run_bytes": []} for M in range(1, mmax + 1)}
    sources = []
    for path in files:
        routes, meta = read_routes(path)
        miss = sorted(set(blocks) - set(routes))
        extra = sorted(set(routes) - set(blocks))
        if extra:
            raise ValueError(f"{path}: route records for non-MoE blocks {extra} (another config?)")
        if not routes:
            raise ValueError(f"{path}: no route.ids.L* tensors (a bolt_ref.py run golden?)")
        first = int(meta.get("n_prompt", 0)) if rows == "gen" else 0
        uc = union_counts(routes, first, mmax)
        n_rows = len(next(iter(routes.values())))
        sources.append({"file": os.path.basename(path), "rows": n_rows, "from": first, "blocks": len(routes),
                        "missing_blocks": miss})
        for M, by_l in uc.items():
            starts = len(next(iter(by_l.values())))
            for l, us in by_l.items():
                E, _ = blocks[l]
                per_M[M]["u"] += us
                per_M[M]["by_E"].setdefault(E, []).extend(us)
            for r in range(starts):
                per_M[M]["run_bytes"].append(sum(by_l[l][r] * block_bytes(blocks[l][1], latent) for l in by_l))
    plain_layer = sum(TOPK * block_bytes(I, latent) for _, I in blocks.values())
    res = {"sources": sources, "plain_per_row_bytes": plain_layer, "M": {}}
    for M, d in per_M.items():
        u, rb = d["u"], d["run_bytes"]
        if not u:
            continue
        res["M"][M] = {
            "n": len(u), "u_mean": sum(u) / len(u), "u_p50": pct(u, 0.5), "u_p90": pct(u, 0.9), "u_max": max(u),
            "by_E": {E: {"mean": sum(v) / len(v), "p50": pct(v, 0.5), "p90": pct(v, 0.9)} for E, v in sorted(d["by_E"].items())},
            "union_bytes_mean": sum(rb) / len(rb), "union_bytes_p50": pct(rb, 0.5), "union_bytes_p90": pct(rb, 0.9),
            "plain_bytes": M * plain_layer}
    return res


def report(res: dict) -> str:
    L = []
    for s in res["sources"]:
        L.append(f"{s['file']}: {s['rows']} rows (from row {s['from']}), {s['blocks']} MoE blocks"
                 + (f", MISSING {s['missing_blocks']}" if s["missing_blocks"] else ""))
    Es = sorted({E for m in res["M"].values() for E in m["by_E"]})
    L.append("")
    L.append("| M | runs x blocks | u mean | u p50 | u p90 | u max | " + " | ".join(f"E {E} mean / p90" for E in Es)
             + " | routed GB union mean | p90 | plain 9M GB | union / plain |")
    L.append("|---:|---:|---:|---:|---:|---:|" + "---:|" * len(Es) + "---:|---:|---:|---:|")
    for M, m in sorted(res["M"].items(), key=lambda kv: int(kv[0])):
        L.append(f"| {M} | {m['n']} | {m['u_mean']:.2f} | {m['u_p50']} | {m['u_p90']} | {m['u_max']} | "
                 + " | ".join(f"{m['by_E'][E]['mean']:.2f} / {m['by_E'][E]['p90']}" if E in m["by_E"] else "-"
                              for E in Es)
                 + f" | {m['union_bytes_mean'] / 1e9:.3f} | {m['union_bytes_p90'] / 1e9:.3f} | "
                   f"{m['plain_bytes'] / 1e9:.3f} | {m['union_bytes_mean'] / m['plain_bytes']:.3f} |")
    L.append("")
    L.append(f"block(I) = up + down at the latent, int4 g64 layout 1 (2 x I x 544 B at Bolt's 1024); a plain row "
             f"reads 9 blocks a MoE block: {res['plain_per_row_bytes'] / 1e9:.3f} GB over the model (DERIVED from the "
             "config)")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("golden", nargs="+", help="*.golden.safetensors files or directories of them")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--config", default=os.path.join(here, "..", "..", "tests", "model", "bolt", "config.json"))
    ap.add_argument("--mmax", type=int, default=8)
    ap.add_argument("--rows", choices=("all", "gen"), default="all")
    ap.add_argument("--json")
    a = ap.parse_args()
    files = []
    for g in a.golden:
        files += sorted(glob.glob(os.path.join(g, "*.golden.safetensors"))) if os.path.isdir(g) else [g]
    if not files:
        print("expert_union: no *.golden.safetensors given", file=sys.stderr)
        return 2
    res = analyse(files, moe_blocks(a.config), a.mmax, a.rows, moe_latent(a.config))
    print(report(res))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
