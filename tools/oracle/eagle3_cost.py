#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""K2's verify cost at M rows, DERIVED from the recorded routes (the EAGLE3 K2 P0).

    eagle3_cost.py run --dumps DIR --out OUT/cost.json [--m 1-6] [--depths 0,4096,32768]
                       [--kmax 5] [--drafter-config CONFIG.json]
    eagle3_cost.py model [--drafter-config CONFIG.json]     the byte model alone (no dumps)

**What a verify of M rows reads** (bytes; the engine bandwidth-bound, a verify list that reads
each touched expert and the KV once for all M rows - spec 19a's M <= 8 verify list). Every
number is derived from model::K2Desc (src/model/k2_horizon.h; spec 18 §10's blocks), the int4
g64 checkpoint the engine loads:

  int4 g64 linear [N][K]   K N / 2 + 2 N K / 64 (GPTQ layout 0, f16 scales)
  dense layer (0-2)        fused q|k|gate|v 2560 -> 10240, o_proj 4096 -> 2560, gate||up
                           2560 -> 12288, down 6144 -> 2560 (int4), norms 2 x 2560 fp32
  sparse layer (3-47)      fused q|k|gate|v_router 2560 -> 9280, o_proj (int4), the MoE router
                           bf16 [128][2560], route biases fp32 [128 + 64], norms; then
                           u_v x 1,392,640 B (the distinct MoVA value experts of the M rows) and
                           (u + 1) x 3,133,440 B (gate||up 2,088,960 + down 1,044,480: the
                           distinct routed MoE experts + the shared one)
  top                      lm_head bf16 250624 x 2560 or int8 + fp32 row scales; final norm;
                           M embedding rows
  KV at depth d            d x 196,608 B (bf16) / 99,840 B (int8 rotkv, spec 18 §13), once

M = 1 is the plain step: u = 8, u_v = 4 -> 3.786 GB (bf16 head) / 3.145 GB (int8 head) of
weights, spec 18 §10's figures (test_eagle3_cost.py holds them). The rows of a verify at anchor
p are positions p .. p + M - 1 (the pending token and M - 1 drafts); their routes are the
RECORDED tokens' (teacher-forced: exact where the drafts are accepted, a stand-in where not).
Anchors are eagle3_accept.py's (n_prompt <= p <= n - kmax), those with p + M - 1 < n.

The drafter's bytes per draft step (eagle3_ref.drafter_bytes): body + 32k head + its own K/V
over min(depth, 2048) positions, fc once per round; per arm format (bf16 / int8 / int8 + int8
head / int4 g64 + int8 head).

`iid` columns: the expected union if every row drew its experts independently and uniformly,
E (1 - (1 - k / E)^M) - the reference for how much consecutive tokens share experts.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402

KMAX = 5
DRAFTER_CONFIG = os.path.join(_HERE, "third_party", "eagle3_k2", "config.json")


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------ the byte model


def int4g64(N: int, K: int) -> int:
    return N * K // 2 + 2 * N * (K // 64)


class K2Bytes:
    """K2Desc's numbers (the operator's int4 checkpoint)."""

    def __init__(self, layers=48, dense_layers=3, hidden=2560, q_n=4096, kv_n=1024, dense_inter=6144,
                 moe_inter=768, experts=100, top_k=8, value_experts=64, value_top_k=4, router_n=128,
                 vocab=250624, kv_pos_bf16=196608, kv_pos_int8=99840):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        H = hidden
        attn_dense_n = 2 * q_n + 2 * kv_n                     # q | k | gate | v
        attn_sparse_n = 2 * q_n + kv_n + value_experts         # q | k | gate | v_router
        norms = 2 * H * 4
        self.dense_layer = (int4g64(attn_dense_n, H) + int4g64(H, q_n) + int4g64(2 * dense_inter, H)
                            + int4g64(H, dense_inter) + norms)
        self.sparse_fixed = (int4g64(attn_sparse_n, H) + int4g64(H, q_n) + router_n * H * 2
                             + (router_n + value_experts) * 4 + norms)
        self.value_block = int4g64(kv_n, H)                   # 1,392,640
        self.moe_block = int4g64(2 * moe_inter, H) + int4g64(H, moe_inter)   # 2,088,960 + 1,044,480
        self.head = {"bf16": vocab * H * 2, "int8": vocab * H + vocab * 4}
        self.top = H * 4 + H * 2                              # final norm fp32 + one embedding row
        self.sparse_layers = layers - dense_layers

    def weights(self, u_moe: float, u_mova: float, head: str = "int8", rows: int = 1) -> float:
        """Bytes of one verify of `rows` rows whose mean distinct experts per sparse layer are
        u_moe (routed) and u_mova (value); the shared expert always."""
        s = self.sparse_layers
        return (self.dense_layers * self.dense_layer + s * self.sparse_fixed
                + s * (u_moe + 1) * self.moe_block + s * u_mova * self.value_block
                + self.head[head] + self.top + (rows - 1) * self.hidden * 2)

    def plain(self, head: str = "int8") -> float:
        return self.weights(self.top_k, self.value_top_k, head)

    def kv(self, depth: int, fmt: str = "int8") -> float:
        return depth * (self.kv_pos_int8 if fmt == "int8" else self.kv_pos_bf16)

    def iid(self, M: int):
        E, k = self.experts, self.top_k
        Ev, kv = self.value_experts, self.value_top_k
        return E * (1 - (1 - k / E) ** M), Ev * (1 - (1 - kv / Ev) ** M)


# ------------------------------------------------------------------------------ the union counter


def union_counts(ids, M: int, n_experts: int):
    """ids [R, L, k] (expert ids per row and layer) -> [R - M + 1, L]: the number of distinct
    experts over rows r .. r + M - 1, per layer."""
    import torch
    R, L, _ = ids.shape
    if M > R:
        return torch.zeros(0, L, dtype=torch.long)
    oh = torch.zeros(R, L, n_experts, dtype=torch.bool)
    oh.scatter_(2, ids.long(), True)
    acc = oh[: R - M + 1].clone()
    for m in range(1, M):
        acc |= oh[m: R - M + 1 + m]
    return acc.sum(-1)


def anchors_of(n_prompt: int, n: int, kmax: int = KMAX):
    return list(range(n_prompt, n - kmax + 1))


def source_unions(dump: dict, ms, kmax: int = KMAX, experts: int = 100, value_experts: int = 64) -> dict:
    """{M: (sum over anchors of the mean-over-layers MoE union, same for MoVA, anchors)}."""
    hf = dump["head_from"]
    out = {}
    anchors = anchors_of(dump["n_prompt"], dump["n"], kmax)
    for M in ms:
        ok = [p for p in anchors if p + M - 1 <= dump["n"] - 1]
        if not ok:
            out[M] = (0.0, 0.0, 0)
            continue
        um = union_counts(dump["moe_ids"], M, experts).double().mean(-1)         # [rows]
        uv = union_counts(dump["mova_ids"], M, value_experts).double().mean(-1)
        idx = [p - hf for p in ok]
        out[M] = (float(um[idx].sum()), float(uv[idx].sum()), len(ok))
    return out


def derive(totals: dict, model: K2Bytes, depths, drafter_cfg=None) -> dict:
    """totals {M: [moe_sum, mova_sum, anchors]} -> the cost table."""
    rows = {}
    for M, (sm, sv, n) in sorted(totals.items()):
        if not n:
            continue
        um, uv = sm / n, sv / n
        im, iv = model.iid(M)
        r = {"anchors": n, "moe_union": um, "mova_union": uv, "moe_iid": im, "mova_iid": iv}
        for head in ("bf16", "int8"):
            w = model.weights(um, uv, head, rows=M)
            r[f"bytes_{head}head"] = w
            r[f"ratio_{head}head"] = w / model.plain(head)
        r["ratio_int8head_depth"] = {str(d): (model.weights(um, uv, "int8", rows=M) + model.kv(d))
                                     / (model.plain("int8") + model.kv(d)) for d in depths}
        rows[str(M)] = r
    out = {"plain_bytes": {"bf16head": model.plain("bf16"), "int8head": model.plain("int8")},
           "kv_per_position": {"bf16": model.kv_pos_bf16, "int8": model.kv_pos_int8}, "M": rows,
           "depths": [int(d) for d in depths]}
    if drafter_cfg is not None:
        E3 = _load("eagle3_ref")
        out["drafter"] = {}
        for arm, (body, head) in E3.ARM_FORMATS.items():
            out["drafter"][arm] = {str(d): E3.drafter_bytes(drafter_cfg, body, head, kv_positions=d) for d in depths}
    return out


def draft_ratio(cost: dict, arm: str, K: int, depth: int) -> float:
    """The drafter's bytes for K drafts (fc once + K steps) over one plain step's (int8 head,
    int8 KV at `depth`)."""
    b = cost["drafter"][arm][str(depth)]
    plain = cost["plain_bytes"]["int8head"] + depth * cost["kv_per_position"]["int8"]
    return (b["fc"] + K * b["step"]) / plain


def verify_ratio(cost: dict, M: int, depth: int, scope: str = "all") -> float:
    if M == 1:
        return 1.0
    tab = cost["M"] if scope == "all" else cost["by_corpus"][scope]["M"]
    return tab[str(M)]["ratio_int8head_depth"][str(depth)]


def format_md(cost: dict) -> str:
    depths = cost["depths"]
    lines = ["| M | anchors | MoE experts / layer (iid) | MoVA (iid) | GB, int8 head | ratio bf16 head | ratio int8 head | "
             + " | ".join(f"ratio + int8 KV @ {d}" for d in depths) + " |",
             "|---:|---:|---|---|---:|---:|---:|" + "---:|" * len(depths)]
    for M, r in cost["M"].items():
        lines.append(f"| {M} | {r['anchors']} | {r['moe_union']:.2f} ({r['moe_iid']:.2f}) | {r['mova_union']:.2f} "
                     f"({r['mova_iid']:.2f}) | {r['bytes_int8head'] / 1e9:.3f} | {r['ratio_bf16head']:.3f} | "
                     f"{r['ratio_int8head']:.3f} | " + " | ".join(f"{r['ratio_int8head_depth'][str(d)]:.3f}" for d in depths) + " |")
    if "drafter" in cost:
        lines += ["", "| drafter arm | bytes per draft step (MB) at depth " + " / ".join(map(str, depths))
                  + " | fc per round (MB) | x plain step per draft at depth " + " / ".join(map(str, depths)) + " |",
                  "|---|---|---:|---|"]
        for arm, by in cost["drafter"].items():
            steps = " / ".join(f"{by[str(d)]['step'] / 1e6:.1f}" for d in depths)
            ratio = " / ".join(f"{by[str(d)]['step'] / (cost['plain_bytes']['int8head'] + d * cost['kv_per_position']['int8']):.4f}"
                               for d in depths)
            lines.append(f"| {arm} | {steps} | {by[str(depths[0])]['fc'] / 1e6:.1f} | {ratio} |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------ main


def cmd_run(a) -> None:
    import torch  # noqa: F401
    KT = _load("k2_taps")
    lo, _, hi = a.m.partition("-")
    ms = list(range(int(lo), int(hi or lo) + 1))
    depths = [int(d) for d in a.depths.split(",")]
    model = K2Bytes()
    files = sorted(glob.glob(os.path.join(a.dumps, "*.k2taps.safetensors")))
    if not files:
        raise SystemExit(f"no dumps in {a.dumps}")
    tot = {M: [0.0, 0.0, 0] for M in ms}
    by_corpus = {}
    for fn in files:
        d = KT.load_dump(fn)
        if d["moe_ids"] is None:
            raise SystemExit(f"{fn}: no routes")
        u = source_unions(d, ms, a.kmax, model.experts, model.value_experts)
        corp = by_corpus.setdefault(d["name"].split("/")[0], {M: [0.0, 0.0, 0] for M in ms})
        for M, (sm, sv, n) in u.items():
            for t in (tot[M], corp[M]):
                t[0] += sm
                t[1] += sv
                t[2] += n
    dc = _load("eagle3_ref").read_config(os.path.dirname(a.drafter_config)) if a.drafter_config else None
    out = derive(tot, model, depths, dc)
    out["by_corpus"] = {c: derive(t, model, depths) for c, t in by_corpus.items()}
    out["sources"] = len(files)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(format_md(out))
    print(f"wrote {a.out} ({len(files)} sources)")


def cmd_model(a) -> None:
    m = K2Bytes()
    print(f"K2 plain step (derived): {m.plain('bf16') / 1e9:.3f} GB bf16 head, {m.plain('int8') / 1e9:.3f} GB int8 head; "
          f"dense layer {m.dense_layer / 1e6:.2f} MB, sparse fixed {m.sparse_fixed / 1e6:.2f} MB, value expert "
          f"{m.value_block} B, MoE expert {m.moe_block} B; KV {m.kv_pos_bf16} / {m.kv_pos_int8} B per position")
    for M in range(1, 7):
        im, iv = m.iid(M)
        print(f"  M {M}: iid unions MoE {im:.2f} MoVA {iv:.2f} -> ratio (int8 head) {m.weights(im, iv, 'int8', M) / m.plain('int8'):.3f}")
    if a.drafter_config:
        E3 = _load("eagle3_ref")
        c = E3.read_config(os.path.dirname(a.drafter_config))
        for arm, (body, head) in E3.ARM_FORMATS.items():
            b = E3.drafter_bytes(c, body, head, kv_positions=2048)
            print(f"  drafter {arm}: {b['step'] / 1e6:.1f} MB per draft step at a full window "
                  f"({b['step'] / m.plain('int8'):.4f} plain steps), fc {b['fc'] / 1e6:.1f} MB per round")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dumps", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--m", default="1-6")
    r.add_argument("--depths", default="0,4096,32768")
    r.add_argument("--kmax", type=int, default=KMAX)
    r.add_argument("--drafter-config", default=DRAFTER_CONFIG)
    m = sub.add_parser("model")
    m.add_argument("--drafter-config", default=DRAFTER_CONFIG)
    a = ap.parse_args()
    (cmd_run if a.cmd == "run" else cmd_model)(a)


if __name__ == "__main__":
    main()
