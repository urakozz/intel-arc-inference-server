#!/usr/bin/env python3
"""Agnes 3.0 Flash: the parallel FFN folded into the MLP, on packed int4 (spec 14 §2).

Per layer, y = down(silu(gate x) * up x) + down_p(silu(gate_p x) * up_p x). With
gate' = [gate ; gate_p], up' = [up ; up_p] (output rows stacked, 17408 + 2048 =
19456) and down' = [down | down_p] (input columns concatenated),
y = down'(silu(gate' x) * up' x): the sum happens inside down's dot product.

On GPTQ's packed layout - `qweight` int32 [K/8, N], `scales` f16 [K/64, N] - that is:
    gate', up'  joined along N (columns):   cat(dim=1) of qweight and of scales;
    down'       joined along K (rows):      cat(dim=0), at qweight row K/8 = 2176 and
                                            scales row K/64 = 272 - a group boundary.
`g_idx` must be the identity grouping (desc_act false) or a row join would
re-group; `check_g_idx` asserts it.

This file is the reference for the C++ loader's fold (src/loader/fold.cc): both
read the shared synthetic fixture tests/loader/agnes_fold_fixture.safetensors
and must produce its folded tensors byte for byte.

    agnes_fold.py --write-fixture tests/loader/agnes_fold_fixture.safetensors
    agnes_fold.py --proof <dir> [--layers 0,3,35,71]   (real packed layers; see --help)
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("dequant", os.path.join(_HERE, "dequant.py"))
_dq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dq)
dequant_gptq = _dq.dequant_gptq

GROUP = 64


def check_packed(qweight: torch.Tensor, scales: torch.Tensor, what: str) -> tuple[int, int]:
    """[K/8, N] int32 and [K/64, N] f16 of one linear; returns (K, N)."""
    if qweight.dtype != torch.int32 or scales.dtype != torch.float16:
        raise ValueError(f"{what}: qweight {qweight.dtype} / scales {scales.dtype}, want int32 / f16")
    if qweight.dim() != 2 or scales.dim() != 2 or qweight.shape[1] != scales.shape[1]:
        raise ValueError(f"{what}: shapes {tuple(qweight.shape)} / {tuple(scales.shape)} disagree")
    k = qweight.shape[0] * 8
    if scales.shape[0] * GROUP != k:
        raise ValueError(f"{what}: {scales.shape[0]} scale rows for K = {k} at g{GROUP}")
    return k, qweight.shape[1]


def check_g_idx(g_idx: torch.Tensor, k: int, what: str) -> None:
    """desc_act false: g_idx[k] == k // 64, the identity grouping the row join needs."""
    want = torch.arange(k, dtype=torch.int32) // GROUP
    if g_idx.dtype != torch.int32 or g_idx.shape != (k,) or not torch.equal(g_idx, want):
        raise ValueError(f"{what}: g_idx is not the identity grouping k // {GROUP}")


def fold_n(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor]):
    """gate' = [gate ; gate_p]: the output columns of b appended to a's."""
    ka, _ = check_packed(*a, "fold_n a")
    kb, _ = check_packed(*b, "fold_n b")
    if ka != kb:
        raise ValueError(f"fold_n: K {ka} != {kb}")
    return torch.cat([a[0], b[0]], dim=1).contiguous(), torch.cat([a[1], b[1]], dim=1).contiguous()


def fold_k(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor]):
    """down' = [down | down_p]: b's input rows appended after a's, at a group boundary."""
    ka, na = check_packed(*a, "fold_k a")
    _, nb = check_packed(*b, "fold_k b")
    if na != nb:
        raise ValueError(f"fold_k: N {na} != {nb}")
    if ka % GROUP:
        raise ValueError(f"fold_k: K {ka} is not a whole number of g{GROUP} groups")
    return torch.cat([a[0], b[0]], dim=0).contiguous(), torch.cat([a[1], b[1]], dim=0).contiguous()


def interleave16_layout0(gate: tuple[torch.Tensor, torch.Tensor], up: tuple[torch.Tensor, torch.Tensor]):
    """The loader's gate||up: common::cols_interleave16 then repack_int4_layout0_cols.

    Output column blocks of 32: gate[16 cols] then up[16 cols], so the lane holding
    gate[n] finds up[n] at +16 (src/common/repack.h). Layout 0 keeps GPTQ's rows.
    """
    (gq, gs), (uq, us) = gate, up
    n = gq.shape[1]
    if uq.shape[1] != n or n % 16:
        raise ValueError("interleave16 needs two parts of equal, 16-divisible N")
    idx_src = []   # (part, column) per output column
    for base in range(0, n, 16):
        idx_src += [(0, base + i) for i in range(16)] + [(1, base + i) for i in range(16)]
    q = torch.stack([(gq if p == 0 else uq)[:, c] for p, c in idx_src], dim=1).contiguous()
    s = torch.stack([(gs if p == 0 else us)[:, c] for p, c in idx_src], dim=1).contiguous()
    return q, s


def swiglu_f32(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor):
    """down(silu(gate x) * up x) in fp32; weights are dequantised [K, N] (in, out)."""
    x = x.float()
    g = x @ gate_w.float()
    u = x @ up_w.float()
    return (torch.nn.functional.silu(g) * u) @ down_w.float()


# --------------------------------------------------------------------------- fixture
FIXTURE_DIMS = {"hidden": 128, "inter": 192, "par": 64}   # 2 / 3 / 1 groups of 64


def synthetic(seed: int = 14):
    """A random GPTQ layer pair at fixture size: every nibble value and sign appears."""
    g = torch.Generator().manual_seed(seed)
    h, i, p = FIXTURE_DIMS["hidden"], FIXTURE_DIMS["inter"], FIXTURE_DIMS["par"]

    def packed(k, n):
        q = torch.randint(-(2**31), 2**31 - 1, (k // 8, n), generator=g, dtype=torch.int64).to(torch.int32)
        s = (torch.rand(k // GROUP, n, generator=g) * 0.06 + 0.01).to(torch.float16)
        return q, s

    t = {}
    for name, (k, n) in {"gate": (h, i), "up": (h, i), "down": (i, h),
                         "gate_p": (h, p), "up_p": (h, p), "down_p": (p, h)}.items():
        q, s = packed(k, n)
        t[f"{name}.qweight"], t[f"{name}.scales"] = q, s
        t[f"{name}.g_idx"] = torch.arange(k, dtype=torch.int32) // GROUP
    return t


def fold_fixture(t: dict) -> dict:
    """The folded tensors the C++ test compares byte for byte."""
    pair = lambda n: (t[f"{n}.qweight"], t[f"{n}.scales"])  # noqa: E731
    for n in ("gate", "up", "down", "gate_p", "up_p", "down_p"):
        check_g_idx(t[f"{n}.g_idx"], t[f"{n}.qweight"].shape[0] * 8, n)
    gq, gs = fold_n(pair("gate"), pair("gate_p"))
    uq, us = fold_n(pair("up"), pair("up_p"))
    dq, ds = fold_k(pair("down"), pair("down_p"))
    iq, is_ = interleave16_layout0((gq, gs), (uq, us))
    return {"gate_fold.qweight": gq, "gate_fold.scales": gs, "up_fold.qweight": uq,
            "up_fold.scales": us, "down_fold.qweight": dq, "down_fold.scales": ds,
            "gate_up_l0.qweight": iq, "gate_up_l0.scales": is_}


def write_fixture(path: str) -> None:
    from safetensors.torch import save_file
    t = synthetic()
    out = dict(t)
    out.update(fold_fixture(t))
    save_file(out, path, metadata={"what": "spec 14 fold fixture: tools/oracle/agnes_fold.py --write-fixture",
                                   "dims": json.dumps(FIXTURE_DIMS)})
    print(f"wrote {path}: {len(out)} tensors")


# --------------------------------------------------------------------------- proof
def layer_tensors(get, layer: int) -> dict:
    """The six linears of one Agnes layer's MLP, packed, by role."""
    base = f"model.language_model.layers.{layer}.mlp."
    names = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj",
             "gate_p": "parallel_ffn.gate_proj", "up_p": "parallel_ffn.up_proj",
             "down_p": "parallel_ffn.down_proj"}
    t = {}
    for role, n in names.items():
        for suf in ("qweight", "scales", "g_idx"):
            t[f"{role}.{suf}"] = get(base + n + "." + suf)
    t["post_ln"] = get(f"model.language_model.layers.{layer}.post_attention_layernorm.weight")
    return t


def proof_layer(t: dict, x: torch.Tensor) -> dict:
    """Folded MLP vs two-branch MLP on the same input rows, both in fp32.

    The fold is exact up to fp32 accumulation order (spec 14 §2), so the bar is a
    per-row cosine >= 0.999999 (computed in fp64); max |diff| and the relative error
    are reported. (The bf16 rounding of each branch before the sum, which
    modeling_agnes.py has and the folded engine does not, is G2's to grade - the
    golden gates against the unfolded reference - not this algebraic check's.)
    """
    f = fold_fixture(t)
    deq = lambda q, s: dequant_gptq(q, s)  # noqa: E731   [K, N] bf16, w = (q - 8) * s
    two = (swiglu_f32(x, deq(t["gate.qweight"], t["gate.scales"]), deq(t["up.qweight"], t["up.scales"]),
                      deq(t["down.qweight"], t["down.scales"]))
           + swiglu_f32(x, deq(t["gate_p.qweight"], t["gate_p.scales"]),
                        deq(t["up_p.qweight"], t["up_p.scales"]),
                        deq(t["down_p.qweight"], t["down_p.scales"])))
    one = swiglu_f32(x, deq(f["gate_fold.qweight"], f["gate_fold.scales"]),
                     deq(f["up_fold.qweight"], f["up_fold.scales"]),
                     deq(f["down_fold.qweight"], f["down_fold.scales"]))
    # fp64 for the cosine itself: in fp32 its own rounding (~1e-7) would be the number.
    cos = torch.nn.functional.cosine_similarity(one.double(), two.double(), dim=-1)
    diff = (one - two).abs()
    # The weights themselves: the folded dequant equals the stacked dequant exactly.
    exact = (torch.equal(deq(f["gate_fold.qweight"], f["gate_fold.scales"]),
                         torch.cat([deq(t["gate.qweight"], t["gate.scales"]),
                                    deq(t["gate_p.qweight"], t["gate_p.scales"])], dim=1))
             and torch.equal(deq(f["down_fold.qweight"], f["down_fold.scales"]),
                             torch.cat([deq(t["down.qweight"], t["down.scales"]),
                                        deq(t["down_p.qweight"], t["down_p.scales"])], dim=0)))
    return {"cos_min": cos.min().item(), "max_abs": diff.max().item(),
            "rel_l2": ((one - two).norm() / two.norm()).item(), "weights_exact": exact,
            "rows": x.shape[0], "out_norm": two.norm(dim=-1).mean().item()}


def rms_inputs(post_ln: torch.Tensor, rows: int, seed: int = 0) -> torch.Tensor:
    """Synthetic MLP inputs with the real layer's scale: N(0,1) rows, RMS-normalised,
    times the checkpoint's post_attention_layernorm (1 + w) - what the MLP sees, up
    to the direction of the residual stream (real directions need a forward: pending)."""
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(rows, post_ln.shape[0], generator=g)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    return (h * (1.0 + post_ln.float())).to(torch.bfloat16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-fixture")
    ap.add_argument("--proof", help="a directory holding the layers' tensors as safetensors files "
                                    "(a snapshot, or tools/oracle/agnes_fetch_layers.py's output)")
    ap.add_argument("--layers", default="0,3,35,71")
    ap.add_argument("--inputs", help="safetensors of real MLP inputs, `mlp_in.L<i>` bf16 [T, 5120] "
                                     "(dump.py's capture); default: rms_inputs() synthetic rows")
    ap.add_argument("--rows", type=int, default=64)
    args = ap.parse_args()
    if args.write_fixture:
        write_fixture(args.write_fixture)
    if args.proof:
        from safetensors import safe_open
        handles = {}
        import glob
        for p in sorted(glob.glob(os.path.join(args.proof, "*.safetensors"))):
            h = safe_open(p, framework="pt", device="cpu")
            for k in h.keys():
                handles[k] = h
        real = None
        if args.inputs:
            real = safe_open(args.inputs, framework="pt", device="cpu")
        worst = 1.0
        for layer in [int(x) for x in args.layers.split(",")]:
            t = layer_tensors(lambda k: handles[k].get_tensor(k), layer)
            x = (real.get_tensor(f"mlp_in.L{layer}") if real is not None
                 else rms_inputs(t["post_ln"], args.rows, seed=layer))
            r = proof_layer(t, x)
            worst = min(worst, r["cos_min"])
            print(f"layer {layer:2d}: rows {r['rows']}, cos_min {r['cos_min']:.9f}, max|d| "
                  f"{r['max_abs']:.3e}, relL2 {r['rel_l2']:.3e}, weights exact {r['weights_exact']}, "
                  f"|y| {r['out_norm']:.3f}, inputs {'real' if real is not None else 'synthetic'}")
        print(f"G1 (CPU) bar cos >= 0.999999: {'PASS' if worst >= 0.999999 else 'FAIL'} (worst {worst:.9f})")
        if worst < 0.999999:
            sys.exit(1)


if __name__ == "__main__":
    main()
