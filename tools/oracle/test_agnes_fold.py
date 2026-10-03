#!/usr/bin/env python3
"""Checks for agnes_fold.py (spec 14 §2). CPU, synthetic, seconds; no checkpoint.

    test_agnes_fold.py [fixture]   (default tests/loader/agnes_fold_fixture.safetensors)

1. the packed fold dequantises to exactly [W_gate | W_gate_p], [W_up | W_up_p]
   (columns, [K, N] = [in, out]) and [W_down ; W_down_p] (rows);
2. the folded MLP equals the two-branch MLP in fp32 (cosine >= 0.999999 per row);
3. a non-identity g_idx and a row join off a group boundary are refused;
4. the committed fixture is what the generator writes today, bit for bit (the
   C++ test, tests/loader/agnes_fold_test.cc, compares against the same file).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("agnes_fold", os.path.join(_HERE, "agnes_fold.py"))
af = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(af)


def main() -> None:
    fixture = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _HERE, "..", "..", "tests", "loader", "agnes_fold_fixture.safetensors")
    t = af.synthetic()
    f = af.fold_fixture(t)
    deq = af.dequant_gptq
    d = {n: deq(t[f"{n}.qweight"], t[f"{n}.scales"]) for n in ("gate", "up", "down", "gate_p", "up_p", "down_p")}

    # 1. exact on the weights
    assert torch.equal(deq(f["gate_fold.qweight"], f["gate_fold.scales"]), torch.cat([d["gate"], d["gate_p"]], 1))
    assert torch.equal(deq(f["up_fold.qweight"], f["up_fold.scales"]), torch.cat([d["up"], d["up_p"]], 1))
    assert torch.equal(deq(f["down_fold.qweight"], f["down_fold.scales"]), torch.cat([d["down"], d["down_p"]], 0))
    h, i, p = af.FIXTURE_DIMS["hidden"], af.FIXTURE_DIMS["inter"], af.FIXTURE_DIMS["par"]
    assert f["down_fold.qweight"].shape == ((i + p) // 8, h) and f["down_fold.scales"].shape == ((i + p) // 64, h)
    assert torch.equal(f["down_fold.qweight"][i // 8], t["down_p.qweight"][0])        # the join row
    assert torch.equal(f["down_fold.scales"][i // 64], t["down_p.scales"][0])         # a group boundary
    # interleave16: block b of 32 columns is gate'[16b:16b+16] then up'[16b:16b+16]
    iq = f["gate_up_l0.qweight"]
    assert iq.shape == (h // 8, 2 * (i + p))
    for b in range((i + p) // 16):
        assert torch.equal(iq[:, 32 * b:32 * b + 16], f["gate_fold.qweight"][:, 16 * b:16 * b + 16])
        assert torch.equal(iq[:, 32 * b + 16:32 * b + 32], f["up_fold.qweight"][:, 16 * b:16 * b + 16])
    print("1. packed fold == stacked dequant, bit-exact; interleave16 layout checked")

    # 2. the MLP identity
    x = torch.randn(32, h, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    two = af.swiglu_f32(x, d["gate"], d["up"], d["down"]) + af.swiglu_f32(x, d["gate_p"], d["up_p"], d["down_p"])
    one = af.swiglu_f32(x, deq(f["gate_fold.qweight"], f["gate_fold.scales"]),
                        deq(f["up_fold.qweight"], f["up_fold.scales"]),
                        deq(f["down_fold.qweight"], f["down_fold.scales"]))
    cos = torch.nn.functional.cosine_similarity(one.double(), two.double(), dim=-1).min().item()
    assert cos >= 0.999999, cos
    print(f"2. folded MLP vs two-branch, fp32: worst row cosine {cos:.9f}, max|d| {(one - two).abs().max():.2e}")

    # 3. refusals
    bad = t["gate.g_idx"].clone()
    bad[[0, 64]] = bad[[64, 0]]
    try:
        af.check_g_idx(bad, h, "gate")
        raise AssertionError("permuted g_idx accepted")
    except ValueError:
        pass
    try:   # 1 x 8 rows of K: not a group boundary
        af.fold_k((t["down.qweight"][:1], t["down.scales"][:1]), (t["down_p.qweight"], t["down_p.scales"]))
        raise AssertionError("off-boundary row join accepted")
    except ValueError:
        pass
    print("3. non-identity g_idx and off-boundary K join refused")

    # 4. the committed fixture
    from safetensors.torch import load_file
    disk = load_file(fixture)
    want = dict(t)
    want.update(f)
    assert set(disk) == set(want), sorted(set(disk) ^ set(want))
    for k in want:
        assert disk[k].dtype == want[k].dtype and torch.equal(disk[k], want[k]), k
    print(f"4. {os.path.basename(fixture)}: {len(disk)} tensors, identical to the generator")
    print("test_agnes_fold OK")


if __name__ == "__main__":
    main()
