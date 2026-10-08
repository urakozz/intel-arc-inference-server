#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for eagle3_cost.py (the EAGLE3 K2 P0). CPU, seconds.

    test_eagle3_cost.py [test_name ...]

- the expert-union counter == Python sets over every window of M rows, every layer;
- the byte model: K2Desc's blocks (value 1,392,640 B, MoE gate||up + down 3,133,440 B) and
  spec 18 §10's plain step (3.786 GB bf16 head, 3.145 GB int8 head); M = 1 costs 1.0; the iid
  union formula's limits;
- source_unions' anchors (n_prompt <= p <= n - kmax, p + M - 1 < n) against a direct count;
- the drafter's bytes and the S_K pieces (verify_ratio, draft_ratio) on a derived table.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("eagle3_cost")
E3 = _load("eagle3_ref")


def test_union_counter():
    g = torch.Generator().manual_seed(1)
    R, L, k, E = 23, 5, 3, 9
    ids = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(R * L)]).view(R, L, k)
    for M in range(1, 7):
        u = C.union_counts(ids.to(torch.uint8), M, E)
        assert u.shape == (R - M + 1, L)
        for r in range(R - M + 1):
            for layer in range(L):
                want = len({int(e) for m in range(M) for e in ids[r + m, layer]})
                assert int(u[r, layer]) == want, (M, r, layer)
    assert C.union_counts(ids, R + 1, E).shape[0] == 0
    print(f"  union_counts == sets over every window (M 1..6, {R} rows x {L} layers)")


def test_byte_model():
    m = C.K2Bytes()
    assert m.value_block == 1392640 and m.moe_block == 2088960 + 1044480
    assert abs(m.plain("bf16") / 1e9 - 3.786) < 5e-4 and abs(m.plain("int8") / 1e9 - 3.145) < 5e-4, \
        (m.plain("bf16"), m.plain("int8"))
    assert m.weights(8, 4, "int8") == m.plain("int8")
    im, iv = m.iid(1)
    assert abs(im - 8) < 1e-9 and abs(iv - 4) < 1e-9
    assert m.iid(1000)[0] > 99.99 and m.iid(1000)[1] > 63.99
    # one more distinct expert of each kind costs exactly its block, on all 45 sparse layers
    assert m.weights(9, 4) - m.weights(8, 4) == 45 * m.moe_block
    assert m.weights(8, 5) - m.weights(8, 4) == 45 * m.value_block
    print(f"  K2 plain step {m.plain('bf16') / 1e9:.4f} / {m.plain('int8') / 1e9:.4f} GB (spec 18 §10: 3.786 / 3.145); "
          f"blocks {m.value_block} / {m.moe_block} B")


def _dump(n, n_prompt, seed=2):
    g = torch.Generator().manual_seed(seed)
    R = n - (n_prompt - 1)
    mk = lambda E, k: torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(R * 45)]).view(R, 45, k)  # noqa: E731
    return {"n": n, "n_prompt": n_prompt, "head_from": n_prompt - 1, "moe_ids": mk(100, 8).to(torch.uint8),
            "mova_ids": mk(64, 4).to(torch.uint8)}


def test_source_unions_and_ratios():
    d = _dump(40, 12)
    u = C.source_unions(d, [1, 2, 6], kmax=5)
    anchors = list(range(12, 40 - 5 + 1))
    for M in (1, 2, 6):
        ok = [p for p in anchors if p + M - 1 <= 39]
        sm = sum(sum(len({int(e) for m in range(M) for e in d["moe_ids"][p - 11 + m, layer]}) for layer in range(45)) / 45
                 for p in ok)
        assert u[M][2] == len(ok) and abs(u[M][0] - sm) < 1e-6, (M, u[M], sm, len(ok))
    assert u[1][0] / u[1][2] == 8.0 and u[1][1] / u[1][2] == 4.0
    m = C.K2Bytes()
    cfg = E3.read_config(os.path.join(_HERE, "third_party", "eagle3_k2"))
    cost = C.derive({M: list(v) for M, v in u.items()}, m, [0, 4096], cfg)
    assert cost["M"]["1"]["ratio_int8head"] == 1.0 and C.verify_ratio(cost, 1, 4096) == 1.0
    r2 = C.verify_ratio(cost, 2, 4096)
    assert 1.0 < r2 < cost["M"]["2"]["ratio_int8head"]          # the KV is read once: depth dilutes
    b = cost["drafter"]["int8h"]["4096"]
    dr = C.draft_ratio(cost, "int8h", 2, 4096)
    assert abs(dr - (b["fc"] + 2 * b["step"]) / (m.plain("int8") + 4096 * m.kv_pos_int8)) < 1e-12
    md = C.format_md(cost)
    assert "| 2 |" in md and "int4h" in md
    print(f"  source_unions == a direct count (M 1 / 2 / 6); V(2) at 4k {r2:.3f}; D(2) int8h {dr:.4f} plain steps")


TESTS = [test_union_counter, test_byte_model, test_source_unions_and_ratios]


def main() -> None:
    only = sys.argv[1:]
    for t in TESTS:
        if not only or t.__name__ in only:
            print(t.__name__)
            t()
    print("ok")


if __name__ == "__main__":
    main()
