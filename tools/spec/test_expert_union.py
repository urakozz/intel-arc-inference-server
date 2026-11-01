#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for expert_union.py (plan 26a Task 3). Dependency-free, seconds: python3 tools/spec/test_expert_union.py"""
import json
import os
import struct
import sys
import tempfile
from array import array

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import expert_union as eu  # noqa: E402

CONFIG = os.path.join(_HERE, "..", "..", "tests", "model", "bolt", "config.json")


def write_golden(path, routes: dict, meta: dict):
    """A safetensors file holding route.ids.L<l> int32 [rows][9] (what bolt_ref.py run writes)."""
    hdr, blobs, off = {}, [], 0
    for l, rows in sorted(routes.items()):
        a = array("i", [x for r in rows for x in r])
        if sys.byteorder != "little":
            a.byteswap()
        b = a.tobytes()
        hdr[f"route.ids.L{l}"] = {"dtype": "I32", "shape": [len(rows), 9], "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hdr["__metadata__"] = {k: str(v) for k, v in meta.items()}
    h = json.dumps(hdr).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_block_bytes_and_config():
    assert eu.block_bytes(2048) == 2 * 2048 * 544 and eu.block_bytes(1536) == 2 * 1536 * 544
    blocks = eu.moe_blocks(CONFIG)
    Es = sorted({E for E, _ in blocks.values()})
    Is = sorted({I for _, I in blocks.values()})
    assert len(blocks) == 40 and Es[0] == 240 and Es[-1] == 496 and Is == [1536, 2048, 2304, 2560], (Es, Is)
    plain = sum(9 * eu.block_bytes(I) for _, I in blocks.values())
    mean_block = sum(eu.block_bytes(I) for _, I in blocks.values()) / 40
    print(f"config: 40 MoE blocks, E {Es}, I {Is}; a block {mean_block / 1e6:.3f} MB on average; a plain row reads "
          f"{plain / 1e9:.3f} GB of routed experts (DERIVED)")


def test_union_counts():
    rows = [tuple(range(0, 9)), tuple(range(0, 9)), tuple(range(5, 14)), tuple(range(100, 109))]
    uc = eu.union_counts({1: rows}, 0, 4)
    assert uc[1][1] == [9, 9, 9, 9]
    assert uc[2][1] == [9, 14, 18]          # rows 0+1 the same set; 1+2 overlap 4; 2+3 disjoint
    assert uc[3][1] == [14, 23] and uc[4][1] == [23]
    assert eu.union_counts({1: rows}, 2, 4)[2][1] == [18]
    print("union: distinct ids of M consecutive rows from every start (and from n_prompt with --rows gen)")


def test_analyse_on_files():
    blocks = {1: (240, 2048), 3: (496, 1536)}
    same = [tuple(range(9))] * 10
    disj = [tuple(range(9 * r, 9 * r + 9)) for r in range(10)]
    with tempfile.TemporaryDirectory() as d:
        write_golden(os.path.join(d, "a.golden.safetensors"), {1: same, 3: disj}, {"n_prompt": 6})
        res = eu.analyse([os.path.join(d, "a.golden.safetensors")], blocks, 8, "all")
        m2 = res["M"][2]
        assert m2["by_E"][240]["mean"] == 9 and m2["by_E"][496]["mean"] == 18 and m2["u_max"] == 18
        b240, b496 = eu.block_bytes(2048), eu.block_bytes(1536)
        assert m2["union_bytes_mean"] == 9 * b240 + 18 * b496
        assert m2["plain_bytes"] == 2 * 9 * (b240 + b496)
        assert res["M"][8]["n"] == 2 * 3 and res["M"][1]["n"] == 20
        g = eu.analyse([os.path.join(d, "a.golden.safetensors")], blocks, 4, "gen")
        assert g["M"][1]["n"] == 2 * 4 and g["M"][4]["n"] == 2 * 1
        text = eu.report(res)
        assert "| 2 |" in text and "union / plain" in text
        write_golden(os.path.join(d, "b.golden.safetensors"), {1: same, 5: same}, {})
        try:
            eu.analyse([os.path.join(d, "b.golden.safetensors")], blocks)
            raise AssertionError("a route record of a non-MoE block accepted")
        except ValueError:
            pass
    print("analyse: per M and E bucket, the union bytes sum_l u_l x block(I_l) against the plain 9 M blocks")


def main():
    for t in (test_block_bytes_and_config, test_union_counts, test_analyse_on_files):
        t()
    print("ok")


if __name__ == "__main__":
    main()
