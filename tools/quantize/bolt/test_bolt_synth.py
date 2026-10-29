#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 25b Task 1's tests (run in the oracle image, tools/oracle/README.md; the Mac under /tmp/b70-cpu-model.lock).

    test_bolt_synth.py                      the unit tests (a few seconds)
    test_bolt_synth.py --e2e <g128> <g64x>  + the two synthetic checkpoints (make_synth.py, same --layers, the g64 one
                                            made with --expand-from <g128>): `check.py` ACCEPTED on both, and
                                            bolt_ref.build_streamed(weights="int4") reads both and its logits on 16 ids
                                            are BITWISE equal (the g128 -> g64 expansion is exact: spec 25 §5.1)
"""
import os
import subprocess
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import make_synth as ms  # noqa: E402


def test_config_is_the_exports():
    mc = ms.make_config()
    assert mc.check() == 0
    c = mc.config(12, "gptq-g128")
    assert len(c["llm_config"]["block_configs"]) == 12 == len(c["llm_config"]["layers_block_type"])
    dyn = c["quantization_config"]["dynamic"]
    assert sum(k.startswith("-:") for k in dyn) == 44 and sum(k.startswith("+:") for k in dyn) == 1
    assert "num_hidden_layers" not in c["llm_config"]


def test_signed_scales_and_expansion_are_exact():
    """A random matrix packed signed at g128 and the same nibbles with each g128 scale row written to both g64 rows
    dequantise to the same weights (dequant.py's rule), and about half the scales are negative."""
    mt = ms.make_tiny()
    g = torch.Generator().manual_seed(3)
    w = (torch.randn(96, 512, generator=g) / 22.0).to(torch.bfloat16)
    qw, qz, sc = mt.pack_int4(w, 128, True)
    assert qw.shape == (64, 96) and sc.shape == (4, 96) and qz.shape == (4, 12)
    neg = float((sc.float() < 0).float().mean())
    assert 0.35 < neg < 0.65, neg
    ex = ms.expand_shard({"x.qweight": qw, "x.scales": sc, "x.qzeros": qz, "x.g_idx": torch.arange(512, dtype=torch.int32) // 128})
    assert "x.g_idx" not in ex and ex["x.scales"].shape == (8, 96) and ex["x.qzeros"].shape == (8, 12)
    assert torch.equal(ex["x.scales"][0::2], sc) and torch.equal(ex["x.scales"][1::2], sc)
    assert bool((ex["x.qzeros"] == 0x77777777).all())
    d128 = mt.dequant(qw, sc, 128)
    d64 = mt.dequant(ex["x.qweight"], ex["x.scales"], 64)
    assert torch.equal(d128, d64)
    # the sign bit survives the expansion bit for bit
    assert torch.equal(ex["x.scales"].view(torch.int16)[0::2], sc.view(torch.int16))


def test_native_g64_differs_from_the_expansion():
    """Packing the same weights at g64 is NOT the expansion (it is a finer quantisation): the two forms are different
    checkpoints, which is why the loader test of Review Focus 1 builds the g64 form FROM the g128 one."""
    mt = ms.make_tiny()
    g = torch.Generator().manual_seed(4)
    w = (torch.randn(32, 256, generator=g) / 16.0).to(torch.bfloat16)
    q128 = mt.pack_int4(w, 128, True)
    q64 = mt.pack_int4(w, 64, True)
    ex = ms.expand_shard({"x.qweight": q128[0], "x.scales": q128[2], "x.qzeros": q128[1]})
    assert not torch.equal(ex["x.scales"], q64[2])


def test_widths_are_the_real_ones():
    llm = ms.make_config().llm_config(88)
    w = ms.widths(llm)
    assert (w["in_proj"], w["d_inner"], w["conv_dim"]) == (18560, 8192, 10240)
    ei = {i: (b["n_routed_experts"], b["moe_intermediate_size"]) for i, b in enumerate(llm["block_configs"]) if b["block_type"] == "moe"}
    assert len(ei) == 40 and sum(e for e, _ in ei.values()) == 13824


def test_check_counts_zero_scales_and_refuses_only_nonfinite():
    import check
    def i16(v):
        return torch.tensor([x - 65536 if x >= 32768 else x for x in v], dtype=torch.int16)
    bits = i16([0x0000, 0x8000, 0x3C00, 0xBC00, 0x0001, 0x8001, 0x7C00, 0x7E00])
    st = check.scale_stats(bits.view(torch.float16))
    assert st == {"count": 8, "negative": 2, "subnormal": 2, "zero": 2, "nonfinite": 2}, st
    clean = i16([0x0000, 0x0000, 0x3C00]).view(torch.float16)   # an all-zero group is legitimate
    st = check.scale_stats(clean)
    assert st["zero"] == 2 and st["nonfinite"] == 0


def e2e(g128: str, g64x: str) -> None:
    for d in (g128, g64x):
        r = subprocess.run([sys.executable, os.path.join(_HERE, "check.py"), d], capture_output=True, text=True)
        print(r.stdout.strip().splitlines()[-1])
        assert r.returncode == 0, r.stdout + r.stderr
    br = ms.bolt_ref()
    out = []
    for d in (g128, g64x):
        cfg = br.config(d)
        model = br.build_streamed(d, cfg, weights="int4", dtype=torch.bfloat16, path="kernel")
        g = torch.Generator().manual_seed(5)
        ids = torch.randint(0, cfg["vocab_size"], (1, 16), generator=g)
        with torch.no_grad():
            lg = model(input_ids=ids, use_cache=False).logits[0].float()
        out.append(lg)
        print(f"{d}: logits {tuple(lg.shape)} finite {bool(torch.isfinite(lg).all())}")
    assert torch.equal(out[0], out[1]), float((out[0] - out[1]).abs().max())
    print(f"e2e OK: the g128 and its exact g64 expansion give bitwise-equal logits on 16 ids "
          f"(max |logit| {float(out[0].abs().max()):.3f})")


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"{n} ok")
    if "--e2e" in sys.argv:
        i = sys.argv.index("--e2e")
        e2e(sys.argv[i + 1], sys.argv[i + 2])
    print(f"test_bolt_synth: {len(tests)} tests OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
