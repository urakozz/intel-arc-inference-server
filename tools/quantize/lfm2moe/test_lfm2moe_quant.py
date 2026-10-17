#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for the LFM2.5-24B-A2B quantisation helpers (tools/quantize/lfm2moe/), spec 24b Task 1. CPU,
seconds, no real model: make_synth.py writes a tiny-N (3 layers: the smallest that keeps an attention layer
and a MoE layer) synthetic checkpoint at the REAL widths and check.py / the reference read it back.

    test_lfm2moe_quant.py [test_name ...]

Needs torch, safetensors, transformers 5.19.0 (tools/oracle/qwen4exp_env.sh's site) on PYTHONPATH; run inside
agnes-ref-img (tools/box_validate/stages.sh r50.synth / r50.host).
"""
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import common as C  # noqa: E402

ref = C.lfm2moe_ref()

# The smallest --layers that holds a dense conv layer, the first attention + MoE layer and a conv + MoE
# layer (the real pattern's layer 2 is the first full_attention) - make_synth's own tests use 3, not the
# default 4, so a run takes a few seconds, not Task 1's ~1.28 GB / minutes.
N_TEST = 3


def _synth(tmp: str, expert_names: str = "w", plant_sign: bool = False) -> str:
    import make_synth as S
    out = os.path.join(tmp, f"synth_{expert_names}")
    sys.argv = ["make_synth.py", out, "--layers", str(N_TEST), "--expert-names", expert_names, "--seed", "1"]
    if plant_sign:
        sys.argv.append("--plant-sign")
    S.main()
    return out


def test_synth_names():
    import make_synth as S
    with tempfile.TemporaryDirectory() as tmp:
        for spelling in ("w", "proj"):
            out = _synth(tmp, spelling)
            with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
                import json
                cfg = json.load(f)
            c = ref.config(None, raw={k: v for k, v in cfg.items() if k != "quantization_config"})
            want = set(ref.expected_names(c, "int4", spelling))
            with open(os.path.join(out, "model.safetensors.index.json"), encoding="utf-8") as f:
                import json as J
                wm = J.load(f)["weight_map"]
            assert set(wm) == want, (sorted(want - set(wm))[:5], sorted(set(wm) - want)[:5])
    print(f"  make_synth's names == lfm2moe_ref.expected_names(cfg, 'int4', spelling) for 'w' and 'proj'")


def test_synth_checked():
    import check
    with tempfile.TemporaryDirectory() as tmp:
        for spelling in ("w", "proj"):
            out = _synth(tmp, spelling)
            errs, info = check.check(out, None, None, 16)
            assert not errs, errs[:5]
            assert info["expert_names"] == spelling
    print("  check.py ACCEPTS make_synth's output for both expert spellings")


def test_synth_loads_in_reference():
    with tempfile.TemporaryDirectory() as tmp:
        out = _synth(tmp, "w")
        with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
            import json
            raw = json.load(f)
        cfg = ref.config(None, raw={k: v for k, v in raw.items() if k != "quantization_config"})
        model = ref.build_streamed(out, cfg, weights="int4")
        ids = list(range(16))
        out_t = ref.port_forward(model, ids, {})
        assert out_t[0].shape[-1] == torch.Size([cfg.vocab_size])[-1]
    print(f"  lfm2moe_ref.build_streamed(weights='int4') runs {16} ids through the N={N_TEST} synthetic")


def test_gate_up_sign():
    """Review Focus 1 at the source: --plant-sign writes w3 = -w1 for expert 0 of the first MoE layer (layer
    2, the real pattern's first full_attention + MoE layer) BEFORE packing. The two tensors' per-group
    scales (amax-derived) must then be bit-identical, and almost every dequantised element must be the exact
    negation - except the (at most one per 64-group) element that IS the group's amax: symmetric int4 at
    zero point 8 represents [-8, 7], not [-8, 8], so a quotient whose f16-rounded scale pushes it fractionally
    past 7.5 clamps on the POSITIVE side only (w1's copy) while w3's negated copy lands exactly at -8 with no
    clamp - a real, documented asymmetry of src/loader/rtn.h's quantiser, not a bug in make_synth or the
    engine's repack (which never re-quantises: it moves already-packed int4 words, so the C++ repack test,
    lfm2moe_repack_test.cc, sees no such exception)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = _synth(tmp, "w", plant_sign=True)
        MT = C.lfm2moe_make_tiny()
        from safetensors import safe_open
        with open(os.path.join(out, "model.safetensors.index.json"), encoding="utf-8") as f:
            import json
            wm = json.load(f)["weight_map"]
        base = "model.layers.2.feed_forward.experts.0."
        with safe_open(os.path.join(out, wm[base + "w1.qweight"]), framework="pt") as h:
            qw1, sc1 = h.get_tensor(base + "w1.qweight"), h.get_tensor(base + "w1.scales")
        with safe_open(os.path.join(out, wm[base + "w3.qweight"]), framework="pt") as h:
            qw3, sc3 = h.get_tensor(base + "w3.qweight"), h.get_tensor(base + "w3.scales")
        assert torch.equal(sc1, sc3), "w1 and w3's per-group scales differ: --plant-sign did not negate w1 exactly"
        w1, w3 = MT.dequant(qw1, sc1), MT.dequant(qw3, sc3)
        mismatches = int((w3 != -w1).sum())
        rate = mismatches / w1.numel()
        assert rate < 0.02, f"{mismatches} of {w1.numel()} elements ({rate:.4%}) are not exact negatives"
    print(f"  --plant-sign: scales identical; {mismatches} / {w1.numel()} elements ({rate:.3%}) differ from "
          f"the exact negation, all at a group's positive-clamp boundary (rtn.h's [-8, 7] asymmetry)")


TESTS = [test_synth_names, test_synth_checked, test_synth_loads_in_reference, test_gate_up_sign]


def main() -> None:
    only = sys.argv[1:]
    for t in TESTS:
        if only and t.__name__ not in only:
            continue
        print(t.__name__, flush=True)
        t()
    print("ok")


if __name__ == "__main__":
    main()
