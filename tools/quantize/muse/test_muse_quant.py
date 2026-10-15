#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for spec 23b's Muse-Glimmer tools (tools/quantize/muse/). CPU, no model download.

    test_muse_quant.py [test_name ...]

Run in agnes-ref-img with 21a's transformers 5.19.0 site first on PYTHONPATH (muse_ref.text_config needs it); the
original's small files (tokenizer, template, generation config) and the -GPTQ repo's config.json from the HF cache
(MUSE_ORIG / MUSE_GPTQ override; /hf/hub inside the container). The checkpoints here are the REAL widths at ONE
layer with a 1024-row vocabulary (make()'s test-only knob) so a test writes ~0.27 GB, not 5.6:

  test_synth_names      make(--layers 1): the names equal muse_ref.expected_names(text_config(out), "gptq") both ways,
                        the config is the -GPTQ repo's truncated, the original's small files copied byte for byte,
                        and check.py ACCEPTS it with every linear within RTN's bound of its twin
  test_check_refuses    check.py refuses a corrupted qzeros word (naming the tensor and the index), a non-identity
                        g_idx, a missing tensor, an extra text tensor, a qweight word changed (its twin), desc_act true
  test_pack_rule        the packer (23a's pack_int4) at a real shape against (q - 8) x scale: within s / 2 of the
                        source, nibble order k = 8r + j low first, qzeros 0x77777777, g_idx i // 64
"""
import filecmp
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import check as CK  # noqa: E402
import make_synth as MS  # noqa: E402

ORIG = os.environ.get("MUSE_ORIG") or MS.default_snapshot("meta-models/Muse-Glimmer-30B")
GPTQ = os.environ.get("MUSE_GPTQ") or MS.default_snapshot("Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ")
SMALL = dict(vocab=1024, verbose=False)
_made = {}


class Skip(Exception):
    pass


def synth(tag: str = "base") -> str:
    """One 1-layer checkpoint per tag, made once (copied for the refusals)."""
    if not ORIG or not GPTQ:
        raise Skip(f"the original's small files ({ORIG}) or the -GPTQ config ({GPTQ}) are not here")
    if "base" not in _made:
        d = os.path.join(tempfile.mkdtemp(prefix="muse_quant_"), "base")
        MS.make(d, 1, ORIG, GPTQ, seed=3, **SMALL)
        _made["base"] = d
    if tag == "base":
        return _made["base"]
    d = os.path.join(os.path.dirname(_made["base"]), tag)
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(_made["base"], d)
    return d


def rewrite(d: str, edit) -> None:
    """Rewrite the one shard holding the tensors `edit` changes: edit(dict of every tensor of that shard) in place."""
    from safetensors.torch import load_file, save_file
    for fn in sorted(f for f in os.listdir(d) if f.endswith(".safetensors")):
        sd = load_file(os.path.join(d, fn))
        if edit(sd):
            save_file(sd, os.path.join(d, fn), metadata={"format": "pt"})
            idx = os.path.join(d, "model.safetensors.index.json")
            j = json.load(open(idx))
            j["weight_map"] = {k: v for k, v in j["weight_map"].items() if v != fn}
            j["weight_map"].update({k: fn for k in sd})
            json.dump(j, open(idx, "w"))
            return
    raise AssertionError("no shard edited")


def refused(d: str, *needles: str) -> None:
    errs, _ = CK.check(d)
    assert errs, f"{d}: ACCEPTED, expected a refusal naming {needles}"
    hit = [e for e in errs if all(n in e for n in needles)]
    assert hit, f"no refusal names {needles}: {errs[:3]}"
    print(f"    refused: {hit[0][:160]}")


L0 = MS.LM + "layers.0."


def test_synth_names():
    d = synth()
    ref = CK.ref()
    tc = ref.text_config(d)
    want = set(ref.expected_names(tc, "gptq"))
    got = set(CK.headers(d))
    assert got == want, f"missing {sorted(want - got)[:5]}, extra {sorted(got - want)[:5]}"
    assert len(want) == 3 + 36, len(want)
    cfg = json.load(open(os.path.join(d, "config.json")))
    t = cfg["text_config"]
    assert t["num_hidden_layers"] == 1 and t["layer_types"] == ["sliding_attention"] and t["layer_rope_theta"] == [500000.0]
    assert cfg["quantization_config"]["quant_method"] == "gptq" and cfg["quantization_config"]["group_size"] == 64
    for fn in MS.SMALL_FILES:
        assert filecmp.cmp(os.path.join(d, fn), os.path.join(ORIG, fn), shallow=False), fn
    errs, info = CK.check(d)
    assert not errs, errs[:3]
    assert info["twins"] == 8 and info["form"] == "gptq" and info["worst_err_in_scales"] <= 0.5 + 1e-3, info
    print(f"    ACCEPTED: {info}")


def test_check_refuses():
    d = synth("qzeros")

    def qz(sd):
        k = L0 + "self_attn.k_proj.qzeros"
        if k not in sd:
            return False
        sd[k][2, 5] = 0x77777776
        return True
    rewrite(d, qz)
    refused(d, L0 + "self_attn.k_proj.qzeros[2][5]")

    d = synth("g_idx")

    def gi(sd):
        k = L0 + "mlp.down_proj.g_idx"
        if k not in sd:
            return False
        sd[k][70] = 0
        return True
    rewrite(d, gi)
    refused(d, L0 + "mlp.down_proj.g_idx[70]")

    d = synth("missing")
    rewrite(d, lambda sd: sd.pop(L0 + "mlp.up_proj.scales", None) is not None)
    refused(d, "missing tensor", L0 + "mlp.up_proj.scales")

    d = synth("extra")

    def extra(sd):
        if L0 + "input_layernorm.weight" not in sd:
            return False
        sd[L0 + "self_attn.q_norm.weight"] = torch.zeros(128, dtype=torch.bfloat16)
        return True
    rewrite(d, extra)
    refused(d, "unexpected tensor", L0 + "self_attn.q_norm.weight")

    d = synth("word")

    def word(sd):
        k = L0 + "self_attn.o_proj.qweight"
        if k not in sd:
            return False
        sd[k][10, 3] = sd[k][10, 3] ^ 0x8   # nibble 0 (k = 80) moved by 8 levels: far beyond RTN's half step
        return True
    rewrite(d, word)
    refused(d, L0 + "self_attn.o_proj", "twin")

    d = synth("desc_act")
    p = os.path.join(d, "config.json")
    cfg = json.load(open(p))
    cfg["quantization_config"]["desc_act"] = True
    json.dump(cfg, open(p, "w"))
    refused(d, "desc_act")


def test_pack_rule():
    pack = MS.tiny().pack_int4
    w = MS.source(L0 + "self_attn.k_proj.weight", (256, 6656), seed=11)       # [N][K]
    qw, qz, sc, gi = pack(w, 64)
    assert qw.dtype == torch.int32 and tuple(qw.shape) == (832, 256)
    assert sc.dtype == torch.float16 and tuple(sc.shape) == (104, 256)
    assert bool(((qz.to(torch.int64) & 0xFFFFFFFF) == 0x77777777).all()) and tuple(qz.shape) == (104, 32)
    assert torch.equal(gi.to(torch.int64), torch.arange(6656) // 64)
    deq = CK._deq(qw, sc)                                                      # [K][N]
    s = sc.float().repeat_interleave(64, dim=0)
    err = (w.float().t() - deq).abs()
    assert bool((err <= s / 2 + w.float().t().abs() * 2 ** -8).all()), float((err / s).max())
    # nibble order: element k = 8r + j of column n is nibble j of word [r][n], low first
    q = ((qw[3, 7].item() & 0xFFFFFFFF) >> (4 * 5)) & 0xF
    assert abs((q - 8) * float(sc[(3 * 8 + 5) // 64, 7]) - float(deq[3 * 8 + 5, 7])) == 0.0
    print(f"    the packer: worst |w - deq| = {float((err / s).max()):.3f} scales")


TESTS = [test_synth_names, test_check_refuses, test_pack_rule]


def main() -> None:
    names = sys.argv[1:]
    run = [t for t in TESTS if not names or t.__name__ in names]
    failed = skipped = 0
    for t in run:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Skip as e:
            skipped += 1
            print(f"SKIP {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    for d in _made.values():
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)
    print(f"{len(run) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
