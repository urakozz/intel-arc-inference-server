#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for spec 21b's Qwen3.8-Flash-Next tools (tools/quantize/qwen4exp/). CPU, no model download.

    test_qwen4exp_quant.py [test_name ...]

Run in agnes-ref-img with 21a's transformers 5.19.0 site first on PYTHONPATH (qwen4exp_ref.text_config
needs it); the original's small files (config.json, tokenizer files) at Q4EXP_ORIG (default
oracle-out-q4exp/orig-small). The synthetic checkpoints here are the REAL widths with 16 experts and a
1024-row vocabulary (make_synth's test-only knobs) so a test writes ~0.4 GB, not 6:

  test_synth_names      make_synth --layers 4 in both forms (and --mtp): the names equal
                        qwen4exp_ref.expected_names both ways, and check.py ACCEPTS each
  test_check_refuses    check.py refuses a missing tensor, a wrong qzeros word, a g64 expert in Intel's form
  test_pack_g64_g128    the packer against dequant.py's rule at both group sizes; qzeros 0x77777777; the
                        nibble order (k = 8r + j in nibble j of word r, low first)
  test_gate_up_order    Review Focus 2 at the source: the reference reads the synthetic's expert as
                        cat(gate_proj, up_proj) - gate rows first - and gate and up differ
  test_ple_int8_rows    Review Focus 3: marker rows at every head's first / last row and at a shard boundary
                        inside a head land at head-local rows; spec 9's scales; a zero row; --scale bf16
                        rounds the scale once; the file equals qwen4exp_make_tiny.write_ple_int8's; the
                        reference's int8 reader dequantises it; check.py --ple ACCEPTS it
  test_hash_tensors     the I64 tensors equal the formula (qwen4exp_facts) for the reduced base and for
                        20,000,000 (21a's facts-sheet values), and transformers 5.19.0's own builders
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import check as CK  # noqa: E402
import make_synth as MS  # noqa: E402
import ple_int8 as PI  # noqa: E402

ORIG = os.environ.get("Q4EXP_ORIG", os.path.join(REPO, "oracle-out-q4exp", "orig-small"))
SMALL = dict(experts=16, vocab=1024, verbose=False)   # 16 >= top-k 10 (the config validator)
_dequant = MS._load(os.path.join(MS.ORACLE, "dequant.py"), "q4test_dequant")
_tiny = None


def tiny():
    global _tiny
    if _tiny is None:
        _tiny = MS._load(os.path.join(MS.ORACLE, "qwen4exp_make_tiny.py"), "q4test_make_tiny")
    return _tiny


class Skip(Exception):
    pass


def need_orig():
    if not os.path.exists(os.path.join(ORIG, "config.json")):
        raise Skip(f"no original config.json at {ORIG} (Q4EXP_ORIG)")


_synth: dict = {}


def synth(form: str, mtp: bool = False) -> str:
    key = (form, mtp)
    if key not in _synth:
        need_orig()
        d = os.path.join(tempfile.mkdtemp(prefix=f"q4synth-{form}-"), "ckpt")
        MS.make(d, form, ORIG, layers=4, ple_base=1000, mtp=mtp, **SMALL)
        _synth[key] = d
    return _synth[key]


def test_synth_names():
    for form, mtp in (("ours", False), ("intel", False), ("intel", True)):
        d = synth(form, mtp)
        MS.check_names(d, mtp)
        errs = CK.check(d, sample=8)
        assert not errs, errs[:5]
        with open(os.path.join(d, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["text_config"]["num_hidden_layers"] == 4 and cfg["text_config"]["ngram_vocab_size_base"] == 1000
        assert cfg["quantization_config"]["group_size"] == (64 if form == "ours" else 128)
        print(f"  {form}{' + mtp' if mtp else ''}: names = expected_names both ways, check.py ACCEPTED")


def test_check_refuses():
    from safetensors.torch import load_file, save_file
    src = synth("intel")
    for what in ("missing", "qzeros", "g64"):
        d = tempfile.mkdtemp(prefix="q4bad-") + "/ckpt"
        shutil.copytree(src, d)
        ix = json.load(open(os.path.join(d, "model.safetensors.index.json")))
        name = "model.language_model.layers.2.mlp.experts.3.down_proj"
        fn = ix["weight_map"][name + ".qweight"]
        sd = load_file(os.path.join(d, fn))
        if what == "missing":
            del sd[name + ".qzeros"]
            del ix["weight_map"][name + ".qzeros"]
        elif what == "qzeros":
            sd[name + ".qzeros"][0, 0] = 0x77777776
        else:
            w = MS.dequant(sd[name + ".qweight"], sd[name + ".scales"], 128)
            qw, sc, qz = MS.pack(w, 64)
            sd[name + ".qweight"], sd[name + ".scales"], sd[name + ".qzeros"] = qw, sc, qz
        save_file(sd, os.path.join(d, fn), metadata={"format": "pt"})
        json.dump(ix, open(os.path.join(d, "model.safetensors.index.json"), "w"))
        errs = CK.check(d, sample=2)
        assert errs and any(name in e for e in errs), (what, errs[:3])
        print(f"  {what}: refused ({errs[0][:90]})")


def test_pack_g64_g128():
    g = torch.Generator().manual_seed(3)
    for grp in (64, 128):
        w = (torch.randn(48, 256, generator=g) / 16).to(torch.bfloat16)
        w[5, :grp] = 0                                                    # an all-zero group: scale 0, q 8
        qw, sc, qz = MS.pack(w, grp)
        assert qw.dtype == torch.int32 and qw.shape == (32, 48)
        assert sc.dtype == torch.float16 and sc.shape == (256 // grp, 48)
        assert qz.shape == (256 // grp, 6) and bool((qz.view(torch.int32) == 0x77777777).all())
        ref = _dequant.dequant_gptq(qw, sc, grp)                          # [K][N] bf16, (q - 8) x scale
        mine = MS.dequant(qw, sc, grp)                                    # [N][K] fp32
        assert torch.equal(ref.t().float(), mine.to(torch.bfloat16).float())
        assert float(sc[0, 5]) == 0.0 and bool((mine[5, :grp] == 0).all())
        err = (mine - w.float()).abs().max() / w.float().abs().max()
        assert err <= 0.1251, err      # scale = amax / 8 and q - 8 <= 7: the largest positive weight loses 1/8
        # Nibble order: one non-zero weight at k = 8r + j of column n -> nibble j of word (r, n), low first.
        one = torch.zeros(16, 128, dtype=torch.bfloat16)
        one[3, 8 * 5 + 6] = 1.0
        qw1, sc1, _ = MS.pack(one, grp)
        word = int(qw1[5, 3]) & 0xFFFFFFFF
        assert float(sc1[0, 3]) == 0.125 and (word >> (4 * 6)) & 0xF == 15    # round(1 / 0.125) + 8 = 16, clamped
        others = [(word >> (4 * j)) & 0xF for j in range(8) if j != 6]
        assert others == [8] * 7, hex(word)
        print(f"  g{grp}: dequant.py's (q - 8) x scale bitwise, qzeros 0x77777777, an all-zero group, nibble order")


def test_gate_up_order():
    r = MS.ref()
    d = synth("intel")
    cfg = json.load(open(os.path.join(d, "config.json")))
    tc = r.text_config(d, raw=cfg)
    ck = r.Ckpt(d)
    lx = r.LazyExperts(ck, tc, "model.language_model.layers.", dtype=torch.bfloat16)
    I = tc.moe_intermediate_size
    for layer, e in ((0, 0), (3, 3)):
        gu, dn = lx.expert(layer, e)
        b = f"model.language_model.layers.{layer}.mlp.experts.{e}."
        gate = _dequant.dequant_gptq(ck.get(b + "gate_proj.qweight"), ck.get(b + "gate_proj.scales"), 128).t()
        up = _dequant.dequant_gptq(ck.get(b + "up_proj.qweight"), ck.get(b + "up_proj.scales"), 128).t()
        assert torch.equal(gu[:I], gate) and torch.equal(gu[I:], up) and not torch.equal(gate, up)
        assert dn.shape == (tc.hidden_size, I)
    print("  the reference reads cat(gate_proj, up_proj): gate rows [0, 640), up [640, 1280); they differ")


def fake_ple_snapshot(base: int = 1000, dtype=torch.bfloat16) -> tuple[str, dict, torch.Tensor]:
    """A snapshot holding only the PLE layer's table (128 shards) and I64 tensors, with marker rows."""
    need_orig()
    cfg = json.load(open(os.path.join(ORIG, "config.json")))
    t = cfg["text_config"]
    t["ngram_vocab_size_base"] = base
    sizes, offsets, total, padded, mult = MS.ple_tables(t)
    parts = t["split_ngram_parts"]
    rows_per = padded // parts
    g = torch.Generator().manual_seed(7)
    table = torch.randn(padded, 160, generator=g)
    marks = []
    for h in range(len(sizes)):
        marks += [offsets[h], offsets[h] + sizes[h] - 1]
    for k in range(1, parts):                                           # a shard boundary inside a head
        bnd = k * rows_per
        h = max(i for i in range(len(sizes)) if offsets[i] <= bnd)
        if offsets[h] < bnd - 1 and bnd < offsets[h] + sizes[h] - 1:
            marks += [bnd - 1, bnd]
            break
    for i, gr in enumerate(marks):
        table[gr] = float(i + 1) * torch.linspace(-1, 1, 160)            # distinctive, exactly known rows
    zero_row = offsets[3] + 5
    table[zero_row] = 0
    d = tempfile.mkdtemp(prefix="q4ple-src-")
    from safetensors.torch import save_file
    pre = "model.language_model.layers.1.ple.ple_embedding."
    sd = {f"{pre}ngram_embedding.shard_{k}.weight": table[k * rows_per:(k + 1) * rows_per].to(dtype).contiguous()
          for k in range(parts)}
    sd[pre + "layer_multipliers"] = torch.tensor(mult, dtype=torch.int64)
    sd[pre + "ngram_heads_vocab_sizes"] = torch.tensor(sizes, dtype=torch.int64)
    sd[pre + "ngram_heads_offsets"] = torch.tensor(offsets, dtype=torch.int64)
    save_file(sd, os.path.join(d, "model-00001-of-00001.safetensors"))
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in sd}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    return d, {"t": t, "sizes": sizes, "offsets": offsets, "marks": marks, "zero": zero_row, "rows_per": rows_per,
               "mult": mult}, table.to(dtype)


def test_ple_int8_rows():
    src, info, table = fake_ple_snapshot()
    sizes, offsets = info["sizes"], info["offsets"]
    w = table.float()
    for scale in ("f32", "bf16"):
        out = tempfile.mkdtemp(prefix=f"q4ple-{scale}-")
        r = PI.convert(src, out, scale, verbose=False)
        assert r["heads"] == 16 and r["rows"] == sum(sizes)
        pm = PI.tensor_map(out)
        for gr in info["marks"] + [info["zero"]]:
            h = max(i for i in range(16) if offsets[i] <= gr)
            loc = gr - offsets[h]
            assert 0 <= loc < sizes[h]
            q = torch.from_numpy(np.asarray(PI.mmap_of(pm[f"ple.h{h}.q"])[loc]).copy()).float()
            sv = np.asarray(PI.mmap_of(pm[f"ple.h{h}.s"])[loc:loc + 1]).copy()
            s = torch.from_numpy(sv).view(torch.bfloat16).float()[0] if scale == "bf16" else float(sv[0])
            s_want = w[gr].abs().max() / 127.0
            if scale == "bf16":
                s_want = s_want.to(torch.bfloat16).float()
            assert float(s) == float(s_want), (gr, float(s), float(s_want))
            if gr == info["zero"]:
                assert float(s) == 0.0 and bool((q == 0).all())
            else:
                assert torch.equal(q, torch.round(w[gr] / s_want).clamp(-127, 127)), gr
        # The same bytes as 21a's writer (qwen4exp_make_tiny.write_ple_int8, the reference's int8 input).
        cfg = json.load(open(os.path.join(src, "config.json")))
        tc = MS.ref().text_config(src, raw=cfg)
        other = tempfile.mkdtemp(prefix="q4ple-tiny-")
        tiny().write_ple_int8(table, tc, other, scale)
        om = PI.tensor_map(other)
        for h in range(16):
            for k in ("q", "s"):
                a = np.asarray(PI.mmap_of(pm[f"ple.h{h}.{k}"]))
                b = np.asarray(PI.mmap_of(om[f"ple.h{h}.{k}"]))
                assert a.shape == b.shape and bool((a.view(np.uint8) == b.view(np.uint8)).all()), (h, k)
        # The reference's int8 reader: a marker row dequantises to bf16(q x s).
        pt = MS.ref().PleTable("int8:" + out, tc)
        gr = info["marks"][2]
        h = max(i for i in range(16) if offsets[i] <= gr)
        ids = torch.full((1, 16), 0, dtype=torch.long)
        for i in range(16):
            ids[0, i] = offsets[i]
        ids[0, h] = gr
        row = pt.rows(ids)[0, h]
        loc = gr - offsets[h]
        q = torch.from_numpy(np.asarray(PI.mmap_of(pm[f"ple.h{h}.q"])[loc]).copy()).float()
        sv = np.asarray(PI.mmap_of(pm[f"ple.h{h}.s"])[loc:loc + 1]).copy()
        s = torch.from_numpy(sv).view(torch.bfloat16).float() if scale == "bf16" else torch.from_numpy(sv)
        assert row.dtype == torch.bfloat16 and torch.equal(row, (q * s).to(torch.bfloat16))
        errs = CK.check_ple(src, out, info["t"], 16, 0)
        assert not errs, errs
        print(f"  --scale {scale}: {len(info['marks'])} marker rows at head-local rows (a shard boundary inside head "
              f"{max(i for i in range(16) if offsets[i] <= info['marks'][-1])}), the zero row s = 0, = write_ple_int8's "
              f"bytes, check.py --ple ACCEPTED")
    # a source whose I64 tensors disagree with the formula is refused
    from safetensors.torch import load_file, save_file
    fn = os.path.join(src, "model-00001-of-00001.safetensors")
    sd = load_file(fn)
    sd["model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets"][1] += 1
    save_file(sd, fn)
    try:
        PI.convert(src, tempfile.mkdtemp(), "bf16", verbose=False)
    except SystemExit as e:
        assert "ngram_heads_offsets" in str(e)
        print("  an I64 tensor off the formula: refused by name")
    else:
        raise AssertionError("ple_int8 accepted wrong offsets")


def test_hash_tensors():
    need_orig()
    t = json.load(open(os.path.join(ORIG, "config.json")))["text_config"]
    sizes, offsets, total, padded, mult = MS.ple_tables(t)
    assert mult == [23703573157769, 20109073645365, 8052911324071]
    assert sizes[0] == 20000003 and sizes[15] == 20000171 and offsets[15] == 300001275
    assert total == 320001446 and padded == 320001536
    import transformers.models.qwen4_exp.modeling_qwen4_exp as M
    assert [int(x) for x in M._build_layer_multipliers(t["vocab_size"], 3, 0, 1234)] == mult
    for base in (1000, 20000000):
        tt = dict(t, ngram_vocab_size_base=base)
        s2, o2, _, _, m2 = MS.ple_tables(tt)
        want = []
        for h in range(16):
            want.append(M._find_nth_prime_after(base - 1, h + 1))
        assert s2 == want and o2 == [sum(want[:h]) for h in range(16)] and m2 == mult
    d = synth("ours")
    tm = PI.tensor_map(d)
    tt = json.load(open(os.path.join(d, "config.json")))["text_config"]
    PI.check_constants(tt, "model.language_model.layers.1.ple.ple_embedding.", tm)
    print(f"  base 20,000,000: multipliers {mult}, 16 primes 20000003..20000171, 320001446 rows (padded 320001536) "
          f"= 21a's facts and transformers' builders; base 1000 likewise; the synthetic's I64 tensors = the formula")


def main(argv) -> None:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    if argv:
        tests = [(n, f) for n, f in tests if n in argv]
    failed = skipped = 0
    for name, fn in tests:
        print(name)
        try:
            fn()
            print("ok")
        except Skip as e:
            skipped += 1
            print(f"SKIP {e}")
        except Exception as e:   # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {e}")
    print(f"{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
