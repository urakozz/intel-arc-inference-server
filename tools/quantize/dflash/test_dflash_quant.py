#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests of the DFlash drafter quantisation tooling on a tiny random drafter (no AutoRound, ~1 min):

- the calibration model (model.py) on calib.py's samples = dflash_ref's context plane and
  dflash_accept.forward_blocks' final-norm mask rows through the selector projection, float32;
- the sample plan: deterministic, anchors inside the continuation and the context, every request used;
- sources.py: the held-out recording refused, EOS / --max-cont cuts, --done skips complete dumps only;
- finalize.py + check_gptq_export.py --dflash on a hand-packed raw export: VERIFIED, then each refusal.
`--make-tiny DIR` only writes the tiny drafter / target / taps (quantize_dflash.sh selftest).

    docker run --rm -v "$PWD":/ws -w /ws agnes-ref-img:latest python3 tools/quantize/dflash/test_dflash_quant.py
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
import calib as C  # noqa: E402
import model as M  # noqa: E402

ref = M.ref
H, HT, VOCAB, MASK = 128, 64, 512, 509
CONFIG = {"architectures": ["DFlash2DraftModel"], "is_causal": False,
          "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": MASK,
                            "selector_rank": 64, "selector_top_k": 4, "target_layer_ids": [1, 3]},
          "head_dim": 64, "hidden_act": "silu", "hidden_size": H, "intermediate_size": 192,
          "layer_types": ["sliding_attention"] * 2, "model_type": "qwen3", "num_attention_heads": 2,
          "num_hidden_layers": 2, "num_key_value_heads": 1, "rms_norm_eps": 1e-6,
          "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "sliding_window": 24,
          "tie_word_embeddings": False, "vocab_size": VOCAB}
WINDOW = 24


def _load(path, name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def packer():
    """Kolibri's RTN g64 GPTQ packer (`(q - 8) x scale`, qzeros 0x77777777)."""
    return _load(os.path.join(TOOLS, "quantize", "kolibri", "make_synth.py"), "kolibri_make_synth").pack_rtn_g64


def make_tiny(root):
    """A tiny DFlash 2 drafter, a target holding the shared embedding, two tap dumps in the probe's format."""
    cfg = ref.config_from_dict(CONFIG)
    g = torch.Generator().manual_seed(0)

    def lin(o, i):
        return (torch.randn(o, i, generator=g) / i ** 0.5).to(torch.bfloat16)

    def norm(n):
        return (1 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)
    t = {"fc.weight": lin(H, HT * 2), "hidden_norm.weight": norm(H), "norm.weight": norm(H),
         "candidate_selector.hidden_projection.weight": lin(64, H),
         "candidate_selector.predecessor_codebook": torch.randn(VOCAB, 64, generator=g).to(torch.bfloat16),
         "candidate_selector.successor_codebook": torch.randn(VOCAB, 64, generator=g).to(torch.bfloat16)}
    for i in range(2):
        L = f"layers.{i}."
        t.update({L + "input_layernorm.weight": norm(H), L + "post_attention_layernorm.weight": norm(H),
                  L + "self_attn.q_proj.weight": lin(128, H), L + "self_attn.k_proj.weight": lin(64, H),
                  L + "self_attn.v_proj.weight": lin(64, H), L + "self_attn.o_proj.weight": lin(H, 128),
                  L + "self_attn.q_norm.weight": norm(64), L + "self_attn.k_norm.weight": norm(64),
                  L + "mlp.gate_proj.weight": lin(192, H), L + "mlp.up_proj.weight": lin(192, H),
                  L + "mlp.down_proj.weight": lin(H, 192)})
        for cv in ("attention_conv", "mlp_conv"):
            base = 0.3 * torch.randn(2, 2, H, generator=g)
            base[:, 0] += 1.0
            t[L + cv + ".base_kernel"] = base.to(torch.bfloat16)
            t[L + cv + ".kernel_projection.weight"] = (0.3 * lin(2 * 2 * H // 16, H).float()).to(torch.bfloat16)
    drafter, target, taps = (os.path.join(root, d) for d in ("drafter", "target", "taps"))
    for d in (drafter, target, taps):
        os.makedirs(d)
    save_file(t, os.path.join(drafter, "model.safetensors"))
    with open(os.path.join(drafter, "config.json"), "w") as f:
        json.dump(CONFIG, f)
    save_file({"model.embed_tokens.weight": torch.randn(VOCAB, H, generator=g).to(torch.bfloat16),
               "lm_head.weight": torch.randn(VOCAB, H, generator=g).to(torch.bfloat16)},
              os.path.join(target, "model.safetensors"))
    with open(os.path.join(target, "config.json"), "w") as f:
        json.dump({"vocab_size": VOCAB, "hidden_size": H}, f)
    for name, n_prompt, n_cont in (("rec-a/000001", 40, 20), ("rec-b/000007", 12, 9)):
        n = n_prompt + n_cont
        tap_from = max(0, n_prompt - WINDOW)
        ids = torch.randint(0, VOCAB - 8, (n,), generator=g, dtype=torch.int32)
        d = {"ids": ids, "greedy": torch.randint(0, VOCAB, (n - tap_from,), generator=g, dtype=torch.int32)}
        for i in (1, 3):
            d[f"tap.L{i}"] = torch.randn(n - tap_from, HT, generator=g).to(torch.bfloat16)
        save_file(d, os.path.join(taps, name.replace("/", "__") + ".taps.safetensors"),
                  metadata={"name": name, "n_prompt": str(n_prompt), "n": str(n), "tap_from": str(tap_from),
                            "layers": "1,3", "head_stats": "greedy_only"})
    return drafter, target, taps


def test_forward_matches_reference(drafter, target, taps):
    acc = _load(os.path.join(TOOLS, "oracle", "dflash_accept.py"), "dflash_accept")
    acc._load_deps()
    d = ref.load_drafter(drafter, target)
    m = M.load(drafter, dtype=torch.float32)
    context = C.context_rows(WINDOW, 20)
    data = C.CalibData(taps, drafter, target, nsamples=4, anchors=3, seed=42, context=context, dtype=torch.float32)
    assert data.rows == 24 and len(data) == 4
    checked = 0
    for e, s in zip(data.entries, data):
        with safe_open(e["path"], framework="pt") as f:          # the engine's dump: no head stats
            md = f.metadata()
            dump = {"tap_from": int(md["tap_from"]), "n": int(md["n"]), "ids": f.get_tensor("ids").long(),
                    "taps": {i: f.get_tensor(f"tap.L{i}") for i in (1, 3)}}
        pos = torch.arange(dump["tap_from"], dump["n"])
        ctx = d.context_kv({i: dump["taps"][i].float() for i in (1, 3)}, pos)
        live = s["kpos"][0] >= 0
        enc = m.encoder(s["taps"])[0][live]
        assert torch.allclose(enc, ctx.fc[s["kpos"][0][live] - dump["tap_from"]], atol=1e-5), "context plane"
        ps = torch.tensor(e["anchors"])
        want_h = acc.forward_blocks(d, [int(dump["ids"][p]) for p in ps], ps, 7, ctx)       # [A, 8, H]
        want = ref.linear(want_h[:, 1:].reshape(-1, H), d.w["candidate_selector.hidden_projection.weight"])
        with torch.no_grad():
            got = m(**s)[0]
        err = (got - want).abs().max().item()
        assert err < 1e-4, f"{e['name']} anchors {e['anchors']}: selector projection differs by {err}"
        checked += 1
    assert checked == 4
    print(f"forward: {checked} samples = dflash_ref (context plane, blocks of 8, selector projection)")


def test_plan(taps):
    metas = C.dumps(taps)
    p1 = C.plan(metas, nsamples=5, anchors=6, seed=42, span=10)
    assert p1 == C.plan(metas, nsamples=5, anchors=6, seed=42, span=10), "deterministic"
    assert p1 != C.plan(metas, nsamples=5, anchors=6, seed=7, span=10)
    by = {m["name"]: m for m in metas}
    assert {e["name"] for e in p1} == set(by), "every request used"
    for e in p1:
        m = by[e["name"]]
        a = e["anchors"]
        assert len(a) == 6 and a == sorted(a)
        assert all(max(m["n_prompt"], m["tap_from"] + 1) <= p < m["n"] for p in a), "inside the continuation"
        assert a[-1] - a[0] < 10, "span"
    print("plan: deterministic, inside the continuation and the span, every request")


def test_sources(root):
    src = _load(os.path.join(HERE, "sources.py"), "dflash_quant_sources")
    rec = os.path.join(root, "opencode-session-x")
    held = os.path.join(root, "opencode-session-2")
    for d in (rec, held):
        os.makedirs(d)
        with open(os.path.join(d, "000001.json"), "w") as f:
            json.dump({"prompt_ids": [1, 2, 3], "out_ids": list(range(10, 20)) + [7, 30, 31]}, f)
        with open(os.path.join(d, "000002.json"), "w") as f:
            json.dump({"prompt_ids": [1], "out_ids": [5]}, f)
    try:
        src.sources([held], {7}, 256)
        raise AssertionError("held-out recording accepted")
    except SystemExit as e:
        assert "held out" in str(e)
    s = src.sources([rec], {7}, 256)
    assert [x["name"] for x in s] == ["opencode-session-x/000001"]
    assert s[0]["cont_ids"] == list(range(10, 20)) + [7], "cut after the first EOS"
    assert src.sources([rec], {7}, 7) == [], "fewer than 8 ids after --max-cont"
    gc = os.path.join(root, "gen.json")
    with open(gc, "w") as f:
        json.dump({"eos_token_id": [7]}, f)
    done = os.path.join(root, "done")
    os.makedirs(done)
    out = os.path.join(root, "src.json")
    run = [sys.executable, os.path.join(HERE, "sources.py"), rec, "--generation-config", gc, "--out", out, "--done", done]
    subprocess.run(run, check=True, capture_output=True)
    assert len(json.load(open(out))) == 1
    dump = os.path.join(done, src.file_name("opencode-session-x/000001"))
    save_file({"x": torch.zeros(100)}, dump)
    subprocess.run(run, check=True, capture_output=True)
    assert json.load(open(out)) == [], "a complete dump is skipped"
    with open(dump, "r+b") as f:
        f.truncate(os.path.getsize(dump) - 4)
    subprocess.run(run, check=True, capture_output=True)
    assert len(json.load(open(out))) == 1, "a truncated dump is redone"
    print("sources: held-out refused, EOS / max-cont cuts, --done skips complete dumps only")


def raw_export(drafter, raw, commit="0123456789abcdef0123456789abcdef01234567", requests=("rec-a/000001",)):
    """What AutoRound writes for model.py's names: every 2-D linear packed (RTN g64), the rest as loaded."""
    m = M.load(drafter, dtype=torch.bfloat16)
    t = {}
    for n, v in m.state_dict().items():
        if n.endswith(".weight") and v.dim() == 2:
            qw, sc, qz = packer()(v.float())
            p = n[: -len(".weight")]
            t.update({p + ".qweight": qw, p + ".scales": sc, p + ".qzeros": qz})
        else:
            t[n] = v.float()                                  # a CPU run's fp32 widening
    os.makedirs(raw)
    save_file(t, os.path.join(raw, "model.safetensors"))
    qc = {"bits": 4, "group_size": 64, "sym": True, "quant_method": "auto-round", "packing_format": "auto_round:auto_gptq",
          "block_name_to_quantize": [["encoder"], ["layers.0", "layers.1"], ["candidate_selector"]],
          "autoround_version": "0.17.0"}
    with open(os.path.join(raw, "config.json"), "w") as f:
        json.dump({"model_type": "dflash2_calib", "quantization_config": qc}, f)
    recipe = {"autoround": {"version": "0.17.0", "commit": commit, "patched": "0", "repo": "fork"},
              "settings": {"scheme": "W4A16", "group_size": 64, "sym": True, "iters": 2, "nsamples": 4,
                           "batch_size": 2, "seed": 42, "format": "auto_round:auto_gptq", "quant_lm_head": False},
              "calibration": {"block_size": 8, "target_layer_ids": [1, 3], "anchors_per_sample": 3,
                              "context_rows": 64, "target": "t", "samples": 4, "requests_used": 1,
                              "recordings": ["rec-a"], "requests": list(requests), "held_out": ["opencode-session-2"]},
              "kept_source_precision": ["norms"], "made": "now", "seconds": 1, "torch": torch.__version__}
    with open(os.path.join(raw, "dflash_recipe.json"), "w") as f:
        json.dump(recipe, f)


def check(out, drafter):
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "quantize", "check_gptq_export.py"), out, "--dflash",
                        "--source", drafter], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_finalize_and_check(root, drafter):
    raw, out = os.path.join(root, "raw"), os.path.join(root, "export")
    raw_export(drafter, raw)
    subprocess.run([sys.executable, os.path.join(HERE, "finalize.py"), raw, drafter, out], check=True,
                   capture_output=True)
    names = set(M.ref._weight_map(out))
    assert "fc.qweight" in names and "hidden_norm.weight" in names and not any(n.startswith("encoder.") for n in names)
    qc = json.load(open(os.path.join(out, "config.json")))["quantization_config"]
    assert qc["block_name_to_quantize"] == [["fc"], ["layers.0", "layers.1"], ["candidate_selector"]]
    rc, txt = check(out, drafter)
    assert rc == 0 and "VERIFIED DFLASH" in txt and "20 packed linears" in txt, txt
    print("finalize + check: " + txt.strip().splitlines()[-1][:160])

    def refused(mutate, why):
        bad = out + "-bad"
        shutil.rmtree(bad, ignore_errors=True)
        shutil.copytree(out, bad)
        mutate(bad)
        rc, txt = check(bad, drafter)
        assert rc == 1 and why in txt, f"expected '{why}':\n{txt}"

    def edit_tensors(f):
        def go(d):
            from safetensors.torch import load_file
            t = load_file(os.path.join(d, "model.safetensors"))
            f(t)
            save_file(t, os.path.join(d, "model.safetensors"))
        return go

    def codebook(t):
        t["candidate_selector.successor_codebook"][0, 0] += 1

    def unpack(t):
        for s in (".qweight", ".scales", ".qzeros"):
            t.pop("layers.1.mlp_conv.kernel_projection" + s)
        t["layers.1.mlp_conv.kernel_projection.weight"] = torch.zeros(32, H, dtype=torch.bfloat16)

    def head(t):
        t["lm_head.weight"] = torch.zeros(4, H, dtype=torch.bfloat16)

    def recipe(change):
        def go(d):
            p = os.path.join(d, "dflash_recipe.json")
            r = json.load(open(p))
            change(r)
            json.dump(r, open(p, "w"))
        return go
    refused(edit_tensors(codebook), "successor_codebook BF16")
    refused(edit_tensors(unpack), "linear layers.1.mlp_conv.kernel_projection is not packed")
    refused(edit_tensors(head), "lm_head.weight shipped")
    refused(recipe(lambda r: r["autoround"].update(commit="?")), "is not a 40-hex commit")
    refused(recipe(lambda r: r["calibration"]["requests"].append("opencode-session-2/000001")), "held-out")
    refused(recipe(lambda r: r["settings"].update(quant_lm_head=True)), "quant_lm_head")
    refused(recipe(lambda r: r["calibration"].update(block_size=16)), "block size / tap layers")
    refused(lambda d: os.remove(os.path.join(d, "dflash_recipe.json")), "no dflash_recipe.json")
    print("check --dflash refuses: a changed codebook, an unpacked linear, a head, a bad commit, a held-out request,"
          " quant_lm_head, another block size, no recipe")


def main():
    if sys.argv[1:2] == ["--make-tiny"]:            # quantize_dflash.sh selftest's inputs
        print("tiny drafter, target, taps: " + " ".join(make_tiny(sys.argv[2])))
        return
    root = tempfile.mkdtemp(prefix="dflash-quant-test-")
    try:
        drafter, target, taps = make_tiny(root)
        test_forward_matches_reference(drafter, target, taps)
        test_plan(taps)
        test_sources(root)
        test_finalize_and_check(root, drafter)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ALL PASS")


if __name__ == "__main__":
    main()
