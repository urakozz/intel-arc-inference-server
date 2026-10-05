#!/usr/bin/env python3
"""Checks for dump_taps.py (spec 19a Task 2). CPU, a tiny random Qwen3.5 checkpoint, seconds.

    test_dump_taps.py [test_name ...]

The checkpoint is written in Qwen3.8's layout (config.json with a text_config,
`model.language_model.*` + `lm_head.weight` over two shards and an index, bf16) and run through
the same layer-streamed path the real dump takes. Review Focus 1: a dumped tap for
target_layer_id i equals the reference model's output of decoder layer i (its forward hook and
transformers' hidden_states[i + 1]) and is not layer i's input; dflash_ref.py's helper agrees.
Also: greedy / top-k / logsumexp against the full logits (vocab_used masking, ties to the lower
id), a right-padded batch against one-sequence forwards, the file round trip and the resume.
"""
import argparse
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DT = _load("dump_taps")
REF = _load("dflash_ref")

VOCAB, VOCAB_USED = 320, 300
LAYERS = [1, 3, 5]          # GDN, FA (layer 3), GDN
TEXT = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=8, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
            full_attention_interval=4, vocab_size=VOCAB, rms_norm_eps=1e-6, attn_output_gate=True,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                             "mrope_section": [2, 1, 1], "mrope_interleaved": True},
            partial_rotary_factor=0.25, tie_word_embeddings=False, model_type="qwen3_5_text",
            layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2)

_STATE = {}


def tiny_checkpoint():
    """(snapshot dir, resident reference model) - built once per process."""
    if "snap" in _STATE:
        return _STATE["snap"], _STATE["model"]
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    tc = Qwen3_5TextConfig(**TEXT)
    tc._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(tc).to(torch.bfloat16).eval()
    model.model.rotary_emb = type(model.model.rotary_emb)(tc)   # fp32 inv_freq, as dump.py rebuilds it
    with torch.no_grad():
        for n, p in model.named_parameters():          # O(1) activations through 8 layers
            if p.dim() == 2 and "embed" not in n:
                p.copy_(torch.randn_like(p.float()) / p.shape[1] ** 0.5)
        # rows the argmax must never pick: larger than any real logit
        model.lm_head.weight[VOCAB_USED:] = model.lm_head.weight[:VOCAB - VOCAB_USED] * 50
    root = tempfile.mkdtemp(prefix="dump_taps_test_")
    shards = ({}, {})
    for n, t in model.state_dict().items():
        name = "lm_head.weight" if n == "lm_head.weight" else "model.language_model." + n[len("model."):]
        shards[0 if ".layers." in name and int(name.split(".layers.")[1].split(".")[0]) < 4 else 1][name] = \
            t.contiguous()
    shards[1]["mtp.fc.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)   # skipped by name
    wm = {}
    for k, sd in enumerate(shards):
        fn = f"model-0000{k + 1}-of-00002.safetensors"
        save_file(sd, os.path.join(root, fn))
        wm.update({n: fn for n in sd})
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": wm}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
                   "text_config": TEXT}, f)
    _STATE["snap"], _STATE["model"] = root, model
    return root, model


def streamed():
    if "streamed" not in _STATE:
        snap, _ = tiny_checkpoint()
        _STATE["streamed"] = DT.build_target(snap, fp32_matmul=True)[0]
    return _STATE["streamed"]


def seq(n, seed):
    return torch.randint(0, VOCAB_USED, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def reference(model, ids):
    """The resident model: per-layer outputs by forward hook, hidden_states, final hidden."""
    outs = {}
    hs = [model.model.layers[i].register_forward_hook(
        (lambda i: lambda m, a, o: outs.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0]))(i))
        for i in range(len(model.model.layers))]
    try:
        with torch.no_grad():
            r = model.model(input_ids=torch.tensor([ids]), use_cache=False, output_hidden_states=True)
    finally:
        for h in hs:
            h.remove()
    return outs, [h[0] for h in r.hidden_states], r.last_hidden_state[0]


def test_tap_offset():
    """Review Focus 1: tap i = OUTPUT of layer i = hidden_states[i + 1], not layer i's input."""
    _, model = tiny_checkpoint()
    m = streamed()                                  # also turns on fp32 matmul, for both models
    ids = seq(23, 1)
    (r,) = DT.forward_taps(m, [ids], LAYERS, VOCAB_USED, topk=8)
    outs, hs, _ = reference(model, ids)
    for i in LAYERS:
        assert torch.equal(r["taps"][i], outs[i].to(torch.bfloat16)), f"layer {i}: tap != layer output"
        assert torch.equal(r["taps"][i], hs[i + 1]), f"layer {i}: tap != hidden_states[{i + 1}]"
        assert not torch.allclose(r["taps"][i].float(), hs[i].float(), atol=1e-2), f"layer {i}: tap = its input"
    via_ref = REF.taps_from_hf_hidden_states(hs, LAYERS)
    assert all(torch.equal(via_ref[i], r["taps"][i]) for i in LAYERS)
    print(f"tap offset: dumped taps {LAYERS} == layer outputs == hidden_states[i+1] (bitwise), != inputs")


def test_greedy_topk():
    _, model = tiny_checkpoint()
    m = streamed()
    ids = seq(19, 2)
    (r,) = DT.forward_taps(m, [ids], LAYERS, VOCAB_USED, topk=8, keep_from=[5])
    _, _, last = reference(model, ids)
    lg = last.float() @ model.lm_head.weight.float().t()
    assert lg.argmax(-1).ge(VOCAB_USED).any(), "fixture: some unmasked argmax lands past vocab_used"
    lu = lg[5:, :VOCAB_USED]
    assert torch.equal(r["greedy"], lu.argmax(-1)), "greedy != argmax over ids < vocab_used"
    v, i = torch.sort(lu, dim=-1, descending=True, stable=True)
    assert torch.equal(r["top_ids"], i[:, :8]) and torch.allclose(r["top_logits"], v[:, :8], atol=1e-5)
    assert torch.allclose(r["lse"], torch.logsumexp(lu.double(), -1).float(), atol=1e-5)
    assert r["taps"][1].shape == (14, 64) and r["greedy"].shape == (14,)
    # ties: equal logits order by id
    h = torch.tensor([[1.0, 0.0]])
    w = torch.tensor([[0.0, 1.0], [2.0, 0.0], [1.0, 5.0], [2.0, 0.0], [2.0, 0.0]], dtype=torch.bfloat16)
    g, ti, tv, _ = DT.head_stats(h, w, 5, 4, chunk=2)
    assert g.tolist() == [1] and ti[0].tolist() == [1, 3, 4, 2], ti
    print("greedy / top-k / lse: masked at vocab_used, ties to the lower id, positions from keep_from")


def test_padded_batch():
    m = streamed()
    a, b, c = seq(31, 3), seq(12, 4), seq(20, 5)
    batch = DT.forward_taps(m, [a, b, c], LAYERS, VOCAB_USED, topk=4, keep_from=[0, 2, 0])
    # 1. the padding never reaches a real position: other pad ids, bitwise the same rows
    other = DT.forward_taps(m, [a, b, c], LAYERS, VOCAB_USED, topk=4, keep_from=[0, 2, 0], pad_id=VOCAB_USED - 1)
    for r, o in zip(batch, other):
        assert torch.equal(r["greedy"], o["greedy"]) and all(torch.equal(r["taps"][i], o["taps"][i]) for i in LAYERS)
    # 2. against one-sequence forwards: NOT bitwise. The projections are (checked while writing
    # this), but the GDN core's fp32 batched matmuls change blocking with the batch shape, even for
    # two copies of one sequence; the WY triangular solve of this random model amplifies that to
    # cos 0.998 after layer 1 and ~0.97 by layer 6. So cosine at the first tap and greedy agreement.
    cos_first, cos_min, agree, n = 1.0, 1.0, 0, 0
    for s, k, r in zip((a, b, c), (0, 2, 0), batch):
        (one,) = DT.forward_taps(m, [s], LAYERS, VOCAB_USED, topk=4, keep_from=[k])
        agree += int((r["greedy"] == one["greedy"]).sum())
        n += len(one["greedy"])
        for i in LAYERS:
            cos = torch.nn.functional.cosine_similarity(r["taps"][i].float(), one["taps"][i].float(), dim=-1)
            cos_min = min(cos_min, cos.min().item())
            if i == LAYERS[0]:
                cos_first = min(cos_first, cos.min().item())
    assert cos_first > 0.995 and agree >= n - 3, (cos_first, agree, n)
    print(f"right-padded batch: pad content bitwise irrelevant; vs one-sequence forwards tap cosine >= "
          f"{cos_first:.5f} at layer {LAYERS[0]} ({cos_min:.4f} at the last tap), greedy {agree}/{n}")


def test_files_and_resume():
    snap, _ = tiny_checkpoint()
    streamed()
    with tempfile.TemporaryDirectory() as d:
        files = {}
        for name, ids in (("p1", seq(15, 6)), ("c1", seq(9, 7)), ("p2", seq(11, 8)), ("c2", seq(4, 9))):
            files[name] = os.path.join(d, name + ".ids")
            with open(files[name], "w") as f:
                f.write(" ".join(map(str, ids)))
        out = os.path.join(d, "out")
        ns = argparse.Namespace(snapshot=snap, out_dir=out, layers="1,3,5", batch_tokens=64, keep_ctx=10,
                                topk=4, vocab_used=VOCAB_USED, fp32_matmul=True, dry_run=False, a4=None,
                                source=[f"prose/one:{files['p1']}:{files['c1']}",
                                        f"code/two:{files['p2']}:{files['c2']}"])
        DT.cmd_run(ns)
        x = DT.load_dump(os.path.join(out, "prose__one.taps.safetensors"))
        assert x["n_prompt"] == 15 and x["n"] == 24 and x["tap_from"] == 5 and x["layers"] == [1, 3, 5]
        assert x["taps"][3].shape == (19, 64) and x["greedy"].shape == (19,) and len(x["ids"]) == 24
        assert x["meta"]["fp32_matmul"] == "True" and 0.0 <= float(x["meta"]["cont_agree"]) <= 1.0
        (direct,) = DT.forward_taps(streamed(), [x["ids"].tolist()], [1, 3, 5], VOCAB_USED, 4, [5])
        assert torch.equal(direct["greedy"], x["greedy"]) and torch.equal(direct["taps"][5], x["taps"][5])
        src = json.load(open(os.path.join(out, "sources.json")))
        assert [s["name"] for s in src] == ["prose/one", "code/two"] and src[1]["n"] == 15
        mt = os.path.getmtime(os.path.join(out, "code__two.taps.safetensors"))
        DT.cmd_run(ns)                                   # everything dumped: nothing reloads
        assert os.path.getmtime(os.path.join(out, "code__two.taps.safetensors")) == mt
    print("files: metadata, tap_from = n_prompt - keep_ctx, round trip, sources.json, resume skips")


def test_plan():
    b = DT.plan_batches([100, 30, 300, 50, 60], 200)
    assert b == [[2], [0, 4], [3, 1]], b
    e = DT.estimate([3000, 1000], DT.plan_batches([3000, 1000], 8192))
    assert e["forwards"] == 1 and e["padded"] == 6000 and e["peak_gib"] > 5
    print(f"plan: longest first, padded size within batch-tokens; estimate {e['est_hours']:.2f} h, "
          f"{e['peak_gib']:.1f} GiB for 2 x 3000 padded")


TESTS = [test_plan, test_tap_offset, test_greedy_topk, test_padded_batch, test_files_and_resume]


def main() -> None:
    torch.set_grad_enabled(False)
    only = sys.argv[1:]
    for t in TESTS:
        if not only or t.__name__ in only:
            t()
    print("ok")


if __name__ == "__main__":
    main()
