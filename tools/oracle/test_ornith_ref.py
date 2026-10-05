#!/usr/bin/env python3
"""Checks for ornith_ref.py (spec 15a). CPU, tiny random weights, ~1 min (most of it building
stream.py's C++ dequant on first use).

    test_ornith_ref.py [test_name ...]

A tiny Qwen3.5-MoE (4 layers: 3 GDN + 1 FA, 8 experts top-2, hidden 128) is written as an int4
checkpoint in the real export's naming - per-expert GPTQ `mlp.experts.E.{gate,up,down}_proj`,
int4 in_proj_a / in_proj_b, bf16 routers / shared gates / norms / embed / head, a bf16 MoE MTP
head under `mtp.*` - and:

  test_streamed_equals_resident  the layer-streamed model with the routed experts dequantised on
      demand (ornith_ref.build_streamed) against transformers' resident model on the same
      dequantised weights (gate||up stacked per expert): every logits row of a prompt forward and
      three cached decode steps BITWISE, both experts implementations; and it really is lazy
      (a decode step dequantises only its top-k experts per layer);
  test_rtn                       rtn_int4_g64 against loader/rtn.h's formula written out per
      element (zero group, signed values, the stored-scale division);
  test_cli_run                   `run` end to end with --mtp-out: the golden file's routing rows
      (T + gen per layer, the router's own logits), the M1 file and the 33 continuation ids.
"""
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("ornith_ref", os.path.join(_HERE, "ornith_ref.py"))
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

E, K_TOP, H, I = 8, 2, 128, 64


def tiny_config() -> dict:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
    text = dict(
        model_type="qwen3_5_moe_text", hidden_size=H, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], full_attention_interval=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, attn_output_gate=True,
        linear_num_key_heads=4, linear_key_head_dim=32, linear_num_value_heads=8,
        linear_value_head_dim=32, linear_conv_kernel_dim=4, num_experts=E, num_experts_per_tok=K_TOP,
        moe_intermediate_size=I, shared_expert_intermediate_size=I, vocab_size=512,
        max_position_embeddings=1024, rms_norm_eps=1e-6, tie_word_embeddings=False,
        mtp_num_hidden_layers=1, hidden_act="silu", dtype="bfloat16",
        rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                         "mrope_interleaved": True, "mrope_section": [3, 3, 2]})
    d = Qwen3_5MoeConfig(text_config=text).to_dict()
    d["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    d["quantization_config"] = {"bits": 4, "group_size": 64, "sym": True, "desc_act": False,
                                "quant_method": "gptq", "provider": "auto-round"}
    return d


def pack(w_q: torch.Tensor, sc: torch.Tensor):
    """[N, K] ints 0..15 + [K/64, N] f16 -> GPTQ tensors (qweight, qzeros, g_idx)."""
    N, K = w_q.shape
    qw = ref.gptq_pack(w_q.t().contiguous())
    qz = torch.full((K // 64, max(N // 8, 1)), 0x77777777, dtype=torch.int32)
    return qw, qz, torch.arange(K, dtype=torch.int32) // 64


def make_checkpoint(dirname: str, seed: int = 0):
    """Writes the tiny int4 checkpoint; returns (config dict, the dequantised bf16 state dict of
    Qwen3_5MoeForCausalLM, the mtp tensors)."""
    from safetensors.torch import save_file
    d = tiny_config()
    tc = ref.text_config_from_dict(d)
    m = ref.mqm()
    with torch.device("meta"):
        model = m.Qwen3_5MoeForCausalLM(tc)
    g = torch.Generator().manual_seed(seed)
    files, deq = {}, {}

    def int4(name, N, K):
        q = torch.randint(0, 16, (N, K), generator=g)
        sc = ((torch.rand(K // 64, N, generator=g) - 0.5) * 0.08).to(torch.float16)
        qw, qz, gi = pack(q, sc)
        files.update({name + ".qweight": qw, name + ".scales": sc, name + ".qzeros": qz, name + ".g_idx": gi})
        return ((q.float() - 8) * sc.float().repeat_interleave(64, 0).t()).to(torch.bfloat16)

    for k, p in model.state_dict().items():
        ck = k.replace("model.layers.", "model.language_model.layers.", 1).replace(
            "model.embed_tokens", "model.language_model.embed_tokens").replace(
            "model.norm.", "model.language_model.norm.")
        if k.endswith("mlp.experts.gate_up_proj"):
            base = ck[: -len("gate_up_proj")]
            w = torch.empty(E, 2 * I, H, dtype=torch.bfloat16)
            for e in range(E):
                w[e, :I] = int4(f"{base}{e}.gate_proj", I, H)
                w[e, I:] = int4(f"{base}{e}.up_proj", I, H)
            deq[k] = w
        elif k.endswith("mlp.experts.down_proj"):
            base = ck[: -len("down_proj")]
            w = torch.empty(E, H, I, dtype=torch.bfloat16)
            for e in range(E):
                w[e] = int4(f"{base}{e}.down_proj", H, I)
            deq[k] = w
        elif (k.startswith("model.layers.") and k.endswith("_proj.weight") and p.dim() == 2
              and p.shape[1] % 64 == 0):
            deq[k] = int4(ck[: -len(".weight")], *p.shape)
        else:
            if k.endswith("A_log"):
                t = -torch.rand(p.shape, generator=g)
            elif "norm" in k:
                t = 0.1 * torch.randn(p.shape, generator=g)
            else:
                t = 0.05 * torch.randn(p.shape, generator=g)
            deq[k] = files[ck] = t.to(torch.bfloat16)
    # the MoE MTP head, bf16, per-expert (the published export's naming)
    mtp = {"mtp.fc.weight": (H, 2 * H), "mtp.norm.weight": (H,), "mtp.pre_fc_norm_embedding.weight": (H,),
           "mtp.pre_fc_norm_hidden.weight": (H,)}
    with torch.device("meta"):
        layer = m.Qwen3_5MoeDecoderLayer(tc, 3)
    for k, p in layer.state_dict().items():
        if k == "mlp.experts.gate_up_proj":
            for e in range(E):
                mtp[f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight"] = (I, H)
                mtp[f"mtp.layers.0.mlp.experts.{e}.up_proj.weight"] = (I, H)
        elif k == "mlp.experts.down_proj":
            for e in range(E):
                mtp[f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"] = (H, I)
        else:
            mtp["mtp.layers.0." + k] = tuple(p.shape)
    mt = {k: (0.05 * torch.randn(s, generator=g)).to(torch.bfloat16) for k, s in mtp.items()}
    files.update(mt)
    shard = {k: v.contiguous() for k, v in files.items() if not k.startswith("mtp.")}
    save_file(shard, os.path.join(dirname, "model-00001-of-00001.safetensors"))
    save_file({k: v.contiguous() for k, v in mt.items()}, os.path.join(dirname, "model_extra_tensors.safetensors"))
    wm = {k: "model-00001-of-00001.safetensors" for k in shard}
    wm.update({k: "model_extra_tensors.safetensors" for k in mt})
    with open(os.path.join(dirname, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f)
    with open(os.path.join(dirname, "config.json"), "w", encoding="utf-8") as f:
        json.dump(d, f)
    return d, deq, mt


def resident(d: dict, deq: dict, experts: str):
    tc = ref.text_config_from_dict(d, experts)
    with torch.device("meta"):
        model = ref.mqm().Qwen3_5MoeForCausalLM(tc)
    model.load_state_dict(deq, strict=True, assign=True)
    model.model.rotary_emb = type(model.model.rotary_emb)(tc)
    return model.eval()


def greedy(model, ids, gen):
    rows = []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=True)
        rows.append(out.logits[0].float())
        cache = out.past_key_values
        for _ in range(gen):
            nxt = int(rows[-1][-1].argmax())
            out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            rows.append(out.logits[0].float())
    return torch.cat(rows, 0)


def test_streamed_equals_resident():
    ids = [5, 17, 300, 42, 7, 99, 128, 3, 250, 11]
    with tempfile.TemporaryDirectory() as tmp:
        d, deq, _ = make_checkpoint(tmp)
        for experts in ("grouped_mm", "eager"):
            want = greedy(resident(d, deq, experts), ids, 3)
            tc = ref.text_config(tmp, experts)
            model, _, lazy = ref.build_streamed(tmp, tc)
            assert model.config._experts_implementation == experts, model.config._experts_implementation
            got = greedy(model, ids, 3)
            assert torch.equal(got, want), (experts, (got - want).abs().max())
            before = lazy.filled
            with torch.no_grad():
                model(input_ids=torch.tensor([[7]]), use_cache=False)
            per_step = lazy.filled - before
            assert per_step <= tc.num_hidden_layers * K_TOP, per_step
            print(f"  {experts}: streamed + lazy experts == resident, bitwise over {got.shape[0]} rows; "
                  f"{per_step} expert dequants for a one-token forward")


def test_rtn():
    import struct

    def f16(x: float) -> float:
        return struct.unpack("<e", struct.pack("<e", x))[0]
    g = torch.Generator().manual_seed(3)
    w = (0.03 * torch.randn(3, 128, generator=g)).to(torch.bfloat16)
    w[1, 64:] = 0                                              # an all-zero group
    w[2, 5] = -0.25                                            # a negative amax
    qw, sc = ref.rtn_int4_g64(w)
    q = torch.stack([(qw >> (4 * i)) & 15 for i in range(8)], 1).reshape(128, 3)
    for n in range(3):
        for gi in range(2):
            vals = [float(v) for v in w[n, gi * 64:(gi + 1) * 64].float()]
            amax = max(abs(v) for v in vals)
            s = f16(float(torch.tensor(2.0 * amax, dtype=torch.float32) / 15.0))
            assert float(sc[gi, n]) == s, (n, gi)
            for k, v in enumerate(vals):
                want = 8 if s == 0 else min(15, max(0, int(torch.round(torch.tensor(v) / torch.tensor(s))) + 8))
                assert int(q[gi * 64 + k, n]) == want, (n, gi, k)
    deq = ref.rtn_dequant(w)
    assert deq.shape == w.shape and torch.equal(deq[1, 64:], torch.zeros(64, dtype=torch.bfloat16))
    print("  rtn_int4_g64 = loader/rtn.h's formula per element (zero group, negative amax)")


def test_cli_run():
    from safetensors import safe_open
    ids = [5, 17, 300, 42, 7, 99]
    with tempfile.TemporaryDirectory() as tmp:
        make_checkpoint(tmp, seed=1)
        pf = os.path.join(tmp, "prose.ids")
        open(pf, "w").write(" ".join(map(str, ids)))
        out = os.path.join(tmp, "out", "prose.golden.safetensors")
        mo = os.path.join(tmp, "mtp")
        a = SimpleNamespace(snapshot=tmp, prompt=pf, out=out, gen=4, max_prompt=64, experts="grouped_mm",
                            mtp_out=mo, mtp_rows=3, mtp_experts="rtn")
        ref.cmd_run(a)
        with safe_open(out, framework="pt") as h:
            keys = set(h.keys())
            assert h.get_tensor("router_logits.L0").shape == (len(ids) + 4, E)
            assert h.get_tensor("router_logits.L0").dtype == torch.bfloat16
            assert h.get_tensor("shared_gate_logits.L3").shape == (len(ids) + 4, 1)
            assert h.get_tensor("route.ids.L2").shape == (len(ids) + 4, K_TOP)
            assert h.get_tensor("logits").shape[0] == len(ids) + 4
            toks = h.get_tensor("tokens").tolist()
            assert h.metadata()["experts_implementation"] == "grouped_mm"
        assert {"gdn_state.L0", "conv_state.L2", "resid.L3", "mixer.L3", "mlp.L0"} <= keys
        cont = [int(x) for x in open(os.path.join(mo, "prose.cont256.ids")).read().split()]
        assert cont[:4] == toks and len(cont) == 5
        with safe_open(os.path.join(mo, "m1", "prose.mtp.safetensors"), framework="pt") as h:
            assert h.get_tensor("logits").shape == (3, 512)
            assert h.get_tensor("pos").tolist() == [5, 6, 7]
            assert h.get_tensor("next").tolist() == cont[:3]
    print("  run: routing rows per layer = T + gen, M1 rows n-1.., 5 continuation ids")


TESTS = [test_rtn, test_streamed_equals_resident, test_cli_run]


def main() -> None:
    only = sys.argv[1:]
    for t in TESTS:
        if only and t.__name__ not in only:
            continue
        print(t.__name__)
        t()
    print("ok")


if __name__ == "__main__":
    main()
