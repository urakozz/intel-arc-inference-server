#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for the Kolibri-1 quantisation helpers (tools/quantize/kolibri/). CPU, seconds, no model.

    test_kolibri_quant.py [test_name ...]

The logic the stages rest on, without the 156 GB model: row packing, the tool-call parser and the
template's tool turn, the coverage top-up's greedy choice, the bf16 restore (refuses a changed
tensor), the engine-acceptance check on a synthetic export (accepts the right one, rejects each
kind of wrong one), the KL / top-1 / perplexity arithmetic, the card's arm choice, and the committed
prompt lists. The stages end to end on a tiny random Kolibri: tools/quantize/README.md.
"""
import json
import math
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import common as C  # noqa: E402

ref = C.kolibri_ref()


class Skip(Exception):
    pass


def test_pack():
    convs = [[1] * 5, [2] * 3, [3] * 9, [4] * 2, [5] * 4]
    rows, starts = C.pack(convs, 10, seqlen=8)
    assert rows[0] == [1] * 5 + [2] * 3 and starts[0] == 2       # two whole conversations
    assert rows[1] == [3] * 8 and starts[1] == 1                 # one cut to fill the row, its tail dropped
    assert len(rows) == 2                                         # [4,4,5,5,5,5] is a partial row: dropped
    assert C.pack(convs, 1, seqlen=8)[0] == [[1] * 5 + [2] * 3]
    print("  rows start at conversation boundaries; the overflowing one is cut; partial rows dropped")


def test_tool_calls():
    sys.path.insert(0, _HERE)
    import calib
    text = ('<think>\nIch rufe das Wetter ab.\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": '
            '{"ort": "Kiel"}}\n</tool_call>\n<tool_call>\n{"name": "rechner", "arguments": {"ausdruck": "1+1"}}\n'
            '</tool_call><tool_call>{not json}</tool_call>')
    assert calib.parse_tool_calls(text) == ["get_weather", "rechner"]
    assert calib.parse_tool_calls("keine Werkzeuge") == []
    tok_path = os.environ.get("KOLIBRI_TOKENIZER")
    if not tok_path:
        print("  parser ok; KOLIBRI_TOKENIZER not set - the template comparison is skipped")
        return
    tok = C.load_tokenizer(tok_path)
    msgs = [{"role": "user", "content": "Wetter in Kiel?"},
            {"role": "assistant", "content": "", "reasoning": "Ich rufe das Wetter ab.",
             "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"ort": "Kiel"}}}]},
            {"role": "tool", "content": '{"t": 7}'}]
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    head = tok.apply_chat_template(msgs[:2], tokenize=False)
    ids = tok(head, add_special_tokens=False)["input_ids"]
    assert ids[-2:] == tok("<|im_end|>\n", add_special_tokens=False)["input_ids"]
    ours = ids[:-1] + calib.tool_response_ids(tok, ["get_weather"], {"get_weather": '{"t": 7}'})
    assert tok.decode(ours) == full, (tok.decode(ours), full)
    print("  parser ok; the tool turn appended after a generated <|im_end|> == the chat template's rendering")


def test_greedy_topup():
    import coverage
    deficit = torch.tensor([[5, 0, 3], [0, 2, 0]])
    per_row = torch.tensor([[[1, 0, 0], [0, 0, 0]],          # gain 1
                            [[4, 9, 0], [0, 2, 0]],          # gain 6
                            [[0, 0, 3], [0, 0, 7]],          # gain 3
                            [[0, 0, 0], [0, 0, 0]]], dtype=torch.int32)
    chosen = coverage.greedy_topup(deficit, per_row)
    assert chosen == [1, 2, 0] and int(deficit.sum()) == 0, (chosen, deficit)
    d2 = torch.tensor([[0, 0, 50]])
    assert coverage.greedy_topup(d2, torch.tensor([[[1, 1, 0]]], dtype=torch.int32)) == [] and int(d2.sum()) == 50
    print("  most deficit filled first; stops when no row helps")


def _tiny_export(tmp: str, attn: str = "int4"):
    """A bf16 source and a matching int4 export (experts, and attention when attn == int4)."""
    sys.path.insert(0, os.path.join(C.ORACLE))
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_kolibri_ref", os.path.join(C.ORACLE, "test_kolibri_ref.py"))
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    d, c, sd = t.tiny(seed=3, hidden_size=64, moe_intermediate_size=64, shared_expert_intermediate_size=64,
                      num_attention_heads=4, head_dim=16, num_key_value_heads=4)
    src = os.path.join(tmp, "src")
    exp = os.path.join(tmp, "exp")
    os.makedirs(src)
    os.makedirs(exp)
    t.write_checkpoint(src, d, sd)
    q = {}
    for name, w in sd.items():
        if ".mlp.experts." in name or (attn == "int4" and ".self_attn." in name and name.endswith("_proj.weight")):
            base = name[:-len(".weight")]
            qw, s, qz, _ = t.gptq_pack(w)
            q[base + ".qweight"], q[base + ".scales"], q[base + ".qzeros"] = qw, s, qz
        else:
            q[name] = w.float()                         # AutoRound on a CPU writes these in fp32
    qc = {"bits": 4, "group_size": 64, "sym": True, "quant_method": "auto-round",
          "packing_format": "auto_round:auto_gptq",
          "extra_config": {f"model.layers.{i}.mlp.shared_experts.down_proj": {"bits": 16, "data_type": "float"}
                           for i in range(c.num_hidden_layers)}}
    t.write_checkpoint(exp, dict(d, quantization_config=qc), q)
    return src, exp, c


def test_restore_and_check():
    import check
    with tempfile.TemporaryDirectory() as tmp:
        src, exp, c = _tiny_export(tmp)
        errs, _ = check.check(exp, src, "int4", 8)
        assert any("must be BF16" in e for e in errs), errs[:3]          # fp32 leftovers are rejected
        stats = C.restore_bf16(exp, src)
        assert stats["restored"] > 0 and stats["shards_rewritten"] > 0, stats
        errs, info = check.check(exp, src, "int4", 8)
        assert not errs, errs[:5]
        assert info["dequant_rel_err_max"] < 0.2 and info["int4_linears"] == c.num_hidden_layers * (3 * c.num_experts + 4)
        assert C.read_json(os.path.join(exp, "config.json"))["dtype"] == "bfloat16"
        print(f"  restored {stats['restored']} fp32 tensors to the source's bf16; ACCEPTED "
              f"(dequant rel err {info['dequant_rel_err_max']:.3f})")
        # each kind of wrong export is rejected
        from safetensors.torch import load_file, save_file
        fn = sorted(f for f in os.listdir(exp) if f.endswith(".safetensors"))
        shards = {f: load_file(os.path.join(exp, f)) for f in fn}

        def mutate(fnc):
            with tempfile.TemporaryDirectory() as t2:
                for f in os.listdir(exp):
                    if not f.endswith(".safetensors"):
                        with open(os.path.join(exp, f), "rb") as a, open(os.path.join(t2, f), "wb") as b:
                            b.write(a.read())
                new = {f: dict(v) for f, v in shards.items()}
                fnc(new, t2)
                for f, v in new.items():
                    save_file(v, os.path.join(t2, f))
                return check.check(t2, src, "int4", 10 ** 6)[0]       # every int4 linear dequantised

        def find(new, suffix):
            for f, v in new.items():
                for k in v:
                    if k.endswith(suffix):
                        return f, k
            raise KeyError(suffix)

        def bad_qz(new, _):
            f, k = find(new, "experts.3.up_proj.qzeros")
            new[f][k] = new[f][k].clone()
            new[f][k][0, 0] = 0x66666666
        assert any("expected 0x77777777" in e for e in mutate(bad_qz))

        def bad_gidx(new, _):
            f, k = find(new, "experts.0.gate_proj.qweight")
            K = new[f][k].shape[0] * 8
            g = (torch.arange(K, dtype=torch.int32) // 64)
            g[0] = 1                                     # k = 0 claims group 1: a permutation
            new[f][k[:-len(".qweight")] + ".g_idx"] = g
        assert any("activation-order" in e for e in mutate(bad_gidx))

        def bias_changed(new, _):
            f, k = find(new, "layers.1.moe.router.expert_bias")
            new[f][k] = new[f][k].clone() + 0.125
        assert any("differs from the bf16 source" in e for e in mutate(bias_changed))

        def shared_quantised(new, _):
            f, k = find(new, "layers.0.mlp.shared_experts.up_proj.weight")
            del new[f][k]
            new[f][k[:-len(".weight")] + ".qweight"] = torch.zeros(8, 64, dtype=torch.int32)
            new[f][k[:-len(".weight")] + ".scales"] = torch.ones(1, 64, dtype=torch.float16)
        assert any("must stay bf16" in e for e in mutate(shared_quantised))

        def transposed(new, _):
            f, k = find(new, "experts.5.down_proj.qweight")
            w = new[f][k]
            new[f][k] = w.flip(1).contiguous()
        assert any("relative error" in e for e in mutate(transposed))

        def asym(_, d2):
            cfg = C.read_json(os.path.join(d2, "config.json"))
            cfg["quantization_config"]["sym"] = False
            C.write_json(os.path.join(d2, "config.json"), cfg)
        assert any("sym true" in e for e in mutate(asym))
        print("  rejected: a bad qzeros word, a permuting g_idx, a changed expert_bias, a quantised shared "
              "expert, a column-permuted qweight, an asymmetric config")


def test_restore_refuses_changed_tensor():
    with tempfile.TemporaryDirectory() as tmp:
        src, exp, _ = _tiny_export(tmp)
        from safetensors.torch import load_file, save_file
        f = sorted(x for x in os.listdir(exp) if x.endswith(".safetensors"))[0]
        sd = load_file(os.path.join(exp, f))
        k = next(k for k in sd if k.endswith("norm.weight"))
        sd[k] = sd[k] * 1.0001
        save_file(sd, os.path.join(exp, f))
        try:
            C.restore_bf16(exp, src)
        except SystemExit as e:
            assert "refusing" in str(e)
            print("  a tensor that is not the widened source is refused, not rounded back")
            return
        raise AssertionError("restore_bf16 accepted a changed tensor")


def test_eval_metrics():
    import evaluate

    class H:                          # a "reference" whose head is the identity on fp32 logits
        def head(self, x):
            return x.float()
    V, T = 7, 5
    lp = torch.randn(1, T, V)
    ids = torch.randint(0, V, (1, T))
    s = evaluate.metrics(H(), lp, H(), lp.clone(), ids, "cpu", chunk=2)
    m = evaluate.finish(s)
    assert m["tokens"] == T - 1 and abs(m["kl"]) < 1e-6 and m["top1"] == 1.0 and abs(m["ppl_bf16"] - m["ppl_quant"]) < 1e-6
    q = lp.clone()
    q[0, :, 0] += 2.0
    m2 = evaluate.finish(evaluate.metrics(H(), lp, H(), q, ids, "cpu"))
    P, Q = torch.log_softmax(lp[0, :-1], -1), torch.log_softmax(q[0, :-1], -1)
    want = float((P.exp() * (P - Q)).sum(-1).mean())
    assert abs(m2["kl"] - want) < 1e-6 and m2["kl"] > 0
    nll = -P.gather(1, ids[0, 1:, None]).mean()
    assert abs(m2["ppl_bf16"] - math.exp(float(nll))) < 1e-4
    b = evaluate.bars({"de": {"kl": 0.055}, "en": {"kl": 0.014}})
    assert b == {"de_target": True, "de_minimum": True, "en": False}
    print(f"  KL(P||Q) = mean over positions of sum p (log p - log q) ({m2['kl']:.4f}); bars")


def test_card_pick():
    import card
    ok = {"bars": {"de_target": False, "de_minimum": True, "en": True}}
    bad = {"bars": {"de_target": False, "de_minimum": False, "en": True}}
    assert card.pick({"tune-attn_int4": ok, "tune-attn_bf16": ok}, "auto") == "tune-attn_int4"
    assert card.pick({"tune-attn_int4": bad, "tune-attn_bf16": ok}, "auto") == "tune-attn_bf16"
    assert card.pick({"tune-attn_int4": bad, "rtn-attn_int4": ok}, "auto") is None
    assert card.pick({"rtn-attn_int4": bad}, "rtn-attn_int4") == "rtn-attn_int4"
    print("  int4 attention if it clears the bars, else bf16 attention, else no card")


def test_prompt_lists():
    P = {n: C.read_jsonl(os.path.join(C.PROMPTS, n + ".jsonl"))
         for n in ("de_chat", "en_chat", "code", "de_tools", "de_doc_tasks")}
    for n, rows in P.items():
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), n
        assert {r["split"] for r in rows} <= {"calib", "eval"}, n
        if n != "de_doc_tasks":
            assert any(r["split"] == "eval" for r in rows) and any(r["split"] == "calib" for r in rows), n
    for r in P["de_tools"]:
        names = [t["function"]["name"] for t in r["tools"]]
        assert set(r["tool_results"]) == set(names), r["id"]
        for v in r["tool_results"].values():
            json.loads(v)
    print("  " + ", ".join(f"{n} {len(v)}" for n, v in P.items()) + "; splits and tool results consistent")


def test_synth_pack_roundtrip():
    # make_synth's RTN packer against dequant.py's rule: w_hat = (q - 8) * scale (spec 20c Task 1)
    import make_synth as S
    g = torch.Generator().manual_seed(0)
    w = torch.randn(128, 256, generator=g).to(torch.bfloat16)          # [N, K]
    qw, sc, qz = S.pack_rtn_g64(w)                                     # I32 [K/8, N], F16 [K/64, N], I32
    assert qw.shape == (32, 128) and sc.shape == (4, 128) and (qz == 0x77777777).all()
    assert qw.dtype == torch.int32 and sc.dtype == torch.float16 and qz.dtype == torch.int32
    deq = S.dequant(qw, sc)                                            # [N, K] fp32, dequant.py's rule
    rel = (deq - w.float()).norm() / w.float().norm()
    assert rel < 0.15
    # nibble order: element k of a column sits in word k // 8, bits 4 (k % 8) - dequant.py's rule
    assert torch.equal(S.dequant(qw, sc)[:, 0], (((qw[0] & 0xF) - 8).float() * sc[0].float()))
    # ... and the same bits as dequant.py itself (its bf16 output, transposed)
    assert torch.equal(ref._dequant.dequant_gptq(qw, sc, 64).t().float(), deq.to(torch.bfloat16).float())
    print(f"  RTN g64 round trip: relative error {float(rel):.4f}; nibble order and dequant.py agree")


def test_synth_config_prefix():
    import make_synth as S
    d = S.synth_config(layers=5, attn="bf16")
    assert d["num_hidden_layers"] == 5 and d["layer_types"][4] == "full_attention"
    assert d["layer_types"][:4] == ["sliding_attention"] * 4 and d["hidden_size"] == 2560
    assert d["quantization_config"]["packing_format"] == "auto_round:auto_gptq"
    assert all(d["quantization_config"]["extra_config"][f"model.layers.{i}.self_attn.q_proj"]["bits"] == 16
               for i in range(5))
    i4 = S.synth_config(layers=1, attn="int4")
    assert i4["layer_types"] == ["sliding_attention"] and not any("self_attn" in k for k in
                                                                    i4["quantization_config"]["extra_config"])
    ref.KConfig.from_dict({k: v for k, v in d.items() if k != "quantization_config"})   # the reference reads it
    print("  5 layers = 4 sliding + full layer 4; auto_round:auto_gptq; bf16 attention excluded per layer")


TESTS = [test_pack, test_tool_calls, test_greedy_topup, test_restore_and_check, test_restore_refuses_changed_tensor,
         test_eval_metrics, test_card_pick, test_prompt_lists, test_synth_pack_roundtrip, test_synth_config_prefix]


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
