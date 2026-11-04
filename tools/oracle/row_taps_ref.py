#!/usr/bin/env python3
"""Spec 28 (the special-token logit gap): a CPU reference of single rows of Qwen3.8's forced contexts, per layer, for a
bisection against the engine's plain row (tools/probe/probe_row_taps.cc).

    row_taps_ref.py <snapshot> --cases FILE --out DIR [--dtype bf16|f32] [--batch-tokens N]

Cases file, one a line: "<label> <prompt ids file> <force ids file> <row>" - the row that decides force[row] after
prompt + force[0, row) (spec_judge.h / forced_judge.h's row numbering). The model is transformers' Qwen3_5ForCausalLM
(eager attention, layer-streamed as dump_taps.py builds it) on the int4 checkpoint dequantised as every oracle
dequantises it:
  --dtype bf16  bf16 weights and activations, standard transformers bf16 semantics; the bf16 GEMMs run as fp32 GEMMs of
                the bf16 values rounded once to bf16 (cpu_bf16.py: the box CPU has no bf16 units)
  --dtype f32   every weight upcast to f32, f32 throughout (the f32 judges' arithmetic, transformers' code)
Per case, DIR/<label>.<dtype>.safetensors at the row's position:
  mid    [64][H]  the residual after layer L's mixer (post_attention_layernorm's input) - the engine's tap[L]
  out    [64][H]  the residual after layer L (transformers' hidden_states[L + 1])
  x      [H]      the final-normalised hidden; pre [H] the final norm's input
  logits_bf16head / logits_i8head  f32 [V]  x (as stored) times the checkpoint's bf16 head upcast / the engine's
                                             per-row int8 head (int8_sim.int8_head_weight), f32 accumulation
Contexts are right-padded into batches of at most --batch-tokens padded ids (every layer is causal: dump_taps.py)."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402


def _load(name):
    mod_name = "row_taps_ref_" + name.replace("/", "_")
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]


def build(snapshot, dtype):
    import torch
    D = _load("dump")
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    quantised = bool(raw.get("quantization_config") or (raw.get("text_config") or {}).get("quantization_config"))
    group_size = D.check_quant_config(snapshot) if quantised else 0
    tc = AutoConfig.from_pretrained(snapshot).get_text_config()
    tc._attn_implementation = "eager"
    with torch.device("meta"):
        model = m.Qwen3_5ForCausalLM(tc)
    if dtype == "f32":
        # Upcast every floating tensor as it is materialised: the resident ones once, each streamed layer per load.
        orig_attach = D._stream.attach

        def up(sd):
            return {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}

        def attach(model_, layers, layer_sd, resident_sd, **kw):
            return orig_attach(model_, layers, lambda i: up(layer_sd(i)), up(resident_sd), **kw)
        D._stream.attach = attach
    pf = D.stream_weights(model, snapshot, group_size, tc, 0)
    model.eval()
    return model, tc, pf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    a = ap.parse_args()
    import torch
    from safetensors.torch import save_file
    torch.set_grad_enabled(False)
    if a.dtype == "bf16":
        _load("cpu_bf16").install(True)
    I8 = _load("int8_sim")
    cases = []
    with open(a.cases, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) != 4:
                continue
            ids = read_ids(p[1])
            force = read_ids(p[2])
            r = int(p[3])
            cases.append({"label": p[0], "seq": ids + force[:r], "pos": len(ids) + r - 1})
    os.makedirs(a.out, exist_ok=True)
    cases.sort(key=lambda c: -len(c["seq"]))
    batches, cur = [], []
    for c in cases:
        if cur and (len(cur) + 1) * len(cur[0]["seq"]) > a.batch_tokens:
            batches.append(cur)
            cur = []
        cur.append(c)
    if cur:
        batches.append(cur)
    t0 = time.time()
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, dtype {a.dtype}; {len(cases)} cases in "
          f"{len(batches)} forwards", flush=True)
    model, tc, pf = build(a.snapshot, a.dtype)
    L = tc.num_hidden_layers
    for b in batches:
        B, T = len(b), len(b[0]["seq"])
        x = torch.zeros((B, T), dtype=torch.long)
        for i, c in enumerate(b):
            x[i, :len(c["seq"])] = torch.tensor(c["seq"], dtype=torch.long)
        pos = torch.tensor([c["pos"] for c in b])
        ar = torch.arange(B)
        mid, out, fin = {}, {}, {}

        def pre_hook(i):
            def fn(_m, args):
                mid[i] = args[0][ar, pos].detach().clone()
            return fn

        def out_hook(i):
            def fn(_m, _a, o):
                t = o[0] if isinstance(o, tuple) else o
                out[i] = t[ar, pos].detach().clone()
            return fn

        def norm_hook(_m, args, o):
            fin["pre"] = args[0][ar, pos].detach().clone()
            fin["x"] = o[ar, pos].detach().clone()
        hs = []
        for i, layer in enumerate(model.model.layers):
            hs.append(layer.post_attention_layernorm.register_forward_pre_hook(pre_hook(i)))
            hs.append(layer.register_forward_hook(out_hook(i)))
        hs.append(model.model.norm.register_forward_hook(norm_hook))
        tb = time.time()
        try:
            model.model(input_ids=x, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        w = model.lm_head.weight
        wb = w.to(torch.bfloat16)          # the checkpoint's head is bf16 (f32 mode: its exact upcast)
        hb = wb.float()
        h8 = I8.int8_head_weight(wb)
        xs = fin["x"].float()
        lg_b = xs @ hb.t()
        lg_8 = xs @ h8.t()
        del h8
        for i, c in enumerate(b):
            t = {"mid": torch.stack([mid[l][i] for l in range(L)]).contiguous(),
                 "out": torch.stack([out[l][i] for l in range(L)]).contiguous(),
                 "x": fin["x"][i].contiguous(), "pre": fin["pre"][i].contiguous(),
                 "logits_bf16head": lg_b[i].contiguous(), "logits_i8head": lg_8[i].contiguous(),
                 "seq": torch.tensor(c["seq"], dtype=torch.int32)}
            save_file(t, os.path.join(a.out, f"{c['label']}.{a.dtype}.safetensors"),
                      metadata={"pos": str(c["pos"]), "dtype": a.dtype, "snapshot": a.snapshot})
            print(f"[{c['label']}] {len(c['seq'])} ids, row at position {c['pos']}: bf16-head top "
                  f"{int(lg_b[i, :248077].argmax())}", flush=True)
        print(f"forward B={B} T={T}: {time.time() - tb:.0f} s (wall {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
