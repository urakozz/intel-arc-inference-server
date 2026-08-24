#!/usr/bin/env python3
"""The oracle: layer-by-layer golden tensors for one prompt, from transformers on CPU.

This is the reference the C++ engine is tested against (doc 04 §testing, doc 10).
The model is built from the config and loaded from a state dict we dequantise
ourselves with tools/oracle/dequant.py - the SAME function the C++ loader is
bit-compared against - so the oracle and the engine start from identical bf16
weights. No `fla` is installed in the reference image on purpose: the pure-torch
gated-delta-rule in modeling_qwen3_5.py IS the contract (doc 03).

    dump.py <snapshot> --prompt <ids-file> --out <out.safetensors> [--gen 32] [--max-prompt 64]

Output tensors (batch dimension squeezed; batch is always 1):
    resid.L{i}      bf16 [T, hidden]   decoder layer i output, all prompt positions
    mixer.L{i}      bf16 [T, hidden]   linear_attn / self_attn submodule output
    mlp.L{i}        bf16 [T, hidden]   mlp submodule output
    gdn_state.L{i}  f32  [Hv, Dk, Dv]  recurrent state after the prompt, GDN layers only
    conv_state.L{i} bf16 [conv_dim, K] conv window after the prompt, GDN layers only
    logits          f32  [T + gen, V]  every prompt position, then one row per generated step
    tokens          i32  [gen]         greedy continuation

    logits[t]      for t < T is prompt position t;  tokens[0]   = argmax(logits[T-1])
    logits[T + j]  is the forward that consumed tokens[j];      tokens[j+1] = argmax(logits[T+j])
    the last row's argmax is deliberately not in `tokens` - it is a free extra check.

Hard aborts (a golden file is worthless if any of these is fudged): strict load,
no meta tensors left, `tokens` length == --gen, every GDN layer contributed a
state, resid of the last layer finite.
"""
import os
import sys

# `tokenize.py` next to this script shadows the stdlib `tokenize` that `inspect`
# imports, and torch/transformers import `inspect`. Drop this directory from
# sys.path BEFORE importing anything third-party (python -P does the same, but
# this works however the script is invoked).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

# The one meaning of the int4 bits lives in dequant.py; load it by path rather
# than by putting _HERE back on sys.path (see the guard above).
_spec = importlib.util.spec_from_file_location("oracle_dequant", os.path.join(_HERE, "dequant.py"))
_dequant = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dequant)
dequant_gptq = _dequant.dequant_gptq

SKIP_PREFIXES = ("model.visual.", "mtp.")
QUANT_SUFFIXES = (".scales", ".qzeros", ".g_idx")


def die(msg: str, code: int = 2) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def map_name(name: str) -> str:
    """Checkpoint name -> Qwen3_5ForCausalLM name."""
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model.") :]
    return name


def build_state_dict(snapshot: str, group_size: int) -> dict[str, torch.Tensor]:
    """Dequantise the checkpoint into a plain bf16 state dict for Qwen3_5ForCausalLM."""
    handles = {}
    where: dict[str, object] = {}
    for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        h = safe_open(path, framework="pt", device="cpu")
        handles[path] = h
        for key in h.keys():
            if key in where:
                die(f"duplicate tensor name across shards: {key}")
            where[key] = h

    sd: dict[str, torch.Tensor] = {}
    n_quant = 0
    for key in sorted(where):
        if key.startswith(SKIP_PREFIXES) or key.endswith(QUANT_SUFFIXES):
            continue
        if key.endswith(".qweight"):
            base = key[: -len(".qweight")]
            scales_key = base + ".scales"
            if scales_key not in where:
                die(f"{key} has no {scales_key}")
            # dequant_gptq returns [K, N] = [in, out]; nn.Linear wants [out, in].
            w = dequant_gptq(where[key].get_tensor(key), where[scales_key].get_tensor(scales_key), group_size)
            sd[map_name(base) + ".weight"] = w.t().contiguous()
            n_quant += 1
        else:
            t = where[key].get_tensor(key)
            if t.dtype != torch.bfloat16:
                die(f"unquantised tensor {key} is {t.dtype}, expected bfloat16")
            sd[map_name(key)] = t
    print(f"state dict: {len(sd)} tensors, {n_quant} dequantised from int4, "
          f"{sum(t.numel() * t.element_size() for t in sd.values()) / 2**30:.2f} GiB")
    return sd


def check_quant_config(snapshot: str) -> int:
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        q = json.load(f).get("quantization_config") or {}
    if q.get("bits") != 4 or not q.get("sym") or q.get("desc_act"):
        die(f"dequant_gptq assumes int4 symmetric desc_act=false; config says "
            f"bits={q.get('bits')} sym={q.get('sym')} desc_act={q.get('desc_act')}")
    gs = q.get("group_size")
    if gs != 64:
        die(f"group_size {gs} != 64 (dequant.py's convention, doc 02)")
    return gs


def dump_cache_layout(cache) -> None:
    """Print enough of the cache API to fix this script when transformers moves."""
    print(f"cache type: {type(cache)}", file=sys.stderr)
    print("cache attrs: " + ", ".join(a for a in dir(cache) if not a.startswith("__")), file=sys.stderr)
    layers = getattr(cache, "layers", None)
    if layers is None:
        print("cache has no .layers", file=sys.stderr)
        return
    for i in (0, 1):
        if i < len(layers):
            attrs = ", ".join(a for a in dir(layers[i]) if not a.startswith("__"))
            print(f"cache.layers[{i}] type: {type(layers[i])}\n  attrs: {attrs}", file=sys.stderr)


def read_state(layer, attr: str, name: str):
    """cache.layers[i].{conv,recurrent}_states is dict[int, Tensor] in transformers 5.15."""
    states = getattr(layer, attr, None)
    if states is None:
        raise AttributeError(f"{name}: cache layer has no `{attr}`")
    try:
        t = states[0]
    except (KeyError, IndexError, TypeError) as exc:
        raise AttributeError(f"{name}: cannot index `{attr}` ({type(states)}): {exc}") from exc
    if t is None:
        raise AttributeError(f"{name}: `{attr}`[0] is None")
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--prompt", required=True, help="file of whitespace-separated token ids (tokenize.py encode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--max-prompt", type=int, default=64)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(0)  # nothing samples, but pin it anyway
    group_size = check_quant_config(args.snapshot)

    with open(args.prompt, encoding="utf-8") as f:
        ids = [int(x) for x in f.read().split()]
    if not ids:
        die(f"{args.prompt} is empty")
    if len(ids) > args.max_prompt:
        die(f"prompt is {len(ids)} ids, --max-prompt is {args.max_prompt}")
    print(f"prompt: {len(ids)} ids from {args.prompt}")

    # 1. the model, from the config alone, on meta so we never hold two copies
    cfg = AutoConfig.from_pretrained(args.snapshot)
    tc = cfg.get_text_config()
    if tc.tie_word_embeddings:
        die("tie_word_embeddings is true; this checkpoint ships a separate lm_head (doc 03)")
    tc._attn_implementation = "eager"  # fp32 softmax, the contract in doc 03
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(tc)
    n_layers = tc.num_hidden_layers
    layer_types = list(tc.layer_types)
    gdn_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    print(f"config: {n_layers} layers, {len(gdn_layers)} linear_attention, "
          f"{n_layers - len(gdn_layers)} full_attention, attn_impl={tc._attn_implementation}, "
          f"hidden {tc.hidden_size}, vocab {tc.vocab_size}")

    # 2. the weights
    sd = build_state_dict(args.snapshot, group_size)
    want = set(model.state_dict().keys())
    missing, unexpected = sorted(want - set(sd)), sorted(set(sd) - want)
    if missing or unexpected:
        print(f"FATAL: strict load would fail: {len(missing)} missing, {len(unexpected)} unexpected",
              file=sys.stderr)
        for k in missing:
            print(f"  missing:    {k}", file=sys.stderr)
        for k in unexpected:
            print(f"  unexpected: {k}", file=sys.stderr)
        sys.exit(2)
    model.load_state_dict(sd, strict=True, assign=True)  # assign: no second copy
    del sd
    # meta construction left non-persistent buffers (rope inv_freq) empty; rebuild on CPU.
    model.model.rotary_emb = type(model.model.rotary_emb)(tc)
    still_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if still_meta:
        die("meta tensors survived the load: " + ", ".join(still_meta[:20]))
    model.eval()
    print(f"loaded strict, {time.time() - t0:.1f}s elapsed, dtype {model.dtype}")

    # 3. prompt forward with per-layer hooks
    caps: dict[str, torch.Tensor] = {}

    def hook(name: str):
        def fn(_mod, _args, out):
            t = out[0] if isinstance(out, tuple) else out
            caps[name] = t.detach()[0].to(torch.bfloat16).clone()
        return fn

    handles = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "linear_attn"):
            mixer = layer.linear_attn
        elif hasattr(layer, "self_attn"):
            mixer = layer.self_attn
        else:
            die(f"layer {i} has neither linear_attn nor self_attn: {type(layer)}")
        handles += [layer.register_forward_hook(hook(f"resid.L{i}")),
                    mixer.register_forward_hook(hook(f"mixer.L{i}")),
                    layer.mlp.register_forward_hook(hook(f"mlp.L{i}"))]

    input_ids = torch.tensor([ids], dtype=torch.long)
    t_fwd = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    for h in handles:
        h.remove()
    print(f"prompt forward: {time.time() - t_fwd:.1f}s")

    # 4. GDN cache state, read straight after the prompt (decode mutates it in place)
    cache = out.past_key_values
    states: dict[str, torch.Tensor] = {}
    try:
        layers = cache.layers
        for i in gdn_layers:
            rec = read_state(layers[i], "recurrent_states", f"gdn_state.L{i}")
            conv = read_state(layers[i], "conv_states", f"conv_state.L{i}")
            states[f"gdn_state.L{i}"] = rec.detach()[0].to(torch.float32).clone()
            states[f"conv_state.L{i}"] = conv.detach()[0].clone()
    except (AttributeError, TypeError, IndexError) as exc:
        print(f"FATAL: cache layout is not what transformers 5.15 had: {exc}", file=sys.stderr)
        dump_cache_layout(cache)
        sys.exit(3)

    # 5. greedy decode through the same cache
    rows = [out.logits[0].to(torch.float32)]
    cur = rows[-1][-1]
    tokens: list[int] = []
    t_gen = time.time()
    for step in range(args.gen):
        nxt = int(torch.argmax(cur).item())
        tokens.append(nxt)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[nxt]], dtype=torch.long),
                        past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        rows.append(out.logits[0].to(torch.float32))
        cur = rows[-1][-1]
    print(f"greedy {args.gen} tokens: {time.time() - t_gen:.1f}s -> {tokens}")

    tensors = dict(caps)
    tensors.update(states)
    tensors["logits"] = torch.cat(rows, dim=0).contiguous()
    tensors["tokens"] = torch.tensor(tokens, dtype=torch.int32)

    # self-checks
    if len(caps) != 3 * n_layers:
        die(f"hooks captured {len(caps)} activations, expected {3 * n_layers} (resid/mixer/mlp per layer)")
    if len(tokens) != args.gen:
        die(f"generated {len(tokens)} tokens, asked for {args.gen}")
    got_gdn = sorted(int(k.split(".L")[1]) for k in states if k.startswith("gdn_state."))
    if got_gdn != gdn_layers:
        die(f"gdn_state missing for layers {sorted(set(gdn_layers) - set(got_gdn))}")
    last = tensors[f"resid.L{n_layers - 1}"]
    if not torch.isfinite(last.to(torch.float32)).all():
        die(f"resid.L{n_layers - 1} has NaN/Inf - the dequantised weights are wrong")
    if len(set(tokens)) == 1:
        print(f"WARNING: degenerate continuation, every token is {tokens[0]}", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(tensors, args.out, metadata={
        "snapshot": os.path.abspath(args.snapshot),
        "prompt_ids": " ".join(str(i) for i in ids),
        "n_prompt": str(len(ids)),
        "gen": str(args.gen),
        "attn_implementation": tc._attn_implementation,
        "group_size": str(group_size),
    })

    total = 0
    for name in sorted(tensors, key=lambda n: (n.split(".L")[0], int(n.split(".L")[1]) if ".L" in n else 0)):
        t = tensors[name]
        total += t.numel() * t.element_size()
        print(f"  {name:<18} {str(t.dtype).replace('torch.', ''):<8} {tuple(t.shape)}")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # KiB on Linux
    print(f"wrote {args.out}: {len(tensors)} tensors, {total / 2**20:.1f} MiB")
    print(f"wall {time.time() - t0:.1f}s, peak RSS {rss:.1f} GiB")


if __name__ == "__main__":
    main()
