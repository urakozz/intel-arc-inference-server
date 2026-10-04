#!/usr/bin/env python3
"""Our Agnes reference against the checkpoint's own `modeling_agnes.py` (spec 14, plan 14a
Review Focus 1): per-position logits on one prompt, both models layer-streamed (stream.py).

    agnes_remote_check.py <agnes snapshot> --prompt <ids file> [--n 64]

Ours: dump.py's build (Qwen3_5ForCausalLM + agnes.AgnesMLP, names mapped). Theirs:
`AgnesForCausalLM` from the snapshot's `modeling_agnes.py` (trust_remote_code), on the
checkpoint's own names. Both get the same dump.convert-dequantised bf16 tensors, eager
attention, no cache, one forward over the first --n ids. Bar: cosine >= 0.99999 and the
same argmax at every position. Needs transformers >= 5 and a writable HF_MODULES_CACHE
(the remote code is copied there). `--save-ours` / `--load-ours` keep our side's logits
so a failure on their side does not cost our forward again.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(f"oracle_{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dump, agnes, stream = _load("dump"), _load("agnes"), _load("stream")


@torch.no_grad()
def ours(snap: str, gs: int, ids: list[int]):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    with open(os.path.join(snap, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    kw, par = agnes.translate_text_config(raw)
    tc = Qwen3_5TextConfig(**kw)
    tc._attn_implementation = "eager"
    with torch.device("meta"):
        model = dump.Qwen3_5ForCausalLM(tc)
        agnes.attach_parallel_ffn(model, tc, par)
    pf = dump.stream_weights(model, snap, gs, tc)
    model.eval()
    t = time.time()
    out = model(input_ids=torch.tensor([ids]), use_cache=False)
    print(f"ours: forward {time.time() - t:.0f}s (dequant wait {pf.seconds:.0f}s)", flush=True)
    return out.logits[0].float()


def _remote_name(n: str) -> str:   # dump.map_name's Qwen names back to the checkpoint's infixes
    for ckpt, qwen in agnes.NAME_MAP:
        n = n.replace(qwen, ckpt, 1)
    return n


@torch.no_grad()
def theirs(snap: str, gs: int, ids: list[int]):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    import transformers.cache_utils as cu
    if not hasattr(cu, "LAYER_TYPE_CACHE_MAPPING"):
        # modeling_agnes.py imports a registry no released transformers has (5.15-5.18,
        # main as of 2026-10-04); it only fills it so a DynamicCache builds Agnes's
        # cache layers. This check runs without a cache (use_cache=False), so an empty
        # dict lets the module import and changes no computation.
        cu.LAYER_TYPE_CACHE_MAPPING = {}
        print("theirs: shimmed transformers.cache_utils.LAYER_TYPE_CACHE_MAPPING = {} (no cache used)")
    cfg = AutoConfig.from_pretrained(snap, trust_remote_code=True)
    tc = cfg.text_config
    tc._attn_implementation = "eager"
    cls = get_class_from_dynamic_module("modeling_agnes.AgnesForCausalLM", snap)
    with torch.device("meta"):
        model = cls(tc)
    where = dump.open_shards(snap)
    n = tc.num_hidden_layers
    per = {i: [] for i in range(n)}
    rest = []
    for key in sorted(where):
        m = re.match(r"model\.language_model\.layers\.(\d+)\.", key)
        (per[int(m.group(1))] if m else rest).append(key)
    resident = {}
    dump.convert(where, rest, gs, resident)
    resident = {_remote_name(k): v for k, v in resident.items()}

    def layer_sd(i):
        sd = {}
        dump.convert(where, per[i], gs, sd, fast=True)
        pre = f"model.layers.{i}."
        return {_remote_name(k[len(pre):]): v for k, v in sd.items()}

    # strict names up front: every layer's mapped key set against the module's
    want = set(model.model.layers[0].state_dict())
    got = set(_remote_name(dump.map_name(k)[len("model.layers.0."):]
                           .replace(".qweight", ".weight"))
              for k in per[0] if not k.endswith(dump.QUANT_SUFFIXES))
    assert want == got, (sorted(want - got)[:5], sorted(got - want)[:5])
    pf = stream.attach(model, list(model.model.layers), layer_sd, resident,
                       rebuild=[(model.model, "rotary_emb", lambda: type(model.model.rotary_emb)(config=tc))])
    meta = [k for k, t in list(model.named_parameters()) + list(model.named_buffers())
            if t.is_meta and not k.startswith("model.layers.")]
    assert not meta, meta
    lb = [k for k, _ in model.model.layers[0].named_buffers()]
    print(f"theirs: {type(model).__name__} from modeling_agnes.py, layer buffers {lb}", flush=True)
    model.eval()
    t = time.time()
    out = model(input_ids=torch.tensor([ids]), use_cache=False)
    print(f"theirs: forward {time.time() - t:.0f}s (dequant wait {pf.seconds:.0f}s)", flush=True)
    return out.logits[0].float()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--save-ours")
    ap.add_argument("--load-ours")
    a = ap.parse_args()
    ids = [int(x) for x in open(a.prompt).read().split()][: a.n]
    assert len(ids) == a.n, f"{a.prompt} has fewer than {a.n} ids"
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}; prompt {len(ids)} ids")
    gs = dump.check_quant_config(a.snapshot)
    if a.load_ours:
        lo = torch.load(a.load_ours)
        print(f"ours: logits loaded from {a.load_ours}")
    else:
        lo = ours(a.snapshot, gs, ids)
        if a.save_ours:
            torch.save(lo, a.save_ours)
    lt = theirs(a.snapshot, gs, ids)
    cos = torch.nn.functional.cosine_similarity(lo.double(), lt.double(), dim=-1)
    am = lo.argmax(-1) == lt.argmax(-1)
    d = (lo - lt).abs().max(-1).values
    for p in range(len(ids)):
        print(f"  pos {p:2d}: cos {cos[p]:.9f}  argmax {int(lo[p].argmax())} / {int(lt[p].argmax())}"
              f"  max|d| {d[p]:.4f}")
    ok = bool(cos.min() >= 0.99999 and am.all())
    print(f"cos min {cos.min():.9f} mean {cos.mean():.9f}; argmax same {int(am.sum())}/{len(ids)}; "
          f"max|d| {d.max():.4f}")
    print(f"modeling_agnes.py vs agnes.py (bar cos >= 0.99999, argmax every position): "
          f"{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
