#!/usr/bin/env python3
"""Logit check for the residual rotation (tools/rotate/rotation.py), on CPU.

    check_rotation.py algebra    <bf16 snapshot> --prompt <ids> [--layers 4]
    check_rotation.py checkpoint <bf16 snapshot> <rotated snapshot> --prompt <ids>
                                 [--layers 0] [--dtype bf16]
    check_rotation.py reference  <bf16 snapshot> <golden.safetensors> --prompt <ids>

algebra     the ORIGINAL weights in fp32, rotated in memory in fp32 with the
            same transform() the writer uses, never rounded: the two models
            must agree to fp32 rounding. This is the proof that the folding and
            the rotation are right. A truncated stack (--layers 4 = three GDN
            layers and one full-attention layer) keeps it to ~40 GB.
checkpoint  the written rotated checkpoint against the original, as stored
            (bf16 weights, one rounding after rotation), same compute dtype on
            both sides. --layers 0 = the full model; each model is loaded,
            run and freed in turn (~55 GB peak in bf16).
reference   the yardstick: the ORIGINAL bf16 model against another model's
            prompt logits from a tools/oracle golden file (e.g. the int4
            AutoRound checkpoint's), same metrics. The prompt ids must match
            the golden file's metadata.

Both run transformers' Qwen3_5ForCausalLM (the oracle's model, tools/oracle/
dump.py) with eager attention, and print: relative L2 of the logits, worst
per-position cosine, argmax agreement, mean top-5 overlap and the largest
|log-softmax| difference.

Run inside the reference container (tools/oracle/run_in_container.sh).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import gc  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

_spec = importlib.util.spec_from_file_location("b70_rotation", os.path.join(_HERE, "rotation.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def text_config(snapshot: str, layers: int):
    tc = AutoConfig.from_pretrained(snapshot).get_text_config()
    tc._attn_implementation = "eager"
    if layers:
        tc.num_hidden_layers = layers
        tc.layer_types = list(tc.layer_types)[:layers]
    return tc


def load_sd(snapshot: str, layers: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Checkpoint-named tensors the text model needs, as `dtype`."""
    with open(os.path.join(snapshot, "model.safetensors.index.json"), encoding="utf-8") as f:
        wmap = json.load(f)["weight_map"]
    want = {}
    for k, shard in wmap.items():
        if k.startswith(("model.visual.", "mtp.")):
            continue
        if layers and k.startswith(f"{R.PREFIX}layers."):
            if int(k[len(f"{R.PREFIX}layers."):].split(".")[0]) >= layers:
                continue
        want.setdefault(shard, []).append(k)
    sd = {}
    for shard, keys in sorted(want.items()):
        with safe_open(os.path.join(snapshot, shard), framework="pt", device="cpu") as h:
            for k in keys:
                sd[k] = h.get_tensor(k).to(dtype)
    return sd


def to_model_names(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {("model." + k[len(R.PREFIX):] if k.startswith(R.PREFIX) else k): v for k, v in sd.items()}


def run(tc, sd_ckpt: dict[str, torch.Tensor], ids: list[int]) -> torch.Tensor:
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(tc)
    sd = to_model_names(sd_ckpt)
    want = set(model.state_dict().keys())
    if set(sd) != want:
        sys.exit(f"FATAL: state dict mismatch: {len(want - set(sd))} missing, "
                 f"{len(set(sd) - want)} unexpected, e.g. {sorted(want ^ set(sd))[:4]}")
    model.load_state_dict(sd, strict=True, assign=True)
    model.model.rotary_emb = type(model.model.rotary_emb)(tc)
    model.eval()
    t = time.time()
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0].to(torch.float32).clone()
    print(f"  forward: {len(ids)} ids, {time.time() - t:.1f}s, dtype {model.dtype}", flush=True)
    del model, sd
    gc.collect()
    return logits


def compare(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    """a = rotated, b = original."""
    a64, b64 = a.double(), b.double()
    rel = ((a64 - b64).norm() / b64.norm()).item()
    cos = torch.nn.functional.cosine_similarity(a64, b64, dim=1)
    top1 = (a.argmax(1) == b.argmax(1)).sum().item()
    ta, tb = a.topk(5, dim=1).indices, b.topk(5, dim=1).indices
    ov = sum(len(set(x.tolist()) & set(y.tolist())) for x, y in zip(ta, tb)) / (5 * a.shape[0])
    lp = (torch.log_softmax(a64, 1) - torch.log_softmax(b64, 1)).abs().max().item()
    print(f"\n## {label}\n")
    print("| metric | value |\n|---|---:|")
    print(f"| logits relative L2 | {rel:.3e} |")
    print(f"| worst per-position cosine | {cos.min().item():.9f} |")
    print(f"| argmax agreement | {top1} / {a.shape[0]} |")
    print(f"| mean top-5 overlap | {ov:.4f} |")
    print(f"| max abs log-softmax difference | {lp:.3e} |", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("algebra", "checkpoint", "reference"))
    ap.add_argument("snapshot")
    ap.add_argument("rotated", nargs="?")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default=None)
    args = ap.parse_args()
    with open(args.prompt, encoding="utf-8") as f:
        ids = [int(x) for x in f.read().split()]
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, prompt {len(ids)} ids")

    if args.mode == "algebra":
        layers = 4 if args.layers is None else args.layers
        dtype = torch.float32 if args.dtype in (None, "fp32") else torch.bfloat16
        tc = text_config(args.snapshot, layers)
        print(f"algebra: {tc.num_hidden_layers} layers {tc.layer_types}, {dtype}")
        sd = load_sd(args.snapshot, layers, dtype)
        base = run(tc, dict(sd), ids)
        norms = {k: sd[k].clone() for k in R.norm_keys(tc.num_hidden_layers)}
        d = R.signs()
        kinds = {}
        for k in list(sd):
            kind, t = R.transform(k, sd[k], norms, d)
            if kind != "copy":
                sd[k] = t.to(dtype)
                kinds[kind] = kinds.get(kind, 0) + 1
        print(f"  rotated in memory, never rounded: {kinds}")
        compare(run(tc, sd, ids), base, f"algebra, {layers} layers, {dtype}, rotated in memory")
        return

    if args.mode == "reference":
        from safetensors.torch import load_file
        if not args.rotated:
            sys.exit("reference mode needs <golden.safetensors>")
        with safe_open(args.rotated, framework="pt", device="cpu") as h:
            meta = h.metadata() or {}
        if [int(x) for x in meta.get("prompt_ids", "").split()] != ids:
            sys.exit("FATAL: the golden file was made from different prompt ids")
        ref = load_file(args.rotated)["logits"][: len(ids)].to(torch.float32)
        tc = text_config(args.snapshot, 0)
        print(f"reference: {meta.get('snapshot')}")
        base = run(tc, load_sd(args.snapshot, 0, torch.bfloat16), ids)
        compare(ref, base, "reference: golden-file model vs the original bf16 model")
        return

    if not args.rotated:
        sys.exit("checkpoint mode needs <rotated snapshot>")
    layers = 0 if args.layers is None else args.layers
    dtype = torch.bfloat16 if args.dtype in (None, "bf16") else torch.float32
    tc = text_config(args.snapshot, layers)
    rc = AutoConfig.from_pretrained(args.rotated)
    if not getattr(rc, "b70_rotation", None) and "b70_rotation" not in rc.to_dict():
        sys.exit("FATAL: the rotated snapshot has no b70_rotation record in config.json")
    print(f"checkpoint: {tc.num_hidden_layers} layers, compute {dtype}")
    base = run(tc, load_sd(args.snapshot, layers, dtype), ids)
    gc.collect()
    rot = run(tc, load_sd(args.rotated, layers, dtype), ids)
    compare(rot, base, f"checkpoint, {tc.num_hidden_layers} layers, {dtype}, rotated as written")


if __name__ == "__main__":
    main()
