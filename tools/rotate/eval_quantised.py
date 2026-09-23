#!/usr/bin/env python3
"""Evaluate an int4 GPTQ checkpoint of the ROTATED model (tools/rotate), any group size.

    eval_quantised.py logits   <bf16 snapshot> <quantised snapshot> --prompt <ids> [--layers 0]
    eval_quantised.py actquant <quantised snapshot> --prompt <ids> [--tokens 512] [--layers 0]
    eval_quantised.py rtn      <bf16 snapshot> <bf16 snapshot to quantise> --prompt <ids> --group <-1|64|128>
    eval_quantised.py sim      <bf16 snapshot> <quantised snapshot> --prompt <ids> [--variants none,a8,a8w8,h8]

logits    the quantised model against the ORIGINAL bf16 model, the metrics of
          check_rotation.py. The yardstick on the 42-id prose prompt is the
          unrotated int4 g64 gate checkpoint: rel L2 7.77 %, worst cos 0.992587,
          argmax 38/42, top-5 0.905 (commit 046f289).
actquant  one CPU forward of the quantised model with a hook on every linear of
          every decoder layer. For each, against y = x W^T with the model's own
          weights, it measures int8 activations, per-token symmetric:
            A8           Q8(x) W^T. The whole error of the W4A8 path for a
                         residual-stream reader, whose x arrives rotated.
            H+A8+W8      Q8(x R) Q8pc(W R)^T with R the 1024-block Hadamard
                         over K: the in-engine rotation, for the linears whose
                         input the checkpoint cannot rotate (down_proj,
                         o_proj, out_proj).
          Bar: worst-row cosine 0.999 (tests/golden/golden_common.h).
rtn       a control with no tuning: round-to-nearest symmetric int4 (the GPTQ
          grid, q in [-8, 7], scale = max|w| / 7.5) at --group, applied in memory
          to every decoder linear AutoRound quantises (all but in_proj_a/b) of
          the second snapshot, logits against the first. Separates what the
          group size costs from what the rotation or the tuning does.
sim       the end-to-end cost of the int8 paths, simulated: every decoder
          linear AutoRound quantised (all but in_proj_a/b) gets its forward
          replaced, fp32 inside, output back in the model dtype:
            none   x W^T                      the fp32-matmul control
            a8     Q8(x) W^T                  per-token int8 activations
            a8w8   Q8(x) Q8pc(W)^T            + per-channel int8 weights (the
                                              unrotated fast path, 1.70x / 1.85x)
            h8     Q8(xR) Q8pc(WR)^T          runtime 1024-block Hadamard on
                                              both sides (1.37x / 1.52x)
          The model is loaded once; each variant is one forward, logits
          against one bf16 reference.

Per-channel checkpoints (group_size -1) store scales [1, N]; the group is read
from each tensor's scale shape, so g64 checkpoints work too. The dequant is the
oracle's own (tools/oracle/dequant.py). Run in the reference container.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORACLE = os.path.join(os.path.dirname(_HERE), "oracle")
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) not in (_HERE, _ORACLE)]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("b70_rotation", os.path.join(_HERE, "rotation.py"))
CR = _load("b70_check_rotation", os.path.join(_HERE, "check_rotation.py"))
DQ = _load("b70_dequant", os.path.join(_ORACLE, "dequant.py"))

STREAM_READERS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "q_proj", "k_proj",
                  "v_proj", "gate_proj", "up_proj")
OTHER_INPUTS = ("down_proj", "o_proj", "out_proj")
BAR = 0.999


def load_quantised(snapshot: str, layers: int) -> dict[str, torch.Tensor]:
    """Checkpoint-named bf16 state dict, int4 tensors dequantised at their own group size."""
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    q = cfg.get("quantization_config") or {}
    if q.get("bits") != 4 or not q.get("sym") or q.get("desc_act"):
        sys.exit(f"FATAL: need int4 symmetric without desc_act, config says {q}")
    with open(os.path.join(snapshot, "model.safetensors.index.json"), encoding="utf-8") as f:
        wmap = json.load(f)["weight_map"]
    if any(k.endswith(".g_idx") for k in wmap) and "desc_act" not in q:
        sys.exit("FATAL: g_idx present without an explicit desc_act: false")
    handles, sd, groups = {}, {}, {}

    def get(k):
        p = os.path.join(snapshot, wmap[k])
        if p not in handles:
            handles[p] = safe_open(p, framework="pt", device="cpu")
        return handles[p].get_tensor(k)

    for k in sorted(wmap):
        if k.startswith(("model.visual.", "mtp.")) or k.endswith((".scales", ".qzeros", ".g_idx")):
            continue
        if layers and k.startswith(f"{R.PREFIX}layers."):
            if int(k[len(f"{R.PREFIX}layers."):].split(".")[0]) >= layers:
                continue
        if k.endswith(".qweight"):
            base = k[: -len(".qweight")]
            qw, sc = get(k), get(base + ".scales")
            kdim = qw.shape[0] * 8
            gs = kdim // sc.shape[0]
            if gs * sc.shape[0] != kdim:
                sys.exit(f"FATAL: {base}: K {kdim} not a multiple of {sc.shape[0]} scale rows")
            if base + ".g_idx" in wmap:   # GPTQ ships a trivial one with desc_act false: prove it
                gi = get(base + ".g_idx")
                if not bool((gi == torch.arange(kdim, dtype=gi.dtype) // gs).all()):
                    sys.exit(f"FATAL: {base}: g_idx is not k // {gs}; the checkpoint is permuted")
            zeros = get(base + ".qzeros")
            if not bool((zeros == 0x77777777).all()):
                sys.exit(f"FATAL: {base}: qzeros are not the symmetric v1 0x77777777")
            groups[gs] = groups.get(gs, 0) + 1
            sd[base + ".weight"] = DQ.dequant_gptq(qw, sc, gs, 8192 if qw.shape[1] > 65536 else 0).t().contiguous()
        else:
            sd[k] = get(k).to(torch.bfloat16)
    print(f"dequantised: {groups} (group size: tensors); {len(sd)} tensors")
    return sd


def q8_token(x: torch.Tensor) -> torch.Tensor:
    s = x.abs().amax(dim=1, keepdim=True) / 127.0
    s[s == 0] = 1.0
    return torch.clamp(torch.round(x / s), -127, 127) * s


def q8_channel(w: torch.Tensor) -> torch.Tensor:
    """w [N, K]: one scale per output row."""
    s = w.abs().amax(dim=1, keepdim=True) / 127.0
    s[s == 0] = 1.0
    return torch.clamp(torch.round(w / s), -128, 127) * s


def rtn_int4(w: torch.Tensor, group: int) -> torch.Tensor:
    """w [N, K] -> symmetric int4 RTN along K in groups of `group` (-1 = all of K)."""
    n, k = w.shape
    g = k if group <= 0 else group
    wf = w.to(torch.float32).reshape(n, k // g, g)
    s = wf.abs().amax(dim=2, keepdim=True) / 7.5
    s[s == 0] = 1.0
    return (torch.clamp(torch.round(wf / s), -8, 7) * s).reshape(n, k).to(w.dtype)


def metrics(y: torch.Tensor, ref: torch.Tensor):
    rel = ((y - ref).norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(y.double(), ref.double(), dim=1).min().item()
    return rel, cos


def actquant(args) -> None:
    with open(args.prompt, encoding="utf-8") as f:
        ids = [int(x) for x in f.read().split()][: args.tokens]
    tc = CR.text_config(args.snapshot, args.layers)
    with open(os.path.join(args.snapshot, "config.json"), encoding="utf-8") as f:
        rotated = "b70_rotation" in json.load(f)
    print(f"checkpoint stream: {'ROTATED (b70_rotation present)' if rotated else 'NOT rotated'}")
    sd = CR.to_model_names(load_quantised(args.snapshot, args.layers))
    with torch.device("meta"):
        model = CR.Qwen3_5ForCausalLM(tc)
    model.load_state_dict(sd, strict=True, assign=True)
    model.model.rotary_emb = type(model.model.rotary_emb)(tc)
    model.eval()
    del sd
    rows = []
    signs = {}

    def hook(name, lin):
        kind = name.rsplit(".", 1)[-1]

        def fn(_mod, inp):
            x = inp[0][0].to(torch.float32)                    # [T, K]
            w = lin.weight.to(torch.float32)                   # [N, K]
            ref = x @ w.t()
            r = {"name": name, "kind": kind, "K": x.shape[1]}
            r["A8"] = metrics(q8_token(x) @ w.t(), ref)
            if kind in OTHER_INPUTS and x.shape[1] % R.BLOCK == 0:
                d = signs.setdefault(x.shape[1], R.signs(x.shape[1], R.SEED + x.shape[1]))
                xr, wr = R.rot(x, d), R.rot(w, d)
                r["H+A8+W8"] = metrics(q8_token(xr) @ q8_channel(wr).t(), ref)
            rows.append(r)
        return fn

    for i, layer in enumerate(model.model.layers):
        for n, m in layer.named_modules():
            if isinstance(m, torch.nn.Linear):
                m.register_forward_pre_hook(hook(f"L{i}.{n}", m))
    t = time.time()
    with torch.no_grad():
        model(input_ids=torch.tensor([ids]))
    print(f"forward + {len(rows)} hooked linears: {len(ids)} tokens, {time.time() - t:.0f}s\n")

    print("| linear | input | layers | A8 worst cos | A8 median rel L2 | A8 fails | "
          "H+A8+W8 worst cos | H+A8+W8 median rel L2 | H+A8+W8 fails |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for kind in STREAM_READERS + OTHER_INPUTS:
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue

        def col(key):
            v = [r[key] for r in rs if key in r]
            if not v:
                return "-", "-", "-"
            rels = sorted(x[0] for x in v)
            return (f"{min(x[1] for x in v):.6f}", f"{rels[len(rels) // 2] * 100:.3f} %",
                    f"{sum(1 for x in v if x[1] < BAR)}")
        a, h = col("A8"), col("H+A8+W8")
        src = ("rotated stream" if rotated else "stream, NOT rotated") if kind in STREAM_READERS \
            else "not rotatable offline"
        print(f"| {kind} | {src} | {len(rs)} | {a[0]} | {a[1]} | {a[2]} | {h[0]} | {h[1]} | {h[2]} |")
    worst = sorted(rows, key=lambda r: r.get("H+A8+W8", r["A8"])[1])[:8]
    print("\nworst 8 (by the column the engine would use):")
    for r in worst:
        best = r.get("H+A8+W8", r["A8"])
        print(f"  {r['name']}: cos {best[1]:.6f}, rel L2 {best[0] * 100:.3f} %"
              f"{'  (A8 ' + format(r['A8'][1], '.6f') + ')' if 'H+A8+W8' in r else ''}")


def sim(args) -> None:
    import torch.nn.functional as F
    with open(args.prompt, encoding="utf-8") as f:
        ids = [int(x) for x in f.read().split()]
    tc = CR.text_config(args.snapshot, args.layers)
    base = CR.run(tc, CR.load_sd(args.snapshot, args.layers, torch.bfloat16), ids)
    sd = CR.to_model_names(load_quantised(args.quantised, args.layers))
    with torch.device("meta"):
        model = CR.Qwen3_5ForCausalLM(tc)
    model.load_state_dict(sd, strict=True, assign=True)
    model.model.rotary_emb = type(model.model.rotary_emb)(tc)
    model.eval()
    del sd
    state = {"v": "none"}
    signs = {}
    targets = []
    for i, layer in enumerate(model.model.layers):
        for n, m in layer.named_modules():
            if isinstance(m, torch.nn.Linear) and not n.endswith(("in_proj_a", "in_proj_b")):
                targets.append(m)

    def make_forward(m):
        def fwd(x):
            shp = x.shape
            x32 = x.reshape(-1, shp[-1]).to(torch.float32)
            w32 = m.weight.to(torch.float32)
            v = state["v"]
            if v == "none":
                y = x32 @ w32.t()
            elif v == "a8":
                y = q8_token(x32) @ w32.t()
            elif v == "a8w8":
                y = q8_token(x32) @ q8_channel(w32).t()
            elif v == "h8":
                k = shp[-1]
                d = signs.setdefault(k, R.signs(k, R.SEED + k))
                y = q8_token(R.rot(x32, d)) @ q8_channel(R.rot(w32, d)).t()
            else:
                raise ValueError(v)
            if m.bias is not None:
                y = y + m.bias.to(torch.float32)
            return y.to(x.dtype).reshape(*shp[:-1], -1)
        return fwd

    for m in targets:
        m.forward = make_forward(m)
    print(f"sim: {len(targets)} linears replaced; {len(ids)} ids")
    for v in args.variants.split(","):
        state["v"] = v
        t = time.time()
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids])).logits[0].to(torch.float32)
        print(f"  {v}: forward {time.time() - t:.0f}s", flush=True)
        CR.compare(logits, base, f"sim {v}: {os.path.basename(args.quantised.rstrip('/'))} vs original bf16")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("logits", "actquant", "rtn", "sim"))
    ap.add_argument("snapshot")
    ap.add_argument("quantised", nargs="?")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--group", type=int, default=-1)
    ap.add_argument("--variants", default="none,a8,a8w8,h8")
    args = ap.parse_args()
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}")
    if args.mode == "actquant":
        actquant(args)
        return
    if not args.quantised:
        sys.exit(f"{args.mode} mode needs a second snapshot")
    if args.mode == "sim":
        sim(args)
        return
    if args.mode == "rtn":
        with open(args.prompt, encoding="utf-8") as f:
            ids = [int(x) for x in f.read().split()]
        tc = CR.text_config(args.snapshot, args.layers)
        base = CR.run(tc, CR.load_sd(args.snapshot, args.layers, torch.bfloat16), ids)
        sd = CR.load_sd(args.quantised, args.layers, torch.bfloat16)
        nq = 0
        for k in list(sd):
            if (k.startswith(f"{R.PREFIX}layers.") and k.endswith("_proj.weight") or
                    k.endswith(("in_proj_qkv.weight", "in_proj_z.weight"))) and sd[k].dim() == 2:
                if "in_proj_a" in k or "in_proj_b" in k:
                    continue
                sd[k] = rtn_int4(sd[k], args.group)
                nq += 1
        print(f"RTN int4 group {args.group}: {nq} linears")
        CR.compare(CR.run(tc, sd, ids), base,
                   f"RTN int4 g{args.group} of {os.path.basename(args.quantised.rstrip('/'))} vs original bf16")
        return
    with open(args.prompt, encoding="utf-8") as f:
        ids = [int(x) for x in f.read().split()]
    tc = CR.text_config(args.snapshot, args.layers)
    base = CR.run(tc, CR.load_sd(args.snapshot, args.layers, torch.bfloat16), ids)
    quant = CR.run(tc, load_quantised(args.quantised, args.layers), ids)
    CR.compare(quant, base, f"quantised {os.path.basename(args.quantised.rstrip('/'))} vs original bf16, "
                            f"{tc.num_hidden_layers} layers")


if __name__ == "__main__":
    main()
