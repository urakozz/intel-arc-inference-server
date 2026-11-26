# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""CPU text oracle for the packed Inkling 2 Mini release.

    python -m tools.oracle.inkling_ref run SNAPSHOT --prompt IDS.txt --out NAME.golden.safetensors --dtype fp32 --gen 64

--kv bf16 stores every cache row rounded to bf16 (what the engine's caches hold); --kv int8 stores the window rings
in bf16 and the 8 full visits' K / V as the engine's `--kv-cache int8` gives them back (PERF-102: kv8.h's 128-point
rotated int8, int8_sim.kv8_fq at head_dim 128). Without --kv the caches keep the run's dtype, as before.

The checkpoint stays memory-mapped. Attention and other dense weights are cast
one layer at a time, and only routed experts are decoded. Trace layer suffixes
index visits in execution order, including each pass over the middle cycle.
"""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from .dequant import mxfp8_to_f32, nvfp4_to_f32
from .int8_sim import kv8_fq

ROOT = "model.llm."
STREAMS = ("attn", "k", "v", "mlp")


class Tensors:
    def __init__(self, snapshot):
        index = snapshot / "model.safetensors.index.json"
        if index.exists():
            self.files = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as f:
                self.files = dict.fromkeys(f.keys(), "model.safetensors")
        self.stack = ExitStack()
        self.shards = {}

        for name in sorted(set(self.files.values())):
            self.shards[name] = self.stack.enter_context(safe_open(snapshot / name, framework="pt", device="cpu"))

    def slice(self, name):
        return self.shards[self.files[name]].get_slice(name)

    def __getitem__(self, name):
        return self.shards[self.files[name]].get_tensor(name)


def rms(x, eps, weight=None):
    z = x.float()
    z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
    return (z if weight is None else weight.float() * z).to(x.dtype)


def mix(blocks, partial, queries):
    states = rms(torch.stack([*blocks, partial], dim=-2).float(), 1e-6)
    q = queries.float().reshape(-1, partial.shape[-1]).expand(*states.shape[:-2], -1, -1)
    score = (q @ states.transpose(-1, -2)) * (partial.shape[-1] ** -0.5)
    return (score.softmax(-1) @ states).unbind(-2)


def softcap(x, limit):
    return torch.tanh(x / limit) * limit


def clip(x, threshold):
    y = x.float()
    inv = threshold / (torch.linalg.vector_norm(y, dim=-1, keepdim=True) + 1e-6)
    return (y * inv.clamp(max=1)).to(x.dtype)


def controller(x, msg, w, stream, eps, dtype):
    key = f"controllers.{stream}."
    x, msg = x.float(), msg.float()
    z = torch.cat([rms(x, eps), rms(msg, eps)], dim=-1).to(dtype)
    low = F.linear(z, w[key + "down_weight"]).float()
    forget = F.linear(low, w[key + "forget_weight"].float()).tanh()
    gain = F.linear(low, w[key + "gain_weight"].float()).tanh()
    return (1 - forget) * x + gain * msg


@dataclass
class VisitCache:
    kv: tuple | None = None
    conv: dict = field(default_factory=dict)


@dataclass
class Cache:
    visits: list
    length: int = 0


@dataclass
class Trace:
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    routes: list = field(default_factory=list)
    final: torch.Tensor | None = None

    def taps(self, visits):
        return torch.stack([self.final if i == len(self.inputs) else self.inputs[i] for i in visits], dim=-2)


def convolve(x, w, key, state, *, dynamic_input=None, heads=None):
    raw = x.float().transpose(1, 2)
    n = x.shape[1]
    if dynamic_input is None:
        weight = w[key + ".weight"].float()
        width = weight.shape[-1]
    else:
        static = w[key + ".static_w"].float()
        width = static.shape[0]
    previous = state.get(key)
    context = raw if previous is None else torch.cat((previous, raw), dim=-1)
    if context.shape[-1] >= width:
        state[key] = context[..., -width:].contiguous()
    else:
        state[key] = F.pad(context, (width - context.shape[-1], 0))

    if dynamic_input is None:
        if previous is None:
            convolved = F.conv1d(raw, weight, padding=width - 1, groups=raw.shape[1])[..., :n]
        else:
            convolved = F.conv1d(context, weight, groups=raw.shape[1])[..., -n:]
        return ((convolved.transpose(1, 2) + x.float()) / math.sqrt(width + 1)).to(x.dtype)

    taps = F.linear(dynamic_input, w[key + ".weight_proj"]).float().view(1, n, heads, 1, width)
    taps = static.T.view(heads, -1, width) + taps
    context = F.pad(context, (width - 1, 0))[..., -(n + width - 1):]
    rows = context.unfold(-1, width, 1).flip(-1).transpose(1, 2).reshape(1, n, heads, -1, width)
    convolved = (rows * taps).sum(-1).reshape_as(x)
    return ((convolved + x.float()) / math.sqrt(2 * width + 1)).to(x.dtype)


def attention(inputs, w, c, positions, state, local, cache=None):
    qx, kx, vx = inputs
    h, kv, d = ((c["swa_num_attention_heads"], c["swa_num_key_value_heads"], c["swa_head_dim"])
                if local else (c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]))
    t = qx.shape[1]
    window = c["sliding_window_size"]
    eps = c["rms_norm_eps"]
    q = F.linear(qx, w["attn.wq_du.weight"]).view(1, t, h, d)
    r = F.linear(qx, w["attn.wr_du.weight"]).view(1, t, h, c["d_rel"])
    q = clip(q.float() * w["attn.q_l2_clip.scale"].float(), c["qk_l2_clip_threshold"]).to(qx.dtype)
    r = clip(r.float() * w["attn.rel_l2_clip.scale"].float(), c["rel_l2_clip_threshold"]).to(qx.dtype)
    if not local:
        tau = 1 + c["log_scaling_alpha"] * ((positions + 1).float() / c["log_scaling_n_floor"]).clamp(min=1).log()
        q = (q.float() * tau[None, :, None, None]).to(q.dtype)
        r = (r.float() * tau[None, :, None, None]).to(r.dtype)

    k = F.linear(kx, w["attn.wk_dv.weight"])
    v = F.linear(vx, w["attn.wv_dv.weight"])
    k = convolve(k, w, "attn.k_sconv", state.conv, dynamic_input=None if local else kx, heads=kv)
    v = convolve(v, w, "attn.v_sconv", state.conv, dynamic_input=None if local else vx, heads=kv)
    k = clip(k.view(1, t, kv, d), c["qk_l2_clip_threshold"]).transpose(1, 2)
    v = rms(v.view(1, t, kv, d), eps).transpose(1, 2)
    if cache in ("int8", "int8-k", "int8-v") and not local:   # int8-k / int8-v: only K / only V int8, the other bf16
        k, v = (kv8_fq(x).to(x.dtype) if cache in ("int8", "int8-" + n) else x.to(torch.bfloat16).to(x.dtype)
                for x, n in ((k, "k"), (v, "v")))
    elif cache is not None:
        k, v = (x.to(torch.bfloat16).to(x.dtype) for x in (k, v))
    if state.kv is not None:
        k, v = (torch.cat((prior, new), dim=2) for prior, new in zip(state.kv, (k, v)))
    keep = window - 1 if local else k.shape[2]
    state.kv = (k[:, :, -keep:].contiguous(), v[:, :, -keep:].contiguous()) if keep else (k[:, :, :0], v[:, :, :0])
    distance = positions[:, None] - (positions[-1] + 1 - k.shape[2] + torch.arange(k.shape[2]))
    relative = (r @ w["attn.rel_logits_proj.proj"]).transpose(1, 2)
    extent = relative.shape[-1]
    bias = relative.gather(-1, distance.clamp(0, extent - 1).expand(1, h, -1, -1))
    bias = bias.masked_fill((distance < 0) | (distance >= extent), 0)
    invalid = distance < 0
    if local:
        invalid = invalid | (distance >= window)
    k, v = (x[:, :, None].expand(1, kv, h // kv, -1, d).reshape(1, h, -1, d) for x in (k, v))
    score = (q.transpose(1, 2) @ k.transpose(2, 3)) * (d ** -0.5) + bias
    score = score.masked_fill(invalid, torch.finfo(score.dtype).min)
    sinks = F.linear(qx, w["attn.attn_sink_proj.weight"]).transpose(1, 2).unsqueeze(-1)
    weight = torch.cat((score, sinks), dim=-1).float().softmax(-1)[..., :-1].to(q.dtype)
    out = (weight @ v).transpose(1, 2).contiguous().view(1, t, h * d)
    gated = (out.float() * F.linear(qx, w["attn.gate_attn.weight"]).float().sigmoid()).to(out.dtype)
    return F.linear(gated, w["attn.wo_ud.weight"])


class Inkling:
    def __init__(self, snapshot, dtype=torch.float32, kv=None):
        self.snapshot = Path(snapshot)
        self.kv = kv
        self.config = json.loads((self.snapshot / "config.json").read_text())
        self.c = self.config["text_config"]
        loop = self.c["loop_config"]
        if loop.get("is_varloop", False) or self.c.get("n_shared_experts", 0):
            raise ValueError("Expected fixed-loop Inkling text weights with routed experts only")
        self.start, self.end, self.passes = (loop[k] for k in ("core_start", "core_end", "num_passes"))
        self.dtype = dtype
        self.weights = Tensors(self.snapshot)
        qc = json.loads((self.snapshot / "hf_quant_config.json").read_text())
        self.formats = {n: g["weight_transform"] for g in qc["model"]["parameter_groups"] for n in g["tensors"]
                        if n.endswith((".w13_weight", ".w2_weight"))}
        self.layers = defaultdict(list)
        for name in self.weights.files:
            if name.startswith(ROOT + "layers.") and ".mlp.experts." not in name:
                _, layer, relative = name.removeprefix(ROOT).split(".", 2)
                self.layers[int(layer)].append(relative)
        self.head = self.weights[ROOT + "unembed.weight"][:self.c["unpadded_vocab_size"]].to(dtype)
        self.vocab = self.head.shape[0]
        self.n_visits = self.c["num_hidden_layers"] + (self.passes - 1) * (self.end - self.start)

    def cache(self):
        return Cache([VisitCache() for _ in range(self.n_visits)])

    def expert(self, layer, part, expert):
        name = f"{ROOT}layers.{layer}.mlp.experts.{part}_weight"
        w = self.weights.slice(name)[expert]
        s = self.weights.slice(name + "_scale")[expert]
        transform = self.formats[name]
        if transform == "nvfp4_nearest":
            amax = self.weights[name + "_amax"][expert]
            denom = self.weights[name + "_outer_scale_denominator"][0]
            w = nvfp4_to_f32(w, s, amax, denom)
        elif transform == "mxfp8_nearest":
            w = mxfp8_to_f32(w, s)
        else:
            raise ValueError(f"{name}: unsupported expert transform {transform}")
        if part == "w13":
            w = torch.cat((w[::2], w[1::2]))
        return w.to(self.dtype)

    def moe(self, x, w, layer):
        flat = x.flatten(0, 1)
        logits = F.linear(flat.float(), w["mlp.gate.weight"].float())
        scores = logits.sigmoid()
        select = scores + w["mlp.gate.bias"].float()
        top = select.topk(self.c["num_experts_per_tok"], sorted=False).indices
        coeff = scores.gather(1, top)
        coeff = (coeff / (coeff.sum(-1, keepdim=True) + 1e-20)) * self.c["route_scale"]
        result = torch.zeros_like(flat, dtype=torch.float32)
        for expert in top.unique(sorted=True).tolist():
            slots, rows = torch.where(top.T == expert)
            g, u = F.linear(flat[rows], self.expert(layer, "w13", expert)).chunk(2, dim=-1)
            g, u = g.float(), u.float()
            h = softcap(g, self.c["mlp_gate_softcap"]) * g.sigmoid() * softcap(u, self.c["mlp_up_softcap"])
            out = F.linear(h.to(x.dtype), self.expert(layer, "w2", expert))
            result.index_add_(0, rows, out.float() * coeff[rows, slots, None])
        ordered = select.topk(self.c["num_experts_per_tok"] + 1).values
        return result.view_as(x).to(x.dtype), (top[None], coeff[None], (ordered[:, -2] - ordered[:, -1])[None])

    def layer(self, x, blocks, i, state, pos, messages, visit=None):
        c, w = self.c, {}
        for key in self.layers[i]:
            value = self.weights[f"{ROOT}layers.{i}.{key}"]
            w[key] = value if value.dtype == torch.float32 else value.to(self.dtype)
        local = i in c["local_layer_ids"]
        eps = c["rms_norm_eps"]
        z = mix(blocks, x, self.weights[f"{ROOT}attn_res.attn_queries.{i - 1}"]) if i else (x,)
        if len(z) == 1:
            z *= 3
        keys = ("attn_norm.weight",) * 3 if local else ("q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight")
        if messages is not None:
            z = tuple(controller(a, m, w, s, eps, self.dtype) for a, m, s in zip(z, messages, STREAMS))
        inp = tuple(rms(a, eps, w[k]).to(self.dtype) for a, k in zip(z, keys))
        kv = self.kv   # "<form>~V1+V2": the form, with visits V1, V2 (walk order) kept bf16
        if kv and "~" in kv:
            kv, keep = kv.split("~")
            kv = "bf16" if str(visit) in keep.split("+") else kv
        y = attention(inp, w, c, pos, state, local, kv)
        y = convolve(y, w, "attn_sconv", state.conv).float()
        if i % c["n_layers_per_attn_res_block"] == 0:
            blocks.append(x)
        else:
            y = x.float() + y
        z, = mix(blocks, y, self.weights[f"{ROOT}attn_res.mlp_queries.{i}"])
        if messages is not None:
            z = controller(z, messages[3], w, "mlp", eps, self.dtype)
        z, route = self.moe(rms(z, eps, w["mlp_norm.weight"]).to(self.dtype), w, i)
        return (y + convolve(z, w, "mlp_sconv", state.conv)).to(self.dtype), route

    @torch.inference_mode()
    def forward(self, ids, cache=None):
        if ids.ndim != 2 or ids.shape[0] != 1 or not ids.shape[1]:
            raise ValueError("Expected one nonempty row of token IDs")
        cache = self.cache() if cache is None else cache
        pos = cache.length + torch.arange(ids.shape[1])
        embed = self.weights[ROOT + "embed.weight"]
        x = rms(F.embedding(ids, embed).to(self.dtype), self.c["rms_norm_eps"],
                self.weights[ROOT + "embed_norm.weight"])
        blocks, trace = [], Trace()

        def walk(indices, upstream=None):
            nonlocal x
            for i in indices:
                v = len(trace.inputs)
                trace.inputs.append(x)
                msg = None
                if upstream is not None:
                    a = (i - self.start) * 2
                    q = self.weights[ROOT + "middle_cycle_cross_pass_queries"]
                    kv = self.weights[ROOT + "middle_cycle_cross_pass_kv_queries"]
                    queries = torch.stack((q[a], kv[a], kv[a + 1], q[a + 1]))
                    msg = mix(*upstream, queries)
                x, route = self.layer(x, blocks, i, cache.visits[v], pos, msg, v)
                trace.outputs.append(x)
                trace.routes.append(route)

        walk(range(self.start))
        prelude, prelude_blocks = x, blocks.copy()
        previous = None
        for _ in range(self.passes):
            x, blocks = prelude, prelude_blocks.copy()
            walk(range(self.start, self.end), previous)
            previous = (blocks[len(prelude_blocks) + 1:], x)
        walk(range(self.end, self.c["num_hidden_layers"]))
        norm_in, = mix(blocks, x, self.weights[ROOT + "attn_res.output_query"])
        trace.final = rms(norm_in, self.c["rms_norm_eps"], self.weights[ROOT + "norm.weight"]).to(self.dtype)
        cache.length += ids.shape[1]
        return softcap(F.linear(trace.final, self.head).float(), self.c["final_logit_softcapping"]), trace


def write_golden(model, ids, out, *, gen=0, chunk=64, force=None, prefill_tail=False):
    prompt = len(ids)
    if prompt == 0 or chunk < 1 or gen < 0:
        raise ValueError("Expected a nonempty prompt, a positive chunk size, and nonnegative generation length")
    if force is not None:
        if gen or not force or any(token < 0 or token >= model.vocab for token in force):
            raise ValueError("Use a nonempty list of live token IDs with --force and omit --gen")
        gen = len(force)
    data = defaultdict(list)
    cache = model.cache()
    taps = model.config.get("mtp_config", {}).get("target_layer_ids", [])
    started = time.monotonic()

    def progress(n):
        if n > 1 or cache.length % 16 == 0:
            print(f"Tokens in cache {cache.length}, elapsed {time.monotonic() - started:.1f} s", flush=True)

    def record(tokens, prompt_tail=False):
        logits, trace = model.forward(torch.tensor([tokens]), cache)
        progress(len(tokens))
        if prefill_tail:
            data["logits"].append(logits[0, -1:].clone())
            if prompt_tail:
                for visit, hidden in enumerate(trace.outputs[:model.start]):
                    data[f"H.L{visit}"].append(hidden[0, -1:].float().clone())
                for visit in dict.fromkeys((0, model.start - 1)):
                    key, value = cache.visits[visit].kv
                    for name, tensor in (("K", key), ("V", value)):
                        data[f"prefill.{name}.L{visit}"].append(tensor[0, :, -4:, :].float().clone())
            return int(logits[0, -1].argmax())
        data["logits"].append(logits[0])
        for visit, (hidden, route) in enumerate(zip(trace.outputs, trace.routes)):
            data[f"H.L{visit}"].append(hidden[0])
            for name, value in zip(("ids", "w", "gap"), route):
                data[f"route.{name}.L{visit}"].append(value[0].to(torch.int32) if name == "ids" else value[0])
        for j, v in enumerate(taps):
            data[f"tap.{j}"].append((trace.final if v == model.n_visits else trace.inputs[v])[0])
        return int(logits[0, -1].argmax())

    # <out>.ckpt: the cache, the rows so far and the next prompt chunk, written after each prompt chunk (write, then
    # rename); a run with the same arguments continues from it, and it is deleted when the golden is saved
    ckpt = out.with_name(out.name + ".ckpt")
    ident = [model.snapshot.name, ids, chunk, gen, force, prefill_tail, str(model.dtype), model.kv]
    resume = 0
    if ckpt.exists():
        saved = torch.load(ckpt, weights_only=False)
        if saved["ident"] == ident:
            cache, data, next_id, resume = saved["cache"], saved["data"], saved["next_id"], saved["next"]
            print(f"Resuming from {ckpt.name}: {resume} of {prompt} prompt tokens done", flush=True)
    next_id = None if not resume else next_id

    for start in range(resume, prompt, chunk):
        batch = ids[start:start + chunk]
        if prefill_tail and start + len(batch) < prompt:
            model.forward(torch.tensor([batch]), cache)
            progress(len(batch))
        else:
            next_id = record(batch, prompt_tail=True)
        torch.save({"ident": ident, "cache": cache, "data": data, "next_id": next_id, "next": start + len(batch)},
                   ckpt.with_name(ckpt.name + ".tmp"))
        os.replace(ckpt.with_name(ckpt.name + ".tmp"), ckpt)
    tokens = []
    for i in range(gen):
        token = force[i] if force is not None else next_id
        tokens.append(token)
        next_id = record([token])
    values = {k: torch.cat(rows).contiguous() for k, rows in data.items()}
    values["prompt_ids"] = torch.tensor(ids, dtype=torch.int32)
    values["tokens"] = torch.tensor(tokens, dtype=torch.int32)
    if not prefill_tail:
        values["nll"] = -values["logits"][:prompt - 1].log_softmax(-1).gather(1, torch.tensor(ids[1:], dtype=torch.long)[:, None])[:, 0]
        for visit, state in enumerate(cache.visits):
            values[f"K.L{visit}"], values[f"V.L{visit}"] = (x[0].contiguous().clone() for x in state.kv)
    save_file(values, out, metadata={"snapshot": str(model.snapshot), "n_prompt": str(prompt), "gen": str(gen),
                                     "dtype": "fp32" if model.dtype == torch.float32 else "bf16",
                                     "layers": str(model.n_visits), "logits_row0": str(prompt - 1 if prefill_tail else 0),
                                     "chunk": str(chunk),
                                     "vocab_used": str(model.vocab), "kv": model.kv or "dtype"})
    ckpt.unlink(missing_ok=True)
    return tokens


def main():
    import importlib.util as _cl_util
    _cl = _cl_util.spec_from_file_location("cpu_lock", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu_lock.py"))
    _cl_mod = _cl_util.module_from_spec(_cl)
    _cl.loader.exec_module(_cl_mod)
    _cl_mod.hold()   # one CPU reference job at a time: refuse beside another
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("run", "acceptance"):
        q = sub.add_parser(command)
        q.add_argument("snapshot", type=Path)
        q.add_argument("--prompt", type=Path, required=True, help="whitespace-separated token IDs")
        q.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
        q.add_argument("--threads", type=int, default=4)
        q.add_argument("--kv", choices=("bf16", "int8"), help="the caches as the engine stores them (see above)")
        if command == "run":
            q.add_argument("--out", type=Path, required=True)
            q.add_argument("--gen", type=int, default=0)
            q.add_argument("--force", type=Path, help="teacher-force the candidate's whitespace-separated output IDs")
            q.add_argument("--chunk", type=int, default=32)
            q.add_argument("--prefill-tail", action="store_true",
                           help="save the last prompt logits, prelude outputs, and sample KV rows")
        else:
            q.add_argument("--rounds", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    ids = list(map(int, a.prompt.read_text().split()))
    model = Inkling(a.snapshot, torch.float32 if a.dtype == "fp32" else torch.bfloat16, a.kv)
    if a.command == "run":
        force = None if a.force is None else list(map(int, a.force.read_text().split()))
        tokens = write_golden(model, ids, a.out, gen=a.gen, chunk=a.chunk, force=force, prefill_tail=a.prefill_tail)
        print("teacher-forced:" if force is not None else "greedy:", tokens, flush=True)
    else:
        from .inkling_dspark_ref import DraftHead, offline_acceptance
        print(json.dumps(offline_acceptance(model, DraftHead(a.snapshot, model), ids, a.rounds), indent=2))


if __name__ == "__main__":
    main()
