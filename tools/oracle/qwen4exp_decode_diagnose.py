#!/usr/bin/env python3
"""Compare a short teacher-forced engine decode capture against controlled reference arms.

Run inside the oracle container with its pinned transformers site on PYTHONPATH:
  python tools/oracle/qwen4exp_decode_diagnose.py SNAP int8:PLE CAPTURE fp32|rounded|bf16
`rounded` changes only int4 weight products to bf16 then widens to fp32.
Writes a reusable safetensors capture and metrics CSV; no gate tolerances are changed.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import csv  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402

import numpy as np
import torch
from safetensors.torch import save_file

spec = importlib.util.spec_from_file_location("ref", os.path.join(os.path.dirname(__file__), "qwen4exp_ref.py"))
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def raw(path, dtype):
    return np.fromfile(path, dtype=dtype)


def bf16(path):
    return (raw(path, np.uint16).astype(np.uint32) << 16).view(np.float32)


def packed_weight(path, k, n):
    tile = raw(path, np.uint32).reshape(n // 16, k // 64, 136)
    words = tile[:, :, :128].reshape(n // 16, k // 64, 8, 16)
    scale = tile[:, :, 128:].copy().view(np.float16).astype(np.float32)
    q = ((words[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).astype(np.int32) - 8
    return (q.transpose(0, 3, 1, 2, 4).reshape(n, k).astype(np.float32)
            * np.repeat(scale.transpose(0, 2, 1).reshape(n, k // 64), 64, axis=1))


@torch.no_grad()
def main():
    snapshot, ple_source, out, arm = sys.argv[1:]
    if arm not in ("fp32", "rounded", "bf16"):
        raise ValueError(arm)
    m = json.load(open(out + "/manifest.json"))
    dtype = torch.bfloat16 if arm == "bf16" else torch.float32
    ref.cpu_bf16.install(True)
    original_dequant = ref._dequant
    if arm == "rounded":
        def rounded(*args, **kwargs):
            return original_dequant(*args, **kwargs).to(torch.bfloat16).float()
        ref._dequant = rounded
    tc = ref.text_config(snapshot, m["layers"], dtype=dtype)
    ple = ref.PleTable(ple_source, tc)
    model, pf, lazy = ref.build_streamed(snapshot, tc, ple)
    model.lm_head.weight = torch.nn.Parameter(ref._load("int8_sim").int8_head_weight(model.lm_head.weight),
                                              requires_grad=False)
    # The engine's int8 head accumulates into f32, including on bf16 activations. Do not round its weights to bf16.
    head_hook = model.lm_head.register_forward_pre_hook(lambda _m, args: (args[0].float(),))
    ids = raw(out + "/prompt.u32", np.uint32).tolist()
    rec = ref.Recorder(model, tc, act_from=0, acts=False, want_scores=True)
    state = ref.StateDump(model)
    tensors, metrics, cache = {}, [], None
    hidden = {}
    hooks = [layer.register_forward_hook(lambda _m, _a, value, l=l: hidden.__setitem__(l, value.detach()[0].float().clone()))
             for l, layer in enumerate(model.model.layers)]

    def compare(row, layer, kind, engine, expected):
        e = np.asarray(engine, dtype=np.float64).reshape(-1)
        r = np.asarray(expected, dtype=np.float64).reshape(-1)
        if e.shape != r.shape:
            raise ValueError((row, layer, kind, e.shape, r.shape))
        if not e.size:
            return
        norms = np.linalg.norm(e) * np.linalg.norm(r)
        distance = 1 - e.dot(r) / norms if norms else float(not np.array_equal(e, r))
        rms = np.sqrt(np.mean((e-r)**2))
        relative = np.linalg.norm(e-r) / max(np.linalg.norm(r), 1e-30)
        metrics.append([arm, row, layer, kind, distance, rms, relative, np.max(np.abs(e-r)), np.mean(e == r)])
        print(f"{arm} row={row} L{layer} {kind}: distance={distance:.9g} relL2={relative:.9g} max={np.max(np.abs(e-r)):.9g}", flush=True)

    # Device packed values against independent checkpoint extraction, not merely two Python dequantizers.
    ck = model.q4_ckpt
    pre = ref.lm_prefix(ck.keys()) + "layers.0.mlp.experts.0."
    gu = packed_weight(out + "/L0.expert0.gate_up.u32", m["model_hidden"], 2*m["moe_inter"])
    dn = packed_weight(out + "/L0.expert0.down.u32", m["moe_inter"], m["model_hidden"])
    for name, engine in (("gate_proj", gu.reshape(-1, 32, m["model_hidden"])[:, :16].reshape(-1, m["model_hidden"])),
                         ("up_proj", gu.reshape(-1, 32, m["model_hidden"])[:, 16:].reshape(-1, m["model_hidden"])),
                         ("down_proj", dn)):
        weight = original_dequant(ck, pre + name, set(), torch.float32).numpy()
        compare(-1, 0, "weight." + name, engine, weight)
        if not np.array_equal(engine, weight):
            raise AssertionError("device int4 products differ from reference: " + name)
    for row, token in enumerate(ids):
        base = out + f"/r{row}"
        rec.base[0] = row
        result = model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        for l, layer in enumerate(cache.layers):
            if m["qsa"][l] and not hasattr(layer, "_q4_blocks"):
                layer._q4_blocks = torch.empty(0, m["idx"], dtype=dtype)
        # StateDump normally removes hooks after its one snapshot. Keep them for this bounded multi-snapshot probe.
        saved_hooks, state.hooks = state.hooks, []
        st = state.tensors(cache, tc, ids[:row+1])
        state.hooks = saved_hooks
        tensors.update({f"r{row}."+k: v for k, v in st.items()})
        for l, h in hidden.items():
            tensors[f"r{row}.H.L{l}"] = h
            compare(row, l, "H", bf16(base + ".H.bf16").reshape(m["layers"], -1)[l], h.numpy())
        logits = result.logits[0, -1].float()
        tensors[f"r{row}.logits"] = logits
        compare(row, m["layers"], "logits", raw(base + ".logits.f32", np.float32)[:248077], logits[:248077].numpy())
        actual_ids = raw(base + ".ple_ids.u64", np.uint64)
        wanted_ids = rec.rows[0]["ple.ids"][-1].numpy().reshape(-1)
        if not np.array_equal(actual_ids, wanted_ids):
            raise AssertionError(f"PLE ids differ at row {row}")
        expected_rows = ple.rows(torch.from_numpy(actual_ids.astype(np.int64)).view(1,-1)).float().numpy()
        compare(row, m["ple_layer"], "ple.lookup", bf16(base + ".ple.bf16"), expected_rows)
        if not np.array_equal(bf16(base + ".ple.bf16"), bf16(base + ".ple.pinned.bf16")):
            raise AssertionError("pinned engine PLE bytes differ")
        q = raw(base + ".ple.q.i8", np.int8).reshape(ple.heads, ple.dim)
        scales = raw(base + ".ple.s.f32", np.float32)
        if not np.array_equal(q.reshape(-1), raw(base + ".ple.pinned.q.i8", np.int8)) or not np.array_equal(scales, raw(base + ".ple.pinned.s.f32", np.float32)):
            raise AssertionError("pinned PLE raw bytes/scales differ")
        for h in range(ple.heads):
            local = int(actual_ids[h] - ple.offsets[h])
            if not np.array_equal(q[h], ple.q[h][local]):
                raise AssertionError("PLE raw int8 bytes differ")
            arr, dt = ple.s[h]
            s = torch.from_numpy(np.array(arr[local:local+1]))
            s = s.view(torch.bfloat16).float() if dt == "BF16" else s.float()
            if scales[h] != s.item():
                raise AssertionError("PLE raw scale differs")
        route = raw(base + ".routes.u32", np.uint32).reshape(m["layers"], -1)
        for l in range(m["layers"]):
            want = rec.rows[0][f"route.ids.L{l}"][-1].numpy().reshape(-1)
            # route ids occupy words 0..9 (route.h kIds).
            compare(row, l, "route_missing", [len(set(want)-set(route[l,:10]))], [0])
            if set(want) != set(route[l, :10]):
                ref_gap = rec.rows[0][f"route.gap.L{l}"][-1].reshape(-1)[0].item()
                engine_cut = route[l, 27:29].view(np.float32)
                print(f"ROUTE_FLIP arm={arm} row={row} L{l} reference_gap={ref_gap:.9g} "
                      f"engine_gap={engine_cut[0]-engine_cut[1]:.9g} "
                      f"reference_ids={want.tolist()} engine_ids={route[l,:10].tolist()}", flush=True)
            data = open(base + f".L{l}.state", "rb").read()
            off = 0

            def read(kind, count, dt=np.uint16, advance=None):
                nonlocal off
                a = np.frombuffer(data, dtype=dt, count=count, offset=off)
                if dt == np.uint16:
                    a = (a.astype(np.uint32)<<16).view(np.float32)
                off += count*np.dtype(dt).itemsize if advance is None else advance
                compare(row, l, kind, a, st[f"state.L{l}.{kind}"].float().numpy())
            if m["qsa"][l]:
                read("k", (row+1)*m["kv"], advance=m["max_len"]*m["kv"]*2)
                read("v", (row+1)*m["kv"], advance=m["max_len"]*m["kv"]*2)
                read("ckeys", ((row+1)//4)*m["idx"], advance=(m["max_len"]//4)*m["idx"]*2)
                read("tail", 8*m["idx"])
            else:
                read("gdn", m["gdn_elems"], np.float32)
                # Only the latest three slots remain live in the engine's 16-slot convolution ring.
                conv = np.frombuffer(data, dtype=np.uint16, count=16*m["conv_rows"], offset=off).copy().reshape(16,-1)
                live = {(row-j)%16 for j in range(min(row+1,3))}
                for slot in range(16):
                    if slot not in live:
                        conv[slot] = 0
                compare(row, l, "conv", (conv.astype(np.uint32)<<16).view(np.float32), st[f"state.L{l}.conv"].numpy())
                off += 16*m["conv_rows"]*2
            if l == m["ple_layer"]:
                off += 64
                read("ple", 16*m["hidden"])
    for hook in hooks + state.hooks + [head_hook]:
        hook.remove()
    tensors.update(rec.tensors())
    rec.remove()
    save_file({k:v.contiguous() for k,v in tensors.items()}, out + f"/reference-{arm}.safetensors")
    with open(out + f"/metrics-{arm}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["arm","row","layer","kind","cosine_distance","rms_error","relative_l2","max_abs","exact_fraction"])
        writer.writerows(metrics)
    print(f"DONE {arm}: PLE raw bytes/scales/ids/rows and pinned rows exact; int4 device products exact; {len(ids)} rows", flush=True)


if __name__ == "__main__":
    main()
