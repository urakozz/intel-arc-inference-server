#!/usr/bin/env python3
"""Qwen3.8's standard-bf16 references of the forced contexts, top-k only (spec 28, the operator's rule of 2026-11-04:
tests/server/spec_judge.h's "bf16-unstable" rows), on the CPU.

    qwen_bf16_top.py plan --ctx DIR [--ctx DIR ...] --judge DIR [--judge DIR ...] --out DIR --plan FILE [--bar ULP]
    qwen_bf16_top.py run <snapshot> --plan FILE --out DIR [--topk K] [--batch-tokens N] [--dry-run]

`plan` runs on the host (numpy only: the judges are read where they are, which the container does not see) and
writes the selected contexts to FILE; `run` (in the container) computes them.

Which contexts: every <key> with a judge (<judge>/<key>.golden.safetensors, qwen_forced.py's) that rejects one of its
forced ids by more than --bar (8: golden::kNearMissUlp) bf16 ulp of the f32 top - only such a row can need the rule.
Keys already in --out are skipped (resumable). Contexts with the same ids share one forward.

The model is transformers' Qwen3_5ForCausalLM in bf16 (eager attention, layer-streamed: row_taps_ref.py's build), on
the int4 checkpoint dequantised as every oracle dequantises it, its bf16 GEMMs as fp32 GEMMs of the bf16 values
rounded once (cpu_bf16.py: the box CPU has no bf16 units). The logits: the final-normalised bf16 hidden times the
head in f32 - the checkpoint's bf16 head, or for the keys qwen_forced.I8HEAD_RE matches the engine's per-row int8
head (int8_sim.int8_head_weight), like the key's judge.

The tests that prefill with l0-int8 (operator 2026-11-04, decision b: every mtp_gpu_ key - mtp_gpu_test and its
128k twins run `l0-int8`): the reference's PROMPT pass applies the same int8 rounding - every int4 linear the h8 walk
takes on Qwen3.8 (runtime/prefill/step.cc's pf_linear / linear_i8_silu: GDN in_proj_qkv || in_proj_z and out_proj, FA
q || k || v and o_proj, the MLP's gate || up and down) computes its prompt rows as int8_sim.fq_x(x) . fq_w(w) (the
rotation, per-token / per-column int8 codes, de-quantised; ornith_ref.py --sim-int8-prefill's functions), rounded
once to bf16. The prompt is the test's own (tests/golden/prompts or tests/golden/toolcall <name>.ids, checked to be
the key's prefix): the rows after it are the test's decode rows and stay standard bf16. `plan` records it per key
(pfi8_prompt: the prompt length, 0 = none) and `run` prints it.

Per key, <out>/<key>.bf16top.safetensors: `ids` int32 [1 + gen][K + 2] and `logits` f32 [1 + gen][K + 2] - row j is
the decision of forced id j (the judge's row numbering); columns 0..K-1 the bf16 row's own top K (descending), K the
judge's f32 top id, K + 1 the forced id, each with its bf16 logit. Run in the oracle container under
/tmp/b70-cpu-model.lock (tools/box_validate/f32_refs.sh bf16top-qwen)."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402

VOCAB_USED = 248077


def _load(name):
    mod_name = "qwen_bf16_top_" + name
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def ulp(x: float) -> float:
    """golden_common.h bf16_ulp_at."""
    return 2.0 ** (math.frexp(abs(float(x)))[1] - 8)


def judge_logits(path):
    """A judge's `logits` [rows][V] f32 as a read-only memmap (numpy; the files are GBs)."""
    import numpy as np
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        v = json.loads(f.read(n))["logits"]
    a, _ = v["data_offsets"]
    return np.memmap(path, dtype=np.float32, mode="r", offset=8 + n + a, shape=tuple(v["shape"]))


def worst_miss(lg, force, vocab_used=VOCAB_USED):
    """The furthest any forced id sits below the judge's top, in bf16 ulps of the top (0 when all are tops); and the
    judge's top id per row (rows 0..gen)."""
    worst, tops = 0.0, []
    for r in range(lg.shape[0]):
        row = lg[r, :vocab_used]
        t = int(row.argmax())
        tops.append(t)
        if r < len(force) and t != force[r]:
            worst = max(worst, (float(row[t]) - float(row[force[r]])) / ulp(row[t]))
    return worst, tops


def test_prompt_len(key, prompt, root):
    """The l0-int8 test's own prompt length for `key` (0 when the key is not such a test's): qwen_forced.py's."""
    qf = _load("qwen_forced")
    return qf.test_prompt_len(key, prompt, root, qf.PFI8_RE)


def sim_int8_prompt(model, I8):
    """Patch the h8 walk's linears: rows t < state['P'][b] of batch row b through fq_x / fq_w (rounded once to bf16),
    the rest standard bf16. Returns the state dict (P: one prompt length per batch row)."""
    import torch
    import torch.nn.functional as F
    state = {"P": []}

    def patch(mod):
        orig = mod.forward

        def fwd(x):
            out = orig(x)
            P = state["P"]
            if not any(P):
                return out
            wq = I8.fq_w(mod.weight.float())
            out = out.clone()
            for b, n in enumerate(P):
                if n:
                    out[b, :n] = F.linear(I8.fq_x(x[b, :n].float()), wq).to(out.dtype)
            return out
        mod.forward = fwd

    for layer in model.model.layers:
        if hasattr(layer, "linear_attn"):
            for n in ("in_proj_qkv", "in_proj_z", "out_proj"):
                patch(getattr(layer.linear_attn, n))
        else:
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                patch(getattr(layer.self_attn, n))
        for n in ("gate_proj", "up_proj", "down_proj"):
            patch(getattr(layer.mlp, n))
    return state


def select(ctx_dirs, judge_dirs, out, bar):
    """[(key, prompt, force, judge path)] of the keys whose judge rejects a forced id by more than `bar` ulp."""
    qf = _load("qwen_forced")
    ctx = qf.load_contexts(ctx_dirs)
    sel = []
    for jd in judge_dirs:
        for p in sorted(glob.glob(os.path.join(jd, "*.golden.safetensors"))):
            k = os.path.basename(p)[: -len(".golden.safetensors")]
            if k not in ctx or os.path.exists(os.path.join(out, k + ".bf16top.safetensors")):
                continue
            pr, fo = ctx[k]
            w, tops = worst_miss(judge_logits(p), fo)
            if w > bar:
                sel.append({"key": k, "prompt": pr, "force": fo, "tops": tops, "worst": float(w),
                            "pfi8_prompt": test_prompt_len(k, pr, os.path.join(_HERE, "..", "..")),
                            "i8": bool(re.search(qf.I8HEAD_RE, k))})
    return sel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--ctx", action="append", required=True)
    p.add_argument("--judge", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--bar", type=float, default=8.0)
    r = sub.add_parser("run")
    r.add_argument("snapshot")
    r.add_argument("--plan", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--topk", type=int, default=64)
    r.add_argument("--batch-tokens", type=int, default=16384)
    r.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "plan":
        sel = select(a.ctx, a.judge, a.out, a.bar)
        with open(a.plan, "w", encoding="utf-8") as f:
            json.dump(sel, f)
        print(f"plan: {len(sel)} keys with a miss beyond {a.bar} ulp -> {a.plan}")
        return
    with open(a.plan, encoding="utf-8") as f:
        sel = [s for s in json.load(f) if not os.path.exists(os.path.join(a.out, s["key"] + ".bf16top.safetensors"))]
    seqs = {}   # one forward per distinct id sequence and prompt pass
    for s in sel:
        seqs.setdefault((tuple(s["prompt"] + s["force"]), s["pfi8_prompt"]), []).append(s)
    order = sorted(seqs, key=lambda q: len(q[0]), reverse=True)
    batches, cur = [], []
    for q in order:
        if cur and (len(cur) + 1) * len(cur[0][0]) > a.batch_tokens:
            batches.append(cur)
            cur = []
        cur.append(q)
    if cur:
        batches.append(cur)
    print(f"{len(sel)} keys to do, {len(seqs)} distinct sequences, "
          f"{sum(len(q[0]) for q in seqs)} ids, {len(batches)} forwards", flush=True)
    for s in sel:
        print(f"  {s['key']}: {len(s['prompt'])} + {len(s['force'])} ids, worst miss {s['worst']:.2f} ulp"
              f"{' (int8 head)' if s['i8'] else ''}"
              f"{f', l0-int8 prompt pass over its first {s['pfi8_prompt']} ids' if s['pfi8_prompt'] else ''}",
              flush=True)
    if a.dry_run or not sel:
        return
    import torch
    from safetensors.torch import save_file
    torch.set_grad_enabled(False)
    _load("cpu_bf16").install(True)
    I8 = _load("int8_sim")
    model, tc, pf = _load("row_taps_ref").build(a.snapshot, "bf16", False)
    sim = sim_int8_prompt(model, I8)
    w = model.lm_head.weight.to(torch.bfloat16)
    heads = {False: w.float(), True: I8.int8_head_weight(w)}
    t0 = time.time()
    for bi, b in enumerate(batches):
        tb = time.time()
        B, T = len(b), len(b[0][0])
        x = torch.zeros((B, T), dtype=torch.long)
        for i, q in enumerate(b):
            x[i, :len(q[0])] = torch.tensor(q[0], dtype=torch.long)
        sim["P"] = [q[1] for q in b]
        hid = model.model(input_ids=x, use_cache=False).last_hidden_state   # final-normalised, bf16
        sim["P"] = []
        for i, q in enumerate(b):
            for s in seqs[q]:
                n_p, gen = len(s["prompt"]), len(s["force"])
                h = hid[i, n_p - 1:n_p + gen].float()                          # rows 0..gen
                lg = h @ heads[s["i8"]].t()                                    # [gen + 1][V] f32
                lg = lg[:, :VOCAB_USED]
                v, ids = torch.topk(lg, a.topk, dim=-1)
                extra = torch.tensor([[s["tops"][j], s["force"][j] if j < gen else s["tops"][j]]
                                      for j in range(gen + 1)], dtype=torch.long)
                ids = torch.cat([ids, extra], -1)
                v = torch.cat([v, lg.gather(-1, extra)], -1)
                save_file({"ids": ids.to(torch.int32).contiguous(), "logits": v.float().contiguous()},
                          os.path.join(a.out, s["key"] + ".bf16top.safetensors"),
                          metadata={"snapshot": a.snapshot, "head": "int8" if s["i8"] else "bf16",
                                    "topk": str(a.topk), "gemm": "cpu_bf16 exact",
                                    "pfi8_prompt": str(s["pfi8_prompt"])})
                print(f"[{s['key']}] {n_p} + {gen}: written", flush=True)
        print(f"batch {bi + 1}/{len(batches)} B={B} T={T}: {time.time() - tb:.0f} s (wall {time.time() - t0:.0f} s)",
              flush=True)
    print(f"done: wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
