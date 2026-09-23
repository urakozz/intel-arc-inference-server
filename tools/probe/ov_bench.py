#!/usr/bin/env python3
"""OpenVINO's own int4 IR of Qwen3.8-27B on the B70: speed and logits, per runtime config.

    ov_bench.py <OpenVINO/Qwen3.8-27B-int4-ov snapshot> --prose <ids> --out-dir <dir>
                [--configs default,dq0,dq0_kvf16] [--pp 4096] [--tg 256] [--runs 3]

Why: the IR stores int4 asymmetric g128 weights and floating-point activations,
but OpenVINO's GPU plugin quantises activations to int8 per token at runtime by
default on XMX devices (DYNAMIC_QUANTIZATION_GROUP_SIZE, "innermost axis") and
keeps the KV cache in int8 by default. That is W4A8 without a rotation, the
combination docs/probe-w4a8-2026-09-23.md section 14 measured per layer. This
turns it into end-to-end numbers.

Per config, one compile, then:
  * logits: the prose golden prompt's full [T, V] prompt logits from one
    prefill, written as <out-dir>/ov_<config>.safetensors in the oracle's golden
    format (logits + prompt_ids metadata), so tools/rotate/check_rotation.py
    reference compares it against the bf16 model unchanged;
  * prefill: --pp ids (the prose prompt repeated, as tools/probe/bench_vllm_prefill.py
    builds them), wall time of embed + language-model infer, median of --runs
    after one warm-up;
  * decode: --tg greedy steps after that prefill (depth --pp), host loop
    included, median of --runs.

Driven through the raw OpenVINO API, no optimum: text-embeddings model ->
inputs_embeds; the stateful language model takes attention_mask, inputs_embeds,
position_ids [4, 1, T] (all four M-RoPE rows equal for text) and beam_idx.
The effective DYNAMIC_QUANTIZATION_GROUP_SIZE, KV_CACHE_PRECISION and
INFERENCE_PRECISION_HINT are read back from the compiled model and printed.
"""
import argparse
import json
import os
import statistics
import time

import numpy as np
import openvino as ov
from safetensors.numpy import save_file

CONFIGS = {
    "default": {},
    "dq0": {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"},
    "dq0_kvf16": {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0", "KV_CACHE_PRECISION": "f16"},
}
READBACK = ("DYNAMIC_QUANTIZATION_GROUP_SIZE", "KV_CACHE_PRECISION", "INFERENCE_PRECISION_HINT",
            "PERFORMANCE_HINT")


class Model:
    def __init__(self, core: ov.Core, snap: str, cfg: dict):
        t = time.time()
        self.emb = core.compile_model(os.path.join(snap, "openvino_text_embeddings_model.xml"), "GPU")
        self.lm = core.compile_model(os.path.join(snap, "openvino_language_model.xml"), "GPU", cfg)
        self.req = self.lm.create_infer_request()
        self.ereq = self.emb.create_infer_request()
        self.compile_s = time.time() - t
        self.props = {}
        for k in READBACK:
            try:
                self.props[k] = str(self.lm.get_property(k))
            except Exception as e:  # noqa: BLE001 - a property the plugin does not expose
                self.props[k] = f"<{type(e).__name__}>"

    def embed(self, ids: np.ndarray) -> np.ndarray:
        self.ereq.infer({0: ids.reshape(1, -1).astype(np.int64)})
        return self.ereq.get_output_tensor(0).data.copy()

    def step(self, ids: np.ndarray, start: int) -> np.ndarray:
        """Run len(ids) tokens at positions start..; returns logits [T, V] (a copy)."""
        n = len(ids)
        emb = self.embed(ids)
        pos = np.broadcast_to(np.arange(start, start + n, dtype=np.int64), (4, 1, n)).copy()
        self.req.infer({"inputs_embeds": emb,
                        "attention_mask": np.ones((1, start + n), dtype=np.int64),
                        "position_ids": pos,
                        "beam_idx": np.zeros((1,), dtype=np.int32)})
        return self.req.get_tensor("logits").data[0]

    def reset(self):
        self.req.reset_state()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--prose", required=True, help="tests/golden/prompts/prose.ids")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--configs", default="default,dq0,dq0_kvf16")
    ap.add_argument("--pp", type=int, default=4096)
    ap.add_argument("--tg", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--logits-ids", help="take the logits on this prompt instead of --prose")
    ap.add_argument("--logits-tokens", type=int, default=0, help="first N ids of --logits-ids")
    ap.add_argument("--no-speed", action="store_true", help="logits only")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.prose, encoding="utf-8") as f:
        prose = [int(x) for x in f.read().split()]
    pp_ids = np.array([prose[i % len(prose)] for i in range(args.pp)], dtype=np.int64)
    core = ov.Core()
    print(f"openvino {ov.__version__}; GPU: {core.get_property('GPU', 'FULL_DEVICE_NAME')}")
    rows = []
    for name in args.configs.split(","):
        cfg = CONFIGS[name]
        print(f"\n## config {name}: {cfg or '(plugin defaults)'}", flush=True)
        m = Model(core, args.snapshot, cfg)
        print(f"compiled in {m.compile_s:.0f}s; effective: {json.dumps(m.props)}", flush=True)

        # logits, one prefill. Dynamic quantisation did NOT engage on the
        # 42-id prose prompt (default and DQ=0 logits bit-identical) while it
        # did at 4096 ids, so a long prompt is needed to see its accuracy.
        lids = prose
        if args.logits_ids:
            with open(args.logits_ids, encoding="utf-8") as f:
                lids = [int(x) for x in f.read().split()]
            if args.logits_tokens:
                lids = lids[: args.logits_tokens]
        m.reset()
        logits = m.step(np.array(lids, dtype=np.int64), 0).astype(np.float32).copy()
        tag = f"_{len(lids)}" if args.logits_ids else ""
        path = os.path.join(args.out_dir, f"ov_{name}{tag}.safetensors")
        save_file({"logits": logits}, path, metadata={
            "prompt_ids": " ".join(map(str, lids)), "snapshot": os.path.basename(args.snapshot.rstrip("/")),
            "runtime": f"openvino {ov.__version__} GPU", "config": json.dumps(cfg),
            "effective": json.dumps(m.props)})
        print(f"logits {logits.shape} -> {path}", flush=True)
        if args.no_speed:
            del m
            continue

        # prefill, then decode at depth pp
        pps, tgs = [], []
        for r in range(args.runs + 1):
            m.reset()
            t = time.perf_counter()
            last = m.step(pp_ids, 0)[-1]
            pp_s = time.perf_counter() - t
            tok = int(np.argmax(last))
            t = time.perf_counter()
            for i in range(args.tg):
                tok = int(np.argmax(m.step(np.array([tok], dtype=np.int64), args.pp + i)[-1]))
            tg_s = time.perf_counter() - t
            tag = "warm-up" if r == 0 else f"run {r}"
            print(f"  {tag}: prefill {args.pp / pp_s:8.2f} t/s ({pp_s * 1000:.1f} ms), "
                  f"decode {args.tg / tg_s:6.2f} t/s", flush=True)
            if r:
                pps.append(args.pp / pp_s)
                tgs.append(args.tg / tg_s)
        rows.append((name, statistics.median(pps), min(pps), max(pps), statistics.median(tgs),
                     min(tgs), max(tgs), m.props))
        del m

    if not rows:
        return
    print(f"\n## OpenVINO int4-ov on the B70, pp{args.pp} / tg{args.tg}, median of {args.runs}\n")
    print("| config | prefill t/s | range | decode t/s | range | dyn-quant group | KV cache | infer precision |")
    print("|---|---:|---|---:|---|---|---|---|")
    for n, p, pl, ph, g, gl, gh, props in rows:
        print(f"| {n} | {p:.2f} | {pl:.2f} .. {ph:.2f} | {g:.2f} | {gl:.2f} .. {gh:.2f} | "
              f"{props['DYNAMIC_QUANTIZATION_GROUP_SIZE']} | {props['KV_CACHE_PRECISION']} | "
              f"{props['INFERENCE_PRECISION_HINT']} |")


if __name__ == "__main__":
    main()
