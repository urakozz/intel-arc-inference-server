#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Calibration samples for the DFlash 2 drafter: blocks of 8 over the QUANTISED target's taps.

    data = CalibData(taps_dir, drafter_cfg, target_dir, drafter_dir, nsamples=128, anchors=32, seed=42)
    for sample in data: model(**sample)                       # model.py's DFlash2Calib

The taps are the engine's (tools/probe/probe_dflash_iter --engine-taps, run by quantize_dflash.sh's `taps`
stage on urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ): one `<recording>__<request>.taps.safetensors` per
recorded request - `ids` (prompt + recorded continuation), `tap.L{i}` (the residual after target layer i, bf16,
positions tap_from .. n - 1), metadata `name`, `n_prompt`, `n`, `tap_from`, `layers` (dump_taps.py's format).

One sample = one request and `anchors` anchor positions p drawn (seed) from its continuation [n_prompt, n): block
a is [ids[p_a], mask x 7] at positions p_a .. p_a + 7 (the drafter's trained block, block_size 8), teacher-forced
on the recording. The context rows are the taps at positions [min p - (window - 1), max p) - every key any row
of the sample can see (dflash_ref's window rule) - padded with kpos -1 to `context` rows. Samples go round-robin
over the requests (seed-shuffled), so 128 samples cover every request when there are fewer than 128.

Block-row inputs are the target's embedding rows (`mask_embedding.pt` beside the drafter replaces the mask row,
as vLLM and dflash_ref do) times input_embedding_scale; RoPE cos / sin in float64, as dflash_ref.rope.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random

import torch
from safetensors import safe_open

try:
    from . import model as M
except ImportError:
    import model as M

ref = M.ref


def dump_meta(path: str) -> dict:
    with safe_open(path, framework="pt", device="cpu") as f:
        md = dict(f.metadata())
    return {"path": path, "name": md["name"], "n_prompt": int(md["n_prompt"]), "n": int(md["n"]),
            "tap_from": int(md["tap_from"]), "layers": [int(i) for i in md["layers"].split(",")]}


def dumps(taps_dir: str) -> list[dict]:
    return [dump_meta(p) for p in sorted(glob.glob(os.path.join(taps_dir, "*.taps.safetensors")))]


def plan(metas: list[dict], nsamples: int, anchors: int, seed: int, span: int) -> list[dict]:
    """[{name, path, anchors}] - `anchors` sorted positions per sample, spread within `span` positions."""
    rng = random.Random(seed)
    usable = sorted((m for m in metas if m["n"] > m["n_prompt"]), key=lambda m: m["name"])
    if not usable:
        raise ValueError("no tap dump with a continuation")
    rng.shuffle(usable)
    pools: dict[str, list[int]] = {}
    out = []
    for i in range(nsamples):
        m = usable[i % len(usable)]
        lo, hi = max(m["n_prompt"], m["tap_from"] + 1), m["n"]
        if hi - lo > span:
            lo = rng.randrange(lo, hi - span + 1)
            hi = lo + span
        pool = pools.setdefault(m["name"], [])
        picked = []
        while len(picked) < anchors:
            if not pool:
                pool.extend(range(lo, hi))
                rng.shuffle(pool)
            p = pool.pop()
            if lo <= p < hi:
                picked.append(p)
        out.append({"name": m["name"], "path": m["path"], "anchors": sorted(picked)})
    return out


def rope_tables(pos: torch.Tensor, head_dim: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    ang = pos.to(torch.float64)[:, None] * inv[None]
    return ang.cos().float(), ang.sin().float()


class EmbedRows:
    """Rows of the target's input embedding, read one by one (the 2.5 GB table never loads whole)."""

    def __init__(self, target_dir: str, drafter_dir: str, cfg):
        wm = ref._weight_map(target_dir)
        names = [n for n in wm if n.endswith("embed_tokens.weight") and not n.startswith(("mtp.", "model.visual", "visual"))]
        if len(names) != 1:
            raise ValueError(f"target embedding ambiguous: {names}")
        self.path, self.name = wm[names[0]], names[0]
        self.mask_id = cfg.mask_token_id
        self.mask = ref.read_mask_embedding(drafter_dir, cfg.mask_token_id)
        self.cache: dict[int, torch.Tensor] = {}

    def __call__(self, ids: list[int]) -> torch.Tensor:
        miss = sorted({i for i in ids if i not in self.cache})
        if miss:
            with safe_open(self.path, framework="pt", device="cpu") as f:
                sl = f.get_slice(self.name)
                for i in miss:
                    self.cache[i] = sl[i:i + 1][0].float()
        rows = torch.stack([self.cache[i] for i in ids])
        if self.mask is not None:
            rows[torch.tensor(ids) == self.mask_id] = self.mask.float()
        return rows


def build(entry: dict, cfg, embed: EmbedRows, context: int, dtype=torch.bfloat16) -> dict:
    """One sample's tensors (batch 1), the keyword arguments of DFlash2Calib.forward."""
    n_block = cfg.block_size
    window = max(cfg.windows) if all(w is not None for w in cfg.windows) else None
    if window is None:
        raise NotImplementedError("a full-attention drafter layer: the context would be every position")
    ps = entry["anchors"]
    with safe_open(entry["path"], framework="pt", device="cpu") as f:
        md = f.metadata()
        tap_from = int(md["tap_from"])
        ids = f.get_tensor("ids").long()
        lo, hi = max(tap_from, ps[0] - (window - 1)), ps[-1]
        if hi - lo > context:
            raise ValueError(f"{entry['name']}: context {hi - lo} rows > {context}")
        taps = torch.cat([f.get_slice(f"tap.L{i}")[lo - tap_from:hi - tap_from] for i in cfg.target_layer_ids], -1)
    C, R = context, len(ps) * n_block
    x = torch.zeros(C, taps.shape[1], dtype=dtype)
    x[: hi - lo] = taps.to(dtype)
    kpos = torch.full((C,), -1, dtype=torch.long)
    kpos[: hi - lo] = torch.arange(lo, hi)
    block_ids = [t for p in ps for t in [int(ids[p])] + [cfg.mask_token_id] * (n_block - 1)]
    qpos = torch.tensor([p + j for p in ps for j in range(n_block)], dtype=torch.long)
    emb = embed(block_ids)
    if cfg.version == 2:
        emb = emb * cfg.input_embedding_scale
    q_cos, q_sin = rope_tables(qpos, cfg.head_dim, cfg.rope_theta)
    k_cos, k_sin = rope_tables(kpos.clamp(min=0), cfg.head_dim, cfg.rope_theta)
    return {"input_ids": torch.tensor([block_ids]), "attention_mask": torch.ones(1, R, dtype=torch.long),
            "taps": x[None], "embeds": emb.to(dtype)[None], "qpos": qpos[None], "kpos": kpos[None],
            "q_cos": q_cos[None], "q_sin": q_sin[None], "k_cos": k_cos[None], "k_sin": k_sin[None]}


class CalibData:
    """AutoRound's `dataset`: an iterable of sample dicts, built on demand (not a list: AutoRound would read a list
    as raw text samples)."""

    def __init__(self, taps_dir: str, drafter_dir: str, target_dir: str, nsamples: int, anchors: int, seed: int,
                 context: int, dtype=torch.bfloat16):
        self.cfg = ref.read_config(drafter_dir)
        window = max(self.cfg.windows)
        self.metas = dumps(taps_dir)
        if not self.metas:
            raise ValueError(f"no *.taps.safetensors in {taps_dir}")
        for m in self.metas:
            if m["layers"] != self.cfg.target_layer_ids:
                raise ValueError(f"{m['path']}: taps of layers {m['layers']}, the drafter reads {self.cfg.target_layer_ids}")
        self.context = context
        self.entries = plan(self.metas, nsamples, anchors, seed, span=context - (window - 1))
        self.embed = EmbedRows(target_dir, drafter_dir, self.cfg)
        self.dtype = dtype

    @property
    def rows(self) -> int:
        """Block rows per sample: AutoRound's `seqlen`."""
        return len(self.entries[0]["anchors"]) * self.cfg.block_size

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        for e in self.entries:
            yield build(e, self.cfg, self.embed, self.context, self.dtype)

    def summary(self) -> dict:
        per = {}
        for e in self.entries:
            per[e["name"]] = per.get(e["name"], 0) + 1
        return {"dumps": len(self.metas), "samples": len(self.entries), "anchors_per_sample": len(self.entries[0]["anchors"]),
                "block_size": self.cfg.block_size, "context_rows": self.context, "rows_per_sample": self.rows,
                "requests_used": len(per), "recordings": sorted({n.split("/")[0] for n in per}),
                "tokens_tap_rows": sum(m["n"] - m["tap_from"] for m in self.metas)}


def context_rows(window: int, max_cont: int) -> int:
    """Context rows that hold any sample whose anchors lie in a continuation of max_cont ids (a multiple of 64)."""
    return int(math.ceil((window - 1 + max_cont) / 64) * 64)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="print the calibration plan's summary")
    ap.add_argument("taps_dir")
    ap.add_argument("drafter")
    ap.add_argument("target")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--anchors", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context", type=int, default=context_rows(2048, 256))
    a = ap.parse_args()
    d = CalibData(a.taps_dir, a.drafter, a.target, a.nsamples, a.anchors, a.seed, a.context)
    print(json.dumps(d.summary(), indent=1))
