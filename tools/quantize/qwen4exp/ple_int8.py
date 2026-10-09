#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next's PLE n-gram table as int8 rows, the file the engine pins in host USM (spec 21b
Task 1; spec 21 §5 and decision 7's proposal: a one-time file beside the checkpoint).

    ple_int8.py <snapshot> <out dir> [--scale bf16|f32]

The source is the checkpoint's own table - 128 shards `...layers.<l>.ple.ple_embedding.ngram_embedding.
shard_K.weight` [rows_per][160] (bf16 in the original and Intel's export, fp32 in the tiny model), which
concatenate to the padded [320,001,536][160] table (102.4 GB bf16). Each of the 16 n-gram heads owns the
global rows [offset_h, offset_h + prime_h) (21a's facts sheet: no shard boundary is a head boundary), so
the file stores the table PER HEAD, head-local row r = global row offset_h + r:

  ple.h<h>.q   I8   [prime_h][160]   q = clamp(rne(w / s), -127, 127)
  ple.h<h>.s   BF16 | F32 [prime_h]  s = max|w| / 127 in fp32 (spec 9's row rule, src/loader/lm_head_int8.h);
                                     --scale bf16 rounds s to bf16 ONCE and quantises with the rounded s;
                                     a zero row has s = 0 and q = 0
  ple.layer_multipliers, ple.ngram_heads_vocab_sizes, ple.ngram_heads_offsets   I64, copied from the
                                     checkpoint after checking them against the formula (qwen4exp_facts)

A row dequantises to bf16(float(q) x float(s)) (qwen4exp_ref.PleTable's int8 reader, 21c's gather).
Written as `<out>/ple_int8-000NN-of-00017.safetensors` (one file per head, then the I64 constants) and a
`model.safetensors.index.json` (loader::SafetensorsSet refuses an index-less directory); every file's
metadata says {source: <repo>@<revision>, rule: "row-int8-spec9", scale}. The source is read by mmap in
chunks of 65,536 rows and each head's q streamed to its file (peak RAM ~0.3 GB, ESTIMATED); on the real
table: 51.2 GB of q + 0.64 GB of bf16 scales (1.28 GB f32) out (derived).
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.abspath(os.path.join(HERE, "..", "..", "oracle"))
sys.path.insert(0, HERE)
import make_synth as MS  # noqa: E402

facts = MS.facts
CHUNK = 65536
RULE = "row-int8-spec9"


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def tensor_map(snapshot: str) -> dict:
    """name -> (file, entry, data start) over the snapshot's index (or its single model.safetensors)."""
    idx = os.path.join(snapshot, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as f:
            wm = json.load(f)["weight_map"]
    else:
        h, _ = read_header(os.path.join(snapshot, "model.safetensors"))
        wm = {k: "model.safetensors" for k in h if k != "__metadata__"}
    heads: dict = {}
    out = {}
    for name, fn in wm.items():
        if fn not in heads:
            heads[fn] = read_header(os.path.join(snapshot, fn))
        h, start = heads[fn]
        out[name] = (os.path.join(snapshot, fn), h[name], start)
    return out


def mmap_of(entry) -> np.ndarray:
    path, e, start = entry
    dt = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16, "I64": np.int64, "I8": np.int8, "I32": np.int32}[e["dtype"]]
    a, _ = e["data_offsets"]
    return np.memmap(path, dtype=dt, mode="r", offset=start + a, shape=tuple(e["shape"]))


def as_f32(arr: np.ndarray, dtype: str) -> torch.Tensor:
    t = torch.from_numpy(np.array(arr))   # a copy: the mmap is read-only
    if dtype == "BF16":
        return t.view(torch.bfloat16).float()
    return t.float()


def quantise_rows(w: torch.Tensor, scale: str) -> tuple[torch.Tensor, torch.Tensor]:
    """[n][160] float -> (q I8 [n][160], s [n] fp32 or bf16): spec 9's row rule (make_tiny.write_ple_int8's)."""
    w = w.float()
    s = w.abs().amax(-1) / 127.0
    if scale == "bf16":
        s = s.to(torch.bfloat16).float()
    q = torch.where(s.unsqueeze(-1) > 0, torch.round(w / torch.where(s > 0, s, 1.0).unsqueeze(-1)),
                    torch.zeros_like(w)).clamp(-127, 127).to(torch.int8)
    return q, (s.to(torch.bfloat16) if scale == "bf16" else s)


def source_name(snapshot: str) -> str:
    """<repo>@<revision> for an HF cache snapshot (models--Org--Name/snapshots/<rev>), else the path."""
    p = os.path.abspath(snapshot).rstrip("/")
    parts = p.split(os.sep)
    if len(parts) >= 3 and parts[-2] == "snapshots" and parts[-3].startswith("models--"):
        return parts[-3][len("models--"):].replace("--", "/", 1) + "@" + parts[-1]
    return p


def ple_source(snapshot: str):
    """(text config, PLE prefix, shard names, tensor map) - the table the snapshot carries, checked."""
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        t = MS.text_of(json.load(f))
    ids = t.get("ple_layer_ids") or []
    if len(ids) != 1:
        raise SystemExit(f"ple_int8: ple_layer_ids {ids}: one PLE layer expected")
    tm = tensor_map(snapshot)
    pre = "model.language_model." if any(n.startswith("model.language_model.") for n in tm) else "model."
    base = f"{pre}layers.{ids[0] - 1}.ple.ple_embedding."
    parts = t.get("split_ngram_parts", 512)
    shards = [f"{base}ngram_embedding.shard_{k}.weight" for k in range(parts)]
    missing = [n for n in shards if n not in tm]
    if missing:
        raise SystemExit(f"ple_int8: {snapshot} has no {missing[0]} ({len(missing)} of {parts} shards missing)")
    return t, base, shards, tm


def check_constants(t: dict, base: str, tm: dict) -> tuple[list, list, list]:
    """The checkpoint's I64 tensors against the formula (M:1040-1105); refuses on any difference."""
    sizes, offsets, _, _ = facts.head_table(t, 0)
    mult = facts.layer_multipliers(t["vocab_size"], t["ngram_size"], 0, t.get("seed", 1234))
    for name, want in (("layer_multipliers", mult), ("ngram_heads_vocab_sizes", sizes), ("ngram_heads_offsets", offsets)):
        got = [int(x) for x in mmap_of(tm[base + name])]
        if got != list(want):
            raise SystemExit(f"ple_int8: {base}{name} = {got[:4]}... differs from the formula {list(want)[:4]}... "
                             f"(ngram_vocab_size_base {t['ngram_vocab_size_base']}, seed {t.get('seed', 1234)})")
    return mult, sizes, offsets


def header_bytes(tensors: list, meta: dict) -> bytes:
    """A safetensors header for [(name, dtype, shape, nbytes)] laid out in order."""
    h, off = {"__metadata__": meta}, 0
    for name, dt, shape, nb in tensors:
        h[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}
        off += nb
    raw = json.dumps(h, separators=(",", ":")).encode()
    raw += b" " * ((8 - (8 + len(raw)) % 8) % 8)
    return struct.pack("<Q", len(raw)) + raw


def convert(snapshot: str, out: str, scale: str = "bf16", verbose: bool = True) -> dict:
    if scale not in ("bf16", "f32"):
        raise ValueError(f"--scale {scale!r}: bf16 or f32")
    t, base, shards, tm = ple_source(snapshot)
    mult, sizes, offsets = check_constants(t, base, tm)
    heads = len(sizes)
    rows_per = int(tm[shards[0]][1]["shape"][0])
    dim = int(tm[shards[0]][1]["shape"][1])
    total = offsets[-1] + sizes[-1]
    if rows_per * len(shards) < total:
        raise SystemExit(f"ple_int8: {len(shards)} shards x {rows_per} rows < the table's {total} addressed rows")
    os.makedirs(out, exist_ok=True)
    meta = {"source": source_name(snapshot), "rule": RULE, "scale": scale, "format": "pt"}
    sdt, sbytes = ("BF16", 2) if scale == "bf16" else ("F32", 4)
    nfiles = heads + 1
    fname = [f"ple_int8-{i + 1:05d}-of-{nfiles:05d}.safetensors" for i in range(nfiles)]
    wm = {}
    t0 = time.time()
    files, svals = [], []
    for h in range(heads):
        n = sizes[h]
        f = open(os.path.join(out, fname[h]), "wb")
        f.write(header_bytes([(f"ple.h{h}.q", "I8", (n, dim), n * dim), (f"ple.h{h}.s", sdt, (n,), n * sbytes)], meta))
        files.append(f)
        svals.append(torch.empty(n, dtype=torch.bfloat16 if scale == "bf16" else torch.float32))
        wm[f"ple.h{h}.q"] = wm[f"ple.h{h}.s"] = fname[h]
    # Global rows in order: shard k holds [k x rows_per, (k + 1) x rows_per); head h [offset_h, offset_h + prime_h).
    h = 0
    for k, name in enumerate(shards):
        arr = mmap_of(tm[name])
        dt = tm[name][1]["dtype"]
        g0 = k * rows_per
        for a in range(0, rows_per, CHUNK):
            b = min(a + CHUNK, rows_per)
            lo, hi = g0 + a, g0 + b
            if lo >= total:
                break
            w = as_f32(arr[a:b], dt)
            while h < heads and lo < hi:
                he = offsets[h] + sizes[h]
                take = min(hi, he) - lo
                q, s = quantise_rows(w[lo - g0 - a: lo - g0 - a + take], scale)
                files[h].write(q.numpy().tobytes())
                r0 = lo - offsets[h]
                svals[h][r0:r0 + take] = s
                lo += take
                if lo == he:
                    files[h].write(svals[h].view(torch.int16).numpy().tobytes() if scale == "bf16"
                                   else svals[h].numpy().tobytes())
                    files[h].close()
                    if verbose:
                        print(f"  head {h}: {sizes[h]} rows (global {offsets[h]}..{he - 1}), {time.time() - t0:.0f}s",
                              flush=True)
                    h += 1
        if h == heads:
            break
    if h != heads:
        raise SystemExit(f"ple_int8: only {h} of {heads} heads written - the shards end early")
    consts = [("ple.layer_multipliers", mult), ("ple.ngram_heads_vocab_sizes", sizes), ("ple.ngram_heads_offsets", offsets)]
    with open(os.path.join(out, fname[heads]), "wb") as f:
        f.write(header_bytes([(n, "I64", (len(v),), 8 * len(v)) for n, v in consts], meta))
        for _, v in consts:
            f.write(np.asarray(v, dtype=np.int64).tobytes())
    for n, _ in consts:
        wm[n] = fname[heads]
    nbytes = sum(sizes) * (dim + sbytes) + sum(8 * len(v) for _, v in consts)
    with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": dict(meta, total_size=nbytes), "weight_map": dict(sorted(wm.items()))}, f, indent=1)
    return {"heads": heads, "rows": sum(sizes), "dim": dim, "scale": scale, "bytes": nbytes,
            "source": meta["source"], "seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--scale", choices=("bf16", "f32"), default="bf16",
                    help="the scale dtype (decision 7: bf16 0.64 GB or f32 1.28 GB on the real table)")
    a = ap.parse_args()
    r = convert(a.snapshot, a.out, a.scale)
    print(f"wrote {a.out}: {r['heads']} heads, {r['rows']} rows x {r['dim']} int8 + {a.scale} scales, "
          f"{r['bytes'] / 1e9:.3f} GB, {r['seconds']:.0f}s (source {r['source']})")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
