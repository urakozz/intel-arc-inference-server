#!/usr/bin/env python3
"""Byte accounting for a HF snapshot: the W behind every t/s ceiling (doc 03, 05).

Reads model.safetensors.index.json as the manifest and the safetensors headers
(no tensor data), groups bytes by role, and prints W = bytes read per decode
token: int4 qweight + f16 scales + bf16 lm_head + small bf16 tensors. qzeros and
g_idx are dropped at load, the embedding is gathered, the vision tower is
skipped, the MTP head is phase 2.

usage: checkpoint_bytes.py <snapshot_dir>
"""
import collections, json, os, struct, sys

def main(d):
    d = d.rstrip("/") + "/"
    ix = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
    hdr = {}
    for f in sorted(set(ix.values())):
        with open(d + f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            hdr.setdefault(k, {})[f] = (v["dtype"], v["shape"], v["data_offsets"][1] - v["data_offsets"][0])
    missing = [k for k in ix if k not in hdr]
    if missing:
        sys.exit(f"{len(missing)} tensors in index but not in files, e.g. {missing[:3]}")
    groups = collections.OrderedDict((g, [0, 0]) for g in
        ["lm_head", "embed_tokens", "lm.qweight", "lm.scales", "lm.qzeros", "lm.g_idx", "lm.other", "mtp", "visual", "other"])
    for k in ix:
        dt, sh, nb = hdr[k][ix[k]]
        if k.startswith("lm_head"): g = "lm_head"
        elif "embed_tokens" in k: g = "embed_tokens"
        elif k.startswith("mtp."): g = "mtp"
        elif k.startswith("model.visual"): g = "visual"
        elif k.startswith("model.language_model"):
            suf = k.rsplit(".", 1)[-1]
            g = {"qweight": "lm.qweight", "scales": "lm.scales", "qzeros": "lm.qzeros", "g_idx": "lm.g_idx"}.get(suf, "lm.other")
        else: g = "other"
        groups[g][0] += 1; groups[g][1] += nb
    print("| group | tensors | GB |\n|---|---|---|")
    for g, (n, b) in groups.items():
        print(f"| {g} | {n} | {b/1e9:.3f} |")
    W = sum(groups[g][1] for g in ["lm_head", "lm.qweight", "lm.scales", "lm.other"])
    lh = groups["lm_head"][1]
    print(f"\nW = {W/1e9:.3f} GB/token  -> ceiling @600 GB/s = {600e9/W:.1f} t/s")
    print(f"W with lm_head int4 = {(W - lh + lh*4.125/16)/1e9:.3f} GB -> {600e9/(W - lh + lh*4.125/16):.1f} t/s")
    print(f"W with lm_head int8 = {(W - lh/2)/1e9:.3f} GB -> {600e9/(W - lh/2):.1f} t/s")
    print(f"MTP head (phase 2, per draft step) = {groups['mtp'][1]/1e9:.3f} GB")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
