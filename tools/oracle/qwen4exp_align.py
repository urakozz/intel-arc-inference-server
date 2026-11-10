#!/usr/bin/env python3
"""Split engine route captures and package paired, independently judged f32 companions."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]
import argparse  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402


def split(capture, directory):
    t = load_file(capture)
    spec = t["trace.spec"].tolist()
    if len(spec) != 10 or spec[0] != 1:
        raise ValueError("unsupported engine route capture")
    os.makedirs(directory, exist_ok=True)
    for mode in ("prefill", "decode"):
        out = {"prompt.ids": t["prompt.ids"], "tokens": t["tokens"], "trace.spec": t["trace.spec"],
               "trace.path": torch.tensor([1 if mode == "prefill" else 0], dtype=torch.int32)}
        for l in range(spec[1]):
            routes = t[f"{mode}.route.ids.L{l}"]
            if routes.shape != (spec[2] + spec[3], 10):
                raise ValueError(f"{mode} L{l}: incomplete route capture")
            out[f"route.ids.L{l}"] = routes
        save_file(out, os.path.join(directory, mode + ".routes.safetensors"))


def merge(original, directory, output):
    base = load_file(original)
    pair = {k: base[k] for k in ("state.spec", "prompt.ids", "tokens", "logits")}
    pair["state.spec"] = pair["state.spec"].clone()
    pair["state.spec"][0] = 2
    for mode in ("prefill", "decode"):
        t = load_file(os.path.join(directory, mode + ".state.safetensors"))
        trace = load_file(os.path.join(directory, mode + ".routes.safetensors"))
        for key in ("prompt.ids", "tokens"):
            if not torch.equal(t[key].long(), pair[key].long()):
                raise ValueError(f"{mode}: reference contexts differ from original f32 continuation")
        if int(t["state.spec"][0]) != 3 or not torch.equal(t["state.spec"][1:], pair["state.spec"][1:]):
            raise ValueError(f"{mode}: incompatible reference specification")
        if t["engine.trace.path"].tolist() != [1 if mode == "prefill" else 0]:
            raise ValueError("prefill/decode companions are swapped")
        for name in ("spec", "path"):
            if not torch.equal(t["engine.trace." + name], trace["trace." + name]):
                raise ValueError("stale engine capture identity")
        for l in range(int(t["state.spec"][2])):
            if not torch.equal(t[f"engine.route.ids.L{l}"].long(), trace[f"route.ids.L{l}"].long()):
                raise ValueError("stale engine route capture")
        for k, value in t.items():
            if k not in ("state.spec", "prompt.ids", "tokens"):
                pair[mode + "." + k] = value
        counts = torch.bincount(torch.cat([t[f"route.judge.L{l}"] for l in range(int(t["state.spec"][2]))]), minlength=3)
        print(f"{mode} f32 route judge: exact {int(counts[0])}, near-tie {int(counts[1])}, REJECT {int(counts[2])}", flush=True)
    save_file(pair, output, metadata={"route_alignment": "engine sets, f32-judged one-bf16-ulp swaps only",
                                      "state_format": "qwen4exp-prefill-paired-v2",
                                      "force_context": "original f32 continuation"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("capture")
    s.add_argument("directory")
    m = sub.add_parser("merge")
    m.add_argument("original")
    m.add_argument("directory")
    m.add_argument("output")
    a = ap.parse_args()
    if a.cmd == "split":
        split(a.capture, a.directory)
    else:
        merge(a.original, a.directory, a.output)
