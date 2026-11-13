#!/usr/bin/env python3
"""gdn_gate_span - phase 3 Task 1: how far the GDN gate's running sum moves inside one 64-row sub-chunk.

The factorised scan applies exp(gl) once and exp(gl - gc_i) to each row (pf_gdn_scan.cl); a sly-radiance / saber
measurement found it wrong above a span of ~160 nats. Over a B70_PF_GDN_DUMP directory (runtime/prefill/gdn.cc:
gdn<call>_g.bin = g_cum fp32 [C][v-heads], gdn<call>.txt = "C k-heads v-heads conv-rows"), per dump: the largest
|gl - gc_i| inside one 64-row sub-chunk of one head (gc_i = g_cum row i of the sub-chunk, gl = its last row), in nats,
with where it is, and how many (sub-chunk, head) pairs exceed 80 and 160 nats. The dump keeps calls 0 / 9 / 21 =
Ornith's layers 0 / 12 / 28 of the first chunk (Flash-Next has no dump hook outside src/: not available).

  gdn_gate_span.py DIR [--calls 0,9,21] [--layers 0,12,28]
Exit 0 always; a span above 80 is printed as "SPAN ABOVE 80" for the record to report to the operator.
"""
import argparse
import array
import os
import sys

SUB = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--calls", default="0,9,21")
    ap.add_argument("--layers", default="0,12,28", help="the layer of each call, for the printout")
    a = ap.parse_args()
    calls = [int(x) for x in a.calls.split(",")]
    layers = [int(x) for x in a.layers.split(",")]
    for call, layer in zip(calls, layers):
        base = os.path.join(a.dir, "gdn%d" % call)
        with open(base + ".txt") as f:
            C, hk, hv, conv = (int(x) for x in f.read().split())
        g = array.array("f")
        with open(base + "_g.bin", "rb") as f:
            g.frombytes(f.read())
        if len(g) != C * hv:
            sys.exit("%s: %d floats, expected C %d x v-heads %d" % (base, len(g), C, hv))
        best, at = 0.0, (0, 0)
        over80 = over160 = total = 0
        for t in range((C + SUB - 1) // SUB):
            b, L = t * SUB, min(SUB, C - t * SUB)
            for h in range(hv):
                gl = g[(b + L - 1) * hv + h]
                span = max(abs(gl - g[(b + i) * hv + h]) for i in range(L))
                total += 1
                over80 += span > 80.0
                over160 += span > 160.0
                if span > best:
                    best, at = span, (t, h)
        print("SPAN {\"call\":%d,\"layer\":%d,\"C\":%d,\"max_nats\":%.4f,\"sub_chunk\":%d,\"head\":%d,"
              "\"pairs\":%d,\"over_80\":%d,\"over_160\":%d}" % (call, layer, C, best, at[0], at[1], total, over80, over160))
        if best > 80.0:
            print("SPAN ABOVE 80: layer %d, %.2f nats (sub-chunk %d, head %d)" % (layer, best, at[0], at[1]))


if __name__ == "__main__":
    main()
