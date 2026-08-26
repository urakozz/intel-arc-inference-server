#!/usr/bin/env python3
"""vLLM's int4 GEMV at our six production shapes - the comparative half of
"skill or silicon" (spec 1.7 §3, plan 5 task 2).

WHAT THIS TIMES, AND WHY THAT PATH
----------------------------------
vLLM picks its quantisation backend from the checkpoint's `quant_method`.
The phase-1 checkpoint (Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ) declares
`quant_method: "gptq"`, `bits: 4`, `sym: true`, `group_size: 64`,
`desc_act: false`, so the production path is

    AutoGPTQConfig  ->  AutoGPTQLinearMethod              (auto_gptq.py)
      -> choose_mp_linear_kernel  ->  XPUwNa16LinearKernel (mixed_precision/xpu.py)
        -> torch.ops._xpu_C.int4_gemm_w4a16                (vllm-xpu-kernels)
          -> oneDNN matmul, f16_int4 / nt, weights streamed in place
             (csrc/xpu/onednn/int4_gemm_w4a16.h)

That last hop is a CLOSED-SOURCE oneDNN matmul. That is fine and it is the
point: it IS the kernel that produced the 31.50 t/s / 83.0% MBU bar in
docs/05-perf-model.md, so it is the right thing to put beside our own GEMV.

This script does not load the 27B model. It drives vLLM's own
`AutoGPTQLinearMethod.create_weights` / `process_weights_after_loading` for
each shape - so the kernel selection, the weight layout transform and the op
call are all vLLM's code, not a reimplementation - then times the op with
random weights. Timing does not care about weight values, but they must be
RANDOM: the B70 compresses uniform fills (docs/01-hardware.md).

BYTE ACCOUNTING - OURS, SO THE COLUMNS COMPARE
----------------------------------------------
GB/s = (qweight bytes + scales bytes) / seconds, GB = 1e9.
  qweight = K*N/2   (int4)          scales = (K/64)*N*2   (fp16)
ZERO POINTS ARE EXCLUDED. This is not a thumb on the scale: for a GPTQ v1 sym
checkpoint `XPUwNa16LinearKernel.process_weights_after_loading` REPLACES the
[K/64, N/8] `qzeros` tensor with the scalar `torch.Tensor([8])`, and the oneDNN
header takes the `zp.dim() == 1` branch (mask 0, one s8 value). So vLLM streams
ONE byte of zero point per matmul, exactly as our engine streams none. Counting
the checkpoint's qzeros tensor would inflate both engines by ~6% for bytes
neither of them reads.

TIMING - THE HOUSE METHOD, TRANSPLANTED
---------------------------------------
tests/kernels/gemv_harness.h: NB = max(2, 72MB/qweight_bytes + 1) weight copies
so one cycle through them exceeds 3x the 24 MB L2; `launches` calls cycling
those copies; 8 replays, drop the first 3, median of the last 5, / launches.
Same NB formula, same 40 launches, same 8/drop-3.

Two differences from the C++ probe, both recorded rather than papered over:
  * our probe records 40 launches into ONE Level Zero command list and replays
    it, so the host is out of the loop. Here every call goes through Python and
    the torch dispatcher. `enqueue us/call` in the output is the control: it is
    the wall time of the submission loop BEFORE the final synchronize. When it
    is well below `us/call` the number is device-bound; when it approaches it,
    the row is a host-bound upper bound and says so.
  * cold-start ramp (docs/probe-gemv-loads-2026-08-26.md): the first recorded
    configuration on a cold device read 355-378 GB/s where the identical binary
    read 532 warm (+42.8%). So: warm the device with a timed dispatch loop
    (>= --warm-seconds of GPU work AND >= --warm-dispatches calls) before the
    first timed shape, discard one whole warm-up pass per shape, and re-time
    the FIRST shape at the end as the ramp/drift control.

PRE-REGISTRATION - written before a single number was measured
--------------------------------------------------------------
Anchor: vLLM does 31.50 t/s on this checkpoint = 489 GB/s effective = 83.0% of
the measured 590 GB/s DRAM wall (docs/05-perf-model.md:47), over W = 15.52 GB
of weight bytes per token. That 489 is an END-TO-END average: it contains
attention, norms, sampling and whatever host gap survives XPU graphs, all of
which run below the wall. The pure int4 GEMV must therefore be ABOVE 489, and
if everything-else costs 5-12% of the step it lands around 510-580 GB/s.

  Point prediction: mean over the six shapes 520-560 GB/s; no single shape
  outside 480-585 GB/s.
  Predicted ordering: `lm_head` (644 MB) best, `out/o_proj` (15.94 MB) worst -
  the same ordering our own probe shows, for the same reason (fixed launch and
  tail cost is a bigger fraction of a small shape).
  Predicted host cost: 8-25 us/call of enqueue. `out/o_proj` (predicted ~30 us
  of device time) is the only shape at real risk of being host-bound.

  H_silicon (primary): oneDNN lands within +/-5% of our `base` at >= 4 of the 6
  shapes. Then the 533-584 plateau is the machine's M=1 int4 wall and Task 1's
  null result is about silicon, not skill.
  H_skill (alternative): oneDNN beats our `base` by >= 5% at >= 4 of 6 shapes.
  Then there is per-shape headroom we have not taken and the plateau is ours,
  not the card's.

  Falsification bars, set BELOW the hypotheses' own ceilings this time (the
  flaw scored in docs/probe-gemv-loads-2026-08-26.md's pre-registration):
    1. mean(theirs) / mean(ours `base`) >= 1.05  -> H_skill.
    2. mean(theirs) / mean(ours `base`) <= 0.95  -> we are already ahead of the
       production path and the comparison stops being a ceiling.
    3. their best shape >= 585 GB/s -> above our best composed cell (579) and
       within 1% of the measured wall: headroom exists at that shape.
    4. any shape > 600 GB/s -> above the DRAM wall; instrument error, not a
       result. Investigate before quoting anything from the run.

USAGE (inside the reference container, read-only, on the box)
------------------------------------------------------------
  python3 /ws/tools/probe/vllm_gemv_bench.py --snap "$SNAP"
Flags: --act {float16,bfloat16,both}, --launches, --warm-seconds,
       --warm-dispatches, --json PATH, --only SUBSTRING.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

# --------------------------------------------------------------------------
# The six production shapes, in the order tools/probe/probe_gemv.cc measures
# them, so the ramp control lines up. `AB` (in_proj_a/b) is bf16 in the
# checkpoint and has no int4 path - deliberately absent.
#   name, K (input size), N (output size)
SHAPES: list[tuple[str, int, int]] = [
    ("out/o_proj", 6144, 5120),
    ("q|k|v", 5120, 14336),
    ("qkv|z", 5120, 16384),
    ("gate|up", 5120, 34816),
    ("down", 17408, 5120),
    ("lm_head int4", 5120, 248320),
]

# Our own numbers, QUOTED from docs/probe-gemv-loads-2026-08-26.md so the
# printed table is directly comparable. `base` is src/kernels/gemv.cl at the
# production layout/S; `best` is that shape's best measured cell in the same
# battery. Both are MEASURED, by the C++ probe, not by this script.
OURS: dict[str, tuple[int, int]] = {
    # name: (base GB/s, best-cell GB/s)
    "out/o_proj": (532, 561),
    "q|k|v": (540, 579),
    "qkv|z": (562, 562),
    "gate|up": (536, 561),
    "down": (533, 574),
    "lm_head int4": (571, 576),
}

WALL_GBS = 590.0  # measured DRAM bandwidth, docs/01-hardware.md
L2_TRIPLE = 72 << 20  # gemv_harness.h: one cycle of weight copies must exceed this
DEV = "xpu"


def weight_bytes(k: int, n: int, group: int) -> tuple[int, int]:
    """(qweight bytes, scales bytes) - our accounting, zeros excluded."""
    return (k * n // 2, (k // group) * n * 2)


def nb_copies(qweight_bytes: int) -> int:
    """gemv_harness.h's NB, verbatim."""
    return max(2, L2_TRIPLE // qweight_bytes + 1)


def rand_i32(shape: tuple[int, ...]) -> torch.Tensor:
    """Uniformly random int32 - i.e. uniformly random int4 nibbles. Built from
    random bytes rather than torch.randint so every one of the 32 bits is
    random: the B70 compresses low-entropy fills and would flatter any engine
    reading them (docs/01-hardware.md)."""
    numel = 1
    for s in shape:
        numel *= s
    raw = torch.randint(0, 256, (numel * 4,), dtype=torch.uint8, device=DEV)
    return raw.view(torch.int32).reshape(shape)


# --------------------------------------------------------------------------


def _noop_weight_loader(*args, **kwargs):  # never called: nothing is loaded
    raise AssertionError("weight_loader must not run in this bench")


def build_layer(method, k: int, n: int, act_dtype: torch.dtype):
    """Run vLLM's own create_weights + process_weights_after_loading for one
    shape, with random tensors of exactly the checkpoint's shapes."""
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=k,
        output_partition_sizes=[n],
        input_size=k,
        output_size=n,
        params_dtype=act_dtype,
        weight_loader=_noop_weight_loader,
    )
    layer.qweight.data = rand_i32(tuple(layer.qweight.shape))
    layer.qzeros.data = rand_i32(tuple(layer.qzeros.shape))
    layer.scales.data = (
        torch.rand(tuple(layer.scales.shape), dtype=torch.float32, device=DEV) * 0.01
        + 1e-3
    ).to(act_dtype)
    layer.g_idx.data = torch.zeros(k, dtype=torch.int32, device=DEV)
    torch.xpu.synchronize()
    method.process_weights_after_loading(layer)
    torch.xpu.synchronize()
    return layer


def time_shape(call, launches: int, replays: int = 8, drop: int = 3):
    """8 replays, drop the first `drop`, median of the rest. Returns
    (us/call, enqueue us/call) - the second is the host-bound control."""
    per, enq = [], []
    for rep in range(replays):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for i in range(launches):
            call(i)
        t1 = time.perf_counter()
        torch.xpu.synchronize()
        t2 = time.perf_counter()
        if rep >= drop:
            per.append((t2 - t0) * 1e6 / launches)
            enq.append((t1 - t0) * 1e6 / launches)
    per.sort()
    enq.sort()
    return per[len(per) // 2], enq[len(enq) // 2]


def run_one(method, name, k, n, act_dtype, group, launches, verify=False):
    qw_b, sc_b = weight_bytes(k, n, group)
    total_b = qw_b + sc_b
    nb = nb_copies(qw_b)

    layer = build_layer(method, k, n, act_dtype)
    kernel = method.kernel
    w_q, w_s, w_zp, w_gidx = kernel._get_weight_params(layer)

    # NB copies of the POST-processing tensors, so consecutive calls miss the
    # 24 MB L2. `.t()` is exactly what apply_weights hands the op.
    qcopies = [w_q.data] + [w_q.data.clone() for _ in range(nb - 1)]
    scopies = [w_s.data] + [w_s.data.clone() for _ in range(nb - 1)]
    wqs = [q.t() for q in qcopies]
    torch.xpu.synchronize()

    x = torch.randn(1, k, dtype=act_dtype, device=DEV)
    op = torch.ops._xpu_C.int4_gemm_w4a16

    def call(i: int):
        j = i % nb
        op(x, wqs[j], None, scopies[j], w_zp, group, w_gidx)

    checked = None
    if verify:
        # Prove the direct op call is bit-identical to vLLM's apply path.
        a = method.apply(layer, x)
        b = op(x, wqs[0], None, scopies[0], w_zp, group, w_gidx)
        torch.xpu.synchronize()
        checked = bool(torch.equal(a, b))

    # oneDNN primitive creation / first-call compile is a one-time host cost
    # and is NOT part of any timed replay. Drain it here.
    for i in range(8):
        call(i)
    torch.xpu.synchronize()

    warm_us, _ = time_shape(call, launches)  # discarded pass - the ramp control
    us, enq = time_shape(call, launches)

    del wqs, qcopies, scopies, w_q, w_s, x, layer
    torch.xpu.empty_cache()

    return {
        "name": name,
        "K": k,
        "N": n,
        "act": str(act_dtype).replace("torch.", ""),
        "NB": nb,
        "MB": total_b / (1 << 20),
        "us": us,
        "enqueue_us": enq,
        "gbs": total_b / us * 1e-3,  # bytes/us -> GB/s (1e9)
        "warm_us": warm_us,
        "warm_gbs": total_b / warm_us * 1e-3,
        "apply_matches_op": checked,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True, help="checkpoint snapshot dir (config.json)")
    ap.add_argument("--act", default="float16", choices=["float16", "bfloat16", "both"])
    ap.add_argument("--launches", type=int, default=40)
    ap.add_argument("--warm-seconds", type=float, default=20.0)
    ap.add_argument("--warm-dispatches", type=int, default=100)
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None, help="substring filter on shape name")
    args = ap.parse_args()

    shapes = [s for s in SHAPES if not args.only or args.only in s[0]]

    # ---- provenance -------------------------------------------------------
    import vllm
    from vllm.model_executor.kernels.linear import (
        MPLinearLayerConfig,
        choose_mp_linear_kernel,
    )
    from vllm.model_executor.layers.quantization import get_quantization_config

    print("## environment (measured)")
    print(f"python              {sys.version.split()[0]}")
    print(f"torch               {torch.__version__}")
    print(f"vllm                {vllm.__version__}")
    props = torch.xpu.get_device_properties(0)
    print(f"device              {props.name}")
    print(f"driver              {getattr(props, 'driver_version', '?')}")
    print(f"total memory        {props.total_memory / (1 << 20):.0f} MB")
    try:
        print(f"oneDNN              {torch.ops._xpu_C.get_onednn_version()}")
    except Exception as e:  # pragma: no cover - provenance only
        print(f"oneDNN              <unavailable: {e}>")

    with open(os.path.join(args.snap, "config.json")) as f:
        hf_cfg = json.load(f)
    qc = hf_cfg["quantization_config"]
    print(f"checkpoint dtype    {hf_cfg.get('dtype') or hf_cfg.get('torch_dtype')}")
    print(
        "quantization_config quant_method={quant_method} bits={bits} "
        "group_size={group_size} sym={sym} desc_act={desc_act}".format(**qc)
    )

    # ---- kernel selection, by vLLM's own registry and chooser -------------
    print("\n## kernel selection (vLLM's own code; log lines verbatim)")
    qcls = get_quantization_config(qc["quant_method"])
    print(f"quant_method {qc['quant_method']!r} -> {qcls.__name__}")
    quant_config = qcls.from_config(qc)
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod

    method = AutoGPTQLinearMethod(quant_config)
    method.input_dtype = None
    group = quant_config.group_size
    print(f"quant_type          {quant_config.quant_type}")
    print(f"group_size          {group}   desc_act {quant_config.desc_act}")

    acts = (
        [torch.float16, torch.bfloat16]
        if args.act == "both"
        else [getattr(torch, args.act)]
    )
    primary = acts[0]

    for act in acts:
        for name, k, n in shapes:
            c = MPLinearLayerConfig(
                full_weight_shape=(k, n),
                partition_weight_shape=(k, n),
                weight_type=quant_config.quant_type,
                act_type=act,
                group_size=group,
                zero_points=False,
                has_g_idx=quant_config.desc_act,
            )
            print(
                f"choose_mp_linear_kernel({name}, {k}x{n}, "
                f"{str(act).replace('torch.', '')}) -> "
                f"{choose_mp_linear_kernel(c).__name__}"
            )

    # ---- device warm-up: the cold-start ramp is worth up to +42.8% --------
    wk, wn = 5120, 14336
    warm_layer = build_layer(method, wk, wn, primary)
    wq, ws, wzp, wgidx = method.kernel._get_weight_params(warm_layer)
    wqt = wq.data.t()
    op = torch.ops._xpu_C.int4_gemm_w4a16
    wx = torch.randn(1, wk, dtype=primary, device=DEV)
    op(wx, wqt, None, ws, wzp, group, wgidx)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    n_warm = 0
    while n_warm < args.warm_dispatches or (
        time.perf_counter() - t0
    ) < args.warm_seconds:
        for _ in range(64):
            op(wx, wqt, None, ws, wzp, group, wgidx)
        torch.xpu.synchronize()
        n_warm += 64
    warm_wall = time.perf_counter() - t0
    del warm_layer, wq, wqt, ws, wx
    torch.xpu.empty_cache()
    print(
        f"\ndevice warm-up      {n_warm} dispatches / {warm_wall:.1f} s of GPU work "
        f"at {wk}x{wn} before the first timed shape"
    )

    print("\n## run")
    rows: list[dict] = []
    for act in acts:
        for i, (name, k, n) in enumerate(shapes):
            r = run_one(method, name, k, n, act, group, args.launches, verify=(i == 0))
            rows.append(r)
            print(
                f"  [{r['act']}] {name:<14} {r['us']:9.1f} us  {r['gbs']:5.0f} GB/s "
                f"(enqueue {r['enqueue_us']:.1f} us/call, NB={r['NB']}"
                + (
                    f", apply==op {r['apply_matches_op']}"
                    if r["apply_matches_op"] is not None
                    else ""
                )
                + ")",
                flush=True,
            )
        # ramp / drift control: the first shape, re-timed after the battery
        name, k, n = shapes[0]
        ctl = run_one(method, name + " (re-timed)", k, n, act, group, args.launches)
        ctl["is_control"] = True
        rows.append(ctl)
        print(
            f"  [{ctl['act']}] {name} re-timed  {ctl['us']:.1f} us  "
            f"{ctl['gbs']:.0f} GB/s",
            flush=True,
        )

    # ---- tables -----------------------------------------------------------
    for act in acts:
        a = str(act).replace("torch.", "")
        sel = [r for r in rows if r["act"] == a and not r.get("is_control")]
        if not sel:
            continue
        print(f"\n## vLLM `int4_gemm_w4a16` (oneDNN) at M=1, activations {a}")
        print(
            "\n| shape | K x N | MB (qweight+scales) | NB | us/call | GB/s | "
            "% of 590 | ours `base` | ours best | theirs / ours `base` |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in sel:
            base, best = OURS[r["name"]]
            print(
                f"| {r['name']} | {r['K']}x{r['N']} | {r['MB']:.2f} | {r['NB']} | "
                f"{r['us']:.1f} | {r['gbs']:.0f} | {r['gbs'] / WALL_GBS * 100:.0f}% | "
                f"{base} | {best} | {r['gbs'] / base:.3f} |"
            )
        mean_theirs = sum(r["gbs"] for r in sel) / len(sel)
        mean_ours = sum(OURS[r["name"]][0] for r in sel) / len(sel)
        print(
            f"\nmean: theirs {mean_theirs:.0f} GB/s, ours `base` {mean_ours:.0f} GB/s, "
            f"ratio {mean_theirs / mean_ours:.3f}"
        )
        print("\n### host-dispatch control (is the row device-bound?)")
        print("\n| shape | us/call | enqueue us/call | enqueue share |")
        print("|---|---|---|---|")
        for r in sel:
            print(
                f"| {r['name']} | {r['us']:.1f} | {r['enqueue_us']:.1f} | "
                f"{r['enqueue_us'] / r['us'] * 100:.0f}% |"
            )
        print("\n### ramp control (discarded warm-up pass beside the recorded pass)")
        print("\n| shape | warm-up (discarded) GB/s | recorded GB/s | ramp |")
        print("|---|---|---|---|")
        for r in sel:
            print(
                f"| {r['name']} | {r['warm_gbs']:.0f} | {r['gbs']:.0f} | "
                f"{(r['gbs'] / r['warm_gbs'] - 1) * 100:+.1f}% |"
            )
        ctls = [r for r in rows if r["act"] == a and r.get("is_control")]
        if ctls:
            first, c = sel[0], ctls[0]
            print(
                f"\ndrift control: {first['name']} {first['gbs']:.0f} GB/s at battery "
                f"start, {c['gbs']:.0f} GB/s at battery end "
                f"({(c['gbs'] / first['gbs'] - 1) * 100:+.2f}%)"
            )

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": rows, "ours": OURS}, f, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
