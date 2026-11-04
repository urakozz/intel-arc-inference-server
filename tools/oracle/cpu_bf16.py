"""Exact-bf16 GEMMs for the CPU oracles on a CPU without bf16 units (shared by the *_ref.py tools).

The box's Xeon E5-2696 v4 has AVX2 only: no AVX-512, no bf16. In the reference image (torch 2.16.0.dev+xpu) a bf16
CPU matmul of 2048 x 2560 x 2560 takes 44.6 s on ONE core (the emulated path) against 38 ms in fp32 on 19 cores;
torch._grouped_mm in bf16 takes 32 s on one core against 40 ms in fp32 (operator, 2026-11-02). A 32k-id golden that
should take two hours took more than eight.

torch's CPU bf16 GEMM accumulates in fp32 and rounds the result once to bf16. So does `fp32 GEMM of the bf16 values,
rounded once to bf16`: the products of two bf16 values are exact in fp32, so the two differ only in the summation
order of the fp32 accumulation (an ulp of bf16 in a few elements; the verification is in the commit message).

    with cpu_bf16.exact_bf16_gemm():     # or cpu_bf16.install(True) once, in main()
        ...                              # F.linear / torch.matmul / @ / bmm / addmm / grouped_mm on bf16 CPU tensors

NOT bitwise torch's emulated GEMM: the summation order differs, and on a random synthetic model one ulp at layer 0 flips
routes and tokens within a few steps (Kolibri synth, 4 sets: tokens differ at step 0 / 1 / 5 / 12). Operator ruling
2026-11-03: the shim is for NEW references judged against f32. Existing goldens (made with torch's own bf16 GEMM) stay as
they are - they are reproducible with --no-exact-bf16-gemm. Check (teacher-forced on the stored ids, vs the f32 judge,
median cosine distance of the logits rows, shim / torch-bf16): 0.915, 0.974, 0.904, 1.145 (means 0.91, 0.98, 0.96, 1.07) -
the shim is as close to f32 as torch's bf16 within noise.

Only bf16 CPU tensors are touched: float32 runs (the judges), every other dtype and every non-CPU tensor go to torch's
own functions unchanged. The tools expose it as --exact-bf16-gemm / --no-exact-bf16-gemm (default on)."""
from __future__ import annotations

import argparse
import contextlib

import torch
import torch.nn.functional as F

_BF16 = torch.bfloat16
_installed: list = []   # [(owner, name, original)] while a patch is active


def _cpu_bf16(*ts) -> bool:
    return all(isinstance(t, torch.Tensor) and t.dtype == _BF16 and t.device.type == "cpu" for t in ts)


def add_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exact-bf16-gemm", action=argparse.BooleanOptionalAction, default=True,
                        help="bf16 GEMMs on the CPU as fp32 GEMMs rounded once to bf16 (torch's own accumulation "
                             "up to summation order; the CPU has no bf16 units, the emulated GEMM is ~1000x slower)")


def _wrap(orig, nargs_tensors, convert):
    def f(*a, **k):
        ts = [x for x in a[:nargs_tensors]]
        if _cpu_bf16(*ts):
            return convert(orig, a, k)
        return orig(*a, **k)
    f.__wrapped__ = orig
    return f


def _linear(orig, a, k):
    x, w = a[0], a[1]
    b = a[2] if len(a) > 2 else k.get("bias")
    return orig(x.float(), w.float(), None if b is None else b.float()).to(_BF16)


def _two(orig, a, k):   # matmul / bmm / mm: out= is not supported (never used by the oracles)
    return orig(a[0].float(), a[1].float(), *a[2:], **k).to(_BF16)


def _addmm_like(orig, a, k):   # addmm / baddbmm(input, m1, m2, *, beta, alpha)
    return orig(a[0].float(), a[1].float(), a[2].float(), *a[3:], **k).to(_BF16)


def _grouped(orig, a, k):   # grouped_mm(mat_a, mat_b, offs=..., bias=...)
    a = [x.float() if isinstance(x, torch.Tensor) and x.dtype == _BF16 else x for x in a]
    k = {n: (v.float() if isinstance(v, torch.Tensor) and v.dtype == _BF16 else v) for n, v in k.items()}
    return orig(*a, **k).to(_BF16)


def install(enabled: bool = True) -> None:
    """Patch torch's entry points (idempotent); `enabled` False is a no-op, so main() can pass the flag straight."""
    if not enabled or _installed:
        return
    plan = [
        (F, "linear", 2, _linear),
        (torch, "matmul", 2, _two), (torch, "bmm", 2, _two), (torch, "mm", 2, _two),
        (torch.Tensor, "matmul", 2, _two), (torch.Tensor, "__matmul__", 2, _two), (torch.Tensor, "bmm", 2, _two),
        (torch.Tensor, "mm", 2, _two),
        (torch, "addmm", 3, _addmm_like), (torch, "baddbmm", 3, _addmm_like),
        (torch, "_grouped_mm", 2, _grouped), (F, "grouped_mm", 2, _grouped),
    ]
    for owner, name, n, conv in plan:
        orig = getattr(owner, name, None)
        if orig is None:
            continue
        _installed.append((owner, name, orig))
        setattr(owner, name, _wrap(orig, n, conv))


def uninstall() -> None:
    while _installed:
        owner, name, orig = _installed.pop()
        setattr(owner, name, orig)


@contextlib.contextmanager
def exact_bf16_gemm(enabled: bool = True):
    was = bool(_installed)
    install(enabled)
    try:
        yield
    finally:
        if not was:
            uninstall()
