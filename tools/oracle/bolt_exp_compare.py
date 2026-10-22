#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 25b: how far is the loader's `A = -exp(A_log)` (src/common/exp_torch.h, Sleef's expf u10 in plain C++) from the
reference's `torch.exp(A_log.float())` (bolt_mamba_ops.A_from_log)? Measured over every bf16 value in [-7, 9] (A_log spans
-6.81 .. 8.81): the recorded figure (spec 25 s14, docs/13) is 196 of 33,266 differ, max 1 ulp, torch 2.14.1 CPU (MKL).
Spec 25c: also bolt_mamba_ops.exp_engine (the same function in torch fp32 ops - bolt_ref's B70_BOLT_A_EXP=engine switch)
against the C++ column: it must differ nowhere.

    clang++ -std=c++17 -O2 -ffp-contract=off -Isrc tools/oracle/bolt_exp_dump.cc -o /tmp/bolt_exp_dump
    /tmp/bolt_exp_dump /tmp/exp.bin
    # in the oracle image (torch needed; on the Mac under /tmp/b70-cpu-model.lock):
    docker run --rm -v "$PWD":/ws -v /tmp:/scratch -w /ws agnes-ref-img:latest python3 tools/oracle/bolt_exp_compare.py /scratch/exp.bin
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]   # tools/oracle shadows stdlib names
import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("bolt_mamba_ops", os.path.join(_HERE, "bolt_mamba_ops.py"))
ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ops)
a = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, 3)
x = torch.from_numpy(a[:, 0].copy())
t = torch.exp(x).numpy()
sleef, libm = a[:, 1], a[:, 2]
def ulp(u, v):
    return np.abs(u.view(np.int32).astype(np.int64) - v.view(np.int32).astype(np.int64))
for name, u in (("exp_torch (Sleef u10 port)", sleef), ("libm expf", libm)):
    d = ulp(u, t)
    print(f"{name} vs torch.exp: {len(d)} values, {int((d != 0).sum())} differ, max {int(d.max())} ulp, histogram {np.bincount(d)[:4].tolist()}")
port = ops.exp_engine(x).numpy()
d = ulp(port, sleef)
print(f"bolt_mamba_ops.exp_engine (torch fp32 ops) vs the C++ exp_torch: {len(d)} values, {int((d != 0).sum())} differ, "
      f"max {int(d.max())} ulp")
print("torch", torch.__version__, "threads", torch.get_num_threads(), "mkl", torch.backends.mkl.is_available())
# the A path: A = -exp(A_log) on the bf16 values of A_log's range (-6.81 .. 8.81)
m = (a[:, 0] >= -6.81) & (a[:, 0] <= 8.81)
d = ulp(sleef[m], t[m])
print(f"on A_log's range [-6.81, 8.81]: {int(m.sum())} values, {int((d != 0).sum())} differ, max {int(d.max())} ulp")
