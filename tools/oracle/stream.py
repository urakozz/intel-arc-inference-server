#!/usr/bin/env python3
"""Layer-streamed weights for the CPU oracle (spec 14, the Mac path).

Agnes 3.0 Flash dequantised to bf16 is ~62 GB; a Mac's Docker VM is 34-64 GB.
So the model is built on `meta` exactly as dump.py builds it, the small
non-layer tensors (embedding, final norm, lm_head) are made resident, and each
decoder layer's weights are materialised by a forward PRE-hook and dropped back
to `meta` by a forward hook. The model's own `forward` runs unchanged - masks,
rotary, the cache, the decode loop - so a streamed run computes what a resident
run computes (same modules, same dequantised bf16 tensors, same op order); only
where the weights live between layers differs. A background thread dequantises
the next layer while the current one runs (prefetch depth 1, two layers resident
at most, ~1.7 GB on Agnes), and `keep` names layers that stay resident once
loaded, as RAM allows.

The cost is one dequant of every streamed layer per forward: a prompt forward
pays it once, a greedy decode once per generated token.

    attach(model, layers, layer_sd, resident_sd)   # the library entry
    checkpoint_reader(snapshot)                    # name -> tensor over the index's shards

`layer_sd(i)` returns layer i's state dict with keys relative to the layer
(strict load); `resident_sd` the remaining tensors by full model name.
"""
import json
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

_spec = importlib.util.spec_from_file_location("stream_dequant", os.path.join(_HERE, "dequant.py"))
_dequant = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dequant)


_CPP = r"""
#include <torch/extension.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
// dequant.py's rule, written transposed for nn.Linear: out[n][k] = bf16_rne(float(q(k, n) - 8)
// * float(scales[k / g][n])), q(k, n) = nibble k % 8 (low first) of qweight[k / 8][n]. The
// product is exact in fp32 (4-bit integer times an f16) and c10::BFloat16(float) rounds to
// nearest even - the same single rounding as dequant_gptq's .to(bfloat16). Per group and
// column the 16 possible values are rounded once into a table. Tiles of 16 columns keep
// the qweight reads on whole cache lines.
torch::Tensor dequant_t(torch::Tensor qw, torch::Tensor sc, int64_t g) {
  const int64_t r = qw.size(0), n = qw.size(1), k = r * 8, G = sc.size(0);
  auto out = torch::empty({n, k}, torch::dtype(torch::kBFloat16));
  const int32_t* q = qw.data_ptr<int32_t>();
  const c10::Half* s = reinterpret_cast<const c10::Half*>(sc.data_ptr<at::Half>());
  c10::BFloat16* o = reinterpret_cast<c10::BFloat16*>(out.data_ptr<at::BFloat16>());
  const int64_t TN = 16;
  at::parallel_for(0, (n + TN - 1) / TN, 1, [&](int64_t a, int64_t b) {
    c10::BFloat16 lut[16][16];
    for (int64_t tb = a; tb < b; ++tb) {
      const int64_t n0 = tb * TN, n1 = std::min(n, n0 + TN);
      for (int64_t grp = 0; grp < G; ++grp) {
        for (int64_t c = n0; c < n1; ++c) {
          const float sv = float(s[grp * n + c]);
          for (int v = 0; v < 16; ++v) lut[c - n0][v] = c10::BFloat16(float(v - 8) * sv);
        }
        for (int64_t j = grp * g / 8; j < (grp + 1) * g / 8; ++j) {
          const int32_t* qj = q + j * n;
          for (int64_t c = n0; c < n1; ++c) {
            const uint32_t w = uint32_t(qj[c]);
            c10::BFloat16* oc = o + c * k + j * 8;
            const c10::BFloat16* L = lut[c - n0];
            for (int t = 0; t < 8; ++t) oc[t] = L[(w >> (4 * t)) & 0xF];
          }
        }
      }
    }
  });
  return out;
}
"""
_ext = None


def dequant_t(qweight: torch.Tensor, scales: torch.Tensor, group_size: int = 64,
              check: bool = True) -> torch.Tensor:
    """dequant_gptq(qweight, scales).t().contiguous(), bit-identical, ~15x faster (C++, OpenMP).

    Built on first use with torch.utils.cpp_extension (needs g++ and ninja). With
    `check`, 64 consecutive columns at a random offset are re-done by dequant.py's
    reference and must be equal (columns are independent, so a column slice is a full
    check of those columns); dump.convert checks every tensor the first time it is seen."""
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline
        _ext = load_inline("oracle_dequant_t3", cpp_sources=_CPP, functions=["dequant_t"],
                           extra_cflags=["-O3", "-fopenmp"], extra_ldflags=["-fopenmp"])
    assert qweight.dtype == torch.int32 and scales.dtype == torch.float16
    assert group_size % 8 == 0 and scales.shape[0] * group_size == qweight.shape[0] * 8
    out = _ext.dequant_t(qweight.contiguous(), scales.contiguous(), group_size)
    if not check:
        return out
    w = min(64, qweight.shape[1])
    c0 = int(torch.randint(0, qweight.shape[1] - w + 1, ()))
    want = _dequant.dequant_gptq(qweight[:, c0:c0 + w].contiguous(), scales[:, c0:c0 + w].contiguous(),
                                 group_size)
    if not torch.equal(out[c0:c0 + w], want.t()):
        raise RuntimeError("dequant_t disagrees with dequant_gptq")
    return out


def conv1d_single_thread() -> None:
    """Run grouped bf16 conv1d (the GDN's causal conv) on one thread.

    On the Mac's Docker VM (i9-9980HK, AVX2, 16 vCPUs, torch 2.14 CPU) a depthwise
    bf16 F.conv1d is pathologically slow multi-threaded: 512 channels x 7 positions
    takes 28 s on 16 threads and 8 ms on one; Agnes's 10240-channel conv did not finish
    in 10 minutes. Outputs are bitwise equal across thread counts (checked at 256 x 42,
    256 x 1, 512 x 7) - each output element is one channel's 4-tap dot product - so this
    changes time only. Patched on torch.nn.functional, which modeling_qwen3_5.py and
    modeling_agnes.py both call through `F`."""
    import torch.nn.functional as F
    if getattr(F.conv1d, "_single_thread", False):
        return
    orig = F.conv1d

    def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if groups > 1 and input.dtype == torch.bfloat16:
            n = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                return orig(input, weight, bias, stride, padding, dilation, groups)
            finally:
                torch.set_num_threads(n)
        return orig(input, weight, bias, stride, padding, dilation, groups)
    conv1d._single_thread = True
    F.conv1d = conv1d


def checkpoint_reader(snapshot: str):
    """(keys, get) over the shards model.safetensors.index.json names."""
    with open(os.path.join(snapshot, "model.safetensors.index.json"), encoding="utf-8") as f:
        wm = json.load(f)["weight_map"]
    handles = {fn: safe_open(os.path.join(snapshot, fn), framework="pt", device="cpu")
               for fn in sorted(set(wm.values()))}
    return sorted(wm), lambda k: handles[wm[k]].get_tensor(k)


class _Prefetch:
    """Layer i's state dict, built on a worker thread ahead of use.

    `keep`: layers whose weights stay materialised after their first forward (RAM
    permitting, they need no dequant again); the prefetch skips them."""

    def __init__(self, layer_sd, n: int, keep=()):
        self.layer_sd, self.n, self.keep = layer_sd, n, set(keep)
        self.lock = threading.Lock()
        self.jobs: dict[int, tuple[threading.Thread, dict]] = {}
        self.loaded: set[int] = set()   # kept layers already materialised
        self.seconds = 0.0   # time the forward thread spent WAITING on dequant

    def _start(self, i: int) -> None:
        if i in self.jobs or not 0 <= i < self.n:
            return
        box: dict = {}

        def run():
            box["sd"] = self.layer_sd(i)
        th = threading.Thread(target=run, daemon=True)
        th.start()
        self.jobs[i] = (th, box)

    def _next(self, i: int) -> int:
        """The next layer after i (wrapping into the next forward) that needs a dequant."""
        for d in range(1, self.n + 1):
            j = (i + d) % self.n
            if j not in self.loaded:
                return j
        return -1

    def get(self, i: int) -> dict:
        with self.lock:
            self._start(i)
            th, box = self.jobs.pop(i)
        t = time.time()
        th.join()
        self.seconds += time.time() - t
        if i in self.keep:
            self.loaded.add(i)
        with self.lock:
            self._start(self._next(i))
        if "sd" not in box:
            raise RuntimeError(f"layer {i}: dequant thread failed")
        return box["sd"]


def attach(model, layers, layer_sd, resident_sd: dict, rebuild=(), keep=()):
    """Make `model` (built on meta) runnable with streamed layers.

    resident_sd: every tensor outside `layers`, by full name (loaded strict over
    the non-layer part). rebuild: (parent, attr, factory) for non-persistent
    buffers meta construction left empty (rotary inv_freq). keep: layer indices whose
    weights stay resident once loaded (~0.83 GB each on Agnes). Returns the prefetcher
    (its `.seconds` is the dequant wait).
    """
    layer_ids = {id(m) for m in layers}
    prefixes = []
    for name, mod in model.named_modules():
        if id(mod) in layer_ids:
            prefixes.append(name + ".")
    want = {k for k in model.state_dict() if not k.startswith(tuple(prefixes))}
    missing, unexpected = sorted(want - set(resident_sd)), sorted(set(resident_sd) - want)
    if missing or unexpected:
        raise RuntimeError(f"resident tensors: missing {missing[:5]}, unexpected {unexpected[:5]}")
    model.load_state_dict(resident_sd, strict=False, assign=True)
    conv1d_single_thread()
    for parent, attr, factory in rebuild:
        setattr(parent, attr, factory())
    pf = _Prefetch(layer_sd, len(layers), keep)
    # Hold mode (layer-major batches, oracle_generate.py --batch): with pf.hold set, a layer stays
    # materialised after its forward, so the next sequence's call of the SAME layer runs on the same
    # tensors without a second dequant; pf.release() drops it once every sequence has passed.
    pf.hold, pf.held = False, None

    def release():
        if pf.held is not None and pf.held not in pf.keep:
            layers[pf.held].to_empty(device="meta")
        pf.held = None
    pf.release = release

    def pre(i):
        def fn(mod, _args, _kwargs=None):
            if i in pf.loaded or pf.held == i:
                return
            release()
            sd = pf.get(i)
            mod.load_state_dict(sd, strict=True, assign=True)
            if pf.hold:
                pf.held = i
        return fn

    def post(i):
        def fn(mod, _args, _out):
            if i not in pf.keep and pf.held != i:
                mod.to_empty(device="meta")
        return fn

    for i, layer in enumerate(layers):
        layer.register_forward_pre_hook(pre(i))
        layer.register_forward_hook(post(i))
    return pf
