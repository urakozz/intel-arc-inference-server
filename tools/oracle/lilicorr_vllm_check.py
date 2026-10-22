#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 26 Decision 1 / plan 26a Task 1 Step 3: dflash_ref.py's LiLiCorr head against vLLM's OWN `LiLiCorrHead`.

    lilicorr_vllm_check.py [--tiny DIR] [--snapshot DIR] [--seed 0] [--vllm-src DIR]   (DRY_RUN=1: where it would run)

vLLM's module (V/models/lilicorr.py at 3901d56800 or later) is instantiated on the CPU in a one-rank gloo process group
(tensor parallel 1, CompilationMode NONE), its weights loaded through its own AutoWeightsLoader from the tiny drafter's
head - UNQUANTISED: every NVFP4 linear as the fp32 product fp32(g x W') (vLLM's ReplicatedLinear / NemotronHMLP with
quant_config None), the BF16 tensors widened - and `materialize_inference_buffers(cpu, float32)` run as its loader
does. The port (dflash_ref.Drafter.lilicorr_lattice) reads the SAME fp32 tensors (no global scale). Both are fed the
same inputs: 2 requests (anchor_valid true / false), the candidates and full-vocabulary log-probs of random logits,
random pass_hidden rows and anchor rows, at 7 slots and at a 3-slot prefix (vLLM's short-draft form: the slot bias
keeps its trained offset) - on the tiny drafter and, when its weights are in the HF cache, on the REAL drafter's head
(the trained weights at the real widths: hidden 1024, 8 heads, factor 1024). Printed: per case the lattice BITWISE, or the max |diff|, how many values differ and the
reason (rounding-order size below 1e-5 relative, else a semantic difference). Exit 0 when every case is bitwise or within 1e-6 (Decision 1's bar), 1
otherwise, 77 (SKIP) when vLLM (with models/lilicorr.py) is not importable here.

Where it runs: vLLM's import needs its full dependency set (pydantic, msgspec, zmq, ...): not in `agnes-ref-img`
(measured 2026-10-23: 16 missing modules, and stubbing pydantic / msgspec breaks vLLM's own classes), so on the box in
the vLLM image (queue row 63's r63.vllmref; `--vllm-src` puts a checkout first on sys.path when the image's vLLM
predates lilicorr.py). If vLLM's platform is CPU and a linear lacks `cpu_linear` (set by process_weights_after_loading
from vLLM's compiled CPU ops), F.linear stands in and the output says so.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import socket  # noqa: E402
import tempfile  # noqa: E402

import torch  # noqa: E402

BAR = 1e-6


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def vllm_status(src: str | None) -> tuple[bool, str]:
    """(importable, what) - vllm with models/lilicorr.py."""
    if src:
        sys.path.insert(0, src)
    try:
        import vllm  # noqa: F401
        from vllm.model_executor.models import lilicorr  # noqa: F401
    except Exception as e:      # ModuleNotFoundError for a dependency, or anything vLLM's import raises
        return False, f"{type(e).__name__}: {e}"
    import vllm
    return True, f"vllm {getattr(vllm, '__version__', '?')} at {os.path.dirname(vllm.__file__)}"


def unquantised_head(d) -> dict[str, torch.Tensor]:
    """The tiny drafter's `lilicorr.*` tensors as fp32, every NVFP4 linear folded once: fp32(g x W')."""
    out = {}
    for n, v in d.w.items():
        if not n.startswith("lilicorr."):
            continue
        g = d.g.get(n)
        out[n] = (v.float() * torch.tensor(g, dtype=torch.float32)) if g is not None else v.float()
    return out


def inputs(d, B: int, S: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    c = d.cfg
    logits = torch.randn(B * S, c.vocab, generator=g) * 2.5
    ref = _load("dflash_ref")
    cand, lp = ref.lilicorr_candidates(logits, c.cand_topk)
    rows = torch.randn(B, S, c.hidden, generator=g)
    rows = rows * torch.rsqrt(rows.pow(2).mean(-1, keepdim=True) + 1e-6)
    anchor = torch.randn(B, c.hidden, generator=g)
    valid = torch.tensor([True, False][:B] + [True] * max(0, B - 2))
    return cand.view(B, S, -1), lp.view(B, S, -1), rows, anchor, valid


def build_vllm_head(cfg, raw_config: dict, weights: dict[str, torch.Tensor]):
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationMode
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.models.lilicorr import LiLiCorrHead
    from vllm.model_executor.models.utils import AutoWeightsLoader
    vc = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    notes = []
    with set_current_vllm_config(vc):
        if not _DIST["ready"]:   # one rank-1 gloo group for the process (a second init would be refused)
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
            init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}",
                                         local_rank=0, backend="gloo")
            initialize_model_parallel(tensor_model_parallel_size=1)
            _DIST["ready"] = True
        torch.set_default_dtype(torch.float32)
        head = LiLiCorrHead(model_hidden_size=cfg.hidden, block_size=cfg.block_size, rms_norm_eps=cfg.eps,
                            config=raw_config["dflash_config"], quant_config=None, prefix="lilicorr")
        loaded = AutoWeightsLoader(head).load_weights((n[len("lilicorr."):], t) for n, t in weights.items())
        want = {n for n, _ in head.named_parameters()}
        missing = sorted(want - set(loaded))
        if missing:
            raise RuntimeError(f"vLLM's head did not load {missing}")
        for m in head.modules():
            if isinstance(m, LinearBase) and isinstance(m.quant_method, UnquantizedLinearMethod):
                try:
                    m.quant_method.process_weights_after_loading(m)
                except Exception as e:      # vLLM's compiled CPU ops absent: recorded below
                    notes.append(f"process_weights_after_loading: {type(e).__name__}")
                if not hasattr(m, "cpu_linear"):
                    m.cpu_linear = lambda x, w, b: torch.nn.functional.linear(x, w, b)
                    if "cpu_linear" not in notes:
                        notes.append("cpu_linear")
        head.materialize_inference_buffers(torch.device("cpu"), torch.float32)
    return head.eval(), vc, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiny", help="a lilicorr_make_tiny.py directory (default: one written to a temp dir)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vllm-src", default=os.environ.get("VLLM_SRC"))
    ap.add_argument("--snapshot", help="the real drafter's snapshot (default: the HF cache's 449d2414 when present)")
    a = ap.parse_args()
    ok, what = vllm_status(a.vllm_src)
    if os.environ.get("DRY_RUN") == "1":
        print(f"DRY_RUN: vLLM {'importable' if ok else 'NOT importable'} here ({what}); "
              + ("this environment runs the check" if ok else "the check runs in the box's vLLM image (r63.vllmref)"))
        return 0
    if not ok:
        print(f"SKIP_REASON lilicorr_vllm_check: vLLM with models/lilicorr.py is not importable here ({what}); "
              "run it in the vLLM image (r63.vllmref)")
        return 77
    print(f"vLLM: {what}")
    torch.set_grad_enabled(False)
    ref = _load("dflash_ref")
    mt = _load("lilicorr_make_tiny")
    rc = 0
    with tempfile.TemporaryDirectory() as tmp:
        tiny = a.tiny or mt.make(os.path.join(tmp, "lil"), seed=a.seed, vocab=4096)
        rc |= check_one(ref, tiny, "the tiny drafter (lilicorr_make_tiny.py)", a.seed)
    snap = a.snapshot or ref.find_snapshot("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr",
                                           ("config.json", "model.safetensors"))
    if snap:
        rc |= check_one(ref, snap, f"the real drafter's head ({snap})", a.seed)
    else:
        print("the real drafter is not in the HF cache: its head not checked (the tiny one is)")
    return rc


_DIST = {"ready": False}


def check_one(ref, d_dir: str, tag: str, seed: int) -> int:
    """The port's lattice against vLLM's on one drafter's head (unquantised fp32 weights on both sides), S 7 and 3."""
    with open(os.path.join(d_dir, "config.json")) as f:
        raw_config = json.load(f)
    cfg = ref.read_config(d_dir)
    t = ref.read_drafter_tensors(d_dir, cfg)
    g = torch.Generator().manual_seed(7)
    emb = (torch.randn(cfg.vocab, cfg.hidden, generator=g, dtype=torch.bfloat16) * 0.02).to(torch.bfloat16)
    head = torch.zeros(1, cfg.hidden, dtype=torch.bfloat16).expand(cfg.vocab, cfg.hidden)   # never read here
    quant = ref.Drafter(cfg, t, emb, head)
    w = unquantised_head(quant)
    plain = {n: v for n, v in quant.w.items() if not n.startswith("lilicorr.")}
    plain.update(w)
    port = ref.Drafter(cfg, plain, emb, head)            # no global scale anywhere: the same fp32 weights as vLLM's
    assert not any(n.startswith("lilicorr.") for n in port.g)
    vh, _vc, notes = build_vllm_head(cfg, raw_config, w)
    print(f"== {tag}: head hidden {cfg.lil_hidden}, {cfg.lil_layers} layers, {cfg.lil_heads} heads, top {cfg.cand_topk}")
    if notes:
        print(f"note: {sorted(set(notes))} - where vLLM's platform has no compiled cpu_linear, F.linear stands in for the "
              "head's linears (vLLM's non-CPU default_unquantized_gemm is F.linear too)")
    worst = 0.0
    rc = 0
    for S in (cfg.block_size - 1, 3):
        cand, lp, rows, anchor, valid = inputs(port, 2, S, seed + S)
        mine = port.lilicorr_lattice(cand, lp, rows, anchor, valid)
        theirs = vh(emb[cand].float(), lp, rows, anchor, valid).float()
        if mine.shape != theirs.shape:
            print(f"S {S}: SHAPE {tuple(mine.shape)} vs vLLM {tuple(theirs.shape)}")
            return 1
        if torch.equal(mine, theirs):
            print(f"S {S}: the lattice [{', '.join(str(x) for x in mine.shape)}] BITWISE vLLM's (anchor_valid true and "
                  "false)")
            continue
        diff = (mine - theirs).abs().max().item()
        worst = max(worst, diff)
        rel = diff / max(theirs.abs().max().item(), 1e-30)
        n = int((mine != theirs).sum())
        why = (f"{n} of {mine.numel()} values differ, {rel:.3g} relative: "
               + ("fp32 rounding-order size (a GEMM / SDPA reduction blocked differently)" if rel < 1e-5 else
                  "beyond rounding - a semantic difference (a misread)"))
        verdict = "within 1e-6" if diff <= BAR else "FAIL"
        print(f"S {S}: max |diff| {diff:.3g} ({verdict}); {why}")
        if diff > BAR:
            rc = 1
    print(("OK" if rc == 0 else "FAIL") + f": {tag}: dflash_ref's LiLiCorr head against vLLM's LiLiCorrHead, worst "
          f"{worst:.3g}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
