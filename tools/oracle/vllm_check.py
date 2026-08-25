#!/usr/bin/env python3
"""The third implementation: vLLM's own greedy continuation, on the same bits.

The golden gate (docs/14) proves engine == oracle. It cannot prove that the
checkpoint was unpacked the way its author packed it: the C++ loader and the
oracle are pinned to the SAME `dequant.py` convention (bit-exactly, by
`dequant_fixture_test`), so a wrong convention moves both together and the gate
still passes. Closing that hole needs a third implementation sharing nothing
with either - vLLM's own GPTQ kernels, its own loader, its own attention.

This script runs that cross-check. It feeds vLLM the three committed prompt id
files as `prompt_token_ids` (no tokenizer in the loop, so a tokenizer
difference cannot show up as a token difference), greedy-decodes 32 tokens with
`ignore_eos=True` (the oracle's `dump.py` is a bare argmax loop with no stop
condition - matching it exactly is what makes position j comparable), reads the
oracle's golden `tokens` tensor straight out of the `.safetensors` files, and
prints the three-way comparison.

    vllm_check.py <snapshot> --prompts tests/golden/prompts --golden oracle-out
                  [--gen 32] [--max-model-len 16384] [--enforce-eager]

Runs INSIDE the reference container, on the box (it needs the XPU). The exact
`docker run` is in docs/14 - the `-u $(id -u):$(id -g)` is not optional, or the
outputs come back root-owned.

Reading the verdict (docs/14 §cross-check, `tools/oracle/README.md` §trust
chain):

    all three agree                    -> the chain is closed.
    engine == oracle, both != vLLM     -> suspect `dequant.py` FIRST: zero point
                                          (`q - 8` vs an explicit `qzeros`), the
                                          group axis, the `[K/8, N]` packing and
                                          nibble order, `desc_act`/`g_idx`.
                                          Do not touch a kernel first.
    engine != oracle                   -> an engine bug; but the gate already
                                          rules that out for these prompts.

Exit status: 0 if every prompt matched all `--gen` ids, 1 otherwise. A
divergence is a result, not a crash - everything is printed either way.
"""
import os
import sys

# `tokenize.py` next to this script shadows the stdlib `tokenize` that `inspect`
# imports, and torch/vllm import `inspect`. Drop this directory from sys.path
# BEFORE importing anything third-party (see tools/oracle/README.md).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

PROMPTS = ("prose", "code", "cjk")


def die(msg: str) -> None:
    print(f"vllm_check: FATAL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def read_ids(path: str) -> list[int]:
    """Whitespace-separated decimal ids - the committed .ids format."""
    with open(path) as f:
        text = f.read()
    try:
        ids = [int(t) for t in text.split()]
    except ValueError as e:
        die(f"{path}: not whitespace-separated ints ({e})")
    if not ids:
        die(f"{path}: empty")
    return ids


def read_golden_tokens(path: str) -> list[int]:
    """The oracle's greedy continuation, read from the golden file itself."""
    from safetensors import safe_open

    with safe_open(path, framework="numpy") as f:
        if "tokens" not in f.keys():
            die(f"{path}: no `tokens` tensor")
        t = f.get_tensor("tokens")
    if t.ndim != 1:
        die(f"{path}: `tokens` has shape {t.shape}, expected 1-D")
    return [int(x) for x in t]


def fmt_ids(ids) -> str:
    """16 ids per line - the shape docs/14 records the CLI cross-check in."""
    out = []
    for i in range(0, len(ids), 16):
        out.append(" ".join(str(int(x)) for x in ids[i : i + 16]))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", help="resolved HF snapshot directory of the checkpoint")
    ap.add_argument("--prompts", default="tests/golden/prompts", help="dir holding <p>.ids")
    ap.add_argument("--golden", default="oracle-out", help="dir holding <p>.golden.safetensors")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument(
        "--enforce-eager",
        action="store_true",
        help="skip torch.compile / xpu graph capture (fallback path, and the "
        "control run if the compiled path is ever suspected)",
    )
    args = ap.parse_args()

    # Read every input BEFORE the 27B load, so a typo costs seconds not minutes.
    prompts: dict[str, list[int]] = {}
    golden: dict[str, list[int]] = {}
    for p in PROMPTS:
        ids_path = os.path.join(args.prompts, f"{p}.ids")
        gold_path = os.path.join(args.golden, f"{p}.golden.safetensors")
        if not os.path.exists(ids_path):
            die(f"missing {ids_path}")
        if not os.path.exists(gold_path):
            die(f"missing {gold_path} (goldens live on the box only)")
        prompts[p] = read_ids(ids_path)
        golden[p] = read_golden_tokens(gold_path)
        if len(golden[p]) != args.gen:
            die(f"{gold_path}: `tokens` is {len(golden[p])} long, --gen is {args.gen}")

    cfg_path = os.path.join(args.snapshot, "config.json")
    with open(cfg_path, "rb") as f:
        cfg_bytes = f.read()
    cfg = json.loads(cfg_bytes)
    qc = cfg.get("quantization_config", {})

    import torch  # noqa: E402
    import vllm  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402

    print("=== vllm_check: the third implementation ===")
    print(f"  vllm        {vllm.__version__}")
    print(f"  torch       {torch.__version__}")
    print(f"  snapshot    {os.path.realpath(args.snapshot)}")
    print(f"  config.json sha256={hashlib.sha256(cfg_bytes).hexdigest()} ({len(cfg_bytes)} B)")
    print(
        "  quant       "
        + " ".join(
            f"{k}={qc.get(k)!r}"
            for k in ("quant_method", "bits", "group_size", "sym", "desc_act", "provider")
        )
    )
    for p in PROMPTS:
        print(f"  prompt {p:5s} {len(prompts[p])} ids")
    print(f"  gen         {args.gen}   enforce_eager={args.enforce_eager}")
    print(flush=True)

    llm_kwargs = dict(
        model=args.snapshot,
        trust_remote_code=True,
        dtype="auto",
        kv_cache_dtype="auto",
        quantization=None,  # auto-detect from the checkpoint's own config
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        # A correctness check has no business caching a prefix or batching two
        # prompts into one GEMM: both change which kernel shape runs.
        enable_prefix_caching=False,
        language_model_only=True,  # text-only, as the engine is (docs/03)
        seed=0,
    )
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    else:
        # The one recorded working override for this stack (docs/BENCHMARKS.md).
        llm_kwargs["compilation_config"] = {"inductor_compile_config": {"pre_grad_fusion_options": {}}}

    t0 = time.time()
    llm = LLM(**llm_kwargs)
    print(f"\n[load] {time.time() - t0:.1f}s\n", flush=True)

    # temperature=0 is greedy; ignore_eos matches dump.py's bare argmax loop,
    # which has no stop condition, so all `gen` positions stay comparable.
    sp = SamplingParams(temperature=0.0, max_tokens=args.gen, min_tokens=args.gen, ignore_eos=True)
    print(f"[sampling] {sp}\n", flush=True)

    got: dict[str, list[int]] = {}
    for p in PROMPTS:
        t = time.time()
        outs = llm.generate([{"prompt_token_ids": prompts[p]}], sp)
        ids = [int(x) for x in outs[0].outputs[0].token_ids]
        got[p] = ids
        print(f"=== {p}: {len(ids)} ids in {time.time() - t:.1f}s")
        print(fmt_ids(ids))
        text = outs[0].outputs[0].text
        if text:
            print(f"  text: {text!r}")
        print(flush=True)

    # ---- the three-way comparison -------------------------------------------
    # engine == oracle is already proven element-exact by the golden gate
    # (docs/14, 96/96), so `golden` stands for both of them here.
    print("\n=== three-way comparison (engine == oracle by the golden gate, docs/14)")
    all_match = True
    for p in PROMPTS:
        g, v = golden[p], got[p]
        n = min(len(g), len(v))
        first = next((i for i in range(n) if g[i] != v[i]), None)
        if len(v) != len(g):
            first = n if first is None else first
        same = sum(1 for i in range(n) if g[i] == v[i])
        print(f"\n--- {p}  ({len(prompts[p])} prompt ids)")
        print(f"  oracle/engine ({len(g)} ids):")
        print("    " + fmt_ids(g).replace("\n", "\n    "))
        print(f"  vllm          ({len(v)} ids):")
        print("    " + fmt_ids(v).replace("\n", "\n    "))
        if first is None and len(v) == len(g):
            print(f"  MATCH {same}/{len(g)}")
        else:
            all_match = False
            print(f"  DIVERGE  matching-position count {same}/{len(g)}")
            if first is not None and first < n:
                print(f"  first divergence at position {first}: vllm={v[first]} golden={g[first]}")
            else:
                print(f"  length mismatch: vllm {len(v)} ids vs golden {len(g)}")
            print(
                "  per-position diff: "
                + " ".join(f"[{i}]{v[i]}!={g[i]}" for i in range(n) if v[i] != g[i])
            )

    total_ok = sum(
        sum(1 for a, b in zip(golden[p], got[p]) if a == b) for p in PROMPTS
    )
    total = sum(len(golden[p]) for p in PROMPTS)
    print(f"\n=== VERDICT: {total_ok}/{total} ids equal across {len(PROMPTS)} prompts")
    if all_match:
        print("=== ALL THREE AGREE - the trust chain is closed (docs/14 §cross-check).")
        return 0
    print(
        "=== vLLM DIVERGES from engine==oracle. docs/14's ruling: suspect\n"
        "=== tools/oracle/dequant.py FIRST (zero point, group axis, [K/8,N]\n"
        "=== packing and nibble order, desc_act/g_idx). Do not touch a kernel."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
