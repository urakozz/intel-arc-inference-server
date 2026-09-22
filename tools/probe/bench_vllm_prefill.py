"""Matched pp4096 reference; run inside the existing vLLM image.

No HTTP or tokenization is timed. Each request has exactly one generated token,
so generate() wall includes prefill, sampling and engine scheduling. Prefix
caching and speculation are disabled. Run GPU profiling separately from timing.
"""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import struct
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    import torch
    import vllm_xpu_kernels._xpu_C  # Register native ops, including version query.
    from vllm import LLM, SamplingParams

    seed = [int(x) for x in (Path(__file__).resolve().parents[2] /
                             "tests/golden/prompts/prose.ids").read_text().split()]
    ids = [seed[i % len(seed)] for i in range(4096)]
    digest = hashlib.sha256(struct.pack("<4096I", *ids)).hexdigest()
    versions = {name: importlib.metadata.version(name)
                for name in ("vllm", "torch", "vllm-xpu-kernels")}
    versions["onednn"] = torch.ops._xpu_C.get_onednn_version()
    print(json.dumps({"versions": versions, "prompt_ids": len(ids),
                      "prompt_u32le_sha256": digest, "model": args.model}), flush=True)

    kwargs = dict(
        model=args.model, tokenizer=args.model, dtype="bfloat16",
        max_model_len=16384, max_num_seqs=1, max_num_batched_tokens=2048,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        kv_cache_dtype="auto", mamba_ssm_cache_dtype="float32",
        tensor_parallel_size=1, pipeline_parallel_size=1,
        gpu_memory_utilization=0.90, trust_remote_code=True,
        language_model_only=True,
    )
    if args.profile:
        kwargs["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": "/results"}
    print(json.dumps({"engine_args": kwargs, "speculation": False}), flush=True)
    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False)
    timings = []
    tokens = []
    for i in range(args.runs + 1):
        if args.profile and i == 1:
            llm.start_profile()
        begin = time.perf_counter()
        output = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
        elapsed = (time.perf_counter() - begin) * 1000
        if args.profile and i == 1:
            llm.stop_profile()
        generated = list(output.outputs[0].token_ids)
        if len(generated) != 1 or output.prompt_token_ids != ids:
            raise RuntimeError("reference did not process the requested prompt/output length")
        if output.num_cached_tokens:
            raise RuntimeError("reference unexpectedly reused cached prompt tokens")
        tokens.append(generated)
        print(json.dumps({"run": i, "cold_request": i == 0,
                          "instrumented": args.profile and i == 1,
                          "wall_ms": elapsed, "tokens_per_second": 4096000 / elapsed,
                          "generated": generated, "cached_tokens": output.num_cached_tokens}),
              flush=True)
        if i:
            timings.append(elapsed)
    if any(t != tokens[0] for t in tokens):
        raise RuntimeError("reference first token changed across identical requests")
    print(json.dumps({"median_ms": statistics.median(timings),
                      "median_tokens_per_second": 4096000 / statistics.median(timings),
                      "scope": "generate wall, one output token, loader excluded",
                      "grade": "diagnostic" if args.profile else "paired reference"}), flush=True)


if __name__ == "__main__":
    main()
