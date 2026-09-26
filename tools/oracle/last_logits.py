#!/usr/bin/env python3
"""The long-context oracle: last-row logits at a few prefix lengths of one long prompt.

    last_logits.py <snapshot> --prompt <ids-file> --out <out.safetensors>
                   [--at 4096,8192,16384,32704] [--chunk 2048]

dump.py's model (the same dequantised bf16 state dict, `eager` attention with its fp32
softmax, strict load), but built for depth: the prompt is forwarded in chunks of
--chunk ids through ONE cache, nothing is hooked, and only the last row of the chunk
ending at each --at length is kept (logits_to_keep=1). dump.py cannot reach 32k: its
hooks alone would hold 192 x T x 5120 bf16 (64 GB at T = 32704) and its eager scores
24 x T^2 fp32 (103 GB). Here the transient peak is one chunk's eager scores,
24 x 2048 x T x 4 B = 6.4 GB at T = 32704 (derived), beside the ~55 GB model.

Output: `logits` f32 [len(at), vocab] (row i is prompt position at[i] - 1), `at` i32.
Spec 6 K3a (the 2026-09-26 ruling): flash_long_test --oracle reads it.
Cost, measured 2026-09-26 on the box (22 threads): 3799 s to 4096 ids, 11359 s to 8192,
so 32704 is day-class. Run in the oracle container, detached:
    tools/oracle/run_in_container.sh 'python3 tools/oracle/last_logits.py "$SNAP" \
        --prompt tests/golden/prompts/long32k.ids --out /ws/oracle-out-long32k/last_logits.safetensors'
"""
import importlib.util
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

_spec = importlib.util.spec_from_file_location("oracle_dump", os.path.join(_HERE, "dump.py"))
_dump = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dump)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--at", default="4096,8192,16384,32704")
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()
    t0 = time.time()
    print(f"torch intra-op threads: {torch.get_num_threads()}", flush=True)
    with open(args.prompt, encoding="utf-8") as f:
        ids = [int(x) for x in f.read().split()]
    at = sorted(int(x) for x in args.at.split(","))
    if not at or at[-1] > len(ids) or at[0] < 1:
        _dump.die(f"--at {at} is outside the prompt's {len(ids)} ids")
    ids = ids[: at[-1]]
    group_size = _dump.check_quant_config(args.snapshot)

    cfg = AutoConfig.from_pretrained(args.snapshot)
    tc = cfg.get_text_config()
    tc._attn_implementation = "eager"   # dump.py's contract: fp32 softmax
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(tc)
    sd = _dump.build_state_dict(args.snapshot, group_size)
    model.load_state_dict(sd, strict=True, assign=True)
    del sd
    model.model.rotary_emb = type(model.model.rotary_emb)(tc)
    still_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if still_meta:
        _dump.die("meta tensors survived the load: " + ", ".join(still_meta[:20]))
    model.eval()
    print(f"loaded strict, {time.time() - t0:.1f}s", flush=True)

    # chunk boundaries: every --chunk ids, plus every --at length
    cuts = sorted(set(list(range(args.chunk, at[-1], args.chunk)) + at))
    rows, cache, start = [], None, 0
    t_fwd = time.time()
    for end in cuts:
        x = torch.tensor([ids[start:end]], dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        if end in at:
            r = out.logits[0, -1].to(torch.float32).clone()
            if not torch.isfinite(r).all():
                _dump.die(f"non-finite logits at position {end - 1}")
            rows.append(r)
            print(f"  at {end}: argmax {int(torch.argmax(r))}, {time.time() - t_fwd:.0f}s", flush=True)
            # Written after every point: the run is day-class at 32k (measured 11359 s to
            # reach 8192 ids on the box), so an interrupted run keeps what it reached.
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            save_file({"logits": torch.stack(rows).contiguous(),
                       "at": torch.tensor(at[: len(rows)], dtype=torch.int32)},
                      args.out, metadata={"snapshot": os.path.abspath(args.snapshot),
                                          "prompt": args.prompt, "chunk": str(args.chunk),
                                          "attn": "eager"})
        start = end
    print(f"wrote {args.out}: {len(rows)} rows, {time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
