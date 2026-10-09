#!/usr/bin/env python3
"""Spec 21e Task 3: the M1 data for qwen4exp_mtp_test - 21a's MTP port (qwen4exp_mtp.py MtpHead, vLLM's form) on the
engine-format head, both pre_fc_norm_hidden forms.

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/qwen4exp_mtp_fixture.py <checkpoint> <out dir>
        [--samples 8] [--ids prompt.ids] [--R model|random] [--layers N] [--ple int8:<dir>|bf16] [--seed 7]

Per sample i: R_i (bf16 [10240], the main model's pre-mixer 4-stream H of prompt row i under --R model - the
reference's own forward of the first `samples` prompt ids, layer-streamed - or a seeded normal draw under --R
random) and t_i (the next prompt id, the head's step-0 token), then chain(R_i, [t_{i-1}, t_i], k=3, share_sel=True):
the head's steps at positions 0, 1, 2 on an empty head cache - step 0 on (R_i, t_i), steps 1 and 2 on their own
pre-mixer H and the previous step's argmax, attending step 0's selection (decision 5, the ruled form) - exactly what
the engine runs after one prefilled id with R_0 replaced (Qwen4ExpEngine::write_mtp_R) and draft(3).

The head is the ENGINE's: its routed experts are the checkpoint's bf16 experts RTN-quantised to int4 g64 at load
(loader::rtn_int4_g64, spec 15e's rule) - here quantised and dequantised the same way (ornith_ref.rtn_int4_g64's
formula: per output row and 64-group, s16 = f16(2 amax / 15), q = clamp(rint(w / s16) + 8, 0, 15)); its dense and
shared arms stay bf16 (loader::q4_mtp_desc).

Writes <out dir>/mtp_single.safetensors and mtp_per_stream.safetensors:
  R       BF16 [S][10240]     the head's step-0 R
  tokens  I32  [S]            t_i
  logits  F32  [S][3][V]      each step's logits (the model dtype's values widened)
  draft   I32  [S][3]         each step's argmax (the reference's chain)
  M       BF16 [S][10240]     the last step's pre-mixer H
Run in agnes-ref-img with the 5.19.0 site (memory capped: the head's layer is resident, its experts lazy).
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ref = _load("qwen4exp_ref")
mtp = _load("qwen4exp_mtp")
K = 3


def premixer_rows(ckpt: str, tc, ids, ple_spec: str) -> torch.Tensor:
    ple = ref.PleTable(ref.ple_source(ckpt, ple_spec), tc)
    model, pf, _ = ref.build_streamed(ckpt, tc, ple=ple)
    step = ref.layer_major(model, pf)
    return step([(list(ids), 0, {})])[0], model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--ids", default=os.path.join(_HERE, "..", "..", "tests", "golden", "prompts", "q4exp_short.ids"))
    ap.add_argument("--R", choices=("model", "random"), default="model")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--ple", default="")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tc = ref.text_config(a.checkpoint, a.layers)
    S = a.samples
    prompt = ref.read_ids(a.ids)[: S + 1]
    if len(prompt) < S + 1:
        ref.die(f"{a.ids}: {len(prompt)} ids, {S + 1} needed")
    H, hcn = tc.hidden_size, tc.hc_count
    model = None
    if a.R == "model":
        ple_spec = a.ple or (f"int8:{a.checkpoint.rstrip('/')}-ple-int8"
                             if os.path.isdir(a.checkpoint.rstrip("/") + "-ple-int8") else "bf16")
        R_all, model = premixer_rows(a.checkpoint, tc, prompt, ple_spec)
        R = R_all[:S].to(torch.bfloat16)
    else:
        R = (torch.randn(S, hcn * H) * 0.5).to(torch.bfloat16)
    toks = torch.tensor(prompt[1: S + 1], dtype=torch.int64)
    os.makedirs(a.out, exist_ok=True)
    from safetensors.torch import save_file
    for norm in mtp.NORMS:
        head = mtp.MtpHead(a.checkpoint, tc, norm=norm, model=model)
        mtp.engine_format(head)
        V = head.lm_head.shape[0]
        logits = torch.empty(S, K, V, dtype=torch.float32)
        draft = torch.empty(S, K, dtype=torch.int32)
        M = torch.empty(S, hcn * H, dtype=torch.bfloat16)
        for i in range(S):
            # a 2-id sequence: step 1 on (R_i, t_i) at position 0, steps 2 and 3 chained (rows=[0])
            res = mtp.chain(R[i:i + 1], head, [int(toks[i]), int(toks[i])], K, share_sel=True, rows=[0])
            for s in range(K):
                logits[i, s] = res["logits"][s][0].float()
            draft[i] = res["tokens"][0].to(torch.int32)
            M[i] = res["M"][0].to(torch.bfloat16)
        path = os.path.join(a.out, f"mtp_{norm}.safetensors")
        save_file({"R": R.contiguous(), "tokens": toks.to(torch.int32).contiguous(), "logits": logits.contiguous(),
                   "draft": draft.contiguous(), "M": M.contiguous()}, path,
                  metadata={"source": "tools/oracle/qwen4exp_mtp_fixture.py", "norm": norm, "R": a.R,
                            "selection": "reuse", "experts": "rtn_int4_g64 dequantised"})
        print(f"{path}: {S} samples x {K} steps, drafts {draft.tolist()}")


if __name__ == "__main__":
    main()
