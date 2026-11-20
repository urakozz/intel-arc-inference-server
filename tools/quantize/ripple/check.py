#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""Stage `check` of tools/quantize/quantize_ripple.sh, the Ripple-specific half (check_gptq_export.py does the
engine's int4 rules and the coverage).

    check.py <export dir> <source snapshot> --attn int4|bf16 [--reload]

- config.json: every key of the source's config.json unchanged (auto_map, model_type, layer_types, sliding_window,
  use_valres, embed_scale, ffn_clamp_threshold, gate_full_precision, ...) beside quantization_config
- the remote code, tokenizer files, chat template and generation_config.json byte-identical to the source's
- ripple_recipe.json (autoround_run.py's): present, its value_residual_seen_layers in 1 .. num_hidden_layers (the
  guard saw the value residual arrive on at least one decoder-layer call)
- the attention arm: per layer mlp.{gate,up,down}_proj packed (.qweight); self_attn.{q,k,v,o,g}_proj packed
  (--attn int4) or shipped as bf16 `.weight` (--attn bf16); embed_tokens, lm_head, the norms and
  value_res_lambda shipped bf16 (bf16_same.py has proven them bitwise the source's)
- --reload (the selftest; never on the real 130 GB model): the export loaded by transformers with
  trust_remote_code (AutoRound's own inference backend for auto_round:auto_gptq) and run on a chat prompt; the
  same prompt through the source model with every packed linear replaced by the engine's dequantisation
  w = (q - 8) * scale (qzeros 0x77777777, group 64): the two must agree (relative L2 of the logits < 1e-2, the
  same argmax at >= 90 % of positions: AutoRound's backend runs in fp16, the reference in fp32, and a random tiny
  model has near ties; a wrong layout is off by O(1)), so the export's layout is the one the engine reads. The distance to the
  bf16 source is printed (random weights: only a sanity number).
Prints "RIPPLE CHECK: OK ..." or exits 1 with every failure.
"""
import argparse
import filecmp
import json
import os
import struct
import sys

SAME_FILES = ("configuration_ricursive.py", "modeling_ricursive.py", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "chat_template.jinja", "generation_config.json")
CONFIG_FREE = {"dtype", "torch_dtype", "transformers_version", "quantization_config", "hybrid_layer_pattern",
               "is_hybrid_swa"}
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj", "g_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def headers(d: str) -> dict:
    with open(os.path.join(d, "model.safetensors.index.json"), encoding="utf-8") as f:
        files = sorted(set(json.load(f)["weight_map"].values()))
    out = {}
    for fn in files:
        with open(os.path.join(d, fn), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            out.update({k: v for k, v in json.loads(f.read(n)).items() if k != "__metadata__"})
    return out


def dequant(qweight, scales, group: int):
    """The engine's int4 rule: element k of column n = nibble k % 8 of qweight[k // 8, n]; w = (q - 8) * s."""
    import torch
    K8, N = qweight.shape
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    q = (qweight.unsqueeze(1) >> shifts[None, :, None]) & 0xF          # [K/8, 8, N]
    q = q.reshape(K8 * 8, N).float() - 8
    s = scales.float().repeat_interleave(group, dim=0)                 # [K, N]
    return (q * s).t().contiguous()                                    # [N, K] like nn.Linear.weight


def reload_check(export: str, source: str, fails: list) -> str:
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    text = tok.apply_chat_template([{"role": "user", "content": "Write a Python function that reverses a list."},
                                    {"role": "assistant", "content": "def rev(xs):\n    return xs[::-1]\n"}],
                                   tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        q = AutoModelForCausalLM.from_pretrained(export, trust_remote_code=True, dtype=torch.float32,
                                                 device_map="cpu")
        lq = q(ids, use_cache=False).logits[0].float()
        del q
        ref = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True, dtype=torch.float32)
        lb = ref(ids, use_cache=False).logits[0].float()
        with open(os.path.join(export, "model.safetensors.index.json"), encoding="utf-8") as f:
            wm = json.load(f)["weight_map"]
        group = json.load(open(os.path.join(export, "config.json")))["quantization_config"]["group_size"]
        mods = dict(ref.named_modules())
        n = 0
        for k in wm:
            if k.endswith(".qweight"):
                p = k[: -len(".qweight")]
                with safe_open(os.path.join(export, wm[k]), "pt") as f:
                    qw = f.get_tensor(k)
                with safe_open(os.path.join(export, wm[p + ".scales"]), "pt") as f:
                    sc = f.get_tensor(p + ".scales")
                mods[p].weight.copy_(dequant(qw, sc, group))
                n += 1
        le = ref(ids, use_cache=False).logits[0].float()
    if not (torch.isfinite(lq).all() and torch.isfinite(le).all()):
        fails.append("reload: non-finite logits")
        return ""
    rel = float((lq - le).norm() / le.norm())
    agree = float((lq.argmax(-1) == le.argmax(-1)).float().mean())
    to_src = float((le - lb).norm() / lb.norm())
    if rel >= 1e-2 or agree < 0.9:
        fails.append(f"reload: transformers' load of the export and the engine's dequantisation disagree "
                     f"(relative L2 {rel:.2e}, argmax agreement {agree:.3f} over {ids.shape[1]} positions)")
    return (f"reload: transformers + trust_remote_code loaded the export; vs the engine's dequantisation of "
            f"{n} packed linears: relative L2 {rel:.2e}, argmax agreement {agree:.3f} over {ids.shape[1]} "
            f"positions; dequantised vs bf16 source: relative L2 {to_src:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    ap.add_argument("source")
    ap.add_argument("--attn", choices=("int4", "bf16"), required=True)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    fails = []
    src_cfg = json.load(open(os.path.join(a.source, "config.json")))
    cfg = json.load(open(os.path.join(a.export, "config.json")))
    for k, v in src_cfg.items():
        if k not in CONFIG_FREE and cfg.get(k) != v:
            fails.append(f"config.json: {k} = {str(cfg.get(k))[:80]!r}, the source has {str(v)[:80]!r}")
    for fn in SAME_FILES:
        p = os.path.join(a.export, fn)
        if not os.path.exists(p):
            fails.append(f"{fn} missing in the export")
        elif not filecmp.cmp(p, os.path.join(a.source, fn), shallow=False):
            fails.append(f"{fn} differs from the source's")
    ts = headers(a.export)
    L = src_cfg["num_hidden_layers"]
    rp = os.path.join(a.export, "ripple_recipe.json")
    seen = json.load(open(rp)).get("value_residual_seen_layers") if os.path.exists(rp) else None
    if seen is None:
        fails.append("ripple_recipe.json (or its value_residual_seen_layers) missing - the quant stage did not finish")
    elif not 0 < seen <= L:
        fails.append(f"ripple_recipe.json: value_residual_seen_layers = {seen}, not 1 .. {L} (the decoder "
                     "layers the value residual guard saw called with it)")
    for i in range(L):
        pre = f"model.layers.{i}."
        for m in MLP:
            if pre + f"mlp.{m}.qweight" not in ts:
                fails.append(f"{pre}mlp.{m} not packed")
        for m in ATTN:
            packed, fp = pre + f"self_attn.{m}.qweight" in ts, ts.get(pre + f"self_attn.{m}.weight")
            if a.attn == "int4" and not packed:
                fails.append(f"{pre}self_attn.{m} not packed (--attn int4)")
            if a.attn == "bf16" and (packed or fp is None or fp["dtype"] != "BF16"):
                fails.append(f"{pre}self_attn.{m} is not a bf16 .weight (--attn bf16)")
        for t in ("self_attn.value_res_lambda", "self_attn.q_norm.weight", "input_layernorm.weight",
                  "post_feedforward_layernorm.weight"):
            if ts.get(pre + t, {}).get("dtype") != "BF16":
                fails.append(f"{pre}{t} not shipped bf16")
    for t in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
        if ts.get(t, {}).get("dtype") != "BF16":
            fails.append(f"{t} not shipped bf16")
    note = reload_check(a.export, a.source, fails) if a.reload and not fails else ""
    if fails:
        print(f"RIPPLE CHECK: FAIL - {len(fails)} problem(s) in {a.export}:")
        for f in fails[:40]:
            print("  - " + f)
        sys.exit(1)
    if note:
        print(note)
    print(f"RIPPLE CHECK: OK {L} layers, MLP int4, attention {a.attn}, embedding / head / norms / value_res_lambda "
          f"bf16, config + remote code + tokenizer + template as the source's -> {a.export}")


if __name__ == "__main__":
    main()
