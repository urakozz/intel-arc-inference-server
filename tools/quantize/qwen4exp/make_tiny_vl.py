#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The tiny stand-in for Qwen/Qwen3.8-Flash-Next that quantize_qwen4exp.sh's `selftest` quantises.

    make_tiny_vl.py <out> --tokenizer <dir with tokenizer.json, tokenizer_config.json, chat_template.jinja>
                          --processor <dir with preprocessor_config.json (Intel's or the original's snapshot)>

tools/oracle/qwen4exp_make_tiny.py's `bf16` form (the original's names: `model.language_model.*`, fused
`mlp.experts.gate_up_proj` / `down_proj`, `mtp.*` with fused experts; hidden 128, layers lllq, 16 experts
top-4, the PLE at layer 1) plus what the downloaded tiny (qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next:
hidden 16, so nothing in it is a multiple of 64) and that form both lack and the real checkpoint has: a vision
tower (`model.visual.*`, one block of width 64, random), so the model loads as the original does -
Qwen4ExpForConditionalGeneration with every parameter in the checkpoint -, the tokenizer files and the
processor files (AutoRound takes the multimodal path for this architecture and needs the processor).
"""
import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "..", "..", "oracle")
sys.path.insert(0, ORACLE)
import qwen4exp_make_tiny as MT  # noqa: E402

VISION = {"deepstack_visual_indexes": [], "depth": 1, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 64,
          "in_channels": 3, "initializer_range": 0.02, "intermediate_size": 128, "model_type": "qwen4_exp",
          "num_heads": 2, "num_position_embeddings": 2304, "patch_size": 16, "spatial_merge_size": 2,
          "temporal_patch_size": 2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--processor", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    MT.make(a.out, form="bf16", mtp=True, seed=a.seed)
    cfg_path = os.path.join(a.out, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["vision_config"] = dict(VISION, out_hidden_size=cfg["text_config"]["hidden_size"])
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    from transformers import AutoConfig, AutoModelForImageTextToText
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(a.out))
    gen = torch.Generator().manual_seed(a.seed + 1)
    st = os.path.join(a.out, "model.safetensors")
    sd = load_file(st)
    for k, p in model.state_dict().items():
        if k.startswith("model.visual."):
            sd[k] = (0.02 * torch.randn(tuple(p.shape), generator=gen)).to(torch.bfloat16)
    save_file(sd, st, metadata={"format": "pt"})
    with open(os.path.join(a.out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in sorted(sd)}}, f)
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copyfile(os.path.join(a.tokenizer, fn), os.path.join(a.out, fn))
    for fn in ("preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"):
        if os.path.exists(os.path.join(a.processor, fn)):
            shutil.copyfile(os.path.join(a.processor, fn), os.path.join(a.out, fn))
    print(f"wrote {a.out}: {len(sd)} tensors, the vision tower included")


if __name__ == "__main__":
    main()
