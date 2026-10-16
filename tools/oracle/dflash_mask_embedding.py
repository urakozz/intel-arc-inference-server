#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert a DFlash drafter's `mask_embedding.pt` into `mask_embedding.safetensors` (spec 19b).

vLLM's qwen3_dflash.py `_read_mask_embedding` reads an optional `mask_embedding.pt` beside the drafter: a tensor, or
{'embedding': tensor, 'mask_token_id': int}; it replaces the embedding row of `mask_token_id`. The engine's loader
(src/loader/dflash_repack.h) does not read torch pickles; it reads the same data from `mask_embedding.safetensors`:
one tensor `embedding` [hidden] in the stored dtype (bf16 or fp32), metadata `mask_token_id` when the .pt names one.

    dflash_mask_embedding.py <drafter dir>            # writes <drafter dir>/mask_embedding.safetensors

The .pt is left in place; the loader refuses a directory that still holds it, so move it aside after converting
(the loader reads only the .safetensors). Run where torch is (the oracle container).
"""
from __future__ import annotations

import os
import sys


def convert(drafter_dir: str) -> str:
    import torch
    from safetensors.torch import save_file

    src = os.path.join(drafter_dir, "mask_embedding.pt")
    state = torch.load(src, weights_only=True, map_location="cpu")
    meta = {}
    if isinstance(state, dict):
        if "mask_token_id" in state:
            meta["mask_token_id"] = str(int(state["mask_token_id"]))
        state = state["embedding"]
    emb = state.reshape(-1).contiguous()
    if emb.dtype not in (torch.bfloat16, torch.float32):
        emb = emb.float()
    dst = os.path.join(drafter_dir, "mask_embedding.safetensors")
    save_file({"embedding": emb}, dst, metadata=meta or None)
    return dst


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(convert(sys.argv[1]))


if __name__ == "__main__":
    main()
