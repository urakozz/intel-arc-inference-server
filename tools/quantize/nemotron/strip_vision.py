#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""Stage `strip` of tools/quantize/quantize_nemotron.sh: the vision tower and projector out of the export.

    strip_vision.py <export dir>

The engine serves Nemotron 3.5 Super text-only and never loads the RADIO tower (`vision_model.*`), its projector
(`mlp1.*`, `vision_projector.*`) or the stray `vision_final_layernorm.*` AutoRound writes beside them - and AutoRound's
export of that tower is not the source's either (its renaming back to checkpoint names loses the fused `qkv` and the
projector shapes; it sits outside the quantised blocks, so nothing checks it). Every shard holding such a tensor is
rewritten without it (tmp file, then rename), a shard left empty is moved aside to `<export>/stripped/`, and the
index is rewritten. config.json keeps its `vision_config`: the omni wrapper still builds (with a random tower) for a
text-only load. Idempotent: a stripped export is left as it is. Resumable: a run killed after moving a shard aside
but before the index rewrite skips that shard on the rerun (it is already in `stripped/`), a rewritten shard is
rewritten again without change, and the index itself is replaced in one rename.
"""
import argparse
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file

VISION = ("vision_model.", "vision_projector.", "mlp1.", "vision_final_layernorm.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    a = ap.parse_args()
    idx_path = os.path.join(a.export, "model.safetensors.index.json")
    with open(idx_path, encoding="utf-8") as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    drop = sorted(k for k in wm if k.startswith(VISION))
    for fn in sorted({wm[k] for k in drop}):
        path = os.path.join(a.export, fn)
        if not os.path.exists(path) and os.path.exists(os.path.join(a.export, "stripped", fn)):
            continue   # moved aside by a run killed before its index rewrite
        with safe_open(path, "pt") as f:
            meta = f.metadata()
            keep = {k: f.get_tensor(k) for k in f.keys() if not k.startswith(VISION)}
        if keep:
            save_file(keep, path + ".tmp", metadata=meta or {"format": "pt"})
            os.replace(path + ".tmp", path)
        else:
            os.makedirs(os.path.join(a.export, "stripped"), exist_ok=True)
            os.replace(path, os.path.join(a.export, "stripped", fn))
    for k in drop:
        del wm[k]
    with open(idx_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)
    os.replace(idx_path + ".tmp", idx_path)
    print(f"strip_vision: {len(drop)} vision / projector tensor(s) dropped, {len(wm)} left")


if __name__ == "__main__":
    main()
