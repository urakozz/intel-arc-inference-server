#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""The `taps` stage's input for Inkling 2 Mini's DFlash 2 drafter: the training recordings' requests rendered by
Inkling's own chat template.

    inkling_sources.py --target <Inkling snapshot> --ids-dir DIR --out SOURCES.json [--done TAPS_DIR] RECORDING_DIR...

A recording is a directory of b70-serve request JSONs (`request`: the OpenAI messages and tools). Their
`prompt_ids` are another model's tokens, so each request is rendered again: its messages and tools through the
target's chat template (`AutoTokenizer.apply_chat_template`, add_generation_prompt, the request's
reasoning_effort when it sets one), the same form tools/oracle/inkling_kv8_kld.py builds. The ids are written to
`DIR/<recording>__<request>.ids`. The taps stage runs `b70-decode --dump-taps` on them: a greedy continuation, so
the drafter is tuned on the text the engine itself writes. The model is greedy-served by default, and its
acceptance is judged on that text.

Held out, refused here: the whole `opencode-session-2` recording (the held-out acceptance source) and the A4 /
golden sets, which never pass through this script. `--done` skips every request whose dump is already complete in
TAPS_DIR (a resumed `taps` stage).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sources import HELD_OUT, dump_complete, file_name  # noqa: E402


def _kv8_kld():
    path = os.path.join(HERE, "..", "..", "oracle", "inkling_kv8_kld.py")
    spec = importlib.util.spec_from_file_location("inkling_kv8_kld", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--target", required=True, help="the Inkling 2 Mini snapshot (its tokenizer and chat template)")
    ap.add_argument("--ids-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--done", help="the taps directory: skip requests whose dump is complete there")
    ap.add_argument("--min-prompt", type=int, default=64, help="skip a request whose rendered prompt is shorter")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.target)
    session_messages = _kv8_kld().session_messages
    os.makedirs(a.ids_dir, exist_ok=True)
    every, todo = [], []
    for d in a.recordings:
        rec = os.path.basename(os.path.normpath(d))
        if rec in HELD_OUT:
            raise SystemExit(f"refused: {d} is held out of calibration ({rec})")
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        if not files:
            raise SystemExit(f"no request JSONs in {d}")
        for path in files:
            with open(path) as f:
                r = json.load(f).get("request") or {}
            if not r.get("messages"):
                continue
            msgs, tools = session_messages(r)
            kw = {"reasoning_effort": r["reasoning_effort"]} if r.get("reasoning_effort") else {}
            ids = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True, **kw)
            ids = [int(x) for x in (ids["input_ids"] if hasattr(ids, "keys") else ids)]
            if len(ids) < a.min_prompt:
                continue
            name = f"{rec}/{os.path.basename(path)[:-5]}"
            ids_path = os.path.join(a.ids_dir, file_name(name).replace(".taps.safetensors", ".ids"))
            with open(ids_path, "w") as f:
                f.write(" ".join(map(str, ids)) + "\n")
            src = {"name": name, "ids": ids_path, "dump": file_name(name), "n_prompt": len(ids)}
            every.append(src)
            if not (a.done and dump_complete(os.path.join(a.done, src["dump"]))):
                todo.append(src)
    with open(a.out, "w") as f:
        json.dump(todo, f)
    print(f"DFLASH_QUANT_SOURCES {len(every)} requests from {len(a.recordings)} recording(s), "
          f"{len(every) - len(todo)} dumped already, {len(todo)} to run -> {a.out}")


if __name__ == "__main__":
    main()
