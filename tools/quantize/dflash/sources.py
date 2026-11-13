#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The `taps` stage's input: tools/probe/probe_dflash_iter --engine-taps SOURCES.json from the training recordings.

    sources.py --generation-config <target>/generation_config.json --out SOURCES.json [--done TAPS_DIR]
               [--max-cont 256] RECORDING_DIR...

A recording is a directory of request JSONs (`prompt_ids`, `out_ids`) written by b70-serve's request log
(~/b70/logs/opencode-session-int8, ~/b70/logs/opencode-session). Each request with a prompt and at least 8
continuation ids (the probe's floor) becomes a source `<recording>/<request>`: the prompt and the recorded
continuation cut at `--max-cont` ids and at the first EOS (dflash_v2_sources.stopped). The probe teacher-forces
that continuation and dumps the quantised target's taps at the drafter's tap layers over its last 2048 prompt
positions and the continuation.

Held out, refused here: the whole `opencode-session-2` recording (spec 19 v2's held-out acceptance source) and
anything that is not a request recording - A4 / golden come from oracle-out-19a dumps and never pass through
this script. `--done` skips every source whose dump is already complete in TAPS_DIR (a resumed `taps` stage).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import struct

_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
HELD_OUT = ("opencode-session-2",)


def _stopped():
    spec = importlib.util.spec_from_file_location(
        "dflash_v2_sources", os.path.join(_TOOLS, "box_validate", "dflash_v2_sources.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.stopped


def file_name(name: str) -> str:
    """probe_dflash_iter's dump name: every '/' of the source name -> '__'."""
    return name.replace("/", "__") + ".taps.safetensors"


def dump_complete(path: str) -> bool:
    """The header parses and the file holds every byte it declares (a stage killed mid-write leaves less)."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        end = max(t["data_offsets"][1] for t in h.values())
        return os.path.getsize(path) >= 8 + n + end
    except (OSError, ValueError, KeyError, struct.error):
        return False


def sources(recordings: list[str], eos: set[int], max_cont: int) -> list[dict]:
    stopped = _stopped()
    out = []
    for d in recordings:
        rec = os.path.basename(os.path.normpath(d))
        if rec in HELD_OUT:
            raise SystemExit(f"refused: {d} is held out of calibration ({rec})")
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        if not files:
            raise SystemExit(f"no request JSONs in {d}")
        for path in files:
            with open(path) as f:
                r = json.load(f)
            prompt = r.get("prompt_ids") or []
            cont = stopped((r.get("out_ids") or [])[:max_cont], eos)
            if prompt and len(cont) >= 8:
                out.append({"name": f"{rec}/{os.path.basename(path)[:-5]}", "prompt_ids": prompt,
                            "cont_ids": cont, "eos_ids": sorted(eos)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--generation-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--done", help="the taps directory: skip sources whose dump is complete there")
    ap.add_argument("--max-cont", type=int, default=256)
    a = ap.parse_args()
    with open(a.generation_config) as f:
        e = json.load(f)["eos_token_id"]
    eos = set(e if isinstance(e, list) else [e])
    all_src = sources(a.recordings, eos, a.max_cont)
    todo = [s for s in all_src if not (a.done and dump_complete(os.path.join(a.done, file_name(s["name"]))))]
    with open(a.out, "w") as f:
        json.dump(todo, f)
    print(f"DFLASH_QUANT_SOURCES {len(all_src)} requests from {len(a.recordings)} recording(s), "
          f"{len(all_src) - len(todo)} dumped already, {len(todo)} to run -> {a.out}")


if __name__ == "__main__":
    main()
