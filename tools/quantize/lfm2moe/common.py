# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the LFM2.5-24B-A2B quantisation tools (spec 24b Task 1; spec 24 §5 / §6).

No calibration or AutoRound pipeline lives here yet (that is 24q's, not built when this was
written) - just the paths, the JSON helpers and a loader for tools/oracle/lfm2moe_ref.py, the
module every lfm2moe quantisation tool reads `config` / `expected_names` from.
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ORACLE = os.path.join(REPO, "tools", "oracle")

AUTOROUND_VERSION = "0.17.0"
QZ = 0x77777777
GROUP = 64


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def resolve_model(model: str, revision: str | None = None) -> str:
    """A local snapshot directory for MODEL: a directory as given, or the HF hub snapshot."""
    if os.path.isdir(model):
        if not os.path.exists(os.path.join(model, "config.json")):
            raise SystemExit(f"{model} has no config.json")
        return os.path.abspath(model)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model, revision=revision)


def find_export(arm_dir: str) -> str:
    """The directory an export sits in, below arm_dir (tools/quantize/kolibri/common.py's helper)."""
    found = sorted(p for p, _, f in os.walk(arm_dir) if "config.json" in f and "quantization_config.json" in f)
    if not found:
        found = sorted(p for p, _, f in os.walk(arm_dir) if "config.json" in f)
    if len(found) != 1:
        raise SystemExit(f"{arm_dir}: expected one exported checkpoint, found {found}")
    return found[0]


def lfm2moe_ref():
    """tools/oracle/lfm2moe_ref.py as a module (its `config` and `expected_names`, spec 24a)."""
    if "lfm2moe_ref" in sys.modules:
        return sys.modules["lfm2moe_ref"]
    spec = importlib.util.spec_from_file_location("lfm2moe_ref", os.path.join(ORACLE, "lfm2moe_ref.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lfm2moe_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


def lfm2moe_make_tiny():
    """tools/oracle/lfm2moe_make_tiny.py as a module (its `rtn_int4` / `dequant`, src/loader/rtn.h's rule -
    make_synth.py reuses this proven packer instead of a second copy of the formula)."""
    if "lfm2moe_make_tiny" in sys.modules:
        return sys.modules["lfm2moe_make_tiny"]
    spec = importlib.util.spec_from_file_location("lfm2moe_make_tiny", os.path.join(ORACLE, "lfm2moe_make_tiny.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lfm2moe_make_tiny"] = mod
    spec.loader.exec_module(mod)
    return mod
