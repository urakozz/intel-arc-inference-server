#!/usr/bin/env bash
# One command: quantize Qwen3.8-27B to MXFP4A16 for the Intel Arc B70 engine.
#
# Recipe (ruling A20, 2026-09-05):
#   - scheme MXFP4A16 = fp4 (e2m1) WEIGHTS, block 32, shared e8m0 scale per
#     block; activations stay bf16. NOT "MXFP4": that preset also quantizes
#     activations to fp4 (W4A4) - the trade the operator declined. The
#     verification below REFUSES a checkpoint whose config carries
#     input_activations, so a wrong scheme cannot slip through.
#   - llm-compressor's model_free_ptq: operates on the safetensors shards
#     directly (no model definition, no full load), data-free, handles the
#     fused q/k/v and gate/up sets across shards. Output is the canonical
#     compressed-tensors MXFP4 layout: <name>.weight_packed (uint8, two e2m1
#     nibbles per byte, [N][K/2]) + <name>.weight_scale (uint8 e8m0, [N][K/32]).
#     That layout is what sycl-tla's blockscaled MXFP4 mainloop and vLLM read,
#     and what src/loader learns to parse.
#   - Bytes/weight 4.25 = identical to GPTQ g64 (docs/16): decode's bandwidth
#     ceiling is unchanged; prefill inherits the in-register upconvert and
#     drops the 210 ms/chunk bf16 dequant scratch (the measured 18.6% term).
#   - lm_head IS quantized by default (the engine's +2.5 t/s decode lever).
#     vLLM cannot load a quantized lm_head (measured, docs/14); set
#     LM_HEAD=keep for a vLLM-loadable variant.
#
# Usage (on the box):  bash tools/quantize_qwen38_mxfp4.sh
#   overridables: MODEL, OUT, DEVICE (cpu | xpu:1), WORKERS, LM_HEAD (quant|keep)
set -euo pipefail

echo "NOTE: model_free_ptq streams shards; expect ~minutes on CPU, less on XPU."
echo "      'Skip processing for weights file' lines are its normal copy log."

MODEL="${MODEL:-Qwen/Qwen3.8-27B}"        # resolved from the local HF cache (HF_HUB_OFFLINE)
OUT="${OUT:-$HOME/models/qwen38-27b-mxfp4a16}"
DEVICE="${DEVICE:-cpu}"                    # xpu:1 if torch.xpu works for this path; cpu is the safe default
WORKERS="${WORKERS:-8}"
LM_HEAD="${LM_HEAD:-quant}"
VENVPY="$HOME/auto-round/.venv/bin/python" # torch-2.13+xpu venv; NOT modified
PKG="$HOME/.cache/llmc-pkg"                 # llm-compressor + pure-python deps, installed once, beside the venv

# --- isolated install (pure-python packages only; torch/transformers come from the venv) ---
if [ ! -d "$PKG/llmcompressor" ]; then
  UV=""
  for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" uv; do
    command -v "$c" >/dev/null 2>&1 && UV="$c" && break
  done
  [ -n "$UV" ] || { echo "uv not found; install it or set UV" >&2; exit 2; }
  "$UV" pip install -q --python "$VENVPY" --target "$PKG" --no-deps \
    llmcompressor compressed-tensors loguru pydantic pydantic-core annotated-types typing-inspection
fi
PYTHONPATH="$PKG" "$VENVPY" - <<'PY'
import importlib, sys
missing = [m for m in ("llmcompressor", "compressed_tensors", "loguru", "pydantic", "safetensors", "transformers", "torch")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"missing modules {missing}: add the pure-python ones to the uv line above (--no-deps into $PKG)")
import llmcompressor, compressed_tensors
print(f"llmcompressor {llmcompressor.__version__}  compressed-tensors {compressed_tensors.__version__}")
PY

# --- quantize (cd /tmp keeps any source checkout off sys.path) ---
IGNORE='["model.embed_tokens", "re:.*visual.*", "re:.*mtp.*", "re:.*in_proj_a$", "re:.*in_proj_b$", "re:.*conv1d.*"]'
[ "$LM_HEAD" = keep ] && IGNORE='["lm_head", "model.embed_tokens", "re:.*visual.*", "re:.*mtp.*", "re:.*in_proj_a$", "re:.*in_proj_b$", "re:.*conv1d.*"]'
cd /tmp
HF_HUB_OFFLINE=1 PYTHONPATH="$PKG" "$VENVPY" - "$MODEL" "$OUT" "$DEVICE" "$WORKERS" "$IGNORE" <<'PY'
import json, sys
from llmcompressor import model_free_ptq
model, out, device, workers, ignore = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), json.loads(sys.argv[5])
model_free_ptq(
    model_stub=model,
    save_directory=out,
    scheme="MXFP4A16",          # weights e2m1 block-32 e8m0; activations bf16. NOT "MXFP4" (W4A4).
    ignore=ignore,
    max_workers=workers,
    device=device,
)
PY

# --- verify: refuse to succeed unless the artifact is B70-ready ---
PYTHONPATH="$PKG" "$VENVPY" - "$OUT" "$LM_HEAD" <<'PY'
import glob, json, os, struct, sys
root, lm_mode = sys.argv[1], sys.argv[2]
cfg_path = glob.glob(os.path.join(root, "**", "config.json"), recursive=True)[0]
d = os.path.dirname(cfg_path)
qc = json.load(open(cfg_path))["quantization_config"]
assert qc.get("quant_method") == "compressed-tensors", qc.get("quant_method")
groups = qc["config_groups"]
assert len(groups) == 1, list(groups)
g = next(iter(groups.values()))
w = g["weights"]
assert w["num_bits"] == 4 and w["type"] == "float" and w["group_size"] == 32 and w["symmetric"] is True, w
assert g.get("input_activations") in (None, {}), f"FAIL: activations are quantized (W4A4) - scheme was not MXFP4A16: {g['input_activations']}"
heads = {}
sample = None
for p in glob.glob(os.path.join(d, "*.safetensors")):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    for k, t in h.items():
        if k.startswith("lm_head"):
            heads[k] = t
        if sample is None and k.endswith("mlp.gate_proj.weight_packed"):
            sample = (k, t, h.get(k.replace("weight_packed", "weight_scale")))
assert sample is not None, "FAIL: no <layer>.mlp.gate_proj.weight_packed found"
k, packed, scale = sample
assert packed["dtype"] == "U8" and scale and scale["dtype"] == "U8", (packed["dtype"], scale and scale["dtype"])
N, Khalf = packed["shape"]; sN, sG = scale["shape"]
assert sN == N and sG * 32 == Khalf * 2, f"scale shape {scale['shape']} vs packed {packed['shape']}"
if lm_mode == "quant":
    assert "lm_head.weight_packed" in heads, f"FAIL: lm_head NOT packed, found only {list(heads)}"
else:
    assert "lm_head.weight" in heads and "lm_head.weight_packed" not in heads, list(heads)
print(f"VERIFIED B70-READY: MXFP4A16 (e2m1 weights, e8m0 block-32 scales, bf16 activations), "
      f"sample {k} [{N}][{Khalf*2}], lm_head={'int4' if lm_mode=='quant' else 'bf16'} -> {d}")
PY
