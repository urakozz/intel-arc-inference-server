#!/usr/bin/env bash
# One command: quantize Qwen3.8-27B optimally for the Intel Arc B70 engine.
# Mode: rtn
#
# Verified recipe (2026-08-25, empirically on this box):
#   - auto-round PINNED to v0.14.2 - current main's shard-writer silently
#     drops lm_head packing (bug diagnosed + reproduced on TinyLlama);
#     v0.14.2 packs it correctly (verified: qweight/qzeros/scales, v1 zeros).
#   - W4A16, group_size 64, symmetric  -> matches the b70 kernels' g64 tiles
#     and the dequant contract w = (q-8)*scale, qzeros 0x77777777.
#   - --quant_lm_head + format auto_round:auto_gptq -> int4 lm_head, the
#     single biggest remaining perf item (2.54 GB -> 0.68 GB read per token).
#   - Note on XMX: W4A16-g64 IS the optimal decode format for the B70
#     (decode is bandwidth-bound). XMX/MXFP4 only matters for compute-bound
#     prefill later; same checkpoint, no format change needed.
#
# Usage (on the box):  bash quantize_qwen38_rtn.sh
#   overridables: MODEL, OUT, DEVICE (default 1 = second B70)
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
DEVICE="${DEVICE:-1}"
VENVPY="$HOME/auto-round/.venv/bin/python"    # torch-2.13+xpu venv
PKG="$HOME/.cache/auto-round-v0142-pkg"       # pinned auto-round, installed once
OUT="${OUT:-$HOME/models/qwen38-27b-w4g64-rtn}"
MODE_FLAGS="--iters 0 --disable_opt_rtn"   # RTN: minutes, format-identical, lower accuracy

# --- pin auto-round v0.14.2 beside the venv (one-time) ----------------------
if [ ! -d "$PKG/auto_round" ]; then
  UV=""
  for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" uv; do
    command -v "$c" >/dev/null 2>&1 && UV="$c" && break
  done
  if [ -n "$UV" ]; then
    "$UV" pip install -q --no-deps --target "$PKG" auto-round==0.14.2
  else
    python3 - "$PKG" <<'PY'
import json, urllib.request, zipfile, io, sys
m = json.load(urllib.request.urlopen("https://pypi.org/pypi/auto-round/0.14.2/json"))
u = next(x["url"] for x in m["urls"] if x["filename"].endswith("py3-none-any.whl"))
zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(u).read())).extractall(sys.argv[1])
PY
  fi
fi
PYTHONPATH="$PKG" "$VENVPY" - <<'PY'
import auto_round
assert auto_round.__version__ == "0.14.2", auto_round.__version__
PY

# --- quantize (cd /tmp keeps the source checkout off sys.path) --------------
cd /tmp
PYTHONPATH="$PKG" "$VENVPY" -m auto_round "$MODEL" \
  --scheme W4A16 --group_size 64 --quant_lm_head \
  --format "auto_round:auto_gptq" \
  $MODE_FLAGS \
  --device "$DEVICE" --low_gpu_mem_usage \
  --output_dir "$OUT"

# --- verify: refuse to succeed unless the artifact is B70-ready -------------
PYTHONPATH="$PKG" "$VENVPY" - "$OUT" <<'PY'
import glob, json, os, struct, sys
import numpy as np
root = sys.argv[1]
cfg = glob.glob(os.path.join(root, "**", "config.json"), recursive=True)[0]
d = os.path.dirname(cfg)
qc = json.load(open(cfg))["quantization_config"]
assert qc["bits"] == 4 and qc["group_size"] == 64 and qc["sym"] is True, qc
lm = {}
for p in glob.glob(os.path.join(d, "*.safetensors")):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n)); off = 8 + n
        for k, t in h.items():
            if k.startswith("lm_head"):
                lm[k] = (t, p, off)
assert "lm_head.qweight" in lm, f"FAIL: lm_head NOT packed, found only {list(lm)}"
t, p, off = lm["lm_head.qzeros"]
with open(p, "rb") as f:
    f.seek(off + t["data_offsets"][0])
    w = np.frombuffer(f.read(16), dtype=np.uint32)
assert all(int(x) == 0x77777777 for x in w), [hex(int(x)) for x in w]
print(f"VERIFIED B70-READY: int4 g64 sym, lm_head packed, v1 zeros -> {d}")
PY
