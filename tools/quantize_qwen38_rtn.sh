#!/usr/bin/env bash
# One command: quantize Qwen3.8-27B optimally for the Intel Arc B70 engine.
# Mode: rtn (plain round-to-nearest, no calibration)
#
# Recipe:
#   - auto-round PINNED to v0.14.2 - a later main's shard writer silently
#     drops the lm_head packing (bug diagnosed + reproduced on TinyLlama,
#     2026-08-25); v0.14.2 packs it correctly (qweight/qzeros/scales, v1 zeros).
#     **Re-checked on 0.17.0 (2026-10-06, tools/quantize/README.md "AutoRound
#     version"): still broken on THIS path.** With --iters 0 --disable_opt_rtn
#     and the default low_cpu_mem_usage, 0.17.0's zero-shot loop quantises
#     lm_head as a "remaining layer" but never packs it, and ShardWriter.finalize
#     writes the plain lm_head.weight - while config.json's extra_config still
#     declares lm_head bits 4. (auto_round/compressors/orchestrator.py,
#     "Quantizing remaining layer", has no immediate_pack; the calibrated loop
#     does.) The tuned path packs it on 0.17.0 and has moved
#     (quantize_qwen38_tuned.sh); this one stays on 0.14.2.
#     --disable_low_cpu_mem_usage avoids the bug on 0.17.0 (byte-identical to
#     0.14.2 on tiny models) at the price of the whole model in RAM; not
#     adopted until tried on the 27B.
#   - W4A16, group_size 64, symmetric  -> matches the b70 kernels' g64 tiles
#     and the dequant contract w = (q-8)*scale, qzeros 0x77777777.
#   - --quant_lm_head + format auto_round:auto_gptq -> int4 lm_head, the
#     single biggest remaining perf item (2.54 GB -> 0.68 GB read per token).
#   - Note on XMX: W4A16-g64 IS the optimal decode format for the B70
#     (decode is bandwidth-bound). XMX/MXFP4 only matters for compute-bound
#     prefill later; same checkpoint, no format change needed.
#   - The export is refused unless tools/quantize/check_gptq_export.py (the
#     engine loader's rules) passes, and README.md in the export records the
#     auto-round version.
#
# Usage (on the box, from the repo):  bash tools/quantize_qwen38_rtn.sh
#   overridables: MODEL, OUT, DEVICE (default 1 = second B70), VENVPY,
#                 EXTRA (more auto_round flags; recorded in the README)
set -euo pipefail

# --- expected console noise (harmless, do not chase) ------------------------
echo "NOTE: two messages you may see and can ignore:"
echo "  1) '[transformers] The fast path is not available ... fla / causal-conv1d'"
echo "     Expected on XPU: those fused GDN kernels are CUDA-oriented"
echo "     (causal-conv1d has no XPU build). transformers falls back to the"
echo "     pure-torch path - quantization math is IDENTICAL, tuned runs are"
echo "     just somewhat slower. Nothing to install."
echo "  2) 'Download complete: ... 0.00B' - the model resolved from the local"
echo "     HF cache; zero bytes were downloaded. Cosmetic progress-bar artifact."

HERE="$(cd "$(dirname "$0")" && pwd)"
CHECK="$HERE/quantize/check_gptq_export.py"
[ -f "$CHECK" ] || { echo "missing $CHECK - run this script from the repo tree" >&2; exit 2; }
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
DEVICE="${DEVICE:-1}"
VENVPY="${VENVPY:-$HOME/auto-round/.venv/bin/python}"   # torch-2.13+xpu venv; NOT modified
AR_VERSION=0.14.2
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
# cd /tmp before any python: `python -` puts the cwd ahead of PYTHONPATH, so an
# auto-round checkout as cwd would shadow the pinned package.
cd /tmp
PYTHONPATH="$PKG" "$VENVPY" - "$PKG" <<'PY'
import sys
import auto_round
assert auto_round.__version__ == "0.14.2", auto_round.__version__
assert auto_round.__file__.startswith(sys.argv[1]), f"auto_round imported from {auto_round.__file__}, not {sys.argv[1]}"
print(f"auto-round {auto_round.__version__} from {auto_round.__file__}")
PY

# --- quantize (from /tmp, so no source checkout is on sys.path) -------------
PYTHONPATH="$PKG" "$VENVPY" -m auto_round "$MODEL" \
  --scheme W4A16 --group_size 64 --quant_lm_head \
  --format "auto_round:auto_gptq" \
  $MODE_FLAGS ${EXTRA:-} \
  --device "$DEVICE" --low_gpu_mem_usage \
  --output_dir "$OUT"

# --- verify: refuse to succeed unless the artifact is B70-ready -------------
"$VENVPY" "$CHECK" "$OUT" --group 64 --lm-head quant

# --- provenance: the export's README records what made it -------------------
D="$(dirname "$(find "$OUT" -name config.json | head -n 1)")"
cat >> "$D/README.md" <<EOF

## Quantisation provenance

- tool: auto-round $AR_VERSION (PyPI wheel; pinned - 0.17.0 drops the lm_head packing on this RTN path)
- recipe: \`tools/quantize_qwen38_rtn.sh\` - \`--scheme W4A16 --group_size 64 --quant_lm_head
  --format auto_round:auto_gptq --iters 0 --disable_opt_rtn --low_gpu_mem_usage ${EXTRA:-}\`
- source: \`$MODEL\`; made $(date -u +%Y-%m-%dT%H:%MZ)
- checked by \`tools/quantize/check_gptq_export.py\`: int4 g64 sym, lm_head packed, qzeros 0x77777777
EOF
echo "provenance -> $D/README.md"
