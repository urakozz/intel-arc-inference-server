#!/usr/bin/env bash
# One command: quantize Qwen3.8-27B optimally for the Intel Arc B70 engine.
# Mode: rtn (plain round-to-nearest, no calibration)
#
# Recipe:
#   - auto-round 0.17.0 at AR_COMMIT + tools/quantize/auto-round-local.patch
#     (tools/quantize/ar_pin.sh, the tuned script's install), with
#     --disable_low_cpu_mem_usage. Until 2026-10-06 this script pinned v0.14.2:
#     with the default low_cpu_mem_usage, 0.17.0's RTN (zero-shot) loop quantises
#     lm_head as a "remaining layer" but never packs it, and ShardWriter.finalize
#     writes the plain lm_head.weight while config.json's extra_config still
#     declares lm_head bits 4 (auto_round/compressors/orchestrator.py, "Quantizing
#     remaining layer", has no immediate_pack). --disable_low_cpu_mem_usage takes
#     the path that packs at export: byte-identical to 0.14.2 on tiny random
#     LLaMA / Qwen3 models (tools/quantize/README.md, "AutoRound version"). Its
#     price: the whole bf16 model in RAM (~54 GB for the 27B), so the script
#     refuses below RTN_MIN_RAM_GB (64) of available memory. The checker below
#     refuses an export whose lm_head is not packed either way.
#   - W4A16, group_size 64, symmetric  -> matches the b70 kernels' g64 tiles
#     and the dequant contract w = (q-8)*scale, qzeros 0x77777777.
#   - --quant_lm_head + format auto_round:auto_gptq -> int4 lm_head, the
#     single biggest remaining perf item (2.54 GB -> 0.68 GB read per token).
#   - Note on XMX: W4A16-g64 IS the optimal decode format for the B70
#     (decode is bandwidth-bound). XMX/MXFP4 only matters for compute-bound
#     prefill later; same checkpoint, no format change needed.
#   - The export is refused unless tools/quantize/check_gptq_export.py (the
#     engine loader's rules) passes, and README.md in the export records the
#     auto-round version, commit and patch.
#
# Usage (on the box, from the repo):  bash tools/quantize_qwen38_rtn.sh
#   overridables: MODEL, OUT, DEVICE (default 1 = second B70), VENVPY,
#                 AR_REPO (git URL or local clone containing AR_COMMIT), AR_PATCH (1 / 0),
#                 RTN_MIN_RAM_GB (64), EXTRA (more auto_round flags; recorded in the README)
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
OUT="${OUT:-$HOME/models/qwen38-27b-w4g64-rtn}"
MODE_FLAGS="--iters 0 --disable_opt_rtn --disable_low_cpu_mem_usage"   # RTN: minutes, format-identical, lower accuracy

# --- RAM: --disable_low_cpu_mem_usage holds the whole bf16 model -------------
if [ -r /proc/meminfo ]; then
  avail=$(awk '/^MemAvailable:/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
  [ "$avail" -ge "${RTN_MIN_RAM_GB:-64}" ] || {
    echo "only ${avail} GB available; RTN holds the whole bf16 model (~54 GB for the 27B)." >&2
    echo "Free memory, or set RTN_MIN_RAM_GB lower for a smaller MODEL." >&2; exit 2; }
fi

# --- auto-round 0.17.0 @ AR_COMMIT + the local patch, beside the venv (one-time); cd /tmp
. "$HERE/quantize/ar_pin.sh"

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

- tool: $AR_DESC
- recipe: \`tools/quantize_qwen38_rtn.sh\` - \`--scheme W4A16 --group_size 64 --quant_lm_head
  --format auto_round:auto_gptq $MODE_FLAGS --low_gpu_mem_usage ${EXTRA:-}\`
- source: \`$MODEL\`; made $(date -u +%Y-%m-%dT%H:%MZ)
- checked by \`tools/quantize/check_gptq_export.py\`: int4 g64 sym, lm_head packed, qzeros 0x77777777
EOF
echo "provenance -> $D/README.md"
