#!/usr/bin/env bash
# One command: quantize Qwen3.8-27B optimally for the Intel Arc B70 engine.
# Mode: tuned (AutoRound sign-SGD with calibration)
#
# Recipe:
#   - auto-round = intel/auto-round main at AR_COMMIT (0.17.0), the version
#     tools/quantize_kolibri1.sh uses (operator's ruling 2026-10-05). Until
#     2026-10-06 this script pinned v0.14.2, because a later main's shard writer
#     silently dropped the lm_head packing under --quant_lm_head. Re-checked on
#     0.17.0 (tools/quantize/README.md, "AutoRound version"): THIS path - the
#     calibrated one - packs lm_head and its export is byte-identical to
#     0.14.2's on tiny random LLaMA / Qwen3 models; the RTN path
#     (--iters 0 --disable_opt_rtn) still drops it on 0.17.0, so
#     quantize_qwen38_rtn.sh keeps the v0.14.2 pin.
#   - W4A16, group_size 64, symmetric  -> matches the b70 kernels' g64 tiles
#     and the dequant contract w = (q-8)*scale, qzeros 0x77777777.
#   - --quant_lm_head + format auto_round:auto_gptq -> int4 lm_head, the
#     single biggest remaining perf item (2.54 GB -> 0.68 GB read per token).
#   - Note on XMX: W4A16-g64 IS the optimal decode format for the B70
#     (decode is bandwidth-bound). XMX/MXFP4 only matters for compute-bound
#     prefill later; same checkpoint, no format change needed.
#   - The export is refused unless tools/quantize/check_gptq_export.py (the
#     engine loader's rules) passes, and README.md in the export records the
#     auto-round version and commit.
#
# Usage (on the box, from the repo):  bash tools/quantize_qwen38_tuned.sh
#   overridables: MODEL, OUT, DEVICE (default 1 = second B70), VENVPY,
#                 AR_SRC (a clean auto-round checkout at AR_COMMIT, or a pip URL),
#                 EXTRA (more auto_round flags, e.g. "--nsamples 512"; recorded in the README)
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
AR_VERSION=0.17.0
AR_COMMIT=6afaecdbe4092dc803f3e91026fe81a65b64b372      # intel/auto-round main, 2026-09-30, version 0.17.0
AR_SRC="${AR_SRC:-git+https://github.com/intel/auto-round@$AR_COMMIT}"
PKG="$HOME/.cache/auto-round-${AR_COMMIT:0:8}-pkg"      # pinned auto-round, installed once, beside the venv
OUT="${OUT:-$HOME/models/qwen38-27b-w4g64-tuned}"
MODE_FLAGS=""                               # tuned sign-SGD (200 iters, 128 x 2048 pile-10k): hours, best accuracy

# --- pin auto-round at AR_COMMIT beside the venv (one-time) -----------------
# --no-deps --target: torch/transformers/accelerate/datasets stay the venv's.
# Building from git needs `git` (setup.py runs `git describe`).
if [ "$(cat "$PKG/.ar_commit" 2>/dev/null || true)" != "$AR_COMMIT" ]; then
  if [ -d "$AR_SRC" ]; then
    head=$(git -C "$AR_SRC" rev-parse HEAD)
    [ "$head" = "$AR_COMMIT" ] || { echo "AR_SRC $AR_SRC is at $head, the recipe pins $AR_COMMIT" >&2; exit 2; }
    [ -z "$(git -C "$AR_SRC" status --porcelain --untracked-files=no)" ] ||
      { echo "AR_SRC $AR_SRC has local changes; the recipe needs $AR_COMMIT as committed" >&2; exit 2; }
  fi
  UV=""
  for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" uv; do
    command -v "$c" >/dev/null 2>&1 && UV="$c" && break
  done
  mkdir -p "$PKG"
  if [ -n "$UV" ]; then
    "$UV" pip install -q --python "$VENVPY" --no-deps --target "$PKG" "$AR_SRC"
  else
    "$VENVPY" -m pip install -q --no-deps --target "$PKG" "$AR_SRC"
  fi
  echo "$AR_COMMIT" > "$PKG/.ar_commit"
fi
# cd /tmp before any python: `python -` puts the cwd ahead of PYTHONPATH, so an
# auto-round checkout as cwd would shadow the pinned package.
cd /tmp
PYTHONPATH="$PKG" "$VENVPY" - "$AR_VERSION" "$PKG" <<'PY'
import importlib.util, sys
assert sys.version_info >= (3, 11), f"auto-round 0.17 needs python >= 3.11, the venv has {sys.version}"
missing = [m for m in ("torch", "transformers", "accelerate", "datasets", "cpuinfo", "pydantic", "numpy")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"the venv lacks {missing}: pip install datasets py-cpuinfo pydantic accelerate into it")
import auto_round
v = tuple(int(x) for x in auto_round.__version__.split(".")[:3])
want = tuple(int(x) for x in sys.argv[1].split("."))
assert v >= want, f"auto-round {auto_round.__version__} < {sys.argv[1]}"
assert auto_round.__file__.startswith(sys.argv[2]), f"auto_round imported from {auto_round.__file__}, not {sys.argv[2]}"
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

- tool: auto-round $AR_VERSION, intel/auto-round commit \`$AR_COMMIT\` (\`$AR_SRC\`)
- recipe: \`tools/quantize_qwen38_tuned.sh\` - \`--scheme W4A16 --group_size 64 --quant_lm_head
  --format auto_round:auto_gptq --low_gpu_mem_usage ${EXTRA:-}\`, defaults otherwise (200 iters,
  128 x 2048 samples of NeelNanda/pile-10k)
- source: \`$MODEL\`; made $(date -u +%Y-%m-%dT%H:%MZ)
- checked by \`tools/quantize/check_gptq_export.py\`: int4 g64 sym, lm_head packed, qzeros 0x77777777
EOF
echo "provenance -> $D/README.md"
