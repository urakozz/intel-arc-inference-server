# Sourced by tools/quantize_qwen38_{tuned,rtn}.sh: installs auto-round at AR_COMMIT (0.17.0),
# with tools/quantize/auto-round-local.patch applied unless AR_PATCH=0, beside the venv, once.
#
# In:  HERE (tools/), VENVPY (python with torch/transformers; NOT modified),
#      AR_REPO  where the source comes from: a git URL or a local clone that contains AR_COMMIT
#               (default https://github.com/intel/auto-round)
#      AR_PATCH 1 (default): apply auto-round-local.patch - neither fix is upstream in 0.17.0:
#               calib_dataset.py resets the BOS/EOS counter per sample (without it every
#               sample that ends in EOS shortens all later calibration rows), wrapper.py gives
#               WrapperLinear in_features / out_features; 0: AR_COMMIT as committed
# Out: AR_VERSION, AR_COMMIT, AR_DESC (for the provenance line), PKG (put it on PYTHONPATH);
#      the shell is left in /tmp: `python -` puts the cwd ahead of PYTHONPATH, so an
#      auto-round checkout as cwd would shadow the pinned package.
AR_VERSION=0.17.0
AR_COMMIT=6afaecdbe4092dc803f3e91026fe81a65b64b372      # intel/auto-round main, 2026-09-30, version 0.17.0
AR_REPO="${AR_REPO:-https://github.com/intel/auto-round}"
AR_PATCH="${AR_PATCH:-1}"
AR_PATCH_FILE="$HERE/quantize/auto-round-local.patch"
case "$AR_PATCH" in
  1) [ -f "$AR_PATCH_FILE" ] || { echo "missing $AR_PATCH_FILE" >&2; exit 2; }
     ptag="p$( (sha256sum "$AR_PATCH_FILE" 2>/dev/null || shasum -a 256 "$AR_PATCH_FILE") | cut -c1-8)"
     AR_DESC="auto-round $AR_VERSION, intel/auto-round commit \`$AR_COMMIT\` + \`tools/quantize/auto-round-local.patch\` (sha256 ${ptag#p}...)" ;;
  0) ptag=plain
     AR_DESC="auto-round $AR_VERSION, intel/auto-round commit \`$AR_COMMIT\` (unpatched, AR_PATCH=0)" ;;
  *) echo "AR_PATCH must be 0 or 1, not '$AR_PATCH'" >&2; exit 2 ;;
esac
PKG="$HOME/.cache/auto-round-${AR_COMMIT:0:8}-$ptag-pkg"   # one directory per (commit, patch)
AR_BUILD="$HOME/.cache/auto-round-src"                    # the clone the package is built from

if [ ! -f "$PKG/.ar_done" ]; then
  command -v git >/dev/null || { echo "git is needed (auto-round's setup.py runs git describe)" >&2; exit 2; }
  [ -d "$AR_BUILD/.git" ] || git clone -q "$AR_REPO" "$AR_BUILD"
  git -C "$AR_BUILD" cat-file -e "$AR_COMMIT^{commit}" 2>/dev/null || git -C "$AR_BUILD" fetch -q --tags "$AR_REPO"
  git -C "$AR_BUILD" cat-file -e "$AR_COMMIT^{commit}" 2>/dev/null ||
    { echo "$AR_REPO does not contain $AR_COMMIT" >&2; exit 2; }
  git -C "$AR_BUILD" checkout -q -f --detach "$AR_COMMIT"     # -f: drops the last build's patch
  if [ "$AR_PATCH" = 1 ]; then
    git -C "$AR_BUILD" apply "$AR_PATCH_FILE"
    echo "auto-round-local.patch applied to $AR_COMMIT"
  fi
  UV=""
  for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" uv; do
    command -v "$c" >/dev/null 2>&1 && UV="$c" && break
  done
  mkdir -p "$PKG"
  if [ -n "$UV" ]; then
    "$UV" pip install -q --python "$VENVPY" --no-deps --target "$PKG" "$AR_BUILD"
  else
    "$VENVPY" -m pip install -q --no-deps --target "$PKG" "$AR_BUILD"
  fi
  git -C "$AR_BUILD" checkout -q -f --detach "$AR_COMMIT"     # leave the clone as committed
  echo "$AR_COMMIT $ptag" > "$PKG/.ar_done"
fi

cd /tmp
PYTHONPATH="$PKG" "$VENVPY" - "$AR_VERSION" "$PKG" "$AR_PATCH" <<'PY'
import importlib.util, inspect, sys
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
from auto_round.wrapper import WrapperLinear
patched = isinstance(inspect.getattr_static(WrapperLinear, "in_features", None), property)
assert patched == (sys.argv[3] == "1"), f"AR_PATCH={sys.argv[3]} but WrapperLinear.in_features patched={patched}"
print(f"auto-round {auto_round.__version__} from {auto_round.__file__}, local patch {'applied' if patched else 'not applied'}")
PY
