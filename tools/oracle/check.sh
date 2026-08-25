#!/usr/bin/env bash
# Re-reads the three golden files in a separate process from the one that wrote
# them - the writer's own asserts are not evidence. Checks `tokens` length and
# distinctness, `resid.L63`/`logits` finiteness and range, the metadata, and
# decodes the continuation back to text through tokenize.py. Run from the repo
# root ON THE BOX, after tools/oracle/golden.sh. Output is the block quoted in
# "Sanity checks on the written files" in tools/oracle/README.md.
#
# OUT_DIR selects the golden set, and the checkpoint must be the one that set
# was dumped from -- the decode step at the end reads the tokenizer out of
# $SNAP, and the two checkpoints' tokenizers are identical, but the ids being
# decoded are not. Same env as golden.sh:
#   OUT_DIR=oracle-out-rtn \
#   ORACLE_SNAP=$HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64 \
#   tools/oracle/check.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT_DIR="${OUT_DIR:-oracle-out}"
cat > "$OUT_DIR"/_check.py <<'PY'
import sys, torch
from safetensors import safe_open
path, ids_out = sys.argv[1], sys.argv[2]
with safe_open(path, framework="pt") as f:
    md = f.metadata()
    tok = f.get_tensor("tokens").tolist()
    print(f"  tokens: len={len(tok)} distinct={len(set(tok))} degenerate={len(set(tok)) == 1}")
    print(f"  ids: {tok}")
    for n in ("resid.L63", "logits"):
        t = f.get_tensor(n).float()
        print(f"  {n}: shape={tuple(t.shape)} finite={bool(torch.isfinite(t).all())} "
              f"min={t.min():.4f} max={t.max():.4f}")
    print(f"  metadata: n_prompt={md['n_prompt']} gen={md['gen']} "
          f"attn={md['attn_implementation']} group_size={md['group_size']}")
open(ids_out, "w").write(" ".join(str(i) for i in tok))
PY
for p in prose code cjk; do
  echo "=== $p  ($(stat -c %s "$OUT_DIR/$p".golden.safetensors) bytes)"
  tools/oracle/run_in_container.sh 'python3 -P /ws/'"$OUT_DIR"'/_check.py /ws/'"$OUT_DIR/$p"'.golden.safetensors /ws/'"$OUT_DIR/$p"'.tokens 2>/dev/null
     python3 tools/oracle/tokenize.py "$SNAP" decode $(cat /ws/'"$OUT_DIR/$p"'.tokens) 2>/dev/null > /ws/'"$OUT_DIR/$p"'.cont.txt
     echo "  decoded:"; sed "s/^/  | /" /ws/'"$OUT_DIR/$p"'.cont.txt
     python3 -P -c "import sys; print(\"  repr:\", repr(open(sys.argv[1]).read()))" /ws/'"$OUT_DIR/$p"'.cont.txt'
done
