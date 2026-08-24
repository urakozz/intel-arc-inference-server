#!/usr/bin/env bash
# Re-reads the three golden files in a separate process from the one that wrote
# them - the writer's own asserts are not evidence. Checks `tokens` length and
# distinctness, `resid.L63`/`logits` finiteness and range, the metadata, and
# decodes the continuation back to text through tokenize.py. Run from the repo
# root ON THE BOX, after tools/oracle/golden.sh. Output is the block quoted in
# "Sanity checks on the written files" in tools/oracle/README.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
cat > oracle-out/_check.py <<'PY'
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
  echo "=== $p  ($(stat -c %s oracle-out/$p.golden.safetensors) bytes)"
  tools/oracle/run_in_container.sh 'python3 -P /ws/oracle-out/_check.py /ws/oracle-out/'"$p"'.golden.safetensors /ws/oracle-out/'"$p"'.tokens 2>/dev/null
     python3 tools/oracle/tokenize.py "$SNAP" decode $(cat /ws/oracle-out/'"$p"'.tokens) 2>/dev/null > /ws/oracle-out/'"$p"'.cont.txt
     echo "  decoded:"; sed "s/^/  | /" /ws/oracle-out/'"$p"'.cont.txt
     python3 -P -c "import sys; print(\"  repr:\", repr(open(sys.argv[1]).read()))" /ws/oracle-out/'"$p"'.cont.txt'
done
