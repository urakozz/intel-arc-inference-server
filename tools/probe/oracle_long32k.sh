#!/bin/bash
# Spec 6 K3a (2026-09-26 ruling): the long-context oracle for flash_long_test, run ON THE
# BOX from the repo root, detached (tools/probe/detach.sh; hours of CPU).
#   1. a check that the chunked-cache forward equals one forward: 3000 ids, chunk 1000 vs
#      chunk 4096 (one chunk), last-row logits compared;
#   2. the real set: tests/golden/prompts/long32k.ids at 4096,8192,16384,32704, chunk 2048,
#      into oracle-out-long32k/last_logits.safetensors (excluded from box.sh's rsync).
cd "$(dirname "$0")/../.."
mkdir -p oracle-out-long32k
tools/oracle/run_in_container.sh '
python3 tools/oracle/last_logits.py "$SNAP" --prompt tests/golden/prompts/long32k.ids \
  --out /scratch/ll_c1000.safetensors --at 3000 --chunk 1000 &&
python3 tools/oracle/last_logits.py "$SNAP" --prompt tests/golden/prompts/long32k.ids \
  --out /scratch/ll_c4096.safetensors --at 3000 --chunk 4096 &&
python3 -c "
from safetensors.torch import load_file
import torch
a = load_file(\"/scratch/ll_c1000.safetensors\")[\"logits\"][0].double()
b = load_file(\"/scratch/ll_c4096.safetensors\")[\"logits\"][0].double()
cs = float(a @ b / (a.norm() * b.norm()))
print(f\"CHUNK CHECK at 3000 ids: chunked-cache vs one forward cos {cs:.9f}, argmax {int(a.argmax())} / {int(b.argmax())}, max abs {float((a - b).abs().max()):.3e}\")
" &&
python3 tools/oracle/last_logits.py "$SNAP" --prompt tests/golden/prompts/long32k.ids \
  --out /ws/oracle-out-long32k/last_logits.safetensors --at 4096,8192,16384,32704 --chunk 2048'
