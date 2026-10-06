# Spec 20a - Kolibri-1 reference, transformers model class, quantisation script

Spec: `docs/superpowers/specs/2026-10-05-spec20-kolibri-1-design.md` (§1, §3.1, §5 KL0, §6, §10 as
built). Branch `spec20a-kolibri`. No card, no 156 GB model on the Mac: tiny random weights plus the
small files (config, tokenizer, index, shard headers by range request).

**Status (2026-10-06): done on the Mac;** the real-model half of KL0 and the 20b run need the weights.

| task | files | gate | status |
|---|---|---|---|
| 1. Facts | `docs/probe-kolibri-2026-10-05.md` | config / index / headers / template / plugin read; `head_dtype`, window edge, tool-call format, EOS pinned | done |
| 2. Model class | `tools/oracle/third_party/kolibri1/` | names = checkpoint, per-expert `nn.Linear`, `trust_remote_code` via `auto_map`, cached `generate()` | done |
| 3. Reference + KL0 | `tools/oracle/kolibri_ref.py`, `test_kolibri_ref.py` | 21 tests in `agnes-ref-img`; port == reference bitwise (bf16 eager, fp32) | done (tiny); real-model perplexity pending |
| 4. Quantisation script | `tools/quantize_kolibri1.sh`, `tools/quantize/kolibri/` | `--dry-run` prints every stage; tiny end to end calib -> card; `test_kolibri_quant.py` | done |
| 5. Docs | `tools/quantize/README.md`, spec 20 §3.1 tool row + §10, this plan, box queue row 20 | - | done |

Run the checks (repo root, Mac):

```sh
docker run --rm --memory 28g --memory-swap 28g -e OMP_NUM_THREADS=3 -v "$PWD":/ws -w /ws \
  agnes-ref-img:latest python3 tools/oracle/test_kolibri_ref.py
docker run --rm --memory 8g -e OMP_NUM_THREADS=3 -v "$PWD":/ws -w /ws \
  agnes-ref-img:latest python3 tools/quantize/kolibri/test_kolibri_quant.py
tools/quantize_kolibri1.sh --dry-run
```

Next (20b): decide where it runs (spec 20 decision 1), `tools/quantize_kolibri1.sh` there, then
`kolibri_ref.py ppl` on German and English text for KL0's second half.
