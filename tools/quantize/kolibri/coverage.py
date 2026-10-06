#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage `coverage` of tools/quantize_kolibri1.sh (spec 20 §3.1 "expert coverage").

    coverage.py --model <snapshot> --calib <dir>/calib.safetensors --out <dir> [--floor 2048] [--device cpu]

Routes every calibration row through the bf16 model (tools/oracle/kolibri_ref.py run_batch, layer
by layer, each layer's weights read once) and counts, per layer and expert, the tokens routed to it
(top-6 on logit + expert_bias, the model's own rule). Every expert under --floor gets targeted German
text: candidate rows of German Wikipedia articles (shard 00000 of the pinned revision, articles the
calibration did not use), chat-formatted as user turns, are routed too, and rows are added greedily
- each time the one that fills the most remaining deficit - until every expert clears the floor.
If that cannot be done in --rounds rounds of --pool candidate rows the stage FAILS (exit 3) and names
the starved experts; nothing is written for the next stage.

Writes coverage.json (counts before / after per layer x expert, the per-category counts, the
deficits, the top-up rows' article ids), coverage.txt (a summary) and calib_final.safetensors (the
calibration rows + the top-up rows, category de_topup) - what `rtn` and `tune` read.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


class Counter:
    """on_route for run_batch: counts per (layer, expert), per category, and optionally per row."""

    def __init__(self, L: int, E: int, row_cat: list, cats: list, per_row: bool = False):
        self.E = E
        self.total = torch.zeros(L, E, dtype=torch.long)
        self.cats = cats
        self.row_cat = torch.tensor([cats.index(c) for c in row_cat])
        self.by_cat = torch.zeros(len(cats), L, E, dtype=torch.long)
        self.per_row = torch.zeros(len(row_cat), L, E, dtype=torch.int32) if per_row else None

    def __call__(self, layer: int, seq_ix: torch.Tensor, rt) -> None:
        ids = rt.ids.cpu()
        k = ids.shape[1]
        self.total[layer] += torch.bincount(ids.flatten(), minlength=self.E)
        cat = self.row_cat[seq_ix].repeat_interleave(k)
        flat = ids.flatten()
        for ci in torch.unique(cat).tolist():
            self.by_cat[ci, layer] += torch.bincount(flat[cat == ci], minlength=self.E)
        if self.per_row is not None:
            rows = seq_ix.repeat_interleave(k)
            for r in torch.unique(rows).tolist():
                self.per_row[r, layer] += torch.bincount(flat[rows == r], minlength=self.E).to(torch.int32)


def route_rows(ref, ids: torch.Tensor, row_cat: list, cats: list, rows_per_fwd: int, per_row: bool):
    c = ref.c
    cnt = Counter(c.num_hidden_layers, c.num_experts, row_cat, cats, per_row)
    ref.run_batch(ids, rows=rows_per_fwd, on_route=cnt, log=C.log)
    return cnt


def greedy_topup(deficit: torch.Tensor, per_row: torch.Tensor) -> list:
    """Indices of candidate rows, chosen one at a time by how much remaining deficit they fill
    (sum over layer x expert of min(deficit, the row's count)), until the deficit is gone or no row
    helps. deficit [L, E] >= 0 (modified in place); per_row [R, L, E]."""
    chosen, avail = [], torch.ones(per_row.shape[0], dtype=torch.bool)
    while int(deficit.sum()) > 0 and bool(avail.any()):
        gain = torch.minimum(per_row.long(), deficit[None]).sum((1, 2))
        gain[~avail] = -1
        best = int(gain.argmax())
        if int(gain[best]) <= 0:
            break
        chosen.append(best)
        avail[best] = False
        deficit -= torch.minimum(per_row[best].long(), deficit)
    return chosen


def summary(counts: torch.Tensor, floor: int) -> list:
    out = []
    for i in range(counts.shape[0]):
        x = counts[i].float()
        out.append({"layer": i, "min": int(x.min()), "median": float(x.median()), "max": int(x.max()),
                    "starved": int((counts[i] < floor).sum()), "unused": int((counts[i] == 0).sum())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the bf16 snapshot directory")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor", type=int, default=2048, help="minimum routed tokens per expert per layer")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rows-per-fwd", type=int, default=4)
    ap.add_argument("--pool", type=int, default=256, help="candidate top-up rows per round")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--doc-tokens", type=int, default=1800, help="max article tokens per top-up conversation")
    a = ap.parse_args()

    ref_mod = C.kolibri_ref()
    snapshot = C.resolve_model(a.model)
    src = ref_mod.Checkpoint(snapshot)
    if src.cfg.group_size is not None:
        raise SystemExit("coverage needs the bf16 model, not a quantised one")
    ids, cats, meta = C.load_rows(a.calib)
    T = ids.shape[1]
    ref = ref_mod.KolibriRef(src.cfg, src, mode="bf16", device=a.device)
    L, E = src.cfg.num_hidden_layers, src.cfg.num_experts
    C.log(f"routing {ids.shape[0]} calibration rows x {T} tokens through {L} layers ({a.device})")
    cnt = route_rows(ref, ids, cats, C.CATS, a.rows_per_fwd, per_row=False)
    before = cnt.total.clone()
    deficit = (a.floor - before).clamp(min=0)
    starved_before = [(i, e) for i, e in (deficit > 0).nonzero().tolist()]
    C.log(f"{len(starved_before)} of {L * E} experts under {a.floor} tokens; deficit {int(deficit.sum())} tokens")

    manifest_path = os.path.join(os.path.dirname(os.path.abspath(a.calib)), "calib_manifest.json")
    used = set(C.read_json(manifest_path).get("calib_doc_ids", [])) if os.path.exists(manifest_path) else set()
    tok = C.load_tokenizer(snapshot)
    docs = C.iter_docs("calib")
    added_ids, added_docs, by_cat_after = [], [], cnt.by_cat.clone()
    rnd, exhausted = 0, False
    while int(deficit.sum()) > 0 and rnd < a.rounds:
        convs, pool_docs = [], []
        for d in docs:
            if d["id"] in used:
                continue
            used.add(d["id"])
            body = d["text"]
            tids = tok(body, add_special_tokens=False)["input_ids"]
            if len(tids) > a.doc_tokens:
                body = tok.decode(tids[: a.doc_tokens])
            text = tok.apply_chat_template([{"role": "user", "content": f"{d['title']}\n\n{body}"}], tokenize=False,
                                           add_generation_prompt=False)
            convs.append(tok(text, add_special_tokens=False)["input_ids"])
            pool_docs.append(d["id"])
            rows, starts = C.pack(convs, a.pool, T)
            if len(rows) >= a.pool:
                break
        rows, starts = C.pack(convs, a.pool, T)
        if not rows:
            exhausted = True
            break
        C.log(f"top-up round {rnd}: routing {len(rows)} candidate rows of German articles")
        pc = route_rows(ref, torch.tensor(rows), ["de_topup"] * len(rows), C.CATS, a.rows_per_fwd, per_row=True)
        chosen = greedy_topup(deficit, pc.per_row)
        doc_at = [0]
        for s in starts:
            doc_at.append(doc_at[-1] + s)
        for r in chosen:
            added_ids.append(rows[r])
            added_docs.append(pool_docs[doc_at[r]:doc_at[r + 1]])
            by_cat_after[C.CATS.index("de_topup")] += pc.per_row[r].long()
        C.log(f"top-up round {rnd}: {len(chosen)} rows added, remaining deficit {int(deficit.sum())} tokens "
              f"over {int((deficit > 0).sum())} experts")
        rnd += 1

    after = by_cat_after.sum(0)
    starved_after = [(i, e, int(after[i, e])) for i, e in (after < a.floor).nonzero().tolist()]
    report = {
        "floor": a.floor, "rows": ids.shape[0], "seqlen": T, "topup_rows": len(added_ids), "rounds": rnd,
        "starved_before": len(starved_before), "starved_after": len(starved_after),
        "starved_after_list": starved_after[:2000], "topup_doc_ids": added_docs,
        "per_layer_before": summary(before, a.floor), "per_layer_after": summary(after, a.floor),
        "counts_before": before.tolist(), "counts_after": after.tolist(),
        "counts_by_cat": {c: by_cat_after[i].tolist() for i, c in enumerate(C.CATS) if int(by_cat_after[i].sum())},
    }
    C.write_json(os.path.join(a.out, "coverage.json"), report)
    lines = [f"floor {a.floor} tokens per expert per layer; {ids.shape[0]} rows + {len(added_ids)} top-up rows",
             f"starved before {len(starved_before)}, after {len(starved_after)} (of {L * E})",
             "layer  min-before  min-after  median-after  max-after  starved-before  starved-after"]
    for b, x in zip(report["per_layer_before"], report["per_layer_after"]):
        lines.append(f"{x['layer']:5d}  {b['min']:10d}  {x['min']:9d}  {x['median']:12.0f}  {x['max']:9d}  "
                     f"{b['starved']:14d}  {x['starved']:13d}")
    with open(os.path.join(a.out, "coverage.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:3]))
    if starved_after:
        _fail(starved_after, a.floor, rnd, exhausted)
    all_ids = torch.cat([ids, torch.tensor(added_ids, dtype=torch.long).view(-1, T)]) if added_ids else ids
    all_cats = cats + ["de_topup"] * len(added_ids)
    lang = {lg: sum(1 for c in all_cats if C.CAT_LANG[c] == lg) / len(all_cats) for lg in ("de", "en", "code")}
    C.save_rows(os.path.join(a.out, "calib_final.safetensors"), all_ids, all_cats,
                {**meta, "stage": "coverage", "rows": len(all_cats), "topup_rows": len(added_ids),
                 "floor": a.floor, "token_share": lang})
    C.log(f"calib_final.safetensors: {len(all_cats)} rows; token share {lang}")


def _fail(starved, floor, rounds, exhausted=False):
    why = " (no unused German documents left for another round)" if exhausted else ""
    print(f"COVERAGE FAILED: {len(starved)} experts stay under {floor} tokens after {rounds} top-up rounds{why}, "
          f"e.g. (layer, expert, tokens) {starved[:10]}.\ncalib_final.safetensors NOT written. More rounds "
          f"(COVERAGE_ROUNDS) or a lower floor (COVERAGE_FLOOR, recorded in the card) are the operator's call.",
          file=sys.stderr, flush=True)
    sys.exit(3)


if __name__ == "__main__":
    main()
