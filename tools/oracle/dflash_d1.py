#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 19 §6 D1 (plan 19b Task 3), the reference's half: every draft tools/probe/dflash_d1 wrote (the engine's
DFlash drafter on the card) against tools/oracle/dflash_ref.py drafting from the SAME taps, anchors and head.

    dflash_d1.py --d1 OUT.jsonl --dumps DIR --drafter DIR --target DIR [--md FILE] [--threads N]

The reference is built to the engine's arm (the run line of OUT.jsonl): the drafter bf16 as shipped, or int8 per-row
RTN simulated (dflash_accept.int8_simulate - spec 9's quantiser, which the loader applies at load); the target's
lm_head bf16 or int8 RTN (dflash_accept.Head); the head's rows >= vocab_used masked (argmax.cl's rule, the engine's
top-16 applies it); with a draft vocabulary every id outside V' masked (the engine's compact head is the full head's
rows at V''s ids, bitwise - spec 8 §11). Context: dump_taps.load_dump's taps from tap_from, context_kv over them,
draft_block(anchor, p, K) at T 0.

D1's bars, per draft row:
  sets    the engine's 16 candidates equal the reference's, tie-aware and noise-aware: an id only one side has must
          sit within max(1e-3, 1e-4 x |v|, d_row) of the reference's 16th logit, where d_row is that row's own
          engine-vs-reference logit difference (the largest |engine unary - reference unary| over the row's common
          candidates). The noise term is the operator's ruling (2026-10-27, the spirit of the MoE gates' f32-judge
          near-tie): measured on the card (plan 19f) every set miss was a swap at the 16th place, the swapped id a
          median 0.012 (max 0.09) logits from the reference's 16th logit, while the engine's logits (bf16 GEMV inputs,
          spec 19 §10) differ from the fp32 reference's by a median 0.078 a row - against a fixed window of ~1.4e-3.
          The ties used are printed: all of them, and those only the row's difference excused.
  scores  per draft the realized scores' cosine (aligned by candidate id) >= 0.9999 (bf16 drafter) / 0.999 (int8)
  walk    the K draft ids equal (T = 0); a split whose selector-score margin is within max(the row's own
          engine-vs-reference score noise, 1 bf16 ulp of the top score) is a near-tie, the cosine then over the
          rows before the split only (operator, 2026-10-29: the 6 splits measured were 0.005-0.09 at noise
          0.03-0.12, scores 12-36)
A draft passes on walk and scores; a top-16 set miss beyond the tie window is reported but passes when the walk
is equal and the scores' cosine meets its bar (operator, 2026-10-28: the misses measured were 16th-place swaps
0.007-0.01 logits past d_row, cosine 0.999996). Exit 0 when every draft passes. Prints a table (and writes it to --md).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]   # tools/oracle/tokenize.py


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cosine(a, b) -> float:
    ab = sum(x * y for x, y in zip(a, b))
    aa = sum(x * x for x in a)
    bb = sum(y * y for y in b)
    if aa == 0 and bb == 0:
        return 1.0
    return ab / math.sqrt(aa * bb)


def compare(rec: dict, out, logits_row) -> dict:
    """One draft: the engine's record against the reference's DraftOut. logits_row(j) -> the reference's masked
    (and scaled) unary logits of mask row j over the whole vocabulary (to judge an engine-only candidate)."""
    K = rec["k"]
    res = {"sets": True, "walk": True, "ties": 0, "noise_ties": 0, "gaps": [], "walk_tie": None}
    eng_u = rec.get("unary")
    pairs = []          # per row: the realized scores aligned by candidate id, (engine, reference)
    split = None        # the first row whose walk id differs
    for j in range(K):
        e = rec["cand"][j]
        r = out.cand_ids[j].tolist()
        ru = out.unary[j].tolist()
        edge = ru[-1]
        tol = max(1e-3, 1e-4 * abs(edge))
        only_e = [i for i in e if i not in r]
        only_r = [i for i in r if i not in e]
        if only_e or only_r:
            lr = logits_row(j)
            gap = max([abs(float(lr[i]) - edge) for i in only_e] + [abs(ru[r.index(i)] - edge) for i in only_r])
            # d_row: this row's measured engine-vs-reference logit difference (operator, 2026-10-27; the docstring)
            d_row = 0.0
            if eng_u is not None:
                d_row = max([abs(v - ru[r.index(c)]) for c, v in zip(e, eng_u[j]) if c in r], default=0.0)
            ok = gap <= max(tol, d_row)
            res["sets"] &= ok
            res["ties"] += int(ok)
            res["noise_ties"] += int(ok and gap > tol)
            res["gaps"].append((gap, d_row))
        rs = dict(zip(r, out.scores[j].tolist()))
        pairs.append([(s, rs[c]) for c, s in zip(e, rec["scores"][j]) if c in rs])
        if split is None and rec["ids"][j] != int(out.ids[j]):
            split = j
    res["walk"] = split is None
    if split is not None:
        # Operator ruling 2026-10-29: a walk split whose selector-score margin is within max(the row's own
        # engine-vs-reference score noise, one bf16 ulp of the top score) is a near-tie. Up to the split both walks
        # stand on the same predecessor, so row `split`'s scores are comparable: the reference's margin of its pick
        # over the engine's under its own scores (the engine's margin when its pick is outside the reference's
        # candidates).
        j = split
        r = out.cand_ids[j].tolist()
        S = out.scores[j].tolist()
        e_id, r_id = rec["ids"][j], int(out.ids[j])
        es = dict(zip(rec["cand"][j], rec["scores"][j]))
        if e_id in r:
            margin = S[r.index(r_id)] - S[r.index(e_id)]
        elif e_id in es and r_id in es:
            margin = es[e_id] - es[r_id]
        else:
            margin = math.inf   # neither side scores both picks: not a tie
        noise = max([abs(a - b) for a, b in pairs[j]], default=0.0)
        top = max(S)
        e = 0
        if top != 0:
            e = math.frexp(abs(top))[1]
        ulp = math.ldexp(1.0, e - 8)
        res["walk_tie"] = (abs(margin) <= max(noise, ulp), split, abs(margin), noise, ulp)
    # The scores' cosine over the rows before the split only (after it the two walks stand on other paths).
    rows = pairs if split is None else pairs[:split]
    res["cos"] = cosine([a for p in rows for a, _ in p], [b for p in rows for _, b in p])
    return res


def reference_drafter(run: dict, drafter_dir: str, target_dir: str):
    """dflash_ref's drafter built to the engine's arm (the module docstring), its head_logits masked."""
    import torch
    ref = _load("dflash_ref")
    acc = _load("dflash_accept")
    acc._load_deps()
    d = ref.load_drafter(drafter_dir, target_dir)
    if run["format"] == "int8":
        acc.int8_simulate(d)
    head = acc.Head(d.lm_head, int8=run["head"] == "int8")
    used = int(run["vocab_used"])
    vmask = None
    if run["draft_vocab"]:
        vmask = torch.ones(d.cfg.vocab, dtype=torch.bool)
        vmask[torch.tensor(run["v_ids"], dtype=torch.long)] = False

    def head_logits(h):
        lg = head.logits(h)
        lg[:, used:] = float("-inf")
        if vmask is not None:
            lg[:, vmask] = float("-inf")
        return lg
    d.head_logits = head_logits
    return d


def evaluate(d1_path: str, dumps_dir: str, drafter_dir: str, target_dir: str, md: str | None = None,
             log=print) -> tuple[bool, dict]:
    import torch
    dt = _load("dump_taps")
    with open(d1_path) as f:
        lines = [json.loads(x) for x in f if x.strip()]
    run = lines[0]["run"]
    recs = lines[1:]
    d = reference_drafter(run, drafter_dir, target_dir)
    bar = 0.9999 if run["format"] == "bf16" else 0.999

    by_dump = {}
    for r in recs:
        by_dump.setdefault(r["dump"], []).append(r)
    files = {}
    for path in glob.glob(os.path.join(dumps_dir, "*.taps.safetensors")):
        files[dt.dump_header(path)["name"]] = path
    n = n_sets = n_walk = n_cos = ties = noise_ties = set_excused = walk_ties = 0
    worst = 1.0
    fails = []
    for name, rs in by_dump.items():
        dump = dt.load_dump(files[name])
        pos = torch.arange(dump["tap_from"], dump["n"])
        ctx = d.context_kv({i: dump["taps"][i] for i in d.cfg.target_layer_ids}, pos)
        for r in rs:
            out = d.draft_block(r["anchor"], r["p"], r["k"], ctx)

            def logits_row(j, out=out):
                lg = d.head_logits(out.hidden[1 + j:2 + j])[0] * d.cfg.output_multiplier \
                    if d.cfg.version == 2 else d.head_logits(out.hidden[1 + j:2 + j])[0] * d.cfg.logit_scale
                if d.cfg.version == 2 and d.cfg.final_logit_softcapping > 0:
                    cap = d.cfg.final_logit_softcapping
                    lg = torch.tanh(lg / cap) * cap
                return lg
            c = compare(r, out, logits_row)
            n += 1
            n_sets += c["sets"]
            wt = c["walk_tie"]
            if wt is not None and wt[0]:
                walk_ties += 1
                log(f"  walk near-tie {name} p {r['p']}: split at row {wt[1]}, margin {wt[2]:.5f} <= max(row noise "
                    f"{wt[3]:.5f}, 1 bf16 ulp {wt[4]:.4f}); cos over the rows before it {c['cos']:.7f}")
            walk_ok = c["walk"] or (wt is not None and wt[0])
            n_walk += walk_ok
            n_cos += c["cos"] >= bar
            ties += c["ties"]
            noise_ties += c["noise_ties"]
            worst = min(worst, c["cos"])
            if walk_ok and c["cos"] >= bar:
                set_excused += not c["sets"]
            else:
                fails.append((name, r["p"], c))
        log(f"{name}: {len(rs)} drafts")
    table = (f"| arm | drafts | top-16 sets (tie- and noise-aware) | scores cos >= {bar} | walk equal (or a near-tie split) | worst cos | "
             f"ties excused | of them by the row's logit difference |\n"
             f"|---|---:|---:|---:|---:|---:|---:|---:|\n"
             f"| drafter {run['format']}, head {run['head']}, V' {run['draft_vocab'] or 'full'} | {n} | {n_sets} | "
             f"{n_cos} | {n_walk} | {worst:.7f} | {ties} | {noise_ties} |\n")
    log(table)
    for name, p, c in fails[:20]:
        gaps = ", ".join(f"gap {g:.4f} > d_row {dr:.4f}" for g, dr in c["gaps"] if g > max(dr, 1e-3))
        log(f"  FAIL {name} p {p}: sets {c['sets']}, walk {c['walk']}, cos {c['cos']:.7f}"
            + (f" ({gaps})" if gaps else ""))
    if md:
        with open(md, "a") as f:
            f.write(table)
    ok = n > 0 and n_walk == n and n_cos == n
    log(f"D1 ties used: {ties} ({noise_ties} by the row's logit difference, operator 2026-10-27)")
    log(f"D1 set misses passed on walk + cosine: {set_excused} (operator 2026-10-28)")
    log(f"D1 walk splits passed as near-ties: {walk_ties} (operator 2026-10-29)")
    log("D1 " + ("PASS" if ok else "FAIL") + f" ({run.get('load', '')})")
    return ok, {"n": n, "sets": n_sets, "walk": n_walk, "cos": n_cos, "worst": worst, "ties": ties,
                "noise_ties": noise_ties, "set_excused": set_excused,
                "walk_ties": walk_ties}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1", required=True)
    ap.add_argument("--dumps", required=True, help="the directory of the *.taps.safetensors the engine read")
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--target", required=True, help="the target snapshot (its embed_tokens and lm_head)")
    ap.add_argument("--md")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    ok, _ = evaluate(a.d1, a.dumps, a.drafter, a.target, a.md, log=lambda s: print(s, flush=True))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
