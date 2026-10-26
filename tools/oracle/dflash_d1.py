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
  sets    the engine's 16 candidates equal the reference's, tie-aware: an id only one side has must sit within
          max(1e-3, 1e-4 x |v|) of the reference's 16th logit (an exact or near tie at the boundary)
  scores  per draft the realized scores' cosine (aligned by candidate id) >= 0.9999 (bf16 drafter) / 0.999 (int8)
  walk    the K draft ids equal (T = 0)
Exit 0 when every draft passes all three. Prints a table (and writes it to --md).
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
    res = {"sets": True, "walk": True, "ties": 0}
    eng_s, ref_s = [], []
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
            ok = all(abs(float(lr[i]) - edge) <= tol for i in only_e) and \
                all(abs(ru[r.index(i)] - edge) <= tol for i in only_r)
            res["sets"] &= ok
            res["ties"] += int(ok)
        rs = dict(zip(r, out.scores[j].tolist()))
        for c, s in zip(e, rec["scores"][j]):
            if c in rs:
                eng_s.append(s)
                ref_s.append(rs[c])
        res["walk"] &= rec["ids"][j] == int(out.ids[j])
    res["cos"] = cosine(eng_s, ref_s)
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
    n = n_sets = n_walk = n_cos = ties = 0
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
            n_walk += c["walk"]
            n_cos += c["cos"] >= bar
            ties += c["ties"]
            worst = min(worst, c["cos"])
            if not (c["sets"] and c["walk"] and c["cos"] >= bar):
                fails.append((name, r["p"], c))
        log(f"{name}: {len(rs)} drafts")
    table = (f"| arm | drafts | top-16 sets (tie-aware) | scores cos >= {bar} | walk equal | worst cos | ties excused |\n"
             f"|---|---:|---:|---:|---:|---:|---:|\n"
             f"| drafter {run['format']}, head {run['head']}, V' {run['draft_vocab'] or 'full'} | {n} | {n_sets} | "
             f"{n_cos} | {n_walk} | {worst:.7f} | {ties} |\n")
    log(table)
    for name, p, c in fails[:20]:
        log(f"  FAIL {name} p {p}: sets {c['sets']}, walk {c['walk']}, cos {c['cos']:.7f}")
    if md:
        with open(md, "a") as f:
            f.write(table)
    ok = n > 0 and n_sets == n and n_walk == n and n_cos == n
    log("D1 " + ("PASS" if ok else "FAIL") + f" ({run.get('load', '')})")
    return ok, {"n": n, "sets": n_sets, "walk": n_walk, "cos": n_cos, "worst": worst, "ties": ties}


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
