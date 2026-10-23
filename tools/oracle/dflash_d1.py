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

Spec 23f: a run line with "form": "muse" (tools/probe/dflash_d1 beside a Muse-Glimmer target) is held to
dflash_ref.py's Muse-Glimmer form (bitwise transformers 5.19.0) in float32: the assistant bf16 as shipped, int8 per-row
RTN or int4 g64 RTN simulated (dflash_accept.int8_simulate / int4g64_simulate - the loader's quantisers), the
target's head bf16 or int8 RTN; the whole 16-row block per draft (draft_block), the head's RAW logits on mask rows
1..K (or, "q": "capped", 20 tanh(z x 0.19611613513818404 / 20) - the order is the raw one); greedy = argmax (the lowest
id among equals). The bars are the same three (scores = the unary logits).

Spec 26b: a run line with "form": "lilicorr" (tools/probe/dflash_d1 --model lilicorr beside a Bolt target, or the host twin
tests/kernels/lilicorr_d1_twin) is held to dflash_ref.py's LiLiCorr form (26a's port of vLLM's models/lilicorr.py), float32:
the drafter at the run's format - exact (W' + g, as decoded), int8 (dflash_accept.lil_int8_simulate: W''s codes, the row
scale fp32(g x s')) or int4 g64 (lil_int4_simulate: rtn from g x W' in fp64) - its head always exact; the target's head
bf16 or int8 RTN (dflash_accept.Head); the whole 8-row block, all 7 slots, T 0. D1's bars per draft (spec 26 §7):
  sets     each slot's 8 candidates equal the reference's, tie-aware (an id one side lacks within max(1e-3, 1e-4 x |v|) of
           the reference's 8th raw logit)
  lp       the candidates' log-probs (aligned by id) within 1e-3
  lattice  the lattice's cosine >= 0.9999 (exact) / 0.999 (int8, int4) - compared where both sides hold the same candidates
           in the same order (another candidate list is another lattice; those drafts are counted, not compared)
  walk     the 7 ids equal (T 0)
  anchor   the anchor row the head read (the engine's anchor buffer) cosine >= 0.9999 to the reference's hidden_norm(fc(taps))
           of p - 1
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


MUSE_CAP_MULT, MUSE_CAP = 0.19611613513818404, 20.0   # Muse-Glimmer-30B's text_config (spec 23 §1)


def muse_reference(run: dict, drafter_dir: str, target_dir: str):
    """Spec 23f: (d, draft) - dflash_ref's Muse-form drafter at the engine's arm, and draft(anchor, p, K, ctx) ->
    (out, logits_row) with out's cand_ids / unary / scores / ids from the arm's head over the block's rows 1..K."""
    import torch
    from types import SimpleNamespace
    ref = _load("dflash_ref")
    acc = _load("dflash_accept")
    acc._load_deps()
    d = ref.load_drafter(drafter_dir, target_dir, dtype=torch.float32)
    if d.cfg.form != "muse":
        raise SystemExit(f"the run line says form muse; {drafter_dir} is not a Muse-Glimmer assistant")
    if run["format"] == "int8":
        acc.int8_simulate(d)
    elif run["format"] in ("int4", "int4g64"):
        acc.int4g64_simulate(d)
    elif run["format"] != "bf16":
        raise SystemExit(f"format {run['format']}: bf16, int8 or int4")
    head = acc.Head(d.lm_head, int8=run["head"] == "int8")
    capped = run.get("q", "raw") == "capped"

    def draft(anchor, p, K, ctx):
        out = d.draft_block(anchor, p, K, ctx)
        lg = head.logits(out.hidden[1:K + 1].float())
        ids = acc.greedy_ids(lg)
        cand, unary = ref.top_k_stable(lg, 16)
        if capped:
            unary = torch.tanh(unary * MUSE_CAP_MULT / MUSE_CAP) * MUSE_CAP
            lg = torch.tanh(lg * MUSE_CAP_MULT / MUSE_CAP) * MUSE_CAP
        o = SimpleNamespace(cand_ids=cand, unary=unary, scores=unary.clone(), ids=ids, hidden=out.hidden)
        return o, (lambda j: lg[j])
    return d, draft


def lilicorr_reference(run: dict, drafter_dir: str, target_dir: str):
    """Spec 26b: dflash_ref's LiLiCorr drafter at the engine's arm (the module docstring), its head_logits the arm's head."""
    ref = _load("dflash_ref")
    acc = _load("dflash_accept")
    acc._load_deps()
    d = ref.load_drafter(drafter_dir, target_dir)
    if d.cfg.form != "lilicorr":
        raise SystemExit(f"the run line says form lilicorr; {drafter_dir} is not a LiLiCorr drafter")
    if run["format"] == "int8":
        acc.lil_int8_simulate(d)
    elif run["format"] in ("int4", "int4g64"):
        acc.lil_int4_simulate(d)
    elif run["format"] != "exact":
        raise SystemExit(f"format {run['format']}: exact, int8 or int4")
    head = acc.Head(d.lm_head, int8=run["head"] == "int8")
    import torch
    # The engine's head reads the final norm's bf16 copy (df_norm's x - the verify list's head reads the same): the arm's
    # head input, not a drafter error, so the reference rounds it too (--fp32-head-input keeps fp32, the plain port).
    if run.get("head_input", "bf16") == "bf16":
        d.head_logits = lambda h: head.logits(h.float().to(torch.bfloat16).float())
    else:
        d.head_logits = head.logits
    return d


def compare_lilicorr(rec: dict, out, logits_row, bar: float, lp_bar: float = 1e-3) -> dict:
    """One LiLiCorr draft (the module docstring's five bars)."""
    res = {"sets": True, "walk": True, "ties": 0, "lp": 0.0, "lattice": None, "anchor": 1.0}
    same = True
    for s in range(len(rec["cand"])):
        e, r = rec["cand"][s], out.cand_ids[s].tolist()
        rv = [float(x) for x in logits_row(s)[out.cand_ids[s]].tolist()]
        edge = rv[-1]
        tol = max(1e-3, 1e-4 * abs(edge))
        only_e = [i for i in e if i not in r]
        only_r = [i for i in r if i not in e]
        if only_e or only_r:
            lr = logits_row(s)
            ok = all(abs(float(lr[i]) - edge) <= tol for i in only_e) and \
                all(abs(rv[r.index(i)] - edge) <= tol for i in only_r)
            res["sets"] &= ok
            res["ties"] += int(ok)
        same &= e == r
        rl = dict(zip(r, out.log_probs[s].tolist()))
        for c, v in zip(e, rec["lp"][s]):
            if c in rl:
                res["lp"] = max(res["lp"], abs(v - rl[c]))
        res["walk"] &= rec["ids"][s] == int(out.ids[s])
    if same:
        flat = [x for sl in rec["lattice"] for row in sl for x in row]
        res["lattice"] = cosine(flat, out.lattice.reshape(-1).tolist())
    res["anchor"] = cosine(rec["anchor_row"], out.anchor.tolist())
    res["ok"] = res["sets"] and res["walk"] and res["lp"] <= lp_bar and res["anchor"] >= 0.9999 and \
        (res["lattice"] is None or res["lattice"] >= bar)
    return res


def evaluate_lilicorr(run: dict, recs: list, dumps_dir: str, drafter_dir: str, target_dir: str, md: str | None,
                      log, lp_bar: float = 1e-3) -> tuple[bool, dict]:
    import torch
    dt = _load("dump_taps")
    d = lilicorr_reference(run, drafter_dir, target_dir)
    bar = 0.9999 if run["format"] == "exact" else 0.999
    files = {dt.dump_header(path)["name"]: path for path in glob.glob(os.path.join(dumps_dir, "*.taps.safetensors"))}
    by_dump = {}
    for r in recs:
        by_dump.setdefault(r["dump"], []).append(r)
    n = n_ok = n_sets = n_walk = n_lat = n_cmp = ties = 0
    worst_lat, worst_anchor, worst_lp = 1.0, 1.0, 0.0
    fails = []
    for name, rs in by_dump.items():
        dump = dt.load_dump(files[name])
        pos = torch.arange(dump["tap_from"], dump["n"])
        ctx = d.context_kv({i: dump["taps"][i] for i in d.cfg.target_layer_ids}, pos)
        for r in rs:
            out = d.draft_block(r["anchor"], r["p"], r["k"], ctx)
            c = compare_lilicorr(r, out, lambda j, out=out: d.head_logits(out.hidden[1 + j:2 + j])[0], bar, lp_bar)
            n += 1
            n_ok += c["ok"]
            n_sets += c["sets"]
            n_walk += c["walk"]
            ties += c["ties"]
            worst_lp = max(worst_lp, c["lp"])
            worst_anchor = min(worst_anchor, c["anchor"])
            if c["lattice"] is not None:
                n_cmp += 1
                n_lat += c["lattice"] >= bar
                worst_lat = min(worst_lat, c["lattice"])
            if not c["ok"]:
                fails.append((name, r["p"], c))
        log(f"{name}: {len(rs)} drafts")
    table = (f"| arm | drafts | sets (tie-aware) | lp within {lp_bar:g} (worst) | lattice cos >= {bar} (compared) | walk equal | "
             f"anchor cos (worst) | ties excused |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
             f"| LiLiCorr {'twin ' + run.get('head_input', 'bf16') + ' head input, ' if run.get('twin') else ''}drafter "
             f"{run['format']}, head {run['head']} | {n} | {n_sets} | {worst_lp:.2e} | "
             f"{n_lat} / {n_cmp} (worst {worst_lat:.7f}) | {n_walk} | {worst_anchor:.7f} | {ties} |\n")
    log(table)
    for name, p, c in fails[:20]:
        log(f"  FAIL {name} p {p}: sets {c['sets']}, walk {c['walk']}, lp {c['lp']:.2e}, lattice {c['lattice']}, "
            f"anchor {c['anchor']:.7f}")
    if md:
        with open(md, "a") as f:
            f.write(table)
    ok = n > 0 and n_ok == n
    log("D1 " + ("PASS" if ok else "FAIL") + f" ({run.get('load', '')})")
    return ok, {"n": n, "ok": n_ok, "sets": n_sets, "walk": n_walk, "lattice": n_lat, "compared": n_cmp,
                "worst_lattice": worst_lat, "worst_anchor": worst_anchor, "worst_lp": worst_lp, "ties": ties}


def evaluate(d1_path: str, dumps_dir: str, drafter_dir: str, target_dir: str, md: str | None = None,
             log=print, lp_bar: float = 1e-3) -> tuple[bool, dict]:
    import torch
    dt = _load("dump_taps")
    with open(d1_path) as f:
        lines = [json.loads(x) for x in f if x.strip()]
    run = lines[0]["run"]
    recs = lines[1:]
    if run.get("form") == "lilicorr":   # spec 26b
        return evaluate_lilicorr(run, recs, dumps_dir, drafter_dir, target_dir, md, log, lp_bar)
    muse = run.get("form") == "muse"
    if muse:
        d, muse_draft = muse_reference(run, drafter_dir, target_dir)
    else:
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
            if muse:
                out, logits_row = muse_draft(r["anchor"], r["p"], r["k"], ctx)
                c = compare(r, out, logits_row)
                n += 1
                n_sets += c["sets"]
                n_walk += c["walk"]
                n_cos += c["cos"] >= bar
                ties += c["ties"]
                worst = min(worst, c["cos"])
                if not (c["sets"] and c["walk"] and c["cos"] >= bar):
                    fails.append((name, r["p"], c))
                continue
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
             f"| {'Muse-Glimmer ' if muse else ''}drafter {run['format']}, head {run['head']}"
             f"{', q ' + run.get('q', 'raw') if muse else ''}, V' {run['draft_vocab'] or 'full'} | {n} | {n_sets} | "
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
    ap.add_argument("--lp-bar", type=float, default=1e-3, help="spec 26b: the LiLiCorr log-probs' bar (spec 26 §7: 1e-3)")
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    ok, _ = evaluate(a.d1, a.dumps, a.drafter, a.target, a.md, log=lambda s: print(s, flush=True), lp_bar=a.lp_bar)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
