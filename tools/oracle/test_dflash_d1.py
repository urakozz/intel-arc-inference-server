#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Checks for dflash_d1.py (spec 19b Task 3): the D1 comparator on a tiny random drafter (dflash_fixture.py's
weights), a tiny target (embedding / head) and a dump in dump_taps.py's format, all written to a temp dir.

    test_dflash_d1.py            (pytest also collects it)

1. Records made by the reference itself (dflash_ref.draft_block, the same arm) pass every bar.
2. A changed draft id fails the walk; a far candidate swapped into a row fails the sets; scaled scores fail the
   cosine bar.
3. A near tie at the 16th slot is excused (sets pass, counted as a tie); with the engine's unary on the record, a
   swap within the row's own logit difference is excused too (operator, 2026-10-27), one beyond it is not.
4. The int8 arm builds the reference with the int8-simulated drafter and head (records from it pass).
"""
import copy
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


d1 = _load("dflash_d1")
fx = _load("dflash_fixture")
dt = _load("dump_taps")


def _world(tmp, fmt="bf16", head="bf16"):
    from safetensors.torch import save_file
    torch.manual_seed(7)
    w = fx.weights()
    drafter = os.path.join(tmp, "drafter")
    target = os.path.join(tmp, "target")
    dumps = os.path.join(tmp, "dumps")
    for p in (drafter, target, dumps):
        os.makedirs(p, exist_ok=True)
    save_file({k: v.contiguous() for k, v in w.items()}, os.path.join(drafter, "model.safetensors"))
    with open(os.path.join(drafter, "config.json"), "w") as f:
        json.dump(fx.config(2), f)
    embed, hd = fx.bf(fx.V, fx.H, std=0.5), fx.bf(fx.V, fx.H, std=0.5)
    save_file({"model.embed_tokens.weight": embed, "lm_head.weight": hd}, os.path.join(target, "model.safetensors"))
    with open(os.path.join(target, "config.json"), "w") as f:
        json.dump({"architectures": ["Tiny"]}, f)
    n, n_prompt = 40, 24
    ids = torch.randint(0, fx.USED, (n,)).tolist()
    res = {"greedy": torch.zeros(n, dtype=torch.int32), "top_ids": torch.zeros(n, 4, dtype=torch.int32),
           "top_logits": torch.zeros(n, 4), "lse": torch.zeros(n),
           "taps": {i: fx.bf(n, fx.TH, std=1.0) for i in fx.TAPS}}
    dt.save_dump(os.path.join(dumps, "t.taps.safetensors"), "tiny/t", ids, n_prompt, 0, res, {})
    run = {"format": fmt, "head": head, "draft_vocab": 0, "vocab_used": fx.USED, "k": 7, "launches": [0, 0],
           "load": "test", "v_ids": []}
    # the "engine": the reference itself on the same arm
    d = d1.reference_drafter(run, drafter, target)
    dump = dt.load_dump(os.path.join(dumps, "t.taps.safetensors"))
    ctx = d.context_kv({i: dump["taps"][i] for i in d.cfg.target_layer_ids}, torch.arange(0, n))
    recs = []
    for p in (24, 28, 32):
        o = d.draft_block(ids[p], p, 7, ctx)
        recs.append({"dump": "tiny/t", "p": p, "anchor": ids[p], "k": 7, "ids": o.ids.tolist(),
                     "chosen": o.chosen.tolist(), "cand": o.cand_ids.tolist(), "unary": o.unary.tolist(),
                     "scores": o.scores.tolist()})
    return run, recs, dumps, drafter, target, d, ctx, ids


def _write(tmp, run, recs):
    path = os.path.join(tmp, "d1.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({"run": run}) + "\n")
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return path


def _eval(tmp, run, recs, dumps, drafter, target):
    return d1.evaluate(_write(tmp, run, recs), dumps, drafter, target, log=lambda s: None)


def test_self_passes_and_failures():
    with tempfile.TemporaryDirectory() as tmp:
        run, recs, dumps, drafter, target, _, _, _ = _world(tmp)
        ok, st = _eval(tmp, run, recs, dumps, drafter, target)
        assert ok and st["n"] == 3 and st["worst"] > 0.9999999, st
        bad = copy.deepcopy(recs)
        bad[1]["ids"][3] = (bad[1]["ids"][3] + 1) % 500
        ok, st = _eval(tmp, run, bad, dumps, drafter, target)
        assert not ok and st["walk"] == 2, st
        bad = copy.deepcopy(recs)
        far = next(i for i in range(500) if i not in bad[0]["cand"][2])
        bad[0]["cand"][2][15] = far
        ok, st = _eval(tmp, run, bad, dumps, drafter, target)
        assert not ok and st["sets"] == 2, st
        bad = copy.deepcopy(recs)
        bad[2]["scores"] = [[s * (1.5 if j % 2 else 0.5) for j, s in enumerate(row)] for row in bad[2]["scores"]]
        ok, st = _eval(tmp, run, bad, dumps, drafter, target)
        assert not ok and st["cos"] == 2, st


def test_near_tie_excused():
    # compare() on a planted row: the reference's 16 candidates 0..15 with unary 30 .. 15 (the 16th at 15.0); the
    # engine has id 99 in the 16th slot. Excused when the reference's logit of 99 is within the tolerance of 15.0,
    # a failure when it is not.
    from types import SimpleNamespace
    cand = torch.arange(16).view(1, 16)
    unary = torch.arange(30, 14, -1, dtype=torch.float32).view(1, 16)
    out = SimpleNamespace(cand_ids=cand, unary=unary, scores=unary.clone(), ids=torch.tensor([0]))
    rec = {"k": 1, "cand": [list(range(15)) + [99]], "scores": [unary[0].tolist()[:15] + [15.0]], "ids": [0]}
    for logit_99, excused in ((15.0004, True), (14.9, False)):
        lg = torch.full((128,), -5.0)
        lg[99] = logit_99
        c = d1.compare(rec, out, lambda j: lg)
        assert c["sets"] == excused and c["ties"] == int(excused) and c["walk"], (logit_99, c)
    # The noise-aware tie (operator, 2026-10-27): with the engine's unary on the record, the window widens to the
    # row's own logit difference over the common candidates - here 0.05 (slot 3) - so 99 at 0.04 from the 16th logit
    # is excused (a noise tie), at 0.1 it is not.
    eng_u = unary[0].tolist()[:15] + [15.0]
    eng_u[3] += 0.05
    rec["unary"] = [eng_u]
    for logit_99, excused in ((14.96, True), (14.9, False)):
        lg = torch.full((128,), -5.0)
        lg[99] = logit_99
        c = d1.compare(rec, out, lambda j: lg)
        assert c["sets"] == excused and c["ties"] == int(excused) and c["noise_ties"] == int(excused), (logit_99, c)


def test_int8_arm():
    with tempfile.TemporaryDirectory() as tmp:
        run, recs, dumps, drafter, target, _, _, _ = _world(tmp, "int8", "int8")
        ok, st = _eval(tmp, run, recs, dumps, drafter, target)
        assert ok, st


if __name__ == "__main__":
    for name in [n for n in dir() if n.startswith("test_")] if len(sys.argv) < 2 else sys.argv[1:]:
        globals()[name]()
        print(f"{name}: ok")
