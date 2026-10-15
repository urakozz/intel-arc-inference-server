#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate Q's tools (spec 23a Task 7) on the tiny variants: muse_gateq.py kl / verdict and oracle_generate.py --model muse.

    PYTHONPATH=<5.19.0 site> python3 tools/oracle/test_muse_gateq.py [test_name ...]

The variants come from muse_make_tiny.py (test_muse_ref.py's helpers): `untied` stands in for the bf16 original and
`int4` (untied's weights RTN-packed g64, the same lm_head) for the GPTQ checkpoint. SKIPs without the tiny model.
"""
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402


def _load(name, path=None):
    spec = importlib.util.spec_from_file_location(name, path or os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load("test_muse_ref")           # the variant builders, Skip
ref = T.ref
gq = _load("muse_gateq")
OG = _load("oracle_generate", os.path.join(_HERE, "..", "toolcall", "oracle_generate.py"))
Skip = T.Skip


def write_set(d, names_ids):
    os.makedirs(d, exist_ok=True)
    man = []
    for n, ids in names_ids:
        text = " ".join(map(str, ids)) + "\n"
        open(os.path.join(d, f"{n}.ids"), "w").write(text)
        man.append({"name": n, "ids": len(ids), "sha256": hashlib.sha256(text.encode()).hexdigest()})
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))


def tiny_set():
    d = os.path.join(T.TMP, "gq")
    s, run = os.path.join(d, "set"), os.path.join(d, "run")
    if not os.path.exists(os.path.join(s, "manifest.json")):
        write_set(s, [("a", T.rand_ids(30, seed=31)), ("b", T.rand_ids(45, seed=32))])
        os.makedirs(run, exist_ok=True)
        for n, seed in (("a", 33), ("b", 34)):
            open(os.path.join(run, f"{n}.bf16.ids"), "w").write(" ".join(map(str, T.rand_ids(6, seed=seed))) + "\n")
        open(os.path.join(d, "prose.ids"), "w").write(" ".join(map(str, T.rand_ids(20, seed=35))) + "\n")
    return d, s, run


def run_kl(gptq, bf16, tag):
    d, s, run = tiny_set()
    rows = os.path.join(d, f"run-{tag}")          # a fresh run dir per pair: the bf16 rows are cached there
    if not os.path.exists(rows):
        import shutil
        shutil.copytree(run, rows)
    out = os.path.join(d, f"kl-{tag}.json")
    gq.main(["kl", rows, gptq, "--set", s, "--bf16", bf16, "--text", f"prose={os.path.join(d, 'prose.ids')}",
             "--out", out])
    return json.load(open(out))


def test_kl_identical_is_zero():
    """KL(p || p) of the same model is exactly 0 and top-1 agreement 1.0 (A4 rows and the text rows)."""
    u = T.variant("untied")
    j = run_kl(u, u, "same")
    assert j["a4"]["rows"] == 12 and j["a4"]["kl_mean"] == 0.0 and j["a4"]["kl_max"] == 0.0 and j["a4"]["top1"] == 1.0
    assert j["text"]["prose"]["rows"] == 19 and j["text"]["prose"]["kl_mean"] == 0.0
    print(f"  identical: {j['a4']['rows']} A4 rows KL 0, top-1 1.0; prose {j['text']['prose']['rows']} rows KL 0")


def test_kl_int4_vs_untied():
    """The RTN int4 twin against its source: KL > 0, finite; the cached bf16 rows are reused on a second call."""
    j = run_kl(T.variant("int4"), T.variant("untied"), "int4")
    a = j["a4"]
    assert a["rows"] == 12 and 0 < a["kl_mean"] < 10 and 0 <= a["top1"] <= 1 and a["kl_p99"] >= a["kl_p50"]
    assert os.path.exists(os.path.join(T.TMP, "gq", "run-int4", "a.bf16.rows.safetensors"))
    j2 = run_kl(T.variant("int4"), T.variant("untied"), "int4")
    assert j2["a4"] == a
    print(f"  int4 vs untied: KL mean {a['kl_mean']:.4g} p99 {a['kl_p99']:.4g}, top-1 {a['top1']:.3f}")


CALL = ('<|start|>assistant to={n}<|message|><atem:function_calls>\n<atem:invoke name="{n}">\n'
        '<atem:parameter name="filePath">{p}</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|>')


def plant(d, n_match, n_mismatch, cand_fail=0, ref_fail=0):
    os.makedirs(d, exist_ok=True)
    i = 0
    for kind, k in (("match", n_match), ("mismatch", n_mismatch)):
        for _ in range(k):
            name = f"s{i:02d}"
            r = " to=self<|message|>look<|eom|>" + CALL.format(n="read", p=f"/w/{i}")
            c = r if kind == "match" else CALL.format(n="grep", p="/w")[len("<|start|>assistant"):]
            if i < ref_fail:
                r = " to=read<|message|><atem:function_calls>\n<atem:invoke name=\"read\">"
            if n_match + n_mismatch - i <= cand_fail:
                c = " to=read<|message|><atem:function_calls>"
            open(os.path.join(d, f"{name}.bf16.txt"), "w").write(r)
            open(os.path.join(d, f"{name}.gptq.txt"), "w").write(c)
            i += 1


def kl_json(path, mean, top1):
    json.dump({"a4": {"rows": 1000, "kl_mean": mean, "kl_p50": mean / 2, "kl_p99": mean * 5, "kl_max": mean * 9,
                      "top1": top1}, "text": {}, "scenarios": {}}, open(path, "w"))
    return path


def test_verdict_bars():
    """verdict reads the A4 outputs (score.py's ATEM reader: parse failures, first-call agreement) and the KL json,
    and prints each of spec 23 §3.3's bars (decision 5) - candidate parse failures <= the reference's, agreement
    >= 25 / 36, mean KL <= 0.02, top-1 >= 0.95 - and WITHIN / OUTSIDE: a recorded measurement (the checkpoint is kept,
    operator 2026-10-15), exit 0."""
    base = os.path.join(T.TMP, "verdict")
    cases = [("pass", (30, 6, 0, 0), (0.01, 0.97), True), ("agree", (24, 12, 0, 0), (0.01, 0.97), False),
             ("fail", (30, 6, 1, 0), (0.01, 0.97), False), ("failboth", (30, 6, 1, 1), (0.01, 0.97), True),
             ("kl", (30, 6, 0, 0), (0.03, 0.97), False), ("top1", (30, 6, 0, 0), (0.01, 0.94), False)]
    for name, table, kl, want in cases:
        d = os.path.join(base, name)
        plant(d, *table)
        ok, lines = gq.verdict(d, kl_json(os.path.join(d, "kl.json"), *kl))
        assert ok == want, (name, lines)
        assert lines[-1].startswith("gate Q: WITHIN" if want else "gate Q: OUTSIDE"), lines
    t = gq.a4_table(os.path.join(base, "fail"))
    assert t["n"] == 36 and t["agree"] == 30 and t["cand_fail"] == 1 and t["ref_fail"] == 0, t
    r = subprocess.run([sys.executable, os.path.join(_HERE, "muse_gateq.py"), "verdict", os.path.join(base, "kl"),
                        os.path.join(base, "kl", "kl.json")], capture_output=True, text=True)
    assert r.returncode == 0 and "gate Q: OUTSIDE the proposed bars on Q-KL mean" in r.stdout, r.stdout + r.stderr
    print("  bars: agreement 25 / 36, parse failures, mean KL 0.02, top-1 0.95 - within / outside as planted; a "
          "recorded measurement (exit 0)")


def test_oracle_generate_muse():
    """oracle_generate.py's muse runner: the sequential greedy (argmax of the capped row) equals muse_ref's own model
    driven step by step, and the layer-major batched path equals the sequential one (two ragged prompts)."""
    snap = T.variant("win16")
    runner = OG.moe_runner(snap, "muse_glimmer")
    eos = {200001, 200008}
    prompts = {"a": T.rand_ids(37, seed=41), "b": T.rand_ids(60, seed=42)}
    seq = {n: runner.generate(p, 10, eos) for n, p in prompts.items()}
    model = runner.ref
    for n, p in prompts.items():
        cache, out, row = None, [], None
        with torch.no_grad():
            o = model(input_ids=torch.tensor([p]), use_cache=True)
            for _ in range(10):
                t = int(o.logits[0, -1].float().argmax())
                out.append(t)
                o = model(input_ids=torch.tensor([[t]]), past_key_values=o.past_key_values, use_cache=True)
        assert out == seq[n], (n, out, seq[n])
    got = {}
    OG.greedy_batched(runner.step_many, runner.new_state, [(n, p, 10) for n, p in prompts.items()], 2, eos,
                      runner.argmax, lambda n, new: got.__setitem__(n, new))
    assert got == seq, (got, seq)
    assert OG.parse_args(["--model", "muse", "--form", "gptq", "s", "d", "o"])["form"] == "gptq"
    print("  muse runner: sequential = muse_ref's model step by step; layer-major batched (2) = sequential")


TESTS = [test_kl_identical_is_zero, test_kl_int4_vs_untied, test_verdict_bars, test_oracle_generate_muse]


def main() -> None:
    import traceback
    only = sys.argv[1:]
    skipped, failed = 0, []
    for t in TESTS:
        if only and t.__name__ not in only:
            continue
        print(t.__name__, flush=True)
        try:
            t()
        except Skip as s:
            skipped += 1
            print(f"SKIP {t.__name__}: {s}")
        except Exception:      # noqa: BLE001
            traceback.print_exc()
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}", flush=True)
    if failed:
        print(f"{len(failed)} FAILED: {failed}")
        sys.exit(1)
    print(f"ok ({skipped} skipped)")


if __name__ == "__main__":
    main()
