#!/usr/bin/env python3
"""oracle_generate.py --batch against its sequential path, BITWISE (the A4 reference stays a reference).

    python3 tools/toolcall/test_oracle_batched.py [test_name ...]     (agnes-ref-img: torch,
                                                                      transformers 5.15, g++ / ninja)

Tiny random checkpoints in each reference's real format - Ornith (Qwen3.5-MoE, int4 GPTQ, layer-
streamed, experts on demand: tools/oracle/test_ornith_ref.py's), K2-Horizon (int4 GPTQ with MoVA:
test_k2_ref.py's tiny config) and Kolibri-1 (bf16, sliding + full layers: test_kolibri_ref.py's) -
are loaded through oracle_generate.moe_runner exactly as the A4 run loads the real ones. Five
ragged prompts (1..17 ids), a different new-id cap per prompt, and an EOS set picked from the
free-running continuations so that the sequences stop at different points (by EOS at different
lengths, and by their caps). The sequential path (runner.generate, one scenario at a time) is the
reference; greedy_batched over runner.step_many with batch 2 (slots refilled mid-run, a new
prompt sharing a pass with decode steps), 3 in reverse order, and all at once must give the same
ids AND every logits row the greedy loop read, torch.equal. The streamed weight reads per
generated id are counted both ways (what batching saves).

test_cli_resume runs oracle_generate.py's main on a tiny Ornith set: --batch 1 into one directory,
--batch 3 into another with one scenario already present (skipped, untouched), the .ids / .txt
files equal; and --batch auto prints the memory plan.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "..", "oracle")
sys.path.insert(0, HERE)
import oracle_generate as OG  # noqa: E402

import torch  # noqa: E402


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TO = _load(os.path.join(ORACLE, "test_ornith_ref.py"), "tb_test_ornith_ref")
TK = _load(os.path.join(ORACLE, "test_k2_ref.py"), "tb_test_k2_ref")
TL = _load(os.path.join(ORACLE, "test_kolibri_ref.py"), "tb_test_kolibri_ref")

LENS = [9, 3, 17, 1, 12]
CAPS = [7, 10, 5, 8, 12]


# --------------------------------------------------------------------------------------------
# the tiny checkpoints

def ornith_snapshot(tmp: str):
    TO.make_checkpoint(tmp, seed=3)
    return "qwen3_5_moe", 512


def k2_snapshot(tmp: str):
    """test_k2_ref.test_gptq_checkpoint's int4 layout: every layer linear int4 g64 except the MoE
    router; embed / head / norms bf16, v_router.bias f16."""
    over = dict(hidden_size=128, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                rope_head_dim=32, moe_intermediate_size=64, mova_num_experts=8, vocab_size=256)
    d, c, sd = TK.tiny(seed=21, **over)
    d["quantization_config"] = {"bits": 4, "group_size": 64, "sym": True, "desc_act": False,
                                "quant_method": "gptq", "provider": "auto-round"}
    g = torch.Generator().manual_seed(21)
    files = {}
    for name, t in sd.items():
        if not (t.dim() == 2 and name.startswith("model.layers.") and not name.endswith("mlp.gate.weight")):
            files[name] = t.to(torch.float16) if name.endswith("v_router.bias") else t
            continue
        N, K = t.shape
        q = torch.randint(0, 16, (N, K), generator=g)
        sc = (torch.rand(K // 64, N, generator=g) * 0.05 + 0.01).to(torch.float16)
        qw, qz, gi = TK.gptq_pack(q, sc, 64)
        base = name[:-len(".weight")]
        files.update({base + ".qweight": qw, base + ".scales": sc, base + ".qzeros": qz, base + ".g_idx": gi})
    TK.write_checkpoint(tmp, d, files)
    return "k2_horizon", c.vocab_size


def kolibri_snapshot(tmp: str):
    d, c, sd = TL.tiny(seed=22)
    TL.write_checkpoint(tmp, d, sd)
    return "kolibri1", c.vocab_size


SNAPSHOTS = {"ornith": ornith_snapshot, "k2": k2_snapshot, "kolibri": kolibri_snapshot}


# --------------------------------------------------------------------------------------------
# the two paths

def prompts_for(vocab: int, seed: int = 5) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {f"s{i}": torch.randint(0, vocab, (n,), generator=g).tolist() for i, n in enumerate(LENS)}


def run_seq(runner, prompts: dict, caps: dict, eos: set):
    ids, rows = {}, {}
    for name, p in prompts.items():
        rows[name] = []
        ids[name] = runner.generate(p, caps[name], eos, on_row=lambda r, k=name: rows[k].append(r.clone()))
    return ids, rows


def run_batched(runner, prompts: dict, caps: dict, eos: set, batch: int, names: list):
    ids, rows = {}, defaultdict(list)
    OG.greedy_batched(runner.step_many, runner.new_state, [(k, prompts[k], caps[k]) for k in names], batch, eos,
                      runner.argmax, on_done=lambda k, o: ids.__setitem__(k, list(o)),
                      on_row=lambda k, r: rows[k].append(r.clone()))
    return ids, dict(rows)


def pick_eos(free: dict, caps: dict) -> set:
    """An EOS set that ends two sequences early at different lengths (the first id each produces
    at a chosen step that no sequence produced before), and leaves at least one at its cap."""
    names = list(free)
    for a in range(len(names)):
        for b in range(len(names)):
            for ka in range(1, 4):
                for kb in range(ka + 1, 7):
                    A, B = free[names[a]], free[names[b]]
                    if a == b or ka >= len(A) or kb >= len(B):
                        continue
                    eos = {A[ka], B[kb]}
                    stops = {}
                    for k, s in free.items():
                        cut = next((j for j, x in enumerate(s) if x in eos), None)
                        stops[k] = cut
                    early = sorted({v for v in stops.values() if v is not None})
                    if len(early) >= 2 and any(v is None for v in stops.values()):
                        return eos
    raise AssertionError(f"no EOS set with two different early stops in {free}")


class Counter:
    """Streamed weight reads: dense layer loads and expert dequants (the cost batching shares)."""

    def __init__(self, runner, mtype: str):
        self.layers = self.experts = 0
        if mtype == "qwen3_5_moe":
            pf = runner.pf
            orig = pf.layer_sd

            def layer_sd(i):
                self.layers += 1
                return orig(i)
            pf.layer_sd = layer_sd
            self._lazy = runner.lazy
        else:
            ref = runner.ref
            if ref.pf is not None:
                orig_pf = ref.pf.layer_sd

                def layer_sd(i):
                    self.layers += 1
                    return orig_pf(i)
                ref.pf.layer_sd = layer_sd
            src = ref.src
            orig_lin = src.linear

            def linear(base):
                if "experts." in base and "shared_experts" not in base:
                    self.experts += 1
                return orig_lin(base)
            src.linear = linear
            self._lazy = None

    def snap(self):
        return (self.layers, self.experts + (self._lazy.filled if self._lazy else 0))


def check(model: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mtype, vocab = SNAPSHOTS[model](tmp)
        runner = OG.moe_runner(tmp, mtype)
        cnt = Counter(runner, mtype)
        prompts = prompts_for(vocab)
        caps = dict(zip(prompts, CAPS))
        free, _ = run_seq(runner, prompts, caps, set())
        eos = pick_eos(free, caps)
        c0 = cnt.snap()
        t = time.time()
        want_ids, want_rows = run_seq(runner, prompts, caps, eos)
        t_seq = time.time() - t
        c1 = cnt.snap()
        stops = {k: (len(v), "eos" if v and v[-1] in eos else "cap") for k, v in want_ids.items()}
        assert len({n for n, why in stops.values() if why == "eos"}) >= 2, stops
        assert any(why == "cap" for _, why in stops.values()), stops
        n_tok = sum(len(v) for v in want_ids.values())
        names = list(prompts)
        notes = []
        for batch, order in ((2, names), (3, names[::-1]), (len(names), names)):
            c2 = cnt.snap()
            t = time.time()
            got_ids, got_rows = run_batched(runner, prompts, caps, eos, batch, order)
            dt = time.time() - t
            c3 = cnt.snap()
            assert got_ids == want_ids, (model, batch, got_ids, want_ids)
            for k in names:
                assert len(got_rows[k]) == len(want_rows[k]), (model, batch, k)
                for j, (a, b) in enumerate(zip(got_rows[k], want_rows[k])):
                    if not torch.equal(a, b):
                        raise AssertionError(f"{model} batch {batch}: {k} row {j} differs, max |d| "
                                             f"{float((a - b).abs().max()):.3g}")
            notes.append(f"batch {batch}: {c3[0] - c2[0]} layer loads, {c3[1] - c2[1]} expert dequants, {dt:.2f}s")
        rows = sum(len(v) for v in want_rows.values())
        print(f"  {model}: ids and {rows} logits rows bitwise over 3 batchings; stops {stops}")
        print(f"    sequential: {c1[0] - c0[0]} layer loads, {c1[1] - c0[1]} expert dequants, {t_seq:.2f}s "
              f"for {n_tok} new ids; " + "; ".join(notes))
        if mtype == "kolibri1":
            return
        # --resident: every layer's dense weights and half the expert layers resident once read
        with open(os.path.join(tmp, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        L = cfg.get("text_config", cfg)["num_hidden_layers"]
        kept = OG.moe_runner(tmp, mtype, keep=(L, max(1, L // 2)))
        kc = Counter(kept, mtype)
        for label, fn in (("sequential", lambda: run_seq(kept, prompts, caps, eos)),
                          ("batch 3", lambda: run_batched(kept, prompts, caps, eos, 3, names))):
            k0 = kc.snap()
            got_ids, got_rows = fn()
            k1 = kc.snap()
            assert got_ids == want_ids, (model, label, got_ids, want_ids)
            for k in names:
                assert len(got_rows[k]) == len(want_rows[k]) and all(
                    torch.equal(a, b) for a, b in zip(got_rows[k], want_rows[k])), (model, label, k)
            print(f"    resident (dense {L}, experts {max(1, L // 2)} layers), {label}: bitwise; "
                  f"{k1[0] - k0[0]} layer loads, {k1[1] - k0[1]} expert dequants")


def test_ornith():
    check("ornith")


def test_k2():
    check("k2")


def test_kolibri():
    check("kolibri")


# --------------------------------------------------------------------------------------------
# the CLI, resumable

def write_tokenizer(path: str, vocab: int) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    tk = Tokenizer(models.WordLevel({f"t{i}": i for i in range(vocab)}, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.save(path)


def write_set(set_dir: str, prompts: dict) -> None:
    os.makedirs(set_dir, exist_ok=True)
    man = []
    for k, p in prompts.items():
        raw = (" ".join(map(str, p)) + "\n").encode()
        with open(os.path.join(set_dir, f"{k}.ids"), "wb") as f:
            f.write(raw)
        man.append({"name": k, "ids": len(p), "sha256": hashlib.sha256(raw).hexdigest()})
    with open(os.path.join(set_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f)


def run_main(args: list[str]) -> str:
    import contextlib
    import io
    buf = io.StringIO()
    old = sys.argv
    sys.argv = ["oracle_generate.py"] + args
    try:
        with contextlib.redirect_stdout(buf):
            OG.main()
    finally:
        sys.argv = old
    return buf.getvalue()


def test_cli_resume():
    with tempfile.TemporaryDirectory() as tmp:
        snap = os.path.join(tmp, "snap")
        os.makedirs(snap)
        mtype, vocab = ornith_snapshot(snap)
        write_tokenizer(os.path.join(snap, "tokenizer.json"), vocab)
        prompts = prompts_for(vocab, seed=6)
        # EOS from a free run, so that the CLI's sequences stop at different points too
        runner = OG.moe_runner(snap, mtype)
        free, _ = run_seq(runner, prompts, {k: 8 for k in prompts}, set())
        eos = sorted(pick_eos(free, {k: 8 for k in prompts}))
        with open(os.path.join(snap, "generation_config.json"), "w", encoding="utf-8") as f:
            json.dump({"eos_token_id": eos}, f)
        set_dir = os.path.join(tmp, "set")
        write_set(set_dir, prompts)
        seq, bat = os.path.join(tmp, "seq"), os.path.join(tmp, "bat")
        log1 = run_main([snap, set_dir, seq, "--new-tokens", "8"])
        assert "sequential: one scenario at a time" in log1, log1
        first = next(iter(prompts))
        os.makedirs(bat)
        for x in ("ids", "txt"):
            shutil.copy(os.path.join(seq, f"{first}.bf16.{x}"), os.path.join(bat, f"{first}.bf16.{x}"))
        before = os.stat(os.path.join(bat, f"{first}.bf16.ids")).st_mtime_ns
        log2 = run_main([snap, set_dir, bat, "--new-tokens", "8", "--batch", "3"])
        assert f"{len(prompts)} scenarios, 1 already done, {len(prompts) - 1} to run" in log2, log2
        assert "memory plan (ESTIMATES)" in log2 and "batched: up to 3" in log2, log2
        assert os.stat(os.path.join(bat, f"{first}.bf16.ids")).st_mtime_ns == before
        for k in prompts:
            for x in ("ids", "txt"):
                with open(os.path.join(seq, f"{k}.bf16.{x}"), "rb") as f1, open(os.path.join(bat, f"{k}.bf16.{x}"), "rb") as f2:
                    assert f1.read() == f2.read(), (k, x)
        import compare_ref
        names = sorted(prompts)
        same, diff, _ = compare_ref.compare(seq, bat, names)
        assert (same, diff) == (len(prompts), 0), (same, diff)
        assert all(os.path.exists(os.path.join(seq, f"{k}.bf16.gap")) for k in prompts)
        only = names[1:3]
        log3 = run_main([snap, set_dir, os.path.join(tmp, "auto"), "--new-tokens", "8", "--batch", "auto",
                         "--resident", "auto", "--only", ",".join(only)])
        assert f"{len(only)} scenarios, 0 already done, {len(only)} to run" in log3, log3
        assert sorted(compare_ref.names_in(os.path.join(tmp, "auto"))) == only
        assert compare_ref.compare(seq, os.path.join(tmp, "auto"), only)[:2] == (len(only), 0)
        plan = next(line for line in log3.splitlines() if line.startswith("memory plan"))
        lens = sorted(len(p) for p in prompts.values())
        print(f"  CLI: --batch 1 == --batch 3 file for file (one scenario resumed); stop lengths "
              f"{sorted(len(v.split()) for v in [open(os.path.join(seq, k + '.bf16.ids')).read() for k in prompts])}, "
              f"prompts {lens}\n    {plan}")


def test_plan_real_configs():
    """The memory plan on the real configs (K2's vendored config.json; Ornith's text config as
    published): the per-sequence and base terms in the expected range, auto within the cap."""
    with open(os.path.join(ORACLE, "third_party", "k2_horizon", "config.json"), encoding="utf-8") as f:
        k2 = json.load(f)
    lens = [960, 1036, 2998, 2200, 2737, 2899, 1144, 2905, 2748, 2816]
    for cap in (28, 60):
        p = OG.batch_plan("k2_horizon", k2, lens, 512, cap * OG.GiB, "auto", "auto")
        print(f"  K2 at {cap} GiB: {p['text']}")
        assert 1 <= p["batch"] <= OG.MAX_AUTO_BATCH["k2_horizon"]
    mm = OG.mem_model("k2_horizon", k2)
    per = mm["per_seq"](3510)
    assert 1.2 * OG.GiB < per < 1.5 * OG.GiB, per / OG.GiB          # 48 layers x 8 x 128 x fp32 K and V
    assert sum(1 for x in mm["experts"] if x) == 45                    # mlp_only_layers 0-2
    assert 1.4 * OG.GiB < max(mm["experts"]) < 1.6 * OG.GiB           # 100 x 3 x 768 x 2560 + 64 MoVA, bf16
    ornith = {"text_config": dict(num_hidden_layers=40, hidden_size=2048, vocab_size=248320, full_attention_interval=4,
                                  linear_num_key_heads=16, linear_key_head_dim=128, linear_num_value_heads=32,
                                  linear_value_head_dim=128, linear_conv_kernel_dim=4, num_experts=256,
                                  moe_intermediate_size=512, shared_expert_intermediate_size=512,
                                  num_attention_heads=16, num_key_value_heads=2, head_dim=256)}
    for cap in (28, 60):
        p = OG.batch_plan("qwen3_5_moe", ornith, lens, 192, cap * OG.GiB, "auto", "auto")
        print(f"  Ornith at {cap} GiB: {p['text']}")
    mo = OG.mem_model("qwen3_5_moe", ornith)
    per = mo["per_seq"](3190)
    assert 0.1 * OG.GiB < per < 0.2 * OG.GiB, per / OG.GiB          # 63 MiB GDN state + 10 layers' bf16 KV
    assert 2.5 * OG.GiB < sum(mo["dense"]) < 3.5 * OG.GiB              # ~2.8 GB a step re-dequantises (plan 15a)
    assert abs(mo["experts"][0] - 1.61e9) < 0.01e9                     # 1.61 GB per layer (ornith_ref.py)
    p = OG.batch_plan("k2_horizon", k2, lens[:3], 512, 28 * OG.GiB, "7")
    assert p["batch"] == 3 and p["keep_dense"] == p["keep_experts"] == 0   # a count, capped by the scenarios


TESTS = [test_plan_real_configs, test_ornith, test_k2, test_kolibri, test_cli_resume]


def main() -> None:
    only = sys.argv[1:]
    for t in TESTS:
        if only and t.__name__ not in only:
            continue
        print(t.__name__, flush=True)
        t()
    print("ok")


if __name__ == "__main__":
    main()
