#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""K2-Horizon taps for the EAGLE3 K2 P0: k2_ref.py over [prompt + recorded continuation].

    k2_taps.py run <K2 snapshot> --out-dir DIR [SOURCES] [--aux 2,24,45] [--keep-ctx 2048]
               [--topk 16] [--matmul fp32|bf16] [--mode bf16|f32] [--dry-run]
    k2_taps.py sources --out-dir DIR [SOURCES]     DIR/sources.json only (no model)
    k2_taps.py golden-ids <run.safetensors> <out.ids>   k2_ref.py run's greedy `tokens` -> ids

SOURCES (any mix; a name is `corpus/label`, the corpus is what eagle3_accept.py pools by):
    --source NAME:PROMPT_IDS:CONT_IDS   whitespace-separated id files (CONT = the recorded K2
                                        greedy continuation, teacher-forced)
    --a4-ref DIR                        tools/toolcall/a4_ref.sh k2's output: DIR/set/manifest.json
                                        names the scenarios; each one whose DIR/<name>.bf16.ids
                                        exists is a4/<name> (prompt DIR/set/<name>.ids)

One forward of k2_ref.K2Ref per source over all N ids (one sequence, no cache), layer at a
time, the int4 checkpoint dequantised (k2_ref.Checkpoint). Per source,
DIR/<corpus>__<label>.k2taps.safetensors holds:

    ids        int32 [N]
    aux.{a}    bf16 [N - aux_from, 2560]   the residual stream at the INPUT of decoder layer a
                                           (= k2_ref's resid.L{a-1}, the output of layer a - 1;
                                           a = 0: the embedding) at positions aux_from..N-1 -
                                           EAGLE-3's aux hidden state a (eagle3_ref.py S1)
    final      bf16 [N - head_from, 2560]  the last layer's output, pre-norm (the plugin's aux
                                           index 48; speculators' training target input)
    greedy     int32 [N - head_from]       argmax (first maximum) of the logits as the A4
                                           reference computes them (grouped final norm, lm_head,
                                           rounded to bf16 in mode bf16): K2's next id after
                                           position head_from + t
    greedy_flatnorm int32 [N - head_from]  the same with the final norm as ONE 2560-wide RMSNorm
                                           group: what speculators' LlamaRMSNorm verifier_norm
                                           computes from `final` (a training-target diagnostic)
    top_ids / top_logits [N - head_from, topk]   torch.topk, ties reordered to the lower id
    lse        f32 [N - head_from]         logsumexp over the vocabulary
    moe_ids    uint8 [N - head_from, 45, 8]   per sparse layer the routed MoE experts (ascending)
    mova_ids   uint8 [N - head_from, 45, 4]   ... and the MoVA value experts (eagle3_cost.py)
    metadata   name, n_prompt, n, aux_from, head_from, aux_ids, sparse_layers, mode, matmul, ...

aux_from = max(0, n_prompt - keep_ctx - 1): the drafter's window (2048) never reads an older
row from an anchor >= n_prompt (a row s carries the aux of s - 1). head_from = n_prompt - 1.

`--matmul fp32` (default): every linear and attention matmul as an fp32 GEMM with ONE bf16
rounding of the output, every other bf16 rounding point of k2_ref's mode bf16 kept. torch's bf16
GEMM is ~6x slower on this Mac's AVX2 (measured 2026-10-08, 2 threads: 18-22 vs 113-135
GFLOP/s); the two differ by accumulation order only (an ulp in places), which test_k2_taps.py
bounds. `--matmul bf16` is bit-identical to k2_ref (= the vendored HF model in bf16).

Resumable: a source whose output exists is skipped. `--dry-run` prints the plan, a time and a
peak-memory estimate, and loads nothing (no torch needed).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

DEFAULT_AUX = "2,24,45"            # the drafter's eagle_aux_hidden_state_layer_ids
ASSUMED_GFLOPS = 300.0             # fp32 sgemm, 16 threads, i9-9980HK (assumed, for --dry-run)
DEQUANT_S_PER_FORWARD = 60.0       # every expert of 45 layers dequantised once (~79 GB bf16 at
                                   # ~1-3 GB/s, k2_ref README) - estimated
K2_FLOP_PER_TOKEN = 9.35e9         # 45 x 98.2 M + 3 x 83.9 M MACs (spec 18 §11) x 2 - derived
K2 = dict(layers=48, hidden=2560, heads=32, head_dim=128, vocab=250624, sparse=45)


def _load(name):
    if name in sys.modules:                    # one instance per process (the tests share it)
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                    # dataclasses resolve annotations through it
    spec.loader.exec_module(mod)
    return mod


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]


def file_name(name: str) -> str:
    return name.replace("/", "__") + ".k2taps.safetensors"


# ------------------------------------------------------------------------------- sources


def parse_sources(a) -> list:
    out = []
    for s in a.source or []:
        parts = s.split(":")
        if len(parts) != 3 or "/" not in parts[0]:
            raise SystemExit(f"--source {s}: want corpus/label:PROMPT_IDS:CONT_IDS")
        out.append({"name": parts[0], "prompt_ids": read_ids(parts[1]), "cont_ids": read_ids(parts[2])})
    if getattr(a, "a4_ref", None):
        man = os.path.join(a.a4_ref, "set", "manifest.json")
        if not os.path.isfile(man):
            raise SystemExit(f"--a4-ref {a.a4_ref}: no {man} (tools/toolcall/a4_ref.sh k2 set)")
        with open(man, encoding="utf-8") as f:
            entries = json.load(f)
        done = 0
        for e in entries:
            cont = os.path.join(a.a4_ref, f"{e['name']}.bf16.ids")
            txt = os.path.join(a.a4_ref, f"{e['name']}.bf16.txt")
            if os.path.isfile(cont) and os.path.isfile(txt):      # .txt last marks a scenario done
                out.append({"name": "a4/" + e["name"], "prompt_ids": read_ids(os.path.join(a.a4_ref, "set", e["name"] + ".ids")),
                            "cont_ids": read_ids(cont)})
                done += 1
        print(f"--a4-ref {a.a4_ref}: {done} of {len(entries)} scenarios have a reference continuation", flush=True)
    # planning placeholders (sizes only, --dry-run): before the references exist
    plans = list(getattr(a, "plan", None) or [])
    for pm in getattr(a, "plan_manifest", None) or []:
        path, _, nc = pm.rpartition(":")
        with open(path, encoding="utf-8") as f:
            plans += [f"a4/{e['name']}:{e['ids']}:{nc}" for e in json.load(f)]
    if plans and not getattr(a, "dry_run", True):
        raise SystemExit("--plan / --plan-manifest are sizes for --dry-run only")
    have = {s["name"] for s in out}
    for p in plans:
        name, n_p, n_c = p.split(":")
        if name not in have:
            out.append({"name": name, "prompt_ids": [0] * int(n_p), "cont_ids": [0] * int(n_c), "planned": True})
    names = [s["name"] for s in out]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate source names: {sorted(n for n in names if names.count(n) > 1)}")
    for s in out:
        if not s["prompt_ids"] or not s["cont_ids"]:
            raise SystemExit(f"{s['name']}: empty prompt or continuation")
    return out


def write_sources_json(out_dir: str, sources: list) -> str:
    path = os.path.join(out_dir, "sources.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"name": s["name"], "n_prompt": len(s["prompt_ids"]), "n": len(s["prompt_ids"]) + len(s["cont_ids"]),
                    "file": file_name(s["name"])} for s in sources], f, indent=1)
    return path


def estimate(lengths, n_prompts, keep_ctx: int = 2048) -> dict:
    """Time and peak memory (derived / assumed rates, see the constants). One forward per
    source: dequant + 9.35 GFLOP per token + eager attention (full T x T, 48 layers) + two head
    GEMMs over the continuation rows."""
    H, V, L, hd, nh = K2["hidden"], K2["vocab"], K2["layers"], K2["head_dim"], K2["heads"]
    flops = sum(K2_FLOP_PER_TOKEN * n + 4.0 * nh * hd * n * n * L + 2 * 2.0 * H * V * (n - p + 1)
                for n, p in zip(lengths, n_prompts))
    t = len(lengths) * DEQUANT_S_PER_FORWARD + flops / (ASSUMED_GFLOPS * 1e9)
    T = max(lengths) if lengths else 0
    attn = nh * T * T * 4 * 4                      # scores, mask fill, softmax, rounding copies (fp32)
    resident = 2 * V * H * 2 + 2 * 1.6e9           # embed + head bf16; a layer's experts bf16, + prefetch
    acts = T * (H * 4 * 12 + 9280 * 4 * 2 + 6144 * 4 * 3)
    disk = sum((n - max(0, p - keep_ctx - 1)) * 3 * H * 2 + (n - p + 1) * (H * 2 + 45 * 12 + 16 * 8 + 12)
               for n, p in zip(lengths, n_prompts))
    return {"tokens": sum(lengths), "tflop": flops / 1e12, "est_hours": t / 3600,
            "peak_gib": (resident + attn + acts + 256 * V * 4) / 2**30, "disk_gb": disk / 1e9}


# ------------------------------------------------------------------------------- the forward


class TapRecorder:
    """k2_ref.Recorder's interface; keeps only the named tensors (one forward: one add each)."""

    def __init__(self, keep):
        self.keep = set(keep)
        self.t = {}

    def add(self, name, x):
        if name in self.keep:
            self.t[name] = x.detach().clone()


class _NoCache(list):
    """A cache list that keeps nothing (one forward over the whole sequence)."""

    def __setitem__(self, i, v):
        pass


def make_arith(ref, mode: str, matmul: str):
    import torch
    if mode == "f32" or matmul == "bf16":
        return ref.Arith(mode)

    class Fp32MatmulArith(ref.Arith):
        """mode bf16's rounding points, the GEMMs in fp32 with one bf16 rounding of the output."""

        def linear(self, x, w):
            return self.r(x.float() @ w.float().t())

        def mm(self, a, b):
            return self.r(torch.matmul(a.float(), b.float()))
    return Fp32MatmulArith(mode)


def build(snapshot_or_src, cfg=None, mode="bf16", matmul="fp32", prefetch=True):
    """(K2Ref with the head left out of forward, its Arith) over a snapshot dir or a DictSource."""
    ref = _load("k2_ref")
    if isinstance(snapshot_or_src, str):
        src = ref.Checkpoint(snapshot_or_src)
        cfg = src.cfg
    else:
        src = snapshot_or_src

    class TapRef(ref.K2Ref):
        def head(self, x, chunk=32768):        # forward returns the post-norm hidden
            return x
    r = TapRef(cfg, src, mode=mode, prefetch=prefetch)
    r.ar = make_arith(ref, mode, matmul)
    return r


def head_rows(r, xn, topk: int, row_chunk: int = 256, vocab_chunk: int = 32768):
    """Over post-norm rows xn [R, H]: greedy (first max), top ids / logits, lse - the logits
    the reference computes (ar.linear on the bf16 lm_head, rounded in mode bf16), in vocab
    chunks so the head is never upcast whole."""
    import torch
    w = r.lm_head
    V = w.shape[0]
    g, ti, tv, lse = [], [], [], []
    for a in range(0, xn.shape[0], row_chunk):
        x = xn[a:a + row_chunk]
        lg = torch.cat([r.ar.linear(x, w[c:c + vocab_chunk]) for c in range(0, V, vocab_chunk)], -1)
        g.append(torch.argmax(lg, -1))
        v, i = torch.topk(lg, topk, dim=-1)
        o = torch.argsort(i, dim=-1, stable=True)
        v, i = v.gather(-1, o), i.gather(-1, o)
        o = torch.sort(v, dim=-1, descending=True, stable=True).indices
        ti.append(i.gather(-1, o))
        tv.append(v.gather(-1, o))
        lse.append(torch.logsumexp(lg.double(), -1).float())
    return torch.cat(g), torch.cat(ti), torch.cat(tv), torch.cat(lse)


def forward_taps(r, ids, n_prompt: int, aux_ids, keep_ctx: int = 2048, topk: int = 16) -> dict:
    """One forward over ids; the tensors of the dump (module docstring), keyed as saved."""
    import torch
    c = r.c
    L = c.num_hidden_layers
    if max(aux_ids) >= L:
        raise SystemExit(f"aux id {max(aux_ids)} >= {L} layers")
    N = len(ids)
    aux_from = max(0, n_prompt - keep_ctx - 1)
    head_from = n_prompt - 1
    sparse = [i for i in range(L) if c.is_sparse(i)]
    keep = {f"resid.L{a - 1}" for a in aux_ids if a > 0} | {f"resid.L{L - 1}"}
    keep |= {f"route.moe.ids.L{i}" for i in sparse} | {f"route.mova.ids.L{i}" for i in sparse if c.is_mova(i)}
    rec = TapRecorder(keep)
    with torch.no_grad():
        xn = r.forward(ids, 0, _NoCache([None] * L), rec)        # post-norm hidden (head left out)
        out = {"ids": torch.tensor(ids, dtype=torch.int32)}
        emb = r.embed[torch.as_tensor(ids, dtype=torch.long)].float()
        for a in aux_ids:
            x = emb if a == 0 else rec.t[f"resid.L{a - 1}"]
            out[f"aux.{a}"] = x[aux_from:].to(torch.bfloat16).contiguous()
        fin = rec.t[f"resid.L{L - 1}"][head_from:]
        out["final"] = fin.to(torch.bfloat16).contiguous()
        g, ti, tv, lse = head_rows(r, xn[head_from:], topk)
        flat = _load("k2_ref").grouped_rms_norm(r.ar, fin, r.norm_w, 1, c.rms_norm_eps)   # ONE group
        gf = head_rows(r, flat, 1)[0]
    out.update(greedy=g.to(torch.int32), greedy_flatnorm=gf.to(torch.int32), top_ids=ti.to(torch.int32),
               top_logits=tv.float(), lse=lse.float())
    if sparse:
        out["moe_ids"] = torch.stack([rec.t[f"route.moe.ids.L{i}"][head_from:] for i in sparse], 1).to(torch.uint8)
        mv = [i for i in sparse if c.is_mova(i)]
        if mv:
            out["mova_ids"] = torch.stack([rec.t[f"route.mova.ids.L{i}"][head_from:] for i in mv], 1).to(torch.uint8)
    meta = {"n_prompt": n_prompt, "n": N, "aux_from": aux_from, "head_from": head_from,
            "aux_ids": ",".join(map(str, aux_ids)), "sparse_layers": ",".join(map(str, sparse)),
            "aux_semantics": "aux.{a} = the input of decoder layer a = resid.L{a-1} (pre-norm)"}
    return out, meta


def save_dump(path: str, name: str, t: dict, meta: dict) -> None:
    import torch
    from safetensors.torch import save_file
    for k, v in t.items():
        if v.is_floating_point() and not bool(torch.isfinite(v.float()).all()):
            raise RuntimeError(f"{name}: non-finite {k}")
    md = {"name": name}
    md.update({k: str(v) for k, v in meta.items()})
    tmp = path + ".tmp"
    save_file({k: v.contiguous() for k, v in t.items()}, tmp, metadata=md)
    os.replace(tmp, path)


def load_dump(path: str) -> dict:
    """A dump as {'name', 'n_prompt', 'n', 'aux_from', 'head_from', 'aux_ids', 'ids', 'aux': {a: ..},
    'greedy', 'greedy_flatnorm', 'final', 'top_ids', 'top_logits', 'lse', 'moe_ids', 'mova_ids', 'meta'}."""
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        md = f.metadata()
        t = {k: f.get_tensor(k) for k in f.keys()}
    aux_ids = [int(i) for i in md["aux_ids"].split(",")]
    d = {"name": md["name"], "n_prompt": int(md["n_prompt"]), "n": int(md["n"]), "aux_from": int(md["aux_from"]),
         "head_from": int(md["head_from"]), "aux_ids": aux_ids, "ids": t["ids"].long(),
         "aux": {a: t[f"aux.{a}"] for a in aux_ids}, "greedy": t["greedy"].long(),
         "greedy_flatnorm": t["greedy_flatnorm"].long(), "final": t["final"], "top_ids": t["top_ids"].long(),
         "top_logits": t["top_logits"], "lse": t["lse"], "moe_ids": t.get("moe_ids"), "mova_ids": t.get("mova_ids"),
         "meta": md}
    return d


def dump_header(path: str) -> dict:
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        return dict(f.metadata())


def aux_cat(dump: dict, aux_ids) -> "torch.Tensor":
    """The drafter's input [N - aux_from, n_aux * 2560]: aux states concatenated in the drafter's
    id order (gpu_model_runner.py:5326-5328 cats in layer order; both are ascending here)."""
    import torch
    return torch.cat([dump["aux"][a] for a in aux_ids], -1)


# ------------------------------------------------------------------------------- main


def cmd_run(a) -> None:
    sources = parse_sources(a)
    if not sources:
        raise SystemExit("no sources (--source / --a4-ref)")
    os.makedirs(a.out_dir, exist_ok=True)
    write_sources_json(a.out_dir, sources)
    aux_ids = [int(i) for i in a.aux.split(",")]
    todo = [s for s in sources if not os.path.exists(os.path.join(a.out_dir, file_name(s["name"])))]
    todo.sort(key=lambda s: len(s["prompt_ids"]) + len(s["cont_ids"]))
    lengths = [len(s["prompt_ids"]) + len(s["cont_ids"]) for s in todo]
    print(f"{len(sources)} sources, {len(sources) - len(todo)} already dumped, {len(todo)} to do; aux = inputs "
          f"of K2 layers {aux_ids}; keep-ctx {a.keep_ctx}; mode {a.mode}, matmul {a.matmul}", flush=True)
    for s in todo:
        print(f"  {s['name']}: {len(s['prompt_ids'])} + {len(s['cont_ids'])} ids"
              + (" (planned sizes)" if s.get("planned") else ""), flush=True)
    if todo:
        e = estimate(lengths, [len(s["prompt_ids"]) for s in todo], a.keep_ctx)
        print(f"estimate: {e['tokens']} tokens, {e['tflop']:.0f} TFLOP, ~{e['est_hours']:.1f} h at "
              f"{DEQUANT_S_PER_FORWARD:.0f} s dequant per forward + {ASSUMED_GFLOPS:.0f} GFLOP/s (assumed); "
              f"peak RSS ~{e['peak_gib']:.1f} GiB, dumps ~{e['disk_gb']:.2f} GB (estimated)", flush=True)
    if a.dry_run or not todo:
        return
    import resource
    import torch
    torch.set_grad_enabled(False)
    t0 = time.time()
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}", flush=True)
    r = build(a.snapshot, mode=a.mode, matmul=a.matmul)
    meta0 = {"snapshot": os.path.abspath(a.snapshot), "mode": a.mode, "matmul": a.matmul, "keep_ctx": a.keep_ctx,
             "weights": f"int4 g{r.c.group_size} dequantised (dequant.py's rule)" if r.c.group_size else "bf16",
             "reference": "tools/oracle/k2_ref.py K2Ref.forward, one sequence, no cache"}
    for s in todo:
        ts = time.time()
        ids = s["prompt_ids"] + s["cont_ids"]
        n_p = len(s["prompt_ids"])
        t, meta = forward_taps(r, ids, n_p, aux_ids, a.keep_ctx, a.topk)
        g = t["greedy"].long()
        cont = torch.tensor(s["cont_ids"])
        agree = float((g[:-1] == cont).float().mean())
        flat = float((t["greedy_flatnorm"].long() == g).float().mean())
        meta.update(meta0)
        meta.update(cont_agree=f"{agree:.4f}", flatnorm_agree=f"{flat:.4f}")
        save_dump(os.path.join(a.out_dir, file_name(s["name"])), s["name"], t, meta)
        print(f"[{s['name']}] {n_p} + {len(s['cont_ids'])} ids: recorded continuation = teacher-forced greedy at "
              f"{agree:.3f}; one-group final norm agrees at {flat:.3f}; {time.time() - ts:.0f} s (dequant wait "
              f"{r.load_seconds:.0f} s), wall {time.time() - t0:.0f} s, peak RSS "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB", flush=True)
    print(f"done: wall {time.time() - t0:.0f} s", flush=True)


def cmd_golden_ids(a) -> None:
    from safetensors import safe_open
    with safe_open(a.run, framework="pt", device="cpu") as f:
        toks = f.get_tensor("tokens").tolist()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, toks)) + "\n")
    print(f"{a.run}: {len(toks)} greedy ids -> {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "sources"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("snapshot")
            p.add_argument("--aux", default=DEFAULT_AUX)
            p.add_argument("--keep-ctx", type=int, default=2048)
            p.add_argument("--topk", type=int, default=16)
            p.add_argument("--mode", choices=("bf16", "f32"), default="bf16")
            p.add_argument("--matmul", choices=("fp32", "bf16"), default="fp32")
            p.add_argument("--dry-run", action="store_true")
        p.add_argument("--out-dir", required=True)
        p.add_argument("--source", action="append")
        p.add_argument("--a4-ref")
        p.add_argument("--plan", action="append", help="NAME:N_PROMPT:N_CONT, a planned source's sizes (--dry-run)")
        p.add_argument("--plan-manifest", action="append",
                       help="MANIFEST.json:N_CONT, an A4 set's prompts with N_CONT ids each (--dry-run)")
    g = sub.add_parser("golden-ids")
    g.add_argument("run")
    g.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "golden-ids":
        return cmd_golden_ids(a)
    if a.cmd == "sources":
        os.makedirs(a.out_dir, exist_ok=True)
        src = parse_sources(a)
        print(f"wrote {write_sources_json(a.out_dir, src)}: {len(src)} sources, "
              f"{sum(len(s['prompt_ids']) for s in src)} prompt + {sum(len(s['cont_ids']) for s in src)} continuation ids")
        return
    cmd_run(a)


if __name__ == "__main__":
    main()
