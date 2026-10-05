#!/usr/bin/env python3
"""Tap dumps for the DFlash P0 (spec 19a Task 2): the bf16 target over [prompt + continuation].

    dump_taps.py run <snapshot> --out-dir DIR [SOURCES] [--layers 5,19,33,47,61]
                 [--batch-tokens 8192] [--keep-ctx 2048] [--topk 64] [--vocab-used 248077]
                 [--no-fp32-matmul] [--dry-run]
    dump_taps.py sources --out-dir DIR [SOURCES] [--rank-logs DIR2]
                                                     write DIR/sources.json only (no model); with
                                                     --rank-logs, the sources as rank.py inputs

SOURCES (any mix, each a name `corpus/label`; the corpus is what dflash_accept.py pools by):
    --source NAME:PROMPT_IDS:CONT_IDS     whitespace-separated id files (the continuation is the
                                          recorded greedy output, teacher-forced)
    --a4 DIR --a4-expected JSON --tokenizer tokenizer.json
                                          the A4 tool-call set: DIR/<scenario>.ids is the
                                          prompt, the reference output is the scenario's text in
                                          tests/server/toolcall_expected.json (bf16 backend
                                          first) re-encoded, + <|im_end|> after a complete call
                                          (tools/spec/lookup_accept.py's load_a4, reused)

One layer-streamed forward (stream.py via dump.stream_weights; the model built exactly as
kv_int8_probe.py's `run` builds it) per batch of sources. A batch is right-padded to its longest
sequence: every layer of Qwen3.5 is causal (attention mask, GDN recurrence, causal conv), so the
padding never reaches a real position - test_dump_taps.py checks a padded batch against
one-sequence forwards. Per source, DIR/<corpus>__<label>.taps.safetensors holds:

    ids        int32 [N]                 prompt + continuation
    tap.L{i}   bf16  [N - tap_from, H]   the residual stream AFTER decoder layer i (dump.py's
                                         resid.L{i}; transformers' hidden_states[i + 1];
                                         dflash_ref.py's taps[i]) at positions tap_from..N-1
    greedy     int32 [N - tap_from]      argmax over ids < vocab_used of the fp32 logits
                                         (final norm, bf16 lm_head upcast), ties to the lower id
    top_ids    int32 [N - tap_from, topk]   the top-k ids by logit (descending, ties lower id)
    top_logits f32   [N - tap_from, topk]   their logits, for a later sampled-acceptance pass
    lse        f32   [N - tap_from]      logsumexp over ids < vocab_used
    metadata   name, n_prompt, n, tap_from, layers, vocab_used, fp32_matmul, snapshot

tap_from = max(0, n_prompt - keep_ctx): the drafter's sliding window (2048) never reads an
older context position from an anchor at or after the first generated token. greedy[t] is the
target's next id AFTER position tap_from + t.

`--fp32-matmul` (default on, as the 12a repeat on this Mac): every bf16 nn.Linear as fp32 sgemm
with one bf16 rounding of the output (kv_int8_probe.fp32_linear) - the engine's fp32
accumulation, and not slower than bf16 GEMM on AVX2.

A re-run skips every source whose output exists (resumable). `--dry-run` prints the batches,
the token counts, a time estimate (this Mac: ~6.5 min of weight streaming per forward, measured
by the 12a repeat; compute at an assumed 300 GFLOP/s) and a peak-memory estimate, and loads
nothing. Run in the oracle container (tools/oracle/README.md, "The DFlash P0").
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

DEFAULT_LAYERS = "5,19,33,47,61"     # z-lab/Qwen3.8-27B-DFlash2 dflash_config.target_layer_ids
VOCAB_USED = 248077                  # Qwen35::kVocabUsed (src/model/qwen35.h), tokenizer.json's ids
STREAM_S_PER_FORWARD = 390.0         # 12a repeat on this Mac: ~6 s per streamed layer x 64 (measured)
ASSUMED_GFLOPS = 300.0               # fp32 sgemm, 16 threads, i9-9980HK (assumed, for --dry-run)


def _load(name):
    mod_name = "dump_taps_" + name.replace("/", "_").replace(".", "_")
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]


def file_name(name: str) -> str:
    return name.replace("/", "__") + ".taps.safetensors"


# ------------------------------------------------------------------------------- sources


def parse_sources(a) -> list[dict]:
    out = []
    for s in a.source or []:
        parts = s.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--source {s}: want NAME:PROMPT_IDS:CONT_IDS")
        name, pf, cf = parts
        if "/" not in name:
            raise SystemExit(f"--source {s}: the name must be corpus/label")
        out.append({"name": name, "prompt_ids": read_ids(pf), "cont_ids": read_ids(cf)})
    if a.a4:
        if not (a.a4_expected and a.tokenizer):
            raise SystemExit("--a4 needs --a4-expected and --tokenizer")
        la = _load("../spec/lookup_accept")
        for s in la.load_a4(a.a4, a.a4_expected, a.tokenizer):
            out.append({"name": "a4/" + s["name"], "prompt_ids": s["prompt_ids"], "cont_ids": s["out_ids"]})
    names = [s["name"] for s in out]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate source names: {sorted(n for n in names if names.count(n) > 1)}")
    for s in out:
        if not s["prompt_ids"] or not s["cont_ids"]:
            raise SystemExit(f"{s['name']}: empty prompt or continuation")
    return out


def write_sources_json(out_dir: str, sources: list[dict]) -> str:
    path = os.path.join(out_dir, "sources.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"name": s["name"], "n_prompt": len(s["prompt_ids"]), "n": len(s["prompt_ids"]) + len(s["cont_ids"]),
                    "file": file_name(s["name"])} for s in sources], f, indent=1)
    return path


def plan_batches(lengths: list[int], batch_tokens: int) -> list[list[int]]:
    """Indices grouped longest first; a batch's padded size (count x longest) stays within
    batch_tokens (a sequence longer than that runs alone)."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[cur[0]] > batch_tokens:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def estimate(lengths, batches, hidden=5120, n_layers_fa=16, heads=24, head_dim=256, ffn=17408,
             params=26.9e9, vocab=248320) -> dict:
    """Time and peak memory, the rough model --dry-run prints (assumed rates, see the header).
    Compute counts the padded batch; memory: the resident embedding + head (bf16), two streamed
    layers (~0.8 GB bf16 each) + one fp32 weight copy (fp32 matmul), the largest batch's fp32
    linear activations, its eager attention scores (bf16 + fp32 softmax + bf16) and a logits chunk."""
    padded = [len(b) * lengths[b[0]] for b in batches]
    flops = sum(2.0 * params * n for n in padded)
    flops += sum(len(b) * 2.0 * heads * head_dim * lengths[b[0]] ** 2 * n_layers_fa for b in batches)
    t = len(batches) * STREAM_S_PER_FORWARD + flops / (ASSUMED_GFLOPS * 1e9)
    worst = 0.0
    for b in batches:
        B, T = len(b), lengths[b[0]]
        act = B * T * (hidden * 4 * 6 + ffn * 4 * 3)
        attn = B * heads * T * T * (2 + 4 + 2)
        worst = max(worst, act + attn)
    resident = 2 * vocab * hidden * 2 + 2 * 0.8e9 + ffn * hidden * 4
    logits = max(lengths) * 16384 * 4 * 2
    return {"tokens": sum(lengths), "padded": sum(padded), "forwards": len(batches),
            "tflop": flops / 1e12, "est_hours": t / 3600, "peak_gib": (resident + worst + logits) / 2**30}


# ------------------------------------------------------------------------------- the target


def build_target(snapshot: str, fp32_matmul: bool = True):
    """Qwen3_5ForCausalLM on meta, layer-streamed (dump.stream_weights), eager attention; the
    same construction as kv_int8_probe.py's `run`. Returns (model, text config, prefetcher)."""
    import torch
    D = _load("dump")
    if fp32_matmul:
        _load("kv_int8_probe").fp32_linear()
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    if D._agnes.is_agnes(raw):
        raise SystemExit("dump_taps.py is for Qwen3.8 (the DFlash drafter's target)")
    quantised = bool(raw.get("quantization_config") or (raw.get("text_config") or {}).get("quantization_config"))
    group_size = D.check_quant_config(snapshot) if quantised else 0
    tc = AutoConfig.from_pretrained(snapshot).get_text_config()
    tc._attn_implementation = "eager"
    with torch.device("meta"):
        model = m.Qwen3_5ForCausalLM(tc)
    pf = D.stream_weights(model, snapshot, group_size, tc, 0)
    model.eval()
    return model, tc, pf


def head_stats(h, w, vocab_used: int, topk: int, chunk: int = 16384):
    """h [T, H] (post-norm hidden), w [V, H] bf16 lm_head. Over ids < vocab_used, fp32 logits:
    greedy [T] (ties to the lower id), top ids / logits [T, topk] (descending, ties lower id),
    logsumexp [T]. The head is upcast per row chunk, so its fp32 copy never exists whole."""
    import torch
    T = h.shape[0]
    hf = h.float()
    vs, ids = [], []
    lse = torch.full((T,), float("-inf"), dtype=torch.float64)
    for c0 in range(0, vocab_used, chunk):
        c1 = min(vocab_used, c0 + chunk)
        lg = hf @ w[c0:c1].float().t()                                     # [T, c]
        lse = torch.logaddexp(lse, torch.logsumexp(lg.double(), -1))
        v, i = torch.topk(lg, min(topk, c1 - c0), dim=-1)
        vs.append(v)
        ids.append(i + c0)
    v, i = torch.cat(vs, -1), torch.cat(ids, -1)
    o = torch.argsort(i, dim=-1, stable=True)                              # by id, then by value:
    v, i = v.gather(-1, o), i.gather(-1, o)                                # equal logits keep id order
    o = torch.sort(v, dim=-1, descending=True, stable=True).indices[:, :topk]
    v, i = v.gather(-1, o), i.gather(-1, o)
    return i[:, 0].clone(), i, v, lse.float()


def forward_taps(model, seqs, layers, vocab_used: int, topk: int = 64, keep_from=None, pad_id: int = 0):
    """One forward of the (right-padded) batch `seqs`. Per sequence: {'taps': {i: bf16 [n - f, H]},
    'greedy', 'top_ids', 'top_logits', 'lse'} from position f = keep_from[b] (default 0)."""
    import torch
    B = len(seqs)
    T = max(len(s) for s in seqs)
    keep_from = keep_from or [0] * B
    x = torch.full((B, T), pad_id, dtype=torch.long)
    for b, s in enumerate(seqs):
        x[b, :len(s)] = torch.tensor(s, dtype=torch.long)
    caps = {}

    def hook(i):
        def fn(_mod, _args, out):
            t = out[0] if isinstance(out, tuple) else out
            caps[i] = t.detach().to(torch.bfloat16)
        return fn
    hs = [model.model.layers[i].register_forward_hook(hook(i)) for i in layers]
    try:
        with torch.no_grad():
            last = model.model(input_ids=x, use_cache=False).last_hidden_state
    finally:
        for h in hs:
            h.remove()
    if sorted(caps) != sorted(layers):
        raise RuntimeError(f"tap hooks captured layers {sorted(caps)}, wanted {sorted(layers)}")
    w = model.lm_head.weight
    out = []
    for b, s in enumerate(seqs):
        f, n = keep_from[b], len(s)
        g, ti, tv, lse = head_stats(last[b, f:n], w, vocab_used, topk)
        out.append({"taps": {i: caps[i][b, f:n].contiguous() for i in layers}, "greedy": g,
                    "top_ids": ti, "top_logits": tv, "lse": lse})
    return out


def save_dump(path: str, name: str, ids, n_prompt: int, tap_from: int, res: dict, meta: dict) -> None:
    import torch
    from safetensors.torch import save_file
    t = {"ids": torch.tensor(ids, dtype=torch.int32), "greedy": res["greedy"].to(torch.int32),
         "top_ids": res["top_ids"].to(torch.int32), "top_logits": res["top_logits"].float(),
         "lse": res["lse"].float()}
    for i, v in res["taps"].items():
        t[f"tap.L{i}"] = v
    if not all(torch.isfinite(v.float()).all() for k, v in t.items() if k.startswith("tap.")):
        raise RuntimeError(f"{name}: non-finite taps")
    md = {"name": name, "n_prompt": str(n_prompt), "n": str(len(ids)), "tap_from": str(tap_from),
          "layers": ",".join(str(i) for i in sorted(res["taps"]))}
    md.update({k: str(v) for k, v in meta.items()})
    tmp = path + ".tmp"
    save_file(t, tmp, metadata=md)
    os.replace(tmp, path)


def load_dump(path: str) -> dict:
    """A dump as {'name', 'n_prompt', 'n', 'tap_from', 'layers', 'ids', 'greedy', 'taps': {i: ...},
    'top_ids', 'top_logits', 'lse', 'meta'}; greedy etc. cover positions tap_from..n-1."""
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        md = f.metadata()
        t = {k: f.get_tensor(k) for k in f.keys()}
    layers = [int(i) for i in md["layers"].split(",")]
    return {"name": md["name"], "n_prompt": int(md["n_prompt"]), "n": int(md["n"]),
            "tap_from": int(md["tap_from"]), "layers": layers, "ids": t["ids"].long(),
            "greedy": t["greedy"].long(), "taps": {i: t[f"tap.L{i}"] for i in layers},
            "top_ids": t["top_ids"].long(), "top_logits": t["top_logits"], "lse": t["lse"], "meta": md}


def dump_header(path: str) -> dict:
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        return dict(f.metadata())


# ------------------------------------------------------------------------------- main


def cmd_run(a) -> None:
    sources = parse_sources(a)
    if not sources:
        raise SystemExit("no sources (--source / --a4)")
    os.makedirs(a.out_dir, exist_ok=True)
    write_sources_json(a.out_dir, sources)
    layers = [int(i) for i in a.layers.split(",")]
    todo = [s for s in sources if not os.path.exists(os.path.join(a.out_dir, file_name(s["name"])))]
    lengths = [len(s["prompt_ids"]) + len(s["cont_ids"]) for s in todo]
    batches = plan_batches(lengths, a.batch_tokens)
    est = estimate(lengths, batches) if todo else None
    print(f"{len(sources)} sources, {len(sources) - len(todo)} already dumped, {len(todo)} to do in "
          f"{len(batches)} forwards (batch-tokens {a.batch_tokens}); taps after layers {layers}", flush=True)
    for bi, b in enumerate(batches):
        print(f"  batch {bi}: " + ", ".join(f"{todo[i]['name']} ({lengths[i]})" for i in b), flush=True)
    if est:
        print(f"estimate (Qwen3.8's sizes): {est['tokens']} tokens ({est['padded']} padded), {est['tflop']:.0f} TFLOP, "
              f"~{est['est_hours']:.1f} h at {STREAM_S_PER_FORWARD:.0f} s streaming per forward + "
              f"{ASSUMED_GFLOPS:.0f} GFLOP/s (assumed); peak RSS ~{est['peak_gib']:.1f} GiB (estimated)", flush=True)
    if a.dry_run or not todo:
        return
    import resource
    import torch
    torch.set_grad_enabled(False)
    t0 = time.time()
    print(f"torch threads {torch.get_num_threads()}, fp32 matmul {a.fp32_matmul}", flush=True)
    model, tc, pf = build_target(a.snapshot, a.fp32_matmul)
    if max(layers) >= tc.num_hidden_layers - 1:
        raise SystemExit(f"tap layer {max(layers)}: the last layer's output is not a residual tap")
    meta = {"snapshot": os.path.abspath(a.snapshot), "vocab_used": a.vocab_used, "fp32_matmul": a.fp32_matmul,
            "keep_ctx": a.keep_ctx, "topk": a.topk, "weights": "layer-streamed (tools/oracle/stream.py)"}
    for bi, b in enumerate(batches):
        tb = time.time()
        seqs = [todo[i]["prompt_ids"] + todo[i]["cont_ids"] for i in b]
        keep = [max(0, len(todo[i]["prompt_ids"]) - a.keep_ctx) for i in b]
        res = forward_taps(model, seqs, layers, a.vocab_used, a.topk, keep)
        for j, i in enumerate(b):
            s = todo[i]
            n_p = len(s["prompt_ids"])
            cont_g = res[j]["greedy"][n_p - 1 - keep[j]:-1]
            agree = (cont_g == torch.tensor(s["cont_ids"])).float().mean().item()
            save_dump(os.path.join(a.out_dir, file_name(s["name"])), s["name"], seqs[j], n_p, keep[j], res[j],
                      dict(meta, cont_agree=f"{agree:.4f}"))
            print(f"[{s['name']}] {n_p} + {len(s['cont_ids'])} ids, taps from {keep[j]}; recorded continuation = "
                  f"bf16 greedy at {agree:.3f} of its ids", flush=True)
        print(f"batch {bi}/{len(batches)}: {time.time() - tb:.0f} s (dequant wait total {pf.seconds:.0f} s), "
              f"wall {time.time() - t0:.0f} s, peak RSS "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB", flush=True)
    print(f"done: wall {time.time() - t0:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "sources"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("snapshot")
            p.add_argument("--layers", default=DEFAULT_LAYERS)
            p.add_argument("--batch-tokens", type=int, default=8192)
            p.add_argument("--keep-ctx", type=int, default=2048)
            p.add_argument("--topk", type=int, default=64)
            p.add_argument("--vocab-used", type=int, default=VOCAB_USED)
            p.add_argument("--no-fp32-matmul", dest="fp32_matmul", action="store_false")
            p.add_argument("--dry-run", action="store_true")
        else:
            p.add_argument("--rank-logs", help="also write the sources as rank.py request logs here")
        p.add_argument("--out-dir", required=True)
        p.add_argument("--source", action="append")
        p.add_argument("--a4")
        p.add_argument("--a4-expected")
        p.add_argument("--tokenizer")
    a = ap.parse_args()
    if a.cmd == "sources":
        os.makedirs(a.out_dir, exist_ok=True)
        src = parse_sources(a)
        print(f"wrote {write_sources_json(a.out_dir, src)}: {len(src)} sources, "
              f"{sum(len(s['cont_ids']) for s in src)} continuation ids")
        if a.rank_logs:
            # tools/draft_vocab/rank.py's request-log input: prompt ids at weight 1, the
            # continuation as generated ids at weight 4 (its defaults)
            os.makedirs(a.rank_logs, exist_ok=True)
            for k, s in enumerate(src):
                with open(os.path.join(a.rank_logs, f"{k + 1:06d}.json"), "w", encoding="utf-8") as f:
                    json.dump({"name": s["name"], "prompt_ids": s["prompt_ids"], "out_ids": s["cont_ids"]}, f)
            print(f"wrote {len(src)} rank.py request logs to {a.rank_logs}")
        return
    cmd_run(a)


if __name__ == "__main__":
    main()
