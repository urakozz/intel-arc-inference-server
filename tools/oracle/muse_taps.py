#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 23f Task 2: Muse-Glimmer tap dumps in dump_taps.py's format, so dflash_accept.py reads them unchanged.

    muse_taps.py run <snapshot> --source NAME:IDS:N_PROMPT [...] --out DIR [--layers 1,13,25,37,49] [--chunk 2048]
                     [--topk 64] [--form auto|gptq|bf16] [--dry-run]
    muse_taps.py from-golden <p.golden.safetensors> --name NAME --out DIR [--prompt <p.ids>] [--topk 64]

`run`: one TEACHER-FORCED forward per source over IDS (= the prompt's N_PROMPT ids followed by the recorded
continuation - the int4 reference's own greedy output, 23a's A4 / golden runs: what we serve) through muse_ref.py's
streamed model (transformers 5.19.0's own MuseGlimmerForConditionalGeneration, eager, bf16; the int4 checkpoint
dequantised on demand), in chunks of --chunk ids (muse_ref.forward_chunks: whole up to 2048, the reference's prefill
arrangement), with muse_ref.Recorder's taps - the OUTPUT of decoder layers --layers (OC:112-131: transformers'
`output_hidden_states=[1, 13, 25, 37, 49]`, hidden_states[i] = layer i's output, no embedding prepended; the
assistant's target_layer_ids). `from-golden`: a golden file muse_ref.py `run --taps` already wrote (taps for every row,
prompt and generated) converted without another forward: ids = its prompt + its `tokens` (the reference's greedy
continuation).

The dump (dump_taps.save_dump; NAME's `/` becomes `__`): ids, `tap.L{i}` bf16 [N - tap_from][H], and per row
t >= tap_from
    greedy      int32  argmax of the RAW lm_head row widened to fp32 (ties: the lower id) - the engine's greedy rule
                       (the device argmax over raw fp32 logits; spec 23 §1.1: the softcap is monotone)
    top_ids     int32  [64] the top ids of the CAPPED row (the model's distribution at T 1.0), descending, ties lower id
    top_logits  f32    [64] their capped logits
    lse         f32    logsumexp of the capped row
tap_from = max(0, N_PROMPT - 2048): the drafter's window (|q - k| <= 2048, decision 13) never reads an older context
position from an anchor at or after the first generated token. In `from-golden`, rows before the golden's
logits_row0 (a long prompt's golden keeps the last 1024 prompt rows' logits) carry greedy / top_ids -1 and NaN
logits - dflash_accept reads rows >= N_PROMPT - 1 only, and the conversion refuses a golden whose logits start later.

Resumable: a source whose dump exists is skipped. `--dry-run` prints the sources, the tokens and an ESTIMATED time
(23a measured ~53 ms per layer per token for a 2048-id chunk on the Mac's CPU, ~2.8 s a token at 52 layers; the
box ~2-3x faster - ESTIMATED) and loads nothing. Run in the oracle image with spec 21a's 5.19.0 site first on
PYTHONPATH (tools/box_validate/muse_oracle.sh <data> taps).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

DEFAULT_LAYERS = "1,13,25,37,49"     # MuseGlimmerAssistantConfig.target_layer_ids (AC:89)
WINDOW = 2048                        # the assistant's sliding_window (|q - k| <= 2048)
MAC_S_PER_LAYER_TOKEN = 0.053        # 23a, measured on the Mac (a 2048-id chunk, the int4 form dequantised)


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]


def parse_sources(specs) -> list[dict]:
    out = []
    for s in specs or []:
        parts = s.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--source {s}: want NAME:IDS:N_PROMPT")
        name, path, n_prompt = parts
        if "/" not in name:
            raise SystemExit(f"--source {s}: the name must be corpus/label (dflash_accept pools by the corpus)")
        ids = read_ids(path)
        n_prompt = int(n_prompt)
        if not 1 <= n_prompt < len(ids):
            raise SystemExit(f"--source {s}: N_PROMPT {n_prompt} must leave a continuation ({len(ids)} ids)")
        out.append({"name": name, "ids": ids, "n_prompt": n_prompt})
    names = [s["name"] for s in out]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate source names: {sorted(n for n in names if names.count(n) > 1)}")
    return out


def tap_from_of(n_prompt: int) -> int:
    return max(0, n_prompt - WINDOW)


def top_k_stable(x, k: int):
    """Top-k per row, descending, equal values in increasing id order: topk, then a stable sort of the rows whose
    k-th value is shared past the boundary (dflash_accept.top_k_fast's rule)."""
    import torch
    v, i = torch.topk(x, k, dim=-1)
    o = torch.argsort(i, dim=-1, stable=True)
    v, i = v.gather(-1, o), i.gather(-1, o)
    o = torch.sort(v, dim=-1, descending=True, stable=True).indices
    v, i = v.gather(-1, o), i.gather(-1, o)
    over = (x >= v[:, -1:]).sum(-1) > k
    if over.any():
        rows = over.nonzero().flatten()
        sv, si = torch.sort(x[rows], dim=-1, descending=True, stable=True)
        v[rows], i[rows] = sv[:, :k], si[:, :k]
    return i, v


def row_stats(raw, cap, topk: int) -> dict:
    """raw / cap [R][V] (the model's dtype) -> greedy (raw fp32 argmax, the lower id on ties), the capped row's top-k
    and logsumexp."""
    import torch
    rf = raw.float()
    best = rf.max(-1, keepdim=True).values
    greedy = torch.where(rf == best, torch.arange(rf.shape[1])[None], rf.shape[1]).min(-1).values
    cf = cap.float()
    ti, tv = top_k_stable(cf, topk)
    return {"greedy": greedy, "top_ids": ti, "top_logits": tv, "lse": torch.logsumexp(cf, -1)}


def estimate(sources, layers=52) -> str:
    toks = sum(len(s["ids"]) for s in sources)
    h = toks * layers * MAC_S_PER_LAYER_TOKEN / 3600
    return (f"{len(sources)} sources, {toks} tokens teacher-forced: ~{h:.1f} h on the Mac's CPU at 23a's "
            f"{MAC_S_PER_LAYER_TOKEN * 1000:.0f} ms a layer-token (measured), the box ~{h / 2.5:.1f} h (ESTIMATED); peak "
            "RSS ~18 GiB (23a's golden runs: 17.7 GiB) + the taps (5 x 6656 x 2 B a token)")


def cmd_run(a) -> None:
    sources = parse_sources(a.source)
    if not sources:
        raise SystemExit("no --source")
    dt = _load("dump_taps")
    layers = [int(i) for i in a.layers.split(",")]
    os.makedirs(a.out, exist_ok=True)
    todo = [s for s in sources if not os.path.exists(os.path.join(a.out, dt.file_name(s["name"])))]
    print(f"{len(sources)} sources, {len(sources) - len(todo)} already dumped; taps after layers {layers}; "
          + estimate(todo), flush=True)
    for s in todo:
        print(f"  {s['name']}: {len(s['ids'])} ids ({s['n_prompt']} prompt), tap_from {tap_from_of(s['n_prompt'])}",
              flush=True)
    if a.dry_run or not todo:
        return
    import torch
    mr = _load("muse_ref")
    torch.set_grad_enabled(False)
    t0 = time.time()
    tc = mr.text_config(a.snapshot)
    model, pf = mr.build_streamed(a.snapshot, tc, a.form)
    print(f"model: {tc.num_hidden_layers} layers, form {model.muse_form}, built {time.time() - t0:.1f}s; "
          f"threads {torch.get_num_threads()}", flush=True)
    for s in todo:
        ts = time.time()
        res = forward_source(mr, model, tc, s["ids"], s["n_prompt"], layers, a.chunk, a.topk)
        meta = {"snapshot": os.path.abspath(a.snapshot), "form": model.muse_form, "transformers": mr.PINNED_TRANSFORMERS,
                "source": "teacher-forced (muse_taps.py run)", "vocab_used": str(tc.vocab_size),
                "chunk": str(mr.auto_chunk(len(s["ids"]), a.chunk))}
        dt.save_dump(os.path.join(a.out, dt.file_name(s["name"])), s["name"], s["ids"], s["n_prompt"],
                     tap_from_of(s["n_prompt"]), res, meta)
        print(f"  {s['name']}: {time.time() - ts:.0f} s", flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"wall {time.time() - t0:.0f} s, peak RSS {rss:.1f} GiB, prefetch wait {pf.seconds:.1f}s", flush=True)


def forward_source(mr, model, tc, ids, n_prompt: int, layers, chunk: int, topk: int) -> dict:
    """One teacher-forced forward over ids: the taps of rows >= tap_from and the per-row head stats."""
    import torch
    f = tap_from_of(n_prompt)
    rec = mr.Recorder(model, tc, acts=False, taps=tuple(layers))
    if list(rec.taps) != list(layers):
        rec.remove()
        raise SystemExit(f"--layers {layers}: the model has {tc.num_hidden_layers} layers")
    parts = []

    def on_rows(r0, raw, cap):
        for b0 in range(0, cap.shape[0], 256):
            ra = r0 + b0
            rb, cb = raw[b0:b0 + 256], cap[b0:b0 + 256]
            k = max(0, f - ra)
            if k < rb.shape[0]:
                parts.append(row_stats(rb[k:], cb[k:], topk))
    try:
        mr.forward_chunks(model, ids, mr.auto_chunk(len(ids), chunk), rec, on_rows=on_rows)
        t = rec.tensors()
    finally:
        rec.remove()
    res = {k: torch.cat([p[k] for p in parts]) for k in ("greedy", "top_ids", "top_logits", "lse")}
    taps = t["taps"][f:]                                   # [N - f][n][H]
    res["taps"] = {L: taps[:, j].contiguous() for j, L in enumerate(layers)}
    if res["greedy"].shape[0] != len(ids) - f or taps.shape[0] != len(ids) - f:
        raise RuntimeError(f"rows: {res['greedy'].shape[0]} head rows, {taps.shape[0]} tap rows, expected {len(ids) - f}")
    return res


def cmd_from_golden(a) -> None:
    import torch
    from safetensors import safe_open
    dt = _load("dump_taps")
    with safe_open(a.golden, framework="pt") as h:
        md = h.metadata()
        t = {k: h.get_tensor(k) for k in ("taps", "logits.raw", "logits.cap", "tokens")}
    prompt_path = a.prompt or a.golden.replace(".golden.safetensors", ".ids")
    prompt = read_ids(prompt_path)
    T, gen, r0 = int(md["n_prompt"]), int(md["gen"]), int(md["logits_row0"])
    layers = json.loads(md["tap_layers"])
    if len(prompt) != T:
        raise SystemExit(f"{prompt_path}: {len(prompt)} ids, the golden's n_prompt is {T}")
    if not layers:
        raise SystemExit(f"{a.golden}: no taps (muse_ref.py run --taps)")
    if gen < 2:
        raise SystemExit(f"{a.golden}: --gen {gen} leaves no continuation to draft over")
    ids = prompt + t["tokens"].tolist()
    N, f = len(ids), tap_from_of(T)
    if r0 > T - 1:
        raise SystemExit(f"{a.golden}: logits from row {r0} > n_prompt - 1 = {T - 1}: the anchors need the rows from "
                         "the last prompt position on (re-run with a larger --logits-tail)")
    if t["taps"].shape[0] != N or t["logits.raw"].shape[0] != N - r0:
        raise SystemExit(f"{a.golden}: {t['taps'].shape[0]} tap rows / {t['logits.raw'].shape[0]} logit rows for {N} ids")
    st = row_stats(t["logits.raw"][max(0, f - r0):], t["logits.cap"][max(0, f - r0):], a.topk)
    pad = max(0, r0 - f)                                    # rows before the golden's logits: never read
    if pad:
        st = {"greedy": torch.cat([torch.full((pad,), -1, dtype=torch.long), st["greedy"]]),
              "top_ids": torch.cat([torch.full((pad, a.topk), -1, dtype=torch.long), st["top_ids"]]),
              "top_logits": torch.cat([torch.full((pad, a.topk), float("nan")), st["top_logits"]]),
              "lse": torch.cat([torch.full((pad,), float("nan")), st["lse"]])}
    st["taps"] = {L: t["taps"][f:, j].contiguous() for j, L in enumerate(layers)}
    os.makedirs(a.out, exist_ok=True)
    meta = {"source": f"from-golden {os.path.abspath(a.golden)}", "golden_tokens_rule": "capped argmax (generate)",
            "logits_from": str(max(r0, f)), "vocab_used": str(t["logits.raw"].shape[1]),
            "snapshot": md.get("snapshot", ""), "form": md.get("form", ""), "transformers": md.get("transformers", "")}
    path = os.path.join(a.out, dt.file_name(a.name))
    dt.save_dump(path, a.name, ids, T, f, st, meta)
    raw_vs_cap = int((st["greedy"][T - 1 - f:N - 1 - f] != torch.tensor(ids[T:])).sum())
    print(f"wrote {path}: {N} ids ({T} prompt, {gen} generated), taps {layers} from row {f}; the recorded continuation "
          f"(capped argmax) differs from the raw-row greedy at {raw_vs_cap} of {gen} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("snapshot")
    r.add_argument("--source", action="append")
    r.add_argument("--out", required=True)
    r.add_argument("--layers", default=DEFAULT_LAYERS)
    r.add_argument("--chunk", type=int, default=-1, help="-1: whole up to 2048 ids, else 2048 (muse_ref.auto_chunk)")
    r.add_argument("--topk", type=int, default=64)
    r.add_argument("--form", default="auto", choices=("auto", "gptq", "bf16"))
    r.add_argument("--dry-run", action="store_true")
    g = sub.add_parser("from-golden")
    g.add_argument("golden")
    g.add_argument("--name", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--prompt")
    g.add_argument("--topk", type=int, default=64)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        if "/" not in a.name:
            raise SystemExit("--name must be corpus/label")
        cmd_from_golden(a)


if __name__ == "__main__":
    main()
