#!/usr/bin/env python3
"""Spec 30 plan 30a Tasks 2-3: the CPU reference of Laguna S 2.1 (`poolside/Laguna-S-2.1`).

The model is the VENDOR's own code (the snapshot's `modeling_laguna.py` / `configuration_laguna.py`, loaded by
transformers' trust_remote_code; the config's auto_map names them), under transformers 5.19.0 (refused on any other
version: laguna_requirements.txt), in float32 arithmetic (the box's CPU has no bf16 units: bf16 weights are widened
once, exactly, and every op runs in f32). It is built on `meta` and LAYER-STREAMED (stream.py): embedding, final norm
and lm_head resident, each decoder layer's weights read from the shards by a forward pre-hook and dropped after. The
file's per-expert `mlp.experts.<e>.{gate,up,down}_proj.weight` are stacked into the vendor's fused
`gate_up_proj` [E][2I][H] (gate rows first: its forward chunks gate, up) / `down_proj` [E][H][I], and
`mlp.experts.e_score_correction_bias` goes to `mlp.gate.e_score_correction_bias` - the vendor's own load mapping.
Attention is sdpa (torch's CPU flash kernel: no [T][T] score matrix, so a 12k-row transcript fits).

    laguna_ref.py run <snapshot> --prompt <ids> --out <p>.golden.safetensors [--gen 32] [--logits-tail L]
                      [--act-tail A]
    laguna_ref.py trace <snapshot> --source NAME:IDS[:ANSWER_FROM] [...] [--list FILE] --out <dir>

`run`: the prompt in one forward, then greedy decode with the cache (each step re-streams every layer: fine for the
tiny model and for a truncated or int4-sized checkpoint, not for the 235 GB original). Writes
    ids           i32 [T]           the prompt
    tokens        i32 [gen]         greedy
    logits        f32 [Lr][V]       rows logits_row0 .. T+gen-1
    nll           f32 [T-1]         the prompt's teacher-forced NLL
    H.L<l>        f32 [A][H]        decoder layer l's output (before the final norm), rows act_row0 .. T+gen-1
    route.ids.L<l> i32 [T+gen][k]   MoE layers: the router's top-k in ITS order (descending biased score)
    route.w.L<l>  f32 [T+gen][k]    their weights: unbiased sigmoid scores / their sum (before the 2.5 scale)

`trace` (Task 3): teacher-forced routes of many sources in ONE pass over the checkpoint - every source's hidden
states held (f32), layer i read once, run over every source, dropped (stream.py's hold mode); no generation. Writes
`<out>/<name>.routes.safetensors` in offload_curve.py's form: ids i32 [T][L][k] ascending expert id (L = the MoE
layers, 47 on the original: layer 0 is dense), p f32 [T][L][k] the chosen experts' weights aligned with ids;
metadata layers / experts / top_k / moe_layers (the decoder layers the L axis holds) / answer_from (the first
answer position: rows from it are the answer, before it the prompt; T when the source has no answer). The log
states seconds per layer and peak RSS. A source already written is skipped (a restart resumes).
`--list FILE`: lines `NAME<TAB>IDS<TAB>ANSWER_FROM` (laguna_prompts.py writes one).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

PINNED_TRANSFORMERS = "5.19.0"
MODEL = "Laguna S 2.1 (the vendor's LagunaForCausalLM, tools/oracle/laguna_ref.py, layer-streamed, f32)"
TRACE_FORMAT = ("ids i32 [T][L][k]: each row's top-k per MoE layer, ascending expert id; p f32 [T][L][k]: the "
                "chosen experts' routing weights (unbiased sigmoid scores / their sum, f32, before the routed "
                "scale), aligned with ids; L = the decoder layers in metadata moe_layers; rows >= answer_from are "
                "the answer")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"laguna_{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stream = _load("stream")


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(2)


def require_version() -> None:
    import transformers
    if transformers.__version__ != PINNED_TRANSFORMERS:
        raise RuntimeError(f"laguna_ref runs the vendor's code under transformers {PINNED_TRANSFORMERS}; this is "
                           f"{transformers.__version__} ({os.path.dirname(transformers.__file__)}) - install "
                           f"tools/oracle/laguna_requirements.txt")


def rss_gib() -> tuple[float, float]:
    """(current, peak) resident set in GiB (Linux: /proc; ru_maxrss is KiB there)."""
    cur = 0.0
    try:
        with open("/proc/self/statm") as f:
            cur = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except OSError:
        pass
    return cur, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def load_config(snapshot: str):
    require_version()
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(snapshot, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    return cfg


def moe_layers(cfg) -> list[int]:
    """The decoder layers with routed experts (the vendor's LagunaDecoderLayer rule)."""
    return [i for i in range(cfg.num_hidden_layers)
            if i not in cfg.mlp_only_layers and cfg.num_experts > 0 and (i + 1) % cfg.decoder_sparse_step == 0]


def layer_state(keys, get, i: int, cfg) -> dict[str, torch.Tensor]:
    """Decoder layer i's module state dict (f32) from the original's file names."""
    p = f"model.layers.{i}."
    sd = {}
    E = cfg.num_experts
    for k in (k for k in keys if k.startswith(p)):
        r = k[len(p):]
        if r.startswith("mlp.experts."):
            continue
        sd[r] = get(k).float()
    if i in moe_layers(cfg):
        sd["mlp.gate.e_score_correction_bias"] = get(p + "mlp.experts.e_score_correction_bias").float()
        e = p + "mlp.experts."
        sd["mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat([get(f"{e}{j}.gate_proj.weight"), get(f"{e}{j}.up_proj.weight")]) for j in range(E)]).float()
        sd["mlp.experts.down_proj"] = torch.stack([get(f"{e}{j}.down_proj.weight") for j in range(E)]).float()
    return sd


def build_streamed(snapshot: str):
    """The vendor's LagunaForCausalLM on meta, f32, layer-streamed. -> (model, prefetcher)."""
    cfg = load_config(snapshot)
    from transformers import AutoModelForCausalLM
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.float32)
    model.eval()
    tm = model.model
    keys, get = _stream.checkpoint_reader(snapshot)
    resident = {k: get(k).float() for k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")}
    rot = type(tm.rotary_emb)
    rebuild = [(tm, "rotary_emb", lambda: rot(config=tm.rotary_emb.config))]
    if tm.swa_rotary_emb is not None:
        rebuild.append((tm, "swa_rotary_emb", lambda: rot(config=tm.swa_rotary_emb.config)))
    layers = list(tm.layers[: cfg.num_hidden_layers])
    pf = _stream.attach(model, layers, lambda i: layer_state(keys, get, i, cfg), resident, rebuild=rebuild)
    return model, pf


class Routes:
    """Forward hooks on every MoE layer's router: rows[key][layer] = [(ids [n][k] in top-k order, w [n][k] f32)]."""

    def __init__(self, model):
        self.key = 0
        self.rows: dict = {}
        self.hooks = []
        cfg = model.model.config
        for i in moe_layers(cfg):
            def hook(_m, _a, out, i=i):
                _logits, w, sel = out
                self.rows.setdefault(self.key, {}).setdefault(i, []).append((sel.to(torch.int32), w.float()))
            self.hooks.append(model.model.layers[i].mlp.gate.register_forward_hook(hook))

    def get(self, key, i):
        r = self.rows[key][i]
        return torch.cat([a for a, _ in r]), torch.cat([b for _, b in r])

    def remove(self):
        for h in self.hooks:
            h.remove()


def prologue(model, ids: list[int]) -> dict:
    """LagunaModel.forward's prologue for one sequence without a cache (vendor modeling_laguna.py, forward): the
    embedding, the position ids, one mask per layer type, one rotary table per layer type."""
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    tm = model.model
    cfg = tm.config
    h = tm.embed_tokens(torch.tensor([list(ids)]))
    position_ids = torch.arange(h.shape[1]).unsqueeze(0)
    mk = {"config": cfg, "inputs_embeds": h, "attention_mask": None, "past_key_values": None,
          "position_ids": position_ids}
    masks = {"full_attention": create_causal_mask(**mk)}
    pe = tm.rotary_emb(h, position_ids)
    pes = {"full_attention": pe}
    if "sliding_attention" in cfg.layer_types:
        masks["sliding_attention"] = create_sliding_window_causal_mask(**mk)
        pes["sliding_attention"] = tm.swa_rotary_emb(h, position_ids) if tm.swa_rotary_emb is not None else pe
    return {"h": h, "pos": position_ids, "masks": masks, "pe": pes}


@torch.no_grad()
def trace_routes(model, pf, sources: list[tuple[str, list[int]]], log=print) -> dict:
    """Teacher-forced routes of every source in one layer-major pass: each decoder layer is materialised once
    (hold mode) and run over every source with the arguments LagunaModel.forward's loop passes. -> {name:
    {"ids": i32 [T][L][k] ascending, "p": f32 aligned, "last": f32 [T][H] the last layer's output}}."""
    tm = model.model
    cfg = tm.config
    rec = Routes(model)
    ps = [prologue(model, ids) for _, ids in sources]
    pf.hold = True
    try:
        for i, layer in enumerate(tm.layers[: cfg.num_hidden_layers]):
            t = time.time()
            for j, p in enumerate(ps):
                rec.key = j
                p["h"] = layer(p["h"], attention_mask=p["masks"][layer.attention_type], position_ids=p["pos"],
                               past_key_values=None, use_cache=False, position_embeddings=p["pe"][layer.attention_type])
            pf.release()
            cur, peak = rss_gib()
            log(f"layer {i}: {time.time() - t:.1f}s over {len(ps)} sources; RSS {cur:.1f} GiB, peak {peak:.1f} GiB",
                flush=True)
    finally:
        pf.hold = False
        pf.release()
        rec.remove()
    L = moe_layers(cfg)
    out = {}
    for j, (name, ids) in enumerate(sources):
        sel = torch.stack([rec.get(j, i)[0] for i in L], 1)
        w = torch.stack([rec.get(j, i)[1] for i in L], 1)
        order = sel.argsort(-1)
        out[name] = {"ids": sel.gather(-1, order).contiguous(), "p": w.gather(-1, order).contiguous(),
                     "last": ps[j]["h"][0]}
        if out[name]["ids"].shape[0] != len(ids):
            raise RuntimeError(f"{name}: {out[name]['ids'].shape[0]} rows, expected {len(ids)}")
    return out


def read_ids(path: str) -> list[int]:
    with open(path, encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]


def read_sources(specs: list[str], list_file: str | None) -> list[tuple[str, str, int | None]]:
    """(name, ids path, answer_from or None) from NAME:IDS[:ANSWER_FROM] and the list file's lines."""
    out = []
    for s in specs:
        parts = s.split(":")
        if len(parts) not in (2, 3):
            die(f"--source {s}: expected NAME:IDS[:ANSWER_FROM]")
        out.append((parts[0], parts[1], int(parts[2]) if len(parts) == 3 else None))
    if list_file:
        base = os.path.dirname(os.path.abspath(list_file))
        with open(list_file, encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    name, path, af = line.rstrip("\n").split("\t")
                    out.append((name, os.path.join(base, path), int(af)))
    return out


def cmd_trace(a) -> None:
    from safetensors.torch import save_file
    t0 = time.time()
    srcs = []
    for name, path, af in read_sources(a.source, a.list):
        ids = read_ids(path)
        af = len(ids) if af is None else af
        if not 0 <= af <= len(ids):
            die(f"{name}: answer_from {af} outside 0..{len(ids)}")
        srcs.append((name, ids, af))
    if not srcs:
        die("trace: no --source / --list")
    os.makedirs(a.out, exist_ok=True)
    todo = [s for s in srcs if not os.path.exists(os.path.join(a.out, f"{s[0]}.routes.safetensors"))]
    print(f"torch {torch.__version__}, transformers {PINNED_TRANSFORMERS}, intra-op threads {torch.get_num_threads()}; "
          f"trace: {len(srcs)} sources, {len(srcs) - len(todo)} already written, {sum(len(s[1]) for s in todo)} rows "
          f"to run", flush=True)
    if not todo:
        return
    model, pf = build_streamed(a.snapshot)
    cfg = model.model.config
    L = moe_layers(cfg)
    res = trace_routes(model, pf, [(n, ids) for n, ids, _ in todo])
    for name, ids, af in todo:
        o = res[name]
        meta = {"model": "Laguna-S-2.1", "checkpoint": os.path.abspath(a.snapshot),
                "checkpoint_revision": os.path.basename(os.path.normpath(a.snapshot)),
                "transformers": PINNED_TRANSFORMERS, "torch": torch.__version__, "source": name, "T": str(len(ids)),
                "layers": str(len(L)), "moe_layers": ",".join(map(str, L)), "top_k": str(cfg.num_experts_per_tok),
                "experts": str(cfg.num_experts), "answer_from": str(af), "dtype": "fp32",
                "attn_implementation": "sdpa", "format": TRACE_FORMAT}
        save_file({"ids": o["ids"], "p": o["p"]}, os.path.join(a.out, f"{name}.routes.safetensors"), metadata=meta)
        distinct = [int(torch.unique(o["ids"][:, i]).numel()) for i in range(len(L))]
        print(f"trace {name}: {len(ids)} rows (answer from {af}); distinct experts per layer min {min(distinct)} "
              f"median {sorted(distinct)[len(L) // 2]} max {max(distinct)}", flush=True)
    cur, peak = rss_gib()
    print(f"wall {time.time() - t0:.0f}s, peak RSS {peak:.1f} GiB, prefetch wait {pf.seconds:.0f}s")


@torch.no_grad()
def run(snapshot: str, ids: list[int], gen: int, logits_tail: int = -1, act_tail: int = -1) -> tuple[dict, dict]:
    """The prompt in one forward with a cache, then `gen` greedy steps. -> (tensors, metadata)."""
    from transformers import DynamicCache
    model, pf = build_streamed(snapshot)
    cfg = model.model.config
    T = len(ids)
    act_from = max(0, T - act_tail) if act_tail >= 0 else 0
    log_from = max(0, T - max(1, logits_tail)) if logits_tail >= 0 else 0
    rec = Routes(model)
    acts: dict[int, list] = {i: [] for i in range(cfg.num_hidden_layers)}
    hooks = [layer.register_forward_hook(lambda _m, _a, out, i=i: acts[i].append(out.float().clone()))
             for i, layer in enumerate(model.model.layers[: cfg.num_hidden_layers])]
    cache = DynamicCache(config=cfg)
    out = model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True)
    lg = out.logits[0].float()
    nll = -torch.log_softmax(lg[:-1], -1).gather(-1, torch.tensor(ids[1:])[:, None])[:, 0]
    rows = [lg[log_from:]]
    tokens = []
    for _ in range(gen):
        nxt = int(torch.argmax(rows[-1][-1]).item())
        tokens.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]]), past_key_values=out.past_key_values, use_cache=True)
        rows.append(out.logits[0].float())
    for h in hooks:
        h.remove()
    rec.remove()
    t = {"ids": torch.tensor(ids, dtype=torch.int32), "tokens": torch.tensor(tokens, dtype=torch.int32),
         "logits": torch.cat(rows).contiguous(), "nll": nll.contiguous()}
    for i in range(cfg.num_hidden_layers):
        t[f"H.L{i}"] = torch.cat(acts[i], 1)[0][act_from:].contiguous()
    for i in moe_layers(cfg):
        t[f"route.ids.L{i}"], t[f"route.w.L{i}"] = (x.contiguous() for x in rec.get(0, i))
    meta = {"snapshot": os.path.abspath(snapshot), "n_prompt": str(T), "gen": str(gen),
            "layers": str(cfg.num_hidden_layers), "moe_layers": ",".join(map(str, moe_layers(cfg))),
            "act_row0": str(act_from), "logits_row0": str(log_from), "dtype": "fp32", "attn_implementation": "sdpa",
            "experts_implementation": str(getattr(cfg, "_experts_implementation", None)),
            "transformers": PINNED_TRANSFORMERS, "torch": torch.__version__,
            "ppl": f"{math.exp(float(nll.double().mean())) if T > 1 else float('nan'):.6g}", "model": MODEL}
    return t, meta


def cmd_run(a) -> None:
    from safetensors.torch import save_file
    t0 = time.time()
    ids = read_ids(a.prompt)
    if not ids:
        die(f"{a.prompt}: no ids")
    t, meta = run(a.snapshot, ids, a.gen, a.logits_tail, a.act_tail)
    meta["prompt"] = os.path.abspath(a.prompt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(t, a.out, metadata=meta)
    print(f"greedy {a.gen}: {t['tokens'].tolist()}")
    print(f"wrote {a.out}: {len(t)} tensors; prompt ppl {meta['ppl']}; wall {time.time() - t0:.1f}s, "
          f"peak RSS {rss_gib()[1]:.1f} GiB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("snapshot")
    r.add_argument("--prompt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gen", type=int, default=32)
    r.add_argument("--logits-tail", type=int, default=1024, help="keep logits from row T - L (-1: all)")
    r.add_argument("--act-tail", type=int, default=256, help="keep H.L<l> from row T - A (-1: all)")
    t = sub.add_parser("trace")
    t.add_argument("snapshot")
    t.add_argument("--source", action="append", default=[], help="NAME:IDS_FILE[:ANSWER_FROM]")
    t.add_argument("--list", help="lines NAME<TAB>IDS<TAB>ANSWER_FROM, paths relative to the file")
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "trace": cmd_trace}[a.cmd](a)


if __name__ == "__main__":
    main()
