#!/usr/bin/env python3
"""Spec 21a Task 4: the Qwen3.8-Flash-Next MTP head, vLLM's semantics on transformers 5.19.0's modules.

transformers drops `mtp.*` (`_keys_to_ignore_on_load_unexpected`, M:1314/1590), so vLLM v0.31.1rc0-138
(`b52ae2aa7f`, vllm/models/qwen4_exp/nvidia/mtp.py) is the reference for the head's wiring (decision 1):

    e  = fc_embedding(pre_fc_norm_embedding(embed(t_{i+1})))                         mtp.py:293-294
    h  = fc_hidden(pre_fc_norm_hidden(R_i) viewed [hc][H]) per stream, flattened       mtp.py:299-306
         pre_fc_norm_hidden is GemmaRMSNorm(hc * H) (mtp.py:227-229): ONE RMS over all 10240 values ("single",
         decision 4);
         "per_stream" reads the same [10240] weight as 4 groups of 2560 (the alternative decision 4 measures)
    H  = h + e on every stream (unit injection: prev_block_output = e, prev_injection = None, mtp.py:307-317;
         a missing injection is unit weight, hyperconnection.py:58-62 / ops/hc.py:225-228)
    M  = the head's decoder layer (layer type qwen_sparse_attention = indexed_attention, its own KV and
         indexer, 512 own experts + the gated shared expert, both gated residuals, no PLE: mtp.py:211-215)
    logits = lm_head(final mixer(M)) (the head's own `mtp.hyper_connection_mixer`, no inject; lm_head and
         embed shared with the main model: mtp_use_dedicated_embeddings false)
    returns (logits, M, the selection): M - the head's PRE-mixer 4-stream hidden - is the next draft
         step's R (scheme A, mtp.py:330-340).

Positions: draft step 1 from row i runs at rope position i (the proposer keeps the target positions,
llm_base_proposer.py "rotate the input ids and leave the positions unchanged"), step d at i + d - 1.
Selection on later steps (decision 5, ruled "reuse"): vLLM reuses step 0's QSA token list on steps >= 1
(`skip_topk`, mtp.py:258-261, indexer_qsa.py:260-261 / 409-412) ONLY when `index_share_for_mtp_iteration`
is set (llm_base_proposer.py:578-613, 1592-1603)
- a speculative-config option whose default is the draft HF config's value, which Qwen3.8-Flash-Next's
config.json does not carry (config/speculative.py:442-444, 845-864) - so vLLM's DEFAULT is a fresh
selection per step. `chain(..., share_sel=True)` is the ruled form; share_sel=False is vLLM's default.
A reused list is the step-0 row's positions verbatim: a later step's query attends neither its own key
nor the earlier draft keys (they are written to the cache, not in the list).

The layer is transformers' own `Qwen4ExpTextDecoderLayer` with qwen4exp_ref's cached indexer; the
weights are read from the checkpoint's `mtp.*` (Intel's per-expert bf16 `.weight` experts, or the
original's fused ones, lazily); `mtp.hyper_connection_mixer.block_inject_weight` is dropped if present
(vLLM's mapper, mtp.py:351 / 438; neither checkpoint ships one).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ref = _load("qwen4exp_ref")
P = "mtp."
NORMS = ("single", "per_stream")


def head_config(tc):
    """The head's layer config: the main text config with one indexed_attention layer and no PLE."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    d = tc.to_dict()
    d.update(num_hidden_layers=1, layer_types=["indexed_attention"], ple_layer_ids=[])
    for k in ("_attn_implementation", "_experts_implementation", "transformers_version", "dtype"):
        d.pop(k, None)
    hc = Qwen4ExpTextConfig(**d)
    hc._attn_implementation = "eager"
    hc._experts_implementation = tc._experts_implementation
    hc.dtype = tc.dtype
    return hc


class _DictCkpt:
    """ref.Ckpt's interface over an in-memory dict (tests)."""

    def __init__(self, d):
        self.d = d

    def __contains__(self, k):
        return k in self.d

    def keys(self):
        return self.d.keys()

    def get(self, k):
        return self.d[k]

    def get_rows(self, k, a, b):
        return self.d[k][a:b]


class MtpHead:
    def __init__(self, src, tc, norm: str = "single", model=None, embed=None, lm_head=None):
        """src: a snapshot dir or a {name: tensor} dict holding `mtp.*`; embed / lm_head: the main model's
        (taken from `model` - build_streamed's - when given, else read from src)."""
        if norm not in NORMS:
            raise ValueError(f"norm {norm!r}: one of {NORMS}")
        m = ref.mq()
        self.tc, self.norm_form = tc, norm
        self.hc = head_config(tc)
        H, hcn, eps, dt = tc.hidden_size, tc.hc_count, tc.rms_norm_eps, tc.dtype
        self.H, self.hcn = H, hcn
        ck = ref.Ckpt(src) if isinstance(src, str) else _DictCkpt(src)
        names = [k for k in ck.keys() if k.startswith(P)]
        if not names:
            raise ValueError("no mtp.* tensors in the source")
        with torch.device("meta"):
            self.layer = m.Qwen4ExpTextDecoderLayer(self.hc, 0)
            self.mixer = m.Qwen4ExpTextGatedResidual(self.hc, use_combine=False)
            self.pre_e = m.Qwen4ExpTextRMSNorm(H, eps=eps)
            self.pre_h = m.Qwen4ExpTextRMSNorm(hcn * H, group_size=None if norm == "single" else H, eps=eps)
            self.fc_e = torch.nn.Linear(H, H, bias=False)
            self.fc_h = torch.nn.Linear(H, H, bias=False)
        LP = P + "layers.0."
        lsd = {}
        for k in names:
            if k.startswith(LP) and ".mlp.experts." not in k:
                lsd[k[len(LP):]] = ref._cast(ck.get(k), dt)
        self.lazy = ref.LazyExperts(ck, self.hc, P + "layers.", keep={0}, dtype=dt)
        lsd["mlp.experts.gate_up_proj"], lsd["mlp.experts.down_proj"] = self.lazy.tensors(0)
        self.layer.load_state_dict(lsd, strict=True, assign=True)
        self.layer.mlp.experts.register_forward_pre_hook(self.lazy.hook(0))

        def own(mod, prefix):
            sd = {k[len(prefix):]: ref._cast(ck.get(k), dt) for k in names
                  if k.startswith(prefix) and not k.endswith("block_inject_weight.weight")}
            mod.load_state_dict(sd, strict=True, assign=True)
            return mod.eval()
        own(self.mixer, P + "hyper_connection_mixer.")
        self.pre_e.load_state_dict({"weight": ref._cast(ck.get(P + "pre_fc_norm_embedding.weight"), dt)}, assign=True)
        self.pre_h.load_state_dict({"weight": ref._cast(ck.get(P + "pre_fc_norm_hidden.weight"), dt)}, assign=True)
        self.fc_e.load_state_dict({"weight": ref._cast(ck.get(P + "fc_embedding.weight"), dt)}, assign=True)
        self.fc_h.load_state_dict({"weight": ref._cast(ck.get(P + "fc_hidden.weight"), dt)}, assign=True)
        if model is not None:
            embed, lm_head = model.model.embed_tokens.weight, model.lm_head.weight
        if embed is None:
            pre = ref.lm_prefix([k for k in ck.keys() if not k.startswith(P)])
            embed = ref._cast(ck.get(pre + "embed_tokens.weight"), dt)
            lm_head = ref._cast(ck.get("lm_head.weight"), dt)
        self.embed, self.lm_head = embed, lm_head
        self.rotary = m.Qwen4ExpTextRotaryEmbedding(self.hc)
        ref.patch_indexers(self.layer)
        self.indexer = self.layer.self_attn.indexer
        self.layer.eval()

    def trace_routes(self, R, ids, chunk: int = 2048) -> dict:
        return trace_routes(self, R, ids, chunk)

    def new_cache(self):
        return ref.mq().DynamicCache(config=self.hc)

    @torch.no_grad()
    def fuse(self, R: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        """The head layer's input H [T][hc*H]: fc_hidden per stream of the normed R, plus e on every stream."""
        T = R.shape[0]
        e = self.fc_e(self.pre_e(F.embedding(tok, self.embed)))
        h = self.fc_h(self.pre_h(R).view(T, self.hcn, self.H)).flatten(-2)
        return h + e.repeat(1, self.hcn)          # unit injection: the product by 1 is exact, then the add

    @torch.no_grad()
    def step(self, R: torch.Tensor, tok: torch.Tensor, pos: int, cache, sel=None):
        """Rows R [T][hc*H] with next tokens tok [T] at positions pos..pos+T-1 (= the cache length) ->
        (logits [T][V] in the model dtype, M [T][hc*H] the head's pre-mixer hidden, the selection [T][W]).
        sel: a forced token list per row (skip_topk) instead of the head's own selection."""
        m = ref.mq()
        T = R.shape[0]
        past = cache.get_seq_length()
        if past != pos:
            raise RuntimeError(f"position {pos} fed to a head cache of {past} entries")
        x = self.fuse(R, tok)[None]
        pid = torch.arange(pos, pos + T)
        mask = m.create_causal_mask(config=self.hc, inputs_embeds=x, attention_mask=None, past_key_values=cache,
                                    position_ids=pid[None], allow_is_causal_skip=False)
        full = torch.arange(0, pos + T).view(1, 1, -1).expand(3, 1, -1)
        pe = self.rotary(x, full)
        self.indexer._q4_force = sel
        try:
            M = self.layer(x, position_embeddings=pe, attention_mask=mask, past_key_values=cache)[0]
        finally:
            self.indexer._q4_force = None
        logits = F.linear(self.mixer(M[None])[0], self.lm_head)
        return logits, M, self.indexer._q4_last


def truncate_cache(head: MtpHead, cache, n: int):
    """A new head cache holding the first n positions of `cache` (views; later appends copy)."""
    out = head.new_cache()
    src, dst = cache.layers[0], out.layers[0]
    dst.lazy_initialization(src.keys[:, :, :n], src.values[:, :, :n])
    dst.keys, dst.values = src.keys[:, :, :n], src.values[:, :, :n]
    dst.lazy_initialization_indexer(src.indexer_keys[:, :n])
    dst.indexer_keys = src.indexer_keys[:, :n]
    blocks = getattr(src, "_q4_blocks", None)
    if blocks is not None:
        dst._q4_blocks = blocks[: n // head.indexer.compress_ratio]
    return out


@torch.no_grad()
def chain(R: torch.Tensor, head: MtpHead, ids, k: int, share_sel: bool = True, rows=None) -> dict:
    """Draft steps 1..k from every row i < T - 1 (or `rows`): step 1 on (R_i, t_{i+1}) at position i,
    teacher-forced for all rows in ONE call (the proposer's step 0 over the accepted tokens); step d >= 2
    on (M_{d-1}, argmax of step d - 1) at position i + d - 1, over the head cache cut to positions 0..i
    plus this chain's own earlier steps; with share_sel the step-1 token list is reused (decision 5).
    Returns {"logits": [k] x [rows][V] f32, "tokens": [rows][k], "M": [rows][hc*H] of the last step,
    "sel": [rows][W] step 1's lists, "rows": the row indices}."""
    ids = torch.as_tensor(ids)
    T = ids.numel()
    cache = head.new_cache()
    lg1, M1, sel1 = head.step(R[: T - 1], ids[1:T], 0, cache)
    rows = list(range(T - 1)) if rows is None else list(rows)
    out_lg = [[lg1[i].float() for i in rows]]
    toks = [[int(lg1[i].float().argmax())] for i in rows]
    lastM = [M1[i] for i in rows]
    for d in range(2, k + 1):
        out_lg.append([])
    for j, i in enumerate(rows):
        if k < 2:
            break
        c = truncate_cache(head, cache, i + 1)
        Mp, tok = M1[i:i + 1], torch.tensor([toks[j][-1]])
        for d in range(2, k + 1):
            lg, Mp, _ = head.step(Mp, tok, i + d - 1, c, sel=sel1[i:i + 1] if share_sel else None)
            tok = torch.tensor([int(lg[0].float().argmax())])
            toks[j].append(int(tok))
            out_lg[d - 1].append(lg[0].float())
        lastM[j] = Mp[0]
    return {"logits": [torch.stack(x) for x in out_lg], "tokens": torch.tensor(toks), "M": torch.stack(lastM),
            "sel": sel1[rows], "rows": rows}


@torch.no_grad()
def accept(R: torch.Tensor, head: MtpHead, ctx_ids, cont_ids, k_max: int = 3, share_sel: bool = True) -> dict:
    """Teacher-forced greedy acceptance by depth (mtp_accept.py's shape): the sequence ctx + cont (cont =
    the main model's greedy continuation, so cont[j] is what the verifier would emit); R = the main model's
    pre-mixer hidden over that sequence. From each row i in the continuation region the chain drafts
    t_{i+2}..t_{i+1+k}; depth d is accepted when steps 1..d all equal the sequence. Returns per-depth
    acceptance rates, the mean accepted length and the row count."""
    seq = list(ctx_ids) + list(cont_ids)
    n0 = len(ctx_ids) - 1
    rows = [i for i in range(n0, len(seq) - 2)]
    res = chain(R, head, seq, k_max, share_sel, rows=rows)
    acc = [0] * k_max
    total = 0
    for j, i in enumerate(rows):
        n = 0
        for d in range(k_max):
            tgt = i + 2 + d
            if tgt >= len(seq) or res["tokens"][j][d] != seq[tgt]:
                break
            n += 1
        for d in range(n):
            acc[d] += 1
        total += n
    R_ = max(len(rows), 1)
    return {"rows": len(rows), "accept_by_depth": [a / R_ for a in acc], "mean_accepted": total / R_,
            "k_max": k_max, "share_sel": share_sel, "norm": head.norm_form}


@torch.no_grad()
def trace_routes(head: MtpHead, R: torch.Tensor, ids, chunk: int = 2048) -> dict:
    """The head's step-1 routes on every row (teacher-forced: (R_t, t_{t+1}) at position t, in chunks):
    mtp_ids i32 [T][k] ascending, mtp_p f32 (the router's pre-cast renormalised fp32 values), mtp_onorm
    (the routed experts' output norms); row T-1 has no next token: -1 / 0."""
    ids = torch.as_tensor(ids)
    T, k = ids.numel(), head.tc.num_experts_per_tok
    cap = {"ids": [], "p": []}

    def router(_m, _a, out):
        logits, _, idx = out
        full = F.softmax(logits.detach(), dtype=torch.float, dim=-1)
        top = full.gather(-1, idx)
        if head.tc.norm_topk_prob:
            top /= top.sum(dim=-1, keepdim=True)
        cap["ids"].append(idx.to(torch.int32))
        cap["p"].append(top)
    h = head.layer.mlp.gate.register_forward_hook(router)
    tap = ref.OnormTap([head.layer])
    cache = head.new_cache()
    try:
        c = chunk if chunk > 0 else T
        for a in range(0, T - 1, c):
            b = min(a + c, T - 1)
            head.step(R[a:b], ids[a + 1:b + 1], a, cache)
    finally:
        h.remove()
        tap.close()
    idx = torch.cat(cap["ids"]) if cap["ids"] else torch.empty(0, k, dtype=torch.int32)
    p = torch.cat(cap["p"]) if cap["p"] else torch.empty(0, k)
    on = torch.cat(tap.rows.get(0, {}).get(0, [])) if T > 1 else torch.empty(0, k)
    order = idx.argsort(-1)
    pad_i = torch.full((1, k), -1, dtype=torch.int32)
    pad_f = torch.zeros(1, k)
    return {"mtp_ids": torch.cat([idx.gather(-1, order), pad_i]).contiguous(),
            "mtp_p": torch.cat([p.gather(-1, order), pad_f]).contiguous(),
            "mtp_onorm": torch.cat([on.gather(-1, order), pad_f]).contiguous()}


# ------------------------------------------------------------------------------------------------
# Spec 21e: the engine-format head and the acceptance run (F5's numbers until spec 22)

def rtn_dequant(w: torch.Tensor) -> torch.Tensor:
    """loader::rtn_int4_g64 then its dequantisation: bf16 [N][K] -> bf16 [N][K]. Per output row and group of 64:
    s16 = f16(2 amax / 15) (round to nearest even), q = clamp(rint(w / f32(s16)) + 8, 0, 15) (an all-zero group:
    s16 = 0, q = 8), the value (q - 8) x f32(s16) rounded to bf16 (ornith_ref.rtn_int4_g64's formula)."""
    N, Kd = w.shape
    if Kd % 64:
        raise ValueError(f"RTN g64 over K = {Kd}")
    wf = w.float().reshape(N, Kd // 64, 64)
    amax = wf.abs().amax(-1)
    s = (2.0 * amax / 15.0).to(torch.float16).float().unsqueeze(-1)
    q = torch.where(s > 0, torch.round(wf / torch.where(s > 0, s, torch.ones_like(s))) + 8.0,
                    torch.full_like(wf, 8.0)).clamp(0.0, 15.0)
    return ((q - 8.0) * s).reshape(N, Kd).to(w.dtype)


def engine_format(head: MtpHead) -> MtpHead:
    """The head's routed experts as the engine loads them (21b: the checkpoints ship the head's experts bf16, the
    loader RTN-quantises them to int4 g64 - spec 15e's rule): every expert quantised and dequantised. Gate and up
    rows quantise independently (the RTN is per output row), so the fused [2I][H] form is quantised as one."""
    lazy = head.lazy
    orig = lazy.expert

    def expert(layer, e):
        gu, dn = orig(layer, e)
        return rtn_dequant(gu), rtn_dequant(dn)

    lazy.expert = expert
    return head


@torch.no_grad()
def greedy_continuation(model, ids, n: int, chunk: int = 2048) -> list:
    """The main model's greedy continuation of `ids` (n ids), the reference's cached decode (qwen4exp_ref's run)."""
    last = {}
    cache = ref.forward_chunks(model, list(ids), chunk if len(ids) > 4096 else 0,
                               on_logits=lambda _r0, lg: last.update(row=lg[-1].float()), logits_to_keep=1)
    out = []
    row = last["row"]
    for _ in range(n):
        nxt = int(torch.argmax(row).item())
        out.append(nxt)
        o = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        row = o.logits[0, -1].float()
    return out


def cmd_accept(argv) -> None:
    """qwen4exp_mtp.py accept <checkpoint> --out accept_<norm>.json [--norm single|per_stream] [--select reuse|fresh|
    both] [--source NAME:CTX_IDS[:CONT_IDS] ...] [--golden DIR] [--k 3] [--ctx-max 2048] [--gen 64] [--layers N]
    [--ple SPEC] [--experts rtn|bf16]

    Teacher-forced greedy acceptance by depth (accept()) per source and pooled (rows-weighted): accept_by_depth[d] =
    the share of continuation rows whose drafts 1..d+1 all equal the main model's greedy text, E_K = 1 + sum_{d<K}
    accept_by_depth[d] the expected tokens an iteration at K drafts. A source's continuation is CONT_IDS, or the
    golden set's `tokens` (--golden: q4exp_short, q4exp_agentic from <DIR>/<p>.golden.safetensors), or the
    reference's own greedy continuation of --gen ids. The head is the engine's (--experts rtn, the default) or the
    checkpoint's bf16 experts. Box CPU (r34.accept), hours on Intel's checkpoint."""
    import argparse
    import json
    import time
    ap = argparse.ArgumentParser(prog="qwen4exp_mtp.py accept")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--norm", choices=NORMS, default="single")
    ap.add_argument("--select", choices=("reuse", "fresh", "both"), default="both")
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--golden", default="")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ctx-max", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--ple", default="")
    ap.add_argument("--experts", choices=("rtn", "bf16"), default="rtn")
    a = ap.parse_args(argv)
    t0 = time.time()
    sources = []
    for s in a.source:
        parts = s.split(":")
        if len(parts) not in (2, 3):
            raise SystemExit(f"--source {s}: NAME:CTX_IDS[:CONT_IDS]")
        sources.append((parts[0], ref.read_ids(parts[1]), ref.read_ids(parts[2]) if len(parts) == 3 else None))
    if a.golden:
        from safetensors import safe_open
        prompts = os.path.join(_HERE, "..", "..", "tests", "golden", "prompts")
        for p in ("q4exp_short", "q4exp_agentic"):
            g = os.path.join(a.golden, f"{p}.golden.safetensors")
            if not os.path.exists(g):
                print(f"golden {p}: no {g} - skipped", flush=True)
                continue
            with safe_open(g, "pt") as f:
                cont = [int(x) for x in f.get_tensor("tokens").tolist()]
            sources.append((p, ref.read_ids(os.path.join(prompts, f"{p}.ids")), cont))
    if not sources:
        raise SystemExit("no source: --source NAME:CTX[:CONT] or --golden DIR")
    tc = ref.text_config(a.checkpoint, a.layers)
    ple_spec = a.ple or (f"int8:{a.checkpoint.rstrip('/')}-ple-int8"
                         if os.path.isdir(a.checkpoint.rstrip("/") + "-ple-int8") else "bf16")
    model, pf, _ = ref.build_streamed(a.checkpoint, tc, ple=ref.PleTable(ref.ple_source(a.checkpoint, ple_spec), tc))
    head = MtpHead(a.checkpoint, tc, norm=a.norm, model=model)
    if a.experts == "rtn":
        engine_format(head)
    step = ref.layer_major(model, pf)
    selects = ("reuse", "fresh") if a.select == "both" else (a.select,)
    out = {"checkpoint": os.path.abspath(a.checkpoint), "norm": a.norm, "experts": a.experts, "k_max": a.k,
           "layers": tc.num_hidden_layers, "sources": {}, "pooled": {}}
    for name, ctx, cont in sources:
        ctx = ctx[-a.ctx_max:]
        if cont is None:
            cont = greedy_continuation(model, ctx, a.gen)
        R = step([(list(ctx) + list(cont), 0, {})])[0]
        out["sources"][name] = {}
        for sel in selects:
            r = accept(R, head, ctx, cont, a.k, share_sel=sel == "reuse")
            out["sources"][name][sel] = r
            print(f"{name} ({len(ctx)} + {len(cont)} ids) {a.norm} {sel}: rows {r['rows']}, accept by depth "
                  f"{[round(x, 4) for x in r['accept_by_depth']]}, mean accepted {r['mean_accepted']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    for sel in selects:
        rows = sum(v[sel]["rows"] for v in out["sources"].values())
        acc = [sum(v[sel]["accept_by_depth"][d] * v[sel]["rows"] for v in out["sources"].values()) / max(rows, 1)
               for d in range(a.k)]
        e = [1.0 + sum(acc[:K]) for K in range(1, a.k + 1)]
        out["pooled"][sel] = {"rows": rows, "accept_by_depth": acc, "expected_tokens_at_K": e}
        print(f"pooled {a.norm} {sel}: rows {rows}, accept by depth {[round(x, 4) for x in acc]}, E_K (K = 1..{a.k}) "
              f"{[round(x, 3) for x in e]}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "accept":
        cmd_accept(sys.argv[2:])
    else:
        raise SystemExit("usage: qwen4exp_mtp.py accept <checkpoint> --out FILE ... (python3 qwen4exp_mtp.py accept -h)")

