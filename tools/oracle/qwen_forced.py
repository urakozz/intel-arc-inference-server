#!/usr/bin/env python3
"""Qwen3.8's f32 judges of the forced contexts (spec 28 N4 / N6, tests/server/spec_judge.h), on the CPU.

    qwen_forced.py run <snapshot> --ctx DIR [--ctx DIR ...] --out DIR --out-i8head DIR
                   [--i8head-re REGEX] [--match REGEX] [--done FILE] [--work DIR] [--dry-run]
    qwen_forced.py plan --ctx DIR [--ctx DIR ...] [--match REGEX] [--done FILE]

A context is tests/golden/forced_judge.h's dump: <key>.prompt.ids and <key>.force.ids (B70_DUMP_FORCED). Its judge
is <out>/<key>.golden.safetensors in ornith_ref.py --lean's format: `logits` f32 [1 + gen][V] (row 0 = the prompt's
last position, row j = the position that consumed forced id j - 1: row j decides forced id j) and `tokens` int32
[gen], the forced ids. The model is Qwen3.8 in float32 throughout: the int4 checkpoint dequantised as dump.py
dequantises it (int4 x f16 scale, one bf16 rounding - every oracle's weights), every weight upcast to f32 and every
activation, GEMM, attention, GDN recurrence and norm in f32; the head is the checkpoint's bf16 lm_head upcast, or -
for the keys --i8head-re matches (the tests that load LmHeadForm::Int8) - the engine's per-row int8 head
(int8_sim.int8_head_weight, as ornith_ref.py --lm-head int8), written to --out-i8head.

Why not one teacher-forced run per context (f32_ref_job.sh / ornith_ref.py): the spec tests' ~250 contexts are
~450k ids, but they are a few prompts each continued a few ways - a token TRIE of all of them holds ~48k distinct
positions. So the contexts are merged into one trie and ONE layer-major forward computes every distinct position
once: per decoder layer (streamed, stream.py via dump.stream_weights), the token-parallel parts (norms, every linear,
the MLP) over all positions at once, the sequence mixing per trie segment (a maximal chain with no branch) from its
parent's state - full attention over the segment's ancestors' K / V, the GDN causal conv over the four positions up
the path, the gated delta rule (modeling_qwen3_5.torch_chunk_gated_delta_rule) from the parent segment's final
recurrent state. Mathematically that is each context's own causal forward; the f32 rounding differs from a
one-sequence forward only in summation order (test_qwen_forced.py checks it against transformers' own forward).

Resumable: the hidden state is checkpointed after every layer in --work (a killed run restarts at the layer it was
on, for the same set of contexts); keys listed in --done (one per line) or already in --out are skipped. Contexts
with the same ids and head share one file (a hard link). Run in the oracle container (run_in_container.sh) under
/tmp/b70-cpu-model.lock - tools/box_validate/f32_refs.sh forced-qwen does both.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

# The tests that load the engine's int8 head (loader::LmHeadForm::Int8): every DFlash test, and mtp_gpu_test with
# --lm-head int8 (its tags carry the head form: mtp_gpu_<form>_, mtp_gpu_rf1_<form>_). mtp_verify_test's
# qwen3.8_int8 is the KV cache, not the head.
I8HEAD_RE = r"^(dflash_|mtp_gpu_(rf1_)?int8_)"
ROW_BLOCK = 4096          # positions per token-parallel GEMM block
Q_BLOCK = 1024            # attention query rows per block
HEAD_ROWS = 2048          # logits rows per head block (2048 x 248320 x 4 B = 2 GB)


def _load(name):
    mod_name = "qwen_forced_" + name
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


def fnv_key_hash(prompt, force) -> str:
    """forced_judge.h ForcedJudge::key's hash: FNV-1a 64 over the ids' little-endian bytes, 0xFFFFFFFF between."""
    h = 1469598103934665603
    for v in list(prompt) + [0xFFFFFFFF] + list(force):
        for b in range(4):
            h = ((h ^ ((v >> (8 * b)) & 0xFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def load_contexts(dirs, match=None, done=()):
    """{key: (prompt, force)} over every <key>.prompt.ids in `dirs` (one key in two dirs must be one context), the
    key's hash checked against its ids; `match` filters keys (re.search), `done` drops them."""
    ctx = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.prompt.ids"))):
            k = os.path.basename(p)[: -len(".prompt.ids")]
            if (match and not re.search(match, k)) or k in done:
                continue
            pr, fo = read_ids(p), read_ids(p[: -len(".prompt.ids")] + ".force.ids")
            if not pr:
                raise SystemExit(f"{p}: an empty prompt")
            if k[-16:] != fnv_key_hash(pr, fo):
                raise SystemExit(f"{k}: the ids do not hash to the key's suffix (got {fnv_key_hash(pr, fo)})")
            if k in ctx and ctx[k] != (pr, fo):
                raise SystemExit(f"{k}: two different contexts under one key")
            ctx[k] = (pr, fo)
    return ctx


class Trie:
    """Every context's prompt + forced ids as one token trie, numbered so that each SEGMENT (a maximal chain whose
    inner nodes have one child) is a contiguous index range and a parent segment precedes its children.

    tok / depth / parent: per node (parent -1 at a root); seg_start / seg_len / seg_parent (segment index, -1);
    seg_anc[s]: the ancestor index ranges of segment s (its parent's, then the parent segment itself)."""

    def __init__(self, seqs):
        tok, par, kids, roots = [], [], [], {}
        for s in seqs:
            node = -1
            for t in s:
                d = roots if node < 0 else kids[node]
                c = d.get(t)
                if c is None:
                    c = len(tok)
                    tok.append(t)
                    par.append(node)
                    kids.append({})
                    d[t] = c
                node = c
        order, self.seg_start, self.seg_len, self.seg_parent = [], [], [], []
        stack = [(c, -1) for c in reversed(list(roots.values()))]
        while stack:
            n, ps = stack.pop()
            si = len(self.seg_start)
            self.seg_start.append(len(order))
            self.seg_parent.append(ps)
            while True:
                order.append(n)
                ch = list(kids[n].values())
                if len(ch) != 1:
                    break
                n = ch[0]
            self.seg_len.append(len(order) - self.seg_start[si])
            for c in reversed(ch):
                stack.append((c, si))
        new = [0] * len(order)
        for i, o in enumerate(order):
            new[o] = i
        self.old2new = new
        self.tok = [tok[o] for o in order]
        self.parent = [-1 if par[o] < 0 else new[par[o]] for o in order]
        self.depth = [0] * len(order)
        for i, p in enumerate(self.parent):   # a parent precedes its child
            self.depth[i] = 0 if p < 0 else self.depth[p] + 1
        self.seg_anc = []
        for s, ps in enumerate(self.seg_parent):
            self.seg_anc.append([] if ps < 0 else self.seg_anc[ps] + [(self.seg_start[ps],
                                                                        self.seg_start[ps] + self.seg_len[ps])])
        self.n_children = [0] * len(self.seg_start)
        for ps in self.seg_parent:
            if ps >= 0:
                self.n_children[ps] += 1
        self._roots, self._kids = roots, kids

    def path(self, seq):
        """The node of every position of `seq` (new numbering)."""
        out, node = [], -1
        for t in seq:
            node = (self._roots if node < 0 else self._kids[node])[t]
            out.append(node)
        return [self.old2new[n] for n in out]

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps([self.tok, self.parent]).encode())
        return h.hexdigest()[:16]


def plan(ctx, i8head_re=I8HEAD_RE):
    """The work: the trie, and per head form the contexts and their distinct (ids, head) outputs."""
    keys = sorted(ctx)
    seqs = [ctx[k][0] + ctx[k][1] for k in keys]
    trie = Trie(seqs)
    jobs = {}
    for k, s in zip(keys, seqs):
        form = "int8" if re.search(i8head_re, k) else "bf16"
        pr, fo = ctx[k]
        rows = trie.path(s)[len(pr) - 1:]          # 1 + gen rows: the prompt's last position, then each forced id's
        assert len(rows) == len(fo) + 1
        jobs[k] = {"form": form, "rows": rows, "force": fo, "n_prompt": len(pr), "seq": tuple(s)}
    return trie, jobs


def describe(trie, jobs, vocab=248320):
    raw = sum(len(j["seq"]) for j in jobs.values())
    uniq = {(j["seq"], j["form"]) for j in jobs.values()}
    rows = {}
    for j in jobs.values():
        rows.setdefault((j["seq"], j["form"]), len(j["rows"]))
    nrow = sum(rows.values())
    forms = {f: sum(1 for j in jobs.values() if j["form"] == f) for f in ("bf16", "int8")}
    return (f"{len(jobs)} contexts ({forms['bf16']} bf16 head, {forms['int8']} int8 head; {len(uniq)} distinct "
            f"ids + head), {raw} ids -> {len(trie.tok)} trie positions in {len(trie.seg_start)} segments; "
            f"{nrow} logits rows written ({nrow * vocab * 4 / 1e9:.1f} GB)"), len(trie.tok), nrow


def estimate_hours(n_pos, n_rows, longest):
    """Rough, from the box's f32 sgemm (2048x2560x2560 in 37.8 ms on ~19 threads, memory cpu-bf16-slow): ~54 GFLOP a
    position over the linears, the head 2.5 GFLOP a written row, attention over a segment's ancestors, the weight
    stream (~6.5 min); at a third of that rate for the GEMMs' shapes and the python-side segment loops."""
    gflops = 700.0 / 3
    lin = n_pos * 54.0 + n_rows * 2.5
    attn = 16 * 24 * 256 * 4 * n_pos * longest / 2 / 1e9
    return (lin + attn) / gflops / 3600 + 390 / 3600


# ------------------------------------------------------------------------------------------------ the forward


def build(snapshot):
    """Qwen3_5ForCausalLM on meta, layer-streamed (dump.stream_weights: the layers dequantised per forward in a
    prefetch thread; embed / final norm / lm_head resident in bf16). Returns (model, text config, prefetcher)."""
    import torch
    D = _load("dump")
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    if D._agnes.is_agnes(raw):
        raise SystemExit("qwen_forced.py is for Qwen3.8")
    quantised = bool(raw.get("quantization_config") or (raw.get("text_config") or {}).get("quantization_config"))
    group_size = D.check_quant_config(snapshot) if quantised else 0
    tc = AutoConfig.from_pretrained(snapshot).get_text_config()
    tc._attn_implementation = "eager"
    with torch.device("meta"):
        model = m.Qwen3_5ForCausalLM(tc)
    pf = D.stream_weights(model, snapshot, group_size, tc, 0)
    model.eval()
    return model, tc, pf


def _rows(f, x, block=ROW_BLOCK):
    import torch
    return torch.cat([f(x[a:a + block]) for a in range(0, x.shape[0], block)], 0)


def _anc_index(trie, s):
    import torch
    parts = [torch.arange(a, b) for a, b in trie.seg_anc[s]]
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long)


def attention(A, h, trie, cos, sin, M):
    """Qwen3_5Attention over the trie: q / k / v / gate token-parallel, softmax(q k^T * scale) v per segment over
    its ancestors' and its own (causal) positions, f32."""
    import torch
    N, hd = h.shape[0], A.head_dim
    qg = _rows(A.q_proj, h).view(N, -1, 2 * hd)
    q, gate = torch.chunk(qg, 2, dim=-1)
    gate = gate.reshape(N, -1)
    q = A.q_norm(q.contiguous())
    k = A.k_norm(_rows(A.k_proj, h).view(N, -1, hd))
    v = _rows(A.v_proj, h).view(N, -1, hd)
    q, k = M.apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin)
    q, k, v = q[0], k[0], v.transpose(0, 1).contiguous()        # [heads, N, hd]
    nH, nKV = q.shape[0], k.shape[0]
    out = torch.empty(nH, N, hd, dtype=q.dtype)
    for s in range(len(trie.seg_start)):
        a, L = trie.seg_start[s], trie.seg_len[s]
        anc = _anc_index(trie, s)
        P = anc.numel()
        idx = torch.cat([anc, torch.arange(a, a + L)])
        K = k[:, idx].repeat_interleave(nH // nKV, dim=0)        # repeat_kv's order: kv head j serves q heads j*g..
        Vv = v[:, idx].repeat_interleave(nH // nKV, dim=0)
        for r0 in range(0, L, Q_BLOCK):
            r1 = min(L, r0 + Q_BLOCK)
            sc = torch.matmul(q[:, a + r0:a + r1], K[:, :P + r1].transpose(1, 2)) * A.scaling
            own = torch.ones(r1 - r0, r1, dtype=torch.bool).triu(r0 + 1)   # own key j > own query r: masked
            sc[:, :, P:].masked_fill_(own, float("-inf"))
            p = torch.softmax(sc, dim=-1, dtype=torch.float32)
            out[:, a + r0:a + r1] = torch.matmul(p, Vv[:, :P + r1])
    o = out.transpose(0, 1).reshape(N, -1) * torch.sigmoid(gate)
    return _rows(A.o_proj, o)


def gdn(G, h, trie, M):
    """Qwen3_5GatedDeltaNet over the trie: the projections token-parallel; the causal conv (kernel K) as the sum of
    K taps up the path (a position's own and its K - 1 ancestors', zero above a root); the gated delta rule per
    segment from its parent segment's final state (modeling_qwen3_5.torch_chunk_gated_delta_rule, fp32)."""
    import torch
    import torch.nn.functional as F
    N = h.shape[0]
    qkv = _rows(G.in_proj_qkv, h)                               # [N, conv_dim]
    z = _rows(G.in_proj_z, h)
    b = _rows(G.in_proj_b, h)
    a_ = _rows(G.in_proj_a, h)
    w = G.conv1d.weight.squeeze(1)                              # [conv_dim, K]
    Kc = w.shape[1]
    par = torch.tensor(trie.parent + [-1])                      # index N: "above a root" (sticky)
    par[par < 0] = N
    conv = qkv * w[:, Kc - 1]
    anc = torch.arange(N)
    xz = torch.cat([qkv, torch.zeros(1, qkv.shape[1], dtype=qkv.dtype)], 0)
    for j in range(1, Kc):
        anc = par[anc]
        conv = conv + xz[anc] * w[:, Kc - 1 - j]
    if G.conv1d.bias is not None:
        conv = conv + G.conv1d.bias
    mixed = F.silu(conv) if G.activation == "silu" else M.ACT2FN[G.activation](conv)
    query, key, value = torch.split(mixed, [G.key_dim, G.key_dim, G.value_dim], dim=-1)
    query = query.reshape(N, -1, G.head_k_dim)
    key = key.reshape(N, -1, G.head_k_dim)
    value = value.reshape(N, -1, G.head_v_dim)
    beta = b.sigmoid()
    g = -G.A_log.float().exp() * F.softplus(a_.float() + G.dt_bias)
    if G.num_v_heads // G.num_k_heads > 1:
        query = query.repeat_interleave(G.num_v_heads // G.num_k_heads, dim=1)
        key = key.repeat_interleave(G.num_v_heads // G.num_k_heads, dim=1)
    core = torch.empty(N, G.num_v_heads, G.head_v_dim, dtype=value.dtype)
    states, left = {}, list(trie.n_children)
    for s in range(len(trie.seg_start)):
        a, L, ps = trie.seg_start[s], trie.seg_len[s], trie.seg_parent[s]
        init = states[ps] if ps >= 0 else None
        o, fin = M.torch_chunk_gated_delta_rule(query[None, a:a + L], key[None, a:a + L], value[None, a:a + L],
                                                g=g[None, a:a + L], beta=beta[None, a:a + L], initial_state=init,
                                                output_final_state=True, use_qk_l2norm_in_kernel=True)
        core[a:a + L] = o[0]
        if left[s]:
            states[s] = fin
        if ps >= 0:
            left[ps] -= 1
            if not left[ps]:
                del states[ps]
    core = G.norm(core.reshape(-1, G.head_v_dim), z.reshape(-1, G.head_v_dim)).reshape(N, -1)
    return _rows(G.out_proj, core)


def layer_forward(lay, X, trie, cos, sin, M):
    h = _rows(lay.input_layernorm, X)
    if lay.block_type == "linear_attention":
        X = X + gdn(lay.linear_attn, h, trie, M)
    else:
        X = X + attention(lay.self_attn, h, trie, cos, sin, M)
    del h
    return X + _rows(lambda x: lay.mlp(lay.post_attention_layernorm(x)), X)


def trie_forward(model, trie, load_layer=None, unload_layer=None, work=None, log=print):
    """The final hidden state (after the last norm) of every trie position, f32 [N, H]. `load_layer(i)` makes layer
    i's weights present (streamed runs), `unload_layer(i)` drops them; `work`: a directory for the per-layer
    checkpoint (resume)."""
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    tok = torch.tensor(trie.tok)
    pos = torch.tensor(trie.depth)
    fp = trie.fingerprint()
    start, X = 0, None
    ck = os.path.join(work, "hidden.pt") if work else None
    if ck and os.path.exists(ck):
        c = torch.load(ck, weights_only=True)
        if c.get("fingerprint") == fp:
            start, X = int(c["next_layer"]), c["X"]
            log(f"resumed at layer {start} from {ck}")
    if X is None:
        X = _rows(lambda t: model.model.embed_tokens(t).float(), tok)
    cos, sin = model.model.rotary_emb(X[:1], pos[None])            # [1, N, rot dim] f32
    layers = model.model.layers
    for i in range(start, len(layers)):
        t = time.time()
        if load_layer:
            load_layer(i)
        lay = layers[i].float()
        with torch.no_grad():
            X = layer_forward(lay, X, trie, cos, sin, M)
        if unload_layer:
            unload_layer(i)
        if ck:
            torch.save({"fingerprint": fp, "next_layer": i + 1, "X": X}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
        log(f"  layer {i} ({layers[i].block_type}): {time.time() - t:.0f} s")
    with torch.no_grad():
        return _rows(lambda x: model.model.norm.float()(x), X)


def head_weight(w, form):
    """The head as an f32 [V, H] weight: the bf16 lm_head upcast, or the engine's per-row int8 head."""
    if form == "int8":
        return _load("int8_sim").int8_head_weight(w)
    return w.float()


def write_outputs(final, jobs, w_bf16, outs, meta, log=print):
    """Per head form, the logits rows of its contexts (in blocks of <= HEAD_ROWS distinct rows), one file a context;
    a context whose ids and head another one already wrote is a hard link to that file."""
    import torch
    from safetensors.torch import save_file
    written = {}
    for form in ("bf16", "int8"):
        ks = [k for k in sorted(jobs) if jobs[k]["form"] == form]
        if not ks:
            continue
        t = time.time()
        W = head_weight(w_bf16, form)
        os.makedirs(outs[form], exist_ok=True)
        i = 0
        while i < len(ks):
            blk, nodes = [], set()
            while i < len(ks) and (not blk or len(nodes | set(jobs[ks[i]]["rows"])) <= HEAD_ROWS):
                blk.append(ks[i])
                nodes |= set(jobs[ks[i]]["rows"])
                i += 1
            nodes = sorted(nodes)
            where = {n: r for r, n in enumerate(nodes)}
            with torch.no_grad():
                lg = final[torch.tensor(nodes)] @ W.t()               # [rows, V] f32
            for k in blk:
                j = jobs[k]
                path = os.path.join(outs[form], k + ".golden.safetensors")
                same = written.get((j["seq"], form))
                if same:
                    if os.path.exists(path):
                        os.remove(path)
                    os.link(same, path)
                    continue
                rows = lg[torch.tensor([where[n] for n in j["rows"]])].contiguous()
                md = dict(meta, n_prompt=str(j["n_prompt"]), gen=str(len(j["force"])), lm_head=form)
                save_file({"logits": rows, "tokens": torch.tensor(j["force"], dtype=torch.int32)}, path + ".tmp",
                          metadata=md)
                os.replace(path + ".tmp", path)
                written[(j["seq"], form)] = path
                vu = int(meta["vocab_used"])
                arg = rows[:-1, :vu].argmax(-1)
                agree = (arg == torch.tensor(j["force"])).float().mean().item() if j["force"] else 1.0
                log(f"[{k}] {form} head, {j['n_prompt']} + {len(j['force'])} ids: the f32 argmax is the forced id "
                    f"at {agree:.3f} of the rows")
        del W
        log(f"{form} head: {len(ks)} contexts in {time.time() - t:.0f} s")
    return written


def cmd_plan(a):
    done = set(read_lines(a.done)) if a.done else set()
    ctx = load_contexts(a.ctx, a.match, done)
    if not ctx:
        print("no context to do")
        return None
    trie, jobs = plan(ctx, a.i8head_re)
    line, n_pos, n_rows = describe(trie, jobs)
    longest = max(len(j["seq"]) for j in jobs.values())
    print(line)
    print(f"estimate: ~{estimate_hours(n_pos, n_rows, longest):.1f} h on the box CPU (22 threads, f32; assumed rates)")
    return ctx, trie, jobs


def read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def cmd_run(a):
    a_done = {f[: -len(".golden.safetensors")] for d in (a.out, a.out_i8head) if os.path.isdir(d)
              for f in os.listdir(d) if f.endswith(".golden.safetensors")}
    done = set(read_lines(a.done)) if a.done else set()
    ctx = load_contexts(a.ctx, a.match, done | a_done)
    if not ctx:
        print("no context to do (every one has its judge)")
        return
    trie, jobs = plan(ctx, a.i8head_re)
    line, n_pos, n_rows = describe(trie, jobs)
    longest = max(len(j["seq"]) for j in jobs.values())
    print(line, flush=True)
    print(f"estimate: ~{estimate_hours(n_pos, n_rows, longest):.1f} h (assumed rates)", flush=True)
    if a.dry_run:
        return
    import resource
    import torch
    torch.set_grad_enabled(False)
    t0 = time.time()
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}", flush=True)
    model, tc, pf = build(a.snapshot)
    layers = model.model.layers

    def load_layer(i):
        layers[i].load_state_dict(pf.get(i), strict=True, assign=True)

    def unload_layer(i):
        layers[i].to_empty(device="meta")
    if a.work:
        os.makedirs(a.work, exist_ok=True)
    final = trie_forward(model, trie, load_layer, unload_layer, a.work,
                         log=lambda s: print(s + f" (wall {time.time() - t0:.0f} s)", flush=True))
    meta = {"snapshot": os.path.abspath(a.snapshot), "dtype": "f32", "lean": "1", "dump_kv": "False",
            "vocab_used": str(a.vocab_used), "model": "qwen3.8 (tools/oracle/qwen_forced.py: one trie forward, f32)"}
    write_outputs(final, jobs, model.lm_head.weight, {"bf16": a.out, "int8": a.out_i8head}, meta,
                  log=lambda s: print(s, flush=True))
    if a.work and os.path.exists(os.path.join(a.work, "hidden.pt")):
        os.remove(os.path.join(a.work, "hidden.pt"))
    print(f"done: {len(jobs)} judges, wall {time.time() - t0:.0f} s, peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "plan"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("snapshot")
            p.add_argument("--out", required=True, help="the bf16 head's judges")
            p.add_argument("--out-i8head", required=True, help="the int8 head's judges")
            p.add_argument("--work", help="the per-layer checkpoint's directory (resume)")
            p.add_argument("--vocab-used", type=int, default=248077)
            p.add_argument("--dry-run", action="store_true")
        p.add_argument("--ctx", action="append", required=True)
        p.add_argument("--match")
        p.add_argument("--done")
        p.add_argument("--i8head-re", default=I8HEAD_RE)
    a = ap.parse_args()
    if a.cmd == "plan":
        cmd_plan(a)
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()
