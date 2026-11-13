#!/usr/bin/env python3
"""Qwen3.8's f32 judges of the forced contexts (spec 28 N4 / N6, tests/server/spec_judge.h), on the CPU.

    qwen_forced.py run <snapshot> --ctx DIR [--ctx DIR ...] --out DIR --out-i8head DIR
                   [--memory-gib 64] [--chunk-rows 256] [--i8head-re REGEX] [--pfi8-re REGEX] [--kv8] [--match REGEX] [--done FILE] [--work DIR] [--dry-run]
    qwen_forced.py plan --ctx DIR [--ctx DIR ...] [--pfi8-re REGEX] [--match REGEX] [--done FILE]

A context is tests/golden/forced_judge.h's dump: <key>.prompt.ids and <key>.force.ids (B70_DUMP_FORCED). Its judge
is <out>/<key>.golden.safetensors in ornith_ref.py --lean's format: `logits` f32 [1 + gen][V] (row 0 = the prompt's
last position, row j = the position that consumed forced id j - 1: row j decides forced id j) and `tokens` int32
[gen], the forced ids. The model is Qwen3.8 in float32 throughout: the int4 checkpoint dequantised as dump.py
dequantises it (int4 x f16 scale, one bf16 rounding - every oracle's weights), every weight upcast to f32 and every
activation, GEMM, attention, GDN recurrence and norm in f32; the head is the checkpoint's bf16 lm_head upcast, or -
for the keys --i8head-re matches (the tests that load LmHeadForm::Int8) - the engine's per-row int8 head
(int8_sim.int8_head_weight, as ornith_ref.py --lm-head int8), written to --out-i8head.

--pfi8-re (the l0-int8 tests' judges, operator ruling 2026-10-23 - ornith_ref.py --sim-int8-prefill's rule on
Qwen3.8): for the keys it matches (PFI8_RE: every mtp_gpu_ key - mtp_gpu_test and its twins run `l0-int8`) the
positions of the TEST'S OWN prompt (tests/golden/prompts or tests/golden/toolcall <name>.ids, checked to be the key's
prefix - the dumped context's prompt also holds the ids decoded before the divergence) take spec 5's int8 prefill
linears: every int4 linear the h8 walk runs on Qwen3.8 (runtime/prefill/step.cc pf_linear / linear_i8_silu: the MLP's
gate || up and down; open-work 17: the mixer linears - GDN in_proj_qkv || in_proj_z and out_proj, FA q || k || v and
o_proj - are the bf16 l0 walk's) computes those rows as int8_sim.fq_x(x) . fq_w(w) (the rotation, per-token /
per-column int8 codes, de-quantised; each weight's fq_w once per layer). What the walk keeps bf16 stays exact (the
mixer linears; in_proj_a / in_proj_b: pf_ab_proj; the head). The positions after
the test's prompt are the engine's M = 1 decode rows and stay plain f32. A trie node is a (token, prompt pass) pair, so
a position is shared only by contexts that computed it the same way.

--kv8 (the int8 KV cache's tests: mtp_gpu_kv8_test, dflash_gpu_kv8_test - `--kv-cache int8`): every context of the
run reads its full-attention K and V as the int8 cache gives them back - int8_sim.kv8_fq on each (position, kv head)
row of the roped K and of V: rounded to bf16, rotated, int8 with an fp16 scale, de-quantised, un-rotated (bitwise
src/common/kv8.h's codes and scales: tests/common/kv8_sim_test.cc). Every position, prompt and decode alike: the
prefill's flash and the decode / verify attention all read the int8 rows. The rest (q, scores, softmax, GDN) stays
f32. A whole run is one KV form - its outputs go to their own dirs (f32_refs.sh forced-qwen-kv8 / -pfi8-kv8).

Why not one teacher-forced run per context (f32_ref_job.sh / ornith_ref.py): the spec tests' ~250 contexts are
~450k ids, but they are a few prompts each continued a few ways - a token TRIE of all of them holds ~48k distinct
positions. So the contexts are merged into one trie and ONE layer-major forward computes every distinct position
once: per decoder layer (streamed, stream.py via dump.stream_weights), the token-parallel parts (norms, every linear,
the MLP) over all positions at once, the sequence mixing per trie segment (a maximal chain with no branch) from its
parent's state - full attention over the segment's ancestors' K / V, the GDN causal conv over the four positions up
the path, the gated delta rule (modeling_qwen3_5.torch_chunk_gated_delta_rule) from the parent segment's final
recurrent state. Mathematically that is each context's own causal forward; the f32 rounding differs from a
one-sequence forward only in summation order (test_qwen_forced.py checks it against transformers' own forward).

Memory: contexts are packed into groups whose estimated peak fits --memory-gib (also capped by the container's
memory.max). An oversized singleton streams each layer in --chunk-rows chunks: its FA KV remains f32 for just
that layer; GDN keeps its f32 recurrent state and the causal-conv halo. No activation is truncated to bf16. The
estimate includes resident weights, prefetched/converting layer weights, hidden buffers, KV, workspaces and head
output copies; impossible caps are refused before weights are loaded. `plan` prints each group's peak estimate.

Resumable: the hidden state is checkpointed after every layer in --work (a killed run restarts at the layer it was
on, for the same set of contexts); keys listed in --done (one per line) or already in --out are skipped. Contexts
with the same prompt, forced ids and head share one file (a hard link; the same ids split at another prompt length
are another file: their rows differ). Run in the oracle container (run_in_container.sh) under
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
# --lm-head int8 (its tags carry the head form: mtp_gpu_<form>_, mtp_gpu_rf1_<form>_), and b70-serve's speculative
# arms (tools/box_validate/serve_judge.py: serve_int8_<set>_<name>; the server loads the int8 head, prefills with l0).
# mtp_verify_test's qwen3.8_int8 is the KV cache, not the head. Spec 17c's tp_prefill_test (tp_prefill_, both KV forms) and
# spec 17d's tp_mtp_test (tp_mtp_)
# runs b70-decode's int8 head; the TP twin count (tools/box_validate/twin_count.py: tp_twin_) b70-decode's and b70-serve's.
I8HEAD_RE = r"^(dflash_|mtp_gpu_(rf1_)?int8_|serve_int8_|slots_|tp_prefill_|tp_mtp_|tp_twin_)"
# The tests that prefill with l0-int8 (tests/CMakeLists.txt: mtp_gpu_test <snapshot> l0-int8 ..., every form).
PFI8_RE = r"^mtp_gpu_"
REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
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


def test_prompt_len(key, prompt, root=REPO, pfi8_re=PFI8_RE):
    """The l0-int8 test's own prompt length for `key` (0 when `pfi8_re` does not match it): its name's ids file,
    which must be a prefix of the key's prompt."""
    if not pfi8_re or not re.search(pfi8_re, key):
        return 0
    m = re.match(r"^mtp_gpu_(?:rf1_)?(?:bf16|int8)_(.+)_[0-9a-f]{16}$", key)
    if not m:
        raise SystemExit(f"{key}: an l0-int8 key whose prompt name cannot be read")
    for d in ("tests/golden/prompts", "tests/golden/toolcall"):
        p = os.path.join(root, d, m.group(1) + ".ids")
        if os.path.exists(p):
            ids = read_ids(p)
            if prompt[:len(ids)] != ids:
                raise SystemExit(f"{key}: {p} is not a prefix of the key's prompt")
            return len(ids)
    raise SystemExit(f"{key}: no ids file for the prompt {m.group(1)}")


class Trie:
    """Every context's prompt + forced ids as one token trie, numbered so that each SEGMENT (a maximal chain whose
    inner nodes have one child) is a contiguous index range and a parent segment precedes its children.

    tok / i8 / depth / parent: per node (i8: the position runs the int8 prompt pass; parent -1 at a root); seg_start /
    seg_len / seg_parent (segment index, -1); seg_anc[s]: the ancestor index ranges of segment s (its parent's, then
    the parent segment itself). `pfi8`: per sequence, its int8 prompt pass's length (0: none) - a node is a (token,
    depth < that length) pair."""

    def __init__(self, seqs, pfi8=None):
        tok, i8, par, kids, roots = [], [], [], [], {}
        for si, s in enumerate(seqs):
            P = pfi8[si] if pfi8 else 0
            node = -1
            for depth, t in enumerate(s):
                d = roots if node < 0 else kids[node]
                key = (t, depth < P)
                c = d.get(key)
                if c is None:
                    c = len(tok)
                    tok.append(t)
                    i8.append(depth < P)
                    par.append(node)
                    kids.append({})
                    d[key] = c
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
        self.i8 = [i8[o] for o in order]
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

    def path(self, seq, pfi8=0):
        """The node of every position of `seq` (new numbering), its first `pfi8` positions the int8 prompt pass."""
        out, node = [], -1
        for depth, t in enumerate(seq):
            node = (self._roots if node < 0 else self._kids[node])[(t, depth < pfi8)]
            out.append(node)
        return [self.old2new[n] for n in out]

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps([self.tok, self.parent] + ([self.i8] if any(self.i8) else [])).encode())
        return h.hexdigest()[:16]


def plan(ctx, i8head_re=I8HEAD_RE, pfi8_re=None, root=REPO):
    """The work: the trie, and per head form the contexts and their distinct (ids, head, prompt pass) outputs.
    `pfi8_re`: the keys whose test's own prompt takes the int8 prompt pass (test_prompt_len); None: no key."""
    keys = sorted(ctx)
    seqs = [ctx[k][0] + ctx[k][1] for k in keys]
    pfi8 = [test_prompt_len(k, ctx[k][0], root, pfi8_re) for k in keys]
    trie = Trie(seqs, pfi8)
    jobs = {}
    for k, s, P in zip(keys, seqs, pfi8):
        form = "int8" if re.search(i8head_re, k) else "bf16"
        pr, fo = ctx[k]
        rows = trie.path(s, P)[len(pr) - 1:]       # 1 + gen rows: the prompt's last position, then each forced id's
        assert len(rows) == len(fo) + 1
        jobs[k] = {"form": form, "rows": rows, "force": fo, "n_prompt": len(pr), "seq": tuple(s), "pfi8": P}
    return trie, jobs


def describe(trie, jobs, vocab=248320):
    raw = sum(len(j["seq"]) for j in jobs.values())
    uniq = {_out_key(j) for j in jobs.values()}
    rows = {}
    for j in jobs.values():
        rows.setdefault(_out_key(j), len(j["rows"]))
    nrow = sum(rows.values())
    forms = {f: sum(1 for j in jobs.values() if j["form"] == f) for f in ("bf16", "int8")}
    npf = sum(1 for j in jobs.values() if j["pfi8"])
    return (f"{len(jobs)} contexts ({forms['bf16']} bf16 head, {forms['int8']} int8 head, {npf} with the l0-int8 "
            f"prompt pass; {len(uniq)} distinct ids + head + prompt pass), {raw} ids -> {len(trie.tok)} trie "
            f"positions ({sum(trie.i8)} int8 prompt pass) in {len(trie.seg_start)} segments; "
            f"{nrow} logits rows written ({nrow * vocab * 4 / 1e9:.1f} GB)"), len(trie.tok), nrow


def _out_key(j):
    """What decides a judge file: the ids, where the prompt ends, the head, the int8 prompt pass."""
    return (j["n_prompt"], j["seq"], j["form"], j["pfi8"])


# The shipped Qwen3.8 text config; `run` reads the actual snapshot (tiny tests use theirs).
DEFAULT_CONFIG = dict(hidden_size=5120, intermediate_size=17408, vocab_size=248320,
                      num_attention_heads=24, num_key_value_heads=4, head_dim=256,
                      linear_num_key_heads=16, linear_num_value_heads=48,
                      linear_key_head_dim=128, linear_value_head_dim=128)


def estimate_peak(n, longest, config, chunk_rows=0, output_rows=0, recurrent_states=2):
    """Conservative live bytes: resident embed/head, current f32 + prefetched bf16 layer and conversion
    scratch, hidden buffers, projections, GDN's 64-wide chunk workspace and retained branch states,
    FA scores and logits copies.
    Chunked single chains retain only this layer's f32 KV (no bf16 truncation) and bounded GDN state.
    Includes 4 GiB for Python/torch, allocator slack and checkpoint/page-cache headroom.
    """
    h, mid, v = (config[k] for k in ("hidden_size", "intermediate_size", "vocab_size"))
    nh, nk, hd = (config[k] for k in ("num_attention_heads", "num_key_value_heads", "head_dim"))
    kh = config["linear_num_key_heads"] * config["linear_key_head_dim"]
    vh = config["linear_num_value_heads"] * config["linear_value_head_dim"]
    conv = 2 * kh + vh
    weights = 4 * v * h + 10 * (3 * h * mid + h * (2 * nh * hd + 2 * nk * hd))
    head = 16 * v * h + 16 * max(HEAD_ROWS, output_rows) * v + 12 * n * h
    rows = min(n, chunk_rows) if chunk_rows else n
    hidden = 12 * n * h
    projections = 4 * rows * (6 * h + 6 * conv + 8 * vh + 4 * nh * hd + 3 * mid)
    gdn_work = 4 * rows * config["linear_num_value_heads"] * (8 * 64 + 8 * config["linear_value_head_dim"])
    kv = 16 * longest * nk * hd
    states = 4 * recurrent_states * config["linear_num_value_heads"] * config["linear_key_head_dim"] * \
        config["linear_value_head_dim"]
    scores = 12 * min(rows, Q_BLOCK) * longest * (1 if chunk_rows else nh)
    return int(4 * 2**30 + max(weights + hidden + projections + gdn_work + kv + scores + states, head))


def memory_cap(gib):
    cap = int(gib * 2**30)
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                value = f.read().strip()
            if value.isdigit():
                cap = min(cap, int(value))
    return cap


def memory_plan(ctx, config, cap, chunk_rows=256, i8head_re=I8HEAD_RE, pfi8_re=None, root=REPO):
    """[(trie, jobs, chunk_rows (0: trie forward), estimated peak bytes)], each <= cap.
    Greedily pack short contexts; an oversized singleton runs one layer at a time in bounded chunks.
    """
    groups, pending = [], {}

    def group(contexts, chunk=0):
        trie, jobs = plan(contexts, i8head_re, pfi8_re, root)
        # Mirror gdn's DFS state lifetime: keep a parent's state until its last child.
        # Two extra states cover the current result and the previous loop's local references.
        left, live, peak_states = list(trie.n_children), 0, 2
        for si, ps in enumerate(trie.seg_parent):
            peak_states = max(peak_states, live + 2)
            if left[si]:
                live += 1
            if ps >= 0:
                left[ps] -= 1
                if not left[ps]:
                    live -= 1
        peak = estimate_peak(len(trie.tok), max(len(j["seq"]) for j in jobs.values()), config, chunk,
                             max(len(j["rows"]) for j in jobs.values()), peak_states)
        return trie, jobs, chunk, peak

    for key in sorted(ctx):
        candidate = group(pending | {key: ctx[key]})
        if candidate[3] <= cap:
            pending[key] = ctx[key]
            continue
        if pending:
            groups.append(group(pending))
            pending = {}
        single = group({key: ctx[key]})
        if single[3] <= cap:
            pending[key] = ctx[key]
        else:
            single = group({key: ctx[key]}, chunk_rows)
            if single[3] > cap:
                raise ValueError(f"{key}: estimated peak {single[3] / 2**30:.1f} GiB exceeds memory cap "
                                 f"{cap / 2**30:.1f} GiB even with {chunk_rows}-row chunks")
            groups.append(single)
    if pending:
        groups.append(group(pending))
    return groups


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


def _rows_m(f, x, m, block=ROW_BLOCK):
    """_rows with the int8 prompt-pass mask's block beside each row block (m None: no int8 row)."""
    import torch
    return torch.cat([f(x[a:a + block], None if m is None else m[a:a + block])
                      for a in range(0, x.shape[0], block)], 0)


class Int8Pass:
    """The l0-int8 prompt pass's linears for one layer: rows where `mask` is set compute int8_sim.fq_x(x) .
    fq_w(w) (each weight's fq_w once, cached for the layer), the others the plain f32 linear."""

    def __init__(self, mask):
        self.mask = mask if mask is not None and bool(mask.any()) else None
        self.wq = {}

    def lin(self, mod, x, m):
        if m is None or not bool(m.any()):
            return mod(x)
        import torch
        import torch.nn.functional as F
        I8 = _load("int8_sim")
        wq = self.wq.get(mod)
        if wq is None:
            wq = self.wq[mod] = I8.fq_w(mod.weight.float())
        out = torch.empty(x.shape[0], mod.out_features, dtype=x.dtype)
        keep = ~m
        if bool(keep.any()):
            out[keep] = mod(x[keep])
        out[m] = F.linear(I8.fq_x(x[m]), wq, mod.bias)
        return out

    def rows(self, mod, x):
        """mod over every row of x (x's rows are the trie's positions, in order)."""
        if self.mask is None:
            return _rows(mod, x)
        return _rows_m(lambda xb, mb: self.lin(mod, xb, mb), x, self.mask)


def _anc_index(trie, s):
    import torch
    parts = [torch.arange(a, b) for a, b in trie.seg_anc[s]]
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long)


def attention(A, h, trie, cos, sin, M, q8=None, kv8=False):
    """Qwen3_5Attention over the trie: q / k / v / gate token-parallel, softmax(q k^T * scale) v per segment over
    its ancestors' and its own (causal) positions, f32. `q8`: the int8 prompt pass's linears (Int8Pass); `kv8`: K and V
    through the int8 cache (int8_sim.kv8_fq)."""
    import torch
    q8 = q8 or Int8Pass(None)
    N, hd = h.shape[0], A.head_dim
    qg = q8.rows(A.q_proj, h).view(N, -1, 2 * hd)
    q, gate = torch.chunk(qg, 2, dim=-1)
    gate = gate.reshape(N, -1)
    q = A.q_norm(q.contiguous())
    k = A.k_norm(q8.rows(A.k_proj, h).view(N, -1, hd))
    v = q8.rows(A.v_proj, h).view(N, -1, hd)
    q, k = M.apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin)
    q, k, v = q[0], k[0], v.transpose(0, 1).contiguous()        # [heads, N, hd]
    if kv8:
        I8 = _load("int8_sim")
        k, v = I8.kv8_fq(k), I8.kv8_fq(v)
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
    return q8.rows(A.o_proj, o)


def gdn(G, h, trie, M, q8=None, initial_state=None, prefix=0, return_state=False):
    """Qwen3_5GatedDeltaNet over the trie: the projections token-parallel; the causal conv (kernel K) as the sum of
    K taps up the path (a position's own and its K - 1 ancestors', zero above a root); the gated delta rule per
    segment from its parent segment's final state (modeling_qwen3_5.torch_chunk_gated_delta_rule, fp32). `q8`: the
    int8 prompt pass's linears (in_proj_qkv, in_proj_z, out_proj; in_proj_b / in_proj_a stay exact: pf_ab_proj).
    For a chunk's single chain, `prefix` rows supply only the conv halo; recurrence starts at `initial_state`,
    skips those halo rows, and `return_state` returns its final state beside the output.
    """
    import torch
    import torch.nn.functional as F
    q8 = q8 or Int8Pass(None)
    N = h.shape[0]
    qkv = q8.rows(G.in_proj_qkv, h)                             # [N, conv_dim]
    z = q8.rows(G.in_proj_z, h)
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
        a, L = a + prefix, L - prefix
        init = states[ps] if ps >= 0 else initial_state
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
    core = G.norm(core[prefix:].reshape(-1, G.head_v_dim),
                  z[prefix:].reshape(-1, G.head_v_dim)).reshape(N - prefix, -1)
    out = _rows_m(lambda x, m: q8.lin(G.out_proj, x, m), core,
                  None if q8.mask is None else q8.mask[prefix:])
    return (out, fin) if return_state else out


def layer_forward(lay, X, trie, cos, sin, M, i8_mask=None, kv8=False):
    """One decoder layer over every trie position; `i8_mask` (bool [N]): the positions of the int8 prompt pass; `kv8`:
    the int8 KV cache's K / V."""
    q8 = Int8Pass(i8_mask)
    h = _rows(lay.input_layernorm, X)
    if lay.block_type == "linear_attention":   # the mixer linears plain: l0-int8 runs them bf16 (open-work 17)
        X = X + gdn(lay.linear_attn, h, trie, M)
    else:
        X = X + attention(lay.self_attn, h, trie, cos, sin, M, kv8=kv8)
    del h
    if q8.mask is None:
        return X + _rows(lambda x: lay.mlp(lay.post_attention_layernorm(x)), X)
    mlp = lay.mlp                                   # Qwen3_5MLP: down(act(gate(x)) * up(x))

    def block(x, m):
        x = lay.post_attention_layernorm(x)
        return q8.lin(mlp.down_proj, mlp.act_fn(q8.lin(mlp.gate_proj, x, m)) * q8.lin(mlp.up_proj, x, m), m)
    return X + _rows_m(block, X, q8.mask)


def chunked_layer(lay, X, trie, cos, sin, M, chunk_rows, i8_mask, kv8=False):
    """One unbranched context, layer-major: f32 KV for this FA layer only, GDN state + conv halo.
    X/Y hold the whole layer's hidden rows; all projection/GDN work is bounded by chunk_rows + halo.
    FA runs one query head at a time without expanded KV or an all-head score tensor.
    """
    import torch
    N = len(trie.tok)
    assert trie.parent == [-1] + list(range(N - 1)), "chunking requires a single context"
    Y = torch.empty_like(X)
    state, K, V = None, None, None
    for a in range(0, N, chunk_rows):
        b = min(N, a + chunk_rows)
        halo = min(a, lay.linear_attn.conv1d.kernel_size[0] - 1) if lay.block_type == "linear_attention" else 0
        h = lay.input_layernorm(X[a - halo:b])
        q8 = Int8Pass(None)   # the mixer linears plain: l0-int8 runs them bf16 (open-work 17)
        if lay.block_type == "linear_attention":
            local = Trie([trie.tok[a - halo:b]])
            mix, state = gdn(lay.linear_attn, h, local, M, q8, state, halo, True)
        else:
            A, L = lay.self_attn, b - a
            hd = A.head_dim
            qg = q8.rows(A.q_proj, h).view(L, -1, 2 * hd)
            q, gate = torch.chunk(qg, 2, dim=-1)
            q = A.q_norm(q.contiguous()).transpose(0, 1)[None]
            k = A.k_norm(q8.rows(A.k_proj, h).view(L, -1, hd)).transpose(0, 1)[None]
            q, k = M.apply_rotary_pos_emb(q, k, cos[:, a:b], sin[:, a:b])
            v = q8.rows(A.v_proj, h).view(L, -1, hd).transpose(0, 1)
            k = k[0]
            if kv8:
                I8 = _load("int8_sim")
                k, v = I8.kv8_fq(k), I8.kv8_fq(v)
            K = k if K is None else torch.cat([K, k], dim=1)
            V = v if V is None else torch.cat([V, v], dim=1)
            nH, nKV = q.shape[1], K.shape[0]
            out = torch.empty(nH, L, hd, dtype=h.dtype)
            for head in range(nH):
                kv = head // (nH // nKV)
                for r0 in range(0, L, Q_BLOCK):
                    r1 = min(L, r0 + Q_BLOCK)
                    sc = (q[0, head, r0:r1] @ K[kv, :a + r1].t()) * A.scaling
                    own = torch.ones(r1 - r0, r1, dtype=torch.bool).triu(r0 + 1)
                    sc[:, a:].masked_fill_(own, float("-inf"))
                    out[head, r0:r1] = torch.softmax(sc, -1) @ V[kv, :a + r1]
            mix = q8.rows(A.o_proj, out.transpose(0, 1).reshape(L, -1) * torch.sigmoid(gate.reshape(L, -1)))
        x = X[a:b] + mix
        q8 = Int8Pass(None if i8_mask is None else i8_mask[a:b])
        mlp = lay.mlp

        def block(xb, mb):
            hn = lay.post_attention_layernorm(xb)
            return q8.lin(mlp.down_proj, mlp.act_fn(q8.lin(mlp.gate_proj, hn, mb)) *
                          q8.lin(mlp.up_proj, hn, mb), mb)
        Y[a:b] = x + _rows_m(block, x, q8.mask)
    return Y


def trie_forward(model, trie, load_layer=None, unload_layer=None, work=None, log=print, chunk_rows=0, kv8=False):
    """The final hidden state (after the last norm) of every trie position, f32 [N, H]. `load_layer(i)` makes layer
    i's weights present (streamed runs), `unload_layer(i)` drops them; `work`: a directory for the per-layer
    checkpoint (resume); `kv8`: the int8 KV cache's K / V."""
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    tok = torch.tensor(trie.tok)
    pos = torch.tensor(trie.depth)
    i8_mask = torch.tensor(trie.i8, dtype=torch.bool) if any(trie.i8) else None
    fp = trie.fingerprint() + (f"-chunk{chunk_rows}" if chunk_rows else "") + ("-kv8" if kv8 else "")
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
            if chunk_rows:
                X = chunked_layer(lay, X, trie, cos, sin, M, chunk_rows, i8_mask, kv8)
            else:
                X = layer_forward(lay, X, trie, cos, sin, M, i8_mask, kv8)
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
                same = written.get(_out_key(j))
                if same:
                    if os.path.exists(path):
                        os.remove(path)
                    os.link(same, path)
                    continue
                rows = lg[torch.tensor([where[n] for n in j["rows"]])].contiguous()
                md = dict(meta, n_prompt=str(j["n_prompt"]), gen=str(len(j["force"])), lm_head=form,
                          pfi8_prompt=str(j["pfi8"]))
                save_file({"logits": rows, "tokens": torch.tensor(j["force"], dtype=torch.int32)}, path + ".tmp",
                          metadata=md)
                os.replace(path + ".tmp", path)
                written[_out_key(j)] = path
                vu = int(meta["vocab_used"])
                arg = rows[:-1, :vu].argmax(-1)
                agree = (arg == torch.tensor(j["force"])).float().mean().item() if j["force"] else 1.0
                pf = f" (the l0-int8 prompt pass over its first {j['pfi8']})" if j["pfi8"] else ""
                log(f"[{k}] {form} head, {j['n_prompt']} + {len(j['force'])} ids{pf}: the f32 argmax is the forced "
                    f"id at {agree:.3f} of the rows")
        del W
        log(f"{form} head: {len(ks)} contexts in {time.time() - t:.0f} s")
    return written


def planned_groups(a, ctx):
    snapshot = getattr(a, "snapshot", None)
    config = DEFAULT_CONFIG
    if snapshot:
        with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
            raw = json.load(f)
        config = raw.get("text_config", raw)
    cap = memory_cap(a.memory_gib)
    groups = memory_plan(ctx, config, cap, a.chunk_rows, a.i8head_re, a.pfi8_re)
    for i, (trie, jobs, chunk, peak) in enumerate(groups):
        print(f"memory group {i + 1}/{len(groups)}: " + describe(trie, jobs, config["vocab_size"])[0], flush=True)
        print(f"  peak estimate {peak / 2**30:.1f} GiB <= cap {cap / 2**30:.1f} GiB; "
              f"{'chunked ' + str(chunk) + ' rows, one layer KV/state' if chunk else 'trie forward'}", flush=True)
        _, n_pos, n_rows = describe(trie, jobs, config["vocab_size"])
        hours = estimate_hours(n_pos, n_rows, max(len(j["seq"]) for j in jobs.values()))
        print(f"  compute estimate ~{hours:.1f} h (assumed rates; chunk/stream overhead excluded)", flush=True)
    return groups


def cmd_plan(a):
    done = set(read_lines(a.done)) if a.done else set()
    ctx = load_contexts(a.ctx, a.match, done)
    if not ctx:
        print("no context to do")
        return None
    return planned_groups(a, ctx)


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
    groups = planned_groups(a, ctx)
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
    meta = {"snapshot": os.path.abspath(a.snapshot), "dtype": "f32", "lean": "1", "dump_kv": "False",
            "vocab_used": str(a.vocab_used), "model": "qwen3.8 (tools/oracle/qwen_forced.py: f32)",
            "pfi8_re": a.pfi8_re or "", "kv": "int8 rotkv" if a.kv8 else "bf16"}
    for trie, jobs, chunk, peak in groups:
        work = os.path.join(a.work, trie.fingerprint() + f"-chunk{chunk}") if a.work else None
        if work:
            os.makedirs(work, exist_ok=True)
        final = trie_forward(model, trie, load_layer, unload_layer, work, chunk_rows=chunk, kv8=a.kv8,
                             log=lambda s: print(s + f" (wall {time.time() - t0:.0f} s)", flush=True))
        write_outputs(final, jobs, model.lm_head.weight, {"bf16": a.out, "int8": a.out_i8head}, meta,
                      log=lambda s: print(s, flush=True))
        del final
        if work and os.path.exists(os.path.join(work, "hidden.pt")):
            os.remove(os.path.join(work, "hidden.pt"))
    print(f"done: {len(ctx)} judges, wall {time.time() - t0:.0f} s, peak RSS "
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
            p.add_argument("--kv8", action="store_true", help="K / V through the int8 KV cache (int8_sim.kv8_fq)")
        if name == "plan":
            p.add_argument("--snapshot", help="config for the memory estimate (default Qwen3.8)")
        p.add_argument("--memory-gib", type=float, default=64, help="peak cap, also bounded by the cgroup cap")
        p.add_argument("--chunk-rows", type=int, default=256, help="rows per oversized singleton chunk")
        p.add_argument("--ctx", action="append", required=True)
        p.add_argument("--match")
        p.add_argument("--done")
        p.add_argument("--i8head-re", default=I8HEAD_RE)
        p.add_argument("--pfi8-re", help="the keys whose test prefills with l0-int8: their test's prompt takes the "
                       "int8 prompt pass (e.g. PFI8_RE, '%s'); default none" % PFI8_RE)
    a = ap.parse_args()
    if a.memory_gib <= 0 or a.chunk_rows <= 0:
        ap.error("--memory-gib and --chunk-rows must be positive")
    if a.cmd == "plan":
        cmd_plan(a)
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()
