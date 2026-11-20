#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The CPU reference of NVIDIA Nemotron 3.5 Super (VL, 120B-A12B): the language model and its MTP layer.

A plain-torch restatement read straight from the safetensors - no transformers model is built - so it runs on the
249 GB BF16 original and on our int4 export alike in a few GB of RAM: LAYER-MAJOR (every prompt row through block l,
then block l + 1), each block's dense weights read when the block runs and dropped after, routed experts read only
for the experts some row routes to. Int4 linears (GPTQ v1 packing, `.qweight` / `.scales` / `.qzeros`, symmetric
`(q - 8) x scale`, group 64 - or 128 through the exact g128 -> g64 expansion) are dequantised exactly: the
ENGINE-FORMAT reference.

The arithmetic is the family's (restored from Nemotron 3.5 Bolt's reference at 55e8923f, which Super shares block
for block: Bolt is Super compressed):
  norm       fp32 variance, x * rsqrt(var + eps), w * that in fp32, one rounding (NemotronHRMSNorm)
  Mamba-2    in_proj -> z | xBC | dt; the causal conv over raw xBC columns + SiLU; the SSD scan; the gated RMSNorm
             over groups of d_inner / n_groups; out_proj - through nemotron_mamba_ops (the training kernels' reading:
             fp32 inside, NO dt clamp; the native class clamps its chunked path at time_step_min)
  attention  32 q / 2 kv heads x 128, no bias, NoPE, causal, scale head_dim^-0.5 (the eager form: scores in the model
             dtype, fp32 softmax rounded back)
  MoE        router: fp32 logits, sigmoid, top-k (22) on s + e_score_correction_bias (n_group 1), weights s at the
             ids renormalised (/ (sum + 1e-20)) x routed_scaling_factor (5.0), fp32; fc1_latent (4096 -> 1024); each
             routed relu^2 expert (up 1024 -> 2688, down -> 1024) over exactly its rows; the combine in fp32, ascending
             expert id, one rounding; fc2_latent; + the shared relu^2 expert (5376) on the normed input
  MTP        (vLLM's nemotron_h_mtp reading) for the token at t+1 and the trunk's final hidden h_t (after norm_f):
             f = eh_proj([enorm(embed(x_{t+1})) | hnorm(h_t)]); r = f + attn(norm0(f)) (its own causal KV over the
             MTP rows); r = r + moe(norm1(r)); m_t = final_layernorm(r); draft logits = lm_head(m_t), predicting
             x_{t+2}. Recursive drafts feed m_t back as the next hidden with the drafted token.

    nemotron_ref.py run <snapshot> --prompt <ids> --out <golden.safetensors> [--gen 8] [--draft 4]
                        [--dtype bf16|fp32] [--logits-tail 64] [--layers N] [--no-mtp]
    nemotron_ref.py compare <a.safetensors> <b.safetensors> [--min-mean-cos C] [--mtp]
                        (--mtp: both files must hold mtp.logits - FAIL when either lacks them)
    nemotron_ref.py hfcheck <tiny snapshot> --prompt <ids>     (the trunk against transformers' native class, fp32)

`<ids>` is a file of whitespace- or comma-separated token ids, or the ids inline ("1,2,3"). The golden file:
    logits     f32 [Lr][V]   rows logits_row0 .. T+gen-1 (metadata), the last --logits-tail prompt rows + gen rows
    tokens     i32 [gen]     greedy (the lower id on an exact tie);  nll f32 [T-1] the prompt's NLL
    route.ids.L<l> i32 [T+gen][k] / route.w.L<l> f32: each MoE block's selected ids ascending and their weights
    mtp.logits f32 [Lr'][V]  the MTP draft logits of the same rows (row t predicts x_{t+2})
    mtp.top1   i32 [T+gen-1] the MTP layer's greedy draft at every row;  drafts i32 [draft] the recursive drafts
               after the last row (the first one is mtp.top1's last)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]   # tools/oracle/tokenize.py

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ops = _load("nemotron_mamba_ops")
cpu_bf16 = _load("cpu_bf16")
INT4 = (".qweight", ".qzeros", ".scales")
KINDS = {"mamba": "mamba", "moe": "moe", "attention": "attention", "linear_attention": "mamba",
         "full_attention": "attention"}


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(2)


# ------------------------------------------------------------------------------------------------
# the config and the names

def config(snapshot: str, layers: int | None = None) -> dict:
    """The language model's config (the omni wrapper's llm_config, or a plain nemotron_h config) with `kinds` (the
    trunk's block kinds, the native spelling read too) and `mtp_kinds`; layers=N keeps blocks [0, N)."""
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    llm = dict(cfg["llm_config"] if cfg.get("model_type") == "nemotron_h_omni" else cfg)
    if llm.get("model_type") != "nemotron_h":
        die(f"{snapshot}: language model type {llm.get('model_type')!r}, expected nemotron_h")
    kinds = [KINDS[k] for k in (llm.get("layers_block_type") or llm["layer_types"])]
    if layers is not None:
        if not 1 <= layers <= len(kinds):
            die(f"--layers {layers}: the checkpoint has {len(kinds)}")
        kinds = kinds[:layers]
    llm["kinds"] = kinds
    llm["mtp_kinds"] = [KINDS[k] for k in llm.get("mtp_layers_block_type") or []] \
        if llm.get("num_nextn_predict_layers") else []
    return llm


def linear_shapes(llm: dict, kind: str) -> dict:
    """mixer-relative linear name -> (K, N) of one block of `kind` (the trunk's; the MTP layer's blocks are the same)."""
    H = llm["hidden_size"]
    if kind == "mamba":
        nh, hd, ng, st = llm["mamba_num_heads"], llm["mamba_head_dim"], llm["n_groups"], llm["ssm_state_size"]
        di = nh * hd
        return {"in_proj": (H, di + di + 2 * ng * st + nh), "out_proj": (di, H)}
    if kind == "attention":
        qn, kvn = llm["num_attention_heads"] * llm["head_dim"], llm["num_key_value_heads"] * llm["head_dim"]
        return {"q_proj": (H, qn), "k_proj": (H, kvn), "v_proj": (H, kvn), "o_proj": (qn, H)}
    L, S, I, E = (llm["moe_latent_size"], llm["moe_shared_expert_intermediate_size"], llm["moe_intermediate_size"],
                  llm["n_routed_experts"])
    out = {"fc1_latent_proj": (H, L), "fc2_latent_proj": (L, H), "shared_experts.up_proj": (H, S),
           "shared_experts.down_proj": (S, H)}
    for e in range(E):
        out[f"experts.{e}.up_proj"] = (L, I)
        out[f"experts.{e}.down_proj"] = (I, L)
    return out


def kept_shapes(llm: dict, kind: str) -> dict:
    """mixer-relative name -> (dtype, shape) of a block's tensors that stay as the original ships them."""
    nh, hd, ng, st = llm["mamba_num_heads"], llm["mamba_head_dim"], llm["n_groups"], llm["ssm_state_size"]
    if kind == "mamba":
        cd = nh * hd + 2 * ng * st
        return {"A_log": ("BF16", (nh,)), "D": ("BF16", (nh,)), "dt_bias": ("BF16", (nh,)),
                "conv1d.weight": ("BF16", (cd, 1, llm["conv_kernel"])), "conv1d.bias": ("BF16", (cd,)),
                "norm.weight": ("BF16", (nh * hd,))}
    if kind == "moe":
        E = llm["n_routed_experts"]
        return {"gate.weight": ("BF16", (E, llm["hidden_size"])), "gate.e_score_correction_bias": ("F32", (E,))}
    return {}


def mtp_bf16(name: str) -> bool:
    """The MTP layer's linears our export keeps bf16 (tools/quantize/nemotron/autoround_run.py IGNORE): all but the
    routed experts."""
    return name.startswith("mtp.") and ".experts." not in name


def expected_names(llm: dict, form: str, trunk: str = "backbone") -> dict:
    """Every language-model tensor of a checkpoint in `form` -> (dtype, shape) (None for an int4 part: check.py
    derives those shapes): "bf16" NVIDIA's original (every linear `.weight`), "int4" our export (every trunk linear
    and the MTP layer's routed experts int4). The trunk is `language_model.<trunk>.*`: both spell it `backbone`
    (AutoRound writes the source's names back). No vision."""
    H, V = llm["hidden_size"], llm["vocab_size"]
    trunk = f"language_model.{trunk}."
    out = {trunk + "embeddings.weight": ("BF16", (V, H)), trunk + "norm_f.weight": ("BF16", (H,)),
           "language_model.lm_head.weight": ("BF16", (V, H))}

    def block(p: str, kind: str, mtp: bool):
        out[p + "norm.weight"] = ("BF16", (H,))
        for s, v in kept_shapes(llm, kind).items():
            out[p + "mixer." + s] = v
        for s, (K, N) in linear_shapes(llm, kind).items():
            base = p + "mixer." + s
            if form == "bf16" or (mtp and ".experts." not in base):
                out[base + ".weight"] = ("BF16", (N, K))
            else:
                for sfx in INT4:
                    out[base + sfx] = None
    for i, kind in enumerate(llm["kinds"]):
        block(f"{trunk}layers.{i}.", kind, False)
    if llm["mtp_kinds"]:
        m = "language_model.mtp.layers."
        out[m + "0.eh_proj.weight"] = ("BF16", (H, 2 * H))
        for s in ("enorm", "hnorm"):
            out[f"{m}0.{s}.weight"] = ("BF16", (H,))
        out[f"{m}{len(llm['mtp_kinds']) - 1}.final_layernorm.weight"] = ("BF16", (H,))
        for j, kind in enumerate(llm["mtp_kinds"]):
            block(f"{m}{j}.", kind, True)
    return out


# ------------------------------------------------------------------------------------------------
# the checkpoint

def dequant(qw: torch.Tensor, sc: torch.Tensor, g: int) -> torch.Tensor:
    """GPTQ v1 int4 [K/8][N] x F16 scales [K/g][N] -> the fp32 weight [N][K], (q - 8) x scale exactly."""
    K, N = qw.shape[0] * 8, qw.shape[1]
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((qw.unsqueeze(1) >> shifts[None, :, None]) & 15).reshape(K, N)       # row k = 8 * (k // 8) + nibble
    return ((q.float() - 8) * sc.float().repeat_interleave(g, 0)).t().contiguous()


class Ckpt:
    """Name -> tensor over a snapshot's shards. Names in are the export's, without `language_model.`: `model.*` (the
    trunk; read as `backbone.*` from NVIDIA's original), `mtp.*`, `lm_head.weight`. A linear's `.weight` comes
    dequantised when the checkpoint has it int4."""

    def __init__(self, snapshot: str, dtype: torch.dtype):
        from safetensors import safe_open
        self.dir, self.dtype = snapshot, dtype
        idx = os.path.join(snapshot, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx, encoding="utf-8") as f:
                self.wm = json.load(f)["weight_map"]
        else:
            with safe_open(os.path.join(snapshot, "model.safetensors"), framework="pt") as h:
                self.wm = {k: "model.safetensors" for k in h.keys()}
        missing = sorted({fn for fn in self.wm.values() if not os.path.exists(os.path.join(snapshot, fn))})
        if missing:
            die(f"{snapshot}: {len(missing)} shard(s) of the index are absent ({missing[:2]})")
        self.h = {fn: safe_open(os.path.join(snapshot, fn), framework="pt", device="cpu")
                  for fn in sorted(set(self.wm.values()))}
        self.backbone = any(k.startswith("language_model.backbone.") for k in self.wm)
        self.form = "int4" if any(k.endswith(".qweight") for k in self.wm) else "bf16"
        self.checked: set = set()

    def c(self, name: str) -> str:
        if self.backbone and name.startswith("model."):
            name = "backbone." + name[len("model."):]
        return "language_model." + name

    def has(self, name: str) -> bool:
        return self.c(name) in self.wm

    def raw(self, name: str) -> torch.Tensor:
        c = self.c(name)
        if c not in self.wm:
            die(f"{self.dir}: {c} absent")
        return self.h[self.wm[c]].get_tensor(c)

    def get(self, name: str) -> torch.Tensor:
        """The tensor in the run's dtype - except the router bias (fp32) and A_log / D / dt_bias (as stored: the ops
        widen them)."""
        if name.endswith(".weight") and self.has(name[:-len(".weight")] + ".qweight"):
            return self.int4(name[:-len(".weight")]).to(self.dtype)
        t = self.raw(name)
        if name.endswith(("e_score_correction_bias", "A_log", ".D", "dt_bias")):
            return t
        return t.to(self.dtype) if t.is_floating_point() else t

    def int4(self, base: str) -> torch.Tensor:
        qw, sc = self.raw(base + ".qweight"), self.raw(base + ".scales")
        g = qw.shape[0] * 8 // sc.shape[0]
        if base not in self.checked:
            self.checked.add(base)
            if not bool((self.raw(base + ".qzeros") == 0x77777777).all()):
                die(f"{base}.qzeros is not 0x77777777: not symmetric int4 (the engine's rule is (q - 8) x scale)")
            if self.has(base + ".g_idx"):
                gi = self.raw(base + ".g_idx")
                if not torch.equal(gi, torch.arange(gi.numel(), dtype=gi.dtype) // g):
                    die(f"{base}.g_idx is not i // {g} (act-order is not the engine's format)")
        if sc.dtype == torch.bfloat16:
            s16 = sc.to(torch.float16)       # the engine converts BF16 scales to F16 exactly or refuses
            if not torch.equal(s16.float(), sc.float()):
                die(f"{base}.scales: a BF16 scale is not exact in F16")
            sc = s16
        elif sc.dtype != torch.float16:
            die(f"{base}.scales is {sc.dtype}, expected F16 (or BF16)")
        if g == 128:
            sc, g = sc.repeat_interleave(2, 0), 64
        elif g != 64:
            die(f"{base}: group {g} (the engine's format is g64, or g128 through its exact expansion)")
        return dequant(qw, sc, g)


# ------------------------------------------------------------------------------------------------
# the ops

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (w.to(torch.float32) * xf).to(x.dtype)


def relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.square(F.relu(x))


def route(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, k: int, scale: float):
    """Rows x [T][H] -> (ids [T][k] ascending, w [T][k] fp32 at those ids)."""
    s = F.linear(x.float(), gate_w.float()).sigmoid()
    ids = torch.topk(s + bias.float()[None], k=k, dim=-1, sorted=False)[1]
    w = s.gather(1, ids)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) * scale
    o = ids.argsort(-1)
    return ids.gather(1, o), w.gather(1, o)


class Model:
    def __init__(self, snapshot: str, dtype: torch.dtype, layers: int | None = None):
        self.cfg = config(snapshot, layers)
        self.ck = Ckpt(snapshot, dtype)
        self.dtype = dtype
        self.eps = self.cfg["layer_norm_epsilon"]
        self.embed = self.ck.get("model.embeddings.weight")
        self.head = self.ck.get("lm_head.weight")
        self.norm_f = self.ck.get("model.norm_f.weight")
        self.routes: dict = {}

    def w(self, name: str) -> torch.Tensor:
        return self.ck.get(name)

    # -- the mixers, on rows x [T][H] (already normed); st is the block's state dict
    def mamba(self, p: str, x: torch.Tensor, st: dict) -> torch.Tensor:
        c = self.cfg
        nh, hd, ng, ns, K = c["mamba_num_heads"], c["mamba_head_dim"], c["n_groups"], c["ssm_state_size"], c["conv_kernel"]
        di = nh * hd
        T = x.shape[0]
        z, xBC, dt = torch.split(F.linear(x, self.w(p + "in_proj.weight")), [di, di + 2 * ng * ns, nh], dim=-1)
        cw, cb = self.w(p + "conv1d.weight").squeeze(1), self.w(p + "conv1d.bias")
        xt = xBC.t()[None]                                                     # [1][C][T]
        prev = st.get("conv")
        if prev is None:
            prev = torch.zeros(1, xt.shape[1], K - 1, dtype=torch.float32)
        conv = ops.conv_seq(xt, cw, cb, prev)[0].t()                           # [T][C]
        st["conv"] = torch.cat([prev, xt.float()], -1)[..., -(K - 1):]
        xs, B, C = torch.split(conv, [di, ng * ns, ng * ns], dim=-1)
        A = ops.A_from_log(self.w(p + "A_log"))
        D, dtb = self.w(p + "D"), self.w(p + "dt_bias")
        if T == 1 and st.get("ssm") is not None:
            y, S = ops.ssm_update(st["ssm"][None], xs.view(1, nh, hd), dt, A, B.view(1, ng, ns), C.view(1, ng, ns),
                                  D, dtb)
            S = S[0]
        else:
            y, S = ops.ssm_chunk(xs.view(T, nh, hd), dt, A, B.view(T, ng, ns), C.view(T, ng, ns), D, dtb,
                                 st.get("ssm"), c["chunk_size"])
        st["ssm"] = S
        out = ops.rmsnorm_gated(y.reshape(T, di), z, self.w(p + "norm.weight"), self.eps, di // ng)
        return F.linear(out, self.w(p + "out_proj.weight"))

    def attention(self, p: str, x: torch.Tensor, st: dict) -> torch.Tensor:
        c = self.cfg
        nq, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        T = x.shape[0]
        q = F.linear(x, self.w(p + "q_proj.weight")).view(T, nq, hd).transpose(0, 1)
        k = F.linear(x, self.w(p + "k_proj.weight")).view(T, nkv, hd).transpose(0, 1)
        v = F.linear(x, self.w(p + "v_proj.weight")).view(T, nkv, hd).transpose(0, 1)
        if "k" in st:
            k, v = torch.cat([st["k"], k], 1), torch.cat([st["v"], v], 1)
        st["k"], st["v"] = k, v
        S = k.shape[1]
        k, v = k.repeat_interleave(nq // nkv, 0), v.repeat_interleave(nq // nkv, 0)
        sc = torch.matmul(q, k.transpose(1, 2)) * hd ** -0.5                  # [nq][T][S]
        mask = torch.arange(S)[None, :] > (torch.arange(T)[:, None] + S - T)
        sc = sc.masked_fill(mask[None], float("-inf"))
        a = torch.softmax(sc, dim=-1, dtype=torch.float32).to(x.dtype)
        o = torch.matmul(a, v).transpose(0, 1).reshape(T, nq * hd)
        return F.linear(o, self.w(p + "o_proj.weight"))

    def moe(self, p: str, x: torch.Tensor, key: str) -> torch.Tensor:
        c = self.cfg
        ids, wt = route(x, self.w(p + "gate.weight"), self.w(p + "gate.e_score_correction_bias"),
                        c["num_experts_per_tok"], c["routed_scaling_factor"])
        self.routes.setdefault(key, []).append((ids.to(torch.int32), wt))
        lat = F.linear(x, self.w(p + "fc1_latent_proj.weight"))
        acc = torch.zeros(x.shape[0], lat.shape[1], dtype=torch.float32)
        for e in sorted(set(ids.reshape(-1).tolist())):                       # ascending id: the combine order
            tok, slot = torch.where(ids == e)
            d = F.linear(relu2(F.linear(lat[tok], self.w(f"{p}experts.{e}.up_proj.weight"))),
                         self.w(f"{p}experts.{e}.down_proj.weight"))
            acc.index_add_(0, tok, d.float() * wt[tok, slot][:, None])
        o = F.linear(acc.to(x.dtype), self.w(p + "fc2_latent_proj.weight"))
        sh = F.linear(relu2(F.linear(x, self.w(p + "shared_experts.up_proj.weight"))),
                      self.w(p + "shared_experts.down_proj.weight"))
        return o + sh

    def block(self, p: str, kind: str, x: torch.Tensor, st: dict, key: str) -> torch.Tensor:
        h = rms_norm(x, self.w(p + "norm.weight"), self.eps)
        if kind == "mamba":
            return x + self.mamba(p + "mixer.", h, st)
        if kind == "attention":
            return x + self.attention(p + "mixer.", h, st)
        return x + self.moe(p + "mixer.", h, key)

    def trunk(self, ids: list[int], states: list[dict]) -> torch.Tensor:
        """Rows `ids` after the cached ones -> the final hidden (after norm_f) [T][H]."""
        x = self.embed[torch.tensor(ids)]
        for i, kind in enumerate(self.cfg["kinds"]):
            x = self.block(f"model.layers.{i}.", kind, x, states[i], f"L{i}")
        return rms_norm(x, self.norm_f, self.eps)

    def mtp(self, next_ids: list[int], h: torch.Tensor, states: list[dict]) -> torch.Tensor:
        """The MTP layer on (embed(next_ids), h) rows -> m [T][H] (after final_layernorm)."""
        m = "mtp.layers."
        e = rms_norm(self.embed[torch.tensor(next_ids)], self.w(m + "0.enorm.weight"), self.eps)
        hh = rms_norm(h, self.w(m + "0.hnorm.weight"), self.eps)
        x = F.linear(torch.cat([e, hh], -1), self.w(m + "0.eh_proj.weight"))
        for j, kind in enumerate(self.cfg["mtp_kinds"]):
            x = self.block(f"{m}{j}.", kind, x, states[j], f"M{j}")
        return rms_norm(x, self.w(f"{m}{len(self.cfg['mtp_kinds']) - 1}.final_layernorm.weight"), self.eps)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.head).float()


def greedy(row: torch.Tensor) -> int:
    m = row.max()
    return int(torch.nonzero(row == m)[0, 0])          # the lower id on an exact tie


def read_ids(spec: str) -> list[int]:
    text = open(spec, encoding="utf-8").read() if os.path.exists(spec) else spec
    return [int(t) for t in re.split(r"[\s,]+", text.strip()) if t]


# ------------------------------------------------------------------------------------------------
# the commands

def cmd_run(a) -> None:
    cpu_bf16.install(a.exact_bf16_gemm)
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    t0 = time.time()
    md = Model(a.snapshot, dtype, a.layers)
    use_mtp = bool(md.cfg["mtp_kinds"]) and not a.no_mtp and a.layers is None
    ids = read_ids(a.prompt)
    T = len(ids)
    st = [dict() for _ in md.cfg["kinds"]]
    mst = [dict() for _ in md.cfg["mtp_kinds"]]
    with torch.no_grad():
        h = md.trunk(ids, st)
        lg = md.logits(h)
        nll = -torch.log_softmax(lg[:-1], -1).gather(1, torch.tensor(ids[1:])[:, None])[:, 0]
        rows_h, rows_lg, toks = [h], [lg], []
        for _ in range(a.gen):
            toks.append(greedy(rows_lg[-1][-1]))
            rows_h.append(md.trunk([toks[-1]], st))
            rows_lg.append(md.logits(rows_h[-1]))
        H = torch.cat(rows_h)
        LG = torch.cat(rows_lg)
        seq = ids + toks
        out = {"tokens": torch.tensor(toks, dtype=torch.int32), "nll": nll.float()}
        r0 = max(0, T - a.logits_tail)
        out["logits"] = LG[r0:].contiguous()
        if use_mtp:
            # row t (0 <= t < len(seq) - 1): x_{t+1} with h_t; the last row's next token is the trunk's own greedy one
            nxt = seq[1:] + [greedy(LG[-1])]
            M = md.mtp(nxt, H, mst)
            ML = md.logits(M)
            out["mtp.logits"] = ML[r0:].contiguous()
            out["mtp.top1"] = torch.tensor([greedy(r) for r in ML], dtype=torch.int32)
            drafts, m, tok = [], M[-1:], greedy(ML[-1])
            for _ in range(a.draft):
                drafts.append(tok)
                m = md.mtp([tok], m, mst)
                tok = greedy(md.logits(m)[0])
            out["drafts"] = torch.tensor(drafts, dtype=torch.int32)
    for key, chunks in md.routes.items():
        out[f"route.ids.{key}"] = torch.cat([c[0] for c in chunks]).contiguous()
        out[f"route.w.{key}"] = torch.cat([c[1] for c in chunks]).contiguous()
    from safetensors.torch import save_file
    meta = {"snapshot": os.path.abspath(a.snapshot), "form": md.ck.form, "dtype": a.dtype, "prompt_len": str(T),
            "logits_row0": str(r0), "gen": str(a.gen), "mtp": str(use_mtp)}
    save_file(out, a.out, metadata=meta)
    mt = f", MTP top-1 at the last row {int(out['mtp.top1'][-1])}, drafts {out['drafts'].tolist()}" if use_mtp else ""
    print(f"run: {md.ck.form} {a.dtype}, {T} prompt rows + {a.gen} generated {toks}, prompt NLL "
          f"{float(nll.mean()) if T > 1 else float('nan'):.4f}{mt}; {time.time() - t0:.1f} s -> {a.out}")


def cos_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.double(), b.double(), dim=-1)


def cmd_compare(a) -> None:
    from safetensors import safe_open
    with safe_open(a.a, "pt") as f, safe_open(a.b, "pt") as g:
        A, B = {k: f.get_tensor(k) for k in f.keys()}, {k: g.get_tensor(k) for k in g.keys()}
    lines, ok = [], True
    for key in ("logits", "mtp.logits"):
        if key not in A or key not in B:
            lines.append(f"{key}: absent in {'both' if key not in A and key not in B else 'a' if key not in A else 'b'}")
            ok = ok and key != "logits" and (key in A) == (key in B) and not (a.mtp and key == "mtp.logits")
            continue
        c = cos_rows(A[key], B[key])
        top = (A[key].argmax(-1) == B[key].argmax(-1)).float().mean()
        lines.append(f"{key}: {A[key].shape[0]} rows, cosine min {float(c.min()):.5f} mean {float(c.mean()):.5f}, "
                     f"top-1 agreement {float(top):.3f}")
        ok = ok and float(c.mean()) >= a.min_mean_cos
    if "tokens" in A and "tokens" in B:
        lines.append(f"tokens: a {A['tokens'].tolist()}, b {B['tokens'].tolist()}")
    for line in lines:
        print("compare:", line)
    print(f"compare: {'OK' if ok else 'FAIL'} (mean cosine bar {a.min_mean_cos})")
    sys.exit(0 if ok else 1)


def cmd_hfcheck(a) -> None:
    """The trunk against transformers' native NemotronHForCausalLM (through the omni wrapper, the checkpoint's remote
    code) on a tiny snapshot, both fp32: the logits must agree to ~1e-4 relative (only the summation orders and the
    attention form differ; the tiny keeps dt above the native class's clamp)."""
    from transformers import AutoModelForCausalLM
    ids = read_ids(a.prompt)
    hf = AutoModelForCausalLM.from_pretrained(a.snapshot, trust_remote_code=True, dtype=torch.float32,
                                              key_mapping={r"^language_model\.backbone\.": "language_model.model."})
    hf = hf.float()        # the inner model is built in llm_config's own dtype (bf16) whatever `dtype` says
    with torch.no_grad():
        want = hf.language_model(input_ids=torch.tensor([ids])).logits[0].float()
        md = Model(a.snapshot, torch.float32)
        got = md.logits(md.trunk(ids, [dict() for _ in md.cfg["kinds"]]))
    c = cos_rows(got, want)
    err = float((got - want).abs().max() / want.abs().max())
    top = float((got.argmax(-1) == want.argmax(-1)).float().mean())
    ok = float(c.min()) > 0.9999 and err < 1e-3 and top == 1.0
    print(f"hfcheck: {len(ids)} rows, cosine min {float(c.min()):.7f}, max |diff| / max |logit| {err:.2e}, top-1 "
          f"agreement {top:.3f} - {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("snapshot")
    r.add_argument("--prompt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gen", type=int, default=8)
    r.add_argument("--draft", type=int, default=4)
    r.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    r.add_argument("--logits-tail", type=int, default=64)
    r.add_argument("--layers", type=int)
    r.add_argument("--no-mtp", action="store_true")
    cpu_bf16.add_argument(r)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--min-mean-cos", type=float, default=0.0)
    c.add_argument("--mtp", action="store_true", help="the files must hold mtp.logits (FAIL when absent)")
    h = sub.add_parser("hfcheck")
    h.add_argument("snapshot")
    h.add_argument("--prompt", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare, "hfcheck": cmd_hfcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
