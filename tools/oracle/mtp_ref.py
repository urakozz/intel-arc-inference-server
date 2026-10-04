#!/usr/bin/env python3
"""A CPU reference of the checkpoint's MTP head (spec 8 P0, plan 8a Task 1).

The wiring is vLLM's, read from the reference image (vllm 0.29.1rc1,
`vllm/model_executor/models/qwen3_5_mtp.py`, which the registry maps
`Qwen3_5MTP` to for this checkpoint's `Qwen3_5ForConditionalGeneration`):

    e = pre_fc_norm_embedding(embed(x[t+1]))          qwen3_5_mtp.py:162
    h = pre_fc_norm_hidden(h_t)                       qwen3_5_mtp.py:163
    y = fc(cat([e, h]))       embed FIRST, hidden second   qwen3_5_mtp.py:164
    y = decoder_layer(y)      one full-attention layer, own KV   qwen3_5_mtp.py:177
    y = mtp.norm(y)           returned hidden, POST-norm   qwen3_5_mtp.py:188
    q = lm_head(y)            shared with the main model

- `h_t` is the main model's hidden AFTER its final norm: the target model's
  returned hidden_states (`qwen3_next.py:729`, `self.norm(hidden_states,
  residual)`) are what the proposer hands the drafter.
- Position: the head's layer runs at position t (the main hidden's), not t+1:
  `llm_base_proposer.py:839` "rotate the input ids and leave the positions
  unchanged"; each further chained step adds 1.
- Chaining: step i+1 takes step i's returned (post-mtp.norm) hidden as `h`
  (`llm_base_proposer.py:759-764`).

The layer is transformers' `Qwen3_5DecoderLayer` (full attention, the main
model's shapes) with its attention replaced by an explicit one so the chained
steps can attend over the depth-1 KV at earlier positions plus their own keys.

Library use (mtp_accept.py, test_mtp_ref.py): `MtpHead.from_snapshot`,
`MtpHead.chain`, `build_main`, `main_forward`. Run inside the reference
container (tools/oracle/run_in_container.sh).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq  # noqa: E402

MTP_SHAPES = {  # docs/03-models.md: 15 tensors, all bf16, 0.849 GB
    "mtp.fc.weight": (5120, 10240),
    "mtp.pre_fc_norm_embedding.weight": (5120,),
    "mtp.pre_fc_norm_hidden.weight": (5120,),
    "mtp.norm.weight": (5120,),
    "mtp.layers.0.input_layernorm.weight": (5120,),
    "mtp.layers.0.post_attention_layernorm.weight": (5120,),
    "mtp.layers.0.self_attn.q_proj.weight": (12288, 5120),
    "mtp.layers.0.self_attn.k_proj.weight": (1024, 5120),
    "mtp.layers.0.self_attn.v_proj.weight": (1024, 5120),
    "mtp.layers.0.self_attn.o_proj.weight": (5120, 6144),
    "mtp.layers.0.self_attn.q_norm.weight": (256,),
    "mtp.layers.0.self_attn.k_norm.weight": (256,),
    "mtp.layers.0.mlp.gate_proj.weight": (17408, 5120),
    "mtp.layers.0.mlp.up_proj.weight": (17408, 5120),
    "mtp.layers.0.mlp.down_proj.weight": (5120, 17408),
}
EMBED = "model.language_model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"

# Wiring variants. "ref" is the reference; the others are controls that must
# lose by a wide margin (plan 8a Review Focus 1, 2).
WIRINGS = {
    "ref":        dict(hidden="post", order="embed_first", pos_shift=0),
    "prenorm":    dict(hidden="pre",  order="embed_first", pos_shift=0),
    "swapped":    dict(hidden="post", order="hidden_first", pos_shift=0),
    "pos_t_plus1": dict(hidden="post", order="embed_first", pos_shift=1),
}


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"oracle_{name}", os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def weight_map(snapshot: str) -> dict[str, str]:
    with open(os.path.join(snapshot, "model.safetensors.index.json"), encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def _checkpoint_name(wm: dict[str, str], n: str) -> str:
    """Qwen3.5 name -> the name this checkpoint ships it under (spec 14: Agnes spells
    `self_attn.` as `global_attn.`, agnes.NAME_MAP run backwards; Qwen3.8 is verbatim)."""
    if n in wm:
        return n
    for ckpt, qwen in _load_module("agnes").NAME_MAP:
        if qwen in n and n.replace(qwen, ckpt, 1) in wm:
            return n.replace(qwen, ckpt, 1)
    raise KeyError(n)


def read_tensors(snapshot: str, names) -> dict[str, torch.Tensor]:
    """By Qwen3.5 name, whatever the checkpoint calls it (returned under the Qwen name)."""
    wm = weight_map(snapshot)
    out = {}
    by_file: dict[str, list[tuple[str, str]]] = {}
    for n in names:
        c = _checkpoint_name(wm, n)
        by_file.setdefault(wm[c], []).append((n, c))
    for fn, ns in by_file.items():
        with safe_open(os.path.join(snapshot, fn), framework="pt", device="cpu") as h:
            for n, c in ns:
                out[n] = h.get_tensor(c)
    return out


def text_config(snapshot: str):
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    agnes = _load_module("agnes")
    if agnes.is_agnes(raw):   # spec 14: the config class is remote code; translate it
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        tc = Qwen3_5TextConfig(**agnes.translate_text_config(raw)[0])
    else:
        tc = AutoConfig.from_pretrained(snapshot).get_text_config()
    tc._attn_implementation = "eager"
    return tc


class MtpHead:
    """The head. `embed` and `lm_head` are the main model's (shared)."""

    def __init__(self, tc, mtp: dict[str, torch.Tensor], embed: torch.Tensor, lm_head: torch.Tensor):
        self.tc = tc
        fa = tc.layer_types.index("full_attention")
        with torch.device("meta"):
            layer = mq.Qwen3_5DecoderLayer(tc, fa)
        sd = {k[len("mtp.layers.0."):]: v for k, v in mtp.items() if k.startswith("mtp.layers.0.")}
        layer.load_state_dict(sd, strict=True, assign=True)
        self.layer = layer.eval()
        self.attn = layer.self_attn

        def norm(key):
            n = mq.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
            n.weight = torch.nn.Parameter(mtp[key], requires_grad=False)
            return n.eval()

        self.pre_e = norm("mtp.pre_fc_norm_embedding.weight")
        self.pre_h = norm("mtp.pre_fc_norm_hidden.weight")
        self.norm = norm("mtp.norm.weight")
        self.fc = mtp["mtp.fc.weight"]          # [5120, 10240]: y = x @ fc^T
        self.embed = embed
        self.lm_head = lm_head
        self.rotary = mq.Qwen3_5TextRotaryEmbedding(tc)

    @classmethod
    def from_snapshot(cls, snapshot: str, tc=None, embed=None, lm_head=None):
        tc = tc or text_config(snapshot)
        need = list(MTP_SHAPES) + ([EMBED] if embed is None else []) + ([LM_HEAD] if lm_head is None else [])
        t = read_tensors(snapshot, need)
        embed = t.pop(EMBED) if embed is None else embed
        lm_head = t.pop(LM_HEAD) if lm_head is None else lm_head
        return cls(tc, t, embed, lm_head)

    def logits(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.lm_head.t()

    def _qkv(self, x: torch.Tensor, pos: torch.Tensor):
        """x [T, H] (already input-normed) -> q [Hq, T, D], gate [T, Hq*D], k, v [Hkv, T, D], roped."""
        a, T, D = self.attn, x.shape[0], self.attn.head_dim
        q, gate = torch.chunk(a.q_proj(x).view(T, -1, D * 2), 2, dim=-1)
        gate = gate.reshape(T, -1)
        q = a.q_norm(q).transpose(0, 1)
        k = a.k_norm(a.k_proj(x).view(T, -1, D)).transpose(0, 1)
        v = a.v_proj(x).view(T, -1, D).transpose(0, 1)
        pid = pos.view(1, 1, T).expand(3, 1, T)
        cos, sin = self.rotary(x, pid)                      # [1, T, rot]
        q, k = mq.apply_rotary_pos_emb(q[None], k[None], cos, sin)
        return q[0], gate, k[0], v

    def step(self, e_ids, h, pos, base_kv, extra_kv, hidden_first=False):
        """One head step for T independent rows.

        e_ids [T] token ids, h [T, H] the hidden, pos [T] positions.
        base_kv: (K1, V1) [Hkv, T, D] depth-1 keys; row t attends to base rows 0..t.
        extra_kv: list of (k, v) [Hkv, T, D], row t's own chained keys (earlier depths).
        Returns (y [T,H] post-mtp.norm hidden, (k, v) of this step).
        """
        e = self.pre_e(self.embed[e_ids])
        hn = self.pre_h(h)
        x = torch.cat([hn, e] if hidden_first else [e, hn], dim=-1) @ self.fc.t()
        L, a = self.layer, self.attn
        resid = x
        xn = L.input_layernorm(x)
        q, gate, k, v = self._qkv(xn, pos)
        T, g = x.shape[0], a.num_key_value_groups
        if base_kv is None:                                  # depth 1: its own keys ARE the base
            base_kv = (k, v)
            own_in_base = True
        else:
            own_in_base = False
        K1, V1 = base_kv
        qf = q.float()
        K1r = K1.float().repeat_interleave(g, 0)             # [Hq, T, D]
        V1r = V1.float().repeat_interleave(g, 0)
        s = (qf @ K1r.transpose(1, 2)) * a.scaling           # [Hq, T, T]
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        s = s.masked_fill(~mask, float("-inf"))
        cols_k = [] if own_in_base else [(k, v)]
        chain = list(extra_kv) + cols_k
        ex = []
        for kk, vv in chain:                                 # per-row own keys: [Hq, T]
            ex.append((qf * kk.float().repeat_interleave(g, 0)).sum(-1) * a.scaling)
        if ex:
            s = torch.cat([s, torch.stack(ex, -1)], -1)      # [Hq, T, T + E]
        p = torch.softmax(s, -1)
        o = p[..., :T] @ V1r
        for j, (kk, vv) in enumerate(chain):
            o = o + p[..., T + j:T + j + 1] * vv.float().repeat_interleave(g, 0)
        o = o.to(x.dtype).transpose(0, 1).reshape(T, -1)
        o = a.o_proj(o * torch.sigmoid(gate))
        x = resid + o
        x = x + L.mlp(L.post_attention_layernorm(x))
        return self.norm(x), (k, v)

    @torch.no_grad()
    def chain(self, h_main, ids_next, pos, depth, feed_ids=None, hidden_first=False):
        """Run the head `depth` steps for every row t.

        h_main [T, H]: the main hidden at row t; ids_next [T]: x[t+1] (step 1's token).
        feed_ids: list of [T] tensors, the token fed at step i+1 (i >= 1) per row;
        None -> the previous step's argmax (greedy chain).
        Returns a list of (logits [T, V] bf16, y [T, H]) per depth.
        """
        out = []
        y, kv1 = self.step(ids_next, h_main, pos, None, [], hidden_first)
        out.append((self.logits(y), y))
        extra = []
        for d in range(1, depth):
            tok = feed_ids[d - 1] if feed_ids is not None else out[-1][0].argmax(-1)
            y, kv = self.step(tok, y, pos + d, kv1, extra, hidden_first)
            extra.append(kv)
            out.append((self.logits(y), y))
        return out


def build_main(snapshot: str, stream: bool = False):
    """The main model exactly as dump.py builds it (dequantised, strict, eager).

    stream: dump.py --stream's layer-streamed weights (stream.py), for a host with less
    RAM than the bf16 model (Agnes on the Mac); the forward is the same."""
    dump = _load_module("dump")
    gs = dump.check_quant_config(snapshot)
    tc = text_config(snapshot)
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    agnes = _load_module("agnes")
    with torch.device("meta"):
        model = dump.Qwen3_5ForCausalLM(tc)
        if agnes.is_agnes(raw):   # spec 14: the parallel FFN, unfolded, as dump.py
            agnes.attach_parallel_ffn(model, tc, agnes.translate_text_config(raw)[1])
    if stream:
        dump.stream_weights(model, snapshot, gs, tc)
        return model.eval(), tc
    sd = dump.build_state_dict(snapshot, gs)
    model.load_state_dict(sd, strict=True, assign=True)
    del sd
    model.model.rotary_emb = type(model.model.rotary_emb)(tc)
    return model.eval(), tc


@torch.no_grad()
def main_forward(model, ids):
    """Teacher-forced forward: logits [T, V] bf16, hidden before and after the final norm."""
    cap = {}

    def fn(_m, args, out):
        cap["pre"] = args[0].detach()[0]
        cap["post"] = out.detach()[0]
    hnd = model.model.norm.register_forward_hook(fn)
    try:
        out = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False, logits_to_keep=0)
    finally:
        hnd.remove()
    return out.logits[0], cap["pre"], cap["post"]


def wiring_inputs(pre, post, w):
    return pre if WIRINGS[w]["hidden"] == "pre" else post


@torch.no_grad()
def dump(snapshot: str, prompts_dir: str, cont_dir: str, out_dir: str, names, n_rows: int,
         stream: bool = False) -> None:
    """Plan 8b M1's reference: per golden prompt, the head's depth-1 logits at the rows the
    engine drafts from. With prompt length n and ids = prompt + cont[:n_rows + 1], row i
    (i < n_rows) is the head on (post-norm h[n-1+i], ids[n+i]) at position n-1+i, attending
    over depth-1 keys of every earlier row -- exactly what the engine's first draft sees
    after prefilling prompt + cont[:i] (teacher forced). Written as one safetensors file per
    prompt: `logits` fp32 [n_rows][V], `pos` int32 [n_rows], `next` int32 [n_rows]
    (the token fed at each row)."""
    from safetensors.torch import save_file
    t0 = time.time()
    model, tc = build_main(snapshot, stream)
    head = MtpHead.from_snapshot(snapshot, tc, model.model.embed_tokens.weight, model.lm_head.weight)
    print(f"loaded {time.time() - t0:.0f}s", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        prompt = [int(x) for x in open(os.path.join(prompts_dir, f"{name}.ids")).read().split()]
        cont = [int(x) for x in open(os.path.join(cont_dir, f"{name}.cont256.ids")).read().split()]
        ids = prompt + cont[: n_rows + 1]
        n, T = len(prompt), len(ids)
        _, _, post = main_forward(model, ids)
        ids_t = torch.tensor(ids)
        rows = torch.arange(0, T - 1)
        (lg, _), = head.chain(post[: T - 1], ids_t[1:T], rows, 1)
        sel = torch.arange(n - 1, n - 1 + n_rows)
        save_file({"logits": lg[sel].float().contiguous(),
                   "pos": sel.to(torch.int32).contiguous(),
                   "next": ids_t[sel + 1].to(torch.int32).contiguous()},
                  os.path.join(out_dir, f"{name}.mtp.safetensors"))
        print(f"{name}: n={n} rows {n - 1}..{n - 2 + n_rows} dumped ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    import argparse
    if len(sys.argv) > 1 and sys.argv[1] == "--dump":
        ap = argparse.ArgumentParser(description="dump plan 8b M1's reference head logits")
        ap.add_argument("--dump", action="store_true")
        ap.add_argument("snapshot")
        ap.add_argument("--prompts", required=True, help="dir with <name>.ids")
        ap.add_argument("--cont", required=True, help="dir with <name>.cont256.ids (engine greedy)")
        ap.add_argument("--out", required=True)
        ap.add_argument("--names", default="code,prose,cjk")
        ap.add_argument("--rows", type=int, default=32)
        ap.add_argument("--stream", action="store_true", help="layer-streamed main model (dump.py --stream)")
        a = ap.parse_args()
        dump(a.snapshot, a.prompts, a.cont, a.out, a.names.split(","), a.rows, a.stream)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--ids", required=True, help="prompt ids file")
    ap.add_argument("--cont", required=True, help="continuation ids file (greedy)")
    ap.add_argument("--n-cont", type=int, default=32)
    args = ap.parse_args()
    t0 = time.time()
    print(f"torch threads {torch.get_num_threads()}")
    prompt = [int(x) for x in open(args.ids).read().split()]
    cont = [int(x) for x in open(args.cont).read().split()][: args.n_cont]
    ids = prompt + cont
    model, tc = build_main(args.snapshot)
    head = MtpHead.from_snapshot(args.snapshot, tc, model.model.embed_tokens.weight, model.lm_head.weight)
    print(f"loaded {time.time() - t0:.0f}s")
    logits, pre, post = main_forward(model, ids)
    g = logits.float().argmax(-1)
    T, c0 = len(ids), len(prompt)
    rows = torch.arange(0, T - 2)
    ids_t = torch.tensor(ids)
    for w, cfg in WIRINGS.items():
        h = wiring_inputs(pre, post, w)
        (lg, _), = head.chain(h[: T - 2], ids_t[1:T - 1], rows + cfg["pos_shift"], 1,
                              hidden_first=cfg["order"] == "hidden_first")
        hit = (lg.float().argmax(-1) == g[1:T - 1])
        cont_rows = rows >= c0 - 1
        print(f"wiring {w:<12} top1 vs main greedy: all rows {hit.float().mean():.3f} "
              f"({int(hit.sum())}/{T - 2}), continuation rows {hit[cont_rows].float().mean():.3f} "
              f"({int(hit[cont_rows].sum())}/{int(cont_rows.sum())})")
    print(f"wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
