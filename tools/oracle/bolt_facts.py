#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 25a Task 4: the Nemotron 3.5 Bolt 74B-A9B facts - the rounding chain cited into the remote code (R) and the
native 5.19.0 class (N), the checkpoint from its small files, its shard headers and a few tensors read by HTTP range,
tokens, the template, and torch's CPU behaviours the reference depends on. docs/probe-bolt-2026-10-21.md quotes it.

    bolt_facts.py <snapshot dir> [--section cite,checkpoint,bytes,quant,tokens,template,prompts,ties,gemm,sum9]
                  [--bf16 <BF16 snapshot dir>] [--cache DIR] [--headers] [--sample] [--write-fixtures]

<snapshot> is our export's pinned revision in the HF cache (small files only; the served checkpoint,
urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ @ 8932aa8d). Shard headers and tensor data come from
the local shard when it exists, else an HTTP range request to huggingface.co at the pinned revision (`--headers`
allows the header requests, ~2 MB a shard; `--sample` the data reads: named small tensors and a few scale /
qzeros / g_idx rows, a few MB in all), cached under --cache (default: <main checkout>/oracle-out-bolt-facts, outside
the tree's tracked files). Without --headers / --sample those sections print what the cache already holds.

Sections:
  cite        every rounding point of spec 25 §2 located by an anchor string in R (`R:line`, the pinned blob), RP / RC
              (the puzzle / base configuration files) and N (transformers' native modeling_nemotron_h.py, `N:line`)
  checkpoint  the index's names grouped (layer / expert -> N), dtypes and shapes from every shard header, the
              per-layer (type, E, I) table, the vision / projector names, the parameter count; with --bf16 the BF16
              original's names (the `backbone` trunk) and its header dtypes
  bytes       spec 25 §3's parameter counts and int4 g64 bytes re-derived from the shapes
  quant       quantization_config (`dynamic`'s `+:` rules / `extra_config` quoted); sampled tensors: qzeros, g_idx (the GPTQ
              g128 spelling's), the scales' sign / subnormal
              / zero counts, A_log / D / dt_bias / e_score_correction_bias ranges and the dt-clamp census (per Mamba
              layer the heads with softplus(dt_bias) < 0.001, R's fallback clamp)
  tokens      tokenizer.json (vocab, the added tokens and their special flags), both generation configs,
              tokenizer_config's fix_mistral_regex and what 5.19.0 does with it
  template    chat_template.jinja sha256, the features minja must render, renders through apply_chat_template;
              --write-fixtures writes them to tests/server/bolt_template_cases.json (25e's byte-identity cases)
  prompts     tests/golden/prompts/{prose,code,long}.txt under Bolt's tokenizer (no BOS)
  ties        torch's CPU topk(sorted=False) on exact ties planted at the 9th / 10th of 240 / 496 fp32 values
  gemm        torch's CPU fp32 and bf16 GEMM at the router's shape: are rows independent of the batch's row count?
  sum9        torch's fp32 sum over the 9 selected scores (R:1049) against fixed orders (the twin's choice)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import urllib.request  # noqa: E402

SECTIONS = ("cite", "checkpoint", "bytes", "quant", "tokens", "template", "prompts", "ties", "gemm", "sum9")
REPO = os.path.dirname(os.path.dirname(_HERE))
OURS = ("urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ", "8932aa8d1844f91056c5b9433448075a842b9156")
NVIDIA_BF16 = ("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16", "567021037ff88364143e24ea05e6abd0f72086ff")
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "U8": 1, "F64": 8}


def main_checkout() -> str:
    """The main checkout of a worktree (data dirs live there), else the repo itself."""
    p = REPO
    m = re.match(r"(.*)/\.claude/worktrees/[^/]+$", p)
    return m.group(1) if m else p


# ------------------------------------------------------------------------------------------------
# bytes: the local snapshot, else range requests at the pinned revision (cached)

class Source:
    def __init__(self, where: str, repo_rev: tuple[str, str], cache: str, allow_headers=False, allow_data=False):
        self.dir, (self.repo, self.rev) = where, repo_rev
        self.cache = os.path.join(cache, self.repo.replace("/", "--"))
        self.allow_headers, self.allow_data = allow_headers, allow_data
        self._h: dict[str, dict] = {}
        self.how: collections.Counter = collections.Counter()
        self.fetched = 0

    def _get(self, fn: str, a: int | None = None, b: int | None = None) -> bytes:
        req = urllib.request.Request(f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{fn}")
        if a is not None:
            req.add_header("Range", f"bytes={a}-{b}")
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.fetched += len(data)
        return data

    def local(self, fn: str) -> str | None:
        p = os.path.join(self.dir, fn)
        return p if os.path.exists(p) else None

    def small(self, fn: str) -> bytes:
        p = self.local(fn)
        return open(p, "rb").read() if p else self._get(fn)

    def json(self, fn: str):
        return json.loads(self.small(fn))

    def header(self, shard: str) -> dict | None:
        if shard in self._h:
            return self._h[shard]
        p = self.local(shard)
        if p:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            self.how["local"] += 1
        else:
            cp = os.path.join(self.cache, shard + ".header.json")
            if os.path.exists(cp):
                h = json.load(open(cp))
                self.how["cached range request"] += 1
            elif self.allow_headers:
                n = struct.unpack("<Q", self._get(shard, 0, 7))[0]
                h = json.loads(self._get(shard, 8, 8 + n - 1))
                h["__header_len__"] = n
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                json.dump(h, open(cp, "w"))
                self.how["range request"] += 1
            else:
                return None
        self._h[shard] = h
        return h

    def tensor_bytes(self, shard: str, name: str, a: int = 0, b: int | None = None) -> bytes | None:
        """Bytes [a, b) of tensor `name`'s data (the whole tensor by default)."""
        h = self.header(shard)
        if h is None:
            return None
        e = h[name]
        n = h.get("__header_len__")
        lo, hi = e["data_offsets"]
        b = hi - lo if b is None else b
        p = self.local(shard)
        if p:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                f.seek(8 + n + lo + a)
                return f.read(b - a)
        key = hashlib.sha256(f"{shard}|{name}|{a}|{b}".encode()).hexdigest()[:24]
        cp = os.path.join(self.cache, "data", key + ".bin")
        if os.path.exists(cp):
            return open(cp, "rb").read()
        if not self.allow_data:
            return None
        if n is None:
            n = struct.unpack("<Q", self._get(shard, 0, 7))[0]
        data = self._get(shard, 8 + n + lo + a, 8 + n + lo + b - 1)
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        open(cp, "wb").write(data)
        return data


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def as_tensor(raw: bytes, dtype: str, shape):
    import torch
    td = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I32": torch.int32}[dtype]
    return torch.frombuffer(bytearray(raw), dtype=td).reshape(shape)


# ------------------------------------------------------------------------------------------------
# cite

CITES = [  # (what, file, anchor regex)
    ("block: residual, norm in the norm weight's dtype, mixer, h = residual + mixer (bf16 add)", "R",
     r"hidden_states = self\.norm\(hidden_states\.to\(dtype=self\.norm\.weight\.dtype\)\)"),
    ("block: residual_in_fp32 false keeps the bf16 residual", "R", r"if self\.residual_in_fp32:"),
    ("block: the add", "R", r"hidden_states = residual \+ hidden_states"),
    ("RMSNorm: fp32 variance", "R", r"variance = hidden_states\.pow\(2\)\.mean\(-1, keepdim=True\)"),
    ("RMSNorm: w (fp32) * x^, ONE rounding", "R", r"return \(self\.weight\.to\(torch\.float32\) \* hidden_states\)\.to\(input_dtype\)"),
    ("mixer dispatch: the kernel path only on CUDA", "R", r'if is_fast_path_available and "cuda" in self\.in_proj\.weight\.device\.type'),
    ("rmsnorm_fn required at import", "R", r"from mamba_ssm\.ops\.triton\.layernorm_gated import rmsnorm_fn"),
    ("gated norm: rmsnorm_fn, norm_before_gate False", "R", r"norm_before_gate=False$"),
    ("gated norm: group_size = d_inner / n_groups (1024)", "R", r"group_size=self\.intermediate_size // self\.n_groups"),
    ("in_proj: z | xBC | dt (d_mlp 0)", "R", r"\[self\.intermediate_size, self\.conv_dim, self\.num_heads\],"),
    ("kernel path decode: causal_conv1d_update on the raw column", "R", r"hidden_states_B_C = causal_conv1d_update\("),
    ("kernel path: A = -exp(A_log) fp32", "R", r"A = -torch\.exp\(self\.A_log\.float\(\)\)"),
    ("kernel path decode: selective_state_update, dt_softplus, no dt_limit", "R", r"hidden_states = selective_state_update\("),
    ("kernel path prefill: dt_limit = time_step_limit (0, inf)", "R", r'dt_limit_kwargs = \{\} if self\.time_step_limit is None else \{"dt_limit": self\.time_step_limit\}'),
    ("kernel path prefill: the conv state holds the RAW xBC (padded to 4)", "R", r"hidden_states_B_C_t, \(self\.conv_kernel_size - hidden_states_B_C_t\.shape\[-1\], 0\)"),
    ("kernel path prefill: causal_conv1d_fn with SiLU", "R", r"hidden_states_B_C = causal_conv1d_fn\("),
    ("kernel path prefill: mamba_chunk_scan_combined", "R", r"scan_output, ssm_state = mamba_chunk_scan_combined\("),
    ("kernel path: y -> gated norm -> out_proj", "R", r"scan_output = self\.norm\(scan_output, gate\)"),
    ("fallback decode: conv products in the cache dtype, torch.sum", "R", r"hidden_states = torch\.sum\(conv_state\.to\(projected_states\.device\) \* self\.conv1d\.weight\[:, 0, :\], dim=-1\)"),
    ("fallback decode: dt + dt_bias in dt's dtype (bf16)", "R", r"dt = torch\.nn\.functional\.softplus\(dt \+ dt_bias\.to\(dt\.dtype\)\)"),
    ("fallback: the dt clamp at time_step_min", "R", r"dt = torch\.clamp\(dt, self\.time_step_min\)"),
    ("fallback decode: dB = dt B (bf16)", "R", r"dB = dt\[\.\.\., None\] \* B\[\.\.\., None, :\]"),
    ("fallback decode: the state cast to C's dtype before C S", "R", r"ssm_states = cache_params\.ssm_states\[self\.layer_idx\]\.to\(C\.dtype\)"),
    ("fallback decode: y + x D in y's dtype", "R", r"y = \(y \+ hidden_states \* D\)\.to\(y\.dtype\)"),
    ("fallback prefill: a bf16 conv1d, SiLU after (two roundings)", "R", r"hidden_states = self\.act\(self\.conv1d\(hidden_states\)\.transpose\(1,2\)\)"),
    ("fallback prefill: softplus(dt + dt_bias)", "R", r"dt = nn\.functional\.softplus\(dt \+ self\.dt_bias\)"),
    ("fallback prefill: state passing summed over dim 2 (the TARGET chunk - wrong past one chunk)", "R", r"result = \(decay_chunk\[\.\.\., None, None\] \* states_permuted\[:, :, None, \.\.\.\]\)\.sum\(dim=2\)"),
    ("fallback prefill: y fp32 into the gated norm", "R", r"scan_output = self\.norm\(y, gate\)"),
    ("MoE: final buffer in the weights' dtype (fp32)", "R", r"final_hidden_states = torch\.zeros_like\(hidden_states, dtype=topk_weights\.dtype\)"),
    ("MoE: experts in ascending id", "R", r"for expert_idx in range\(len\(self\.experts\)\):"),
    ("MoE: bf16 expert out x fp32 w (fp32 product)", "R", r"weighted_output = expert_output \* expert_weights\.unsqueeze\(-1\)"),
    ("MoE: index_add_ (fp32), then one rounding", "R", r"final_hidden_states\.index_add_\(0, token_indices, weighted_output\)"),
    ("MoE: the dummy forward of an unrouted expert (adds +0)", "R", r"dummy_out = expert\(torch\.zeros_like"),
    ("MoE: the combine rounded to the hidden dtype", "R", r"return final_hidden_states\.type\(hidden_states\.dtype\)"),
    ("MoE: fc1 latent, moe, fc2 latent", "R", r"hidden_states = self\.fc2_latent_proj\(hidden_states\)"),
    ("MoE: + shared expert on the NORMED input (residuals = the mixer's input)", "R", r"hidden_states = hidden_states \+ self\.shared_experts\(residuals\)"),
    ("expert MLP: down(act(up(x))), relu2, non-gated", "R", r"return self\.down_proj\(self\.act_fn\(self\.up_proj\(x\)\)\)"),
    ("routed experts in the latent (moe_latent_size)", "R", r"input_size = self\.hidden_size if not use_latent_size else config\.moe_latent_size"),
    ("router: fp32 logits from widened x and W", "R", r"router_logits = F\.linear\(hidden_states\.type\(torch\.float32\), self\.weight\.type\(torch\.float32\)\)"),
    ("router: sigmoid fp32", "R", r"scores = router_logits\.sigmoid\(\)"),
    ("router: sel = s + e_score_correction_bias", "R", r"scores_for_choice = scores\.view\(-1, self\.n_routed_experts\) \+ self\.e_score_correction_bias\.unsqueeze\(0\)"),
    ("router: top-9, sorted=False", "R", r"topk_indices = torch\.topk\(scores_for_choice, k=self\.top_k, dim=-1, sorted=False\)\[1\]"),
    ("router: w = s gathered", "R", r"topk_weights = scores\.gather\(1, topk_indices\)"),
    ("router: / (sum + 1e-20)", "R", r"denominator = topk_weights\.sum\(dim=-1, keepdim=True\) \+ 1e-20"),
    ("router: x routed_scaling_factor (5.0)", "R", r"topk_weights = topk_weights \* self\.routed_scaling_factor"),
    ("router: the bias kept fp32", "R", r"self\.e_score_correction_bias\.data = self\.e_score_correction_bias\.data\.to\(torch\.float32\)"),
    ("attention: scale head_dim^-0.5", "R", r"self\.scaling = self\.head_dim \*\* -0\.5"),
    ("attention: q, k, v, o without bias (attention_bias false), no norm, no RoPE", "R", r"self\.q_proj = nn\.Linear\(self\.hidden_size, self\.num_heads \* self\.head_dim, bias=config\.attention_bias\)"),
    ("eager: Q K^T x scaling", "R", r"attn_weights = torch\.matmul\(query, key_states\.transpose\(2, 3\)\) \* scaling"),
    ("eager: softmax fp32 then the query dtype", "R", r"attn_weights = F\.softmax\(attn_weights, dim=-1, dtype=torch\.float32\)\.to\(query\.dtype\)"),
    ("model: hidden_states tuple holds block INPUTS", "R", r"all_hidden_states \+= \(hidden_states,\)"),
    ("model: norm_f, appended last", "R", r"hidden_states = self\.norm_f\(hidden_states\)"),
    ("model: has_previous_state set after a forward", "R", r"past_key_values\.has_previous_state = True"),
    ("cache: seq length from an attention block", "R", r"layer_idx = self\.transformer_layers\[0\] if layer_idx not in self\.transformer_layers else layer_idx"),
    ("cache: built in the model's dtype by generate", "R", r"self\.config, input_ids\.shape\[0\], dtype=self\.dtype, device=self\.device"),
    ("head: lm_head(h in the head's dtype).float()", "R", r"logits = self\.lm_head\(hidden_states\.to\(self\.lm_head\.weight\.dtype\)\)\.float\(\)"),
    ("puzzle: per-layer config from block_configs", "RP", r"def get_nemotron_h_config_for_layer"),
    ("puzzle: block config merged into the global dict", "RP", r"config_dict\.update\(block_config\)"),
    ("config: mamba_dt_limit default (0, inf)", "RC", r'mamba_dt_limit=\(0\.0, float\("inf"\)\)'),
    ("N: time_step_limit = (time_step_min, inf) - N clamps the chunk path", "N", r'self\.time_step_limit = \(config\.time_step_min, float\("inf"\)\)'),
    ("N: the chunk path's clamp", "N", r"dt = torch\.clamp\(dt, min=dt_limit\[0\], max=dt_limit\[1\]\)"),
    ("N: state passing over the source chunk (the fix R lacks)", "N", r"new_states = torch\.einsum\(\"bzch,bzhpn->bchpn\", decay_chunk, states\)"),
    ("N: the router (same chain)", "N", r"denominator = topk_weights\.sum\(dim=-1, keepdim=True\) \+ 1e-20"),
]


def transformers_dir() -> str:
    spec = importlib.util.find_spec("transformers")
    if spec is None or not spec.origin:
        raise SystemExit("cite: transformers is not importable (the 5.19.0 site first on PYTHONPATH)")
    return os.path.dirname(spec.origin)


def sec_cite(snap: str) -> None:
    td = transformers_dir()
    v = re.search(r'__version__ = "([^"]+)"', open(os.path.join(td, "__init__.py"), encoding="utf-8").read()).group(1)
    files = {"R": os.path.join(snap, "modeling_nemotron_h.py"), "RP": os.path.join(snap, "configuration_nemotron_h_puzzle.py"),
             "RC": os.path.join(snap, "configuration_nemotron_h.py"),
             "N": os.path.join(td, "models", "nemotron_h", "modeling_nemotron_h.py")}
    rb = open(files["R"], "rb").read()
    print(f"## cite: R = {files['R']} (sha256 {sha256(rb)}, git blob "
          f"{hashlib.sha1(b'blob %d' % len(rb) + bytes([0]) + rb).hexdigest()}); N = transformers {v}")
    print("| point | where |\n|---|---|")
    bad = 0
    for what, f, pat in CITES:
        lines = open(files[f], encoding="utf-8").read().split("\n")
        hits = [i + 1 for i, ln in enumerate(lines) if re.search(pat, ln)]
        bad += not hits
        print(f"| {what} | {f}:{','.join(map(str, hits)) if hits else 'NOT FOUND'} |")
    print(f"{len(CITES) - bad} of {len(CITES)} anchors found")


# ------------------------------------------------------------------------------------------------
# checkpoint, bytes

def natural(x: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", x)]


def group(n: str) -> str:
    n = re.sub(r"\.layers\.\d+\.", ".layers.L.", n)
    n = re.sub(r"\.experts\.\d+\.", ".experts.E.", n)
    return re.sub(r"(vision_model|vision_projector|mlp1)\..*", r"\1.*", n)


def read_headers(src: Source, wm: dict) -> tuple[dict, dict, list]:
    shapes, dtypes, missing = {}, {}, []
    for shard in sorted(set(wm.values())):
        h = src.header(shard)
        if h is None:
            missing.append(shard)
            continue
        for k, e in h.items():
            if not k.startswith("__"):
                shapes[k], dtypes[k] = e["shape"], e["dtype"]
    return shapes, dtypes, missing


def layer_table(llm: dict) -> list[tuple]:
    rows = []
    for i, b in enumerate(llm["block_configs"]):
        rows.append((i, b["block_type"], b.get("n_routed_experts"), b.get("moe_intermediate_size")))
    return rows


def sec_checkpoint(src: Source, llm: dict, bf16: Source | None) -> dict:
    idx = src.json("model.safetensors.index.json")
    wm = idx["weight_map"]
    print(f"## checkpoint: {src.repo}@{src.rev[:8]}: {len(wm)} tensors in {len(set(wm.values()))} files; index metadata "
          f"{idx.get('metadata')}")
    shapes, dtypes, missing = read_headers(src, wm)
    print(f"headers: {dict(src.how)}" + (f"; NOT READ: {missing} (--headers)" if missing else ""))
    groups = collections.OrderedDict()
    for k in sorted(wm, key=natural):
        groups.setdefault(group(k), []).append(k)
    print("| group | count | dtype | shapes |\n|---|---:|---|---|")
    for g, ks in groups.items():
        ds = sorted({dtypes.get(k, "?") for k in ks})
        ss = sorted({str(shapes.get(k, "?")) for k in ks})
        print(f"| `{g}` | {len(ks)} | {', '.join(ds)} | {'; '.join(ss[:6])}{' ...' if len(ss) > 6 else ''} |")
    vision = [k for k in wm if not k.startswith("language_model.")]
    extra = [k for k in vision if wm[k] == "model_extra_tensors.safetensors"]
    print(f"vision / projector names: {len(vision)} ({len(extra)} in model_extra_tensors.safetensors); language "
          f"names {len(wm) - len(vision)}")
    numel = sum(math.prod(s) for k, s in shapes.items() if k.startswith("language_model."))
    tb = sum(math.prod(s) * DTYPE_BYTES[dtypes[k]] for k, s in shapes.items())
    print(f"stored language-model elements (headers): {numel:,}; tensor bytes (all, headers): {tb:,}")
    rows = layer_table(llm)
    pairs = collections.Counter((e, i) for _, t, e, i in rows if t == "moe")
    letter = {"mamba": "M", "moe": "E", "attention": "*"}
    print("per-layer (type, E, I): " + " ".join(f"{i}:{letter[t]}{'' if e is None else f'({e},{ii})'}"
                                               for i, t, e, ii in rows))
    print("pattern: " + "".join(letter[t] for _, t, *_ in rows))
    print(f"MoE (E, I) pairs: {len(pairs)} distinct - " + ", ".join(f"({e}, {i}) x{n}" for (e, i), n in sorted(pairs.items())))
    print(f"E counts: {dict(sorted(collections.Counter(e for _, t, e, _ in rows if t == 'moe').items()))}; I counts: "
          f"{dict(sorted(collections.Counter(i for _, t, _, i in rows if t == 'moe').items()))}; experts in all "
          f"{sum(e for _, t, e, _ in rows if t == 'moe'):,}; attention at {[i for i, t, *_ in rows if t == 'attention']}")
    out = {"wm": wm, "shapes": shapes, "dtypes": dtypes}
    if bf16 is not None:
        bidx = bf16.json("model.safetensors.index.json")
        bwm = bidx["weight_map"]
        bshapes, bdtypes, bmiss = read_headers(bf16, bwm)
        lang = [k for k in bwm if k.startswith("language_model.")]
        print(f"## the BF16 original {bf16.repo}@{bf16.rev[:8]}: {len(bwm)} tensors in {len(set(bwm.values()))} shards, "
              f"index total_size {bidx.get('metadata', {}).get('total_size')}; language names {len(lang)} (trunk "
              f"{'backbone' if any(k.startswith('language_model.backbone.') for k in lang) else 'model'}), vision "
              f"{len(bwm) - len(lang)}; headers {dict(bf16.how)}" + (f"; NOT READ: {len(bmiss)} shards" if bmiss else ""))
        if bdtypes:
            dc = collections.Counter((group(k).split(".")[-1] if "." in k else k, bdtypes[k]) for k in lang if k in bdtypes)
            print("BF16 original dtypes by tensor kind: " + ", ".join(f"{a} {d} x{n}" for (a, d), n in sorted(dc.items())))
            bnum = sum(math.prod(bshapes[k]) for k in lang if k in bshapes)
            print(f"BF16 original language-model parameters (headers): {bnum:,}")
            out["bf16_params"] = bnum
    return out


def sec_bytes(info: dict, llm: dict) -> None:
    """Spec 25 §3 re-derived from the config (every linear's logical [N][K]) - int4 g64 at 0.53125 B a weight."""
    H, V = llm["hidden_size"], llm["vocab_size"]
    di = llm["mamba_num_heads"] * llm["mamba_head_dim"]
    conv = di + 2 * llm["n_groups"] * llm["ssm_state_size"]
    nh = llm["mamba_num_heads"]
    qd, kvd = llm["num_attention_heads"] * llm["head_dim"], llm["num_key_value_heads"] * llm["head_dim"]
    lat, sh = llm["moe_latent_size"], llm["moe_shared_expert_intermediate_size"]
    parts = collections.OrderedDict((k, 0) for k in ("mamba", "attention", "moe_dense", "experts", "routers", "small"))
    active = 0
    for b in llm["block_configs"]:
        t = b["block_type"]
        if t == "mamba":
            n = H * (di + conv + nh) + di * H
            parts["mamba"] += n
            parts["small"] += conv * 4 + conv + 3 * nh + di + H
            active += n
        elif t == "attention":
            n = H * qd + 2 * H * kvd + qd * H
            parts["attention"] += n
            parts["small"] += H
            active += n
        else:
            E, I = b["n_routed_experts"], b["moe_intermediate_size"]
            d = H * lat + lat * H + H * sh + sh * H
            parts["moe_dense"] += d
            parts["experts"] += E * 2 * lat * I
            parts["routers"] += E * H
            parts["small"] += E + H
            active += d + 9 * 2 * lat * I + E * H
    parts["small"] += H
    emb, head = V * H, V * H
    total = sum(parts.values()) + emb + head
    active += H + head
    i4 = 0.53125
    print("## bytes (derived from the config; int4 g64 = 0.53125 B a weight, bf16 2 B)")
    print("| part | parameters | on the cards |\n|---|---:|---:|")
    for k, n in parts.items():
        b = n * (i4 if k in ("mamba", "attention", "moe_dense", "experts") else (2 if k == "routers" else 4))
        print(f"| {k} | {n:,} | {b / 1e9:.3f} GB |")
    print(f"| embeddings (bf16) | {emb:,} | {emb * 2 / 1e9:.3f} GB |\n| lm_head (int8 + fp32 row scales) | {head:,} | "
          f"{(head + V * 4) / 1e9:.3f} GB |")
    print(f"total parameters {total:,}; active per token {active:,} (9 experts a MoE layer, the embedding row aside)")
    ssm = sum(1 for b in llm["block_configs"] if b["block_type"] == "mamba") * nh * llm["mamba_head_dim"] * llm["ssm_state_size"]
    print(f"SSM state a sequence: {ssm:,} values = {ssm * 4:,} B fp32; KV a position "
          f"{sum(1 for b in llm['block_configs'] if b['block_type'] == 'attention') * 2 * kvd * 2:,} B")
    if info.get("shapes"):
        lang = {k: s for k, s in info["shapes"].items() if k.startswith("language_model.")}
        q = sum(math.prod(s) * 8 for k, s in lang.items() if k.endswith(".qweight"))
        print(f"cross-check: int4 weights in the checkpoint's qweights {q:,} (= the linears' parameters "
              f"{parts['mamba'] + parts['attention'] + parts['moe_dense'] + parts['experts']:,})")


# ------------------------------------------------------------------------------------------------
# quant: the config and sampled tensors

SAMPLE_LINEARS = ("layers.0.mixer.out_proj", "layers.0.mixer.in_proj", "layers.1.mixer.experts.0.down_proj",
                  "layers.1.mixer.experts.0.up_proj", "layers.7.mixer.q_proj", "layers.1.mixer.shared_experts.up_proj")


def trunk_of(wm: dict) -> str:
    """The trunk prefix: `language_model.backbone.` (ours) or `language_model.model.` (the GPTQ g128 spelling)."""
    return "language_model.backbone." if "language_model.backbone.embeddings.weight" in wm else "language_model.model."


def f16_stats(t) -> str:
    import torch
    a = t.float()
    sub = (t != 0) & (a.abs() < 2.0 ** -14)
    return (f"{t.numel()} scales: negative {int((a < 0).sum())}, zero {int((a == 0).sum())}, subnormal {int(sub.sum())} "
            f"(min |s| {a[a != 0].abs().min().item():.3g}), max |s| {a.abs().max().item():.3g}")


def sec_quant(src: Source, info: dict, llm: dict) -> None:
    import torch
    cfg = src.json("config.json")
    q = cfg["quantization_config"]
    print("## quant: quantization_config " + json.dumps({k: v for k, v in q.items() if k not in ("dynamic", "extra_config")}))
    ec = q.get("extra_config", {})
    print(f"extra_config: {len(ec)} module(s): " + ", ".join(sorted({json.dumps(v) for v in ec.values()})))
    dyn = q.get("dynamic", {})
    plus = {k: v for k, v in dyn.items() if k.startswith("+:")}
    minus = [k for k in dyn if k.startswith("-:")]
    print(f"dynamic: {len(minus)} `-:` exclusions, {len(plus)} `+:` rule(s): " + json.dumps(plus))
    print("  -: " + ", ".join(minus[:6]) + (" ..." if len(minus) > 6 else ""))
    wm, shapes, dtypes = info["wm"], info["shapes"], info["dtypes"]
    if not shapes:
        print("(no headers: --headers)")
        return
    p = trunk_of(wm)
    for b in (p + s for s in SAMPLE_LINEARS):
        sc = b + ".scales"
        raw = src.tensor_bytes(wm[sc], sc)
        if raw is None:
            print(f"{b}: not sampled (--sample)")
            continue
        s = as_tensor(raw, dtypes[sc], shapes[sc])
        qz = as_tensor(src.tensor_bytes(wm[b + ".qzeros"], b + ".qzeros", 0, 64), "I32", (16,))
        gi_shape = shapes.get(b + ".g_idx")
        gi = as_tensor(src.tensor_bytes(wm[b + ".g_idx"], b + ".g_idx"), "I32", gi_shape) if gi_shape else None
        g_ok = gi is not None and torch.equal(gi, torch.arange(gi.numel(), dtype=torch.int32) // 128)
        print(f"{b}: qweight {shapes[b + '.qweight']} scales {dtypes[sc]} {shapes[sc]} - {f16_stats(s)}; qzeros[0:16] all "
              f"0x77777777 {bool((qz == 0x77777777).all())}; g_idx == i // 128 {g_ok}")
    print("| layer | heads softplus(dt_bias) < 0.001 | dt_bias min / max | A_log min / max | D min / max |")
    print("|---:|---:|---|---|---|")
    tot = 0
    dts = []
    for i, b in enumerate(llm["block_configs"]):
        if b["block_type"] != "mamba":
            continue
        vals = {}
        for nm in ("dt_bias", "A_log", "D"):
            k = f"{p}layers.{i}.mixer.{nm}"
            raw = src.tensor_bytes(wm[k], k)
            vals[nm] = None if raw is None else as_tensor(raw, dtypes[k], shapes[k])
        if vals["dt_bias"] is None:
            print(f"| {i} | (not sampled) | | | |")
            continue
        dt = vals["dt_bias"].float()
        dts.append(dt)
        n = int((torch.nn.functional.softplus(dt) < 1e-3).sum())
        tot += n
        bf_exact = all(torch.equal(v, v.to(torch.bfloat16).float()) for v in vals.values() if v is not None)
        print(f"| {i} | {n} / {dt.numel()} | {dt.min().item():.3f} / {dt.max().item():.3f} | "
              f"{vals['A_log'].min().item():.3f} / {vals['A_log'].max().item():.3f} | {vals['D'].min().item():.3f} / "
              f"{vals['D'].max().item():.3f} |" + ("" if bf_exact else " (NOT bf16 values)"))
    if dts:
        alld = torch.cat(dts)
        print(f"dt-clamp census: {tot} of {alld.numel()} heads ({len(dts)} Mamba layers) have softplus(dt_bias) < 0.001; "
              f"dt_bias over all layers {alld.min().item():.3f} .. {alld.max().item():.3f}")
    lo, hi, n_e = math.inf, -math.inf, 0
    per = []
    for i, b in enumerate(llm["block_configs"]):
        if b["block_type"] != "moe":
            continue
        k = f"{p}layers.{i}.mixer.gate.e_score_correction_bias"
        raw = src.tensor_bytes(wm[k], k)
        if raw is None:
            continue
        t = as_tensor(raw, dtypes[k], shapes[k]).float()
        lo, hi, n_e = min(lo, t.min().item()), max(hi, t.max().item()), n_e + 1
        srt = t.sort(descending=True).values
        per.append(f"L{i} {t.min().item():.4f}..{t.max().item():.4f} (9th-10th {srt[8] - srt[9]:.2e})")
    if n_e:
        print(f"e_score_correction_bias over {n_e} MoE layers: {lo:.4f} .. {hi:.4f}; " + "; ".join(per[:8]) + " ...")


# ------------------------------------------------------------------------------------------------
# tokens, template, prompts

def sec_tokens(src: Source, bf16: Source | None) -> None:
    tj = src.small("tokenizer.json")
    t = json.loads(tj)
    vocab = t["model"]["vocab"]
    added = t["added_tokens"]
    print(f"## tokens: tokenizer.json sha256 {sha256(tj)}; model {t['model']['type']}, vocab {len(vocab)}, merges "
          f"{len(t['model'].get('merges', []))}, ignore_merges {t['model'].get('ignore_merges')}, byte_fallback "
          f"{t['model'].get('byte_fallback')}; added tokens {len(added)} (special {sum(a['special'] for a in added)})")
    named = {a["id"]: (a["content"], a["special"]) for a in added if a["id"] < 30 or a["id"] == 990}
    print("added ids 0-29 and 990: " + ", ".join(f"{i} {c!r}{'' if s else ' (not special)'}" for i, (c, s) in sorted(named.items())))
    print(f"max id {max(max(vocab.values()), max(a['id'] for a in added))}; ids < 1000 added: "
          f"{sum(1 for a in added if a['id'] < 1000)}")
    print(f"post_processor: {json.dumps(t.get('post_processor'))[:200]}; normalizer {json.dumps(t.get('normalizer'))[:120]}")
    pre = json.dumps(t.get("pre_tokenizer"))
    print(f"pre_tokenizer: {pre[:400]}")
    tc = src.json("tokenizer_config.json")
    print(f"tokenizer_config: {json.dumps({k: v for k, v in tc.items() if k not in ('added_tokens_decoder',)})[:600]}")
    for s, what in ((src, "our export"), (bf16, "NVIDIA BF16")):
        if s is not None:
            print(f"generation_config ({what}): {json.dumps(s.json('generation_config.json'))}")
    try:
        td = transformers_dir()
        lines = open(os.path.join(td, "tokenization_utils_tokenizers.py"), encoding="utf-8").read().split("\n")
        hits = [i + 1 for i, ln in enumerate(lines) if "fix_mistral_regex" in ln]
        print(f"fix_mistral_regex in transformers' tokenization_utils_tokenizers.py at lines {hits[:12]}")
        from transformers import AutoTokenizer
        a = AutoTokenizer.from_pretrained(src.dir)
        from tokenizers import Tokenizer
        b = Tokenizer.from_str(tj.decode())
        probe = "Hello world!  It's 2026 - naïve café, x_1=foo(y) // 漢字 🙂\n\tdef f():\n    return 'a'"
        ia, ib = a(probe, add_special_tokens=False)["input_ids"], b.encode(probe, add_special_tokens=False).ids
        print(f"AutoTokenizer (5.19.0, fix_mistral_regex in the config) vs tokenizers on the raw tokenizer.json: equal "
              f"{ia == ib} on a mixed probe ({len(ia)} ids); with special tokens: "
              f"{a(probe)['input_ids'][:2]} (no BOS added: {a(probe)['input_ids'] == ia})")
    except Exception as e:  # noqa: BLE001
        print(f"(tokenizer comparison skipped: {e})")


TEMPLATE_FEATURES = [("macro", r"\{%-? macro "), ("namespace", r"namespace\("), ("is mapping", r"is mapping"),
                     ("is sequence", r"is sequence"), ("is string", r"is string"), ("is iterable", r"is iterable"),
                     ("is none / is defined", r"is none|is defined"), ("|items", r"\|items"),
                     ("| tojson | safe", r"tojson \| safe"), ("| string", r"\| ?string"), ("| trim", r"\| trim"),
                     ("| length", r"\| length"), ("| default('', true)", r"default\('', true\)"),
                     (".split", r"\.split\("), ("slicing", r"\[1:\]|\[-1\]|\[0\]"),
                     ("loop.previtem / nextitem", r"loop\.(previtem|nextitem)"), ("loop.index0 / last", r"loop\.(index0|last)"),
                     ("for ... if (filtered loop)", r"for \w+ in \w+ if "), ("~ concatenation", r" ~ "),
                     ("in / not in", r" not in | in ")]


def template_cases() -> list[tuple[str, list, list | None, dict]]:
    tools = [{"type": "function", "function": {"name": "read", "description": "Read a file.\n", "parameters": {
        "type": "object", "properties": {"filePath": {"type": "string", "description": "the path"},
                                         "limit": {"type": "integer"}, "mode": {"type": "string", "enum": ["a", "b"]}},
        "required": ["filePath"]}}},
        {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"], "additionalProperties": False}}}]

    def call(name, args, content="", reasoning=None):
        m = {"role": "assistant", "content": content,
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": args}}]}
        if reasoning is not None:
            m["reasoning_content"] = reasoning
        return m
    sys_ = {"role": "system", "content": "You are a coding agent."}
    user = {"role": "user", "content": "Open the file."}
    later = {"role": "user", "content": "Now the next one."}
    return [
        ("no system, no tools, generation prompt", [user], None, {}),
        ("system, no tools", [sys_, user], None, {}),
        ("system + tools (descriptions, enum, extra keys)", [sys_, user], tools, {}),
        ("tools without a system message", [user], tools, {}),
        ("a call: string / number / bool / null / list / dict arguments",
         [sys_, user, call("f", {"s": "a\nb", "n": 3, "x": 2.5, "b": True, "z": None, "l": [1, "two"], "d": {"k": "v"}}),
          {"role": "tool", "content": "ok"}], tools, {}),
        ("content before the call, reasoning", [user, call("read", {"filePath": "/w/x"}, "Reading.", "I should read.")],
         tools, {}),
        ("two tool results grouped into one user turn", [user, {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "a", "arguments": {"x": 1}}},
            {"type": "function", "function": {"name": "b", "arguments": {}}}]},
            {"role": "tool", "content": "r1"}, {"role": "tool", "content": "r2"}], tools, {}),
        ("reasoning before the last user turn (truncate_history_thinking default: dropped)",
         [user, {"role": "assistant", "content": "Done.", "reasoning_content": "think a"}, later], None, {}),
        ("reasoning before the last user turn, truncate_history_thinking false",
         [user, {"role": "assistant", "content": "Done.", "reasoning_content": "think a"}, later], None,
         {"truncate_history_thinking": False}),
        ("a call with reasoning before the last user turn (dropped)",
         [user, call("read", {"filePath": "/w/x"}, "Reading.", "think b"), {"role": "tool", "content": "data"}, later],
         tools, {}),
        ("reasoning after the last user turn (kept)", [user, {"role": "assistant", "content": "Done.",
                                                             "reasoning_content": "think a"}], None, {}),
        ("enable_thinking false", [user], None, {"enable_thinking": False}),
        ("medium_effort true", [sys_, user], None, {"medium_effort": True}),
        ("content as a list of text parts", [{"role": "user", "content": [{"type": "text", "text": "a"},
                                                                        {"type": "text", "text": "b"}]}], None, {}),
    ]


def sec_template(src: Source, write: bool) -> None:
    raw = src.small("chat_template.jinja")
    text = raw.decode()
    print(f"## template: chat_template.jinja sha256 {sha256(raw)}, {len(raw)} B, {len(text.splitlines())} lines")
    print("features minja must render: " + "; ".join(
        f"{n} (line {', '.join(str(i + 1) for i, ln in enumerate(text.splitlines()) if re.search(p, ln))[:30]})"
        for n, p in TEMPLATE_FEATURES if re.search(p, text)))
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("(no transformers: the renders need the 5.19.0 site)")
        return
    tok = AutoTokenizer.from_pretrained(src.dir)
    cases = []
    for what, msgs, tools, kw in template_cases():
        add = msgs[-1]["role"] != "assistant"
        r = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=add, tokenize=False, **kw)
        ids = tok(r, add_special_tokens=False)["input_ids"]
        cases.append({"name": what, "messages": msgs, "tools": tools, "add_generation_prompt": add, "kwargs": kw,
                      "text": r, "ids": ids})
        print(f"### render: {what} (add_generation_prompt {add}{', ' + json.dumps(kw) if kw else ''}; {len(ids)} ids)")
        print("```text\n" + r + "\n```")
    if write:
        p = os.path.join(REPO, "tests", "server", "bolt_template_cases.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"source": f"{OURS[0]}@{OURS[1]} chat_template.jinja sha256 {sha256(raw)}, rendered by "
                                 f"transformers' apply_chat_template (tools/oracle/bolt_facts.py --section template "
                                 f"--write-fixtures)", "cases": cases}, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {p} ({len(cases)} cases)")


def sec_prompts(src: Source) -> None:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_str(src.small("tokenizer.json").decode())
    d = os.path.join(REPO, "tests", "golden", "prompts")
    print("## prompts: tests/golden/prompts under Bolt's tokenizer (no BOS)")
    for name in ("prose", "code", "long"):
        p = os.path.join(d, f"{name}.txt")
        if os.path.exists(p):
            text = open(p, encoding="utf-8").read().rstrip("\n")
            ids = tok.encode(text, add_special_tokens=False).ids
            back = tok.decode(ids, skip_special_tokens=False)
            print(f"{name}.txt: {len(ids)} ids; decodes back to the text {back == text}")


# ------------------------------------------------------------------------------------------------
# torch's CPU behaviours

def sec_ties(rows: int = 50) -> None:
    import torch
    print(f"## ties: torch {torch.__version__} CPU topk(k=9, sorted=False) with an exact tie at the 9th / 10th")
    g = torch.Generator().manual_seed(0)
    for n in (240, 496):
        lower = higher = stable = 0
        for _ in range(rows):
            v = 4.5 + torch.rand(n, generator=g)
            order = v.argsort(descending=True)
            a_, b_ = sorted(order[8:10].tolist())
            v[b_] = v[a_]
            idx = v.topk(9, sorted=False).indices.tolist()
            lower += (a_ in idx) and (b_ not in idx)
            higher += (b_ in idx) and (a_ not in idx)
            stable += idx == v.topk(9, sorted=False).indices.tolist()
        print(f"of {n}: the lower id wins {lower}/{rows}, the higher id {higher}/{rows}; repeat calls identical "
              f"{stable}/{rows}")


def sec_gemm() -> None:
    import torch
    import torch.nn.functional as F
    print(f"## gemm: torch {torch.__version__} CPU, {torch.get_num_threads()} threads - are a linear's rows independent of "
          f"the batch's row count? (rows of a 512-row batch recomputed in sub-batches of M)")
    g = torch.Generator().manual_seed(1)
    for dt, what in ((torch.float32, "fp32 (R's router: F.linear of widened x and W, R:1044)"),
                     (torch.bfloat16, "bf16 (every other linear)")):
        for N, K in ((496, 4096), (240, 4096), (1024, 4096), (4096, 1024)):
            x = torch.randn(512, K, generator=g).to(torch.bfloat16).to(dt)
            w = (torch.randn(N, K, generator=g) / 64).to(torch.bfloat16).to(dt)
            full = F.linear(x, w)
            res = []
            for M in (1, 2, 9, 64, 128, 2048):
                part = torch.cat([F.linear(x[a:a + M], w) for a in range(0, 512, M)])
                res.append(f"M={M}: {int((part != full).any(-1).sum())} rows / {int((part != full).sum())} elements")
            print(f"{what} [512][{K}] x [{N}][{K}]^T: " + "; ".join(res))


def sec_sum9(rows: int = 100000) -> None:
    import torch
    print(f"## sum9: torch {torch.__version__} CPU fp32 sum over the last dim of [{rows}][9] (the router's sum of the "
          f"nine s, R:1049) against fixed orders")
    g = torch.Generator().manual_seed(2)
    s = torch.rand(rows, 9, generator=g)
    t = s.sum(-1)
    seq = s[:, 0].clone()
    for j in range(1, 9):
        seq = seq + s[:, j]
    pair = ((s[:, 0] + s[:, 1]) + (s[:, 2] + s[:, 3])) + ((s[:, 4] + s[:, 5]) + (s[:, 6] + s[:, 7])) + s[:, 8]
    eightlane = s[:, :8].sum(-1) + s[:, 8]
    print(f"torch.sum = sequential left to right on {int((t == seq).sum())} / {rows} rows; = pairwise-of-8 then + the "
          f"9th on {int((t == pair).sum())}; = sum(first 8) + 9th on {int((t == eightlane).sum())}")


# ------------------------------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--section", default=",".join(SECTIONS))
    ap.add_argument("--bf16", help="the BF16 original's snapshot (small files)")
    ap.add_argument("--cache", default=os.path.join(main_checkout(), "oracle-out-bolt-facts"))
    ap.add_argument("--headers", action="store_true", help="range-read the shard headers that are not cached")
    ap.add_argument("--sample", action="store_true", help="range-read the sampled tensors that are not cached")
    ap.add_argument("--write-fixtures", action="store_true")
    a = ap.parse_args(argv)
    secs = [s for s in a.section.split(",") if s]
    bad = [s for s in secs if s not in SECTIONS]
    if bad:
        raise SystemExit(f"unknown sections {bad}")
    src = Source(a.snapshot, OURS, a.cache, a.headers, a.sample)
    bf = Source(a.bf16, NVIDIA_BF16, a.cache, a.headers, False) if a.bf16 else None
    llm = json.loads(src.small("config.json"))["llm_config"]
    info: dict = {"shapes": {}}
    if "cite" in secs:
        sec_cite(a.snapshot)
    if any(s in secs for s in ("checkpoint", "bytes", "quant")):
        info = sec_checkpoint(src, llm, bf)
    if "bytes" in secs:
        sec_bytes(info, llm)
    if "quant" in secs:
        sec_quant(src, info, llm)
    if "tokens" in secs:
        sec_tokens(src, bf)
    if "template" in secs:
        sec_template(src, a.write_fixtures)
    if "prompts" in secs:
        sec_prompts(src)
    if "ties" in secs:
        sec_ties()
    if "gemm" in secs:
        sec_gemm()
    if "sum9" in secs:
        sec_sum9()
    print(f"(range-read {src.fetched + (bf.fetched if bf else 0):,} B this run)")


if __name__ == "__main__":
    main()
