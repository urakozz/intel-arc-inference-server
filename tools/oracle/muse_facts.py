#!/usr/bin/env python3
"""Spec 23a Task 2: the Muse-Glimmer-30B facts, from SMALL FILES, shard HEADERS and transformers' source.

    muse_facts.py <bf16 snapshot> <gptq snapshot> [--assistant <snapshot>] [--scan-quant]
                  [--section cites,config,tensors,quant,bytes,tokens,drafter] [--cache DIR]

A snapshot is an HF cache directory (`.../models--ORG--NAME/snapshots/REV/`). Its small files are read from
disk; a shard that is not there (the Mac: only the small files are cached) is read by HTTP RANGE requests
from `huggingface.co/ORG/NAME/resolve/REV/<shard>` - the 8-byte length and the JSON header, and with
`--scan-quant` / the quant samples the bytes of single `qzeros` / `g_idx` tensors. Nothing is downloaded
whole. Headers are cached in --cache (default oracle-out-muse-facts/, git-ignored). The token for gated
repos: $HF_TOKEN, else the file $HF_TOKEN_PATH, $HF_HOME/token or ~/.cache/huggingface/token (never printed).

Sections (docs/probe-muse-glimmer-2026-10-15.md quotes every table):
  cites    every transformers 5.19.0 line spec 23 / the facts sheet cites (M:, C:, A:, AC:, G:, CU:, MU:, OC:,
           RP:, TB:) held to the INSTALLED source: the cited range must contain the quoted text (needs
           `import transformers` == 5.19.0; skipped with a line otherwise)
  config   the text config the reference builds; layer types; NoPE layers; layer_rope_theta vs the global theta
  tensors  both repos' tensor groups (layer index -> N): dtype, shape, count, bytes; text / vision totals;
           the vision names by prefix; lm_head / embed_tokens both shipped
  quant    quantization_config; the sampled qzeros / g_idx (layer 0 q_proj, 30 o_proj, 51 down_proj);
           with --scan-quant EVERY qzeros (== 0x77777777) and g_idx (== i // 64) tensor, counted (~210 MB of
           range reads: the box's, plan 23a Task 6)
  bytes    spec 23 §2's per-layer bytes and the read-per-token table, derived from the headers
  tokens   tokenizer.json sha256 in both repos; the added tokens; the post-processor; both templates' sha256
           and what differs; response_template; generation_config; with transformers a one-message render
           (apply_chat_template(tokenize=True) ids, the BOS count) through the ORIGINAL's template
  drafter  the assistant's config and its header table (spec 23 §8.1)
  --twin   <the other int4 snapshot>: the two Vishva007 repos (-AutoRound: auto_round:auto_gptq, no g_idx; -AutoRound-
           GPTQ: gptq, g_idx) are one export - sampled tensors' bytes compared (sha256)
<gptq snapshot> is either int4 spelling (g_idx is checked when shipped).
No torch is needed except for `cites` / the render (transformers imports torch).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]   # tools/oracle/tokenize.py

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import urllib.request  # noqa: E402

PINNED_TRANSFORMERS = "5.19.0"
ORIGINAL_TEMPLATE_SHA = "cfc67e5f"      # spec 23 decision 1 (prefix)
GPTQ_TEMPLATE_SHA = "114f55eb"
TAP_LAYERS = (1, 13, 25, 37, 49)
SAMPLES = ((0, "self_attn.q_proj"), (30, "self_attn.o_proj"), (51, "mlp.down_proj"))
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
SYM_ZERO = 0x77777777
NAMED = {200000: "BOS <|begin_of_text|>", 200001: "EOS <|end_of_text|>", 200007: "<|eom|>", 200008: "EOS <|eot|>",
         200018: "pad <|finetune_right_pad|>", 200022: "<|start|>", 200023: "<|message|>",
         201818: "mask (the drafter's) <|reserved_special_token_1818|>"}

# (tag, path relative to the transformers package, first line, last line, text that must be in the range)
CITES = [
    ("M:101-119", "models/muse_glimmer/modeling_muse_glimmer.py", 101, 119, "class MuseGlimmerRMSNorm"),
    ("M:110-113 (pow, not rsqrt)", "models/muse_glimmer/modeling_muse_glimmer.py", 110, 113,
     "hidden_states * torch.pow(mean_squared, -0.5)"),
    ("M:116-119 (with_scale: plain w, one cast)", "models/muse_glimmer/modeling_muse_glimmer.py", 116, 119,
     "normed_output = normed_output * self.weight.float()"),
    ("M:122-136 (centered: 1 + w)", "models/muse_glimmer/modeling_muse_glimmer.py", 122, 136,
     "output = output * (1.0 + self.weight.float())"),
    ("M:128-129 (rsqrt)", "models/muse_glimmer/modeling_muse_glimmer.py", 128, 129,
     "x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)"),
    ("M:142-155 (SwiGLU)", "models/muse_glimmer/modeling_muse_glimmer.py", 142, 155,
     "self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))"),
    ("M:186-191 (one rotary table from rope_parameters)", "models/muse_glimmer/modeling_muse_glimmer.py", 186, 191,
     'base = config.rope_parameters["rope_theta"]'),
    ("M:202 (cos / sin cast to the dtype)", "models/muse_glimmer/modeling_muse_glimmer.py", 202, 202,
     "return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)"),
    ("M:233-234 (RoPE: y*cos + rot(y)*sin)", "models/muse_glimmer/modeling_muse_glimmer.py", 233, 234,
     "q_embed = (q * cos) + (rotate_half(q) * sin)"),
    ("M:263-269 (eager: scores * scaling, softmax fp32, cast)", "models/muse_glimmer/modeling_muse_glimmer.py", 263, 269,
     "softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)"),
    ("M:291 (scaling head_dim^-0.5)", "models/muse_glimmer/modeling_muse_glimmer.py", 291, 291,
     "self.scaling = self.head_dim**-0.5"),
    ("M:295-311 (q, k, v, o, gate_proj; no bias on gate)", "models/muse_glimmer/modeling_muse_glimmer.py", 295, 311,
     "self.gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)"),
    ("M:312 (weight-less qk norm)", "models/muse_glimmer/modeling_muse_glimmer.py", 312, 312,
     "MuseGlimmerRMSNorm(eps=config.rms_norm_eps, with_scale=False)"),
    ("M:330-331 (q x 3.87 AFTER the norm)", "models/muse_glimmer/modeling_muse_glimmer.py", 330, 331,
     "query_states = self.qk_norm(query_states) * self.qk_scale_factor"),
    ("M:333-336 (NoPE: position_embeddings None)", "models/muse_glimmer/modeling_muse_glimmer.py", 333, 336,
     "if position_embeddings is not None:"),
    ("M:357-359 (output gate before o_proj)", "models/muse_glimmer/modeling_muse_glimmer.py", 357, 359,
     "attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states))"),
    ("M:370-373 (eps 1e-5 in / pre-ff, post_norm_eps post)", "models/muse_glimmer/modeling_muse_glimmer.py", 370, 373,
     "self.post_attention_layernorm = MuseGlimmerTextCenteredRMSNorm(config.hidden_size, eps=config.post_norm_eps)"),
    ("M:397-398 (post-attention norm, residual add)", "models/muse_glimmer/modeling_muse_glimmer.py", 397, 398,
     "hidden_states = residual + hidden_states"),
    ("M:400-404 (MLP half)", "models/muse_glimmer/modeling_muse_glimmer.py", 400, 404,
     "hidden_states = self.post_feedforward_layernorm(hidden_states)"),
    ("M:428-436 (the normed embedding)", "models/muse_glimmer/modeling_muse_glimmer.py", 428, 436,
     "return self.embed_norm(super().forward(input_ids))"),
    ("M:451-453 (embed norm eps = rms_norm_eps)", "models/muse_glimmer/modeling_muse_glimmer.py", 451, 453,
     "config.vocab_size, config.hidden_size, self.padding_idx, config.rms_norm_eps"),
    ("M:457 (final norm WITH weight)", "models/muse_glimmer/modeling_muse_glimmer.py", 457, 457,
     "self.norm = MuseGlimmerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)"),
    ("M:499-502 (masks per layer type)", "models/muse_glimmer/modeling_muse_glimmer.py", 499, 502,
     '"sliding_attention": create_sliding_window_causal_mask(**mask_kwargs)'),
    ("M:512 (NoPE by layer_rope_theta[i] == 0)", "models/muse_glimmer/modeling_muse_glimmer.py", 512, 512,
     "position_embeddings=position_embeddings if self.config.layer_rope_theta[i] else None"),
    ("M:1010-1013 (image / video ids embedded as id 0)", "models/muse_glimmer/modeling_muse_glimmer.py", 1010, 1013,
     "llm_input_ids[multimodal_mask] = 0"),
    ("M:1063 (_tied_weights_keys)", "models/muse_glimmer/modeling_muse_glimmer.py", 1063, 1063,
     '_tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}'),
    ("M:1172-1181 (lm_head, then the softcap)", "models/muse_glimmer/modeling_muse_glimmer.py", 1172, 1181,
     "logits = self.lm_head(hidden_states[:, slice_indices, :])"),
    ("M:1178-1181 (four bf16 roundings)", "models/muse_glimmer/modeling_muse_glimmer.py", 1178, 1181,
     "logits = logits * self.config.text_config.output_multiplier"),
    ("C:152-156 (layer_types default: full at (L-1-i) % 4 == 0)", "models/muse_glimmer/configuration_muse_glimmer.py",
     152, 156, '"full_attention" if (self.num_hidden_layers - 1 - i) % 4 == 0 else "sliding_attention"'),
    ("A:50-55 (the drafter's norm: x-hat rounded, then w x)",
     "models/muse_glimmer_assistant/modeling_muse_glimmer_assistant.py", 50, 55,
     "return self.weight * hidden_states.to(input_dtype)"),
    ("AC:81-89 (block 16, mask 201818, taps)", "models/muse_glimmer_assistant/configuration_muse_glimmer_assistant.py",
     81, 89, "self.target_layer_ids = [1, 13, 25, 37, 49]"),
    ("G:1548 (taps via output_hidden_states=list)", "generation/candidate_generator.py", 1548, 1548,
     '{"output_hidden_states": self.target_layer_ids}'),
    ("G:1551 (15 mask ids after the anchor)", "generation/candidate_generator.py", 1551, 1551,
     "torch.tensor([self.mask_token_id] * (self.block_size - 1))"),
    ("G:1563-1579 (noise embeds WITHOUT the embed norm)", "generation/candidate_generator.py", 1563, 1579,
     "self.main_model_input_embeddings.embed_norm = nn.Identity()"),
    ("G:1620-1621 (crop(-block_size))", "generation/candidate_generator.py", 1620, 1621,
     "self.cache.crop(-self.block_size)"),
    ("G:1633-1638 (anchor + masks at p .. p + 15)", "generation/candidate_generator.py", 1633, 1638,
     "noise_ids = torch.cat([input_ids[:, -1:], self.noise_ids_mask.to(input_ids.device)], dim=-1)"),
    ("G:1658-1660 (raw lm_head on rows 1..15)", "generation/candidate_generator.py", 1658, 1660,
     "outputs.last_hidden_state[:, 1:]"),
    ("MU:93-102 (sliding: kv > q - window)", "masking_utils.py", 93, 102, "return kv_idx > q_idx - sliding_window"),
    ("MU:142-152 (bidirectional window |q - k| <= w)", "masking_utils.py", 142, 152,
     "return abs(q_idx - kv_idx) <= sliding_window"),
    ("OC:112-131 (a list captures only those layers' outputs)", "utils/output_capturing.py", 112, 131,
     "if current_layer_idx not in hidden_states_layers:"),
    ("OC:294-305 (tie_last_hidden_states: a non-None last entry becomes last_hidden_state)",
     "utils/output_capturing.py", 294, 305, "collected_outputs[key].append(outputs.last_hidden_state)"),
    ("CU:296-300 (sliding layer crop keeps sliding_window - 1)", "cache_utils.py", 296, 300,
     "def crop(self, tokens_to_remove: int) -> None:"),
    ("CU:259-260 (a sliding layer caches the last sliding_window - 1 keys)", "cache_utils.py", 259, 260,
     "sliding_window - 1"),
    ("M:205-209 (rotate_half: NEOX halves)", "models/muse_glimmer/modeling_muse_glimmer.py", 205, 209,
     "return torch.cat((-x2, x1), dim=-1)"),
    ("M:510 (each layer's mask by layer_types[i])", "models/muse_glimmer/modeling_muse_glimmer.py", 510, 510,
     "attention_mask=causal_mask_mapping[self.config.layer_types[i]]"),
    ("RP:352-356 (_close_current)", "utils/chat_parsing/response_parser.py", 352, 356, "def _close_current"),
    ("TB:3335-3345 (parse_response(..., prefix=, tools=))", "tokenization_utils_base.py", 3335, 3345, "prefix:"),
]


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(2)


def _token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for p in (os.environ.get("HF_TOKEN_PATH"), os.path.join(os.environ.get("HF_HOME", ""), "token"),
              os.path.expanduser("~/.cache/huggingface/token")):
        if p and os.path.isfile(p):
            return open(p).read().strip() or None
    return None


class Snapshot:
    """An HF cache snapshot: small files from disk, shards from disk if present, else by HTTP range."""

    def __init__(self, path: str, cache: str):
        self.dir = os.path.abspath(path)
        m = re.search(r"models--([^/]+?)--([^/]+)/snapshots/([0-9a-f]+)/?$", self.dir)
        if not m:
            die(f"{path}: not an HF cache snapshot (.../models--ORG--NAME/snapshots/REV)")
        self.repo, self.rev = f"{m.group(1)}/{m.group(2)}", m.group(3)
        self.cache = os.path.join(cache, self.repo.replace("/", "--") + "@" + self.rev[:8])
        self._hdr: dict[str, dict] = {}
        self.ranged = 0

    def has(self, fn: str) -> bool:
        return os.path.exists(os.path.join(self.dir, fn))

    def small(self, fn: str) -> bytes:
        with open(os.path.join(self.dir, fn), "rb") as f:
            return f.read()

    def json(self, fn: str):
        return json.loads(self.small(fn))

    def _get(self, fn: str, a: int, b: int) -> bytes:
        url = f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{fn}"
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
        tok = _token()
        if tok:
            req.add_header("Authorization", "Bearer " + tok)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                break
            except OSError as e:          # URLError / SSL EOF on a flaky link: retry, then fail loudly
                if attempt == 3:
                    die(f"{self.repo}/{fn} bytes {a}-{b}: {e}")
        if len(data) != b - a + 1:
            die(f"{self.repo}/{fn}: asked bytes {a}-{b}, got {len(data)}")
        self.ranged += len(data)
        return data

    def read(self, fn: str, a: int, n: int) -> bytes:
        p = os.path.join(self.dir, fn)
        if os.path.exists(p):
            with open(p, "rb") as f:
                f.seek(a)
                return f.read(n)
        return self._get(fn, a, a + n - 1)

    def header(self, shard: str) -> dict:
        if shard in self._hdr:
            return self._hdr[shard]
        cp = os.path.join(self.cache, shard + ".header.json")
        if not self.has(shard) and os.path.exists(cp):
            h = json.load(open(cp))
        else:
            n = struct.unpack("<Q", self.read(shard, 0, 8))[0]
            h = json.loads(self.read(shard, 8, n))
            h["__data_start__"] = 8 + n
            if not self.has(shard):
                os.makedirs(self.cache, exist_ok=True)
                json.dump(h, open(cp, "w"))
        self._hdr[shard] = h
        return h

    def weight_map(self) -> dict[str, str]:
        if self.has("model.safetensors.index.json"):
            return self.json("model.safetensors.index.json")["weight_map"]
        h = self.header("model.safetensors")
        return {k: "model.safetensors" for k in h if not k.startswith("__")}

    def entry(self, name: str, wm: dict) -> dict:
        return self.header(wm[name])[name]

    def tensor(self, name: str, wm: dict) -> tuple[str, list[int], bytes]:
        h = self.header(wm[name])
        e = h[name]
        a, b = e["data_offsets"]
        return e["dtype"], e["shape"], self.read(wm[name], h["__data_start__"] + a, b - a)

    def file_size(self, shard: str) -> int:
        h = self.header(shard)
        return h["__data_start__"] + max(e["data_offsets"][1] for k, e in h.items() if not k.startswith("__"))


DT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "I8": 1, "U8": 1}


def nbytes(e: dict) -> int:
    a, b = e["data_offsets"]
    return b - a


def text_cfg(cfg: dict) -> dict:
    return cfg["text_config"] if "text_config" in cfg else cfg


def is_text(name: str) -> bool:
    return name.startswith("model.language_model.") or name == "lm_head.weight"


def group(name: str) -> str:
    return re.sub(r"(^|\.)layers\.\d+\.", r"\1layers.N.", name)


def human(n: float) -> str:
    return f"{n / 1e9:.3f} GB" if n >= 1e8 else f"{n / 1e6:.2f} MB" if n >= 1e5 else f"{int(n)} B"


# ------------------------------------------------------------------------------------------------

def sec_cites() -> None:
    print("## cites: transformers 5.19.0 lines held to the installed source")
    try:
        import transformers
    except Exception as e:      # noqa: BLE001
        print(f"  skipped: transformers not importable ({type(e).__name__})")
        return
    if transformers.__version__ != PINNED_TRANSFORMERS:
        print(f"  skipped: transformers {transformers.__version__}, the citations are {PINNED_TRANSFORMERS}'s")
        return
    root = os.path.dirname(transformers.__file__)
    bad = 0
    for tag, rel, a, b, needle in CITES:
        lines = open(os.path.join(root, rel), encoding="utf-8").read().split("\n")
        text = "\n".join(lines[a - 1:b])
        ok = needle in text
        bad += not ok
        print(f"  {'OK ' if ok else 'BAD'} {tag:<70} {rel}:{a}-{b}" + ("" if ok else f"  (wanted {needle!r})"))
    print(f"  {len(CITES) - bad}/{len(CITES)} citations hold" + ("" if not bad else " - FIX THE SHEET"))


def sec_config(bf: Snapshot, gq: Snapshot) -> dict:
    print("## config")
    cb, cg = bf.json("config.json"), gq.json("config.json")
    tb, tg = text_cfg(cb), text_cfg(cg)
    keys = ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings", "sliding_window",
            "rms_norm_eps", "post_norm_eps", "qk_scale_factor", "output_multiplier", "final_logit_softcapping",
            "hidden_activation", "attention_bias", "tie_word_embeddings", "rope_parameters", "bos_token_id",
            "eos_token_id", "pad_token_id")
    print(f"  {'key':<26} {'bf16 ' + bf.rev[:8]:<34} gptq {gq.rev[:8]}")
    for k in keys:
        print(f"  {k:<26} {json.dumps(tb.get(k)):<34} {json.dumps(tg.get(k))}" + ("" if tb.get(k) == tg.get(k) else "  DIFFERS"))
    for c, n in ((cb, "bf16"), (cg, "gptq")):
        print(f"  {n}: top-level architectures {c.get('architectures')} model_type {c.get('model_type')} "
              f"tie_word_embeddings {c.get('tie_word_embeddings')} image_token_id {c.get('image_token_id')} "
              f"video_token_id {c.get('video_token_id')} transformers_version {c.get('transformers_version')}")
    lt, th = tb["layer_types"], tb["layer_rope_theta"]
    full = [i for i, t in enumerate(lt) if t == "full_attention"]
    nope = [i for i, x in enumerate(th) if not x]
    print(f"  layer_types: {len(lt)} = {sum(t == 'sliding_attention' for t in lt)} sliding + {len(full)} full; full at "
          f"{full} (l % 4 == 3: {all(i % 4 == 3 for i in full) and len(full) == len(lt) // 4})")
    print(f"  layer_rope_theta: zero (NoPE) at {nope} - the full layers: {nope == full}; non-zero values "
          f"{sorted(set(x for x in th if x))} vs rope_parameters.rope_theta {tb['rope_parameters']['rope_theta']} "
          f"(equal: {all(x == tb['rope_parameters']['rope_theta'] for x in th if x)})")
    def flat(d, p=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, p + k + "."))
            else:
                out[p + k] = v
        return out
    fb, fg = flat(cb), flat(cg)
    diff = sorted(k for k in set(fb) | set(fg) if fb.get(k) != fg.get(k) and not k.startswith("quantization_config"))
    print("  config.json keys that differ between the repos (quantization_config aside): "
          + (", ".join(f"{k} {fb.get(k)!r} / {fg.get(k)!r}" for k in diff) or "none"))
    return tb


def sec_tensors(src: Snapshot, label: str) -> dict:
    print(f"## tensors: {label} {src.repo}@{src.rev[:8]}")
    wm = src.weight_map()
    shards = sorted(set(wm.values()))
    groups: dict[str, list] = collections.OrderedDict()
    tot = {"text": 0, "vision": 0}
    cnt = {"text": 0, "vision": 0}
    vis_prefix = collections.Counter()
    vis_bytes = collections.Counter()
    for name in sorted(wm):
        e = src.entry(name, wm)
        kind = "text" if is_text(name) else "vision"
        tot[kind] += nbytes(e)
        cnt[kind] += 1
        if kind == "vision":
            p = ".".join(name.split(".")[:2])
            vis_prefix[p] += 1
            vis_bytes[p] += nbytes(e)
            continue
        g = group(name)
        groups.setdefault(g, [e["dtype"], e["shape"], 0, 0])
        groups[g][2] += 1
        groups[g][3] += nbytes(e)
    files = sum(src.file_size(s) for s in shards)
    print(f"  {len(shards)} shards, {len(wm)} tensors ({cnt['text']} text + {cnt['vision']} vision), files "
          f"{files:,} B; data: text {tot['text']:,} B ({human(tot['text'])}), vision {tot['vision']:,} B "
          f"({human(tot['vision'])})")
    print(f"  {'group':<62} {'dtype':<5} {'shape (layer 0)':<18} {'n':>3} {'bytes':>16}")
    for g, (dt, shape, n, b) in groups.items():
        print(f"  {g.replace('model.language_model.', 'LM.'):<62} {dt:<5} {str(shape):<18} {n:>3} {b:>16,}")
    print(f"  vision by prefix: " + ", ".join(f"{p} {n} ({human(vis_bytes[p])})" for p, n in sorted(vis_prefix.items()))
          + f"; total {cnt['vision']}")
    lm, emb = "lm_head.weight", "model.language_model.embed_tokens.weight"
    print(f"  lm_head.weight shipped: {lm in wm} {src.entry(lm, wm)['shape'] if lm in wm else ''}; embed_tokens "
          f"shipped: {emb in wm} {src.entry(emb, wm)['shape'] if emb in wm else ''} - separate tensors (untied)")
    return {"wm": wm, "groups": groups, "tot": tot, "cnt": cnt}


def _i32(data: bytes):
    import array
    a = array.array("i")
    a.frombytes(data)
    return a


def check_quant_tensor(src: Snapshot, wm: dict, base: str):
    """(qzeros all 0x77777777, g_idx == i // 64 - None when the spelling ships no g_idx, bytes read)."""
    dtz, shz, z = src.tensor(base + ".qzeros", wm)
    zero_ok = dtz == "I32" and all((v & 0xFFFFFFFF) == SYM_ZERO for v in _i32(z))
    if base + ".g_idx" not in wm:
        return zero_ok, None, len(z)
    dtg, shg, g = src.tensor(base + ".g_idx", wm)
    gidx_ok = dtg == "I32" and all(v == i // 64 for i, v in enumerate(_i32(g)))
    return zero_ok, gidx_ok, len(z) + len(g)


def sec_twin(a: Snapshot, b: Snapshot) -> None:
    """The two Vishva007 int4 repos are one export in two spellings (auto_round:auto_gptq / gptq): sampled tensors'
    bytes compared (local reads, else HTTP range) - sha256 per tensor."""
    print(f"## twin: {a.repo}@{a.rev[:8]} vs {b.repo}@{b.rev[:8]}")
    wa, wb = a.weight_map(), b.weight_map()
    names = [f"model.language_model.layers.0.self_attn.k_proj{s}" for s in (".qweight", ".scales", ".qzeros")]
    names += ["model.language_model.layers.30.self_attn.o_proj.scales", "model.language_model.layers.51.mlp.down_proj.qzeros",
              "model.language_model.layers.30.post_feedforward_layernorm.weight", "model.language_model.norm.weight"]
    same = 0
    for n in names:
        if n not in wa or n not in wb:
            print(f"  {n}: absent in {'the first' if n not in wa else 'the second'}")
            continue
        da, sa, ba = a.tensor(n, wa)
        db, sb, bb = b.tensor(n, wb)
        ha, hb = hashlib.sha256(ba).hexdigest(), hashlib.sha256(bb).hexdigest()
        same += ha == hb and da == db and sa == sb
        print(f"  {n}: {da} {sa}, {len(ba):,} B - sha256 {ha[:12]} / {hb[:12]}: {'IDENTICAL' if ha == hb else 'DIFFER'}")
    gi = sum(k.endswith(".g_idx") for k in wa), sum(k.endswith(".g_idx") for k in wb)
    print(f"  {same} / {len(names)} sampled tensors byte-identical; g_idx tensors {gi[0]} / {gi[1]}; text names "
          f"{sum(k.startswith('model.language_model.') or k == 'lm_head.weight' for k in wa)} / "
          f"{sum(k.startswith('model.language_model.') or k == 'lm_head.weight' for k in wb)}")


def sec_quant(gq: Snapshot, wm: dict, scan: bool) -> None:
    print("## quant")
    qc = gq.json("config.json").get("quantization_config", {})
    print(f"  config.json quantization_config: {json.dumps(qc, sort_keys=True)}")
    if gq.has("quantization_config.json"):
        print(f"  quantization_config.json equal to config.json's (as JSON): "
              f"{gq.json('quantization_config.json') == qc}")
    acc = (qc.get("bits") == 4 and qc.get("group_size") in (64, 128) and qc.get("sym") is True
           and qc.get("desc_act") in (False, None) and qc.get("quant_method") in ("gptq", "auto-round")
           and not any(str(r).startswith("+:") for r in (qc.get("dynamic") or {}))
           and "extra_config" not in qc and qc.get("packing_format") in (None, "auto_round:auto_gptq"))
    print(f"  loader::QuantConfig::parse's rule (src/loader/quant.h:21-60): bits 4, group 64|128, sym, desc_act false or "
          f"absent, quant_method gptq|auto-round (packing auto_round:auto_gptq), no '+:' dynamic rule -> accepted: {acc} (dynamic "
          f"{'absent' if 'dynamic' not in qc else len(qc['dynamic'])}, extra_config "
          f"{'absent' if 'extra_config' not in qc else 'present'}, packing_format {qc.get('packing_format', 'absent')})")
    for layer, lin in SAMPLES:
        base = f"model.language_model.layers.{layer}.{lin}"
        zo, go, nb = check_quant_tensor(gq, wm, base)
        e = gq.entry(base + ".qzeros", wm)
        gtxt = "not shipped (the auto_round spelling)" if go is None else go
        print(f"  sample {base}: qzeros {e['shape']} all 0x77777777: {zo}; g_idx == i // 64: {gtxt} ({nb:,} B read)")
    if not scan:
        print("  full scan: not run (--scan-quant; the box's, plan 23a Task 6)")
        return
    bases = sorted({k[: -len(".qzeros")] for k in wm if k.endswith(".qzeros")})
    zbad, gbad, total, ng = [], [], 0, 0
    for i, b in enumerate(bases):
        zo, go, nb = check_quant_tensor(gq, wm, b)
        total += nb
        ng += go is not None
        if not zo:
            zbad.append(b)
        if go is False:
            gbad.append(b)
        if (i + 1) % 52 == 0:
            print(f"    scanned {i + 1}/{len(bases)} ({total / 1e6:.0f} MB)", flush=True)
    print(f"  FULL SCAN: {len(bases)} linears, {total:,} B read: qzeros != 0x77777777 in {len(zbad)} {zbad[:4]}; "
          f"g_idx shipped for {ng}, != i // 64 in {len(gbad)} {gbad[:4]}"
          + ("  -> CLEAN" if not zbad and not gbad else "  -> NOT SYM g64"))


def sec_bytes(tc: dict, bfi: dict, gqi: dict) -> None:
    print("## bytes (derived from the headers)")
    H, I, L = tc["hidden_size"], tc["intermediate_size"], tc["num_hidden_layers"]
    nq, nkv, hd, V = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"], tc["vocab_size"]
    shapes = {"q": (nq * hd, H), "k": (nkv * hd, H), "v": (nkv * hd, H), "gate": (nq * hd, H), "o": (H, nq * hd),
              "ffn gate": (I, H), "ffn up": (I, H), "ffn down": (H, I)}
    per_layer = sum(a * b for a, b in shapes.values())
    lin_bf16 = sum(b for g, (_dt, _s, _n, b) in bfi["groups"].items() if g.endswith("_proj.weight")) // 2
    print(f"  per layer: " + ", ".join(f"{k} {a}x{b}" for k, (a, b) in shapes.items()) + f" = {per_layer:,} parameters")
    print(f"  52 layers' linears: {per_layer * L:,} parameters (the bf16 headers: {lin_bf16:,}: equal {per_layer * L == lin_bf16})")
    qw = sum(b for g, (_dt, _s, _n, b) in gqi["groups"].items() if g.endswith(".qweight"))
    sc = sum(b for g, (_dt, _s, _n, b) in gqi["groups"].items() if g.endswith(".scales"))
    qz = sum(b for g, (_dt, _s, _n, b) in gqi["groups"].items() if g.endswith(".qzeros"))
    gi = sum(b for g, (_dt, _s, _n, b) in gqi["groups"].items() if g.endswith(".g_idx"))
    print(f"  int4 g64 in the GPTQ headers: qweight {qw:,} + scales {sc:,} = {qw + sc:,} B ({human(qw + sc)}; "
          f"{8 * (qw + sc) / (per_layer * L):.3f} bit) - per layer {(qw + sc) / L / 1e6:.2f} MB; not loaded by the engine: "
          f"qzeros {qz:,} B, g_idx {gi:,} B (docs/13's asserts read them)")
    norms = sum(b for g, (_dt, _s, _n, b) in gqi["groups"].items() if g.endswith("layernorm.weight") or g.endswith("norm.weight"))
    norms_f32 = norms * 2
    head_i8 = V * H + V * 4
    emb = V * H * 2
    card = qw + sc + norms_f32 + head_i8 + emb
    tok = qw + sc + norms_f32 + head_i8
    print(f"  norms bf16 {norms:,} B -> fp32 on the device {norms_f32:,} B ({human(norms_f32)})")
    print(f"  lm_head int8 + fp32 row scales (spec 9): {head_i8:,} B; embed_tokens bf16 {emb:,} B (one {2 * H:,} B row a token)")
    print(f"  weights on the card {card:,} B ({human(card)}); read per decoded token (weights) {tok:,} B ({human(tok)})")
    full = sum(t == "full_attention" for t in tc["layer_types"])
    slide = L - full
    kv_pos = full * nkv * hd * 2 * 2
    win = tc["sliding_window"]
    ring = slide * 4096 * nkv * hd * 2 * 2
    print(f"  KV a position (full layers only): {full} x {nkv} x {hd} x 2 x 2 B = {kv_pos:,} B; at 131072: "
          f"{human(kv_pos * 131072)}; rings {slide} x 4096 slots: {ring:,} B ({human(ring)})")
    for d in (4096, 32768, 131072):
        r = tok + kv_pos * d + slide * min(d, win) * nkv * hd * 2 * 2
        print(f"  read per token at depth {d}: {human(r)}; roofline at 590 GB/s {590e9 / r:.1f} t/s")


def sec_tokens(bf: Snapshot, gq: Snapshot, render: bool) -> None:
    print("## tokens and templates")
    sha = {n: hashlib.sha256(s.small("tokenizer.json")).hexdigest() for n, s in (("bf16", bf), ("gptq", gq))}
    print(f"  tokenizer.json sha256 bf16 {sha['bf16'][:16]}... gptq {sha['gptq'][:16]}... equal {sha['bf16'] == sha['gptq']}")
    tj = json.loads(bf.small("tokenizer.json"))
    at = tj["added_tokens"]
    print(f"  model {tj['model']['type']}: vocab {len(tj['model']['vocab'])}, merges {len(tj['model']['merges'])}; "
          f"normalizer {tj['normalizer']}; decoder {tj['decoder']['type'] if tj.get('decoder') else None}")
    print(f"  added tokens: {len(at)}, ids {at[0]['id']}..{at[-1]['id']}, all special {all(a['special'] for a in at)}")
    content = {a["id"]: a["content"] for a in at}
    print("  named (not reserved_special_token_N): " + ", ".join(f"{a['id']} {a['content']}" for a in at
                                                                 if "reserved_special" not in a["content"]))
    for i, what in NAMED.items():
        print(f"    {i} {content.get(i)!r:<38} {what}")
    pp = tj["post_processor"]
    single = [list(x.values())[0]["id"] for x in pp.get("single", [])]
    print(f"  post_processor {pp['type']} single {single} -> encode(add_special_tokens=True) prepends BOS")
    tpl = {n: s.small("chat_template.jinja") for n, s in (("bf16", bf), ("gptq", gq))}
    for n, t in tpl.items():
        s = hashlib.sha256(t).hexdigest()
        txt = t.decode()
        print(f"  chat_template.jinja {n}: sha256 {s[:8]}...{s[-6:]}, {len(t)} B, {txt.count(chr(10)) + (not txt.endswith(chr(10)))} "
              f"lines; 'Reasoning effort' normalised: {'Reasoning effort' in txt}; error text names: "
              f"{sorted(set(re.findall(r'(?:Onyx|Muse[- ]Glimmer)', txt)))}")
    print(f"  original's template is the spec's ({ORIGINAL_TEMPLATE_SHA}...): "
          f"{hashlib.sha256(tpl['bf16']).hexdigest().startswith(ORIGINAL_TEMPLATE_SHA)}; GPTQ's is {GPTQ_TEMPLATE_SHA}...: "
          f"{hashlib.sha256(tpl['gptq']).hexdigest().startswith(GPTQ_TEMPLATE_SHA)}")
    tcfg = bf.json("tokenizer_config.json")
    print(f"  tokenizer_config.json: bos {tcfg.get('bos_token')!r} eos {tcfg.get('eos_token')!r} pad {tcfg.get('pad_token')!r} "
          f"model_max_length {tcfg.get('model_max_length')}; chat_template key present: {'chat_template' in tcfg}")
    print(f"  response_template: {json.dumps(tcfg.get('response_template'), sort_keys=True)}")
    print(f"  generation_config.json bf16: {json.dumps(bf.json('generation_config.json'), sort_keys=True)}")
    print(f"  generation_config.json gptq: {json.dumps(gq.json('generation_config.json'), sort_keys=True)}")
    if not render:
        return
    try:
        import transformers
        from transformers import AutoTokenizer
    except Exception as e:      # noqa: BLE001
        print(f"  render: skipped (transformers not importable: {type(e).__name__})")
        return
    tok = AutoTokenizer.from_pretrained(bf.dir)
    msgs = [{"role": "user", "content": "Hello"}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, current_date="2026-10-15",
                                  return_dict=False)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, current_date="2026-10-15")
    plain = tok(text, add_special_tokens=False)["input_ids"]
    special = tok(text)["input_ids"]
    print(f"  render (transformers {transformers.__version__}, the original's template, one user message 'Hello', "
          f"current_date 2026-10-15, add_generation_prompt): {len(ids)} ids, BOS count {ids.count(200000)}; equals "
          f"tok(text, add_special_tokens=False): {ids == plain}; tok(text) (special tokens on) has BOS x "
          f"{special.count(200000)}")
    print(f"  rendered text: {text!r}")
    print(f"  ids: {ids}")


def sec_drafter(asst: Snapshot) -> None:
    print(f"## drafter {asst.repo}@{asst.rev[:8]}")
    c = asst.json("config.json")
    print(f"  config: {json.dumps(c, sort_keys=True)}")
    wm = asst.weight_map()
    groups = collections.OrderedDict()
    tot = 0
    for name in sorted(wm):
        e = asst.entry(name, wm)
        tot += nbytes(e)
        g = group(name)
        groups.setdefault(g, [e["dtype"], e["shape"], 0, 0])
        groups[g][2] += 1
        groups[g][3] += nbytes(e)
    size = asst.file_size("model.safetensors")
    params = tot // 2
    print(f"  model.safetensors: {len(wm)} tensors, file {size:,} B, data {tot:,} B = {params:,} bf16 parameters")
    for g, (dt, shape, n, b) in groups.items():
        print(f"  {g:<52} {dt:<5} {str(shape):<16} {n:>3} {b:>14,}")
    print(f"  no embedding / lm_head tensor: {not any('embed' in k or 'lm_head' in k for k in wm)}; "
          f"target_layer_ids {c.get('target_layer_ids')} == TAP_LAYERS: {tuple(c.get('target_layer_ids', ())) == TAP_LAYERS}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bf16")
    ap.add_argument("gptq")
    ap.add_argument("--assistant")
    ap.add_argument("--scan-quant", action="store_true")
    ap.add_argument("--section", default="cites,config,tensors,quant,bytes,tokens,drafter")
    ap.add_argument("--cache", default="oracle-out-muse-facts")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--twin", help="the other Vishva007 int4 spelling's snapshot: sampled tensors compared byte for byte")
    a = ap.parse_args(argv)
    secs = set(a.section.split(","))
    bf, gq = Snapshot(a.bf16, a.cache), Snapshot(a.gptq, a.cache)
    print(f"bf16 {bf.repo}@{bf.rev}; gptq {gq.repo}@{gq.rev}")
    if "cites" in secs:
        sec_cites()
    tc = sec_config(bf, gq)
    bfi = gqi = None
    if secs & {"tensors", "bytes", "quant"}:
        bfi = sec_tensors(bf, "bf16")
        gqi = sec_tensors(gq, "gptq")
    if "quant" in secs:
        sec_quant(gq, gqi["wm"], a.scan_quant)
    if "bytes" in secs:
        sec_bytes(tc, bfi, gqi)
    if "tokens" in secs:
        sec_tokens(bf, gq, not a.no_render)
    if "drafter" in secs and a.assistant:
        sec_drafter(Snapshot(a.assistant, a.cache))
    if a.twin:
        sec_twin(gq, Snapshot(a.twin, a.cache))
    print(f"range bytes read: bf16 {bf.ranged:,}, gptq {gq.ranged:,}")


if __name__ == "__main__":
    main()
