#!/usr/bin/env python3
"""Spec 24a Task 4: the LFM2.5-24B-A2B (`lfm2_moe`) facts - the rounding chain cited into the installed
transformers 5.19.0, the checkpoint from its small files and shard headers, tokens, the template, torch's CPU
tie order and the conv's accumulation. docs/probe-lfm2moe-2026-10-15.md quotes its output.

    lfm2moe_facts.py <snapshot dir | hf:REPO[@REV]> [--section cite,checkpoint,bytes,tokens,template,prompts,
                     licence,ties,conv] [--tiny <snapshot>] [--rows 100000] [--cache DIR]

Sections:
  cite        every rounding point of spec 24 §2 located in the INSTALLED 5.19.0 sources by an anchor string
              (`M:line` = modeling_lfm2_moe.py, `C:line` = configuration_lfm2_moe.py, other files by name) - the
              citations are found, not typed
  checkpoint  the index's names grouped (layer / expert index -> N), dtype and shape per group from the shard
              headers (a local shard's first 8 + n bytes, else an HTTP range request), the shard map, the
              parameter count with and without the fp32 expert biases against the index's total_parameters
  bytes       spec 24 §3's per-token decode bytes and the memory table re-derived from the shapes
  tokens      tokenizer.json: BPE vocab, added tokens (id, content, special), contiguity -> vocab_used; the
              pre-tokenizer, normalizer; generation_config; tokenizer.json sha256
  template    chat_template.jinja sha256, the Jinja features it uses (what minja must render), and one render
              per feature through apply_chat_template (24e's byte-identity cases); the pythonic call grammar
  prompts     whether tests/golden/prompts/{prose,code}.txt re-tokenise to Qwen's committed ids (they cannot:
              another tokenizer) - the lengths
  licence     LICENSE sha256 and its section headings (§5's threshold sentence quoted)
  ties        torch's CPU topk on planted exact ties at the 4th / 5th of 64 fp32 sel values (50 rows)
  conv        torch's CPU bf16 depthwise conv1d (the short conv's F.conv1d, 3 taps) against fp32 taps-ascending
              (= fma: bf16 x bf16 products are exact in fp32) and one rounding, on --rows random rows of 2048
              channels; the rows (and elements) that differ, and against other accumulation orders
--tiny adds nothing measured here (Task 1's lfm2moe_tiny_check.py prints the tiny-model rows).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import urllib.request  # noqa: E402

SECTIONS = ("cite", "checkpoint", "bytes", "tokens", "template", "prompts", "licence", "ties", "conv")
REPO = os.path.dirname(os.path.dirname(_HERE))


# ------------------------------------------------------------------------------------------------
# where the bytes come from (a local snapshot, partial ones too: absent shards by range request)

class Source:
    def __init__(self, where: str, cache: str | None = None):
        self.remote = where.startswith("hf:")
        if self.remote:
            self.repo, _, self.rev = where[3:].partition("@")
            self.rev = self.rev or "main"
            self.dir = None
        else:
            self.dir = where
            self.repo, self.rev = "LiquidAI/LFM2.5-24B-A2B", os.path.basename(os.path.normpath(where))
        self.cache = cache
        self._h: dict[str, dict] = {}
        self.how: dict[str, str] = {}

    def _get(self, fn: str, a: int | None = None, b: int | None = None) -> bytes:
        req = urllib.request.Request(f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{fn}")
        if a is not None:
            req.add_header("Range", f"bytes={a}-{b}")
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    def local(self, fn: str) -> str | None:
        if self.dir is None:
            return None
        p = os.path.join(self.dir, fn)
        return p if os.path.exists(p) else None

    def small(self, fn: str) -> bytes:
        p = self.local(fn)
        if p:
            return open(p, "rb").read()
        return self._get(fn)

    def json(self, fn: str):
        return json.loads(self.small(fn))

    def header(self, shard: str) -> dict:
        if shard in self._h:
            return self._h[shard]
        p = self.local(shard)
        if p:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            self.how[shard] = "local"
        else:
            cp = os.path.join(self.cache, shard + ".header.json") if self.cache else None
            if cp and os.path.exists(cp):
                h = json.load(open(cp))
                self.how[shard] = "cached range request"
            else:
                n = struct.unpack("<Q", self._get(shard, 0, 7))[0]
                h = json.loads(self._get(shard, 8, 8 + n - 1))
                self.how[shard] = "range request"
                if cp:
                    os.makedirs(os.path.dirname(cp), exist_ok=True)
                    json.dump(h, open(cp, "w"))
        self._h[shard] = h
        return h


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------
# cite: the rounding chain located in the installed sources

CITES = [  # (what, file, anchor regex) - file "M" modeling, "C" configuration, else relative to transformers/
    ("RMSNorm: fp32 variance", "M", r"variance = hidden_states\.pow\(2\)\.mean\(-1, keepdim=True\)"),
    ("RMSNorm: x^ cast back to bf16, then w * x^ (two roundings)", "M", r"return self\.weight \* hidden_states\.to\(input_dtype\)"),
    ("q / k layernorm over head_dim (64)", "M", r"self\.q_layernorm = Lfm2MoeRMSNorm\(self\.head_dim"),
    ("q / k norm applied before RoPE", "M", r"query_states = self\.q_layernorm\("),
    ("RoPE inv_freq in fp32", "M", r"inv_freq = 1\.0 / \(base \*\* \(torch\.arange\(0, dim, 2, dtype=torch\.float\)"),
    ("RoPE freqs = pos x inv_freq in fp32", "M", r"freqs = position_ids\[\.\.\., None\]\.float\(\) \* self\.inv_freq"),
    ("RoPE cos / sin cast to the model dtype", "M", r"return cos\.to\(dtype=x\.dtype\), sin\.to\(dtype=x\.dtype\)"),
    ("rotate_half", "M", r"def rotate_half\(x\)"),
    ("RoPE: rne(rne(q cos) + rne(rotate_half(q) sin))", "M", r"q_embed = \(q \* cos\) \+ \(rotate_half\(q\) \* sin\)"),
    ("attention scale head_dim^-0.5 = 0.125", "M", r"self\.scaling = self\.head_dim\*\*-0\.5"),
    ("eager: bf16 Q K^T then x scaling", "M", r"attn_weights = torch\.matmul\(query, key_states\.transpose\(2, 3\)\) \* scaling"),
    ("eager: softmax in fp32, cast to bf16", "M", r"softmax\(attn_weights, dim=-1, dtype=torch\.float32\)\.to\(query\.dtype\)"),
    ("eager: bf16 P V", "M", r"attn_output = torch\.matmul\(attn_weights, value_states\)"),
    ("short conv: in_proj then B, C, x = chunk(3) in that order", "M", r"B, C, x = BCx\.chunk\(3, dim=-2\)"),
    ("short conv: Bx = B * x (bf16)", "M", r"hidden_states = B \* x$"),
    ("short conv: cached decode step (seq 1)", "M", r"hidden_states = causal_conv1d_update\("),
    ("decode: cat([state(3), new]) - the state is Bx", "M", r"hidden_states_new = torch\.cat\(\[conv_state, hidden_states\], dim=-1\)"),
    ("decode: F.conv1d without padding, the last output", "M", r"out = F\.conv1d\(hidden_states_new, weight\.unsqueeze\(1\), bias, padding=0"),
    ("prefill: F.conv1d at padding K - 1 = 2 (zeros before position 0)", "M", r"padding = weight\.shape\[-1\] - 1"),
    ("short conv: y = C * conv(Bx)", "M", r"y = C \* hidden_states"),
    ("short conv: out_proj", "M", r"y = self\.out_proj\(y\)"),
    ("cache: prefill pads the state to conv_L_cache with zeros", "cache_utils.py", r"full_conv_states = torch\.nn\.functional\.pad\(full_conv_states, \(padding_length, 0\), value=0\)"),
    ("cache: later chunks see cat(state, new)", "cache_utils.py", r"full_conv_states = torch\.cat\(\[self\.conv_states\[state_idx\], conv_states\], dim=-1\)"),
    ("cache: keeps the last conv_L_cache columns", "cache_utils.py", r"self\.conv_states\[state_idx\]\.copy_\(full_conv_states\[\.\.\., -self\.conv_kernel_size\[state_idx\] :\]\)"),
    ("cache: layer type conv -> LinearAttentionLayer", "cache_utils.py", r'"conv": LinearAttentionLayer'),
    ("router: logits = F.linear (bf16)", "M", r"router_logits = F\.linear\(hidden_states, self\.weight\)"),
    ("router: s = sigmoid(logits) in the logits' dtype", "M", r"routing_weights = router_logits\.sigmoid\(\)"),
    ("router: sel = s + expert_bias (fp32 buffer: promotes)", "M", r"scores_for_routing = routing_weights \+ expert_bias"),
    ("router: topk on sel", "M", r"_, selected_experts = torch\.topk\(scores_for_routing, k=self\.top_k, dim=-1\)"),
    ("router: weights = s gathered (not sel), bf16", "M", r"routing_weights = torch\.gather\(routing_weights, dim=1, index=selected_experts\)\.type_as\(router_logits\)"),
    ("router: w / (sum(w) + 1e-6), bf16", "M", r"routing_weights = routing_weights / \(routing_weights\.sum\(dim=-1, keepdim=True\) \+ 1e-6\)"),
    ("router: x routed_scaling_factor (1.0)", "M", r"routing_weights = routing_weights \* self\.routed_scaling_factor"),
    ("expert_bias is an fp32 buffer", "M", r"self\.expert_bias = nn\.Buffer\(torch\.zeros\(config\.num_experts, dtype=torch\.float32\)\)"),
    ("experts: eager fallback, bf16 index_add_ by ascending expert id", "M", r"final_hidden_states\.index_add_\(0, token_idx"),
    ("experts: grouped_mm, sort by expert", "integrations/moe.py", r"expert_ids_g, perm = torch\.sort\(expert_ids\)"),
    ("experts: grouped_mm, a = act(gate) * up", "integrations/moe.py", r"return self\.act_fn\(gate\) \* up"),
    ("experts: grouped_mm, t_k = proj * w (bf16)", "integrations/moe.py", r"weighted_out = proj_out \* sample_weights_g\.unsqueeze\(-1\)"),
    ("experts: grouped_mm, sum over the k slots (fp32 accumulation, one rounding)", "integrations/moe.py", r"final_hidden_states = weighted_out\.view\(num_tokens, num_top_k, hidden_dim\)\.sum\(dim=1\)"),
    ("experts implementation default grouped_mm, eager when it cannot dispatch", "modeling_utils.py", r'applicable_experts = "grouped_mm" if requested_experts is None else requested_experts'),
    ("conv kernels: hub / package / torch fallback order", "integrations/hub_kernels.py", r"def use_kernel_func_from_hub_with_fallback"),
    ("dense MLP: w2(silu(w1 x) * w3 x)", "M", r"return self\.w2\(F\.silu\(self\.w1\(x\)\) \* self\.w3\(x\)\)"),
    ("dense layers: layer_idx < num_dense_layers", "M", r"if layer_idx < config\.num_dense_layers"),
    ("layer: h = mixer(operator_norm(h)) + residual", "M", r"hidden_states = hidden_states \+ residual"),
    ("layer: h = h + FFN(ffn_norm(h))", "M", r"hidden_states = hidden_states \+ self\.feed_forward\(self\.ffn_norm\(hidden_states\)\)"),
    ("model: embedding_norm AFTER the last layer", "M", r"hidden_states = self\.embedding_norm\(hidden_states\)"),
    ("model: the conv layers' mask (None unpadded)", "M", r'"conv": create_recurrent_attention_mask'),
    ("head tied to the embedding", "M", r'_tied_weights_keys = \{"lm_head\.weight": "model\.embed_tokens\.weight"\}'),
    ("head: logits = lm_head(h)", "M", r"logits = self\.lm_head\(hidden_states\[:, slice_indices, :\]\)"),
    ("sdpa supported (the default when not pinned)", "M", r"_supports_sdpa = True"),
    ("config: tie_word_embeddings default True", "C", r"tie_word_embeddings: bool = True"),
    ("config: norm_eps default 1e-5", "C", r"norm_eps: float = 0\.00001"),
    ("checkpoint conversion: experts.*.w1 / w3 -> gate_up_proj", "conversion_mapping.py", r'"feed_forward\.experts\.\*\.w1\.weight",'),
    ("recurrent mask: None for an unpadded batch", "masking_utils.py", r"def create_recurrent_attention_mask"),
    ("repetition penalty (generation)", "generation/logits_process.py", r"score = torch\.where\(score < 0, score \* self\.penalty, score / self\.penalty\)"),
]


def transformers_dir() -> str:
    import importlib.util
    spec = importlib.util.find_spec("transformers")
    if spec is None or not spec.origin:
        raise SystemExit("cite: transformers is not importable (the 5.19.0 site first on PYTHONPATH)")
    return os.path.dirname(spec.origin)


def sec_cite() -> None:
    td = transformers_dir()
    ver = open(os.path.join(td, "__init__.py"), encoding="utf-8").read()
    v = re.search(r'__version__ = "([^"]+)"', ver).group(1)
    print(f"## cite: the rounding chain in transformers {v} ({td})")
    files = {"M": "models/lfm2_moe/modeling_lfm2_moe.py", "C": "models/lfm2_moe/configuration_lfm2_moe.py"}
    print("| point | where |\n|---|---|")
    for what, f, pat in CITES:
        rel = files.get(f, f)
        lines = open(os.path.join(td, rel), encoding="utf-8").read().split("\n")
        hits = [i + 1 for i, ln in enumerate(lines) if re.search(pat, ln)]
        tag = f if f in files else rel.rsplit("/", 1)[-1]
        print(f"| {what} | {tag}:{','.join(map(str, hits)) if hits else 'NOT FOUND'} |")


# ------------------------------------------------------------------------------------------------
# checkpoint, bytes

def group(n: str) -> str:
    return re.sub(r"\.experts\.\d+\.", ".experts.E.", re.sub(r"^model\.layers\.\d+\.", "model.layers.L.", n))


def sec_checkpoint(src: Source) -> dict:
    idx = src.json("model.safetensors.index.json")
    wm = idx["weight_map"]
    print(f"## checkpoint: {len(wm)} tensors in {len(set(wm.values()))} shards; index metadata {idx.get('metadata')}")
    shapes, dtypes, numel = {}, {}, 0
    nbytes = 0
    expert_bias = 0
    for shard in sorted(set(wm.values())):
        h = src.header(shard)
        for k, e in h.items():
            if k.startswith("__"):
                continue
            n = 1
            for d in e["shape"]:
                n *= d
            shapes[k], dtypes[k] = e["shape"], e["dtype"]
            numel += n
            nbytes += e["data_offsets"][1] - e["data_offsets"][0]
            if k.endswith("expert_bias"):
                expert_bias += n
    missing = sorted(set(wm) - set(shapes))
    print(f"headers: {', '.join(f'{s} {how}' for s, how in sorted(src.how.items()))}")
    if missing:
        print(f"names of the index absent from the headers: {missing[:5]}")
    groups = collections.OrderedDict()
    for k in sorted(wm, key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", x)]):
        g = group(k)
        groups.setdefault(g, []).append(k)
    print("| group | count | dtype | shape |\n|---|---:|---|---|")
    for g, ks in groups.items():
        ds = sorted({dtypes.get(k, "?") for k in ks})
        ss = sorted({str(shapes.get(k, "?")) for k in ks})
        print(f"| `{g}` | {len(ks)} | {', '.join(ds)} | {'; '.join(ss)} |")
    by_shard = collections.Counter(wm.values())
    print("shard map: " + ", ".join(f"{s.split('-')[1]}: {n}" for s, n in sorted(by_shard.items())))
    tp = idx.get("metadata", {}).get("total_parameters")
    print(f"parameters (derived from the headers): {numel:,} including the {expert_bias:,} fp32 expert_bias values; "
          f"{numel - expert_bias:,} without them; index total_parameters {tp:,}" if tp else
          f"parameters (derived): {numel:,}")
    print(f"tensor bytes {nbytes:,}; index total_size {idx.get('metadata', {}).get('total_size')}")
    print(f"lm_head.weight stored: {'lm_head.weight' in wm}")
    return {"shapes": shapes, "dtypes": dtypes, "wm": wm}


def sec_bytes(info: dict, cfg: dict) -> None:
    sh = info["shapes"]
    H, E, k, L = cfg["hidden_size"], cfg["num_experts"], cfg["num_experts_per_tok"], cfg["num_hidden_layers"]
    types = cfg["layer_types"]

    def numel(name):
        n = 1
        for d in sh[name]:
            n *= d
        return n

    def i4(n):            # int4 g64 sym: 0.5 B + an f16 scale per 64 = 0.53125 B a weight
        return n * 0.53125
    ex1 = sum(numel(f"model.layers.2.feed_forward.experts.0.{s}.weight") for s in ("w1", "w2", "w3"))
    moe = [i for i in range(L) if i >= cfg["num_dense_layers"]]
    conv = [i for i, t in enumerate(types) if t == "conv"]
    att = [i for i, t in enumerate(types) if t == "full_attention"]
    experts_tok = len(moe) * k * ex1
    conv_tok = sum(numel(f"model.layers.{i}.conv.in_proj.weight") + numel(f"model.layers.{i}.conv.out_proj.weight")
                   for i in conv)
    dense_tok = sum(numel(f"model.layers.{i}.feed_forward.{s}.weight") for i in range(cfg["num_dense_layers"])
                    for s in ("w1", "w2", "w3"))
    att_tok = sum(numel(f"model.layers.{i}.self_attn.{s}.weight") for i in att for s in ("q_proj", "k_proj", "v_proj", "out_proj"))
    router = sum(numel(f"model.layers.{i}.feed_forward.gate.weight") for i in moe)
    V = sh["model.embed_tokens.weight"][0]
    head8 = V * H + V * 4
    rows = [("routed experts (%d x %d x 3 x %d x %d)" % (len(moe), k, H, cfg["moe_intermediate_size"]), "int4 g64", i4(experts_tok)),
            ("conv in_proj + out_proj (%d layers)" % len(conv), "int4 g64", i4(conv_tok)),
            ("dense FFNs (%d)" % cfg["num_dense_layers"], "int4 g64", i4(dense_tok)),
            ("attention q, k, v, o (%d layers)" % len(att), "int4 g64", i4(att_tok)),
            ("routers (%d x %d x %d)" % (len(moe), E, H), "bf16", router * 2),
            ("tied head, int8 per row + fp32 scales", "int8", head8)]
    print("## bytes (derived, per decode token)\n| part | format | MB |\n|---|---|---:|")
    tot = 0
    for what, fmt, b in rows:
        tot += b
        print(f"| {what} | {fmt} | {b / 1e6:,.1f} |")
    print(f"| weights read per token | | {tot / 1e6:,.1f} |")
    kvh = cfg["num_key_value_heads"]
    hd = H // cfg["num_attention_heads"]
    kv_pos = len(att) * kvh * hd * 2 * 2
    print(f"KV bytes per position: {len(att)} layers x {kvh} kv x {hd} x 2 (K, V) x 2 B = {kv_pos:,}; at 131072: "
          f"{kv_pos * 131072 / 1e9:.3f} GB; conv state per sequence {len(conv)} x 2 x {H} x 2 B = {len(conv) * 2 * H * 2:,} B")
    all_experts = len(moe) * E * ex1
    other = conv_tok + dense_tok + att_tok
    print(f"memory: routed experts {i4(all_experts) / 1e9:.3f} GB, other int4 linears {i4(other) / 1e9:.3f} GB, routers "
          f"{router * 2 / 1e9:.3f} GB, embedding bf16 {V * H * 2 / 1e9:.3f} GB, int8 head {head8 / 1e9:.3f} GB")
    act = moe and (len(moe) * k * ex1 + other + router + sum(
        numel(k_) for k_ in sh if k_.endswith(("norm.weight", "layernorm.weight", "conv.conv.weight"))) + V * H)
    print(f"active parameters per token (derived, the head included): {act / 1e9:.3f} B")


# ------------------------------------------------------------------------------------------------
# tokens, template, prompts, licence

def sec_tokens(src: Source) -> dict:
    raw = src.small("tokenizer.json")
    tj = json.loads(raw)
    model = tj["model"]
    vocab = model.get("vocab", {})
    added = tj.get("added_tokens", [])
    ids = sorted(set(vocab.values()) | {a["id"] for a in added})
    contiguous = ids == list(range(len(ids)))
    print(f"## tokens: tokenizer.json sha256 {sha256(raw)}; model {model['type']}, ignore_merges "
          f"{model.get('ignore_merges')}, byte_fallback {model.get('byte_fallback')}; {len(vocab):,} BPE entries, "
          f"{len(model.get('merges', [])):,} merges, {len(added)} added tokens; ids 0..{ids[-1]} contiguous {contiguous} "
          f"-> vocab_used {ids[-1] + 1 if contiguous else '?'}")
    print(f"normalizer {json.dumps(tj.get('normalizer'))}; pre_tokenizer {json.dumps(tj.get('pre_tokenizer'))[:400]}")
    print(f"post_processor {json.dumps(tj.get('post_processor'))[:300]}; decoder {json.dumps(tj.get('decoder'))[:200]}")
    print("| id | content | special |\n|---:|---|---|")
    for a in added:
        if a["special"] or a["content"].startswith("<|") or a["content"] in ("<think>", "</think>"):
            print(f"| {a['id']} | `{a['content']}` | {a['special']} |")
    nonspecial = [a for a in added if not a["special"]]
    print(f"added tokens not special: {len(nonspecial)}: " + ", ".join(f"{a['id']} `{a['content']}`" for a in nonspecial[:40]))
    for fn in ("generation_config.json", "tokenizer_config.json"):
        print(f"{fn}: {json.dumps(src.json(fn), sort_keys=True)}")
    return {"vocab_used": ids[-1] + 1, "added": added}


TEMPLATE_FEATURES = [("namespace", r"namespace\("), ("macro", r"\{%-? macro "), ("is mapping", r"is mapping"),
                     ("is iterable", r"is iterable"), ("is string", r"is string"), ("tojson", r"\| tojson"),
                     ("replace", r"\| replace\("), ("slicing", r"\[\d*:\w*\]|\[:_"), (".endswith", r"\.endswith\("),
                     (".split", r"\.split\("), ("generation block", r"\{%-? generation"), ("raise_exception",
                     r"raise_exception\("), ("loop.index0 / loop.last", r"loop\.(index0|last)"), ("trim", r"\| trim"),
                     ("length", r"\| length"), ("join", r"\| join\("), ("default", r"\| default\("),
                     ("string filter", r"\| string"), ("dict .get / .items", r"\.get\(|\.items\(\)")]


def template_cases() -> list[tuple[str, list, list | None, dict]]:
    tools = [{"type": "function", "function": {"name": "read", "description": "Read a file.", "parameters": {
        "type": "object", "properties": {"filePath": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["filePath"]}}}]

    def call(name, args):
        return {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                                   "function": {"name": name, "arguments": args}}]}
    sys_ = {"role": "system", "content": "You are a coding agent."}
    user = {"role": "user", "content": "Open the file."}
    return [
        ("system + tools + generation prompt", [sys_, user], tools, {}),
        ("no system, tools only", [user], tools, {}),
        ("a call: string with quote / backslash / newline / CR", [sys_, user, call("read", {
            "filePath": "/w/it's\\a\nb\rc"})], tools, {}),
        ("a call: int, float, bool, null", [user, call("f", {"a": 1, "b": 2.5, "c": True, "d": None})], None, {}),
        ("a call: list and nested dict (tojson)", [user, call("f", {"xs": [1, "two", [3]], "m": {"k": "v'q", "n": {"x": 1}}})], None, {}),
        ("two calls in one turn", [user, {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "a", "arguments": {"x": 1}}},
            {"type": "function", "function": {"name": "b", "arguments": {}}}]}], None, {}),
        ("content then calls; a tool result", [user, dict(call("read", {"filePath": "/w/x"}), content="Reading."),
                                               {"role": "tool", "content": "line 1"}], tools, {}),
        ("reasoning after the last user turn (kept)", [user, {"role": "assistant", "content": "Done.",
                                                             "reasoning_content": "think a"}], None, {}),
        ("reasoning before the last user turn (dropped)", [user, {"role": "assistant", "content": "Done.",
                                                                  "reasoning_content": "think a"},
                                                           {"role": "user", "content": "Again."}], None, {}),
        ("preserve_thinking true", [user, {"role": "assistant", "content": "Done.", "reasoning_content": "think a"},
                                    {"role": "user", "content": "Again."}], None, {"preserve_thinking": True}),
        ("content as a list of text parts", [{"role": "user", "content": [{"type": "text", "text": "a"},
                                                                        {"type": "text", "text": "b"}]}], None, {}),
    ]


def sec_template(src: Source) -> None:
    raw = src.small("chat_template.jinja")
    text = raw.decode()
    print(f"## template: chat_template.jinja sha256 {sha256(raw)}, {len(text.splitlines())} lines")
    print("features minja must render: " + "; ".join(
        f"{n} (line {', '.join(str(i + 1) for i, ln in enumerate(text.splitlines()) if re.search(p, ln))[:40]})"
        for n, p in TEMPLATE_FEATURES if re.search(p, text)))
    m = re.search(r"macro format_arg_value.*?endmacro", text, re.S)
    print("the pythonic call grammar (format_arg_value, the template's own lines):")
    for ln in m.group(0).splitlines():
        print("    " + ln.strip())
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("(no transformers: the renders need the 5.19.0 site)")
        return
    tok = AutoTokenizer.from_pretrained(src.dir)
    for what, msgs, tools, kw in template_cases():
        add = msgs[-1]["role"] != "assistant"
        r = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=add, tokenize=False, **kw)
        ids = tok(r, add_special_tokens=False)["input_ids"]
        print(f"### render: {what} (add_generation_prompt {add}{', ' + json.dumps(kw) if kw else ''}; {len(ids)} ids)")
        print("```text\n" + r + "\n```")


def sec_prompts(src: Source) -> None:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_bytes(src.small("tokenizer.json"))
    d = os.path.join(REPO, "tests", "golden", "prompts")
    print("## prompts: tests/golden/prompts re-tokenised by LFM2's tokenizer")
    for name in ("prose", "code", "long"):
        p = os.path.join(d, f"{name}.txt")
        if not os.path.exists(p):
            continue
        text = open(p, encoding="utf-8").read().rstrip("\n")
        new = tok.encode(text, add_special_tokens=False).ids
        old_p = os.path.join(d, f"{name}.ids")
        old = [int(x) for x in open(old_p).read().split()] if os.path.exists(old_p) else None
        print(f"{name}.txt: {len(new)} LFM2 ids (Qwen's committed {len(old) if old else '-'}; equal {new == old})")


def sec_licence(src: Source) -> None:
    raw = src.small("LICENSE")
    text = raw.decode(errors="replace")
    print(f"## licence: LICENSE sha256 {sha256(raw)}, {len(text.splitlines())} lines")
    for ln in text.splitlines():
        s = ln.strip()
        if re.match(r"^(\d+\.|[A-Z][A-Za-z ]+ License|LFM)", s) and len(s) < 120:
            print("  " + s)
    m = re.search(r"[^.]*Threshold[^.]*\.", text)
    if m:
        print("  threshold: " + " ".join(m.group(0).split()))


# ------------------------------------------------------------------------------------------------
# measured: torch's tie order, the conv's accumulation

def sec_ties(rows: int = 50) -> None:
    import torch
    g = torch.Generator().manual_seed(0)
    lower = higher = stable = 0
    for _ in range(rows):
        v = torch.rand(64, generator=g)
        order = v.argsort(descending=True)
        a, b = sorted(order[3:5].tolist())
        v[b] = v[a]
        idx = v.topk(4).indices.tolist()
        lower += (a in idx) and (b not in idx)
        higher += (b in idx) and (a not in idx)
        stable += idx == v.topk(4).indices.tolist()
    print(f"## ties: torch {torch.__version__} CPU topk(k=4) over 64 fp32 with an exact tie at the 4th / 5th, {rows} "
          f"rows: the lower id wins {lower}, the higher id {higher}; repeat calls identical {stable}/{rows}")
    # the router's real shape: a [T][64] batch (topk along the last dim), ties planted per row
    T = rows
    v = torch.rand(T, 64, generator=g)
    pairs = []
    for r in range(T):
        o = v[r].argsort(descending=True)
        a, b = sorted(o[3:5].tolist())
        v[r, b] = v[r, a]
        pairs.append((a, b))
    idx = v.topk(4, dim=-1).indices
    lo = sum(int(a in idx[r].tolist() and b not in idx[r].tolist()) for r, (a, b) in enumerate(pairs))
    print(f"   batched [{T}][64]: the lower id wins {lo}/{T} (the engine's rule is the lower id; the reference's "
          f"choice on an exact tie is torch's - such rows are undetermined: route.gap == 0)")


def sec_conv(rows: int, H: int = 2048) -> None:
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)               # depthwise bf16 conv1d is slow multi-threaded; bitwise across threads
    g = torch.Generator().manual_seed(0)
    T = 1000
    n_rows = n_elem = n_desc = n_bf = n_pair = 0
    total = 0
    while total < rows:
        t = min(T, rows - total)
        # Bx as the model makes it: a bf16 product of two bf16 rows; the taps bf16 ~ the real conv weights' scale
        bx = (torch.randn(t + 2, H, generator=g) * torch.randn(t + 2, H, generator=g)).to(torch.bfloat16)
        w = (0.5 * torch.randn(H, 3, generator=g)).to(torch.bfloat16)
        bx[:2] = 0
        x = bx.t().unsqueeze(0)                                                # [1][H][t + 2]
        out = F.conv1d(x, w.unsqueeze(1), None, padding=2, groups=H)[0, :, 2:t + 2].t()   # rows of positions 2..t+1
        xf, wf = bx.float(), w.float()
        p0, p1, p2 = wf[:, 0] * xf[0:t], wf[:, 1] * xf[1:t + 1], wf[:, 2] * xf[2:t + 2]
        asc = ((p0 + p1) + p2).to(torch.bfloat16)                              # taps ascending (= fma: exact products)
        desc = ((p2 + p1) + p0).to(torch.bfloat16)
        pair = (p0 + (p1 + p2)).to(torch.bfloat16)
        bf = ((p0.to(torch.bfloat16).float() + p1).to(torch.bfloat16).float() + p2).to(torch.bfloat16)
        d = asc != out
        n_rows += int(d.any(-1).sum())
        n_elem += int(d.sum())
        n_desc += int((desc != out).any(-1).sum())
        n_pair += int((pair != out).any(-1).sum())
        n_bf += int((bf != out).any(-1).sum())
        total += t
    print(f"## conv: torch {torch.__version__} CPU bf16 depthwise F.conv1d (3 taps, padding 2) on {total:,} rows x {H} "
          f"channels: fp32 taps-ascending ((w0 x0 + w1 x1) + w2 x2, one rounding) differs on {n_rows} rows "
          f"({n_elem} elements); descending on {n_desc} rows; w0 x0 + (w1 x1 + w2 x2) on {n_pair}; a bf16 accumulator "
          f"(rounding after each add) on {n_bf}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--section", default=",".join(SECTIONS))
    ap.add_argument("--tiny")
    ap.add_argument("--rows", type=int, default=100000)
    ap.add_argument("--cache")
    a = ap.parse_args(argv)
    want = a.section.split(",")
    bad = [s for s in want if s not in SECTIONS]
    if bad:
        raise SystemExit(f"--section: {bad} not in {SECTIONS}")
    src = Source(a.snapshot, a.cache)
    cfg = src.json("config.json")
    info = None
    for s in want:
        if s == "cite":
            sec_cite()
        elif s == "checkpoint":
            info = sec_checkpoint(src)
        elif s == "bytes":
            sec_bytes(info or sec_checkpoint(src), cfg)
        elif s == "tokens":
            sec_tokens(src)
        elif s == "template":
            sec_template(src)
        elif s == "prompts":
            sec_prompts(src)
        elif s == "licence":
            sec_licence(src)
        elif s == "ties":
            sec_ties()
        elif s == "conv":
            sec_conv(a.rows)
        print()


if __name__ == "__main__":
    main()
