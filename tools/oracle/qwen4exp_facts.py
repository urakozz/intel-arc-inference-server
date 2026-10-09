#!/usr/bin/env python3
"""Spec 21a Task 2: the Qwen3.8-Flash-Next (qwen4_exp) checkpoint facts, from SMALL FILES ONLY.

    qwen4exp_facts.py <snapshot dir | hf:REPO[@REV]> [--qwen38-tokenizer FILE] [--cache DIR]
                      [--section config,ple,tensors,quant,tokens,bytes]

Reads config.json, model.safetensors.index.json, every shard's safetensors HEADER (8 + n bytes), the
three I64 PLE hash tensors (by their byte range: ~280 bytes), generation_config.json,
tokenizer_config.json, tokenizer.json, chat_template.jinja and the model card. A local snapshot is
read from disk (only the header bytes of a shard); `hf:REPO` reads the same bytes by HTTP range
requests (nothing is downloaded whole except the small files; headers are cached in --cache). No
torch: this runs on the Mac host as well as in the oracle image. docs/probe-qwen4exp-2026-10-09.md
quotes its tables.

Sections:
  config  the text config the reference builds (layer types, widths, PLE, QSA, MoE, MTP keys)
  ple     the 16 n-gram heads: prime (recomputed: the first `ple_index * 16 + h + 1`-th prime after
          base - 1), offset, rows [offset, offset + prime) and the shards they fall in; the padded
          table; layer_multipliers recomputed by splitmix64 (seed 1234) - both held to the checkpoint's
          I64 tensors when they can be read
  tensors the tensor list grouped (layer index -> N), dtype, shape, count, bytes
  quant   quantization_config (Intel's: packing, group, sym, ignore list, extra_config); one routed
          expert's qzeros words (sym = 0x77777777 under auto_gptq's zero - 1 convention)
  tokens  tokenizer.json against Qwen3.8's (byte sha256, then vocab / merges / added tokens /
          normalizer / pre_tokenizer / decoder), the chat template's sha256, generation_config
  bytes   spec 21 §3's per-token decode bytes and the planner's per-layer bytes, derived
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]   # tools/oracle/tokenize.py

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import urllib.request  # noqa: E402

EOS = 248044
QWEN38_TEMPLATE_SHA = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"  # chat_template.cc:142


# ------------------------------------------------------------------------------------------------
# the PLE hash constants, restated from modeling_qwen4_exp.py M:1026-1069 in pure Python integers

_MASK64 = (1 << 64) - 1
_GAMMA, _M1, _M2, _PRIME_1 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 10007


def splitmix64(v: int) -> int:
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def layer_multipliers(vocab: int, ngram: int, ple_index: int, seed: int) -> list[int]:
    """M:1040-1049: odd, at most (2^63 - 1) // vocab, so t * m never overflows int64."""
    mmax = ((1 << 63) - 1) // max(vocab, 1)
    half = max(1, mmax // 2)
    base = seed + _PRIME_1 * ple_index
    return [2 * (splitmix64((base + _GAMMA * (i + 1)) & _MASK64) % half) + 1 for i in range(ngram)]


def is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    for d in range(3, math.isqrt(v) + 1, 2):
        if v % d == 0:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not is_prime(p):
            p += 1
    return p


def head_table(tc: dict, ple_index: int = 0):
    """(sizes, offsets, total, padded) for one PLE layer: M:1088-1105."""
    heads = (tc["ngram_size"] - 1) * tc["heads_per_ngram"]
    sizes, offsets, total = [], [], 0
    p = tc["ngram_vocab_size_base"] - 1
    # head h's size is the (ple_index * heads + h + 1)-th prime after base - 1; walking the primes once
    # gives the same numbers as calling _find_nth_prime_after per head
    p = nth_prime_after(p, ple_index * heads)
    for _ in range(heads):
        p = nth_prime_after(p, 1)
        sizes.append(p)
        offsets.append(total)
        total += p
    div = tc["make_ngram_vocab_size_divisible_by"]
    return sizes, offsets, total, math.ceil(total / div) * div


# ------------------------------------------------------------------------------------------------
# where the bytes come from

class Source:
    def __init__(self, where: str, cache: str | None):
        self.remote = where.startswith("hf:")
        if self.remote:
            repo = where[3:]
            self.repo, _, self.rev = repo.partition("@")
            self.rev = self.rev or "main"
            self.name = f"{self.repo}@{self.rev}"
        else:
            self.dir = where
            self.name = os.path.abspath(where)
        self.cache = cache
        self._headers: dict[str, dict] = {}

    def _url(self, fn: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{fn}"

    def _get(self, fn: str, a: int | None = None, b: int | None = None) -> bytes:
        req = urllib.request.Request(self._url(fn))
        if a is not None:
            req.add_header("Range", f"bytes={a}-{b}")
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    def exists(self, fn: str) -> bool:
        if not self.remote:
            return os.path.exists(os.path.join(self.dir, fn))
        try:
            self._get(fn, 0, 0)
            return True
        except Exception:
            return False

    def small(self, fn: str) -> bytes:
        if not self.remote:
            with open(os.path.join(self.dir, fn), "rb") as f:
                return f.read()
        if self.cache:
            p = os.path.join(self.cache, self.repo.replace("/", "--"), fn)
            if os.path.exists(p):
                return open(p, "rb").read()
        data = self._get(fn)
        if self.cache:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(data)
        return data

    def json(self, fn: str):
        return json.loads(self.small(fn))

    def header(self, shard: str) -> dict:
        """The shard's safetensors header (8-byte length, then JSON), and the data start offset."""
        if shard in self._headers:
            return self._headers[shard]
        cp = os.path.join(self.cache, self.repo.replace("/", "--"), shard + ".header.json") \
            if (self.remote and self.cache) else None
        if cp and os.path.exists(cp):
            h = json.load(open(cp))
        else:
            if self.remote:
                n = struct.unpack("<Q", self._get(shard, 0, 7))[0]
                h = json.loads(self._get(shard, 8, 8 + n - 1))
            else:
                with open(os.path.join(self.dir, shard), "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    h = json.loads(f.read(n))
            h["__data_start__"] = 8 + n
            if cp:
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                json.dump(h, open(cp, "w"))
        self._headers[shard] = h
        return h

    def tensor_bytes(self, shard: str, name: str) -> tuple[str, list[int], bytes]:
        h = self.header(shard)
        e = h[name]
        a, b = e["data_offsets"]
        s = h["__data_start__"]
        if self.remote:
            data = self._get(shard, s + a, s + b - 1)
        else:
            with open(os.path.join(self.dir, shard), "rb") as f:
                f.seek(s + a)
                data = f.read(b - a)
        return e["dtype"], e["shape"], data

    def weight_map(self) -> dict[str, str]:
        if self.exists("model.safetensors.index.json"):
            return self.json("model.safetensors.index.json")["weight_map"]
        h = self.header("model.safetensors")
        return {k: "model.safetensors" for k in h if not k.startswith("__")}


def text_cfg(cfg: dict) -> dict:
    return cfg["text_config"] if "text_config" in cfg else cfg


def lm_prefix(names) -> str:
    return "model.language_model." if any(n.startswith("model.language_model.") for n in names) else "model."


# ------------------------------------------------------------------------------------------------
# sections

def sec_config(src: Source) -> dict:
    cfg = src.json("config.json")
    tc = text_cfg(cfg)
    lt = tc.get("layer_types") or []
    kinds = sorted(set(lt))
    print(f"## config ({src.name})")
    print(f"architectures {cfg.get('architectures')}, model_type {cfg.get('model_type')} / text {tc.get('model_type')}, "
          f"saved by transformers {cfg.get('transformers_version')}")
    print(f"layers {tc['num_hidden_layers']}: " + ", ".join(f"{k} x{lt.count(k)} at "
          f"{[i for i, t in enumerate(lt) if t == k][:4]}..." for k in kinds))
    keys = ["hidden_size", "vocab_size", "hc_count", "hc_lowrank", "head_dim", "num_attention_heads",
            "num_key_value_heads", "partial_rotary_factor", "rope_parameters", "linear_num_key_heads",
            "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
            "output_gate_type", "indexer_n_heads", "indexer_kv_heads", "indexer_head_dim", "indexer_budget",
            "indexer_compress_ratio", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
            "shared_expert_intermediate_size", "norm_topk_prob", "ple_layer_ids", "ple_embed_dim",
            "ple_conv_kernel_size", "ngram_size", "heads_per_ngram", "ngram_vocab_size_base",
            "make_ngram_vocab_size_divisible_by", "split_ngram_parts", "seed", "rms_norm_eps", "eos_token_id",
            "bos_token_id", "max_position_embeddings", "tie_word_embeddings", "mtp", "mtp_num_hidden_layers",
            "mtp_use_dedicated_embeddings", "index_share_for_mtp_iteration"]
    for k in keys:
        print(f"  {k}: {json.dumps(tc.get(k, '<absent>'))}")
    return cfg


def sec_ple(src: Source, wm: dict) -> None:
    tc = text_cfg(src.json("config.json"))
    sizes, offsets, total, padded = head_table(tc, 0)
    parts = tc["split_ngram_parts"]
    rows = padded // parts
    pre = lm_prefix(wm)
    ple_l = tc["ple_layer_ids"][0] - 1
    base = f"{pre}layers.{ple_l}.ple.ple_embedding."
    shard_names = [k for k in wm if k.startswith(base + "ngram_embedding.shard_")]
    print(f"## PLE (layer_idx {ple_l} = ple_layer_ids {tc['ple_layer_ids']} one-indexed)")
    print(f"16 heads (8 bigram + 8 trigram) x {tc['ple_embed_dim'] // 16} dims; total {total} rows, padded to a "
          f"multiple of {tc['make_ngram_vocab_size_divisible_by']}: {padded} = {parts} shards x {rows} rows "
          f"({len(shard_names)} shard tensors in the checkpoint)")
    print("| head | n-gram | prime (rows) | offset | first row's shard:row | last row's shard:row |")
    print("|---:|---:|---:|---:|---:|---:|")
    for h, (s, o) in enumerate(zip(sizes, offsets)):
        a, b = o, o + s - 1
        print(f"| {h} | {2 + h // tc['heads_per_ngram']} | {s} | {o} | {a // rows}:{a % rows} | {b // rows}:{b % rows} |")
    mult = layer_multipliers(tc["vocab_size"], tc["ngram_size"], 0, tc.get("seed", 1234))
    print(f"layer_multipliers recomputed (splitmix64, seed {tc.get('seed', 1234)} - the config default when absent, C:155): {mult} "
          f"(odd: {all(m % 2 for m in mult)}, max product {max(mult) * (tc['vocab_size'] - 1)} < 2^63-1: "
          f"{max(mult) * (tc['vocab_size'] - 1) < (1 << 63) - 1})")
    for nm, want in (("layer_multipliers", mult), ("ngram_heads_vocab_sizes", sizes), ("ngram_heads_offsets", offsets)):
        k = base + nm
        if k not in wm:
            print(f"  {nm}: not in the checkpoint")
            continue
        try:
            dt, sh, data = src.tensor_bytes(wm[k], k)
        except Exception as e:      # noqa: BLE001 - a missing local shard is reported, not fatal
            print(f"  {nm}: {wm[k]} not readable here ({type(e).__name__}); recomputed only")
            continue
        got = list(struct.unpack(f"<{len(data) // 8}q", data))
        print(f"  {nm} {dt} {sh} from {wm[k]}: {got} -> equal to the formula: {got == want}")


def group_name(n: str) -> str:
    n = re.sub(r"\.layers\.\d+\.", ".layers.N.", n)
    n = re.sub(r"\.experts\.\d+\.", ".experts.E.", n)
    return re.sub(r"\.shard_\d+\.", ".shard_K.", n)


def sec_tensors(src: Source, wm: dict) -> dict:
    nb = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "I8": 1, "U8": 1}
    info = {}
    for shard in sorted(set(wm.values())):
        h = src.header(shard)
        for k, v in h.items():
            if k.startswith("__"):
                continue
            info[k] = (v["dtype"], v["shape"], math.prod(v["shape"]) * nb[v["dtype"]])
    missing = sorted(set(wm) - set(info))
    print(f"## tensors: {len(info)} in {len(set(wm.values()))} shards; index-only names {len(missing)}; "
          f"{sum(b for _, _, b in info.values()) / 1e9:.3f} GB")
    groups: dict = {}
    for k, (dt, sh, b) in info.items():
        if k.startswith("model.visual."):
            g = ("model.visual.* (vision tower, not read by the text reference)", dt, "-")
        else:
            g = (group_name(k), dt, json.dumps(sh))
        e = groups.setdefault(g, [0, 0])
        e[0] += 1
        e[1] += b
    print("| tensor (layer N, expert E, shard K) | dtype | shape | count | GB |")
    print("|---|---|---|---:|---:|")
    for (g, dt, sh), (c, b) in sorted(groups.items()):
        print(f"| `{g}` | {dt} | {sh} | {c} | {b / 1e9:.3f} |")
    return info


def sec_quant(src: Source, wm: dict, cfg: dict) -> None:
    q = cfg.get("quantization_config")
    print("## quantization_config")
    if not q:
        print("none (an unquantised checkpoint)")
        return
    ex = q.get("extra_config", {})
    print(json.dumps({k: v for k, v in q.items() if k != "extra_config"}))
    kinds: dict = {}
    for k, v in ex.items():
        kinds.setdefault(json.dumps(v, sort_keys=True), []).append(k)
    print(f"extra_config: {len(ex)} entries; " + "; ".join(f"{len(v)} x {k}" for k, v in kinds.items()))
    pats = [k for k in ex if not re.search(r"\.\d+\.", k)]
    print(f"  pattern entries: {pats}")
    pre = lm_prefix(wm)
    k = f"{pre}layers.0.mlp.experts.0.gate_proj.qzeros"
    if k in wm:
        dt, sh, data = src.tensor_bytes(wm[k], k)
        words = set(struct.unpack(f"<{len(data) // 4}I", data))
        print(f"  {k} {dt} {sh}: words {sorted(hex(w) for w in words)} (sym under auto_gptq's zero-1 convention: "
              f"{{0x77777777}} -> dequant (q - 8) x scale: {words == {0x77777777}})")


def tok_summary(tj: dict) -> dict:
    m = tj["model"]
    merges = [x if isinstance(x, str) else " ".join(x) for x in m.get("merges", [])]
    return {"model.type": m.get("type"), "vocab": m.get("vocab"), "merges": merges,
            "model.other": {k: v for k, v in m.items() if k not in ("vocab", "merges")},
            "added_tokens": tj.get("added_tokens"), "normalizer": tj.get("normalizer"),
            "pre_tokenizer": tj.get("pre_tokenizer"), "post_processor": tj.get("post_processor"),
            "decoder": tj.get("decoder")}


def sec_tokens(src: Source, q38: str | None) -> None:
    print("## tokens and template")
    gen = src.json("generation_config.json")
    print(f"generation_config.json: {json.dumps(gen, sort_keys=True)}")
    for fn in ("chat_template.jinja",):
        if src.exists(fn):
            sha = hashlib.sha256(src.small(fn)).hexdigest()
            print(f"{fn}: sha256 {sha} (Qwen3.8's / Agnes's c3cf9e34...: {sha == QWEN38_TEMPLATE_SHA})")
    tcfg = src.json("tokenizer_config.json")
    print(f"tokenizer_config.json: class {tcfg.get('tokenizer_class')}, eos {tcfg.get('eos_token')}, pad "
          f"{tcfg.get('pad_token')}, chat_template key present: {'chat_template' in tcfg}")
    if not src.exists("tokenizer.json"):
        print("tokenizer.json: absent")
        return
    raw = src.small("tokenizer.json")
    tj = json.loads(raw)
    a = tok_summary(tj)
    n_ids = len(a["vocab"]) + len(a["added_tokens"])
    added = a["added_tokens"]
    print(f"tokenizer.json: sha256 {hashlib.sha256(raw).hexdigest()}, {len(raw)} B; vocab {len(a['vocab'])}, "
          f"merges {len(a['merges'])}, added {len(added)} (ids {added[0]['id']}..{added[-1]['id']}), len {n_ids}")
    if not q38:
        return
    rb = open(q38, "rb").read()
    b = tok_summary(json.loads(rb))
    print(f"Qwen3.8's tokenizer.json ({q38}): sha256 {hashlib.sha256(rb).hexdigest()}, {len(rb)} B; byte-identical: "
          f"{raw == rb}")
    for k in a:
        same = a[k] == b[k]
        extra = ""
        if not same and isinstance(a[k], dict) and isinstance(b[k], dict) and k == "vocab":
            d1 = {t: i for t, i in a[k].items() if b[k].get(t) != i}
            d2 = {t: i for t, i in b[k].items() if a[k].get(t) != i}
            extra = f" ({len(d1)} entries differ / {len(d2)} the other way)"
        print(f"  {k}: equal {same}{extra}")
    if a == b:
        print("  -> the same tokenizer (formatting differs only): Qwen3.8's .ids files are valid ids for this model")


def sec_bytes(info: dict, cfg: dict) -> None:
    tc = text_cfg(cfg)
    H, I, E, k = tc["hidden_size"], tc["moe_intermediate_size"], tc["num_experts"], tc["num_experts_per_tok"]
    L = tc["num_hidden_layers"]
    lt = tc["layer_types"]
    nq = sum(1 for t in lt if t != "linear_attention")
    ng = L - nq
    HC = tc["hc_count"] * H

    def i4(n_out, n_in, g=64):           # int4 sym + f16 scales per group of g along K
        return n_out * n_in // 2 + (n_in // g) * n_out * 2

    def bf(n_out, n_in):
        return n_out * n_in * 2
    expert = 2 * i4(I, H) + i4(H, I)          # gate, up [I][H] and down [H][I]
    gdn = {"in_proj_qkv": (10240, H), "in_proj_z": (6144, H), "out_proj": (H, 6144)}
    qsa = {"q_proj": (2 * tc["num_attention_heads"] * tc["head_dim"], H),
           "k_proj": (tc["num_key_value_heads"] * tc["head_dim"], H),
           "v_proj": (tc["num_key_value_heads"] * tc["head_dim"], H),
           "o_proj": (H, tc["num_attention_heads"] * tc["head_dim"])}
    hc_one = bf(tc["hc_lowrank"], HC) + bf(HC, tc["hc_lowrank"]) + bf(tc["hc_count"], HC) + HC * 2
    mixer = bf(tc["hc_lowrank"], HC) + bf(HC, tc["hc_lowrank"]) + HC * 2
    rows = [
        ("routed experts (L x top-k x 3 x H x I)", "int4 g64", L * k * expert),
        ("shared experts", "int4 g64", L * 3 * i4(I, H)),
        ("routers (E x H)", "bf16", L * bf(E, H)),
        ("hyper-connections (2 per layer + the final mixer)", "bf16", 2 * L * hc_one + mixer),
        ("GDN projections (int4) + a / b (bf16)", "int4 g64 + bf16",
         ng * (sum(i4(o, i) for o, i in gdn.values()) + 2 * bf(tc["linear_num_value_heads"], H))),
        ("QSA projections (int4) + indexer (bf16)", "int4 g64 + bf16",
         nq * (sum(i4(o, i) for o, i in qsa.values()) + bf(5 * tc["indexer_head_dim"], H))),
        ("PLE projections key 10240 x 2560 + value 2560 x 2560", "bf16", bf(HC, tc["ple_embed_dim"]) + bf(H, tc["ple_embed_dim"])),
        ("lm_head (spec 9: int8 + an f32 scale a row)", "int8", tc["vocab_size"] * H + tc["vocab_size"] * 4),
    ]
    print("## derived bytes (spec 21 §3), per decoded token")
    print("| part | format | GB |")
    print("|---|---|---:|")
    for n, f, b in rows:
        print(f"| {n} | {f} | {b / 1e9:.3f} |")
    print(f"| **total** | | **{sum(b for _, _, b in rows) / 1e9:.3f}** |")
    print(f"decode roofline at 590 GB/s: {590 / (sum(b for _, _, b in rows) / 1e9):.1f} t/s (derived)")
    # the planner's per-layer bytes with Intel's interim forms (21b Task 2): experts at int4 g64 after the exact
    # g128 -> g64 scale expansion, everything else bf16 as shipped
    ex_layer = E * expert
    gdn_bf = sum(bf(o, i) for o, i in gdn.values()) + 2 * bf(tc["linear_num_value_heads"], H) \
        + 10240 * tc["linear_conv_kernel_dim"] * 2 + 3 * tc["linear_num_value_heads"] * 2 + 128 * 2
    qsa_bf = sum(bf(o, i) for o, i in qsa.values()) + bf(5 * tc["indexer_head_dim"], H) + 2 * 256 * 2 + 2 * 128 * 2
    common = 2 * hc_one + bf(E, H) + 3 * bf(I, H) + H * 2
    print("## the planner's per-layer bytes, Intel's interim forms (experts g64 after the expansion, the rest bf16)")
    print(f"  experts {ex_layer} B ({ex_layer / 1e9:.3f} GB); GDN dense {gdn_bf} B; QSA dense {qsa_bf} B; "
          f"HC + router + shared {common} B")
    print(f"  a GDN layer {(ex_layer + gdn_bf + common) / 1e9:.3f} GB, a QSA layer {(ex_layer + qsa_bf + common) / 1e9:.3f} GB "
          f"(derived)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("where")
    ap.add_argument("--qwen38-tokenizer", help="Qwen3.8's tokenizer.json (tok::default_tokenizer_json()'s file)")
    ap.add_argument("--cache", help="where hf: headers and small files are cached")
    ap.add_argument("--section", default="config,ple,tensors,quant,tokens,bytes")
    a = ap.parse_args(argv)
    src = Source(a.where, a.cache)
    want = set(a.section.split(","))
    cfg = src.json("config.json")
    wm = src.weight_map()
    if "config" in want:
        sec_config(src)
    if "ple" in want:
        sec_ple(src, wm)
    info = sec_tensors(src, wm) if ("tensors" in want or "bytes" in want) else {}
    if "quant" in want:
        sec_quant(src, wm, cfg)
    if "tokens" in want:
        sec_tokens(src, a.qwen38_tokenizer)
    if "bytes" in want:
        sec_bytes(info, cfg)


if __name__ == "__main__":
    sys.exit(main())
