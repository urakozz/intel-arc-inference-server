#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A synthetic Nemotron 3.5 Bolt checkpoint at the REAL widths, in either int4 spelling (spec 25b Task 1).

    make_synth.py <out-dir> --layers N --form gptq-g128|ar-g64 [--seed 0] [--tokenizer <dir>]
                  [--expand-from <g128 synthetic dir>]

Writes `config.json` (tests/model/bolt/make_config.py: the real omni wrapper + `llm_config`, `num_hidden_layers` is not a
key of this model - the count is `block_configs`' length - truncated to the first N blocks; N in 8..88: the first
attention block is 7, R's cache wants one), the real tensor names (`language_model.model.layers.<l>.{norm.weight,
mixer.*}`, `embeddings`, `norm_f`, `language_model.lm_head.weight` - bolt_ref.expected_names, checked at the end), one
shard a layer (shard 1: embeddings, norm_f, lm_head), `model_extra_tensors.safetensors` with a few vision / projector
tensors the loader must skip by name (`vision_model.*`, `vision_projector.*`, `mlp1.*`) and one of them inside a
language shard, the tokenizer files and chat template copied from the HF cache at run time (the engine reads none;
tools/oracle does), `generation_config.json` (SergiioB's: eos 11).

  gptq-g128   SergiioB's spelling: `quantization_config` the published one (quant_method gptq, group 128, the 44 `-:`
              exclusions and the restating `+:` out_proj rule), every linear `.qweight` I32 [K/8][N] / `.scales` F16
              [K/128][N] SIGNED (AutoRound's: s = f16(-e / 8), e the group's value of largest magnitude, so about half
              the groups are negative) / `.qzeros` I32 [K/128][N/8] = 0x77777777 / `.g_idx` I32 [K] = i // 128
  ar-g64      ours (25q): AutoRound's own writer's spelling (auto-round, auto_round:auto_gptq), group 64, no g_idx, scales
              F16 [K/64][N] signed. By default packed from the SAME random weights at g64 (a native g64 checkpoint);
              `--expand-from DIR` instead writes the EXACT g64 expansion of the g128 synthetic in DIR (every g128 scale
              row written to both g64 rows it covers, the nibbles untouched) - the pair whose device bytes and logits
              must be identical (spec 25b Review Focus 1)

Real widths: hidden 4096, Mamba in_proj 18560 / out_proj 8192 -> 4096, attention q 4096 / k, v 256 / o 4096, latent 1024,
shared expert 5376, the real E and I of the first N blocks (make_config.py), the router [E][4096] bf16. The kept tensors
at the checkpoint's dtypes: `A_log` / `D` / `dt_bias` / `e_score_correction_bias` F32 (holding bf16 values, the first
three), conv, norms, router, embeddings, lm_head bf16. The draws are tools/oracle/bolt_make_tiny.py's (its ranges: dt_bias
in [-10.5, 8.5], A_log in [-4.7, 4.4], the router bias ~ an offset a layer + N(0, 0.02)), its `pack_int4` packs them.
One layer is drawn, packed and written at a time. N 8 is the smallest (~3.9 GB, ESTIMATED), N 12 the default (~5.5 GB).

Needs torch + safetensors + (through bolt_make_tiny / bolt_ref) transformers 5.19.0: run it in the oracle image
(tools/oracle/README.md, "The Nemotron 3.5 Bolt reference"), the Mac under /tmp/b70-cpu-model.lock. Afterwards
`check.py <out-dir>` must print ACCEPTED.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_ORACLE = os.path.join(_REPO, "tools", "oracle")

import argparse  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

FORMS = ("gptq-g128", "ar-g64")
F32_NAMES = (".A_log", ".D", ".dt_bias", ".e_score_correction_bias")
VISION_PREFIXES = ("vision_model.", "vision_projector.", "mlp1.")
SMALL_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def _load(path: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_config():
    return _load(os.path.join(_REPO, "tests", "model", "bolt", "make_config.py"), "bolt_make_config")


def make_tiny():
    """bolt_make_tiny's packers (pack_int4) and draws' ranges - reused until 25q's land (24b's precedent)."""
    if _ORACLE not in sys.path:
        sys.path.append(_ORACLE)
    return _load(os.path.join(_ORACLE, "bolt_make_tiny.py"), "bolt_make_tiny")


def bolt_ref():
    if _ORACLE not in sys.path:
        sys.path.append(_ORACLE)
    return _load(os.path.join(_ORACLE, "bolt_ref.py"), "bolt_ref")


# ------------------------------------------------------------------------------------------------
# the draws (R's names, `model.` trunk), one layer at a time

def _bf(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16)


def _uni(n: int, lo: float, hi: float, g: torch.Generator) -> torch.Tensor:
    t = lo + (hi - lo) * torch.rand(n, generator=g)
    if n >= 2:
        t[0], t[-1] = lo, hi
    return t


def _norm(n: int, g: torch.Generator) -> torch.Tensor:
    return _bf(1.0 + 0.1 * torch.randn(n, generator=g))


def _lin(n: int, k: int, g: torch.Generator) -> torch.Tensor:
    return _bf(torch.randn(n, k, generator=g) / math.sqrt(k))


def layer_tensors(i: int, block: dict, seed: int, wid: dict) -> dict:
    """Layer i's tensors, R's names, bf16 (fp32 for the kept F32 ones); `wid` the real widths."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + 17 + i)
    H = wid["hidden"]
    p = f"model.layers.{i}."
    out = {p + "norm.weight": _norm(H, g)}
    m = p + "mixer."
    t = block["block_type"]
    if t == "mamba":
        nh = wid["mamba_num_heads"]
        out[m + "A_log"] = _bf(_uni(nh, -4.7, 4.4, g)).float()
        out[m + "D"] = _bf(1.0 + 0.3 * torch.randn(nh, generator=g)).float()
        out[m + "dt_bias"] = _bf(_uni(nh, -10.5, 8.5, g)).float()
        out[m + "conv1d.weight"] = _bf(0.5 * torch.randn(wid["conv_dim"], 1, wid["conv_kernel"], generator=g))
        out[m + "conv1d.bias"] = _bf(0.2 * torch.randn(wid["conv_dim"], generator=g))
        out[m + "norm.weight"] = _norm(wid["d_inner"], g)
        out[m + "in_proj.weight"] = _lin(wid["in_proj"], H, g)
        out[m + "out_proj.weight"] = _lin(H, wid["d_inner"], g)
    elif t == "attention":
        qn, kvn = wid["q_heads"] * wid["head_dim"], wid["kv_heads"] * wid["head_dim"]
        out[m + "q_proj.weight"] = _lin(qn, H, g)
        out[m + "k_proj.weight"] = _lin(kvn, H, g)
        out[m + "v_proj.weight"] = _lin(kvn, H, g)
        out[m + "o_proj.weight"] = _lin(H, qn, g)
    else:
        E, I, L, S = block["n_routed_experts"], block["moe_intermediate_size"], wid["latent"], wid["shared"]
        out[m + "gate.weight"] = _bf(2.0 * torch.randn(E, H, generator=g) / math.sqrt(H))
        out[m + "gate.e_score_correction_bias"] = (min(4.5 + 1.2 * i, 48.0) + 0.02 * torch.randn(E, generator=g)).float()
        out[m + "fc1_latent_proj.weight"] = _lin(L, H, g)
        out[m + "fc2_latent_proj.weight"] = _lin(H, L, g)
        out[m + "shared_experts.up_proj.weight"] = _lin(S, H, g)
        out[m + "shared_experts.down_proj.weight"] = _lin(H, S, g)
        for e in range(E):
            out[f"{m}experts.{e}.up_proj.weight"] = _lin(I, L, g)
            out[f"{m}experts.{e}.down_proj.weight"] = _lin(L, I, g)
    return out


def top_tensors(seed: int, wid: dict) -> dict:
    g = torch.Generator().manual_seed(seed * 1_000_003 + 5)
    V, H = wid["vocab"], wid["hidden"]
    return {"model.embeddings.weight": _bf(torch.randn(V, H, generator=g)),
            "model.norm_f.weight": _norm(H, g),
            "lm_head.weight": _bf(torch.randn(V, H, generator=g) / math.sqrt(H))}


def widths(llm: dict) -> dict:
    nh, hd = llm["mamba_num_heads"], llm["mamba_head_dim"]
    ng, st = llm["n_groups"], llm["ssm_state_size"]
    d_inner = nh * hd
    conv_dim = d_inner + 2 * ng * st
    return {"hidden": llm["hidden_size"], "vocab": llm["vocab_size"], "mamba_num_heads": nh, "d_inner": d_inner,
            "conv_dim": conv_dim, "conv_kernel": llm["conv_kernel"], "in_proj": d_inner + conv_dim + nh,
            "q_heads": llm["num_attention_heads"], "kv_heads": llm["num_key_value_heads"], "head_dim": llm["head_dim"],
            "latent": llm["moe_latent_size"], "shared": llm["moe_shared_expert_intermediate_size"]}


# ------------------------------------------------------------------------------------------------
# packing

def is_linear(name: str) -> bool:
    return name.endswith((".up_proj.weight", ".down_proj.weight", ".fc1_latent_proj.weight", ".fc2_latent_proj.weight",
                          ".in_proj.weight", ".out_proj.weight", ".q_proj.weight", ".k_proj.weight",
                          ".v_proj.weight", ".o_proj.weight")) and ".layers." in name


def ckpt_name(r_name: str) -> str:
    return "language_model." + r_name


def pack(sd: dict, form: str, group: int) -> dict:
    """R-named tensors -> the checkpoint's: every linear packed signed at `group` in `form`'s tensors, the rest at the
    checkpoint's dtypes."""
    mt = make_tiny()
    out = {}
    for k, v in sd.items():
        if is_linear(k):
            qw, qz, sc = mt.pack_int4(v, group, True)
            base = ckpt_name(k)[: -len(".weight")]
            out[base + ".qweight"], out[base + ".qzeros"], out[base + ".scales"] = qw, qz, sc
            if form == "gptq-g128":
                out[base + ".g_idx"] = (torch.arange(v.shape[1], dtype=torch.int32) // group).contiguous()
        else:
            out[ckpt_name(k)] = v.float() if k.endswith(F32_NAMES) else _bf(v)
    return out


def expand_shard(t: dict) -> dict:
    """The exact g128 -> g64 expansion of one shard's tensors: scales and qzeros rows written twice, g_idx dropped."""
    out = {}
    for k, v in t.items():
        if k.endswith(".g_idx"):
            continue
        out[k] = v.repeat_interleave(2, 0).contiguous() if k.endswith((".scales", ".qzeros")) else v
    return out


def vision_tensors(extra: bool) -> dict:
    g = torch.Generator().manual_seed(99)
    out = {"vision_projector.mlp1.0.weight": _bf(torch.randn(16, generator=g)),
           "mlp1.1.weight": _bf(torch.randn(16, 16, generator=g))}
    if extra:
        out["vision_model.embeddings.patch_projection.weight"] = _bf(torch.randn(8, 24, generator=g))
    return out


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def find_cache_snapshot():
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--SergiioB--Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym/snapshots/*/")))
    return hits[-1] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--form", required=True, choices=FORMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", help="a Bolt snapshot directory (tokenizer.json, tokenizer_config.json, "
                                        "chat_template.jinja); default: the HF cache's")
    ap.add_argument("--expand-from", help="ar-g64: the g128 synthetic whose exact g64 expansion to write")
    a = ap.parse_args()
    if a.expand_from and a.form != "ar-g64":
        ap.error("--expand-from is ar-g64's")
    mc = make_config()
    if not 8 <= a.layers <= mc.LAYERS:
        ap.error(f"--layers {a.layers}: 8..{mc.LAYERS} (the first attention block is 7)")
    cfg = mc.config(a.layers, a.form)
    llm = cfg["llm_config"]
    wid = widths(llm)
    group = 128 if a.form == "gptq-g128" else 64
    os.makedirs(a.out, exist_ok=True)
    n_shards = 1 + a.layers
    wm: dict = {}
    total = 0
    t0 = time.time()

    def write(k: int, sd: dict, fn: str | None = None) -> None:
        nonlocal total
        fn = fn or f"model-{k + 1:05d}-of-{n_shards:05d}.safetensors"
        sd = {n: v.contiguous() for n, v in sd.items()}
        save_file(sd, os.path.join(a.out, fn), metadata={"format": "pt"})
        wm.update({n: fn for n in sd})
        total += sum(v.numel() * v.element_size() for v in sd.values())

    if a.expand_from:
        src = a.expand_from
        with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
            sc = json.load(f)
        if sc["llm_config"]["block_configs"] != llm["block_configs"]:
            raise SystemExit(f"--expand-from {src}: another layer count / pattern than --layers {a.layers}")
        for k in range(n_shards):
            fn = f"model-{k + 1:05d}-of-{n_shards:05d}.safetensors"
            with safe_open(os.path.join(src, fn), framework="pt") as h:
                t = {n: h.get_tensor(n) for n in h.keys()}
            write(k, expand_shard(t), fn)
            print(f"  shard {k} expanded, {time.time() - t0:.0f}s", flush=True)
    else:
        write(0, pack(top_tensors(a.seed, wid), a.form, group))
        for i in range(a.layers):
            t = pack(layer_tensors(i, llm["block_configs"][i], a.seed, wid), a.form, group)
            if i == a.layers - 1:
                t.update({"vision_model.summary_idxs": torch.arange(4, dtype=torch.int64)})
            write(1 + i, t)
            print(f"  layer {i} ({llm['block_configs'][i]['block_type']}) written, {time.time() - t0:.0f}s", flush=True)
    write(0, vision_tensors(True), "model_extra_tensors.safetensors")

    br = bolt_ref()
    want = set(br.expected_names(llm, "gptq-g128" if a.form == "gptq-g128" else "ar-g64"))
    got = {n for n in wm if not n.startswith(VISION_PREFIXES)}
    if want != got:
        raise SystemExit(f"make_synth: names differ from bolt_ref.expected_names: missing {sorted(want - got)[:5]}, "
                         f"extra {sorted(got - want)[:5]}")
    write_json(os.path.join(a.out, "model.safetensors.index.json"),
               {"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))})
    write_json(os.path.join(a.out, "config.json"), cfg)
    write_json(os.path.join(a.out, "generation_config.json"),
               {"_from_model_config": True, "bos_token_id": 1, "eos_token_id": 11, "pad_token_id": 0,
                "use_cache": True})
    write_json(os.path.join(a.out, "bolt_synth.json"),
               {"synthetic": True, "layers": a.layers, "form": a.form, "seed": a.seed,
                "expanded_from": os.path.abspath(a.expand_from) if a.expand_from else None,
                "maker": "tools/quantize/bolt/make_synth.py (random weights, bolt_make_tiny.pack_int4 signed, spec 25b)"})
    snap = a.tokenizer or find_cache_snapshot()
    for fn in SMALL_FILES:
        s = os.path.join(snap, fn) if snap else ""
        if snap and os.path.exists(s):
            shutil.copyfile(s, os.path.join(a.out, fn))
        else:
            print(f"make_synth: no {fn} ({'--tokenizer / the HF cache' if not snap else snap}) - not copied",
                  file=sys.stderr)
    kinds = [b["block_type"] for b in llm["block_configs"]]
    print(f"wrote {a.out}: {len(wm)} tensors in {n_shards} shards + model_extra_tensors, {total / 1e9:.2f} GB, "
          f"{a.layers} layers ({kinds.count('mamba')} mamba, {kinds.count('moe')} moe, {kinds.count('attention')} "
          f"attention), form {a.form}, {time.time() - t0:.0f}s")
    print(json.dumps({"layers": a.layers, "form": a.form, "bytes": total}))


if __name__ == "__main__":
    main()
