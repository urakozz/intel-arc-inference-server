#!/usr/bin/env python3
"""Greedy bf16 baseline for the tool-call set (spec 5 T0, gate A4), on CPU.

    oracle_generate.py <bf16 snapshot> <set dir> <out dir> [--new-tokens N]

Builds transformers' Qwen3_5ForCausalLM the way tools/rotate/check_rotation.py
does (text_config, load_sd in bf16, eager attention), then for every scenario
in <set dir>/manifest.json, after checking its .ids file's SHA-256 against the
manifest, runs greedy generate with 192 new tokens and writes
<out>/<name>.bf16.ids (the new ids) and <out>/<name>.bf16.txt (decoded with
the snapshot's tokenizer, special tokens kept).

Resumable: a scenario whose two outputs exist is skipped, because the whole set
takes hours. Outputs are written to a temporary name and renamed, so a killed
run never leaves a half-written file behind.

Run inside the reference container (tools/oracle/run_in_container.sh with
ORACLE_MODEL=models--Qwen--Qwen3.8-27B).

The model follows config.json's model_type:
  qwen3_5 / agnes   transformers' Qwen3_5ForCausalLM on the bf16 checkpoint (above; spec 14
                    for Agnes), HF generate;
  qwen3_5_moe       Ornith 1.5 (spec 15e Task 3): tools/oracle/ornith_ref.py's layer-streamed
                    Qwen3_5MoeForCausalLM on its INT4 checkpoint dequantised by the repo's one
                    rule (15a: the engine's own weights, so a mismatch is the engine's, not the
                    quantisation's), greedy through its KV cache;
  k2_horizon        K2-Horizon (spec 18d, K4): tools/oracle/k2_ref.py's port of
                    modeling_k2_horizon.py, layer at a time, mode bf16 (bitwise the vendored
                    model in bf16, eager attention), on the checkpoint given - the int4 one
                    (dequantised, 18a's reference) or the bf16 original - greedy through its
                    own cache.
The MoE paths stop after an EOS id of generation_config.json (kept, as HF generate keeps it)
or --new-tokens ids (default 192; K2's A4 run uses 512: its replies open with reasoning).
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "bf16"
NEW_TOKENS = 192


def load_manifest(set_dir: str) -> list[dict]:
    with open(os.path.join(set_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def checked_ids(set_dir: str, entry: dict) -> list[int]:
    with open(os.path.join(set_dir, f"{entry['name']}.ids"), "rb") as f:
        raw = f.read()
    got = hashlib.sha256(raw).hexdigest()
    if got != entry["sha256"]:
        sys.exit(f"FATAL: {entry['name']}.ids SHA-256 {got} != manifest {entry['sha256']}")
    ids = [int(x) for x in raw.split()]
    if len(ids) != entry["ids"]:
        sys.exit(f"FATAL: {entry['name']}.ids has {len(ids)} ids, manifest says {entry['ids']}")
    return ids


def write_atomic(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def greedy(step, prompt: list[int], n: int, eos: set[int], argmax) -> list[int]:
    """The MoE paths' greedy loop: `step(ids, pos)` runs ids at positions pos.. through the
    model's cache and returns the last row's logits; `argmax(row)` -> int. Returns the new ids:
    at most n, ending at the first EOS id (kept)."""
    out: list[int] = []
    row = step(prompt, 0)
    while len(out) < n:
        nxt = int(argmax(row))
        out.append(nxt)
        if nxt in eos or len(out) == n:
            break
        row = step([nxt], len(prompt) + len(out) - 1)
    return out


def moe_runner(snap: str, mtype: str):
    """-> generate(ids, n, eos) for Ornith (qwen3_5_moe) or K2-Horizon (k2_horizon)."""
    import torch

    def load(name: str):
        spec = importlib.util.spec_from_file_location(f"b70_{name}", os.path.join(HERE, "..", "oracle", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    def argmax(row):
        return int(torch.argmax(row).item())

    if mtype == "k2_horizon":
        K2 = load("k2_ref")
        src = K2.Checkpoint(snap)
        ref = K2.K2Ref(src.cfg, src, mode="bf16")
        print(f"K2-Horizon reference: k2_ref.py, {src.cfg.num_hidden_layers} layers, mode bf16, "
              f"{'int4 g%d dequantised' % src.cfg.group_size if src.cfg.group_size else 'bf16'}", flush=True)

        def generate(ids, n, eos):
            cache = ref.new_cache()
            return greedy(lambda chunk, pos: ref.forward(chunk, pos, cache)[-1], ids, n, eos, argmax)
        return generate

    OR = load("ornith_ref")
    tc = OR.text_config(snap)
    model, _pf, lazy = OR.build_streamed(snap, tc)
    print(f"Ornith reference: ornith_ref.py, {tc.num_hidden_layers} layers, int4 dequantised, streamed", flush=True)

    def generate(ids, n, eos):
        state = {"cache": None}

        def step(chunk, pos):
            with torch.no_grad():
                o = model(input_ids=torch.tensor([chunk]), past_key_values=state["cache"], use_cache=True)
            state["cache"] = o.past_key_values
            return o.logits[0, -1].float()
        return greedy(step, ids, n, eos, argmax)
    return generate


def main() -> None:
    args = sys.argv[1:]
    new_tokens = NEW_TOKENS
    if len(args) == 5 and args[3] == "--new-tokens":
        new_tokens = int(args[4])
        args = args[:3]
    if len(args) != 3:
        sys.exit(__doc__)
    snap, set_dir, out = args
    os.makedirs(out, exist_ok=True)
    manifest = load_manifest(set_dir)
    todo = [e for e in manifest
            if not all(os.path.exists(os.path.join(out, f"{e['name']}.{RUN}.{x}")) for x in ("ids", "txt"))]
    # Check every ids file before the hour-long load, not one by one after it.
    prompts = {e["name"]: checked_ids(set_dir, e) for e in manifest}
    print(f"{len(manifest)} scenarios, {len(manifest) - len(todo)} already done, {len(todo)} to run",
          flush=True)
    if not todo:
        return

    with open(os.path.join(snap, "config.json"), encoding="utf-8") as f:
        mtype = json.load(f).get("model_type", "")
    if mtype in ("qwen3_5_moe", "k2_horizon"):
        moe_main(snap, out, todo, prompts, new_tokens, mtype)
        return

    import torch
    from tokenizers import Tokenizer
    from transformers import GenerationConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    spec = importlib.util.spec_from_file_location(
        "b70_check_rotation", os.path.join(HERE, "..", "rotate", "check_rotation.py"))
    CR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CR)

    tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    gen_cfg = GenerationConfig.from_pretrained(snap)
    eos = gen_cfg.eos_token_id
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, eos {eos}", flush=True)

    t = time.time()
    # Spec 14: Agnes 3.0 Flash (config.json model_type agnes) is Qwen3.5 + the parallel
    # FFN under other tensor names (tools/oracle/agnes.py); its bf16 checkpoint
    # (Agnes-AI/Agnes-3.0-Flash) is the A4 reference (spec 14 G3).
    aspec = importlib.util.spec_from_file_location(
        "b70_agnes", os.path.join(HERE, "..", "oracle", "agnes.py"))
    AG = importlib.util.module_from_spec(aspec)
    aspec.loader.exec_module(AG)
    with open(os.path.join(snap, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    if AG.is_agnes(raw):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        kwargs, parallel = AG.translate_text_config(raw)
        tc = Qwen3_5TextConfig(**kwargs)
        tc._attn_implementation = "eager"
    else:
        tc = CR.text_config(snap, 0)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(tc)
        if AG.is_agnes(raw):
            AG.attach_parallel_ffn(model, tc, parallel)
    sd = CR.to_model_names(CR.load_sd(snap, 0, torch.bfloat16))
    if AG.is_agnes(raw):
        sd = {AG.map_name(k): v for k, v in sd.items()}
    want = set(model.state_dict().keys())
    if set(sd) != want:
        sys.exit(f"FATAL: state dict mismatch: {len(want - set(sd))} missing, "
                 f"{len(set(sd) - want)} unexpected, e.g. {sorted(want ^ set(sd))[:4]}")
    model.load_state_dict(sd, strict=True, assign=True)
    del sd
    model.model.rotary_emb = type(model.model.rotary_emb)(tc)
    model.eval()
    print(f"loaded {tc.num_hidden_layers} layers, dtype {model.dtype}, {time.time() - t:.0f}s", flush=True)

    gen_cfg = GenerationConfig(max_new_tokens=new_tokens, do_sample=False, eos_token_id=eos,
                               pad_token_id=eos[0] if isinstance(eos, list) else eos)
    for i, e in enumerate(todo, 1):
        name = e["name"]
        ids = prompts[name]
        t = time.time()
        with torch.no_grad():
            seq = model.generate(torch.tensor([ids]), generation_config=gen_cfg)
        new = [int(x) for x in seq[0, len(ids):].tolist()]
        save(out, name, new, tok.decode(new, skip_special_tokens=False), i, len(todo), len(ids), t)
    print("done", flush=True)


def save(out: str, name: str, new: list[int], text: str, i: int, n: int, n_prompt: int, t: float) -> None:
    # .txt last: its presence (with .ids) marks the scenario done.
    write_atomic(os.path.join(out, f"{name}.{RUN}.ids"), " ".join(map(str, new)) + "\n")
    write_atomic(os.path.join(out, f"{name}.{RUN}.txt"), text)
    call = "<tool_call>" in text or "<ifm|tool_call>" in text
    print(f"[{i}/{n}] {name}: {n_prompt} prompt ids, {len(new)} new, {time.time() - t:.0f}s, call={call}",
          flush=True)


def moe_main(snap: str, out: str, todo: list[dict], prompts: dict, new_tokens: int, mtype: str) -> None:
    import torch
    from tokenizers import Tokenizer
    with open(os.path.join(snap, "generation_config.json"), encoding="utf-8") as f:
        eos = json.load(f)["eos_token_id"]
    eos = set(eos if isinstance(eos, list) else [eos])
    tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, eos {sorted(eos)}, "
          f"model_type {mtype}, {new_tokens} new tokens", flush=True)
    t = time.time()
    generate = moe_runner(snap, mtype)
    print(f"ready in {time.time() - t:.0f}s", flush=True)
    for i, e in enumerate(todo, 1):
        name = e["name"]
        ids = prompts[name]
        t = time.time()
        new = generate(ids, new_tokens, eos)
        save(out, name, new, tok.decode(new, skip_special_tokens=False), i, len(todo), len(ids), t)
    print("done", flush=True)


if __name__ == "__main__":
    main()
