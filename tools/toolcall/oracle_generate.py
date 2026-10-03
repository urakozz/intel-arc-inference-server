#!/usr/bin/env python3
"""Greedy bf16 baseline for the tool-call set (spec 5 T0, gate A4), on CPU.

    oracle_generate.py <bf16 snapshot> <set dir> <out dir>

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


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    snap, set_dir, out = sys.argv[1:]
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

    greedy = GenerationConfig(max_new_tokens=NEW_TOKENS, do_sample=False, eos_token_id=eos,
                              pad_token_id=eos[0] if isinstance(eos, list) else eos)
    for i, e in enumerate(todo, 1):
        name = e["name"]
        ids = prompts[name]
        t = time.time()
        with torch.no_grad():
            seq = model.generate(torch.tensor([ids]), generation_config=greedy)
        new = [int(x) for x in seq[0, len(ids):].tolist()]
        text = tok.decode(new, skip_special_tokens=False)
        # .txt last: its presence (with .ids) marks the scenario done.
        write_atomic(os.path.join(out, f"{name}.{RUN}.ids"), " ".join(map(str, new)) + "\n")
        write_atomic(os.path.join(out, f"{name}.{RUN}.txt"), text)
        print(f"[{i}/{len(todo)}] {name}: {len(ids)} prompt ids, {len(new)} new, "
              f"{time.time() - t:.0f}s, call={'<tool_call>' in text}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
