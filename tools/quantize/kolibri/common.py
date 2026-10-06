# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the Kolibri-1 quantisation stages (tools/quantize_kolibri1.sh, spec 20 §3.1).

Paths, the model shim (the bf16 snapshot plus our transformers port, wired through `auto_map`), the
packed-row files every stage passes on, the pinned German document source, and the reference
(tools/oracle/kolibri_ref.py) the coverage and evaluation stages run on.
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ORACLE = os.path.join(REPO, "tools", "oracle")
PORT_DIR = os.path.join(ORACLE, "third_party", "kolibri1")
PROMPTS = os.path.join(HERE, "prompts")

SEQLEN = 2048
IM_END, ENDOFTEXT = 127906, 127901          # <|im_end|> (EOS), <|endoftext|> (pad, the 2nd EOS)
EOS_IDS = (IM_END, ENDOFTEXT)               # generation_config.json eos_token_id
SAMPLING = {"temperature": 1.0, "top_p": 0.97, "top_k": 128}   # generation_config.json / the model card

# Calibration categories and their share of the base rows (spec 20 §3.1: >= 50 % German - chat with
# reasoning and tool calls, long documents - ~30 % English chat, ~20 % code). de_topup rows are what
# the coverage stage adds on top.
CAT_FRACTIONS = {"de_chat": 0.20, "de_tool": 0.10, "de_doc": 0.20, "en_chat": 0.30, "code": 0.20}
CATS = list(CAT_FRACTIONS) + ["de_topup"]
CAT_LANG = {"de_chat": "de", "de_tool": "de", "de_doc": "de", "de_topup": "de", "en_chat": "en", "code": "code"}
EVAL_SETS = ("de_wiki", "de_chat", "en_chat", "code")
EVAL_LANG = {"de_wiki": "de", "de_chat": "de", "en_chat": "en", "code": "code"}

# German long documents: Wikipedia (de), a pinned dataset revision. Calibration and the coverage
# top-up read shard 00000 only; evaluation reads shard 00019 only - disjoint files, so no held-out
# article can have been calibrated on (the manifest records every article id used).
WIKI_REPO = "wikimedia/wikipedia"
WIKI_REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKI_CALIB_FILE = "20231101.de/train-00000-of-00020.parquet"
WIKI_EVAL_FILE = "20231101.de/train-00019-of-00020.parquet"

# The bars (spec 20 §3 / §3.1): Aleph Alpha's FP8 KL DE 0.060 (target), the community GPTQ 0.069
# (minimum), English no worse than KL 0.013.
BAR_DE_TARGET, BAR_DE_MIN, BAR_EN = 0.060, 0.069, 0.013
PUBLISHED = [("Aleph-Alpha/Kolibri-1 (FP8, official)", 0.060, 0.928, 0.010, 0.972),
             ("Padakovec/Kolibri-1-W4A16-GPTQ (asym int4 g128, experts only)", 0.069, 0.915, None, None)]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------------
# the model

def resolve_model(model: str, revision: str | None = None) -> str:
    """A local snapshot directory for MODEL: a directory as given, or the HF hub snapshot (the cache
    is used when present; HF_HUB_OFFLINE=1 makes it never download)."""
    if os.path.isdir(model):
        if not os.path.exists(os.path.join(model, "config.json")):
            raise SystemExit(f"{model} has no config.json")
        return os.path.abspath(model)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model, revision=revision)


def make_shim(src: str, dst: str) -> str:
    """dst: every file of the snapshot `src` symlinked, config.json copied with an `auto_map` to our
    port, and the port's two files copied in - so `AutoModelForCausalLM.from_pretrained(dst,
    trust_remote_code=True)` (AutoRound's loader) builds third_party/kolibri1's Kolibri1ForCausalLM on
    the original weights. Idempotent; the snapshot is never written."""
    os.makedirs(dst, exist_ok=True)
    for fn in sorted(os.listdir(src)):
        if fn in ("config.json",) or fn.endswith(".py"):
            continue
        link = os.path.join(dst, fn)
        target = os.path.realpath(os.path.join(src, fn))
        if os.path.islink(link) and os.path.realpath(link) == target:
            continue
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(target, link)
    cfg = read_json(os.path.join(src, "config.json"))
    if cfg.get("model_type") != "kolibri1":
        raise SystemExit(f"{src}: model_type {cfg.get('model_type')!r}, expected kolibri1")
    cfg["auto_map"] = {"AutoConfig": "configuration_kolibri1.Kolibri1Config",
                       "AutoModel": "modeling_kolibri1.Kolibri1Model",
                       "AutoModelForCausalLM": "modeling_kolibri1.Kolibri1ForCausalLM"}
    write_json(os.path.join(dst, "config.json"), cfg)
    for fn in ("modeling_kolibri1.py", "configuration_kolibri1.py"):
        with open(os.path.join(PORT_DIR, fn), "rb") as f:
            data = f.read()
        with open(os.path.join(dst, fn), "wb") as f:
            f.write(data)
    return dst


def load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.chat_template is None:
        raise SystemExit(f"{model_dir}: tokenizer has no chat template")
    return tok


def kolibri_ref():
    """tools/oracle/kolibri_ref.py as a module."""
    if "kolibri_ref" in sys.modules:
        return sys.modules["kolibri_ref"]
    spec = importlib.util.spec_from_file_location("kolibri_ref", os.path.join(ORACLE, "kolibri_ref.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kolibri_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------
# packed rows: input_ids [N, T] int32 + a category per row, metadata as JSON

def save_rows(path: str, ids: torch.Tensor, cats: list[str], meta: dict) -> None:
    from safetensors.torch import save_file
    assert ids.dim() == 2 and len(cats) == ids.shape[0]
    names = sorted(set(cats))
    code = torch.tensor([names.index(c) for c in cats], dtype=torch.int32)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file({"input_ids": ids.to(torch.int32).contiguous(), "cat": code}, path + ".tmp",
              metadata={"cats": json.dumps(names), "meta": json.dumps(meta, ensure_ascii=False)})
    os.replace(path + ".tmp", path)


def load_rows(path: str):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as h:
        ids = h.get_tensor("input_ids").long()
        code = h.get_tensor("cat").tolist()
        md = h.metadata()
    names = json.loads(md["cats"])
    return ids, [names[i] for i in code], json.loads(md["meta"])


def pack(convs: list[list[int]], n_rows: int, seqlen: int = SEQLEN) -> tuple[list[list[int]], list[int]]:
    """Greedy packing into rows of exactly `seqlen`: conversations are appended whole while they fit;
    the one that does not fit is cut to fill the row (its tail is dropped), so every row starts at a
    conversation boundary. Returns (rows, how many conversations each row starts with). Stops at
    n_rows; a final partial row is dropped."""
    rows, starts, cur, k = [], [], [], 0
    for c in convs:
        if len(rows) >= n_rows:
            break
        if not c:
            continue
        cur = cur + c[: seqlen - len(cur)]
        k += 1
        if len(cur) == seqlen:
            rows.append(cur)
            starts.append(k)
            cur, k = [], 0
    return rows, starts


# --------------------------------------------------------------------------------------------
# German documents

def iter_docs(which: str, min_chars: int = 2000):
    """German long documents, in file order: {"id", "title", "text"}. which = "calib" (shard 00000)
    or "eval" (shard 00019) of the pinned Wikipedia revision; KOLIBRI_DE_DOCS / KOLIBRI_DE_DOCS_EVAL
    (a JSONL file of the same records) replace them - the tests use that."""
    env = os.environ.get("KOLIBRI_DE_DOCS" if which == "calib" else "KOLIBRI_DE_DOCS_EVAL")
    if env:
        for d in read_jsonl(env):
            if len(d["text"]) >= min_chars:
                yield d
        return
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    fn = WIKI_CALIB_FILE if which == "calib" else WIKI_EVAL_FILE
    path = hf_hub_download(WIKI_REPO, fn, repo_type="dataset", revision=WIKI_REV)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=512, columns=["id", "title", "text"]):
        for rec in batch.to_pylist():
            if len(rec["text"]) >= min_chars:
                yield {"id": str(rec["id"]), "title": rec["title"], "text": rec["text"]}


def doc_source() -> dict:
    if os.environ.get("KOLIBRI_DE_DOCS"):
        return {"calib": os.environ["KOLIBRI_DE_DOCS"], "eval": os.environ.get("KOLIBRI_DE_DOCS_EVAL")}
    return {"dataset": WIKI_REPO, "revision": WIKI_REV, "calib": WIKI_CALIB_FILE, "eval": WIKI_EVAL_FILE}


# --------------------------------------------------------------------------------------------
# the export

def find_export(arm_dir: str) -> str:
    """The directory AutoRound exported into under arm_dir (it appends `<model>-w4g64`)."""
    found = sorted(p for p, _, f in os.walk(arm_dir) if "config.json" in f and "quantization_config.json" in f)
    if not found:
        found = sorted(p for p, _, f in os.walk(arm_dir) if "config.json" in f)
    if len(found) != 1:
        raise SystemExit(f"{arm_dir}: expected one exported checkpoint, found {found}")
    return found[0]


def restore_bf16(export: str, source: str) -> dict:
    """Every tensor of the export that is not int4 (qweight / scales / qzeros / g_idx) must be the
    source's bf16 tensor. AutoRound on a CPU without bf16 support runs (and writes) the model in fp32:
    those tensors are widened copies - restored here to the source's bf16 bytes after checking they
    are BITWISE the widened source (anything else means a tensor was changed, and this refuses).
    Rewrites only the shards that change; sets config.json dtype to bfloat16."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    ref = kolibri_ref()
    src = ref.Checkpoint(source)
    if src.cfg.group_size is not None:
        raise SystemExit(f"{source} is quantised; restore_bf16 needs the bf16 snapshot")
    stats = {"restored": 0, "already_bf16": 0, "shards_rewritten": 0}
    for fn in sorted(f for f in os.listdir(export) if f.endswith(".safetensors")):
        path = os.path.join(export, fn)
        with safe_open(path, framework="pt") as h:
            md = h.metadata() or {}
            tensors = {k: h.get_tensor(k) for k in h.keys()}
        changed = False
        for k, t in tensors.items():
            if k.endswith((".qweight", ".scales", ".qzeros", ".g_idx")):
                continue
            if not src.has(k):
                raise SystemExit(f"{fn}: {k} is not in the source checkpoint")
            s = src.tensor(k)
            if s.dtype != torch.bfloat16:
                raise SystemExit(f"source {k} is {s.dtype}, expected bf16")
            if t.shape != s.shape or not torch.equal(t.float(), s.float()):
                raise SystemExit(f"{fn}: {k} ({t.dtype}) is not the source tensor - refusing to restore")
            if t.dtype == torch.bfloat16:
                stats["already_bf16"] += 1
                continue
            tensors[k] = s.contiguous()
            stats["restored"] += 1
            changed = True
        if changed:
            save_file(tensors, path + ".tmp", metadata=md)
            os.replace(path + ".tmp", path)
            stats["shards_rewritten"] += 1
    cfg = read_json(os.path.join(export, "config.json"))
    if cfg.get("dtype") != "bfloat16" or cfg.get("torch_dtype") not in (None, "bfloat16"):
        cfg["dtype"] = "bfloat16"
        cfg.pop("torch_dtype", None)
        write_json(os.path.join(export, "config.json"), cfg)
    return stats
