# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
"""What Inkling 2 Mini's int8 cache costs (plan 41d Task 0): teacher-forced, fp32, the bf16 cache against the int8 one.

    python -m tools.oracle.inkling_kv8_kld ids SNAPSHOT OUT --wikitext TEST.parquet --session REQUEST.json \
        [--a4 SET_DIR] [--a4-every 6]
    python -m tools.oracle.inkling_kv8_kld run SNAPSHOT OUT [--threads 40] [--chunk 512] [--arms int8,int8-k,int8-v]
        [--only NAME]
    python -m tools.oracle.inkling_kv8_kld summary OUT

`ids` writes the corpus into OUT as <name>.ids: 20 consecutive wikitext-2 (raw, test) chunks of 512 ids (wiki-00..19),
the recorded opencode session's last request rendered by the checkpoint's template (reasoning effort none), its last
8192 ids (session), and every --a4-every-th scenario of the A4 tool-call set, sorted by name (a4-<scenario>).

`run` reads each sequence twice through inkling_ref.Inkling in fp32 - `--kv bf16` (every cache row rounded to bf16, the
engine's default) and `--kv int8` (the 8 full visits' K / V through the 128-point rotated int8, rings bf16) - chunk by
chunk, the two caches side by side, and writes OUT/<name>.kld.json: per position KL(bf16 || int8) of the softmaxed
logits and whether the top-1 ids agree. Resumable: a sequence with its json is skipped. `--arms` names the int8 forms
compared against the one bf16 forward (int8; int8-k / int8-v: only K / only V int8, the other bf16; <arm>~V1+V2: the
arm with visits V1, V2 in walk order kept bf16 - the full ones are 5, 11, 17, 23, 29, 35, 41, 47; fp32: no cache
rounding, the control of how far any change of the caches moves these contexts), each arm but int8
written as OUT/<name>.<arm>.kld.json; `--only` runs the one sequence NAME.

`summary` prints mean KLD and top-1 agreement per corpus (wiki, session, a4) and over all positions.
"""

import argparse
import json
import math
from pathlib import Path

WIKI_CHUNKS, WIKI_LEN, SESSION_LEN = 20, 512, 8192


def corpus(name):
    return name.split("-")[0]


def plain(content):
    """An OpenAI message content (string or parts) as text."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def session_messages(request):
    """The recorded request's messages and tools in the form the checkpoint's template takes: text contents, tool call
    arguments as objects."""
    msgs = []
    for m in request["messages"]:
        m = dict(m, content=plain(m.get("content")))
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                f = dict(c["function"])
                if isinstance(f.get("arguments"), str):
                    f["arguments"] = json.loads(f["arguments"] or "{}")
                calls.append(dict(c, function=f))
            m["tool_calls"] = calls
        msgs.append(m)
    return msgs, request.get("tools")


def make_ids(a):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.snapshot)
    text = "".join(pq.read_table(a.wikitext).column("text").to_pylist())
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < WIKI_CHUNKS * WIKI_LEN:
        raise SystemExit(f"wikitext gives {len(ids)} ids, fewer than {WIKI_CHUNKS} x {WIKI_LEN}")
    seqs = {f"wiki-{i:02d}": ids[i * WIKI_LEN:(i + 1) * WIKI_LEN] for i in range(WIKI_CHUNKS)}
    msgs, tools = session_messages(json.loads(Path(a.session).read_text())["request"])
    r = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True,
                                reasoning_effort="none")
    r = [int(x) for x in (r["input_ids"] if hasattr(r, "keys") else r)]
    seqs["session"] = r[-SESSION_LEN:]
    if a.a4:
        names = sorted(m["name"] for m in json.loads((Path(a.a4) / "manifest.json").read_text()))
        for n in names[::a.a4_every]:
            seqs[f"a4-{n}"] = list(map(int, (Path(a.a4) / f"{n}.ids").read_text().split()))
    for n, s in seqs.items():
        (out / f"{n}.ids").write_text(" ".join(map(str, s)) + "\n")
        print(f"{n}: {len(s)} ids", flush=True)


FULL_VISITS = ("5", "11", "17", "23", "29", "35", "41", "47")


def arms(value):
    """--arms: each name one of fp32, int8, int8-k, int8-v, or one of the int8 forms with ~V1+V2 of FULL_VISITS."""
    names = value.split(",")
    for name in names:
        form, _, keep = name.partition("~")
        if form not in ("fp32", "int8", "int8-k", "int8-v") or (keep and (form == "fp32" or not all(
                v in FULL_VISITS for v in keep.split("+")))):
            raise argparse.ArgumentTypeError(f"unknown arm {name!r}: fp32, int8, int8-k, int8-v, or an int8 form with"
                                             f" ~V1+V2 of the full visits {', '.join(FULL_VISITS)}")
    return names


def run(a):
    import time

    import torch

    from .inkling_ref import Inkling

    torch.set_num_threads(a.threads)
    out = Path(a.out)
    model = Inkling(a.snapshot, torch.float32)      # one model, its cache form switched a forward (Inkling.kv)
    for path in sorted(out.glob("*.ids")):
        name = path.stem
        if a.only and name != a.only:
            continue
        done = {arm: out / (f"{name}.kld.json" if arm == "int8" else f"{name}.{arm}.kld.json") for arm in a.arms}
        arms = [arm for arm in a.arms if not done[arm].exists()]
        if not arms:
            continue
        ids = list(map(int, path.read_text().split()))
        caches = [model.cache() for _ in range(len(arms) + 1)]
        kld, agree, t0 = {arm: [] for arm in arms}, {arm: [] for arm in arms}, time.monotonic()
        with torch.inference_mode():
            for s in range(0, len(ids), a.chunk):
                x = torch.tensor([ids[s:s + a.chunk]])
                model.kv = "bf16"
                p = model.forward(x, caches[0])[0][0].log_softmax(-1)
                for arm, c in zip(arms, caches[1:]):
                    model.kv = None if arm == "fp32" else arm
                    q = model.forward(x, c)[0][0].log_softmax(-1)
                    kld[arm] += (p.exp() * (p - q)).sum(-1).tolist()
                    agree[arm] += (p.argmax(-1) == q.argmax(-1)).tolist()
                print(f"{name}: {s + x.shape[1]} / {len(ids)} ids, {time.monotonic() - t0:.0f} s", flush=True)
        for arm in arms:
            done[arm].write_text(json.dumps({"name": name, "positions": len(kld[arm]), "kld": kld[arm],
                                             "top1": agree[arm]}) + "\n")


def summary(a):
    rows = {}
    for path in sorted(Path(a.out).glob("*.kld.json")):
        if path.name.count(".") > 2:   # another arm's (<name>.<arm>.kld.json)
            continue
        r = json.loads(path.read_text())
        for key in (corpus(r["name"]), "all"):
            k, t = rows.setdefault(key, ([], []))
            k += r["kld"]
            t += r["top1"]
    print("| corpus | positions | mean KLD | max KLD | top-1 agree |\n|---|---:|---:|---:|---:|")
    for key, (k, t) in rows.items():
        print(f"| {key} | {len(k)} | {math.fsum(k) / len(k):.6f} | {max(k):.4f} | {100 * sum(t) / len(t):.2f} % |")


def main():
    import os
    import importlib.util as _cl_util
    _cl = _cl_util.spec_from_file_location("cpu_lock", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu_lock.py"))
    _cl_mod = _cl_util.module_from_spec(_cl)
    _cl.loader.exec_module(_cl_mod)
    _cl_mod.hold()   # one CPU reference job at a time: refuse beside another
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("ids")
    q.add_argument("snapshot")
    q.add_argument("out")
    q.add_argument("--wikitext", required=True)
    q.add_argument("--session", required=True)
    q.add_argument("--a4")
    q.add_argument("--a4-every", type=int, default=6)
    q = sub.add_parser("run")
    q.add_argument("snapshot")
    q.add_argument("out")
    q.add_argument("--threads", type=int, default=40)
    q.add_argument("--chunk", type=int, default=512)
    q.add_argument("--arms", type=arms, default=["int8"])
    q.add_argument("--only")
    q = sub.add_parser("summary")
    q.add_argument("out")
    a = p.parse_args()
    {"ids": make_ids, "run": run, "summary": summary}[a.command](a)


if __name__ == "__main__":
    main()
