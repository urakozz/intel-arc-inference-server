#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage `eval` of tools/quantize_kolibri1.sh: KL and top-1 against bf16, German and English apart.

    evaluate.py --model <bf16 snapshot> --sets <dir with eval_*.safetensors> --arm <name>=<export-dir> ...
                --out <dir> [--device cpu]

For every held-out set (de_wiki, de_chat, en_chat, code - calib.py wrote them, none was calibrated
on) the bf16 model and each quantised export run teacher-forced over every row
(tools/oracle/kolibri_ref.py run_batch, layer-major; an export's int4 linears dequantised by
dequant.py's rule, the engine's arithmetic). Per position: the fp32 logits (head_dtype float32),
KL(P_bf16 || P_quant) in nats, whether the argmaxes agree, and both models' NLL of the next token.

Reported per set and per language - DE = de_wiki + de_chat (token-weighted), EN = en_chat, code -
as mean KL, top-1 agreement and perplexity, with spec 20 §3.1's bars: KL DE < 0.060 (target, the
official FP8), < 0.069 (minimum, the community GPTQ), KL EN <= 0.013. The bf16 pass is cached as
final hidden states (eval/bf16_<set>.pt), so arms can be added later. Writes eval/eval.json and
prints the table.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def hidden(ref_mod, snapshot: str, ids: torch.Tensor, device: str, rows: int) -> torch.Tensor:
    src = ref_mod.Checkpoint(snapshot)
    ref = ref_mod.KolibriRef(src.cfg, src, mode="bf16", device=device)
    h = ref.run_batch(ids, rows=rows, log=C.log)
    return h, ref


def metrics(ref_p, hp: torch.Tensor, ref_q, hq: torch.Tensor, ids: torch.Tensor, device: str, chunk: int = 512):
    """Sums over every position with a next token: KL(P || Q), top-1 agreement, NLL under P and Q."""
    s = {"n": 0, "kl": 0.0, "top1": 0, "nll_p": 0.0, "nll_q": 0.0}
    N, T = ids.shape
    for r in range(N):
        for a in range(0, T - 1, chunk):
            b = min(T - 1, a + chunk)
            lp = torch.log_softmax(ref_p.head(hp[r, a:b].to(device)), dim=-1)
            lq = torch.log_softmax(ref_q.head(hq[r, a:b].to(device)), dim=-1)
            tgt = ids[r, a + 1:b + 1].to(device)
            s["kl"] += float((lp.exp() * (lp - lq)).sum())
            s["top1"] += int((lp.argmax(-1) == lq.argmax(-1)).sum())
            s["nll_p"] += float(-lp.gather(1, tgt[:, None]).sum())
            s["nll_q"] += float(-lq.gather(1, tgt[:, None]).sum())
            s["n"] += b - a
    return s


def finish(s: dict) -> dict:
    import math
    n = max(s["n"], 1)
    return {"tokens": s["n"], "kl": s["kl"] / n, "top1": s["top1"] / n, "ppl_bf16": math.exp(s["nll_p"] / n),
            "ppl_quant": math.exp(s["nll_q"] / n)}


def bars(lang: dict) -> dict:
    de, en = lang.get("de", {}).get("kl"), lang.get("en", {}).get("kl")
    return {"de_target": de is not None and de < C.BAR_DE_TARGET, "de_minimum": de is not None and de < C.BAR_DE_MIN,
            "en": en is not None and en <= C.BAR_EN}


def table(results: dict) -> str:
    lines = ["arm                       set        tokens      KL      top-1   ppl bf16  ppl quant"]
    for arm, r in results.items():
        for name, m in list(r["sets"].items()) + [(f"[{k.upper()}]", v) for k, v in r["lang"].items()]:
            lines.append(f"{arm:25s} {name:9s} {m['tokens']:7d}  {m['kl']:.4f}  {m['top1'] * 100:6.2f}%  "
                         f"{m['ppl_bf16']:8.3f}  {m['ppl_quant']:9.3f}")
        b = r["bars"]
        lines.append(f"{arm:25s} bars: KL DE < {C.BAR_DE_TARGET} target {'PASS' if b['de_target'] else 'fail'}, "
                     f"< {C.BAR_DE_MIN} minimum {'PASS' if b['de_minimum'] else 'FAIL'}, "
                     f"EN <= {C.BAR_EN} {'PASS' if b['en'] else 'FAIL'}")
    lines.append("published: " + "; ".join(f"{n}: KL DE {d}, top-1 DE {t}" + (f", KL EN {e}, top-1 EN {te}" if e else "")
                                           for n, d, t, e, te in C.PUBLISHED))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the bf16 snapshot")
    ap.add_argument("--sets", required=True, help="directory with eval_<set>.safetensors")
    ap.add_argument("--arm", action="append", required=True, help="name=export-dir (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rows-per-fwd", type=int, default=4)
    a = ap.parse_args()
    ref_mod = C.kolibri_ref()
    snapshot = C.resolve_model(a.model)
    os.makedirs(a.out, exist_ok=True)
    sets = {}
    for name in C.EVAL_SETS:
        p = os.path.join(a.sets, f"eval_{name}.safetensors")
        if os.path.exists(p):
            sets[name] = C.load_rows(p)[0]
    if not sets:
        raise SystemExit(f"no eval_<set>.safetensors under {a.sets}")
    arms = dict(x.split("=", 1) for x in a.arm)
    out_path = os.path.join(a.out, "eval.json")
    results = C.read_json(out_path)["arms"] if os.path.exists(out_path) else {}
    base = {}
    for name, ids in sets.items():
        cache = os.path.join(a.out, f"bf16_{name}.pt")
        if os.path.exists(cache):
            base[name] = torch.load(cache)
        else:
            C.log(f"bf16 pass over {name}: {tuple(ids.shape)}")
            base[name], _ = hidden(ref_mod, snapshot, ids, a.device, a.rows_per_fwd)
            torch.save(base[name], cache)
    src_p = ref_mod.Checkpoint(snapshot)
    ref_p = ref_mod.KolibriRef(src_p.cfg, src_p, mode="bf16", device=a.device, prefetch=False)
    for arm, d in arms.items():
        export = C.find_export(d) if not os.path.exists(os.path.join(d, "config.json")) else d
        per, lang_sum = {}, {}
        for name, ids in sets.items():
            C.log(f"{arm}: pass over {name}")
            hq, ref_q = hidden(ref_mod, export, ids, a.device, a.rows_per_fwd)
            s = metrics(ref_p, base[name], ref_q, hq, ids, a.device)
            per[name] = finish(s)
            lg = C.EVAL_LANG[name]
            acc = lang_sum.setdefault(lg, {"n": 0, "kl": 0.0, "top1": 0, "nll_p": 0.0, "nll_q": 0.0})
            for k in acc:
                acc[k] += s[k]
        lang = {k: finish(v) for k, v in lang_sum.items()}
        results[arm] = {"export": export, "sets": per, "lang": lang, "bars": bars(lang)}
        C.write_json(out_path, {"model": snapshot, "arms": results, "bars": {"de_target": C.BAR_DE_TARGET,
                                "de_minimum": C.BAR_DE_MIN, "en": C.BAR_EN}, "published": C.PUBLISHED})
    t = table(results)
    with open(os.path.join(a.out, "eval.txt"), "w", encoding="utf-8") as f:
        f.write(t + "\n")
    print(t)


if __name__ == "__main__":
    main()
