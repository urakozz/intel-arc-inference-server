#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage `card` of tools/quantize_kolibri1.sh: the model card for
urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ (spec 20 §3.1 "output").

    card.py --work <OUT> [--arm auto|<arm name>] [--repo urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ]

Picks the arm to publish - `auto`: the tuned arm with int4 attention if it clears every bar (KL DE <
0.069, KL EN <= 0.013), else the tuned arm with bf16 attention if that does, else none (exit 2: no
card for a checkpoint below the minimum bar) - and writes README.md into its export directory: the
KL / top-1 / perplexity table for every measured arm against the bars and the published numbers, the
calibration recipe (mix, sizes, sources and revisions, prompt-file hashes), the per-expert coverage
summary, the AutoRound commit and call, what is int4 and what stays bf16, and how to load it.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def pick(results: dict, want: str) -> str | None:
    if want != "auto":
        if want not in results:
            raise SystemExit(f"--arm {want}: not in eval.json ({sorted(results)})")
        return want
    for arm in ("tune-attn_int4", "tune-attn_bf16"):
        r = results.get(arm)
        if r and r["bars"]["de_minimum"] and r["bars"]["en"]:
            return arm
    return None


def pct(x):
    return f"{x * 100:.2f} %"


def render(arm: str, work: str, repo: str) -> str:
    ev = C.read_json(os.path.join(work, "eval", "eval.json"))
    cov_p = os.path.join(work, "coverage.json")
    cov = C.read_json(cov_p) if os.path.exists(cov_p) else None
    man = C.read_json(os.path.join(work, "calib_manifest.json"))
    r = ev["arms"][arm]
    export = r["export"]
    recipe = C.read_json(os.path.join(export, "kolibri_recipe.json"))
    chk_p = os.path.join(work, f"check-{arm}.json")
    chk = C.read_json(chk_p) if os.path.exists(chk_p) else {}
    attn = recipe.get("attn", "?")
    ar = recipe.get("autoround", {})
    L = []
    L += ["---", "license: apache-2.0", "language:", "- de", "- en", "base_model: Aleph-Alpha/Kolibri-1-BF16",
          "pipeline_tag: text-generation", "tags:", "- kolibri1", "- moe", "- auto-round", "- gptq", "- int4", "---", ""]
    L += [f"# {repo.split('/')[-1]}", "",
          "Aleph Alpha's [Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16) (78B total / 3.46B active, "
          "German and English) quantised to **int4, group size 64, symmetric** with "
          "[AutoRound](https://github.com/intel/auto-round), exported in the GPTQ layout "
          "(`auto_round:auto_gptq`), with a **German-first calibration**. Made for the "
          "[b70-inference-server](https://github.com/urakozz) engine (Intel Arc Pro B70), loadable by anything "
          "that reads symmetric GPTQ v1 int4 (`w = (q - 8) * scale`, `qzeros` 0x77777777, no `g_idx` permutation).", ""]
    L += ["## What is quantised", "",
          "| tensors | precision |", "|---|---|",
          "| routed experts `mlp.experts.*.{gate,up,down}_proj` (50 x 384 x 3) | int4 g64 symmetric |",
          f"| attention `self_attn.{{q,k,v,o}}_proj` | {'int4 g64 symmetric' if attn == 'int4' else 'bf16'} |",
          "| router `mlp.gate`, `moe.router.expert_bias` | bf16 (routing must not move) |",
          "| shared experts `mlp.shared_experts.*` | bf16 (every token reads them) |",
          "| norms, `embed_tokens`, `lm_head` | bf16 |", ""]
    if chk:
        L += [f"Size: {chk.get('bytes_total', 0) / 1e9:.1f} GB ({chk.get('int4_linears', '?')} int4 linears). "
              f"Every bf16 tensor is bitwise the original's.", ""]
    L += ["## Quality against bf16 (held-out text, never calibrated on)", "",
          "KL(bf16 || quantised) in nats per token, top-1 agreement and perplexity, teacher-forced over every "
          "position; DE = German Wikipedia (raw text) + German chat, EN = English chat; the chat sets are the "
          "bf16 model's own answers (reasoning on) to held-out prompts.", "",
          "| arm | set | tokens | KL | top-1 | ppl bf16 | ppl int4 |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, rr in ev["arms"].items():
        rows = [(s.upper(), m) for s, m in rr["lang"].items() if s != "code"] + list(rr["sets"].items())
        for s, m in rows:
            mark = " **(this checkpoint)**" if name == arm and s in ("DE", "EN") else ""
            L.append(f"| {name}{mark} | {s} | {m['tokens']} | "
                     f"{m['kl']:.4f} | {pct(m['top1'])} | {m['ppl_bf16']:.3f} | {m['ppl_quant']:.3f} |")
    L += ["", f"Bars (spec 20 §3.1): KL DE < {C.BAR_DE_TARGET} (Aleph Alpha's FP8) target, < {C.BAR_DE_MIN} "
              f"(the community asymmetric g128 GPTQ) minimum, KL EN <= {C.BAR_EN}. This checkpoint: "
              f"DE {r['lang'].get('de', {}).get('kl', float('nan')):.4f}, EN {r['lang'].get('en', {}).get('kl', float('nan')):.4f} - "
              f"target {'met' if r['bars']['de_target'] else 'not met'}, minimum {'met' if r['bars']['de_minimum'] else 'NOT met'}.",
          "", "Published references (their own text sets, so comparable in kind, not to the digit):", ""]
    for n, d, t, e, te in C.PUBLISHED:
        L.append(f"- {n}: KL DE {d}, top-1 DE {pct(t)}" + (f"; KL EN {e}, top-1 EN {pct(te)}" if e else ""))
    L += ["", "## Calibration", "",
          f"{man['nsamples_base']} base rows of {man['seqlen']} tokens"
          + (f" + {cov['topup_rows']} coverage top-up rows" if cov else "")
          + f", every row starting at a conversation boundary, all chat-formatted through the model's own template "
            f"(reasoning on). Token share of the base rows: German {pct(man['token_share']['de'])}, English "
            f"{pct(man['token_share']['en'])}, code {pct(man['token_share']['code'])}.", "",
          "| category | rows | source |", "|---|---:|---|"]
    src = {"de_chat": "German prompts (prompts/de_chat.jsonl), answered by the bf16 model",
           "de_tool": "German tool-calling scenarios (prompts/de_tools.jsonl): the model's <tool_call>, a canned "
                      "<tool_response>, the model's answer",
           "de_doc": f"German Wikipedia articles ({man['documents'].get('dataset', 'local')} @ "
                     f"{str(man['documents'].get('revision', ''))[:12]}, {man['documents'].get('calib')}) under an "
                     "instruction, answered by the bf16 model",
           "en_chat": "English prompts (prompts/en_chat.jsonl), answered by the bf16 model",
           "code": "code prompts, German and English (prompts/code.jsonl), answered by the bf16 model"}
    for c, n in man["rows_by_cat"].items():
        L.append(f"| {c} | {n} | {src.get(c, '')} |")
    if cov:
        L.append(f"| de_topup | {cov['topup_rows']} | German Wikipedia articles chosen to fill starved experts |")
    L += ["", f"Generation: {man['generator']}, sampling {man['sampling']}, max {man['gen_max']} new tokens per turn, "
              f"seed {man['seed']}. Evaluation sets: {man['documents'].get('eval')} (documents disjoint from "
              "calibration) and the prompts' held-out split.", ""]
    if cov:
        before = cov["per_layer_before"]
        after = cov["per_layer_after"]
        L += ["## Expert coverage", "",
              f"Routed tokens per expert per layer on the calibration set (floor {cov['floor']}): "
              f"{cov['starved_before']} of {len(before) * len(cov['counts_before'][0])} experts were under the floor "
              f"before the top-up, {cov['starved_after']} after. Per layer, the least-routed expert's tokens "
              f"(min over layers / median over layers): before {min(x['min'] for x in before)} / "
              f"{sorted(x['min'] for x in before)[len(before) // 2]}, after {min(x['min'] for x in after)} / "
              f"{sorted(x['min'] for x in after)[len(after) // 2]}.", ""]
    L += ["## Recipe", "",
          f"- AutoRound {ar.get('version', '?')} at commit `{ar.get('commit', '?')}`",
          f"- call: `{recipe.get('call', '?')}`",
          f"- {'RTN (iters 0)' if recipe.get('iters') == 0 else 'sign-SGD tuning, ' + str(recipe.get('iters')) + ' iterations'}, "
          f"{recipe.get('calib_rows')} calibration rows (sha256 `{recipe.get('calib_sha256', '')[:16]}...`)",
          "- the model class: `modeling_kolibri1.py` / `configuration_kolibri1.py` (in this repository; "
          "a transformers port of Aleph Alpha's vLLM plugin, Apache-2.0)",
          "- reproduce: `tools/quantize_kolibri1.sh` in b70-inference-server (stages calib, coverage, rtn, tune, "
          "eval, check, card)", ""]
    L += ["## Loading", "",
          "- transformers: `AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)` with an int4 GPTQ "
          "backend (the checkpoint is AutoRound's `auto_round:auto_gptq` export).",
          "- vLLM: with Aleph Alpha's `aleph-alpha-inference` plugin (the `kolibri1` architecture), as a GPTQ "
          "checkpoint.",
          "- Chat template, reasoning switch and tool-call format are the original's (ChatML, `<think>`, "
          "hermes-style JSON in `<tool_call>`).", "",
          "## License", "", "Apache-2.0, as the original weights.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--arm", default="auto")
    ap.add_argument("--repo", default="urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ")
    a = ap.parse_args()
    ev = C.read_json(os.path.join(a.work, "eval", "eval.json"))
    arm = pick(ev["arms"], a.arm)
    if arm is None:
        print("no tuned arm clears the minimum bars (KL DE < %.3f, KL EN <= %.3f); no card written. Measured: %s"
              % (C.BAR_DE_MIN, C.BAR_EN, {k: (round(v["lang"].get("de", {}).get("kl", -1), 4),
                                               round(v["lang"].get("en", {}).get("kl", -1), 4))
                                           for k, v in ev["arms"].items()}), file=sys.stderr)
        sys.exit(2)
    text = render(arm, a.work, a.repo)
    export = ev["arms"][arm]["export"]
    with open(os.path.join(export, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)
    C.log(f"card for {arm}: {os.path.join(export, 'README.md')}")
    print(f"publish: huggingface-cli upload {a.repo} {export} .")


if __name__ == "__main__":
    main()
