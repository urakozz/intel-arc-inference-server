#!/usr/bin/env python3
"""A per-request client for b70-serve (stdlib only), for the box runbook's server rows.

  serve_client.py run --url http://127.0.0.1:8013 --out arm.r1/requests.jsonl \\
      [--set golden] [--set toolcall] [--set DIR] [--only NAME,...] [--max-tokens 256]
      [--label ARM] [--round N] [--server-log DIR] [--warmup 1] [--stop-at-eos]
  serve_client.py table   --dir D [--ref ARM] [--ratio A/B]... [--metric gen_tps|wall_tps]
  serve_client.py compare --dir D --ref ARM [--round N]

`run` sends one OpenAI request per prompt, one at a time, temperature 0 (greedy), with
`return_token_ids` so the output ids come back:
  golden    tests/golden/prompts/{prose,code,cjk}.txt as /v1/completions (text prompts)
  toolcall  the A4 set, tests/golden/toolcall/<name>.json (messages, tools, enable_thinking)
            as /v1/chat/completions, every scenario of manifest.json
  DIR       a directory: manifest.json + <name>.json as toolcall, else every *.txt as golden
Each prompt's ids as the server encoded them are checked against the set's committed
<name>.ids (`prompt_ids_match`; a mismatch is reported, the request still counts). By default
the request asks for exactly --max-tokens ids (`ignore_eos`), so arms time the same work;
--stop-at-eos lets the model stop. One JSON line per request in --out: label, round, set,
name, endpoint, prompt / output token counts, out_ids, text, finish_reason, usage, wall
seconds, wall_tps (output ids / wall, prefill included) and - with --server-log (the
server's --log-requests directory) - prefill_s, gen_s and gen_tps ((ids - 1) / (t_end -
t_first_token): decode only) from the server's own record of that request. Accepted drafts:
`accepted` is usage.completion_tokens_details.accepted_prediction_tokens or any usage field
naming drafts / acceptance, when the server reports one (b70-serve's usage does not yet:
None). Exit 0 when every request was answered.

`table` reads every requests.jsonl under --dir and prints, per set and prompt, the median
over rounds of --metric per arm, then `RATIO <arm>/<ref> <set> geomean x` per arm (and per
--ratio A/B pair): the geometric mean over the set's prompts of the median ratio.
`compare` checks greedy identity: every request of every other arm's round has the --ref
arm's output ids; prints `IDENTICAL a vs ref: n/n` or the first differing index; exit 1 on
any difference or missing request.
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN = ("prose", "code", "cjk")


def read_ids(path):
    try:
        with open(path) as f:
            return [int(x) for x in f.read().split()]
    except OSError:
        return None


def load_set(spec, root):
    """[(set name, prompt name, endpoint, body without the sampling fields, expected ids)]"""
    out = []
    if spec == "golden":
        d = os.path.join(root, "tests", "golden", "prompts")
        for n in GOLDEN:
            with open(os.path.join(d, n + ".txt"), encoding="utf-8") as f:
                text = f.read()
            out.append(("golden", n, "/v1/completions", {"prompt": text}, read_ids(os.path.join(d, n + ".ids"))))
        return out
    d = os.path.join(root, "tests", "golden", "toolcall") if spec == "toolcall" else spec
    tag = "toolcall" if spec == "toolcall" else os.path.basename(os.path.normpath(d))
    manifest = os.path.join(d, "manifest.json")
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            names = [e["name"] for e in json.load(f)]
        for n in names:
            with open(os.path.join(d, n + ".json"), encoding="utf-8") as f:
                sc = json.load(f)
            body = {"messages": sc["messages"]}
            if sc.get("tools"):
                body["tools"] = sc["tools"]
            body["chat_template_kwargs"] = {"enable_thinking": bool(sc.get("enable_thinking", False))}
            out.append((tag, n, "/v1/chat/completions", body, read_ids(os.path.join(d, n + ".ids"))))
        return out
    for p in sorted(glob.glob(os.path.join(d, "*.txt"))):
        n = os.path.splitext(os.path.basename(p))[0]
        with open(p, encoding="utf-8") as f:
            text = f.read()
        out.append((tag, n, "/v1/completions", {"prompt": text}, read_ids(os.path.join(d, n + ".ids"))))
    if not out:
        raise SystemExit(f"serve_client: no prompts in set '{spec}'")
    return out


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def log_files(d):
    return set(glob.glob(os.path.join(d, "*.json"))) if d else set()


def accepted_of(usage):
    if not isinstance(usage, dict):
        return None
    det = usage.get("completion_tokens_details")
    if isinstance(det, dict) and "accepted_prediction_tokens" in det:
        return det["accepted_prediction_tokens"]
    for k, v in usage.items():
        if any(w in k for w in ("accept", "draft", "speculat")):
            return v
    return None


def run(a):
    prompts = []
    for spec in a.set or ["golden"]:
        prompts += load_set(spec, a.root)
    if a.only:
        keep = set(a.only.split(","))
        prompts = [p for p in prompts if p[1] in keep or f"{p[0]}/{p[1]}" in keep]
    base = a.url.rstrip("/")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def body_for(endpoint, extra, max_tokens):
        b = {"model": a.model, "max_tokens": max_tokens, "temperature": 0, "return_token_ids": True}
        if not a.stop_at_eos:
            b["ignore_eos"] = True
        b.update(extra)
        return b

    for i in range(a.warmup):   # not recorded: the first request after a load pays one-off costs
        _, _, ep, extra, _ = prompts[i % len(prompts)]
        post(base + ep, body_for(ep, extra, 16), a.timeout)
    failed = 0
    with open(a.out, "a", encoding="utf-8") as out:
        for sname, name, ep, extra, want in prompts:
            before = log_files(a.server_log)
            t0 = time.monotonic()
            try:
                resp = post(base + ep, body_for(ep, extra, a.max_tokens), a.timeout)
            except (urllib.error.URLError, OSError, ValueError) as e:
                failed += 1
                print(f"REQ {a.label} {sname}/{name} FAILED: {e}", flush=True)
                continue
            wall = time.monotonic() - t0
            ch = resp["choices"][0]
            ids = ch.get("token_ids") or []
            if "message" in ch:
                m = ch["message"]
                text = (m.get("content") or "") + ("".join(json.dumps(c, sort_keys=True) for c in m["tool_calls"])
                                                   if m.get("tool_calls") else "")
            else:
                text = ch.get("text", "")
            pids = resp.get("prompt_token_ids")
            usage = resp.get("usage", {})
            rec = {"label": a.label, "round": a.round, "set": sname, "name": name, "endpoint": ep,
                   "prompt_tokens": usage.get("prompt_tokens", len(pids) if pids else None),
                   "prompt_ids_match": (pids == want) if (pids is not None and want is not None) else None,
                   "completion_tokens": usage.get("completion_tokens", len(ids)), "out_ids": ids,
                   "text": text, "finish_reason": ch.get("finish_reason"), "usage": usage,
                   "accepted": accepted_of(usage), "wall_s": round(wall, 4),
                   "wall_tps": round(len(ids) / wall, 3) if wall > 0 else None,
                   "prefill_s": None, "gen_s": None, "gen_tps": None}
            new = sorted(log_files(a.server_log) - before)
            if new:
                try:
                    with open(new[-1], encoding="utf-8") as f:
                        lg = json.load(f)
                    gen = lg["t_end"] - lg["t_first_token"]
                    rec["prefill_s"] = round(lg["t_first_token"] - lg["t_start"], 4)
                    rec["gen_s"] = round(gen, 4)
                    n = len(lg.get("out_ids", ids))
                    rec["gen_tps"] = round((n - 1) / gen, 3) if gen > 0 and n > 1 else None
                    rec["server_log"] = os.path.basename(new[-1])
                except (OSError, ValueError, KeyError) as e:
                    rec["server_log_error"] = str(e)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            match = {True: "ids match", False: "ids DIFFER from the set's", None: "ids unchecked"}[rec["prompt_ids_match"]]
            print(f"REQ {a.label} r{a.round} {sname}/{name} prompt={rec['prompt_tokens']} ({match}) "
                  f"out={len(ids)} finish={rec['finish_reason']} wall={wall:.2f}s wall_tps={rec['wall_tps']} "
                  f"gen_tps={rec['gen_tps']} accepted={rec['accepted']}", flush=True)
    print(f"serve_client: {len(prompts) - failed}/{len(prompts)} requests answered -> {a.out}")
    return 1 if failed else 0


def records(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "**", "requests.jsonl"), recursive=True)):
        with open(p, encoding="utf-8") as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


def table(a):
    recs = records(a.dir)
    if not recs:
        print(f"serve_client table: no requests.jsonl under {a.dir}")
        return 1
    arms = []
    for r in recs:
        if r["label"] not in arms:
            arms.append(r["label"])
    vals = {}
    for r in recs:
        v = r.get(a.metric)
        if v is None and a.metric == "gen_tps":
            v = r.get("wall_tps")
        if v is not None:
            vals.setdefault((r["set"], r["name"]), {}).setdefault(r["label"], []).append(v)
    print(f"| prompt | " + " | ".join(f"{x} {a.metric} (n)" for x in arms) + " |")
    print("|---|" + "---:|" * len(arms))
    for key in sorted(vals, key=lambda k: (k[0] != "golden", k)):
        cells = []
        for x in arms:
            v = vals[key].get(x)
            cells.append(f"{statistics.median(v):.2f} ({len(v)})" if v else "-")
        print(f"| {key[0]}/{key[1]} | " + " | ".join(cells) + " |")
    ref = a.ref or arms[0]
    pairs = [(x, ref) for x in arms if x != ref] + [tuple(r.split("/", 1)) for r in a.ratio or []]
    for x, y in pairs:
        for s in sorted({k[0] for k in vals}):
            logs = [math.log(statistics.median(v[x]) / statistics.median(v[y]))
                    for k, v in vals.items() if k[0] == s and v.get(x) and v.get(y)]
            if logs:
                print(f"RATIO {x}/{y} {s} geomean {math.exp(sum(logs) / len(logs)):.4f} over {len(logs)} prompts")
    return 0


def compare(a):
    recs = records(a.dir)
    by = {}
    for r in recs:
        by.setdefault((r["label"], r["round"]), {})[(r["set"], r["name"])] = r
    refs = {k: v for k, v in by.items() if k[0] == a.ref and (a.round is None or k[1] == a.round)}
    if not refs:
        print(f"serve_client compare: no requests of arm '{a.ref}' under {a.dir}")
        return 1
    ref = refs[min(refs)]
    bad = 0
    for (label, rnd), reqs in sorted(by.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
        if label == a.ref or (a.round is not None and rnd != a.round):
            continue
        same = 0
        for key, r0 in sorted(ref.items()):
            r = reqs.get(key)
            if r is None:
                bad += 1
                print(f"MISSING {label} r{rnd} {key[0]}/{key[1]}")
                continue
            x, y = r0["out_ids"], r["out_ids"]
            if x == y:
                same += 1
                continue
            bad += 1
            i = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
            print(f"DIFFER {label} r{rnd} vs {a.ref} {key[0]}/{key[1]} at index {i} "
                  f"({len(y)} vs {len(x)} ids)")
        print(f"{'IDENTICAL' if same == len(ref) else 'NOT IDENTICAL'} {label} r{rnd} vs {a.ref}: {same}/{len(ref)}")
    unmatched = sum(1 for r in recs if r.get("prompt_ids_match") is False)
    if unmatched:
        print(f"NOTE {unmatched} requests' prompt ids differ from the set's committed .ids "
              f"(the server's tokenisation; the arms still compare like for like)")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8013")
    r.add_argument("--out", required=True)
    r.add_argument("--set", action="append", help="golden, toolcall or a directory (repeatable)")
    r.add_argument("--only", help="comma list of prompt names (or set/name) to keep")
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--model", default="b70")
    r.add_argument("--label", default="arm")
    r.add_argument("--round", type=int, default=1)
    r.add_argument("--server-log", help="the server's --log-requests directory")
    r.add_argument("--warmup", type=int, default=1)
    r.add_argument("--stop-at-eos", action="store_true")
    r.add_argument("--timeout", type=float, default=1800)
    r.add_argument("--root", default=ROOT)
    t = sub.add_parser("table")
    t.add_argument("--dir", required=True)
    t.add_argument("--ref")
    t.add_argument("--ratio", action="append", help="A/B: another pair of arms to compare")
    t.add_argument("--metric", default="gen_tps", choices=("gen_tps", "wall_tps"))
    c = sub.add_parser("compare")
    c.add_argument("--dir", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--round", type=int)
    a = ap.parse_args(argv)
    return {"run": run, "table": table, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
