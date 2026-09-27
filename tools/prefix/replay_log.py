#!/usr/bin/env python3
"""Replay a request log against a running b70-serve and compare two replays (spec 7 C3).

    replay_log.py run <log dir> <base url> <out.json> [--seed 0] [--cap N]
    replay_log.py compare <a.json> <b.json>
    replay_log.py cases <a log dir> <b log dir> <cases file>

<log dir> is what `b70-serve --log-requests DIR` writes (NNNNNN.json, sorted by name), or
tools/prefix/make_synth_log.py's synthetic one. `run` sends every logged request in order,
as logged except: temperature 0 and a fixed seed (greedy), stream on with usage, and
max_tokens capped at the logged response's length (out_ids, or `max_tokens_logged`, or
--cap) so the replay generates what the session generated. Per request it records the
reasoning, content and tool calls, usage.prompt_tokens_details.cached_tokens, the time to
the first streamed chunk with choices (as llama-benchy and clients time it) and the total.

`compare` is the C3 verdict over two runs (cache on and off, same build): per request the
text equal, or the first differing character with both sides' context; every tool call's
arguments parse as JSON and the two runs have the same calls (names and arguments). The
server exposes no logits, so a differing token cannot be judged against the tie rule here:
a divergence is reported for the judge. `cases` reads the two runs' server request logs
(b70-serve --log-requests, prompt_ids and out_ids) and writes, per request whose generated
ids differ, the prompt, the common generated prefix and the two ids at the first
difference, for tools/probe/probe_tie_judge (the tie-aware rule against the cold run,
B = the cache-off run). Exit 0 when every request agrees, 1 otherwise.
"""
import argparse
import json
import os
import sys
import time
import urllib.request


def load_log(log_dir):
    names = sorted(n for n in os.listdir(log_dir) if n.endswith(".json") and n[:-5].isdigit())
    out = []
    for n in names:
        with open(os.path.join(log_dir, n), encoding="utf-8") as f:
            out.append((n, json.load(f)))
    return out


def stream_request(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    reasoning, content, calls, usage, finish, error = "", "", [], None, None, None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            j = json.loads(data)
            if "error" in j:
                error = j["error"]
                continue
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                d = ch.get("delta", {})
                reasoning += d.get("reasoning_content") or ""
                content += d.get("content") or ""
                for tc in d.get("tool_calls") or []:
                    calls.append({"name": tc["function"]["name"],
                                  "arguments": tc["function"].get("arguments", "")})
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    total = time.perf_counter() - t0
    return {"reasoning": reasoning, "content": content, "tool_calls": calls, "usage": usage,
            "finish_reason": finish, "error": error, "ttft_s": ttft, "total_s": total}


def cmd_run(a):
    records = load_log(a.log)
    url = a.url.rstrip("/")
    out = []
    for name, r in records:
        body = dict(r["request"])
        body["temperature"] = 0
        body["seed"] = a.seed
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        logged = len(r.get("out_ids") or []) or r.get("max_tokens_logged") or a.cap
        if a.cap:
            logged = min(logged, a.cap) if logged else a.cap
        if logged:
            body["max_tokens"] = min(int(body.get("max_tokens", logged)), int(logged))
        endpoint = r.get("endpoint", "/v1/chat/completions")
        res = stream_request(url + endpoint, body)
        res["name"] = name
        res["cached_tokens"] = ((res["usage"] or {}).get("prompt_tokens_details") or {}).get(
            "cached_tokens")
        res["prompt_tokens"] = (res["usage"] or {}).get("prompt_tokens")
        out.append(res)
        print(f"{name}: prompt {res['prompt_tokens']} cached {res['cached_tokens']} "
              f"ttft {res['ttft_s'] or 0:.3f} s total {res['total_s']:.3f} s "
              f"calls {len(res['tool_calls'])} finish {res['finish_reason']}"
              + (f" ERROR {res['error']}" if res["error"] else ""), flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


def first_diff(x, y):
    n = min(len(x), len(y))
    i = 0
    while i < n and x[i] == y[i]:
        i += 1
    return i if i < n or len(x) != len(y) else -1


def parse_calls(calls):
    out, bad = [], 0
    for c in calls:
        try:
            out.append((c["name"], json.loads(c["arguments"]) if c["arguments"] else {}))
        except json.JSONDecodeError:
            bad += 1
            out.append((c["name"], None))
    return out, bad


def cmd_compare(a):
    with open(a.a, encoding="utf-8") as f:
        ra = json.load(f)
    with open(a.b, encoding="utf-8") as f:
        rb = json.load(f)
    if len(ra) != len(rb):
        print(f"different request counts: {len(ra)} vs {len(rb)}")
        return 1
    ok_all = True
    print("| # | prompt | cached A | cached B | ttft A s | ttft B s | total A s | total B s |"
          " calls | text | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (x, y) in enumerate(zip(ra, rb)):
        tx = x["reasoning"] + "\x00" + x["content"]
        ty = y["reasoning"] + "\x00" + y["content"]
        d = first_diff(tx, ty)
        cx, bx = parse_calls(x["tool_calls"])
        cy, by = parse_calls(y["tool_calls"])
        calls_ok = bx == 0 and by == 0 and cx == cy
        errs = bool(x["error"] or y["error"])
        ok = d < 0 and calls_ok and not errs
        ok_all = ok_all and ok
        text = "equal" if d < 0 else f"differs at char {d}"
        calls = f"{len(cx)}" + ("" if calls_ok else f" (bad {bx}/{by}, same {cx == cy})")
        print(f"| {i + 1} | {x['prompt_tokens']} | {x['cached_tokens']} | {y['cached_tokens']} |"
              f" {x['ttft_s'] or 0:.3f} | {y['ttft_s'] or 0:.3f} | {x['total_s']:.2f} |"
              f" {y['total_s']:.2f} | {calls} | {text} | {'ok' if ok else 'DIFF'} |")
        if d >= 0:
            print(f"    A: ...{tx[max(0, d - 40):d + 40]!r}")
            print(f"    B: ...{ty[max(0, d - 40):d + 40]!r}")
        if errs:
            print(f"    errors: A {x['error']} B {y['error']}")
    sa = sum(x["ttft_s"] or 0 for x in ra)
    sb = sum(y["ttft_s"] or 0 for y in rb)
    print(f"sum of time to first token: A {sa:.2f} s, B {sb:.2f} s")
    print("C3 machinery: " + ("every request agrees" if ok_all else "divergences above"))
    return 0 if ok_all else 1


def cmd_cases(a):
    ra, rb = load_log(a.a), load_log(a.b)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for (name, x), (_, y) in zip(ra, rb):
            if x["prompt_ids"] != y["prompt_ids"]:
                print(f"{name}: prompts differ between the runs")
                continue
            ox, oy = x["out_ids"], y["out_ids"]
            k = 0
            while k < min(len(ox), len(oy)) and ox[k] == oy[k]:
                k += 1
            if k == len(ox) == len(oy):
                continue
            if k == min(len(ox), len(oy)):
                print(f"{name}: one run stopped at {k} (end of sequence), the other did not")
                continue
            f.write("P " + " ".join(map(str, x["prompt_ids"])) + "\n")
            f.write("G " + " ".join(map(str, ox[:k])) + "\n")
            f.write(f"C {ox[k]} {oy[k]} {name}@{k}\n")
            n += 1
            print(f"{name}: first differing generated id at {k}: {ox[k]} vs {oy[k]}")
    print(f"{n} case(s) in {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("log")
    r.add_argument("url")
    r.add_argument("out")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--cap", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    k = sub.add_parser("cases")
    k.add_argument("a")
    k.add_argument("b")
    k.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "cases":
        cmd_cases(a)
        return 0
    if a.cmd == "run":
        cmd_run(a)
        return 0
    return cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
