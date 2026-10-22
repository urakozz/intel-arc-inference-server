#!/usr/bin/env python3
"""Replay a recorded coding session as a LIVE agent loop against b70-serve and measure the prefix cache (spec 24e Task 5).

    agentic_loop.py run <transcript.json> <base url> <out.json> [--turns N] [--max-tokens N] [--seed S] [--extra JSON]
    agentic_loop.py report <out.json> [<server --log-requests dir>]

<transcript.json> is a chat session ({"messages": [...], "tools": [...]}: system, user, then assistant / tool turns -
tests/golden/prompts/lfm2moe_agentic.json, 24a's). `run` plays the client's side of it: it sends [system, first user]
and, after every response, appends the MODEL's own turn as an OpenAI client does - content, reasoning_content and
tool_calls exactly as the server returned them - and then the next recorded tool result (in order, one per call
the model made, whatever the model called; a turn with more calls than recorded results left wraps to the first of them, and
a turn that makes a call when none is left ends the loop) or, when the model made no call, the next recorded user message; it stops
when the transcript has nothing left to say, the model answered without a call and no user message remains, or after
--turns requests. Because the assistant turns are the server's own, the prefix cache sees what it would see in a
real session: a turn the template re-renders in another order than the model generated it (content then calls; the
model may write a call before its text) is a miss at that point.

--extra JSON (spec 25e): a JSON object merged into every request body - Bolt's {"chat_template_kwargs":
{"truncate_history_thinking": false}} arm, say; the default sends none, so a model's own template defaults apply.

Per request it records (stream, temperature 0, a fixed seed): prompt tokens, usage.prompt_tokens_details.cached_tokens,
the time to the first streamed chunk with choices, the total, the completion tokens, the decode rate
(completion tokens - 1) / (total - ttft), the finish reason, and `reordered`: the turn streamed a tool call before some
content (the order the template cannot re-render).

`report` prints one table row per request, the hit rate (cached / prompt tokens over the session), the cached share of each
turn that followed another, the number of reordered turns, and - given the server's --log-requests directory - the ideal
reuse of tools/prefix/analyze_log.py's rule (the longest common prefix of a prompt with the previous request's prompt +
generated ids) beside what was cached.
"""
import argparse
import json
import os
import sys
import time
import urllib.request


def stream_request(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    reasoning, content, calls, usage, finish, error = "", "", [], None, None, None
    reordered = False
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
                piece = d.get("content") or ""
                if piece and calls:
                    reordered = True   # content after a call: the template renders content first
                content += piece
                for tc in d.get("tool_calls") or []:
                    calls.append({"id": tc.get("id", ""), "name": tc["function"]["name"],
                                  "arguments": tc["function"].get("arguments", "")})
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    total = time.perf_counter() - t0
    return {"reasoning": reasoning, "content": content, "tool_calls": calls, "usage": usage, "finish_reason": finish,
            "error": error, "ttft_s": ttft, "total_s": total, "reordered": reordered}


def assistant_message(res):
    """The message a client sends back for the server's response."""
    msg = {"role": "assistant", "content": res["content"] or None}
    if res["reasoning"]:
        msg["reasoning_content"] = res["reasoning"]
    if res["tool_calls"]:
        msg["tool_calls"] = [{"id": c["id"], "type": "function",
                              "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                             for c in res["tool_calls"]]
    return msg


def plan(transcript):
    """(system messages, user messages, tool results) of the transcript, in order."""
    system = [m for m in transcript["messages"] if m["role"] == "system"]
    users = [m for m in transcript["messages"] if m["role"] == "user"]
    tools = [m for m in transcript["messages"] if m["role"] == "tool"]
    return system, users, tools


def cmd_run(a):
    with open(a.transcript, encoding="utf-8") as f:
        t = json.load(f)
    extra = json.loads(a.extra) if a.extra else {}
    if not isinstance(extra, dict):
        sys.exit("--extra must be a JSON object")
    system, users, results = plan(t)
    if not users:
        sys.exit("the transcript has no user message")
    messages = system + [users[0]]
    next_user, next_result = 1, 0
    url = a.url.rstrip("/") + "/v1/chat/completions"
    out = []
    for turn in range(a.turns):
        body = {"model": "b70", "messages": messages, "tools": t.get("tools"), "temperature": 0, "seed": a.seed,
                "stream": True, "stream_options": {"include_usage": True}, "max_tokens": a.max_tokens}
        if not body["tools"]:
            del body["tools"]
        body.update(extra)
        res = stream_request(url, body)
        usage = res["usage"] or {}
        res["turn"] = turn + 1
        res["prompt_tokens"] = usage.get("prompt_tokens")
        res["completion_tokens"] = usage.get("completion_tokens")
        res["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        decode_s = (res["total_s"] - (res["ttft_s"] or 0.0))
        res["decode_tps"] = (res["completion_tokens"] - 1) / decode_s if res["completion_tokens"] and decode_s > 0 else None
        res["calls_n"] = len(res["tool_calls"])
        out.append(res)
        print(f"turn {turn + 1}: prompt {res['prompt_tokens']} cached {res['cached_tokens']} ttft {res['ttft_s'] or 0:.3f} s "
              f"decode {res['decode_tps'] or 0:.1f} t/s calls {res['calls_n']} finish {res['finish_reason']}"
              f"{' REORDERED' if res['reordered'] else ''}" + (f" ERROR {res['error']}" if res["error"] else ""),
              flush=True)
        if res["error"]:
            break
        messages = messages + [assistant_message(res)]
        if res["tool_calls"]:
            if next_result >= len(results):
                break
            for c in res["tool_calls"]:
                r = results[next_result % len(results)]
                next_result += 1
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": r["content"]})
        elif next_user < len(users):
            messages.append(users[next_user])
            next_user += 1
        else:
            break
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


def load_log(log_dir):
    names = sorted(n for n in os.listdir(log_dir) if n.endswith(".json") and n[:-5].isdigit())
    out = []
    for n in names:
        with open(os.path.join(log_dir, n), encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def ideal_reuse(log):
    """Per request: the longest common prefix of its prompt with the previous request's prompt + generated ids."""
    ideal, prev = [], []
    for r in log:
        p = r["prompt_ids"]
        n = min(len(p), len(prev))
        k = 0
        while k < n and p[k] == prev[k]:
            k += 1
        ideal.append(k)
        prev = p + r["out_ids"]
    return ideal


def divergences(log):
    """Per request: (k, the prompt's id at k, the previous prompt + generated id at k) where the prompt stops matching the
    previous request's prompt + generated ids - the first token a miss diverges at (spec 25e Review Focus 5); None for the first
    request or a prompt that is a prefix of the previous one."""
    out, prev = [], []
    for r in log:
        p = r["prompt_ids"]
        k = 0
        n = min(len(p), len(prev))
        while k < n and p[k] == prev[k]:
            k += 1
        out.append(None if not prev or k >= len(p) else (k, p[k], prev[k] if k < len(prev) else None))
        prev = p + r["out_ids"]
    return out


def cmd_report(a):
    with open(a.run, encoding="utf-8") as f:
        runs = json.load(f)
    ideal = ideal_reuse(load_log(a.log)) if a.log else None
    print("| turn | prompt | cached | cached % | ideal | ttft s | decode t/s | calls | order | finish |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    cached_sum = prompt_sum = reordered = 0
    for i, r in enumerate(runs):
        p, c = r.get("prompt_tokens") or 0, r.get("cached_tokens") or 0
        cached_sum += c
        prompt_sum += p
        reordered += 1 if r.get("reordered") else 0
        pct = f"{100.0 * c / p:.0f}" if p else "-"
        ide = str(ideal[i]) if ideal is not None and i < len(ideal) else "-"
        print(f"| {r['turn']} | {p} | {c} | {pct} | {ide} | {r['ttft_s'] or 0:.3f} | "
              f"{(r['decode_tps'] or 0):.1f} | {r['calls_n']} | {'call before text' if r.get('reordered') else 'ok'} | "
              f"{r['finish_reason']} |")
    follow = [r for r in runs[1:] if r.get("prompt_tokens")]
    follow_hit = sum(r.get("cached_tokens") or 0 for r in follow) / max(1, sum(r["prompt_tokens"] for r in follow))
    print(f"\nhit rate over the session: {cached_sum}/{prompt_sum} = {100.0 * cached_sum / max(1, prompt_sum):.1f} %")
    print(f"hit rate of the turns that followed another: {100.0 * follow_hit:.1f} %")
    print(f"turns whose call the template would re-render in another order than generated: {reordered} of {len(runs)}")
    if ideal is not None:
        print(f"ideal prefix reuse (analyze_log.py's rule): {sum(ideal)} of {sum(r.get('prompt_tokens') or 0 for r in runs)}")
        for r, d in zip(runs, divergences(load_log(a.log))):
            if d is not None and (r.get("cached_tokens") or 0) < (r.get("prompt_tokens") or 0):
                other = f"id {d[2]}" if d[2] is not None else "nothing (it ended)"
                print(f"turn {r['turn']} diverges from the previous prompt + generated ids at token {d[0]}: its prompt has id "
                      f"{d[1]}, the previous request's text has {other}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("transcript")
    r.add_argument("url")
    r.add_argument("out")
    r.add_argument("--turns", type=int, default=12)
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--extra", default="", help="a JSON object merged into every request body (e.g. chat_template_kwargs)")
    q = sub.add_parser("report")
    q.add_argument("run")
    q.add_argument("log", nargs="?")
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
