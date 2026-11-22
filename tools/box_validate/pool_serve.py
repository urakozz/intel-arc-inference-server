#!/usr/bin/env python3
"""The KV page pool through b70-serve (spec 31, plan 31 Task 7), stdlib only. Each case talks to one running server
(tools/box_validate/pool_serve.sh starts it) and writes its ids to --out; `compare` checks two runs' files.

  pool_serve.py lone  --url U --out F            the golden prompts and long.txt one at a time, greedy
  pool_serve.py long  --url U --out F [--long-tokens 100000]
                                                 a long prompt beside two short ones sent together, then the short
                                                 ones alone: the long one is served, the short ones' ids as alone
  pool_serve.py evict --url U --out F [--long-tokens 100000] [--short-tokens 15000] [--no-shorts]
                                                 a long session's first turn, three short requests together (enough
                                                 pages that the idle long session leaves the card), its second turn
  pool_serve.py compare A.json B.json            the same ids in both (IDENTICAL / NOT IDENTICAL per request)
  pool_serve.py check-long F                     a `long` run's verdict: POOL_LONG served / FAIL

Greedy, ignore_eos, return_token_ids. Exit 0 when every request was answered (compare: when every one matches).
"""
import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def text(name):
    with open(os.path.join(ROOT, "tests", "golden", "prompts", name), encoding="utf-8") as f:
        return f.read()


def complete(url, prompt, max_tokens, user):
    body = {"model": "b70", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
            "return_token_ids": True, "user": user}
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"user": user, "error": e.code, "body": e.read().decode(errors="replace")[:300]}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"user": user, "error": str(e)}
    c = out["choices"][0]
    usage = out.get("usage", {})
    return {"user": user, "ids": c.get("token_ids") or c.get("output_token_ids"), "text": c.get("text", ""),
            "finish": c.get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


def together(url, reqs):
    """Send (prompt, max_tokens, user) requests at once; their answers in the same order."""
    out = [None] * len(reqs)

    def one(i):
        out[i] = complete(url, *reqs[i])

    ts = [threading.Thread(target=one, args=(i,)) for i in range(len(reqs))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def long_text(tokens):
    # long.txt is about 3,000 tokens; numbered copies up to about `tokens` (4 characters a token, roughly).
    base = text("long.txt")
    n = max(1, tokens * 4 // len(base))
    return "".join(f"[part {i}]\n{base}\n" for i in range(n))


def say(r):
    if "error" in r:
        print(f"  {r['user']}: ERROR {r['error']} {r.get('body', '')}")
    else:
        print(f"  {r['user']}: prompt {r['prompt_tokens']} tokens, cached {r['cached']}, {len(r['ids'] or [])} ids, "
              f"finish {r['finish']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lone", "long", "evict"):
        p = sub.add_parser(name)
        p.add_argument("--url", default="http://127.0.0.1:8013")
        p.add_argument("--out", required=True)
        p.add_argument("--max-tokens", type=int, default=64)
        p.add_argument("--long-tokens", type=int, default=100000)
        p.add_argument("--short-tokens", type=int, default=15000)
        p.add_argument("--no-shorts", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    k = sub.add_parser("check-long")
    k.add_argument("file")
    a = ap.parse_args()

    if a.cmd == "check-long":
        with open(a.file) as f:
            r = json.load(f)
        ok = "error" not in r["long.beside"] and all(
            r[n + ".beside"].get("ids") is not None and r[n + ".beside"].get("ids") == r[n + ".alone"].get("ids")
            for n in ("prose", "code"))
        print(f"POOL_LONG {'served, the short ones as alone' if ok else 'FAIL'} (prompt "
              f"{r['long.beside'].get('prompt_tokens')} tokens)")
        return 0 if ok else 1

    if a.cmd == "compare":
        with open(a.a) as f:
            x = json.load(f)
        with open(a.b) as f:
            y = json.load(f)
        ok = True
        for k in sorted(set(x) & set(y)):
            same = x[k].get("ids") is not None and x[k].get("ids") == y[k].get("ids")
            ok &= same
            print(f"{'IDENTICAL' if same else 'NOT IDENTICAL'} {k}")
        missing = sorted(set(x) ^ set(y))
        if missing:
            print(f"NOT IDENTICAL: in one run only: {', '.join(missing)}")
            ok = False
        return 0 if ok else 1

    res = {}
    if a.cmd == "lone":
        for name in ("prose.txt", "code.txt", "long.txt"):
            res[name] = complete(a.url, text(name), a.max_tokens, name)
    elif a.cmd == "long":
        shorts = [(text("prose.txt"), a.max_tokens, "prose"), (text("code.txt"), a.max_tokens, "code")]
        both = together(a.url, [(long_text(a.long_tokens), a.max_tokens, "long")] + shorts)
        for r in both:
            res[r["user"] + ".beside"] = r
        for p in shorts:
            res[p[2] + ".alone"] = complete(a.url, *p)
        same = all(res[n + ".beside"].get("ids") == res[n + ".alone"].get("ids") for n in ("prose", "code"))
        print(f"POOL_LONG served={'error' not in res['long.beside']} short_ids_as_alone={same}")
    else:
        first = long_text(a.long_tokens)
        t1 = complete(a.url, first, a.max_tokens, "turn1")
        res["turn1"] = t1
        if not a.no_shorts:
            filler = long_text(a.short_tokens)
            for r in together(a.url, [(f"[{k}]\n" + filler, 16, f"short{k}") for k in range(3)]):
                res[r["user"]] = r
        res["turn2"] = complete(a.url, first + t1.get("text", "") + "\nContinue.", a.max_tokens, "turn2")
    for r in res.values():
        say(r)
    with open(a.out, "w") as f:
        json.dump(res, f)
    return 0 if all("error" not in r for r in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
