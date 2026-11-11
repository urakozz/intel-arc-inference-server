#!/usr/bin/env python3
"""The card gates of b70-serve --parallel replica (plan 13b Task 5, spec 13 §11.3), stdlib only.

  replica_check.py run --url URL --mode pairs|alone|b3|b4 --out FILE [--max-tokens 128] [--timeout 1800]
  replica_check.py compare --replica FILE --server-log DIR --card0 FILE --card1 FILE
  replica_check.py b3 --replica FILE --server-log DIR --card0 FILE --card1 FILE
  replica_check.py b4 --replica FILE [--wall 120]

Every request carries a key in the OpenAI `user` field; the server's --log-requests record keeps the request body,
and under --parallel replica the `card` and `slot` that served it. One JSON line per request in --out: key, mode, ids,
text, finish_reason, t0 / t1 (client wall clock), the error if any.

The requests (the golden prompts, tests/golden/prompts/{prose,code,cjk}.txt, as /v1/completions and as one chat
message, greedy and seeded sampled - temperature 0.7, seed 11 -, ignore_eos, --max-tokens ids):
  pairs   R1 / B2: the requests two at a time, sent together (both cards busy at once);
  alone   the same requests one at a time - run against `--parallel 1` on ONE card (ZE_AFFINITY_MASK=0 and =1);
  b3      the cross-card restore, in order: A (a long chat turn) alone; X (A's first message, a new second one -
          its best match is A's card) started; while X decodes, A' (A's conversation continued) - on replica it
          lands on the other card and restores A's request-end snapshot from the one host store; on one card
          (`--parallel 1`) the same order restores the same snapshot there;
  b4      a streamed request the client drops after 8 chunks while a neighbour decodes (the neighbour's ids must
          not change, and two requests right after must both start at once); a stop string ending one row of a
          pair; a stream beside a plain request (the streamed text equals the same request's non-streamed text).

`compare` (R1 / B2): requests alone on their card must equal that card's serial run bitwise. Requests overlapping
on different slots of the same card may share batched steps: their first differences go through serve_judge.py's
f32 judge (missing references SKIP, exit 77). Also checks that pairs really overlapped and both cards were used.
`b3` checks A' was
restored on the other card (cached_tokens > 0, slot != A's) and equals A' on that card alone. `b4` checks the
dropped stream's neighbour finished, the two requests after it both answered within --wall seconds, the stop row
ended with "stop" and its neighbour ran to its length, and the streamed text equals the non-streamed one. Exit 1 on
any difference.
"""
import argparse
import glob
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN = ("prose", "code", "cjk")
SEEDED = {"temperature": 0.7, "seed": 11}


def text_of(name):
    with open(os.path.join(ROOT, "tests", "golden", "prompts", name + ".txt"), encoding="utf-8") as f:
        return f.read()


def requests_for(max_tokens):
    """[(key, endpoint, body)] - the golden prompts x {completion, chat} x {greedy, seeded}."""
    out = []
    for n in GOLDEN:
        t = text_of(n)
        for kind in ("cmpl", "chat"):
            for samp in ("greedy", "seeded"):
                key = f"{n}.{kind}.{samp}"
                body = {"max_tokens": max_tokens, "ignore_eos": True, "return_token_ids": True, "user": key}
                body.update(SEEDED if samp == "seeded" else {"temperature": 0})
                if kind == "cmpl":
                    body["prompt"] = t
                    out.append((key, "/v1/completions", body))
                else:
                    body["messages"] = [{"role": "user", "content": t}]
                    body["chat_template_kwargs"] = {"enable_thinking": False}
                    out.append((key, "/v1/chat/completions", body))
    return out


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def one(base, key, ep, body, timeout, mode):
    rec = {"key": key, "mode": mode, "t0": time.time()}
    try:
        j = post(base + ep, body, timeout)
        c = j["choices"][0]
        rec["ids"] = c.get("token_ids", [])
        rec["prompt_ids"] = j.get("prompt_token_ids")
        rec["text"] = c["message"]["content"] if "message" in c else c.get("text", "")
        rec["finish_reason"] = c.get("finish_reason")
        rec["cached"] = j.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens")
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        rec["error"] = str(e)
    rec["t1"] = time.time()
    return rec


def together(base, items, timeout, mode):
    out = [None] * len(items)

    def go(i):
        k, ep, b = items[i]
        out[i] = one(base, k, ep, b, timeout, mode)

    ts = [threading.Thread(target=go, args=(i,)) for i in range(len(items))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def stream_drop(base, body, chunks, timeout):
    """A streamed request whose client hangs up after `chunks` reads."""
    host, port = base.replace("http://", "").split(":")
    data = json.dumps(dict(body, stream=True)).encode()
    s = socket.create_connection((host, int(port)), timeout=timeout)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              + f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    got = 0
    while got < chunks:
        if not s.recv(4096):
            break
        got += 1
    s.close()


def stream_text(base, body, timeout):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    text = ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r.read().decode("utf-8").split("\n\n"):
            if not raw.startswith("data: ") or raw == "data: [DONE]":
                continue
            j = json.loads(raw[6:])
            if j.get("choices"):
                d = j["choices"][0].get("delta", {})
                if isinstance(d.get("content"), str):
                    text += d["content"]
    return text


def chat(key, messages, max_tokens, **extra):
    b = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
         "return_token_ids": True, "user": key, "chat_template_kwargs": {"enable_thinking": False}}
    b.update(extra)
    return (key, "/v1/chat/completions", b)


def run(a):
    base = a.url.rstrip("/")
    reqs = requests_for(a.max_tokens)
    recs = []
    if a.mode == "alone":
        for k, ep, b in reqs:
            recs.append(one(base, k, ep, b, a.timeout, a.mode))
    elif a.mode == "pairs":
        # each request beside another prompt's (both cards busy); every request once
        for i in range(0, len(reqs), 2):
            recs += together(base, reqs[i:i + 2], a.timeout, a.mode)
    elif a.mode == "b3":
        first = text_of("prose")
        conv = [{"role": "user", "content": first}]
        A = chat("b3.A", conv, 64)
        recs.append(one(base, *A, a.timeout, a.mode))
        x_msgs = [{"role": "user", "content": first[: len(first) // 2]}]
        X = chat("b3.X", x_msgs, 400)
        conv2 = conv + [{"role": "assistant", "content": recs[0].get("text", "")},
                        {"role": "user", "content": "Now summarise that in one sentence."}]
        A2 = chat("b3.A2", conv2, 64)
        holder = {}
        t = threading.Thread(target=lambda: holder.update(x=one(base, *X, a.timeout, a.mode)))
        t.start()
        time.sleep(a.hold)   # X is granted and decoding
        recs.append(one(base, *A2, a.timeout, a.mode))
        t.join()
        recs.append(holder["x"])
    elif a.mode == "b4":
        nb = chat("b4.neighbour", [{"role": "user", "content": text_of("code")}], a.max_tokens)
        th = {}
        t = threading.Thread(target=lambda: th.update(n=one(base, *nb, a.timeout, a.mode)))
        t.start()
        stream_drop(base, chat("b4.dropped", [{"role": "user", "content": text_of("prose")}], 20000)[2], 8,
                    a.timeout)
        t.join()
        recs.append(th["n"])
        t0 = time.time()
        after = together(base, [chat("b4.after0", [{"role": "user", "content": "Say hello."}], 32),
                                chat("b4.after1", [{"role": "user", "content": "Say goodbye."}], 32)],
                         a.timeout, a.mode)
        recs += after
        recs.append({"key": "b4.after.wall", "mode": a.mode, "wall": time.time() - t0})
        # a stop string ends one row; the neighbour runs on. The stop string is a piece of the request's own greedy
        # text (from a first run of it), so the row surely meets it.
        probe = one(base, *chat("b4.stop_probe", [{"role": "user", "content": text_of("cjk")}], a.max_tokens),
                    a.timeout, a.mode)
        recs.append(probe)
        ptext = probe.get("text") or ""
        stop_at = ptext[len(ptext) // 2: len(ptext) // 2 + 4] or "\n"
        stop_req = chat("b4.stopped", [{"role": "user", "content": text_of("cjk")}], a.max_tokens, stop=[stop_at])
        recs += together(base, [stop_req, chat("b4.stop_neighbour", [{"role": "user", "content": text_of("prose")}],
                                               a.max_tokens)], a.timeout, a.mode)
        # streaming order: a stream beside a plain request; its text equals the same request non-streamed
        sb = chat("b4.stream", [{"role": "user", "content": text_of("code")}], 64)[2]
        holder = {}
        t = threading.Thread(target=lambda: holder.update(s=stream_text(base, sb, a.timeout)))
        t.start()
        recs.append(one(base, *chat("b4.stream_neighbour", [{"role": "user", "content": text_of("cjk")}], 64),
                        a.timeout, a.mode))
        t.join()
        plain = one(base, "b4.stream_plain", "/v1/chat/completions", dict(sb, user="b4.stream_plain"), a.timeout,
                    a.mode)
        recs.append(plain)
        recs.append({"key": "b4.stream_text", "mode": a.mode, "text": holder.get("s", ""),
                     "equal": holder.get("s", "") == plain.get("text")})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    errs = [r for r in recs if "error" in r]
    for r in recs:
        print(f"{r['mode']} {r['key']}: {len(r.get('ids', []))} ids {r.get('finish_reason', '')} "
              f"{r.get('error', '')}".rstrip())
    return 1 if errs else 0


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["key"]: r for r in (json.loads(x) for x in f if x.strip())}


def server_records(log_dir):
    """key -> the server's --log-requests record (card, slot, timings and exact prompt ids)."""
    out = {}
    for p in glob.glob(os.path.join(log_dir, "*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        req = r.get("request")
        if isinstance(req, dict) and "user" in req:
            out[req["user"]] = r
    return out


def slots(log_dir):
    """key -> (card, t_start, t_end); legacy logs used slot as the card number."""
    return {key: (r.get("card", r.get("slot", -1)), r.get("t_start"), r.get("t_end"))
            for key, r in server_records(log_dir).items()}


def shared_card(key, records):
    """Different slots on one card, overlapping in the server. Legacy logs have no per-card batching."""
    r = records[key]
    if "card" not in r or "slot" not in r:
        return False
    return any(k != key and q.get("card") == r["card"] and "slot" in q and q["slot"] != r["slot"]
               and all(x.get(t) is not None for x in (r, q) for t in ("t_start", "t_end"))
               and r["t_start"] < q["t_end"] and q["t_start"] < r["t_end"]
               for k, q in records.items())


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def compare(a):
    rep = load(a.replica)
    cards = [load(a.card0), load(a.card1)]
    where = slots(a.server_log)
    logs = server_records(a.server_log)
    alone_logs = [server_records(os.path.join(os.path.dirname(path), "server-log")) for path in (a.card0, a.card1)]
    import serve_judge as sj
    reviewed, cache, awaiting = sj.reviewed_rows(a.header), {}, set()
    bad = skipped = 0
    used = set()
    expected = set(cards[0]) | set(cards[1])
    for key in sorted(expected | set(rep)):
        r = rep.get(key, {})
        if "ids" not in r:
            print(f"R1 {key}: missing reply or request error: FAIL")
            bad += 1
            continue
        slot = where.get(key, (-1,))[0]
        if slot not in (0, 1):
            print(f"R1 {key}: no slot in the server's record")
            bad += 1
            continue
        used.add(slot)
        ref = cards[slot].get(key)
        if ref is None or "ids" not in ref:
            print(f"R1 {key}: no alone run on card {slot}")
            bad += 1
            continue
        d = first_diff(r["ids"], ref["ids"])
        if d < 0:
            print(f"R1 {key} card {slot} IDENTICAL ({len(r['ids'])} ids)")
        elif shared_card(key, logs):
            plain = dict(set="replica", name=key, out_ids=ref["ids"],
                         prompt_ids=ref.get("prompt_ids") or alone_logs[slot].get(key, {}).get("prompt_ids"))
            batched = dict(set="replica", name=key, out_ids=r["ids"], prompt_ids=logs[key].get("prompt_ids"))
            verdict, text, keys = sj.judge_pair(plain, batched, a, cache, reviewed)
            awaiting.update(keys)
            bad += verdict == sj.Verdict.FAIL
            skipped += verdict == sj.Verdict.SKIP
            print(f"R1 {key} card {slot} JUDGED (shared card): {verdict}: {text}")
        else:
            print(f"R1 {key} card {slot} DIFFERS at id {d} of {len(r['ids'])} (alone on card: bitwise required)")
            bad += 1
    if used != {0, 1}:
        print(f"R1: only card(s) {sorted(used)} served - not two at once")
        bad += 1
    # overlap: consecutive pairs (the order requests_for gives) ran at the same time
    keys = [k for k, _, _ in requests_for(1)]
    overlapped = 0
    for i in range(0, len(keys) - 1, 2):
        p, q = where.get(keys[i]), where.get(keys[i + 1])
        if p and q and p[1] is not None and q[1] is not None and p[1] < q[2] and q[1] < p[2]:
            overlapped += 1
    print(f"R1: {overlapped}/{len(keys) // 2} pairs overlapped in the server")
    if overlapped == 0:
        bad += 1
    for key in sorted(awaiting):
        print(f"    awaiting f32 reference: {key}")
    if skipped:
        print(f"SKIP_REASON replica_check: {skipped} differing request(s) not judged; "
              f"{len(awaiting)} awaiting f32 reference (contexts dumped to {a.dump})")
    print(f"R1 FAIL ({bad})" if bad else (f"R1 SKIP ({skipped})" if skipped else "R1 PASS"))
    return 1 if bad else (77 if skipped else 0)


def b3(a):
    rep = load(a.replica)
    where = slots(a.server_log)
    cards = [load(a.card0), load(a.card1)]
    bad = 0
    sa, sa2 = where.get("b3.A", (-1,))[0], where.get("b3.A2", (-1,))[0]
    cached = rep.get("b3.A2", {}).get("cached")
    print(f"B3: A on card {sa}, A' on card {sa2}, A' cached_tokens {cached}")
    if sa == sa2 or sa not in (0, 1) or sa2 not in (0, 1):
        print("B3: A' did not move to the other card")
        bad += 1
    if not cached:
        print("B3: A' reused nothing from the host store")
        bad += 1
    if sa2 in (0, 1):
        ref = cards[sa2].get("b3.A2")
        got = rep.get("b3.A2", {}).get("ids")
        if not ref or got is None or got != ref.get("ids"):
            print(f"B3: A' differs from the same order on card {sa2} alone "
                  f"(first at {first_diff(got or [], (ref or {}).get('ids', []))})")
            bad += 1
        else:
            print(f"B3: A' as a prefix cache hit on card {sa2} IDENTICAL to card {sa2} alone ({len(got)} ids)")
    print("B3 PASS" if bad == 0 else f"B3 FAIL ({bad})")
    return 1 if bad else 0


def b4(a):
    rep = load(a.replica)
    bad = 0

    def ok(cond, what):
        nonlocal bad
        print(("B4 ok: " if cond else "B4 FAIL: ") + what)
        bad += 0 if cond else 1

    ok("ids" in rep.get("b4.neighbour", {}), "the dropped stream's neighbour finished")
    ok(all("ids" in rep.get(k, {}) for k in ("b4.after0", "b4.after1")), "two requests after the drop answered")
    ok(rep.get("b4.after.wall", {}).get("wall", 1e9) < a.wall,
       f"they took {rep.get('b4.after.wall', {}).get('wall', -1):.1f} s (< {a.wall} s: the dropped request no "
       "longer holds a card)")
    ok(rep.get("b4.stopped", {}).get("finish_reason") == "stop", "the stop string ended its row")
    ok(rep.get("b4.stop_neighbour", {}).get("finish_reason") == "length", "its neighbour ran to max_tokens")
    ok(rep.get("b4.stream_text", {}).get("equal") is True, "the stream beside a request: text equal to non-streamed")
    print("B4 PASS" if bad == 0 else f"B4 FAIL ({bad})")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--mode", choices=("pairs", "alone", "b3", "b4"), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--timeout", type=float, default=1800)
    r.add_argument("--hold", type=float, default=3.0, help="b3: seconds between X's start and A'")
    for name in ("compare", "b3"):
        c = sub.add_parser(name)
        c.add_argument("--replica", required=True)
        c.add_argument("--server-log", required=True)
        c.add_argument("--card0", required=True)
        c.add_argument("--card1", required=True)
        if name == "compare":
            c.add_argument("--dump", default=os.path.join(ROOT, "oracle-forced-dumps", "qwen-serve"))
            c.add_argument("--judge", default=os.path.join(ROOT, "oracle-out-qwen-f32-i8head-forced"))
            c.add_argument("--bf16", default=os.path.join(ROOT, "oracle-out-qwen-bf16top-forced"))
            c.add_argument("--tag", default="serve_int8")
            c.add_argument("--header", default=os.path.join(ROOT, "tests", "server", "spec_judge.h"))
            c.add_argument("--vocab-used", type=int, default=248077)
            c.add_argument("--root", default=ROOT)
    c = sub.add_parser("b4")
    c.add_argument("--replica", required=True)
    c.add_argument("--wall", type=float, default=120)
    a = p.parse_args()
    if a.cmd == "run":
        return run(a)
    return {"compare": compare, "b3": b3, "b4": b4}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
