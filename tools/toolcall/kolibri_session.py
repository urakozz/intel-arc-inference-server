#!/usr/bin/env python3
"""A live ~20-turn agentic tool session against b70-serve (Kolibri-1), with real tools on a sandbox project.

    kolibri_session.py <base url> <sandbox dir> <out.json> [--turns 20] [--effort medium] [--max-tokens 8192]

The client echoes every response back as an OpenAI client does (content, reasoning_content, tool_calls as returned),
executes the calls on the sandbox (list_dir, read_file, write_file, run_tests, search), appends the results, and sends
the next user task when the model answers without a call. Sampling: the card's T 1.0, top-p 0.97, top-k 128.
Per turn it records the finish reason, reasoning / content sizes, the calls, and a verdict:
  answer   - content, no call, finish stop
  call     - one or more parsed tool calls, finish tool_calls
  STALL    - reasoning but no content and no call (the community report's symptom)
  LEFTOVER - "<tool_call>" text left in content (a call the parser could not read)
  LENGTH   - finish length (the max_tokens budget hit)
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

TOOLS = [
    {"type": "function", "function": {"name": "list_dir", "description": "List the files of a directory of the project.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory, relative to the project root."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file of the project.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File, relative to the project root."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write (create or replace) a text file of the project with the full new content.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string", "description": "The complete file content."}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the project's unit tests (python3 -m unittest) and return the output.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "search", "description": "Search the project's files for a text (grep -rn).",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
]

PROJECT = {
    "calc.py": '''"""Kleine Rechenhilfen für Messreihen."""


def summe(werte):
    total = 0
    for w in werte[1:]:
        total += w
    return total


def maximum(werte):
    if not werte:
        raise ValueError("leere Liste")
    best = werte[0]
    for w in werte:
        if w < best:
            best = w
    return best


def prozent(teil, ganzes):
    return teil / ganzes * 10
''',
    "test_calc.py": '''import unittest

from calc import maximum, prozent, summe


class CalcTest(unittest.TestCase):
    def test_summe(self):
        self.assertEqual(summe([1, 2, 3]), 6)

    def test_maximum(self):
        self.assertEqual(maximum([3, 9, 4]), 9)

    def test_maximum_leer(self):
        with self.assertRaises(ValueError):
            maximum([])

    def test_prozent(self):
        self.assertAlmostEqual(prozent(1, 4), 25.0)


if __name__ == "__main__":
    unittest.main()
''',
    "README.md": "# calc\n\nKleine Rechenhilfen: `summe`, `maximum`, `prozent`.\n",
}

USERS = [
    "Im Projekt schlagen Tests fehl. Finde die Ursachen und behebe sie in calc.py, ohne die Tests zu ändern. "
    "Prüfe am Ende mit run_tests.",
    "Danke. Füge bitte eine Funktion `mittelwert(werte)` hinzu (ValueError bei leerer Liste), mit Tests in "
    "test_calc.py, und ergänze die README. Lass danach die Tests laufen.",
    "Gibt es im Projekt noch Stellen ohne Docstring? Ergänze sie kurz und prüfe die Tests erneut.",
    "Fasse bitte in drei Sätzen zusammen, was du in dieser Sitzung geändert hast.",
    "Schreibe eine kurze CHANGELOG.md mit den Änderungen dieser Sitzung.",
    "Danke, das war alles. Gibt es noch etwas, das du empfehlen würdest?",
]


def safe(root, path):
    p = os.path.realpath(os.path.join(root, path or "."))
    if not p.startswith(os.path.realpath(root)):
        raise ValueError("path outside the project")
    return p


def execute(root, name, args):
    try:
        if name == "list_dir":
            return "\n".join(sorted(os.listdir(safe(root, args.get("path", ".")))))
        if name == "read_file":
            with open(safe(root, args["path"]), encoding="utf-8") as f:
                return f.read()
        if name == "write_file":
            p = safe(root, args["path"])
            with open(p, "w", encoding="utf-8") as f:
                f.write(args["content"])
            return f"wrote {len(args['content'])} characters to {args['path']}"
        if name == "run_tests":
            r = subprocess.run([sys.executable, "-m", "unittest", "-v"], cwd=root, capture_output=True, text=True, timeout=60)
            return (r.stdout + r.stderr)[-4000:]
        if name == "search":
            r = subprocess.run(["grep", "-rn", "--", args["pattern"], "."], cwd=root, capture_output=True, text=True, timeout=30)
            return r.stdout[-4000:] or "(no match)"
        return f"error: unknown tool {name}"
    except Exception as e:  # the model sees the error, as a real agent would
        return f"error: {e}"


def stream(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    res = {"reasoning": "", "content": "", "tool_calls": [], "finish_reason": None, "usage": None, "error": None}
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            j = json.loads(line[6:])
            if "error" in j:
                res["error"] = j["error"]
                continue
            if j.get("usage"):
                res["usage"] = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {})
                res["reasoning"] += d.get("reasoning_content") or ""
                res["content"] += d.get("content") or ""
                for tc in d.get("tool_calls") or []:
                    res["tool_calls"].append({"id": tc["id"], "name": tc["function"]["name"],
                                              "arguments": tc["function"].get("arguments") or "{}"})
                if ch.get("finish_reason"):
                    res["finish_reason"] = ch["finish_reason"]
    res["seconds"] = time.time() - t0
    return res


def verdict(res):
    if res["finish_reason"] == "length":
        return "LENGTH"
    if "<tool_call>" in res["content"]:
        return "LEFTOVER"
    if res["tool_calls"]:
        return "call"
    if res["content"].strip():
        return "answer"
    return "STALL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("sandbox")
    ap.add_argument("out")
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()
    os.makedirs(a.sandbox, exist_ok=True)
    for name, text in PROJECT.items():
        with open(os.path.join(a.sandbox, name), "w", encoding="utf-8") as f:
            f.write(text)
    url = a.url.rstrip("/") + "/v1/chat/completions"
    messages = [{"role": "user", "content": USERS[0]}]
    next_user = 1
    out = []
    for turn in range(1, a.turns + 1):
        body = {"model": "b70", "messages": messages, "tools": TOOLS, "stream": True,
                "stream_options": {"include_usage": True}, "temperature": 1.0, "top_p": 0.97, "top_k": 128,
                "seed": a.seed + turn, "max_tokens": a.max_tokens,
                "chat_template_kwargs": {"reasoning_effort": a.effort}}
        res = stream(url, body)
        v = verdict(res)
        u = res["usage"] or {}
        rec = {"turn": turn, "verdict": v, "finish": res["finish_reason"], "prompt_tokens": u.get("prompt_tokens"),
               "completion_tokens": u.get("completion_tokens"),
               "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "reasoning_chars": len(res["reasoning"]), "content": res["content"], "reasoning": res["reasoning"],
               "calls": res["tool_calls"], "seconds": round(res["seconds"], 1), "error": res["error"]}
        out.append(rec)
        names = ",".join(c["name"] for c in res["tool_calls"])
        print(f"turn {turn:2d}: {v:8s} finish {res['finish_reason']} prompt {u.get('prompt_tokens')} gen "
              f"{u.get('completion_tokens')} reasoning {len(res['reasoning'])} ch content {len(res['content'])} ch "
              f"calls [{names}] {res['seconds']:.0f} s", flush=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        if res["error"]:
            break
        msg = {"role": "assistant", "content": res["content"] or None}
        if res["reasoning"]:
            msg["reasoning_content"] = res["reasoning"]
        if res["tool_calls"]:
            msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"], "arguments": c["arguments"]}} for c in res["tool_calls"]]
        messages.append(msg)
        if res["tool_calls"]:
            for c in res["tool_calls"]:
                try:
                    args = json.loads(c["arguments"])
                except ValueError:
                    args = {}
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": execute(a.sandbox, c["name"], args)})
        elif v in ("answer", "STALL", "LENGTH", "LEFTOVER"):
            if next_user >= len(USERS):
                break
            messages.append({"role": "user", "content": USERS[next_user]})
            next_user += 1
    counts = {}
    for r in out:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("SUMMARY", json.dumps(counts), f"turns {len(out)}")


if __name__ == "__main__":
    main()
