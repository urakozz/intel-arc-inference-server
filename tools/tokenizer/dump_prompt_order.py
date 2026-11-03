#!/usr/bin/env python3
"""Reference ids for tests/server/prompt_order_test.cc: one agentic session's turns N and N+1 (Qwen3.8).

    dump_prompt_order.py <Qwen3.8 snapshot> [out.json]    (default tests/server/prompt_order_cases.json)
    (on the Mac: uv run --no-project --with transformers --with jinja2 python -I tools/tokenizer/dump_prompt_order.py <snap>)

Each case is a session at turn N: the messages as an OpenAI client sends them (a history whose assistant tool calls
carry `arguments` as a JSON STRING, keys in a non-alphabetical order), the text the model writes next (one or more
calls in the template's own XML form, parameters in the model's order), and the tool results the client appends for
turn N+1. The reference is transformers' apply_chat_template on the checkpoint's chat_template.jinja (the template
iterates `arguments | items`, so a string is parsed first - json.loads, insertion order kept, as vLLM does):
  want_n    the ids of turn N's prompt;
  want_n1   the ids of turn N+1's prompt: turn N's messages + the assistant turn the model wrote (its calls as vLLM's
            qwen3_coder parser reads them: parameters in output order, values typed by the tool's schema) + the
            tool results.
The script also checks the reference itself is session-stable: render(N+1) starts with render(N) + output +
"<|im_end|>" (a template that re-renders a turn differently from how it was written would fail here, not in C++).
The cases: nested objects with non-alphabetical keys, unicode (raw and \\u-escaped in the client's string), numbers
(integers, a float, a negative, a large integer), an empty arguments object (and an empty string), several calls in
one turn.
"""
import json
import os
import sys

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Key order as clients write tools: type, function{name, description, parameters{type, properties, required}} -
# not alphabetical at any level.
TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file.",
        "parameters": {"type": "object", "properties": {
            "filePath": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
            "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}, "include": {"type": "string"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "configure", "description": "Change build settings.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string"}, "settings": {"type": "object"}, "threshold": {"type": "number"},
            "retries": {"type": "integer"}, "seed": {"type": "integer"}, "dry_run": {"type": "boolean"}},
            "required": ["target"]}}},
    {"type": "function", "function": {"name": "list_todos", "description": "List open TODOs.",
        "parameters": {"type": "object", "properties": {}}}},
]

SYSTEM = "You are a coding agent. The repository is at /work. Make tool calls to inspect it."


def call(i, name, args_text):
    """An assistant turn as an OpenAI client echoes it: arguments a JSON string."""
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args_text}}


def assistant(calls):
    return {"role": "assistant", "content": None, "tool_calls": calls}


def tool(i, text):
    return {"role": "tool", "tool_call_id": f"call_{i}", "content": text}


def xml_call(name, args):
    """One call as the model writes it (the template's own rendering of a call)."""
    s = f"<tool_call>\n<function={name}>\n"
    for k, v in args.items():
        s += f"<parameter={k}>\n" + (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) + "\n</parameter>\n"
    return s + "</function>\n</tool_call>"


def output_of(calls):
    return "\n".join(xml_call(n, a) for n, a in calls)


CASES = [
    {"name": "nested",
     "history": [("grep", '{"pattern": "TODO", "path": "/work/src", "include": "*.cc"}',
                  "Found 1 match\n/work/src/a.cc:3: // TODO")],
     "user": "Turn on the fast path for the server target.",
     "next": [("configure", {"target": "server", "settings": {"zeta": {"y": 1, "b": [3, 1, 2]}, "alpha": "on",
                                                              "mid": {"q": None, "c": True}}, "dry_run": False})],
     "result": "ok"},
    {"name": "unicode",
     "history": [("grep", '{"pattern": "Gr\\u00f6\\u00dfe", "path": "/work/docs"}',
                  "Found 1 match\n/work/docs/de.md:1: Größe"),
                 ("grep", '{"pattern": "日本語|✓", "path": "/work/docs"}', "No matches")],
     "user": "Mark the naïve note — ünïcödé ✓ 日本語.",
     "next": [("configure", {"target": "docs", "settings": {"note": "naïve — ünïcödé ✓ 日本語", "emoji": "🚀"}})],
     "result": "done ✓"},
    {"name": "numbers",
     "history": [("read", '{"filePath": "/work/src/b.cc", "offset": 120, "limit": 40}', "<file>\n00120| int b;\n</file>")],
     "user": "Tune the retry policy.",
     "next": [("configure", {"target": "net", "threshold": 0.25, "retries": -3, "seed": 12345678901234567,
                             "settings": {"timeout_s": 1.5, "backoff": [1, 2.5, 100]}})],
     "result": "ok"},
    {"name": "empty",
     "history": [("list_todos", "{}", "none"), ("list_todos", "", "none")],
     "user": "List the TODOs again.",
     "next": [("list_todos", {})],
     "result": "none"},
    {"name": "multi",
     "history": [("read", '{"filePath": "/work/a.cc", "offset": 1, "limit": 5}', "<file>\n00001| a\n</file>"),
                 ("read", '{"filePath": "/work/b.cc", "limit": 5, "offset": 9}', "<file>\n00009| b\n</file>")],
     "user": "Read c.cc and d.cc too, both from line 3.",
     "next": [("read", {"filePath": "/work/c.cc", "offset": 3, "limit": 2}),
              ("grep", {"pattern": "main", "path": "/work", "include": "d.cc"})],
     "result": "<file>\n00003| c\n</file>"},
]


def turn_n(case):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Look around first."}]
    if len(case["history"]) > 1 and case["name"] == "multi":   # several calls in ONE assistant turn
        msgs.append(assistant([call(i, n, a) for i, (n, a, _) in enumerate(case["history"])]))
        msgs += [tool(i, r) for i, (_, _, r) in enumerate(case["history"])]
    else:
        for i, (n, a, r) in enumerate(case["history"]):
            msgs += [assistant([call(i, n, a)]), tool(i, r)]
    msgs.append({"role": "user", "content": case["user"]})
    return msgs


def for_template(msgs):
    """vLLM's (and the server's) reading of a client's messages: a string `arguments` parsed, order kept."""
    out = json.loads(json.dumps(msgs))
    for m in out:
        for c in m.get("tool_calls") or []:
            a = c["function"]["arguments"]
            c["function"]["arguments"] = json.loads(a) if a else {}
    return out


def main():
    snap = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "tests", "server", "prompt_order_cases.json")
    tok = AutoTokenizer.from_pretrained(snap)

    def render(msgs, tokenize):
        r = tok.apply_chat_template(for_template(msgs), tools=TOOLS, add_generation_prompt=True,
                                    enable_thinking=False, tokenize=tokenize)
        return [int(x) for x in (r["input_ids"] if hasattr(r, "keys") else r)] if tokenize else r

    cases = []
    for c in CASES:
        msgs = turn_n(c)
        text = output_of(c["next"])
        n1 = msgs + [assistant([call(100 + i, n, json.dumps(a, ensure_ascii=False)) for i, (n, a) in enumerate(c["next"])])]
        n1 += [tool(100 + i, c["result"]) for i in range(len(c["next"]))]
        pn, pn1 = render(msgs, False), render(n1, False)
        if not pn1.startswith(pn + text + "<|im_end|>"):
            sys.exit(f"FATAL: {c['name']}: the reference itself does not continue turn N's prompt + output")
        cases.append({"name": c["name"], "messages": msgs, "output": text,
                      "results": [c["result"]] * len(c["next"]),
                      "want_n": render(msgs, True), "want_n1": render(n1, True)})
        print(f"{c['name']}: turn N {len(cases[-1]['want_n'])} ids, turn N+1 {len(cases[-1]['want_n1'])} ids")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"snapshot": os.path.basename(os.path.normpath(snap)), "tools": TOOLS, "cases": cases}, f,
                  ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
