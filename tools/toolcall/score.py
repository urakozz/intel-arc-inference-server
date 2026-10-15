"""Score tool calls: the first tool call of each output against a reference run.

    score.py [--format qwen|hermes|atem] <out dir> <reference run> <candidate run>...

Reads <out dir>/<scenario>.<run>.txt (the generated text, special tokens kept) and compares
each candidate's first call - its name and raw argument strings - with the reference's.

Two call syntaxes, told apart by their tags (the two sets share none):
  Qwen XML (Qwen3.8, Agnes, Ornith): <tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n</parameter>
      ...</function>\\n</tool_call>; the first <tool_call> block wins.
  K2-Horizon (spec 18d; its template's tool_call_format, server/toolcall_k2.h): a block
      <ifm|tool_calls> ... </ifm|tool_calls> of <ifm|tool_call> ... </ifm|tool_call> calls in
        xml        NAME\\n<ifm|arg_key>K</ifm|arg_key>\\n<ifm|arg_value>V</ifm|arg_value>\\n ...
        xml_typed  the same with <ifm|arg_type>T</ifm|arg_type>\\n between key and value
        json       {"name": NAME, "arguments": {...}}  (arguments may be a JSON string of one)
      and REASONING first: K2's prompt ends in its think tag (<ifm|think>\\n, <ifm|think_fast>\\n or
      <ifm|think_faster>\\n by reasoning_effort), so the output opens with reasoning, closed by the
      matching close tag or, implicitly, by <ifm|tool_calls> (vLLM's k2_horizon reasoning parser,
      and ours). A call is read only after the reasoning; an output that ends inside it is
      'reasoning'. xml values are the raw text; json values are kept as strings when they are
      strings and as their JSON text otherwise, so two runs in one format compare like for like.

  hermes (Kolibri-1, spec 20e; `--format hermes`: its tags are Qwen's, so it is never detected):
      <tool_call>\n{"name": NAME, "arguments": {...}}\n</tool_call> - the body JSON ("arguments"
      may be a JSON string holding an object), the first block wins; reasoning the MODEL opens
      (optional whitespace, then <think> ... </think>) is skipped first, and an output that ends
      inside it is 'reasoning'. Arguments are kept as JSON values and compared as such.

  atem (Muse-Glimmer, spec 23a; `--format atem`): Harmony-style messages - the output continues the prompt's
      `<|start|>assistant` with ` to=<recipient><|message|>body<|eom|>` and more `<|start|>assistant ...` messages.
      to=self bodies are reasoning, to=user (or none) content; any other recipient is a tool channel whose first
      complete <atem:invoke name="NAME"> ... </atem:invoke> is the call (spec 23 §6.5's selection rule: an invoke
      echoed in reasoning or content is never one); NAME from the invoke, `ns.ns` collapsed to a registered `ns`;
      <atem:parameter name="K">V</atem:parameter> values JSON-decoded per value, else the raw string. A tool
      channel with no complete invoke is a parse failure ('incomplete'); every run's count is printed.

Kinds: 'call', 'no call', 'incomplete' (a call opened and not closed, or one that does not
parse), 'reasoning' (K2 / hermes: generation ended inside the reasoning).
"""
import json
import os
import re
import sys

CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FUNC = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.S)

# K2-Horizon (spec 18d): every tag one special token (250043/44, 250054-61), kept in the text.
K2_THINK_CLOSE = ("</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>")
K2_THINK_OPEN = ("<ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>")
K2_BLOCK = "<ifm|tool_calls>"
K2_CALL = re.compile(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", re.S)
K2_ARG = re.compile(r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*(?:<ifm\|arg_type>(.*?)</ifm\|arg_type>\s*)?"
                    r"<ifm\|arg_value>(.*?)</ifm\|arg_value>", re.S)


def _trim(v: str) -> str:
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def is_k2(text: str) -> bool:
    """K2-Horizon's output: any of its tags. Qwen XML has none of them."""
    return "<ifm|" in text or "</ifm|" in text


def k2_split_reasoning(text: str):
    """-> (reasoning, rest) or (text, None) when the text ends inside the reasoning. An open think
    tag the model repeats at the very start is part of the reasoning text here."""
    ends = [(text.find(t), t) for t in K2_THINK_CLOSE + (K2_BLOCK,)]
    ends = [(i, t) for i, t in ends if i >= 0]
    if not ends:
        return text, None
    i, t = min(ends)
    if t == K2_BLOCK:
        return text[:i], text[i:]
    return text[:i], text[i + len(t):]


def k2_parse_call(body: str):
    """One <ifm|tool_call> body -> (name, params) or None when it does not parse."""
    s = body.strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
        except ValueError:
            return None
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"]:
            return None
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return None
        if not isinstance(args, dict):
            return None
        return obj["name"], {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                             for k, v in args.items()}
    first, _, rest = body.lstrip("\n").partition("\n")
    name = first.strip()
    if not name or "<ifm|" in name:
        return None
    params = {k.strip(): v for k, _t, v in K2_ARG.findall(rest)}
    return name, params


def parse_first_call_k2(text: str):
    """-> (kind, name, params, n_calls) for K2-Horizon's output."""
    _reasoning, rest = k2_split_reasoning(text)
    if rest is None:
        return ("reasoning", None, {}, 0)
    calls = K2_CALL.findall(rest)
    if not calls:
        return ("incomplete" if "<ifm|tool_call>" in rest else "no call", None, {}, 0)
    got = k2_parse_call(calls[0])
    if got is None:
        return ("incomplete", None, {}, len(calls))
    return ("call", got[0], got[1], len(calls))


HERMES_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


def hermes_parse_call(body: str):
    """One hermes call body -> (name, arguments) or None (server::parse_json_call's rule)."""
    try:
        obj = json.loads(body.strip())
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"].strip() \
            or any(c.isspace() for c in obj["name"]):
        return None
    args = obj.get("arguments", {})
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    return obj["name"], args


def parse_first_call_hermes(text: str):
    """-> (kind, name, arguments, n_calls) for Kolibri-1's output (hermes JSON calls)."""
    rest = text.lstrip()
    if rest.startswith("<think>"):
        end = rest.find("</think>")
        if end < 0:
            return ("reasoning", None, {}, 0)
        rest = rest[end + len("</think>"):]
    calls = HERMES_CALL.findall(rest)
    if not calls:
        return ("incomplete" if "<tool_call>" in rest else "no call", None, {}, 0)
    got = hermes_parse_call(calls[0])
    if got is None:
        return ("incomplete", None, {}, len(calls))
    return ("call", got[0], got[1], len(calls))


ATEM_INVOKE = re.compile(r'<atem:invoke\b[^>]*?\bname="([^"]+)">(.*?)</atem:invoke>', re.S)
ATEM_PARAM = re.compile(r'<atem:parameter\b[^>]*?\bname="([^"]+)"[^>]*?>(.*?)</atem:parameter>', re.S)
ATEM_ENDS = ("<|eom|>", "<|eot|>")
_TOOL_NAMES = None


def registered_tools() -> set:
    """The A4 set's tool names (tools.json beside this script)."""
    global _TOOL_NAMES
    if _TOOL_NAMES is None:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools.json"), encoding="utf-8") as f:
            _TOOL_NAMES = {t.get("function", t)["name"] for t in json.load(f)}
    return _TOOL_NAMES


def atem_messages(text: str):
    """Muse-Glimmer's output as messages (spec 23 §6.5, vLLM's _iter_messages rule): a header is
    `[<|start|>assistant] [ to=<recipient>]<|message|>` (the first message has no <|start|>: the prompt ended
    with it); a body ends at <|eom|> / <|eot|>, or at a <|start|> (a header still being generated).
    -> [(recipient or None, body, terminated)]."""
    out, pos = [], 0
    while True:
        m = text.find("<|message|>", pos)
        if m < 0:
            return out
        header = text[pos:m].strip()
        if header.startswith("<|start|>"):
            header = header[len("<|start|>"):].strip()
        if header.startswith("assistant"):
            header = header[len("assistant"):].strip()
        rec = header[3:].strip() if header.startswith("to=") else None
        b0 = m + len("<|message|>")
        ends = [(text.find(t, b0), t) for t in ATEM_ENDS + ("<|start|>",)]
        ends = [(i, t) for i, t in ends if i >= 0]
        if not ends:
            out.append((rec, text[b0:], False))
            return out
        i, t = min(ends)
        out.append((rec, text[b0:i], t != "<|start|>"))
        pos = i + (len(t) if t in ATEM_ENDS else 0)


def atem_value(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def atem_name(name: str, tools) -> str:
    """A doubled `ns.ns` maps to a registered bare `ns` (the template's recipient is "ns.*"); others pass through."""
    tools = registered_tools() if tools is None else tools
    a, dot, b = name.partition(".")
    return a if dot and a == b and a in tools and name not in tools else name


def parse_first_call_atem(text: str, tools=None):
    """-> (kind, name, arguments, n_calls) for Muse-Glimmer's ATEM output: the first call is the first complete
    <atem:invoke> in a channel that is not `self` (reasoning) and not `user` / none (content) - an invoke echoed in
    reasoning or content is never a call; the name is the invoke's, never the recipient's; a tool channel with no
    complete invoke is a parse failure ('incomplete'); a generation that ends inside reasoning is 'reasoning'."""
    msgs = atem_messages(text)
    first, n_calls = None, 0
    for rec, body, _term in msgs:
        if rec is None or rec in ("self", "user"):
            continue
        invs = ATEM_INVOKE.findall(body)
        n_calls += len(invs)
        if first is None:
            if not invs:
                first = ("incomplete", None, {})
            else:
                name, inner = invs[0]
                first = ("call", atem_name(name, tools), {k: atem_value(v) for k, v in ATEM_PARAM.findall(inner)})
    if first is not None:
        return first + (n_calls,)
    if msgs and msgs[-1][0] == "self" and not msgs[-1][2]:
        return ("reasoning", None, {}, 0)
    return ("no call", None, {}, 0)


def parse_first_call(text: str, fmt: str = "auto", tools=None):
    """-> (kind, name, params, n_calls); kind is 'call', 'no call', 'incomplete' or (K2, hermes, atem)
    'reasoning'. fmt 'hermes' reads Kolibri-1's calls, 'atem' Muse-Glimmer's (`tools`: the registered names for
    the ns.ns collapse, default tools.json's); 'auto' (and 'qwen') K2's tags when present, else Qwen XML."""
    if fmt == "atem":
        return parse_first_call_atem(text, tools)
    if fmt == "hermes":
        return parse_first_call_hermes(text)
    if is_k2(text):
        return parse_first_call_k2(text)
    calls = CALL.findall(text)
    if not calls:
        return ("incomplete" if "<tool_call>" in text else "no call", None, {}, 0)
    f = FUNC.search(calls[0])
    if not f:
        return ("incomplete", None, {}, len(calls))
    params = {k.strip(): _trim(v) for k, v in PARAM.findall(f.group(2))}
    return ("call", f.group(1).strip(), params, len(calls))


def main() -> None:
    args = sys.argv[1:]
    fmt = "auto"
    if args[:1] == ["--format"]:
        if len(args) < 2 or args[1] not in ("qwen", "hermes", "atem"):
            sys.exit("--format expects qwen, hermes or atem")
        fmt, args = args[1], args[2:]
    if len(args) < 3:
        sys.exit(__doc__)
    out, ref, cands = args[0], args[1], args[2:]
    names = sorted(n[: -len(f".{ref}.txt")] for n in os.listdir(out) if n.endswith(f".{ref}.txt"))
    matches = {c: 0 for c in cands}
    print(f"| scenario | {ref} | " + " | ".join(cands) + " |")
    print("|---|---|" + "---|" * len(cands))
    for n in names:
        r = parse_first_call(open(os.path.join(out, f"{n}.{ref}.txt"), encoding="utf-8").read(), fmt)
        cells = []
        for c in cands:
            p = os.path.join(out, f"{n}.{c}.txt")
            if not os.path.exists(p):
                cells.append("missing")
                continue
            got = parse_first_call(open(p, encoding="utf-8").read(), fmt)
            if got[0] != "call":
                cells.append(got[0])
            elif r[0] == "call" and got[1:3] == r[1:3]:
                cells.append("match")
                matches[c] += 1
            else:
                cells.append("mismatch")
        ref_cell = f"{r[1]}" if r[0] == "call" else r[0]
        print(f"| {n} | {ref_cell} | " + " | ".join(cells) + " |")
    print()
    for c in cands:
        print(f"{c}: {matches[c]} / {len(names)} match {ref}")
    if fmt == "hermes":   # KL4's hard bar: the parser reads every call the reference wrote
        bad = [n for n in names if parse_first_call(
            open(os.path.join(out, f"{n}.{ref}.txt"), encoding="utf-8").read(), fmt)[0] == "incomplete"]
        print(f"{ref}: {len(bad)} parse failure(s)" + (f": {', '.join(bad)}" if bad else ""))
    if fmt == "atem":     # gate Q (spec 23 §3.3): no more parse failures than the reference - every run's count
        for run in [ref] + cands:
            bad = [n for n in names if os.path.exists(os.path.join(out, f"{n}.{run}.txt")) and parse_first_call(
                open(os.path.join(out, f"{n}.{run}.txt"), encoding="utf-8").read(), fmt)[0] == "incomplete"]
            print(f"{run}: {len(bad)} parse failure(s)" + (f": {', '.join(bad)}" if bad else ""))


if __name__ == "__main__":
    main()
