"""Score tool calls: the first tool call of each output against a reference run.

    score.py <out dir> <reference run> <candidate run>...

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

Kinds: 'call', 'no call', 'incomplete' (a call opened and not closed, or one that does not
parse), 'reasoning' (K2: generation ended inside the reasoning).
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


def parse_first_call(text: str):
    """-> (kind, name, params, n_calls); kind is 'call', 'no call', 'incomplete' or (K2)
    'reasoning'."""
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
    out, ref, cands = sys.argv[1], sys.argv[2], sys.argv[3:]
    names = sorted(n[: -len(f".{ref}.txt")] for n in os.listdir(out) if n.endswith(f".{ref}.txt"))
    matches = {c: 0 for c in cands}
    print(f"| scenario | {ref} | " + " | ".join(cands) + " |")
    print("|---|---|" + "---|" * len(cands))
    for n in names:
        r = parse_first_call(open(os.path.join(out, f"{n}.{ref}.txt"), encoding="utf-8").read())
        cells = []
        for c in cands:
            p = os.path.join(out, f"{n}.{c}.txt")
            if not os.path.exists(p):
                cells.append("missing")
                continue
            got = parse_first_call(open(p, encoding="utf-8").read())
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


if __name__ == "__main__":
    main()
