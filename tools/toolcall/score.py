"""Score tool calls: the first tool call of each output against a reference run.

    score.py [--format qwen|hermes|atem|pythonic] <out dir> <reference run> <candidate run>...

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

  pythonic (LFM2.5-24B-A2B, spec 24; `--format pythonic`): the prompt ends in <think>, so the output OPENS
      inside reasoning, closed by </think> (an output that ends inside it is 'reasoning'; a block inside the
      reasoning is reasoning text). Then <|tool_call_start|>[name(k=v, ...), ...]<|tool_call_end|>: a list of
      calls with keyword arguments whose values are Python literals, read by a safe literal parser (never
      eval): strings in single, double or triple quotes, ints, floats, True / False / None and JSON's true /
      false / null, lists, tuples, dicts, nested. Escapes are the template's inverse per context
      (chat_template.jinja's format_arg_value): a TOP-LEVEL string decodes only a doubled backslash, its own
      quote's escape, n and r; a string inside a list / dict (written by tojson) decodes JSON's set; every
      other backslash is kept verbatim. A quote closes a string only when what follows (after whitespace) is
      , ) ] } : or the end of the block, so 'it's here' is one string; when the rest then fails to parse,
      the last quote that closed is kept in its string and the parse retried (once per quote: code such as
      'if __name__ == '__main__':' survives). A bare call without the brackets is one call; a lone positional
      dict is the arguments; any other positional argument, or a name without a value, makes the call
      'incomplete'. A block cut by the end of generation (no <|tool_call_end|>) counts the calls whose )
      closed; an unfinished first call is 'incomplete'. The first block's first call wins; values compare as
      JSON values (True == true, tuples == lists).

Kinds: 'call', 'no call', 'incomplete' (a call opened and not closed, or one that does not
parse), 'reasoning' (K2 / hermes / atem / pythonic: generation ended inside the reasoning).
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


LFM_THINK_CLOSE = "</think>"
LFM_START, LFM_END = "<|tool_call_start|>", "<|tool_call_end|>"
_BS = chr(92)                                              # a backslash
_TOP_ESC = {_BS: _BS, "n": "\n", "r": "\r"}                # + the string's own quote
_JSON_ESC = {'"': '"', _BS: _BS, "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_WS = " \t\r\n"


class _PyLit:
    """The pythonic call list's literal parser (spec 24 §4.6, decision 7) over a block's body."""

    CLOSERS = ",)]}:"
    WORDS = {"True": True, "true": True, "False": False, "false": False, "None": None, "null": None}

    def __init__(self, s: str):
        self.s, self.i = s, 0
        self.skip: set[int] = set()     # quotes kept in their string (quote recovery)
        self.closed: list[int] = []     # quotes that closed a string, in order
        self.done: list = []            # the calls that closed

    def peek(self) -> str:
        while self.i < len(self.s) and self.s[self.i] in _WS:
            self.i += 1
        return self.s[self.i] if self.i < len(self.s) else ""

    def eat(self, c: str) -> None:
        if self.peek() != c:
            raise ValueError(f"expected {c!r} at {self.i}")
        self.i += 1

    def closes(self, j: int, extra: str = "") -> bool:
        """A quote ending at j closes its string when the next non-space character is a closer or the end
        (`extra`: a quoted NAME also closes before ( or =)."""
        while j < len(self.s) and self.s[j] in _WS:
            j += 1
        return j >= len(self.s) or self.s[j] in self.CLOSERS + extra

    def string(self, nested: bool, extra: str = "") -> str:
        self.peek()
        q = self.s[self.i]
        triple = self.s.startswith(q * 3, self.i)
        self.i += 3 if triple else 1
        out = []
        while True:
            if self.i >= len(self.s):
                raise ValueError("unterminated string")
            c = self.s[self.i]
            if c == _BS and self.i + 1 < len(self.s):
                e = self.s[self.i + 1]
                if nested and e == "u" and self.i + 6 <= len(self.s):
                    out.append(chr(int(self.s[self.i + 2:self.i + 6], 16)))
                    self.i += 6
                    continue
                table = _JSON_ESC if nested else dict(_TOP_ESC, **{q: q})
                out.append(table[e] if e in table else c + e)   # anything else verbatim ("C:\Users", "\d")
                self.i += 2
                continue
            if triple and self.s.startswith(q * 3, self.i):
                self.i += 3
                return "".join(out)
            if not triple and c == q and self.i not in self.skip and self.closes(self.i + 1, extra):
                self.closed.append(self.i)
                self.i += 1
                return "".join(out)
            out.append(c)
            self.i += 1

    def name(self) -> str:
        if self.peek() in ("'", '"'):
            return self.string(False, "(=")
        j = self.i
        while self.i < len(self.s) and (self.s[self.i].isalnum() or self.s[self.i] in "_.-"):
            self.i += 1
        if j == self.i:
            raise ValueError(f"expected a name at {j}")
        return self.s[j:self.i]

    def value(self, nested: bool, depth: int):
        if depth > 64:
            raise ValueError("nested deeper than 64")
        c = self.peek()
        if c in ("'", '"'):
            return self.string(nested)
        if c in "[(":
            close = "]" if c == "[" else ")"
            self.i += 1
            out = []
            while self.peek() != close:
                out.append(self.value(True, depth + 1))
                if self.peek() == ",":
                    self.i += 1
                elif self.peek() != close:
                    raise ValueError("list separator")
            self.i += 1
            return out
        if c == "{":
            self.i += 1
            out = {}
            while self.peek() != "}":
                k = self.value(True, depth + 1)
                self.eat(":")
                out[k if isinstance(k, str) else json.dumps(k)] = self.value(True, depth + 1)
                if self.peek() == ",":
                    self.i += 1
                elif self.peek() != "}":
                    raise ValueError("dict separator")
            self.i += 1
            return out
        j = self.i
        while self.i < len(self.s) and (self.s[self.i].isalnum() or self.s[self.i] in "+-._"):
            self.i += 1
        tok = self.s[j:self.i]
        if tok in self.WORDS:
            return self.WORDS[tok]
        for conv in (int, float):
            try:
                return conv(tok)
            except ValueError:
                pass
        raise ValueError(f"not a literal: {tok!r}")

    def call(self):
        name = self.name()
        self.eat("(")
        args, positional = {}, []
        while self.peek() != ")":
            if self.peek() in ("'", '"', "[", "(", "{") or self.peek().isdigit() or self.peek() in "+-.":
                v = self.string(False, "=") if self.peek() in ("'", '"') else self.value(False, 1)
                if self.peek() == "=" and isinstance(v, str):     # a quoted argument name: "path"="x"
                    self.i += 1
                    args[v] = self.value(False, 1)
                else:
                    positional.append(v)
            else:
                k = self.name()
                if self.peek() != "=":
                    raise ValueError(f"{k}: a name without '='")
                self.i += 1
                if self.peek() in (",", ")", ""):
                    raise ValueError(f"{k}= without a value")
                args[k] = self.value(False, 1)
            if self.peek() == ",":
                self.i += 1
            elif self.peek() != ")":
                raise ValueError("argument separator")
        self.i += 1
        if positional:
            if len(positional) == 1 and isinstance(positional[0], dict) and not args:
                return name, positional[0]
            raise ValueError("a positional argument")
        return name, args

    def _calls(self) -> list:
        bracket = self.peek() == "["
        if bracket:
            self.i += 1
        while self.peek() not in ("]", ""):
            self.done.append(self.call())
            if self.peek() == ",":
                self.i += 1
        if bracket:
            self.eat("]")
        if self.peek() != "" or not self.done:
            raise ValueError("text after the call list, or no call")
        return self.done

    def calls(self, partial: bool = False) -> list:
        """The block's calls. Quote recovery (spec 24 §4.6): when a parse fails, a quote that closed a string in that
        attempt is kept IN the string instead and the parse restarts - a depth-first search over which quotes to
        keep, latest quote first, each quote kept at most once on a path and only quotes AFTER the last one kept
        tried next (so every set of kept quotes is tried at most once), bounded by MAX_ATTEMPTS parses (the block's
        length, at least 64). `partial` (generation ended inside the block): the calls whose ) closed, from the
        attempt that closed the most; ValueError when none did."""
        best: list = []
        budget = [max(64, len(self.s))]

        def attempt(skip: frozenset):
            self.i, self.skip, self.closed, self.done = 0, set(skip), [], []
            budget[0] -= 1
            try:
                return self._calls(), []
            except (ValueError, IndexError):
                return None, (list(self.done), list(self.closed))

        def search(skip: frozenset, floor: int):
            nonlocal best
            got, info = attempt(skip)
            if got is not None:
                return got
            done, closed = info
            if len(done) > len(best):
                best = done
            for q in reversed(closed):            # the latest closing quote first
                if budget[0] <= 0:
                    return None
                if q in skip or q <= floor:
                    continue
                r = search(skip | {q}, q)
                if r is not None:
                    return r
            return None
        got = search(frozenset(), -1)
        if got is not None:
            return got
        if partial and best:
            return best
        raise ValueError("the call list does not parse")


def parse_pythonic_block(body: str, partial: bool = False) -> list:
    """The calls of one <|tool_call_start|> ... <|tool_call_end|> body: [(name, arguments dict)]; ValueError when
    it does not parse. `partial`: the body was cut by the end of generation - the calls that closed."""
    return _PyLit(body).calls(partial)


def parse_first_call_pythonic(text: str):
    """-> (kind, name, arguments, n_calls) for LFM2.5's output (its prompt ends in <think>)."""
    end = text.find(LFM_THINK_CLOSE)
    if end < 0:
        return ("reasoning", None, {}, 0)
    rest = text[end + len(LFM_THINK_CLOSE):]
    a = rest.find(LFM_START)
    if a < 0:
        return ("no call", None, {}, 0)
    b = rest.find(LFM_END, a)
    try:   # no end tag: generation ended inside the block - the calls that closed count (spec 24 §4.6)
        calls = parse_pythonic_block(rest[a + len(LFM_START):] if b < 0 else rest[a + len(LFM_START):b], b < 0)
    except (ValueError, IndexError):
        return ("incomplete", None, {}, 0)
    name, args = calls[0]
    return ("call", name, args, len(calls))


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
    """-> (kind, name, params, n_calls); kind is 'call', 'no call', 'incomplete' or (K2, hermes, atem,
    pythonic) 'reasoning'. fmt 'hermes' reads Kolibri-1's calls, 'atem' Muse-Glimmer's (`tools`: the
    registered names for the ns.ns collapse, default tools.json's), 'pythonic' LFM2.5's; 'auto' (and
    'qwen') K2's tags when present, else Qwen XML."""
    if fmt == "atem":
        return parse_first_call_atem(text, tools)
    if fmt == "pythonic":
        return parse_first_call_pythonic(text)
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
        if len(args) < 2 or args[1] not in ("qwen", "hermes", "atem", "pythonic"):
            sys.exit("--format expects qwen, hermes, atem or pythonic")
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
    if fmt in ("hermes", "pythonic"):   # KL4's / spec 24 F6's bar: the parser reads every call the reference wrote
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
