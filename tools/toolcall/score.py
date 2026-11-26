"""Score tool calls: the first tool call of each output against a reference run.

    score.py [--format qwen|hermes|inkling] [--min N] <out dir> <reference run> <candidate run>...

Reads <out dir>/<scenario>.<run>.txt (the generated text, special tokens kept) and compares
each candidate's first call - its name and raw argument strings - with the reference's.

Validity (open-work 3o): every first call is also checked against the scenario's
files - tests/golden/toolcall/tree.json, the repository the 36 conversations were made from (its file
list and the full text of the six files they read; revision in the file, the one whose lines every
prompt's read results show; Ornith's, Agnes's and Flash-Next's sets re-render the same
conversations, and so does Inkling 2 Mini's). Not for `--format hermes`: Kolibri-1's German set was made from a later tree. A call is
'invalid' when it does not parse ('incomplete') or when its
arguments would not work there:
  edit   filePath one of the six files; oldString occurs in it exactly once (opencode's edit refuses
         none and several); newString differs from it
  read   filePath a file of the tree; offset / limit integers, offset within a known file's lines
  grep   pattern compiles; path a file or directory of the tree
  glob   path a directory of the tree
  A search that finds nothing is an answer, not an error (the tools print "No files found"): the
  reference itself globs tests/**/*linear* in t5_test-linear_l0, which matched no file then.
  bash / write and any other tool: parsing only. A path may be relative (to /work).
A valid alternative (a narrower grep, a broader glob, other wording) stays valid; it is a mismatch
against the reference, not a failure. An invalid call never counts as a match. --min N (A4's bar):
exit 1 when a candidate has fewer than N matches or any invalid call.

Call syntaxes:
  Qwen XML (Qwen3.8, Ornith): <tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n</parameter>
      ...</function>\\n</tool_call>; the first <tool_call> block wins.
  hermes (Kolibri-1, spec 20e; `--format hermes`: its tags are Qwen's, so it is never detected):
      <tool_call>\n{"name": NAME, "arguments": {...}}\n</tool_call> - the body JSON ("arguments"
      may be a JSON string holding an object), the first block wins; reasoning the MODEL opens
      (optional whitespace, then <think> ... </think>) is skipped first, and an output that ends
      inside it is 'reasoning'. Arguments are kept as JSON values and compared as such.
  Inkling 2 Mini (`--format inkling`): blocks opened by <|message_model|> and closed by <|end_message|>;
      a call is NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}, the first such block wins.
      The JSON's name must be the block's NAME and "args" an object; values are compared as JSON values.
      A <|content_thinking|> block is skipped, and an output that ends inside one is 'reasoning'.

Kinds: 'call', 'no call', 'incomplete' (a call opened and not closed, or one that does not
parse), 'reasoning' (hermes, inkling: generation ended inside the reasoning).
"""
import json
import os
import re
import sys

TREE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "golden", "toolcall",
                    "tree.json")
WORK = "/work"

CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FUNC = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.S)

def _trim(v: str) -> str:
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


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


INK_HEADER = "<|message_model|>"
INK_THINK = "<|content_thinking|>"
INK_JSON = "<|content_invoke_tool_json|>"
INK_END = "<|end_message|>"


def inkling_parse_call(name: str, body: str):
    """One Inkling call block -> (name, args) or None: the JSON names the block's tool and "args" is an object."""
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    name = name.strip()
    if not isinstance(obj, dict) or obj.get("name") != name or not name or any(c.isspace() for c in name) \
            or not isinstance(obj.get("args"), dict):
        return None
    return name, obj["args"]


def parse_first_call_inkling(text: str):
    """-> (kind, name, arguments, n_calls) for Inkling 2 Mini's output. Text before the first <|message_model|>
    counts as a block too (a prompt that already opened the model's message)."""
    blocks = text.split(INK_HEADER)
    calls = [b for b in blocks if INK_JSON in b and not b.startswith(INK_THINK)]
    if not calls:
        last = blocks[-1]
        return ("reasoning" if last.startswith(INK_THINK) and INK_END not in last else "no call", None, {}, 0)
    name, rest = calls[0].split(INK_JSON, 1)
    end = rest.find(INK_END)
    got = inkling_parse_call(name, rest[:end]) if end >= 0 else None
    if got is None:
        return ("incomplete", None, {}, len(calls))
    return ("call", got[0], got[1], len(calls))


def parse_first_call(text: str, fmt: str = "auto"):
    """-> (kind, name, params, n_calls); kind is 'call', 'no call', 'incomplete' or (hermes)
    'reasoning'. fmt 'hermes' reads Kolibri-1's calls, 'inkling' Inkling 2 Mini's; 'auto' (and 'qwen') Qwen XML."""
    if fmt == "hermes":
        return parse_first_call_hermes(text)
    if fmt == "inkling":
        return parse_first_call_inkling(text)
    calls = CALL.findall(text)
    if not calls:
        return ("incomplete" if "<tool_call>" in text else "no call", None, {}, 0)
    f = FUNC.search(calls[0])
    if not f:
        return ("incomplete", None, {}, len(calls))
    params = {k.strip(): _trim(v) for k, v in PARAM.findall(f.group(2))}
    return ("call", f.group(1).strip(), params, len(calls))


def load_tree(path: str = TREE):
    """-> (files, dirs, text): the scenario tree's paths relative to /work and the six files' text."""
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    files = set(t["files"])
    dirs = {""} | {os.path.dirname(f) for f in files}
    for d in list(dirs):
        while d:
            d = os.path.dirname(d)
            dirs.add(d)
    return files, dirs, t["text"]


def _rel(p: str):
    """A tool path -> its path under /work ('' for /work itself), or None outside it."""
    p = os.path.normpath(p if p.startswith("/") else os.path.join(WORK, p))
    if p == WORK:
        return ""
    return p[len(WORK) + 1:] if p.startswith(WORK + "/") else None


def check_call(name: str, params: dict, tree) -> str:
    """'' when the call works against the scenario's files, else why not."""
    files, dirs, text = tree
    p = {k: v if isinstance(v, str) else json.dumps(v) for k, v in params.items()}
    if name == "edit":
        f = _rel(p.get("filePath", ""))
        if f not in text:
            return f"edit of {p.get('filePath')!r}, not a file the conversation read"
        old = p.get("oldString", "")
        n = text[f].count(old) if old else 0
        if n != 1 and not (n > 1 and p.get("replaceAll") == "true"):
            return f"edit's oldString occurs {n} times in {f}"
        if p.get("newString") == old:
            return "edit's newString equals its oldString"
        return ""
    if name == "read":
        f = _rel(p.get("filePath", ""))
        if f not in files:
            return f"read of {p.get('filePath')!r}, not a file of the tree"
        for k in ("offset", "limit"):
            if k in p and not re.fullmatch(r"\d+", p[k].strip()):
                return f"read's {k} {p[k]!r} is not an integer"
        if f in text and "offset" in p and int(p["offset"]) > text[f].count("\n"):
            return f"read's offset {p['offset']} is past the end of {f}"
        return ""
    if name == "grep":
        try:
            re.compile(p.get("pattern", ""))
        except re.error as e:
            return f"grep's pattern does not compile: {e}"
        root = _rel(p.get("path", WORK))
        if root is None or (root not in files and root not in dirs):
            return f"grep's path {p.get('path')!r} is not in the tree"
        return ""
    if name == "glob":
        root = _rel(p.get("path", WORK))
        if root is None or root not in dirs:
            return f"glob's path {p.get('path')!r} is not a directory of the tree"
        return ""
    return ""


def validity(got, tree) -> str:
    """'' for a valid first call (or none at all), else why it is invalid."""
    if got[0] == "incomplete":
        return "the call does not parse"
    if got[0] != "call":
        return ""
    return check_call(got[1], got[2], tree)


def main() -> None:
    args = sys.argv[1:]
    fmt = "auto"
    if args[:1] == ["--format"]:
        if len(args) < 2 or args[1] not in ("qwen", "hermes", "inkling"):
            sys.exit("--format expects qwen, hermes or inkling")
        fmt, args = args[1], args[2:]
    need = None
    if args[:1] == ["--min"]:
        need, args = int(args[1]), args[2:]
    if len(args) < 3:
        sys.exit(__doc__)
    out, ref, cands = args[0], args[1], args[2:]
    names = sorted(n[: -len(f".{ref}.txt")] for n in os.listdir(out) if n.endswith(f".{ref}.txt"))
    matches = {c: 0 for c in cands}
    invalid = {c: [] for c in cands}
    tree = load_tree() if fmt != "hermes" else None
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
            why = validity(got, tree) if tree else ""
            if why:
                cells.append("invalid")
                invalid[c].append(f"{n} ({why})")
            elif got[0] != "call":
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
    for c in cands:
        print(f"{c}: {len(invalid[c])} invalid call(s)" + (": " + "; ".join(invalid[c]) if invalid[c] else ""))
    if fmt in ("hermes", "inkling"):   # KL4's bar: every reference call parses
        bad = [n for n in names if parse_first_call(
            open(os.path.join(out, f"{n}.{ref}.txt"), encoding="utf-8").read(), fmt)[0] == "incomplete"]
        print(f"{ref}: {len(bad)} parse failure(s)" + (f": {', '.join(bad)}" if bad else ""))
    if need is not None:
        short = [c for c in cands if matches[c] < need or invalid[c]]
        print(f"bar: >= {need} / {len(names)} matches and no invalid call: "
              + (f"FAIL ({', '.join(short)})" if short else "PASS"))
        if short:
            sys.exit(1)


if __name__ == "__main__":
    main()
