#!/usr/bin/env python3
"""Build the tool-call acceptance set (spec 5 T0, gate A4).

    make_set.py <snapshot> <out dir>

36 multi-turn agentic-coding conversations, six templates times six real files
of this repository, each ending where the assistant's next turn is a tool call.
The checkpoint's own chat template renders them (tools from tools.json,
enable_thinking=False). Writes <name>.json (messages, tools, what the next call
is expected to be), <name>.ids (space-separated) and manifest.json (name, id
count, SHA-256 of the .ids file). A scenario outside 800 to 3000 ids is refused;
tool results are shortened until it fits.

The repository is read from this script's tree (/ws in the reference
container); the prompts call it /work. Run inside the reference container
(tools/oracle/run_in_container.sh) so the tokenizer is the pinned one.
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORK = "/work"
MIN_IDS, MAX_IDS = 800, 3000

FILES = [
    "src/runtime/prefill/linear_l0.cc",
    "src/runtime/prefill/step.cc",
    "src/loader/loader.cc",
    "src/server/openai.cc",
    "tools/rotate/rotation.py",
    "tools/box.sh",
]

SYSTEM = """You are a coding agent working in a local checkout of b70-inference-server, an LLM inference engine for the Intel Arc Pro B70 written in C++ with Level Zero kernels, plus Python and shell tooling. The repository is at /work.

You help the user with software engineering tasks: finding code, explaining it, editing it and running builds and tests. Use the tools you are given to inspect the repository; never guess the contents of a file you have not read. Prefer grep and glob to locate code, read to look at it, edit to change it (the oldString must match the file exactly, including indentation) and bash to run commands.

Guidelines:
- Be concise. Do not explain what you are about to do at length; act.
- Make one tool call at a time and wait for its result.
- Keep edits minimal and in the style of the surrounding code.
- Paths passed to tools are absolute and start with /work.
- The C++ tree builds with CMake into /work/build; tests are ctest targets named after their source file (for example tests/prefill/dequant_test.cc is dequant_test). The Python tools run with python3 and their tests with python3 -m pytest.
- Do not commit, push or change git state unless the user asks."""

FUNC_RE = {
    "cc": re.compile(r"^(?!\s)(?!return\b|if\b|namespace\b|using\b|static_assert\b)"
                     r"(?:[\w:<>,*&]+\s+)+[*&]?(\w+)\([^;]*$", re.M),
    "py": re.compile(r"^def (\w+)\(", re.M),
    "sh": re.compile(r"^(\w+)\(\) \{", re.M),
}
LOCAL_RE = {
    "cc": re.compile(r"^[ \t]+const [^=;()\n]*?\b(\w{3,}) = ", re.M),
    "py": re.compile(r"^[ \t]+(\w{3,}) = ", re.M),
    "sh": re.compile(r"^(\w{3,})=", re.M),
}
SKIP_FUNCS = {"require"}   # the per-file error helper; defined identically in several files
SCAN_DIRS = ("src", "tools", "tests")
SCAN_EXT = (".cc", ".h", ".py", ".sh", ".cl")
SCAN_SKIP = ("tests/golden", "tools/toolcall")   # generated data must not feed back into the set


def lang(path: str) -> str:
    return {".cc": "cc", ".py": "py", ".sh": "sh"}[os.path.splitext(path)[1]]


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def functions(path: str, text: str) -> list[tuple[str, int]]:
    out, seen = [], set()
    for m in FUNC_RE[lang(path)].finditer(text):
        n = m.group(1)
        if n in SKIP_FUNCS or n in seen:
            continue
        seen.add(n)
        out.append((n, line_of(text, m.start())))
    return out


def first_local(path: str, text: str) -> tuple[str, int]:
    m = LOCAL_RE[lang(path)].search(text)
    if not m:
        sys.exit(f"FATAL: no local in {path}")
    return m.group(1), line_of(text, m.start(1))


def comment_target(path: str, text: str):
    """-> (kind, target name, first line, last line) of a multi-line comment
    block directly above a definition; for Python, a function's docstring."""
    lines = text.split("\n")
    if lang(path) == "py":
        for i, ln in enumerate(lines):
            m = re.match(r"^def (\w+)\(", ln)
            if m and i + 1 < len(lines) and lines[i + 1].strip().startswith('"""'):
                j = i + 1
                if not (lines[j].strip().endswith('"""') and len(lines[j].strip()) > 3):
                    while not lines[j].rstrip().endswith('"""') or j == i + 1:
                        j += 1
                return ("docstring", m.group(1), i + 2, j + 1)
        sys.exit(f"FATAL: no docstring in {path}")
    mark = "//" if lang(path) == "cc" else "#"
    for i, ln in enumerate(lines):
        if i < 2 or not ln.strip() or ln.lstrip().startswith(mark):
            continue
        j = i
        while j > 0 and lines[j - 1].lstrip().startswith(mark) and not lines[j - 1].startswith("#!"):
            j -= 1
        if 2 <= i - j <= 6:   # long blocks would not fit an edit in 192 new tokens
            m = re.search(r"\b(\w+)\s*(?:\(|=)", ln)
            if m:
                return ("comment", m.group(1), j + 1, i)
    # No multi-line comment in the file (src/server/openai.cc has none): ask for
    # one to be added above the first function after the anonymous namespace.
    for i, ln in enumerate(lines):
        if ln.startswith("}  // namespace"):
            for n, k in functions(path, text):
                if k > i + 1:
                    return ("none", n, k, k)
    sys.exit(f"FATAL: no comment block in {path}")


def scan_repo() -> list[tuple[str, int, str]]:
    out = []
    for d in SCAN_DIRS:
        for dp, dns, fns in os.walk(os.path.join(ROOT, d)):
            dns.sort()
            rel_dir = os.path.relpath(dp, ROOT)
            if rel_dir.startswith(SCAN_SKIP):
                dns[:] = []
                continue
            for fn in sorted(fns):
                if not fn.endswith(SCAN_EXT):
                    continue
                rel = os.path.join(rel_dir, fn)
                with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
                    for k, ln in enumerate(f.read().split("\n"), 1):
                        out.append((rel, k, ln))
    return out


def hits(corpus, sym: str):
    r = re.compile(rf"\b{re.escape(sym)}\b")
    return [(p, k, ln) for p, k, ln in corpus if r.search(ln)]


def read_result(text: str, offset: int, limit: int) -> str:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    sel = lines[offset - 1: offset - 1 + limit]
    body = "\n".join(f"{offset + i:05d}| {ln}" for i, ln in enumerate(sel))
    end = offset - 1 + len(sel)
    tail = (f"\n\n(File has more lines. Use 'offset' parameter to read beyond line {end})"
            if end < len(lines) else f"\n\n(End of file - total {len(lines)} lines)")
    return f"<file>\n{body}{tail}\n</file>"


def call(i: int, name: str, args: dict) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": f"call_{i}", "type": "function",
                            "function": {"name": name, "arguments": args}}]}


def tool(i: int, content: str) -> dict:
    return {"role": "tool", "tool_call_id": f"call_{i}", "content": content}


def scenarios(corpus):
    """Yield (name, expect, builder(limit) -> messages, sizable)."""
    for rel in FILES:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            text = f.read()
        slug = os.path.splitext(os.path.basename(rel))[0]
        path = f"{WORK}/{rel}"
        funcs = functions(rel, text)
        if not funcs:
            sys.exit(f"FATAL: no function in {rel}")
        fn, fn_line = funcs[0]
        # a usage question wants a distinctive name: short ones (`at`) hit prose
        used = [(n, k) for n, k in funcs if len(n) >= 5 and len(hits(corpus, n)) >= 3]
        sym = next((n for n, _ in used if n != fn), used[0][0] if used else fn)
        local, local_line = first_local(rel, text)
        ckind, ctarget, c0, c1 = comment_target(rel, text)
        sys_msg = {"role": "system", "content": SYSTEM}

        # 1: find a definition; no history.
        yield (f"t1_define-{slug}", "grep", lambda lim, fn=fn: [
            sys_msg, {"role": "user", "content": f"Find where `{fn}` is defined."}], False)

        # 2: explain a function; glob gave the path.
        def b2(lim, fn=fn, rel=rel, path=path):
            base = os.path.basename(rel)
            return [sys_msg,
                    {"role": "user", "content": f"Explain `{fn}` in {base}."},
                    call(1, "glob", {"pattern": f"**/{base}", "path": WORK}), tool(1, path)]
        yield (f"t2_explain-{slug}", "read", b2, False)

        # 3: rename a local; a read of the file around it.
        new = f"{local}_val"

        def b3(lim, local=local, new=new, path=path, text=text, ll=local_line):
            off = max(1, ll - 20)
            return [sys_msg,
                    {"role": "user", "content": f"Rename `{local}` to `{new}` in {path}, "
                                                f"only the one defined on line {ll}."},
                    call(1, "read", {"filePath": path, "offset": off, "limit": lim}),
                    tool(1, read_result(text, off, lim))]
        yield (f"t3_rename-{slug}", "edit", b3, True)

        # 4: fix a multi-line comment (Python: a docstring); a read of the file.
        if ckind == "comment":
            task = (f"The comment above `{ctarget}` in {path} (lines {c0} to {c1}) is too long "
                    f"for what it says. Replace it with a single-line comment that keeps the "
                    f"essential point.")
        elif ckind == "none":
            task = (f"`{ctarget}` in {path} (line {c0}) has no comment. Add a short comment above "
                    f"it saying what it validates and what it throws.")
        else:
            task = (f"Fix the docstring of `{ctarget}` in {path} (line {c0}): it does not say what "
                    f"the function returns. Rewrite it as a summary line followed by a sentence "
                    f"about the return value.")

        def b4(lim, task=task, path=path, text=text, c0=c0):
            off = max(1, c0 - 10)
            return [sys_msg, {"role": "user", "content": task},
                    call(1, "read", {"filePath": path, "offset": off, "limit": lim}),
                    tool(1, read_result(text, off, lim))]
        yield (f"t4_comment-{slug}", "edit", b4, True)

        # 5: run the tests; a read of the file from the top.
        def b5(lim, path=path, text=text):
            return [sys_msg,
                    {"role": "user", "content": f"I just changed {path}. Run the tests for this file."},
                    call(1, "read", {"filePath": path, "limit": lim}),
                    tool(1, read_result(text, 1, lim))]
        yield (f"t5_test-{slug}", "bash", b5, True)

        # 6: where is a symbol used; grep gave three hits.
        hs = hits(corpus, sym)
        shown = "\n".join(f"{WORK}/{p}:{k}: {ln.strip()}" for p, k, ln in hs[:3])
        more = f"\n\n(Showing 3 of {len(hs)} matches)" if len(hs) > 3 else ""

        def b6(lim, sym=sym, shown=shown, more=more, n=min(3, len(hs))):
            return [sys_msg,
                    {"role": "user", "content": f"Where is `{sym}` used?"},
                    call(1, "grep", {"pattern": rf"\b{sym}\b", "path": WORK}),
                    tool(1, f"Found {n} matches\n{shown}{more}")]
        yield (f"t6_usage-{slug}", "read", b6, False)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    snap, out = sys.argv[1], sys.argv[2]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap)
    with open(os.path.join(HERE, "tools.json"), encoding="utf-8") as f:
        tools = json.load(f)

    def render(msgs) -> list[int]:
        r = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                    enable_thinking=False, tokenize=True)
        if hasattr(r, "keys"):          # transformers 5 returns a BatchEncoding
            r = r["input_ids"]
        return [int(x) for x in r]

    os.makedirs(out, exist_ok=True)
    corpus = scan_repo()
    manifest = []
    print("| scenario | expect | ids |\n|---|---|---:|")
    for name, expect, build, sizable in scenarios(corpus):
        lim = 160
        msgs = build(lim)
        ids = render(msgs)
        while sizable and len(ids) > MAX_IDS and lim > 10:
            lim -= 10
            msgs = build(lim)
            ids = render(msgs)
        if not MIN_IDS <= len(ids) <= MAX_IDS:
            sys.exit(f"FATAL: {name} renders to {len(ids)} ids, outside {MIN_IDS} to {MAX_IDS}")
        text = " ".join(map(str, ids)) + "\n"
        with open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump({"name": name, "expect": expect, "enable_thinking": False,
                       "messages": msgs, "tools": tools}, f, indent=1, ensure_ascii=False)
            f.write("\n")
        with open(os.path.join(out, f"{name}.ids"), "w", encoding="utf-8") as f:
            f.write(text)
        manifest.append({"name": name, "ids": len(ids),
                         "sha256": hashlib.sha256(text.encode()).hexdigest()})
        print(f"| {name} | {expect} | {len(ids)} |", flush=True)
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    n = [m["ids"] for m in manifest]
    print(f"\n{len(manifest)} scenarios, ids min {min(n)} max {max(n)} total {sum(n)}")


if __name__ == "__main__":
    main()
