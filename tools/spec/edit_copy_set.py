#!/usr/bin/env python3
"""The edit / copy prompt set for prompt lookup inside DFlash (spec 19f; spec 19 §13.1).

    edit_copy_set.py <snapshot> <out dir>            (tokenizer: transformers; on the Mac
        uv run --no-project --with transformers==4.57.1 --with jinja2 \\
            python -I tools/spec/edit_copy_set.py <Qwen3.8 snapshot> tests/golden/edit-copy)
    edit_copy_set.py --check <set dir>               (the committed set: sha256, depths, distances; no tokenizer)

12 agentic-coding conversations built from this repository's own files (no external data), 3 kinds at each of
the depths 8k / 32k / 64k / 99k of prompt ids, each ending where the assistant's next turn re-emits text that lies
FAR back in the context - beyond the DFlash drafter's 2048-id window, where only prompt lookup can see it:
  reemit  the first 120 lines of a file printed verbatim by a bash sed (the source), more files were read after it,
          and the last turn asks to apply a rename across those lines and output them;
  rename  a function was shown by a read (the source), more files were read, and the last turn asks for the
          function back with one local renamed;
  quote   a grep's output (the source), more files were read, and the last turn asks to quote that output back
          exactly as the tool printed it.
The reads in between (the filler) are whole files of this tree in a fixed order, the last one cut so the rendered
prompt lands at the depth (within DEPTH_TOL). The checkpoint's own chat template renders them (tools from
tools/toolcall/tools.json, enable_thinking False) - make_set.py's helpers and conventions, so serve_client.py runs
the set as it runs A4 (`--set tests/golden/edit-copy`).

Writes <name>.json (messages, tools, the kind, the depth, the source's id span and its distance), <name>.ids and
manifest.json (name, ids, sha256, kind, depth, source span, distance = ids from the source's end to the prompt's
end). Every distance must exceed MIN_DISTANCE (the drafter's window, 2048): the builder refuses a prompt whose
source is not far.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools", "toolcall"))
import make_set as ms  # noqa: E402

DEPTHS = {"8k": 8192, "32k": 32768, "64k": 65536, "99k": 99000}
DEPTH_TOL = 0.02          # the rendered prompt within 2 % below the depth
MIN_DISTANCE = 2048       # the drafter's window: the source must lie beyond it
SOURCE_LINES = 120        # a source read's line limit (the copy the last turn asks for)

# The source of each kind at each depth (a different file per depth, so no two prompts share a source).
REEMIT = {"8k": "src/server/adaptive_k.cc", "32k": "src/server/chat_format.cc",
          "64k": "src/server/spec_accept.cc", "99k": "tools/box_validate/serve_client.py"}
RENAME = {"8k": ("src/server/prompt_lookup.cc", "PromptLookup::match"),
          "32k": ("src/server/prefix_cache.cc", "PrefixCache::plan"),
          "64k": ("src/server/scheduler.cc", "PrefixSlots::store"),
          "99k": ("src/server/toolcall.cc", "schema_type")}
QUOTE = {"8k": "accept_greedy", "32k": "MtpCost", "64k": "PromptLookup", "99k": "kVocab"}

# The filler, in order (whole files of this tree; a file used as a source is skipped).
FILLER = [
    "src/server/server.cc", "src/server/openai.cc", "src/runtime/prefill/step.cc", "src/loader/loader.cc",
    "src/runtime/engine.cc", "src/runtime/engine.h", "src/cli/b70_serve.cc", "src/cli/b70_decode.cc",
    "src/runtime/capture.cc", "src/loader/safetensors.cc", "src/model/qwen35.cc", "src/runtime/prefill/linear_l0.cc",
    "src/tokenizer/chat_template.cc", "src/server/scheduler.h", "src/server/prefix_cache.h", "src/l0/memory.cc",
    "src/l0/context.cc", "src/l0/cmdlist.cc", "tools/box_validate/lib.sh", "tools/box.sh",
    "tools/toolcall/make_set.py", "tools/rotate/rotation.py", "src/runtime/dflash/dflash_drafter.cc",
    "src/loader/dflash_loader.cc", "src/runtime/memory_plan.cc",
]

USER = {
    "reemit_read": "Print the first {n} lines of {path}.",
    "reemit_ask": ("In those first lines of {path}, rename `{old}` to `{new}` everywhere (the definition and every "
                   "use) and output the lines with the change, nothing else."),
    "rename_read": "Show me `{fn}` in {path}.",
    "rename_ask": ("Rename the local `{local}` to `{local}_v` in `{fn}` ({path}) and print the whole function "
                   "with the change, nothing else."),
    "quote_grep": "Where is `{sym}` used?",
    "quote_ask": ("Quote the grep output for `{sym}` from earlier back to me exactly as the tool printed it, every "
                  "line, then say which hit you would edit first."),
    "filler": "Before we go on, read the files the build touches, one at a time.",
}


def read_file(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
        return f.read()


def n_lines(text):
    ls = text.split("\n")
    return len(ls) - 1 if ls and ls[-1] == "" else len(ls)


class Builder:
    """Renders with `tok` (anything with HF's apply_chat_template) and sizes the filler."""

    def __init__(self, tok, tools, corpus=None, depths=None, filler=None, min_distance=MIN_DISTANCE):
        self.tok = tok
        self.tools = tools
        self.corpus = corpus
        self.depths = depths or DEPTHS
        self.filler = filler or FILLER
        self.min_distance = min_distance

    def render(self, msgs, gen=True):
        r = self.tok.apply_chat_template(msgs, tools=self.tools, add_generation_prompt=gen,
                                         enable_thinking=False, tokenize=True)
        if hasattr(r, "keys"):
            r = r["input_ids"]
        return [int(x) for x in r]

    # --- the source part of each kind: (messages up to and including the source's tool result, last turn) ---
    def reemit(self, depth):
        rel = REEMIT[depth]
        text, path = read_file(rel), f"{ms.WORK}/{rel}"
        funcs = ms.functions(rel, text)
        old = next((n for n, _ in funcs if len(n) >= 5), funcs[0][0])
        # the source verbatim (a bash sed of the lines, no line-number prefixes): the copy the last turn asks for is
        # the text itself, as in the published far-copy measurement
        shown = "\n".join(text.split("\n")[:SOURCE_LINES])
        head = [{"role": "user", "content": USER["reemit_read"].format(path=path, n=SOURCE_LINES)},
                ms.call(1, "bash", {"command": f"sed -n 1,{SOURCE_LINES}p {path}",
                                    "description": f"Print the first {SOURCE_LINES} lines of {os.path.basename(rel)}"}),
                ms.tool(1, shown)]
        return head, USER["reemit_ask"].format(path=path, old=old, new=old + "_impl"), [rel]

    def rename(self, depth):
        rel, fn = RENAME[depth]
        text, path = read_file(rel), f"{ms.WORK}/{rel}"
        lines = text.split("\n")
        line = next((i + 1 for i, ln in enumerate(lines) if not ln[:1].isspace() and f"{fn}(" in ln), None)
        if line is None:
            sys.exit(f"FATAL: no definition of {fn} in {rel}")
        local, _ = ms.first_local(rel, "\n".join(lines[line - 1:]))
        off = max(1, line - 2)
        head = [{"role": "user", "content": USER["rename_read"].format(fn=fn, path=path)},
                ms.call(1, "read", {"filePath": path, "offset": off, "limit": SOURCE_LINES}),
                ms.tool(1, ms.read_result(text, off, SOURCE_LINES))]
        return head, USER["rename_ask"].format(local=local, fn=fn, path=path), [rel]

    def quote(self, depth):
        sym = QUOTE[depth]
        if self.corpus is None:
            self.corpus = ms.scan_repo()
        hs = ms.hits(self.corpus, sym)
        shown = "\n".join(f"{ms.WORK}/{p}:{k}: {ln.strip()}" for p, k, ln in hs[:40])
        more = f"\n\n(Showing 40 of {len(hs)} matches)" if len(hs) > 40 else ""
        head = [{"role": "user", "content": USER["quote_grep"].format(sym=sym)},
                ms.call(1, "grep", {"pattern": rf"\b{sym}\b", "path": ms.WORK}),
                ms.tool(1, f"Found {min(40, len(hs))} matches\n{shown}{more}")]
        return head, USER["quote_ask"].format(sym=sym), []

    def build(self, kind, depth):
        """-> (record, ids): the conversation at the depth, the source's id span and distance."""
        sys_msg = {"role": "system", "content": ms.SYSTEM_EN}
        head, ask, used = getattr(self, kind)(depth)
        source = [sys_msg] + head
        target = self.depths[depth]
        files = [f for f in self.filler if f not in used]

        def conv(n_files, last_lines):
            msgs = list(source) + [{"role": "user", "content": USER["filler"]}]
            for i in range(n_files):
                rel = files[i]
                text = read_file(rel)
                lim = n_lines(text) if i < n_files - 1 or last_lines is None else last_lines
                path = f"{ms.WORK}/{rel}"
                msgs.append(ms.call(2 + i, "read", {"filePath": path, "limit": lim}))
                msgs.append(ms.tool(2 + i, ms.read_result(text, 1, lim)))
            msgs.append({"role": "user", "content": ask})
            return msgs

        # whole filler files while the prompt stays below the depth, then the next one cut by lines
        n = 0
        while True:
            if n == len(files):
                sys.exit(f"FATAL: {kind}/{depth}: the filler runs out at {len(self.render(conv(n, None)))} ids")
            if len(self.render(conv(n + 1, None))) > target:
                break
            n += 1
        if len(self.render(conv(n + 1, 1))) > target:   # not even one line of the next file fits
            msgs = conv(n, None)
        else:
            lo, hi = 1, n_lines(read_file(files[n]))   # the cut file's lines: the largest that fits
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(self.render(conv(n + 1, mid))) <= target:
                    lo = mid
                else:
                    hi = mid - 1
            msgs = conv(n + 1, lo)
        ids = self.render(msgs)
        if not target * (1 - DEPTH_TOL) <= len(ids) <= target:
            sys.exit(f"FATAL: {kind}/{depth}: {len(ids)} ids, outside {target * (1 - DEPTH_TOL):.0f} .. {target}")
        # the source's span: the render of the messages before it and with it (the template is prefix-stable over
        # these turns - checked)
        before = self.render(source[:-1], gen=False)
        with_src = self.render(source, gen=False)
        if ids[:len(with_src)] != with_src or with_src[:len(before)] != before:
            sys.exit(f"FATAL: {kind}/{depth}: the template is not prefix-stable here")
        span = [len(before), len(with_src)]
        distance = len(ids) - span[1]
        if distance <= self.min_distance:
            sys.exit(f"FATAL: {kind}/{depth}: the source ends {distance} ids before the prompt's end "
                     f"(<= {self.min_distance}): not beyond the drafter's window")
        name = f"{kind}-{depth}"
        rec = {"name": name, "kind": kind, "depth": depth, "expect": "copy", "enable_thinking": False,
               "source_span": span, "distance": distance, "messages": msgs, "tools": self.tools}
        return rec, ids


def build_set(tok, out, tools=None, **kw):
    if tools is None:
        with open(os.path.join(ROOT, "tools", "toolcall", "tools.json"), encoding="utf-8") as f:
            tools = json.load(f)
    b = Builder(tok, tools, **kw)
    os.makedirs(out, exist_ok=True)
    manifest = []
    print("| prompt | ids | source span | distance |\n|---|---:|---|---:|")
    for depth in b.depths:
        for kind in ("reemit", "rename", "quote"):
            rec, ids = b.build(kind, depth)
            text = " ".join(map(str, ids)) + "\n"
            with open(os.path.join(out, rec["name"] + ".json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1, ensure_ascii=False)
                f.write("\n")
            with open(os.path.join(out, rec["name"] + ".ids"), "w", encoding="utf-8") as f:
                f.write(text)
            manifest.append({"name": rec["name"], "ids": len(ids), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                             "kind": rec["kind"], "depth": rec["depth"], "source_span": rec["source_span"],
                             "distance": rec["distance"]})
            print(f"| {rec['name']} | {len(ids)} | {rec['source_span']} | {rec['distance']} |", flush=True)
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    return manifest


def check_set(d, depths=None, min_distance=MIN_DISTANCE):
    """The committed set: every .ids file's sha256 and count as the manifest says, at its depth, its source far."""
    depths = depths or DEPTHS
    with open(os.path.join(d, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    problems = []
    for e in manifest:
        with open(os.path.join(d, e["name"] + ".ids"), encoding="utf-8") as f:
            text = f.read()
        ids = text.split()
        if hashlib.sha256(text.encode()).hexdigest() != e["sha256"] or len(ids) != e["ids"]:
            problems.append(f"{e['name']}: .ids does not match the manifest")
        t = depths[e["depth"]]
        if not t * (1 - DEPTH_TOL) <= e["ids"] <= t:
            problems.append(f"{e['name']}: {e['ids']} ids, not at depth {e['depth']}")
        if e["distance"] <= min_distance or e["distance"] != e["ids"] - e["source_span"][1]:
            problems.append(f"{e['name']}: distance {e['distance']}")
        with open(os.path.join(d, e["name"] + ".json"), encoding="utf-8") as f:
            rec = json.load(f)
        if rec["messages"][-1]["role"] != "user" or rec["source_span"] != e["source_span"]:
            problems.append(f"{e['name']}: the record")
    return manifest, problems


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        manifest, problems = check_set(sys.argv[2])
        for p in problems:
            print("FAIL", p)
        print(f"edit-copy set: {len(manifest)} prompts, {len(problems)} problems")
        sys.exit(1 if problems else 0)
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    build_set(ms.load_tokenizer(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
