#!/usr/bin/env python3
"""README.md's flag tables against the binaries' generated --help (src/cli/flags.h).

Every flag a table names is the binary's (a b70-decode table's row that names `b70-serve` is b70-serve's), every flag
either --help names is in a table, and a row of one flag whose default is one `literal` has the default --help gives.
Usage: readme_flags_test.py README.md <b70-serve> <b70-decode>
"""
import re
import subprocess
import sys


def help_flags(binary):
    out = subprocess.run([binary, "--help"], capture_output=True, text=True, check=True).stderr
    flags, text = {}, None
    for line in out.splitlines():
        m = re.match(r"  (--[a-z0-9-]+)", line)
        if m:
            text = flags.setdefault(m.group(1), [])
        if text is not None and line.startswith("  "):
            text.append(line)
        elif not line.strip():
            text = None
    defaults = {}
    for f, lines in flags.items():
        d = re.findall(r"\(default ([^)]*)\)", " ".join(lines))
        defaults[f] = d[-1] if d else None
    return defaults


def readme_rows(path):
    rows, owner = [], None
    for line in open(path, encoding="utf-8"):
        if line.startswith("`b70-serve`"):
            owner = "serve"
        elif line.startswith("`b70-decode`") or line.startswith("Developer flags (`b70-decode`)"):
            owner = "decode"
        elif line.startswith("## "):
            owner = None
        elif owner and line.startswith("| `"):
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
            rows.append(("serve" if "`b70-serve`" in line else owner, cells))
    return rows


def main():
    readme, serve, decode = sys.argv[1:4]
    helps = {"serve": help_flags(serve), "decode": help_flags(decode)}
    named, bad = set(), []
    for owner, cells in readme_rows(readme):
        flags = re.findall(r"--[a-z0-9-]+", cells[0])
        named.update(flags)
        for f in flags:
            if f not in helps[owner]:
                bad.append(f"README names {f} for b70-{owner}, whose --help does not")
        m = re.fullmatch(r"`([^`]+)`", cells[1])
        if len(flags) == 1 and m and flags[0] in helps[owner] and helps[owner][flags[0]] != m.group(1):
            bad.append(f"README's default of {flags[0]} is {m.group(1)}, b70-{owner} --help says {helps[owner][flags[0]]}")
    for owner, h in helps.items():
        for f in h:
            if f not in named:
                bad.append(f"b70-{owner} --help names {f}, no README table does")
    for b in bad:
        print(b, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
