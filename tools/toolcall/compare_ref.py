#!/usr/bin/env python3
"""Two A4 reference runs, id for id (oracle_generate.py's <name>.bf16.ids; the batched run's
cross-check against the sequential one).

    compare_ref.py <dir A> <dir B> [name ...]

For every scenario with a .bf16.ids in both (or the names given): equal, or the first position
the two continuations differ at, the two ids there and - where the runs wrote <name>.bf16.gap
(each greedy step's top-1 minus top-2 logit, fp32; oracle_generate.py writes it since the
batched mode) - both runs' gaps at that position: a flip at a near tie has a gap near 0 on one
side. Where both have gap files and the ids agree, the gaps are compared too (bitwise runs give
identical gaps). Exit 0 when everything compared is equal, 1 otherwise, 2 when nothing compared.
"""
import os
import sys

RUN = "bf16"


def read(path: str, cast):
    with open(path, encoding="utf-8") as f:
        return [cast(x) for x in f.read().split()]


def names_in(d: str) -> set[str]:
    suf = f".{RUN}.ids"
    return {f[: -len(suf)] for f in os.listdir(d) if f.endswith(suf)}


def compare(a: str, b: str, names: list[str]) -> tuple[int, int, list[str]]:
    same = diff = 0
    lines = []
    for n in names:
        ia, ib = read(os.path.join(a, f"{n}.{RUN}.ids"), int), read(os.path.join(b, f"{n}.{RUN}.ids"), int)
        ga_p, gb_p = os.path.join(a, f"{n}.{RUN}.gap"), os.path.join(b, f"{n}.{RUN}.gap")
        ga = read(ga_p, float) if os.path.exists(ga_p) else None
        gb = read(gb_p, float) if os.path.exists(gb_p) else None
        if ia == ib:
            note = ""
            if ga is not None and gb is not None:
                note = ", gaps identical" if ga == gb else ", ids equal but the GAPS DIFFER (not bitwise)"
                if ga != gb:
                    diff += 1
                    lines.append(f"DIFF  {n}: {len(ia)} ids equal{note}")
                    continue
            same += 1
            lines.append(f"same  {n}: {len(ia)} ids{note}")
            continue
        diff += 1
        j = next((k for k, (x, y) in enumerate(zip(ia, ib)) if x != y), min(len(ia), len(ib)))
        at = (f"A {ia[j] if j < len(ia) else 'end'} / B {ib[j] if j < len(ib) else 'end'}")
        gaps = " ".join(f"{s} gap {g[j]:.4g}" if g is not None and j < len(g) else f"{s} gap n/a (no .gap file)"
                        for s, g in (("A", ga), ("B", gb)))
        lines.append(f"DIFF  {n}: first differs at new id {j} ({at}; {len(ia)} vs {len(ib)} ids); {gaps}")
    return same, diff, lines


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a, b = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or sorted(names_in(a) & names_in(b))
    if not names:
        print(f"nothing to compare: no <name>.{RUN}.ids in both {a} and {b}")
        sys.exit(2)
    same, diff, lines = compare(a, b, names)
    print("\n".join(lines))
    print(f"{same} same, {diff} different of {len(names)} ({a} vs {b})")
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
