#!/usr/bin/env python3
"""The box validation run's summary, and the medians of interleaved arms (stdlib only).

  summary.py medians ROWS [--ratio A/B]...
        medians over the rounds of tools/box_validate/interleave.sh's ROWS file
  summary.py report --state DIR --registry TSV [--meta K=V]... --out FILE
        box-validation-<date>.md from a fetched state directory: a table per queue row
        (PASS / FAIL / SKIP, the reason, the log), the numbers each stage recorded
        (METRIC lines, JUnit counts, interleaved medians), what no stage runs (row notes),
        and the commands of the opt-in and manual stages that did not run.

State directory (written on the box by tools/box_validate/remote.sh):
  <id>.status   one line: id, result, rc, start, end, seconds, reason (tab separated)
  <id>.log      the stage's log: METRIC / NOTE / BAD / SKIP_REASON / JUNIT lines among the rest
  <id>.rows     interleave.sh's ROW / WARM lines, for timed stages
  meta.env      KEY=value lines (shas, trees, times)
Registry TSV (written by tools/box_validate.sh from tools/box_validate/stages.sh):
  row<TAB>id<TAB>title | note<TAB>row<TAB>text | stage<TAB>id<TAB>row<TAB>sel<TAB>kind<TAB>needs<TAB>after<TAB>title
  | cmd<TAB>id<TAB>line  (the commands of opt-in and manual stages)
"""
import argparse
import os
import re
import statistics
import sys
from collections import OrderedDict, defaultdict


# ---- medians --------------------------------------------------------------------------
def parse_rows(lines):
    """-> {label: {"tg": [...], "pp": [...], "ms": [...]}}, {(label, prompt): {"tps": [...], "acc": [...]}}"""
    dec = defaultdict(lambda: defaultdict(list))
    bench = defaultdict(lambda: defaultdict(list))
    for line in lines:
        parts = line.rstrip("\n").split(" ", 2)
        if len(parts) < 3 or parts[0] != "ROW":
            continue
        label, rest = parts[1], parts[2]
        if rest.startswith("| b70-decode"):
            f = [c.strip() for c in rest.split("|")]
            try:
                if f[1].endswith(" pp"):   # the prefill row: | ... pp | N | C | ms | t/s |
                    dec[label]["pp"].append(float(f[5]))
                else:
                    dec[label]["tg"].append(float(f[4]))
                    dec[label]["ms"].append(float(f[5]))
            except (IndexError, ValueError):
                pass
        elif rest.startswith("BENCH"):
            kv = dict(t.split("=", 1) for t in rest.split()[1:] if "=" in t)
            key = (label, kv.get("prompt", "?"))
            try:
                bench[key]["tps"].append(float(kv["tps"]))
                bench[key]["acc"].append(float(kv.get("acc", "nan")))
            except (KeyError, ValueError):
                pass
    return dec, bench


def med(xs):
    return statistics.median(xs) if xs else None


def spread(xs):
    m = med(xs)
    return 100.0 * (max(xs) - min(xs)) / m if xs and m else None


def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def medians_md(lines, ratios=()):
    dec, bench = parse_rows(lines)
    out = []
    if dec:
        out += ["| arm | tg t/s (median) | ms/token | spread | pp t/s (median) | spread | runs |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for label, d in dec.items():
            out.append(f"| {label} | {fmt(med(d['tg']))} | {fmt(med(d['ms']))} | {fmt(spread(d['tg']))}% | "
                       f"{fmt(med(d['pp']))} | {fmt(spread(d['pp']))}% | {max(len(d['tg']), len(d['pp']))} |")
    if bench:
        if out:
            out.append("")
        out += ["| arm | prompt | t/s (median) | spread | acceptance (median) | runs |",
                "|---|---|---:|---:|---:|---:|"]
        for (label, prompt), d in bench.items():
            out.append(f"| {label} | {prompt} | {fmt(med(d['tps']))} | {fmt(spread(d['tps']))}% | "
                       f"{fmt(med(d['acc']), 4)} | {len(d['tps'])} |")
    if ratios and out:
        out.append("")   # a line after a table must not read as one of its rows
    for r in ratios:
        a, _, b = r.partition("/")
        if a in dec and b in dec:
            parts = []
            for k in ("tg", "pp"):
                ma, mb = med(dec[a][k]), med(dec[b][k])
                if ma and mb:
                    parts.append(f"{k} {ma / mb:.4f}")
            out.append(f"RATIO {a}/{b} " + (", ".join(parts) if parts else "(no common rows)"))
        else:
            prompts = sorted({p for (l, p) in bench if l == a} & {p for (l, p) in bench if l == b})
            parts = []
            for p in prompts:
                ma, mb = med(bench[(a, p)]["tps"]), med(bench[(b, p)]["tps"])
                if ma and mb:
                    parts.append(f"{p} {ma / mb:.4f}")
            out.append(f"RATIO {a}/{b} " + ("; ".join(parts) if parts else "(no common rows)"))
    return out


def cmd_medians(a):
    with open(a.rows, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print("\n".join(medians_md(lines, a.ratio or [])))
    return 0


# ---- report ---------------------------------------------------------------------------
def read_registry(path):
    rows, notes, stages, cmds = OrderedDict(), defaultdict(list), OrderedDict(), defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] == "row" and len(p) >= 3:
                rows[p[1]] = p[2]
            elif p[0] == "note" and len(p) >= 3:
                notes[p[1]].append(p[2])
            elif p[0] == "stage" and len(p) >= 8:
                stages[p[1]] = dict(row=p[2], sel=p[3], kind=p[4], needs=p[5], after=p[6], title=p[7])
            elif p[0] == "cmd" and len(p) >= 3:
                cmds[p[1]].append(p[2])
    return rows, notes, stages, cmds


def read_status(state, sid):
    path = os.path.join(state, sid + ".status")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        p = f.read().strip().split("\t")
    p += [""] * (7 - len(p))
    return dict(result=p[1], rc=p[2], start=p[3], end=p[4], secs=p[5], reason=p[6])


def read_log(state, sid):
    path = os.path.join(state, sid + ".log")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readlines()


def cell(s, n=300):
    s = (s or "").replace("|", "\\|").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def hms(secs):
    try:
        s = int(float(secs))
    except ValueError:
        return ""
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def cmd_report(a):
    rows, notes, stages, cmds = read_registry(a.registry)
    meta = OrderedDict(kv.split("=", 1) for kv in a.meta or [] if "=" in kv)
    mp = os.path.join(a.state, "meta.env")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.rstrip("\n").split("=", 1)
                    meta.setdefault(k, v)
    logdir = a.logdir or a.state

    res = OrderedDict()
    for sid, s in stages.items():
        st = read_status(a.state, sid)
        if s["sel"] == "manual":
            r = dict(result="MANUAL", reason="operator run - commands below")
        elif st is None:
            r = dict(result="OPT-IN" if s["sel"] == "optin" else "NOT RUN",
                     reason="not selected (opt-in)" if s["sel"] == "optin" else "no status in this run")
        else:
            r = st
            if r["result"] == "RUNNING":
                r = dict(r, result="INCOMPLETE", reason="started, never finished (the orchestrator died?)")
        res[sid] = r

    out = [f"# Box validation - {a.date or meta.get('finished', '')[:10] or 'run'}", ""]
    for k in ("tree_sha", "baseline_sha", "baseline_ref", "state", "tree", "baseline_tree", "started",
              "finished", "selected"):
        if k in meta:
            out.append(f"- {k.replace('_', ' ')}: `{meta[k]}`")
    g0 = [sid for sid, s in stages.items() if s["sel"] == "g0"]
    g0_ok = all(res[s]["result"] == "PASS" for s in g0)
    out += ["", f"**G0 (bitwise neutrality, stop-the-line): {'PASS' if g0_ok else 'NOT PASSED'}** - "
            + ", ".join(f"{s} {res[s]['result']}" for s in g0), ""]

    out += ["| row | what | PASS | FAIL | SKIP | other |", "|---|---|---:|---:|---:|---|"]
    for rid, title in rows.items():
        ids = [sid for sid, s in stages.items() if s["row"] == rid]
        c = defaultdict(int)
        for sid in ids:
            c[res[sid]["result"]] += 1
        other = ", ".join(f"{k.lower()} {v}" for k, v in c.items() if k not in ("PASS", "FAIL", "SKIP"))
        out.append(f"| {rid} | {cell(title, 90)} | {c['PASS']} | {c['FAIL']} | {c['SKIP']} | {other} |")
    out.append("")

    for rid, title in rows.items():
        ids = [sid for sid, s in stages.items() if s["row"] == rid]
        out += [f"## Row {rid} - {title}", "",
                "| stage | what | result | reason / notes | time | log |", "|---|---|---|---|---|---|"]
        numbers = []
        for sid in ids:
            s, r = stages[sid], res[sid]
            log = read_log(a.state, sid)
            extra = []
            for line in log:
                if line.startswith("NOTE "):
                    extra.append(line[5:].strip())
                elif line.startswith("JUNIT total="):
                    extra.append(line.strip()[6:])
            reason = "; ".join([r.get("reason", "")] + extra[:6]).strip("; ")
            logp = os.path.join(logdir, sid + ".log") if log else ""
            out.append(f"| `{sid}` | {cell(s['title'], 120)} | **{r['result']}** | {cell(reason)} | "
                       f"{hms(r.get('secs', ''))} | {('`' + logp + '`') if logp else ''} |")
            mets = OrderedDict()
            for line in log:
                if line.startswith("METRIC "):
                    lab, _, val = line[7:].rstrip("\n").partition("\t")
                    mets.setdefault((lab, val.strip()), None)
            fails = [l.strip() for l in log if l.startswith(("FAILED ", "BAD "))]
            rows_path = os.path.join(a.state, sid + ".rows")
            med_lines = []
            if os.path.exists(rows_path):
                with open(rows_path, encoding="utf-8", errors="replace") as f:
                    ratios = [l.split()[1] for l in log if l.startswith("RATIO ")]
                    med_lines = medians_md(f.readlines(), ratios)
            if mets or fails or med_lines:
                numbers.append(f"**`{sid}`**")
                numbers.append("")
                for (lab, val) in mets:
                    numbers.append(f"- {lab}: `{cell(val, 400)}`")
                for l in fails[:20]:
                    numbers.append(f"- {cell(l, 400)}")
                if med_lines:
                    numbers += [""] + med_lines
                numbers.append("")
        out.append("")
        if numbers:
            out += ["Measured / recorded:", ""] + numbers
        if notes.get(rid):
            out += ["Not run by this runbook:", ""] + [f"- {n}" for n in notes[rid]] + [""]
        waiting = [sid for sid in ids if res[sid]["result"] in ("MANUAL", "OPT-IN") and cmds.get(sid)]
        for sid in waiting:
            out += [f"`{sid}` ({res[sid]['result'].lower()}): {stages[sid]['title']}", "", "```"]
            out += cmds[sid] + ["```", ""]
    text = "\n".join(out).rstrip() + "\n"
    if a.out == "-":
        sys.stdout.write(text)
    else:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"summary: {a.out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("medians")
    p.add_argument("rows")
    p.add_argument("--ratio", action="append")
    p.set_defaults(fn=cmd_medians)
    p = sub.add_parser("report")
    p.add_argument("--state", required=True)
    p.add_argument("--registry", required=True)
    p.add_argument("--meta", action="append")
    p.add_argument("--logdir", help="how to print log paths (default: --state)")
    p.add_argument("--date")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
