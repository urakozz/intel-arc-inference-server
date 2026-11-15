#!/usr/bin/env python3
"""The f32 judge on b70-serve's speculative arms (spec 28 N6, tests/server/spec_judge.h's streams()), for the server
rows whose bar was "greedy output identical to --spec off" (r2.auto_rows, r12.a4).

  serve_judge.py --dir D --ref ARM --dump DIR --judge DIR [--bf16 DIR] [--tag serve_int8] [--header FILE]
                 [--snapshot DIR] [--vocab-used 248077]

THE END OF TURN (item 30 (3), operator 2026-11-15; golden_common.h turn_end): a server stops at the model's end-of-turn
id and sends it, the arms run with ignore_eos. Each stream is graded up to and including its first end-of-turn id -
the model's eos list, from --snapshot's generation_config.json, else the `eos [..]` list b70-serve printed in the arms'
server.log. The rows after it are not graded and not counted (a rejected one is printed "past end of turn (row R): not
graded"); a first difference after the shared end of turn is not judged at all (TURN, nothing dumped).

Since the small-batch verify kernels (spec 28b / 28c) a speculative arm's verify rows round differently from the
M = 1 rows, so its greedy stream may leave the plain one; identity is a recorded number, not the bar. The bar is
spec_judge.h's N6 rule, applied to every request of every other arm against the --ref arm (serve_client.py
compare's pairing: the ref arm's lowest round):

  identical                        passes.
  first difference at d            the two share the context prompt + plain[0, d); the plain suffix plain[d, n) and
                                   the speculative suffix spec[d, n) (n: the shorter stream) are each a forced
                                   context, key <tag>_<set>_<name>_<FNV-1a 64> exactly as golden::ForcedJudge::key -
                                   both are dumped to --dump as <key>.prompt.ids / <key>.force.ids (what
                                   tools/box_validate/f32_refs.sh forced-qwen turns into references), and their f32
                                   references are read from --judge (<key>.golden.safetensors, `logits` [1 + gen][V]).
  both references there            the f32 judge (golden_common.h judge_row's one-ulp rule) counts the ids it rejects
                                   in each suffix; the arm PASSES when it rejects no more speculative ids than plain
                                   ids, or when every speculative miss is a near-miss (<= kNearMissUlp = 8 bf16 ulp
                                   of the f32 top below it). Otherwise each miss beyond 8 ulp goes to the
                                   bf16-unstable rule (operator 2026-11-04): it needs the ENGINE's plain M = 1 logits
                                   row on that context, which b70-serve does not return - so such a miss stays on
                                   the strict bar and FAILS (spec_judge.h: "a miss with no plain row stays on the
                                   strict bar"), unless the bf16 reference itself picks the speculative id there (bf16-
                                   unstable, operator 2026-11-08: PASS; unstable is one-sided, item 30 (4), operator
                                   2026-11-15: the bf16 gap more than 8 ulp below the f32 gap, a move toward the
                                   speculative pick) or it matches a kReviewedRows entry of spec_judge.h (read from
                                   --header; exactly, or by tag / row / ids / f32 gap as match_reviewed). The bf16
                                   gap is printed when --bf16 has the key's <key>.bf16top.safetensors
                                   (tools/oracle/qwen_bf16_top.py), as spec_judge.h does.
  a reference missing              NOT judged and NOT passed: the arm is SKIPPED, the missing keys named (spec
                                   28's "awaiting f32 reference"). Without the plain logits rows b70-serve does not
                                   return, the 1-ulp tie rule cannot be applied here either.
  no ids after d                   one stream is a prefix of the other (a length difference, not a token one): FAIL.
  prompts differ / missing         the two arms' prompt ids differ: FAIL; a request missing: FAIL (as compare).
                                   No prompt ids in the record (prompt_ids, serve_client.py run) and no committed
                                   .ids file matching: SKIP, said so.

The tag carries the head form (qwen_forced.py's I8HEAD_RE matches serve_int8_: b70-serve loads the int8 head and
prefills with l0, so the judges are the plain int8-head f32 references, oracle-out-qwen-f32-i8head-forced).

Prints one JUDGE / SKIP line per differing request, `JUDGED <arm> r<round> vs <ref>: ...` per arm-round and a
`SKIP_REASON serve_judge: ...` line naming the awaiting contexts. Exit 1 when anything failed, else 77 (SKIP) when
anything awaits a reference, else 0. numpy is used when it is installed (the box host has it: qwen_bf16_top.py
plan), else plain Python (slow, for tests).
"""
import argparse
import array
import glob
import json
import math
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import serve_client  # noqa: E402

try:
    import numpy as np
except ImportError:   # the Mac: the tests run without it
    np = None

NEAR_MISS_ULP = 8.0      # golden::kNearMissUlp (tests/golden/forced_judge.h)
REVIEWED_GAP_ULP = 0.01  # spec_judge.h kReviewedGapUlp
VOCAB_USED = 248077      # Qwen3.8's tokenizer ids (Qwen35::kVocabUsed)
HEADER = os.path.join(ROOT, "tests", "server", "spec_judge.h")


# ---- golden::ForcedJudge::key ----------------------------------------------------------------------------------
def fnv_key_hash(prompt, force):
    """FNV-1a 64 over the ids' little-endian bytes, 0xFFFFFFFF between (qwen_forced.py fnv_key_hash)."""
    h = 1469598103934665603
    for v in list(prompt) + [0xFFFFFFFF] + list(force):
        for b in range(4):
            h = ((h ^ ((v >> (8 * b)) & 0xFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def forced_key(name, prompt, force):
    n = "".join("_" if c in " :/," else c for c in name)
    return n + "_" + fnv_key_hash(prompt, force)


def dump(d, k, prompt, force):
    os.makedirs(d, exist_ok=True)
    for ext, v in ((".prompt.ids", prompt), (".force.ids", force)):
        with open(os.path.join(d, k + ext), "w") as f:
            f.write(" ".join(map(str, v)) + "\n")


# ---- safetensors (no torch) ------------------------------------------------------------------------------------
class Tensors:
    """A safetensors file's header; tensors read as numpy memmaps (numpy) or arrays."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.h = json.loads(f.read(n))
        self.base = 8 + n

    def shape(self, name):
        return tuple(self.h[name]["shape"])

    def has(self, name):
        return name in self.h

    def flat(self, name):
        t = self.h[name]
        code = {"F32": "f", "I32": "i"}[t["dtype"]]
        a, b = t["data_offsets"]
        if np is not None:
            dt = np.float32 if code == "f" else np.int32
            return np.memmap(self.path, dtype=dt, mode="r", offset=self.base + a, shape=((b - a) // 4,))
        out = array.array(code)
        with open(self.path, "rb") as f:
            f.seek(self.base + a)
            out.frombytes(f.read(b - a))
        if sys.byteorder != "little":
            out.byteswap()
        return out


class Judge:
    """<key>.golden.safetensors: `logits` [1 + gen][V], row j the decision of forced id j."""

    def __init__(self, path, force):
        self.t = Tensors(path)
        self.rows, self.V = self.t.shape("logits")
        if self.rows != len(force) + 1:
            raise ValueError(f"{path}: {self.rows} logits rows, the context forces {len(force)} ids (+1 expected)")
        if self.t.has("tokens"):
            tok = list(self.t.flat("tokens"))
            if [int(x) for x in tok] != list(force):
                raise ValueError(f"{path}: its `tokens` are not this context's forced ids")
        self.lg = self.t.flat("logits")

    def row(self, r):
        return self.lg[r * self.V:(r + 1) * self.V]


class Bf16Top:
    """<key>.bf16top.safetensors (qwen_bf16_top.py): `ids` int32 / `logits` f32 [1 + gen][C]."""

    def __init__(self, path):
        t = Tensors(path)
        self.rows, self.cols = t.shape("ids")
        self.ids, self.lg = t.flat("ids"), t.flat("logits")

    def at(self, r, i):
        for c in range(self.cols):
            if int(self.ids[r * self.cols + c]) == i:
                return float(self.lg[r * self.cols + c])
        return math.nan


# ---- golden_common.h judge_row / forced_judge.h row_accepts ----------------------------------------------------
def bf16_ulp_at(v):
    return 2.0 ** (math.frexp(abs(float(v)))[1] - 8)


def judge_row(row, used):
    """(top, accepted set, ulp at the top): the f32 argmax (lowest index on an exact tie); the set is every exact
    tie, or - when the top is unique but the top-2 gap is within one bf16 ulp - every id within one ulp of it."""
    lim = min(len(row), used)
    if np is not None:
        r = np.asarray(row[:lim], dtype=np.float32)
        bv = float(r.max())
        tied = np.flatnonzero(r == r.max()).tolist()
        top = tied[0]
        second = max(float(r[:top].max()) if top > 0 else -math.inf,
                     float(r[top + 1:].max()) if top + 1 < lim else -math.inf)
        ulp = bf16_ulp_at(bv)
        if len(tied) == 1 and bv - second <= ulp:
            tied = np.flatnonzero(bv - r.astype(np.float64) <= ulp).tolist()
        return top, tied, ulp
    r = [float(x) for x in row[:lim]]
    bv = max(r)
    tied = [i for i, v in enumerate(r) if v == bv]
    top = tied[0]
    second = max((v for i, v in enumerate(r) if i != top), default=-math.inf)
    ulp = bf16_ulp_at(bv)
    if len(tied) == 1 and bv - second <= ulp:
        tied = [i for i, v in enumerate(r) if bv - v <= ulp]
    return top, tied, ulp


def row_accepts(row, i, used):
    top, s, _ = judge_row(row, used)
    return i == top if len(s) == 1 else i in s


def misses(j, force, used, graded):
    """([(row, f32 top, the forced id, its distance below the top in bf16 ulps of the top)] of the rows < graded where
    the judge rejects force[row], the same of the rows from graded on - past the stream's end of turn, not graded)
    (spec_judge.h misses())."""
    out, past = [], []
    for r, fid in enumerate(force):
        row = j.row(r)
        top, s, ulp = judge_row(row, used)
        if (fid == top) if len(s) == 1 else (fid in s):
            continue
        (out if r < graded else past).append((r, top, fid, (float(row[top]) - float(row[fid])) / ulp))
    return out, past


# ---- the end of turn (golden_common.h turn_end, item 30 (3), operator 2026-11-15) -------------------------------
def turn_end(ids, stops):
    """The graded rows of a generated stream: [0, turn_end) - up to and including its first end-of-turn id."""
    return next((i + 1 for i, x in enumerate(ids) if x in stops), len(ids))


def end_of_turn_ids(a):
    """(ids, source): the model's eos list - --snapshot's generation_config.json eos_token_id (an integer or a list),
    else the `eos [..]` list b70-serve printed at startup in the arms' server.log files (one list, or an error)."""
    if a.snapshot:
        p = os.path.join(a.snapshot, "generation_config.json")
        with open(p, encoding="utf-8") as f:
            e = json.load(f)["eos_token_id"]
        return set(e if isinstance(e, list) else [e]), p
    found = {}
    for log in glob.glob(os.path.join(a.dir, "**", "server.log"), recursive=True):
        with open(log, encoding="utf-8", errors="replace") as f:
            m = re.search(r"\beos \[([0-9, ]+)\]", f.read())
        if m:
            found.setdefault(tuple(sorted(int(x) for x in m.group(1).split(","))), log)
    if len(found) > 1:
        raise SystemExit(f"serve_judge: the arms' server.log files print different eos lists: {sorted(found)}")
    if found:
        (ids, log), = found.items()
        return set(ids), log
    return set(), ""


# ---- spec_judge.h kReviewedRows --------------------------------------------------------------------------------
def reviewed_rows(header):
    """[(key, row, f32 top, accepted, gap, date)] from spec_judge.h's kReviewedRows table."""
    try:
        with open(header, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return []
    m = re.search(r"kReviewedRows\[\]\s*=\s*\{(.*?)\n\};", text, re.S)
    if not m:
        return []
    body = "\n".join(line.split("//")[0] for line in m.group(1).splitlines())
    return [(k, int(r), int(a), int(b), float(g), d) for k, r, a, b, g, d in
            re.findall(r'\{\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*,\s*"([^"]+)"', body)]


def key_tag(key):
    """spec_judge.h key_tag: the key without its "_<16 hex>" hash."""
    return key[:-17] if len(key) > 17 and key[-17] == "_" else key


def match_reviewed(reviewed, key, row, a, b, g32):
    """spec_judge.h match_reviewed: the entry of exactly (key, row, f32 top, pick); else (operator 2026-11-07) the
    entry of the same test tag, row and ids whose f32 gap is within REVIEWED_GAP_ULP of g32. (entry, by_gap)."""
    for e in reviewed:
        if e[0] == key and e[1] == row and e[2] == a and e[3] == b:
            return e, False
    for e in reviewed:
        if (key_tag(e[0]) == key_tag(key) and e[1] == row and e[2] == a and e[3] == b
                and abs(g32 - e[4]) <= REVIEWED_GAP_ULP):
            return e, True
    return None, False


# ---- one request pair ------------------------------------------------------------------------------------------
def committed_ids(rec, root):
    sub = {"golden": "prompts", "toolcall": "toolcall"}.get(rec["set"])
    return serve_client.read_ids(os.path.join(root, "tests", "golden", sub, rec["name"] + ".ids")) if sub else None


def prompt_of(rec, root):
    """The prompt ids the server encoded (the record's prompt_ids), or the set's committed .ids when the server
    reported a match; None when neither is known."""
    if rec.get("prompt_ids"):
        return list(rec["prompt_ids"])
    if rec.get("prompt_ids_match") is True:
        return committed_ids(rec, root)
    return None


class Verdict:
    PASS, FAIL, SKIP, TURN = "PASS", "FAIL", "SKIP", "TURN"   # TURN: identical to the shared end of turn


def judge_pair(r0, r, a, cache, reviewed):
    """(verdict, text, awaiting keys) for the ref record r0 against the arm's record r (already known to differ)."""
    x, y = r0["out_ids"], r["out_ids"]
    n = min(len(x), len(y))
    d = next((i for i in range(n) if x[i] != y[i]), n)
    where = f"at index {d} ({len(y)} vs {len(x)} ids)"
    ex = turn_end(x, a.stops)
    if ex <= d:
        return Verdict.TURN, (f"{where}: past end of turn (row {d}): not graded - the streams are identical through "
                              f"their shared end-of-turn id at {ex - 1}"), []
    if d == n:
        return Verdict.FAIL, f"{where}: one stream is a prefix of the other - a length difference, not judged: FAIL", []
    p0, p1 = prompt_of(r0, a.root), prompt_of(r, a.root)
    if p0 is None or p1 is None:
        return Verdict.SKIP, (f"{where}: no prompt ids recorded (serve_client.py run's prompt_ids) and the "
                              f"server's tokenisation does not match the committed .ids - the context is unknown, "
                              f"not judged"), []
    if p0 != p1:
        return Verdict.FAIL, f"{where}: the two arms' prompt ids differ - not like for like: FAIL", []
    ctx = p0 + list(x[:d])
    fp, fs = list(x[d:n]), list(y[d:n])
    tag = f"{a.tag}_{r0['set']}_{r0['name']}"
    kp, ks = forced_key(tag, ctx, fp), forced_key(tag, ctx, fs)
    dump(a.dump, kp, ctx, fp)
    dump(a.dump, ks, ctx, fs)
    tail = f" (ids [{n}, {max(len(x), len(y))}) of the longer stream have no counterpart: not judged)" \
        if len(x) != len(y) else ""
    # each stream graded to its own end of turn: the judge rows [0, graded) of its suffix from d
    gp, gs = min(ex, n) - d, min(turn_end(y, a.stops), n) - d
    if (kp, ks) not in cache:
        cache[(kp, ks)] = judge_context(a, kp, ks, fp, fs, gp, gs, reviewed)
    v, text, keys = cache[(kp, ks)]
    return v, f"{where}: {text}{tail}", keys


def judge_context(a, kp, ks, fp, fs, gp, gs, reviewed):
    """(verdict, text, awaiting keys) of one forced context pair: the plain suffix fp (key kp), the speculative fs,
    each graded on its rows [0, gp) / [0, gs) - to its end of turn."""
    paths = {k: os.path.join(a.judge, k + ".golden.safetensors") for k in (kp, ks)}
    absent = [k for k in (kp, ks) if not os.path.exists(paths[k])]
    if absent:
        return (Verdict.SKIP, f"no f32 reference for {', '.join(absent)} in {a.judge} - AWAITING f32 reference "
                              f"(not judged, not passed)", absent)
    try:
        jp, js = Judge(paths[kp], fp), Judge(paths[ks], fs)
    except (ValueError, KeyError) as e:
        return Verdict.FAIL, f"a reference does not fit its context: {e}: FAIL", []
    used = min(a.vocab_used, jp.V)
    (mp, pp), (ms, ps) = misses(jp, fp, used, gp), misses(js, fs, used, gs)
    worst = max((m[3] for m in ms), default=0.0)
    ok = len(ms) <= len(mp) or worst <= NEAR_MISS_ULP
    lines = []
    if not ok:   # more misses than plain, one beyond the near-miss bar: each such miss by the bf16-unstable rule
        bt_path = os.path.join(a.bf16, ks + ".bf16top.safetensors") if a.bf16 else ""
        bt = Bf16Top(bt_path) if bt_path and os.path.exists(bt_path) else None
        ok = True
        for row, top, fid, g32 in ms:
            if g32 <= NEAR_MISS_ULP:
                continue
            ulp = bf16_ulp_at(js.row(row)[top])
            g16 = (bt.at(row, top) - bt.at(row, fid)) / ulp if bt else math.nan
            if math.isnan(g16):
                why = "no bf16 reference - the strict bar 8"
            elif g32 - g16 <= NEAR_MISS_ULP:   # one-sided (item 30 (4)): only a move toward the speculative pick
                why = (f"bf16 gap {g16:.2f} ulp - bf16 stable (moved {g32 - g16:.2f} toward the speculative pick, "
                       f"<= 8): the strict bar")
            else:
                why = (f"bf16 gap {g16:.2f} ulp - bf16-UNSTABLE, but no plain row (b70-serve returns no logits): "
                       f"the strict bar")
            rv, by_gap = match_reviewed(reviewed, ks, row, top, fid, g32)
            if not math.isnan(g16) and g32 - g16 > NEAR_MISS_ULP and g16 < 0:
                # operator 2026-11-08 (A): bf16-unstable and the standard bf16 model itself picks the speculative id -
                # a precision flip, passed without the plain row b70-serve cannot return
                lines.append(f"row {row}: ids {top} (f32 top) / {fid} (speculative): f32 gap {g32:.2f} ulp; bf16 gap "
                             f"{g16:.2f} ulp - bf16-UNSTABLE, the bf16 reference picks the speculative id: PASS")
            elif rv:
                lines.append(f"row {row}: ids {top} (f32 top) / {fid} (speculative): f32 gap {g32:.2f} ulp; {why} - "
                             f"REVIEWED (operator {rv[5]}: {rv[0]}, {rv[4]:.2f} ulp when reviewed"
                             f"{', matched by tag / row / ids / f32 gap' if by_gap else ''}): PASS as REVIEWED")
            else:
                ok = False
                lines.append(f"row {row}: ids {top} (f32 top) / {fid} (speculative): f32 gap {g32:.2f} ulp; {why}: "
                             f"FAIL")
    v = Verdict.PASS if ok else Verdict.FAIL
    text = (f"the f32 judge rejects {len(ms)} speculative / {len(mp)} plain ids from there (worst "
            f"speculative miss {worst:.2f} ulp) - {v}")
    if gs < len(fs) or gp < len(fp):
        text += f"; graded to each stream's end of turn: {gs} speculative / {gp} plain of {len(fs)} rows"
    for ln in lines:
        text += "\n      ** the bf16-unstable rule (operator 2026-11-04): " + ln
    for who, past in (("plain", pp), ("speculative", ps)):
        for row, top, fid, g in past:
            text += (f"\n      past end of turn (row {row}): not graded - the judge rejects the {who} id {fid} there, "
                     f"{g:.2f} bf16 ulp below its top {top}")
    return v, text, []


def run(a):
    recs = serve_client.records(a.dir)
    by = {}
    for r in recs:
        by.setdefault((r["label"], r["round"]), {})[(r["set"], r["name"])] = r
    refs = {k: v for k, v in by.items() if k[0] == a.ref}
    if not refs:
        print(f"serve_judge: no requests of arm '{a.ref}' under {a.dir}")
        return 1
    ref = refs[min(refs)]
    a.stops, src = end_of_turn_ids(a)
    print(f"serve_judge: end of turn: ids {sorted(a.stops)} ({src}) - each stream graded up to and including its "
          f"first one" if a.stops else
          "serve_judge: end of turn: no eos list (no --snapshot, no server.log 'eos [..]' line) - every row graded")
    reviewed = reviewed_rows(a.header)
    cache = {}
    fail = skip = 0
    awaiting = set()
    for (label, rnd), reqs in sorted(by.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
        if label == a.ref:
            continue
        c = {Verdict.PASS: 0, Verdict.FAIL: 0, Verdict.SKIP: 0, Verdict.TURN: 0}
        same = 0
        for key, r0 in sorted(ref.items()):
            r = reqs.get(key)
            if r is None:
                c[Verdict.FAIL] += 1
                print(f"MISSING {label} r{rnd} {key[0]}/{key[1]}: FAIL")
                continue
            if r0["out_ids"] == r["out_ids"]:
                same += 1
                continue
            v, text, keys = judge_pair(r0, r, a, cache, reviewed)
            c[v] += 1
            awaiting.update(keys)
            word = "SKIP" if v == Verdict.SKIP else "TURN" if v == Verdict.TURN else "JUDGE"
            print(f"{word} {label} r{rnd} vs {a.ref} {key[0]}/{key[1]} {text}")
        fail += c[Verdict.FAIL]
        skip += c[Verdict.SKIP]
        print(f"JUDGED {label} r{rnd} vs {a.ref}: {len(ref)} requests - {same} identical, {c[Verdict.PASS]} differ "
              f"and pass the f32 judge, {c[Verdict.FAIL]} FAIL, {c[Verdict.SKIP]} SKIPPED (not judged), "
              f"{c[Verdict.TURN]} differ only past the end of turn (not graded)")
    if awaiting:
        print(f"serve_judge: the contexts are dumped to {a.dump}; tools/box_validate/f32_refs.sh <data> forced-qwen "
              f"(QWEN_KEYS_RE='^{a.tag}_') makes their references into {a.judge}")
        for k in sorted(awaiting):
            print(f"    awaiting f32 reference: {k}")
    if skip:
        print(f"SKIP_REASON serve_judge: {skip} differing request(s) not judged ({len(awaiting)} context(s) await "
              f"their f32 reference; the SKIP lines say why each)")
    print(f"serve_judge: {fail} FAIL, {skip} SKIPPED")
    return 1 if fail else (77 if skip else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="the serve_arms directory (requests.jsonl per arm and round)")
    ap.add_argument("--ref", required=True, help="the plain arm (--spec off)")
    ap.add_argument("--dump", required=True, help="where the forced contexts are written")
    ap.add_argument("--judge", required=True, help="the f32 references (<key>.golden.safetensors)")
    ap.add_argument("--bf16", default="", help="the bf16 top-k references (<key>.bf16top.safetensors)")
    ap.add_argument("--tag", default="serve_int8", help="the key's test tag (carries the head form)")
    ap.add_argument("--header", default=HEADER, help="spec_judge.h (its kReviewedRows)")
    ap.add_argument("--snapshot", default="", help="the model's snapshot (its generation_config.json eos_token_id is "
                    "the end of turn); default: the eos list in the arms' server.log")
    ap.add_argument("--vocab-used", type=int, default=VOCAB_USED)
    ap.add_argument("--root", default=ROOT)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
