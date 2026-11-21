"""Tests for pool_fit.py on synthetic request logs.

    python3 tools/prefix/test_pool_fit.py   (or python3 -m pytest tools/prefix)
"""
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_log import load  # noqa: E402
from pool_fit import analyze, main, parse, verdict  # noqa: E402


def rec(t0, t1, prompt, out):
    return {"t_start": t0, "t_first_token": t0 + 0.1, "t_end": t1, "endpoint": "/v1/chat/completions",
            "request": {}, "prompt_ids": prompt, "out_ids": out, "response_text": ""}


def ids(n, base=0):
    return [base + i for i in range(n)]


def labels(runs, model, kv):
    o = runs[(model, kv)][0]
    return {n: ("cut" if x["cut"] else "refused" if x["refused"] else "waited" if x["wait"] > 0 else "ok")
            for n, x in o.items()}


def test_serial_short_fits_everything():
    a = parse(["x"])
    log = [rec(0, 1, ids(1000), ids(100, 10**6)), rec(2, 3, ids(1500), ids(100, 10**6))]
    reqs, runs = analyze(log, a)
    for m in runs:
        assert set(labels(runs, *m).values()) == {"ok"}, m
    assert verdict([(reqs, runs)]).startswith("stop")


def test_long_beside_short_bf16_only():
    # A 100k prompt beside two short ones: bf16 slot 0 (46,080) refuses it, int8 slot 0 and both pools serve it.
    a = parse(["x"])
    log = [rec(0, 10, ids(100000), ids(500, 10**6)),
           rec(1, 2, ids(3000, 10**7), ids(50, 10**6)),
           rec(1, 2, ids(3000, 2 * 10**7), ids(50, 10**6))]
    reqs, runs = analyze(log, a)
    assert labels(runs, "fixed", "bf16")[1] == "refused"
    assert labels(runs, "pool", "bf16")[1] == "ok"
    assert set(labels(runs, "fixed", "int8").values()) == {"ok"}
    assert set(labels(runs, "pool", "int8").values()) == {"ok"}
    assert verdict([(reqs, runs)]).startswith("decide")


def test_two_long_int8_go():
    # Two 60k requests at once: only slot 0 holds one in int8, the other waits; the int8 pool serves both.
    a = parse(["x"])
    log = [rec(0, 10, ids(60000), ids(100, 10**6)), rec(1, 11, ids(60000, 10**7), ids(100, 10**6))]
    reqs, runs = analyze(log, a)
    assert labels(runs, "fixed", "int8") == {1: "ok", 2: "waited"}
    assert runs[("fixed", "int8")][0][2]["wait"] == 9
    assert labels(runs, "pool", "int8") == {1: "ok", 2: "ok"}
    assert verdict([(reqs, runs)]).startswith("go")


def test_growth_cut_when_pool_full():
    # 2 sequences, 6 pages: request 1 takes 2 pages at admission, request 2 four; request 1's growth into a
    # third page finds none free and nothing idle - it is cut where its pages end, request 2 is untouched.
    a = parse(["x", "--seqs", "2", "--pages-bf16", "6"])
    log = [rec(0, 10, ids(100), ids(5000, 10**6)), rec(1, 11, ids(5000, 10**7), ids(10, 10**6))]
    reqs, runs = analyze(log, a)
    o = runs[("pool", "bf16")][0]
    assert o[1]["cut"] and not o[2]["cut"]
    assert runs[("pool", "bf16")][1]["peak_running"] == 6


def test_idle_session_evicted_lru():
    # 3 sequences, 6 pages: requests 1 and 2 end idle (2 pages each, 1 older); request 3 needs 4 pages and
    # evicts the least recently used session only.
    a = parse(["x", "--seqs", "3", "--pages-bf16", "6"])
    log = [rec(0, 1, ids(3000), ids(10, 10**6)), rec(2, 3, ids(3000, 10**7), ids(10, 10**6)),
           rec(4, 5, ids(5000, 2 * 10**7), ids(10, 10**6))]
    reqs, runs = analyze(log, a)
    assert set(labels(runs, "pool", "bf16").values()) == {"ok"}
    st = runs[("pool", "bf16")][1]
    assert st["evictions"] == 1
    assert st["peak_all"] == 6


def test_continuation_reuses_own_pages_not_shared():
    # A session's next turn on its own sequence keeps its pages; it shares nothing with another sequence.
    a = parse(["x"])
    first = ids(10000)
    log = [rec(0, 1, first, ids(100, 10**6)), rec(2, 3, first + ids(100, 10**6) + ids(50, 10**7), ids(10, 10**6))]
    reqs, runs = analyze(log, a)
    assert runs[("pool", "int8")][0][2]["shared"] == 0
    assert runs[("pool", "int8")][1]["evictions"] == 0


def test_shared_pages_and_overlap():
    # Two requests at once with the same first 20,000 ids: the second shares 9 whole pages with the first.
    a = parse(["x"])
    common = ids(20000)
    log = [rec(0, 10, common + ids(100, 10**7), ids(50, 10**6)), rec(1, 5, common + ids(100, 2 * 10**7), ids(50, 10**6))]
    reqs, runs = analyze(log, a)
    assert runs[("pool", "bf16")][0][2]["shared"] == 9 * 2048
    assert runs[("pool", "bf16")][1]["peak_shared"] == 9
    assert "shared pages: go" in verdict([(reqs, runs)])
    with tempfile.TemporaryDirectory() as d:
        for i, r in enumerate(log, 1):
            with open(os.path.join(d, f"{i:06d}.json"), "w") as f:
                json.dump(r, f)
        assert len(load(d)) == 2
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main([d])
    text = buf.getvalue()
    assert "requests overlap in time (up to 2 at once)" in text
    assert "verdict: " in text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
