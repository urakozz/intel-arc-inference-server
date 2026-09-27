"""Tests for analyze_log.py on a synthetic 3-request log.

    python3 tools/prefix/test_analyze_log.py   (or python3 -m pytest tools/prefix)
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_log import analyze, load  # noqa: E402


def write_log(d, records):
    for i, r in enumerate(records, 1):
        with open(os.path.join(d, f"{i:06d}.json"), "w") as f:
            json.dump(r, f)


def rec(prompt, out, endpoint="/v1/chat/completions"):
    return {"t_start": 0.0, "t_first_token": 0.1, "t_end": 0.2, "endpoint": endpoint,
            "request": {}, "prompt_ids": prompt, "out_ids": out, "response_text": ""}


def synthetic():
    # 1: main turn. 2: side request (diverges at 1). 3: main continues after 1's
    # prompt + output -- its best earlier match is request 1, not the previous one.
    r1 = rec([1, 2, 3, 4], [5, 6])
    r2 = rec([1, 9, 9], [7])
    r3 = rec([1, 2, 3, 4, 5, 6, 8, 8], [0])
    return [r1, r2, r3]


def test_rows():
    with tempfile.TemporaryDirectory() as d:
        write_log(d, synthetic())
        with open(os.path.join(d, "notes.txt"), "w") as f:
            f.write("ignored")
        rows, totals = analyze(load(d))
    assert [r["n"] for r in rows] == [1, 2, 3]
    assert [r["prompt"] for r in rows] == [4, 3, 8]
    assert [r["lcp_prev"] for r in rows] == [0, 1, 1]
    assert [r["lcp_any"] for r in rows] == [0, 1, 6]
    assert [r["best"] for r in rows] == [None, 1, 1]
    # 3's match (6) ends past request 1's prompt end (4), inside its generated ids.
    assert [r["at_prompt_end"] for r in rows] == [False, False, False]
    assert [r["in_generated"] for r in rows] == [False, False, True]
    assert totals["prompt_tokens"] == 15
    assert totals["reusable"] == 7
    assert abs(totals["fraction"] - 7 / 15) < 1e-9


def test_prompt_end_and_cap():
    # The match ends exactly at an earlier prompt's end; a prompt fully matched still
    # prefills one token (reusable = len - 1).
    a = rec([1, 2, 3], [4])
    b = rec([1, 2, 3, 7], [])
    c = rec([1, 2], [])
    rows, totals = analyze([a, b, c])
    assert rows[1]["lcp_any"] == 3 and rows[1]["at_prompt_end"]
    assert rows[2]["lcp_any"] == 2 and rows[2]["reusable"] == 1
    assert totals["reusable"] == 0 + 3 + 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
