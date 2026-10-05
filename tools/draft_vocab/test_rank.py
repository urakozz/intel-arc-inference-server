"""Tests for rank.py on synthetic request logs and .ids files.

    python3 tools/draft_vocab/test_rank.py   (or python3 -m pytest tools/draft_vocab)
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank import count, main, rank, read_ids  # noqa: E402


def write_log(d, records):
    for i, r in enumerate(records, 1):
        with open(os.path.join(d, f"{i:06d}.json"), "w") as f:
            json.dump(r, f)


def rec(prompt, out):
    return {"t_start": 0.0, "t_first_token": 0.1, "t_end": 0.2, "endpoint": "/v1/chat/completions",
            "request": {}, "prompt_ids": prompt, "out_ids": out, "response_text": ""}


def test_weights_and_session_prompts():
    # Request 2 resends request 1's prompt + output and adds [9, 9]: only [9, 9] is new.
    with tempfile.TemporaryDirectory() as d:
        write_log(d, [rec([1, 2, 3], [5, 6]), rec([1, 2, 3, 5, 6, 9, 9], [5])])
        with open(os.path.join(d, "notes.txt"), "w") as f:
            f.write("ignored")
        counts, seen = count([d])
        assert seen["logs"] == 2 and seen["gen"] == 3
        assert seen["prompt"] == 3 + 2 and seen["prompt_skipped"] == 5
        # gen x4, new prompt x1
        assert counts == {1: 1, 2: 1, 3: 1, 5: 8, 6: 4, 9: 2}
        counts_all, _ = count([d], prompt_all=True)
        assert counts_all[1] == 2 and counts_all[5] == 9 and counts_all[9] == 2


def test_rank_order_ties_and_mask():
    counts = {7: 3, 2: 3, 9: 5, 100: 9, 4: 0}
    assert rank(counts) == [100, 9, 2, 7]            # ties to the lower id; zero dropped
    assert rank(counts, vocab_used=100) == [9, 2, 7]
    assert rank(counts, top=2) == [100, 9]


def test_ids_files_and_cli():
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.ids")
        with open(a, "w") as f:
            f.write("760 72103 506\n506\n")
        logs = os.path.join(d, "logs")
        os.mkdir(logs)
        write_log(logs, [rec([506], [11, 11])])
        out = os.path.join(d, "ranked.ids")
        assert main([a, logs, "-o", out, "--gen-weight", "2"]) == 0
        lines = open(out).read().splitlines()
        assert lines[0].startswith("# tools/draft_vocab/rank.py: 1 request logs")
        # 506: 2 (ids) + 1 (prompt); 11: 2 x 2 (gen); 760, 72103: 1 each.
        assert read_ids(out) == [11, 506, 760, 72103]
        # The C++ reader's format: '#' lines skipped, one id per line.
        assert all(s.isdigit() for s in lines[1:])


def test_empty_and_bad_input():
    with tempfile.TemporaryDirectory() as d:
        e = os.path.join(d, "empty.ids")
        open(e, "w").close()
        assert main([e]) == 1
        b = os.path.join(d, "bad.ids")
        with open(b, "w") as f:
            f.write("1 two 3\n")
        try:
            read_ids(b)
            raise AssertionError("a non-id token must be refused")
        except ValueError:
            pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
