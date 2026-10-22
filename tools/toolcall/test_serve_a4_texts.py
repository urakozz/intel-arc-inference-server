#!/usr/bin/env python3
"""Tests for serve_a4_texts.py: python3 tools/toolcall/test_serve_a4_texts.py"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_a4_texts  # noqa: E402


class ServeA4Texts(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.set = os.path.join(self.dir, "set")
        self.log = os.path.join(self.dir, "log")
        self.out = os.path.join(self.dir, "out")
        os.makedirs(self.set)
        os.makedirs(self.log)
        json.dump([{"name": "a"}, {"name": "b"}], open(os.path.join(self.set, "manifest.json"), "w"))
        open(os.path.join(self.set, "a.ids"), "w").write("1 2 3\n")
        open(os.path.join(self.set, "b.ids"), "w").write("4 5\n")

    def record(self, n, ids, text):
        json.dump({"prompt_ids": ids, "out_ids": [9], "response_text": text}, open(os.path.join(self.log, f"{n:06d}.json"), "w"))

    def test_matches_by_prompt_ids_and_keeps_the_last(self):
        self.record(1, [7], "warm-up")
        self.record(2, [1, 2, 3], "first a")
        self.record(3, [4, 5], "text bé")
        self.record(4, [1, 2, 3], "second a")
        rc = serve_a4_texts.main(["x", self.log, self.set, self.out, "serve"])
        self.assertEqual(rc, 0)
        self.assertEqual(open(os.path.join(self.out, "a.serve.txt"), encoding="utf-8").read(), "second a")
        self.assertEqual(open(os.path.join(self.out, "b.serve.txt"), encoding="utf-8").read(), "text bé")

    def test_a_scenario_the_server_never_saw_fails(self):
        self.record(1, [1, 2, 3], "only a")
        rc = serve_a4_texts.main(["x", self.log, self.set, self.out, "serve"])
        self.assertEqual(rc, 1)
        self.assertTrue(os.path.exists(os.path.join(self.out, "a.serve.txt")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "b.serve.txt")))


if __name__ == "__main__":
    unittest.main()
