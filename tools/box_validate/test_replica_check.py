#!/usr/bin/env python3
"""Replica comparison on synthetic request logs and tiny f32 judges; no model or cards."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_serve_judge import write_judge
import serve_judge as sj


class ReplicaCompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="replica_check_test_")
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.logs = os.path.join(self.root, "server-log")
        self.judge = os.path.join(self.root, "judge")
        os.makedirs(self.logs)
        os.makedirs(self.judge)
        self.keys = ["prose.cmpl.greedy", "prose.cmpl.seeded", "prose.chat.greedy"]
        self.records = [{"key": key, "ids": [4, 5], "prompt_ids": [1, 2, 3]} for key in self.keys]
        self.logs_data = [dict(request={"user": key}, card=card, slot=slot, t_start=0, t_end=10,
                               prompt_ids=[1, 2, 3])
                          for key, card, slot in zip(self.keys, (0, 0, 1), (0, 1, 3))]
        for name in ("card0", "card1"):
            self.write(name, self.records)

    def write(self, name, records):
        path = os.path.join(self.root, name + ".jsonl")
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    def run_compare(self):
        for i, r in enumerate(self.logs_data):
            with open(os.path.join(self.logs, f"{i}.json"), "w") as f:
                json.dump(r, f)
        rep = self.write("replica", self.records)
        r = subprocess.run([sys.executable, os.path.join(HERE, "replica_check.py"), "compare", "--replica", rep,
                            "--server-log", self.logs, "--card0", os.path.join(self.root, "card0.jsonl"),
                            "--card1", os.path.join(self.root, "card1.jsonl"), "--dump", os.path.join(self.root, "dump"),
                            "--judge", self.judge, "--vocab-used", "8"], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    def differ(self, index=0):
        self.records[index]["ids"] = [4, 6]

    def refs(self, bad=False):
        tag = "serve_int8_replica_prose.cmpl.greedy"
        for force in ([5], [6]):
            key = sj.forced_key(tag, [1, 2, 3, 4], force)
            rows = [(5 if bad and force == [6] else force[0], {}), (0, {})]
            write_judge(os.path.join(self.judge, key + ".golden.safetensors"), rows, force)

    def test_identical_and_card_field_overrides_slot(self):
        rc, out = self.run_compare()
        self.assertEqual(rc, 0, out)
        self.assertIn("card 1 IDENTICAL", out)

    def test_shared_card_missing_reference_skips(self):
        self.differ()
        rc, out = self.run_compare()
        self.assertEqual(rc, 77, out)
        self.assertIn("awaiting f32 reference", out)
        self.assertNotIn("R1 PASS", out)

    def test_shared_card_judge_pass_and_fail(self):
        self.differ()
        self.refs()
        rc, out = self.run_compare()
        self.assertEqual(rc, 0, out)
        self.assertIn("JUDGED", out)
        self.refs(bad=True)
        rc, out = self.run_compare()
        self.assertEqual(rc, 1, out)

    def test_isolated_card_difference_stays_bitwise(self):
        self.differ(2)
        rc, out = self.run_compare()
        self.assertEqual(rc, 1, out)
        self.assertIn("DIFFERS", out)

    def test_nonoverlapping_same_card_stays_bitwise(self):
        self.differ()
        self.logs_data[1].update(t_start=11, t_end=20)
        rc, out = self.run_compare()
        self.assertEqual(rc, 1, out)
        self.assertIn("DIFFERS", out)

    def test_legacy_slot_is_card(self):
        for r, card in zip(self.logs_data, (0, 1, 0)):
            r.pop("card")
            r["slot"] = card
        rc, out = self.run_compare()
        self.assertEqual(rc, 0, out)

    def test_missing_reply_fails(self):
        self.records.pop(0)
        rc, out = self.run_compare()
        self.assertEqual(rc, 1, out)
        self.assertIn("missing reply", out)

    def test_failure_takes_precedence_over_awaiting(self):
        self.differ()
        self.differ(2)
        rc, out = self.run_compare()
        self.assertEqual(rc, 1, out)
        self.assertIn("awaiting f32 reference", out)
        self.assertIn("R1 FAIL", out)

    def test_unknown_prompt_skips_and_mismatched_prompt_fails(self):
        self.differ()
        self.logs_data[0].pop("prompt_ids")
        rc, out = self.run_compare()
        self.assertEqual(rc, 77, out)
        self.assertIn("context is unknown", out)
        self.logs_data[0]["prompt_ids"] = [7]
        rc, out = self.run_compare()
        self.assertEqual(rc, 1, out)
        self.assertIn("prompt ids differ", out)

    def test_same_slot_overlap_stays_bitwise(self):
        self.differ()
        self.logs_data[1]["slot"] = self.logs_data[0]["slot"]
        rc, out = self.run_compare()
        self.assertEqual(rc, 1, out)
        self.assertIn("bitwise required", out)

    def test_r77_stage_preserves_an_earlier_skip(self):
        script = r"""
source tools/box_validate/lib.sh
source tools/box_validate/stages.sh
chk() { BV_OK=$((BV_OK + 1)); }
x() { case "$1" in *r1/off/*) return 77 ;; *) return 0 ;; esac; }
grab_all() { :; }
DRY=0
st_r77_r1
"""
        r = subprocess.run(["bash", "-c", script], cwd=os.path.dirname(os.path.dirname(HERE)),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)

    def test_missing_request_and_error_fail(self):
        self.records[0].pop("ids")
        self.records[0]["error"] = "request failed"
        rc, out = self.run_compare()
        self.assertEqual(rc, 1, out)


if __name__ == "__main__":
    unittest.main()
