#!/usr/bin/env python3
"""Host-only dry-run checks of the capped Flash-Next reference job."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class OracleJobTest(unittest.TestCase):
    def test_intel_f32_uses_checkpoint_and_sibling_ple(self):
        with tempfile.TemporaryDirectory() as d:
            hf = Path(d) / "hub/models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound"
            snap = hf / "snapshots/abc"
            ple = hf / "snapshots/abc-ple-int8"
            snap.mkdir(parents=True)
            ple.mkdir()
            (snap / "config.json").write_text("{}")
            (ple / "model.safetensors.index.json").write_text("{}")
            (hf / "refs").mkdir()
            (hf / "refs/main").write_text("abc")
            env = dict(os.environ, DRY_RUN="1", HF_HOME=d)
            env.pop("Q4_PLE", None)
            env.pop("SNAP_Q4EXP_INTEL", None)
            cmd = ["bash", str(ROOT / "tools/box_validate/qwen4exp_oracle.sh"), d, "intel-f32", "18"]
            r = subprocess.run(cmd, env=env, text=True, capture_output=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            for arg in ("--dtype fp32", "--cycle 4500", "--state-dump", "--layers 18", "--lm-head int8",
                        "q4exp_4k.state.safetensors", "64g", "ORACLE_SNAP=" + str(snap),
                        "--ple int8:/hf/hub/models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound/snapshots/abc-ple-int8"):
                self.assertIn(arg, r.stdout)
            self.assertEqual(list(Path(d).glob("oracle-out-*")), [])
            r = subprocess.run(cmd, env=dict(env, Q4_PLE="bf16"), text=True, capture_output=True)
            self.assertNotEqual(r.returncode, 0, "accepted a PLE table different from the engine's")


if __name__ == "__main__":
    unittest.main()
