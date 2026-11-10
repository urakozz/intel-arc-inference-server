#!/usr/bin/env python3
"""Host-only dry-run checks of the capped Flash-Next reference job."""
import os
import json
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
            external_ple = Path(d) / "external-ple"
            external_ple.mkdir()
            ple.symlink_to(external_ple)
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
                        "ORACLE_PLE=" + str(external_ple.resolve()), "--ple int8:/ple"):
                self.assertIn(arg, r.stdout)
            self.assertEqual(list(Path(d).glob("oracle-out-*")), [])
            r = subprocess.run(cmd, env=dict(env, Q4_PLE="bf16"), text=True, capture_output=True)
            self.assertNotEqual(r.returncode, 0, "accepted a PLE table different from the engine's")
            cmd[-2] = "intel-f32-aligned"
            r = subprocess.run(cmd, env=dict(env, ORACLE_MEMORY="110g"), text=True, capture_output=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            for arg in ("qwen4exp_align.py split", "qwen4exp_align.py merge", "prefill.routes.safetensors",
                        "decode.routes.safetensors", "--engine-routes", "--dtype fp32", "--lm-head int8",
                        "--layers 18", "--cycle 4500", "--gen 32", "ORACLE_MEMORY=64g",
                        "ORACLE_PLE=" + str(external_ple.resolve())):
                self.assertIn(arg, r.stdout)
            self.assertEqual(list(Path(d).glob("oracle-out-*")), [])

    def test_container_mounts_external_ple(self):
        # HF snapshots on the box link their PLE sibling outside the cache mount.
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            snap, ple, binaries = d / "snap", d / "external-ple", d / "bin"
            for p in (snap, ple, binaries):
                p.mkdir()
            (snap / "config.json").write_text("{}")
            (ple / "model.safetensors.index.json").write_text("{}")
            docker = binaries / "docker"
            docker.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
            docker.chmod(0o755)
            env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                       ORACLE_SNAP=str(snap), ORACLE_PLE=str(ple), ORACLE_MEMORY="64g")
            r = subprocess.run(["bash", str(ROOT / "tools/oracle/run_in_container.sh"), "true"], env=env,
                               text=True, capture_output=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            args = json.loads(r.stdout)
            self.assertIn(str(ple) + ":/ple:ro", args)
            self.assertEqual(args[args.index("--memory") + 1], "64g")
            self.assertEqual(args[args.index("--memory-swap") + 1], "64g")


if __name__ == "__main__":
    unittest.main()
