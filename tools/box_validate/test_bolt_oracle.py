#!/usr/bin/env python3
"""Host-only wrapper contract: force the golden context, publish last, resume per prompt."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest


class GoldenF32Test(unittest.TestCase):
    def test_force_and_resume(self):
        script = Path(__file__).with_name("bolt_oracle.sh")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tools/box_validate").mkdir(parents=True)
            (root / "tools/oracle").mkdir()
            (root / "tests/golden/prompts").mkdir(parents=True)
            shutil.copy(script, root / "tools/box_validate/bolt_oracle.sh")
            (root / "tests/golden/prompts/bolt_short.ids").write_text("1 2 3\n")
            snap = root / "hf/hub/models--tiny/snapshots/rev"
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}")
            (snap / "model.safetensors").touch()
            src = root / "data/oracle-out-bolt"
            src.mkdir(parents=True)
            (src / "SNAPSHOT").write_text("models--tiny/snapshots/rev\n")
            h = json.dumps({"tokens": {"dtype": "I32", "shape": [2], "data_offsets": [0, 8]}}).encode()
            (src / "bolt_short.golden.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + struct.pack("<ii", 19, 23))
            (root / "tools/oracle/bolt_ref.py").write_text("reference implementation")
            (root / "tools/oracle/bolt_mamba_ops.py").write_text("reference ops")
            # Replace only the expensive external container boundary, retaining the wrapper's real filesystem flow.
            runner = root / "tools/oracle/run_in_container.sh"
            runner.write_text('''#!/usr/bin/env python3
import os, pathlib, shlex, sys
cmd = sys.argv[1]
with open("calls.txt", "a") as f: f.write(cmd + "\\n")
args = shlex.split(cmd)
assert "--dtype" in args and args[args.index("--dtype") + 1] == "fp32"
force = args[args.index("--force") + 1].removeprefix("/ws/")
assert pathlib.Path(force).read_text().split() == ["19", "23"]
out = args[args.index("--out") + 1].removeprefix("/ws/")
log = args[[i for i, a in enumerate(args) if a == ">"][ -1] + 1].removeprefix("/ws/")
pathlib.Path(log).write_text("wall 1s, peak RSS 1 GiB\\n")
if os.environ.get("FAIL_CONTAINER"): sys.exit(1)
pathlib.Path(out).write_bytes(b"finished")
''')
            runner.chmod(0o755)
            env = dict(os.environ, HF_HOME=str(root / "hf"), SNAP_BOLT="models--tiny", SNAP_BOLT_BF16="models--absent",
                       BOLT_PROMPTS="bolt_short", BOLT_BF16_PROMPTS="bolt_short", BOLT_GEN="2", BOLT_REF_MIN_GB="0",
                       BOLT_LOCK_HELD="1", BOLT_KEEP="")
            cmd = ["bash", "tools/box_validate/bolt_oracle.sh", "data", "golden-f32"]
            result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            out = root / "data/oracle-out-bolt-f32/bolt_short.golden.safetensors"
            self.assertEqual(out.read_bytes(), b"finished")
            subprocess.run(cmd, cwd=root, env=env, check=True, capture_output=True)
            self.assertEqual(len((root / "calls.txt").read_text().splitlines()), 1)
            # A changed input invalidates a completed result rather than silently keeping it.
            (root / "tools/oracle/bolt_ref.py").write_text("reference implementation v2")
            result = subprocess.run(cmd, cwd=root, env=dict(env, FAIL_CONTAINER="1"), capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out.exists())
            self.assertEqual(len(list(out.parent.glob("stale-*/*.golden.safetensors"))), 1)


if __name__ == "__main__":
    unittest.main()
