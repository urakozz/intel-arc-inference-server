#!/usr/bin/env python3
"""Normal DFlash validation selects the serving arm; the real probe CLI rejects bad counts before a device."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class DFlashModesTest(unittest.TestCase):
    def assert_serving_mode(self, argv):
        self.assertIn('--format', argv)
        self.assertEqual(argv[argv.index('--format')+1], 'int8')
        self.assertIn('--candidate-head', argv)

    def test_cost_stage_uses_int8_candidate_at_both_depths(self):
        shell = '''row() { :; }; stage() { :; }; rownote() { :; }
source tools/box_validate/stages.sh
SNAP_QWEN=/target; SNAP_DFLASH=/drafter
idle() { :; }; grab_all() { :; }; finish() { :; }; chk() { printf '%s\\n' "$1"; }
st_r82_iter
'''
        r = subprocess.run(['bash','-c',shell],cwd=ROOT,capture_output=True,text=True,check=True)
        commands = [shlex.split(line) for line in r.stdout.splitlines()]
        self.assertEqual(len(commands), 2)
        for argv, depth in zip(commands, ('4096','32768')):
            self.assert_serving_mode(argv)
            self.assertEqual(argv[argv.index('--depth')+1], depth)

    def test_acceptance_helper_uses_int8_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = root/'tools/box_validate'; tools.mkdir(parents=True)
            for name in ('dflash_v2.sh','data.sh'):shutil.copy2(ROOT/'tools/box_validate'/name,tools/name)
            probe = root/'build/tools/probe/probe_dflash_iter';probe.parent.mkdir(parents=True)
            probe.write_text(f'#!{sys.executable}\nimport json,os,sys\nopen(os.environ["PROBE_ARGS_LOG"],"w").write(json.dumps(sys.argv[1:]))\n')
            probe.chmod(0o755)
            target, drafter = root/'target', root/'drafter';target.mkdir();drafter.mkdir()
            output = root/'args.json'
            subprocess.run([str(tools/'dflash_v2.sh'),'gpu',str(root/'out'),str(target),str(drafter)],
                           env={**os.environ,'PROBE_ARGS_LOG':str(output)},check=True,capture_output=True)
            self.assert_serving_mode(json.loads(output.read_text()))

    @unittest.skipUnless(os.getenv('DFLASH_PROBE'), 'set DFLASH_PROBE to the box-built probe for CLI checks')
    def test_candidate_flag_at_either_position_is_parsed_before_count_validation(self):
        for flags in (['--candidate-head','--calls','0'], ['--calls','0','--candidate-head']):
            r = subprocess.run([os.environ['DFLASH_PROBE'],'/unused-target','/unused-drafter',*flags],
                               capture_output=True,text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn('calls, rounds and anchors must be positive', r.stderr)
            self.assertNotIn('missing value', r.stderr)
            self.assertNotIn('device:', r.stderr)


if __name__ == '__main__':unittest.main()
