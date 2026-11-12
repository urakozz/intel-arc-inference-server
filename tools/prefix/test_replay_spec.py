#!/usr/bin/env python3
"""The speculation replay uses identical requests/cache settings, controlled children and the f32 judge."""
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_spec

FAKE = r'''
import json,os,sys,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
args=sys.argv[1:]
value=lambda k: args[args.index(k)+1]
arm=value('--spec')
logs=Path(value('--log-requests'))
(logs.parent/'argv.json').write_text(json.dumps({'args':args,'step_k':os.getenv('B70_MTP_STEP_K')}))
if arm=='mtp' and os.getenv('FAKE_FAIL_MTP'): raise SystemExit(3)
rate={'off':40,'mtp':100,'dflash':120}[arm]
print('startup adaptive per step' if arm=='mtp' and os.getenv('B70_MTP_STEP_K')!='0' else 'startup',flush=True)
class Handler(BaseHTTPRequestHandler):
 count=0
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(b'{}')
 def do_POST(self):
  Handler.count+=1
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  i=Handler.count
  ids=[10*i+1,10*i+2,10*i+3]
  if arm=='dflash' and os.getenv('FAKE_DIFF'): ids[0]=99
  t=time.monotonic()
  rec={'request':body,'prompt_ids':[1,2,3,i],'out_ids':ids,'t_start':t,'t_first_token':t+.01,
       't_end':t+.01+2/rate,'endpoint':self.path,'response_text':'ok'}
  # Completion order must not be used to pair requests: reverse the server log names.
  (logs/f'{3-i:06}.json').write_text(json.dumps(rec))
  usage={'prompt_tokens':4,'completion_tokens':3,'prompt_tokens_details':{'cached_tokens':2 if i==2 else 0}}
  frames=[{'choices':[{'delta':{'content':'ok'},'finish_reason':None}]},
          {'choices':[{'delta':{},'finish_reason':'length'}]}, {'choices':[],'usage':usage}]
  self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
  for f in frames:self.wfile.write(('data: '+json.dumps(f)+'\n\n').encode())
  self.wfile.write(b'data: [DONE]\n\n')
 def log_message(self,*a):pass
HTTPServer(('127.0.0.1',int(value('--port'))),Handler).serve_forever()
'''


class ReplaySpecTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.recording = self.root / 'session'
        self.recording.mkdir()
        for i in (1, 2):
            (self.recording / f'{i:06}.json').write_text(json.dumps({
                'endpoint': '/v1/chat/completions', 'out_ids': [1, 2, 3],
                'request': {'messages': [{'role': 'user', 'content': str(i)}], 'temperature': .8,
                            'max_tokens': 99, 'ignore_eos': True}}))
        self.bin = self.root / 'serve'
        self.bin.write_text(f'#!{sys.executable}\n' + FAKE)
        self.bin.chmod(0o755)
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            self.port = s.getsockname()[1]

    def args(self, rounds=1):
        return SimpleNamespace(log=self.recording, out=self.root / 'out', model='/target', drafter='/drafter',
                               rounds=rounds, port=self.port, wait=5,
                               judge=self.root / 'judge', bf16=self.root / 'bf16')

    def run_replay(self, args, **env):
        with patch.dict(os.environ, dict(B70_SERVE=str(self.bin), B70_MTP_STEP_K='0', **env)), \
             contextlib.redirect_stdout(io.StringIO()):
            return replay_spec.run(args)

    def test_public_entry_takes_the_whole_box_lock(self):
        with patch.object(replay_spec.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            self.assertEqual(replay_spec.main([str(self.recording), str(self.root/'out')]), 0)
        argv = run.call_args.args[0]
        self.assertEqual(argv[1:3], ['box', '--'])
        self.assertTrue(argv[0].endswith('/tools/gpu_lock.sh'))
        self.assertIn('--under-lock', argv)

    def test_three_rotated_rounds_preserve_shape_ids_and_cache(self):
        args = self.args(3)
        self.assertEqual(self.run_replay(args), 0)
        manifest = json.loads((args.out/'run.json').read_text())
        self.assertEqual(manifest['orders'], [['off','mtp','dflash'], ['mtp','dflash','off'], ['dflash','off','mtp']])
        for round_ in range(1,4):
            for arm in ('off','mtp','dflash'):
                d=args.out/f'{arm}.r{round_}'
                argv=json.loads((d/'argv.json').read_text())
                for k,v in (('--prefix-cache-gb','32'),('--max-len','131072'),('--parallel','1')):
                    self.assertEqual(argv['args'][argv['args'].index(k)+1],v)
                self.assertIsNone(argv['step_k'])
                if arm=='mtp':self.assertEqual(argv['args'][argv['args'].index('--mtp')+1],'auto')
                rows=[json.loads(s) for s in (d/'requests.jsonl').read_text().splitlines()]
                self.assertEqual([r['out_ids'] for r in rows],[[11,12,13],[21,22,23]])
                self.assertEqual([r['cached_tokens'] for r in rows],[0,2])
                self.assertAlmostEqual(rows[0]['gen_tps'],{'off':40,'mtp':100,'dflash':120}[arm],places=4)
                for f in (d/'server-log').glob('*.json'):
                    b=json.loads(f.read_text())['request']
                    self.assertEqual((b['temperature'],b['seed'],b['max_tokens']),(0,0,3))
        self.assertIn('RATIO dflash/mtp', (args.out/'table.txt').read_text())
        with self.assertRaises(OSError):socket.create_connection(('127.0.0.1',self.port),timeout=.2)

    def test_missing_f32_reference_is_pending_not_pass(self):
        self.assertEqual(self.run_replay(self.args(), FAKE_DIFF='1'),77)

    def test_legacy_replay_preserves_existing_user_without_a_prefix(self):
        p = self.recording/'000001.json'
        source = json.loads(p.read_text())
        source['request']['user'] = 'existing'
        p.write_text(json.dumps(source))
        seen = []
        def reply(url, body):
            seen.append(body)
            return {'error': None, 'usage': {'prompt_tokens': 4}, 'tool_calls': [],
                    'ttft_s': .1, 'total_s': 1, 'finish_reason': 'length'}
        with patch.object(replay_spec.replay_log, 'stream_request', side_effect=reply), \
             contextlib.redirect_stdout(io.StringIO()):
            replay_spec.replay_log.cmd_run(SimpleNamespace(log=self.recording, url='http://unused',
                                                          out=self.root/'legacy.json', seed=0, cap=0))
        self.assertEqual(seen[0]['user'], 'existing')
        self.assertNotIn('user', seen[1])

    def test_server_failure_stops_without_hanging(self):
        with self.assertRaises(RuntimeError):self.run_replay(self.args(), FAKE_FAIL_MTP='1')
        with self.assertRaises(OSError):socket.create_connection(('127.0.0.1',self.port),timeout=.2)


if __name__=='__main__':unittest.main()
