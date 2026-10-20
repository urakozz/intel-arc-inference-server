#!/usr/bin/env python3
"""Tests for agentic_loop.py against a scripted server (no model): python3 tools/prefix/test_agentic_loop.py"""
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentic_loop  # noqa: E402

TRANSCRIPT = {
    "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
    "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "c1", "content": "RESULT-1"},
        {"role": "assistant", "content": "x"},
        {"role": "tool", "tool_call_id": "c2", "content": "RESULT-2"},
        {"role": "user", "content": "second"},
    ],
}


def frame(delta=None, finish=None, usage=None):
    j = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}] if delta is not None or finish else []}
    if usage:
        j["usage"] = usage
    return "data: " + json.dumps(j) + "\n\n"


class Scripted(BaseHTTPRequestHandler):
    bodies = []
    script = []   # per request: (list of deltas, finish, usage)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Scripted.bodies.append(body)
        deltas, finish, usage = Scripted.script[len(Scripted.bodies) - 1]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        out = frame({"role": "assistant"})
        for d in deltas:
            out += frame(d)
        out += frame({}, finish) + frame(usage=usage) + "data: [DONE]\n\n"
        self.wfile.write(out.encode())

    def log_message(self, *args):
        pass


def call_delta(i, name, args, cid):
    return {"tool_calls": [{"index": i, "id": cid, "type": "function", "function": {"name": name, "arguments": args}}]}


class AgenticLoop(unittest.TestCase):
    def setUp(self):
        Scripted.bodies = []
        self.httpd = HTTPServer(("127.0.0.1", 0), Scripted)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        self.dir = tempfile.mkdtemp()
        self.transcript = os.path.join(self.dir, "t.json")
        json.dump(TRANSCRIPT, open(self.transcript, "w"))
        self.out = os.path.join(self.dir, "out.json")

    def tearDown(self):
        self.httpd.shutdown()

    def run_loop(self, script, turns=12):
        Scripted.script = script
        agentic_loop.main(["run", self.transcript, self.url, self.out, "--turns", str(turns)])
        return json.load(open(self.out))

    def test_loop_feeds_back_the_models_own_turns(self):
        usage = lambda p, c, cached: {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
                                      "prompt_tokens_details": {"cached_tokens": cached}}
        script = [
            # turn 1: reasoning, a call BEFORE its text (the order the template cannot re-render)
            ([{"reasoning_content": "think "}, call_delta(0, "read", '{"path":"/a"}', "call_A"), {"content": "text after"}],
             "tool_calls", usage(100, 11, 0)),
            # turn 2: a plain call, two calls in one turn
            ([call_delta(0, "read", '{"path":"/b"}', "call_B"), call_delta(1, "read", '{"path":"/c"}', "call_C")],
             "tool_calls", usage(160, 7, 96)),
            # turn 3: an answer with no call: the next recorded user message follows
            ([{"content": "done"}], "stop", usage(220, 3, 192)),
            # turn 4: an answer again, nothing left to say: the loop stops
            ([{"content": "bye"}], "stop", usage(260, 2, 256)),
        ]
        runs = self.run_loop(script)
        self.assertEqual(len(runs), 4)
        b = Scripted.bodies
        self.assertEqual([m["role"] for m in b[0]["messages"]], ["system", "user"])
        self.assertEqual(b[0]["temperature"], 0)
        self.assertEqual(b[0]["tools"], TRANSCRIPT["tools"])
        # turn 2 holds the model's turn exactly as returned, then one tool result for its one call
        m = b[1]["messages"]
        self.assertEqual([x["role"] for x in m], ["system", "user", "assistant", "tool"])
        self.assertEqual(m[2]["content"], "text after")
        self.assertEqual(m[2]["reasoning_content"], "think ")
        self.assertEqual(m[2]["tool_calls"][0]["id"], "call_A")
        self.assertEqual(m[2]["tool_calls"][0]["function"], {"name": "read", "arguments": '{"path":"/a"}'})
        self.assertEqual((m[3]["tool_call_id"], m[3]["content"]), ("call_A", "RESULT-1"))
        # turn 3: two calls -> two results, in order, the second recorded result next
        m = b[2]["messages"]
        self.assertEqual([x["role"] for x in m][-3:], ["assistant", "tool", "tool"])
        self.assertEqual([x["content"] for x in m[-2:]], ["RESULT-2", "RESULT-1"])   # the results are taken in order, wrapping within a turn
        self.assertEqual([x["tool_call_id"] for x in m[-2:]], ["call_B", "call_C"])
        # turn 4: the answer had no call -> the next recorded user message
        m = b[3]["messages"]
        self.assertEqual((m[-1]["role"], m[-1]["content"]), ("user", "second"))
        self.assertIsNone(m[-2]["tool_calls"] if "tool_calls" in m[-2] else None)
        self.assertEqual(m[-2]["content"], "done")
        # the records
        self.assertEqual([r["cached_tokens"] for r in runs], [0, 96, 192, 256])
        self.assertEqual([r["reordered"] for r in runs], [True, False, False, False])
        self.assertEqual([r["calls_n"] for r in runs], [1, 2, 0, 0])
        self.assertEqual(runs[0]["prompt_tokens"], 100)

    def test_stops_when_the_transcript_is_spent(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 0}}
        script = [([{"content": "a"}], "stop", usage), ([{"content": "b"}], "stop", usage),
                  ([{"content": "c"}], "stop", usage)]
        runs = self.run_loop(script)
        self.assertEqual(len(runs), 2)   # one user message left after the first, then nothing

    def test_report(self):
        usage = lambda p, c, cached: {"prompt_tokens": p, "completion_tokens": c, "prompt_tokens_details": {"cached_tokens": cached}}
        script = [([call_delta(0, "read", "{}", "k1")], "tool_calls", usage(100, 5, 0)),
                  ([{"content": "ok"}], "stop", usage(150, 3, 100))]
        self.run_loop(script, turns=2)
        log = os.path.join(self.dir, "log")
        os.mkdir(log)
        # the server's request log: turn 2's prompt shares 120 ids with turn 1's prompt + output
        json.dump({"prompt_ids": list(range(100)), "out_ids": list(range(1000, 1005))}, open(os.path.join(log, "000001.json"), "w"))
        json.dump({"prompt_ids": list(range(100)) + [1000, 1001, 1002, 7, 7], "out_ids": [1]}, open(os.path.join(log, "000002.json"), "w"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            agentic_loop.main(["report", self.out, log])
        text = buf.getvalue()
        self.assertIn("hit rate over the session: 100/250 = 40.0 %", text)
        self.assertIn("hit rate of the turns that followed another: 66.7 %", text)
        self.assertIn("ideal prefix reuse (analyze_log.py's rule): 103 of 250", text)
        self.assertIn("in another order than generated: 0 of 2", text)


if __name__ == "__main__":
    unittest.main()
