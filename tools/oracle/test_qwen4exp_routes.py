#!/usr/bin/env python3
"""Negative controls for the f32-judged engine-route replay policy."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]
import unittest  # noqa: E402
import tempfile  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file, load_file  # noqa: E402

spec = importlib.util.spec_from_file_location("routes", os.path.join(HERE, "qwen4exp_routes.py"))
routes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routes)
spec = importlib.util.spec_from_file_location("align", os.path.join(HERE, "qwen4exp_align.py"))
align = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align)


class RouteReplayTest(unittest.TestCase):
    def test_capture_identity_is_required(self):
        trace = {"trace.spec": torch.tensor([1, 1, 2, 0, 2048, 256, 1, 1, 4, 4], dtype=torch.int32),
                 "trace.path": torch.tensor([0], dtype=torch.int32)}
        self.assertEqual(routes.validate_trace(trace, 1, 2, 0, True, True, 4, 4), 0)
        for field in (0, 1, 2, 3, 4, 6, 7, 8, 9):
            bad = {k:v.clone() for k,v in trace.items()}
            bad["trace.spec"][field] += 1
            with self.assertRaises(ValueError):
                routes.validate_trace(bad, 1, 2, 0, True, True, 4, 4)

    def test_actual_swapped_expert_must_be_in_window(self):
        lg = torch.tensor([4., 2., 1.999, -7.])
        self.assertEqual(routes.judge_route(lg, [0, 1], [2, 0])[0], 1)
        self.assertEqual(routes.judge_route(lg, [0, 1], [0, 3])[0], 2)
        self.assertEqual(routes.judge_route(lg, [0, 1], [1, 2])[0], 2)

    def test_multiway_cut_and_ulp_boundary(self):
        self.assertEqual(routes.judge_route(torch.tensor([4., 2.006, 2.004, 2.002, 2.]),
                                            [0, 1, 2], [0, 3, 4])[0], 1)
        for candidate, verdict in ((2.-1/64, 1), (2.-1/64-1/1024, 2)):
            self.assertEqual(routes.judge_route(torch.tensor([4., 2., candidate]), [0, 1], [0, 2])[0], verdict)
        self.assertEqual(routes.judge_route(torch.tensor([4., 2., 1.]), [0, 1], [1, 0])[0], 0)

    def test_invalid_sets_are_rejected(self):
        for ids in ([0, 0], [0, -1], [0, 4]):
            with self.assertRaises(ValueError):
                routes.judge_route(torch.tensor([4., 2., 1., 0.]), [0, 1], ids)

    def test_replay_recomputes_weights_and_keeps_bad_route_unforced(self):
        logits = torch.tensor([[4., 2., 1.999, -7.], [4., 2., 1.999, -7.]])
        p = logits.softmax(-1)
        idx = p.topk(2).indices
        weights = p.gather(-1, idx)
        weights /= weights.sum(-1, keepdim=True)
        replay = routes.RouteReplay({"route.ids.L0": torch.tensor([[2, 0], [0, 3]], dtype=torch.int32)}, 1, 2)
        result = replay.apply(0, (logits, weights, idx))
        self.assertEqual(result[2].tolist(), [[2, 0], [0, 1]])
        expected = p[0, [2, 0]] / p[0, [2, 0]].sum()
        self.assertTrue(torch.equal(result[1][0], expected))
        self.assertTrue(torch.equal(result[1][1], weights[1]))
        self.assertEqual(replay.tensors()["route.judge.L0"].tolist(), [1, 2])
        self.assertEqual(replay.bad, 1)


class PairedReferenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = self.tmp.name
        self.original = os.path.join(self.directory, "original.safetensors")
        self.output = os.path.join(self.directory, "paired.safetensors")
        base = {"state.spec": torch.tensor([1, 1, 1, 2, 0, 1, 1, 1, 4], dtype=torch.int32),
                "prompt.ids": torch.tensor([1, 2], dtype=torch.int32),
                "tokens": torch.tensor([3], dtype=torch.int32), "logits": torch.ones(2, 4)}
        save_file(base, self.original)
        capture = {"prompt.ids": base["prompt.ids"].long(), "tokens": base["tokens"].long(),
                   "trace.spec": torch.tensor([1, 1, 2, 1, 2048, 256, 1, 1, 4, 4], dtype=torch.int32)}
        for mode in ("prefill", "decode"):
            capture[mode + ".route.ids.L0"] = torch.arange(10, dtype=torch.int32).repeat(3, 1)
        p = os.path.join(self.directory, "capture.safetensors")
        save_file(capture, p)
        align.split(p, self.directory)
        for mode in ("prefill", "decode"):
            t = {k: v.clone() for k, v in base.items()}
            t["state.spec"][0] = 3
            t["route.judge.L0"] = torch.tensor([0, 1, 2], dtype=torch.int32)
            trace = load_file(os.path.join(self.directory, mode + ".routes.safetensors"))
            t.update({"engine.trace.spec": trace["trace.spec"], "engine.trace.path": trace["trace.path"],
                      "engine.route.ids.L0": trace["route.ids.L0"]})
            save_file(t, os.path.join(self.directory, mode + ".state.safetensors"))

    def test_pair_keeps_judge_rejections_and_fixed_context(self):
        align.merge(self.original, self.directory, self.output)
        t = load_file(self.output)
        self.assertEqual(int(t["state.spec"][0]), 2)
        self.assertEqual(t["prefill.route.judge.L0"].tolist(), [0, 1, 2])
        self.assertEqual(t["decode.route.judge.L0"].tolist(), [0, 1, 2])
        self.assertTrue(torch.equal(t["tokens"], load_file(self.original)["tokens"]))

    def test_merge_rejects_stale_swapped_or_unaudited_companions(self):
        p = os.path.join(self.directory, "prefill.state.safetensors")
        original = load_file(p)
        for key in ("engine.trace.path", "engine.trace.spec", "engine.route.ids.L0", "prompt.ids", "tokens", "state.spec"):
            bad = {k: v.clone() for k, v in original.items()}
            bad[key].view(-1)[0] += 1
            save_file(bad, p)
            with self.subTest(key=key), self.assertRaises(ValueError):
                align.merge(self.original, self.directory, self.output)


if __name__ == "__main__":
    unittest.main()
