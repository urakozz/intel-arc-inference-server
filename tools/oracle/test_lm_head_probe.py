#!/usr/bin/env python3
"""Checks for lm_head_probe.py (plan 9a Task 1 Step 1). Synthetic, no model: seconds.

    test_lm_head_probe.py

1. int8 per-row round trip: |w - q * s| <= s / 2 on every element, q in [-127, 127],
   s = max|row| / 127 in fp32, and an all-zero row survives (q = 0, s = 0);
2. argmax equality between the bf16 and the int8 head on well-separated logits;
3. KL of identical filtered distributions == 0, and the filter matches the
   engine's sample() order (top-k, softmax at temperature, top-p inclusive cut);
4. ids >= vocab_used never win the argmax, the top-20 or the filter.
Runs anywhere torch is installed (the reference container, or the Mac).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("lm_head_probe", os.path.join(_HERE, "lm_head_probe.py"))
lp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lp)


def test_round_trip():
    g = torch.Generator().manual_seed(1)
    w = (torch.randn(1000, 64, generator=g) * torch.rand(1000, 1, generator=g)).to(torch.bfloat16)
    w[7] = 0
    q, s = lp.quantise_rows(w)
    assert q.dtype == torch.int8 and s.dtype == torch.float32 and s.shape == (1000,)
    assert int(q.abs().max()) <= 127
    wf = w.float()
    assert torch.allclose(s, wf.abs().amax(1) / 127)
    # in fp64; the slack is fp32's: s is max/127 rounded (|q| <= 127 times half an ulp of s)
    # and w / s is one fp32 quotient before the rounding, so <= s * (0.5 + 127 * 2^-23)
    err = (wf.double() - q.double() * s.double()[:, None]).abs()
    assert bool((err <= s.double()[:, None] * lp.BOUND).all()), float((err / s.double()[:, None]).max())
    assert int(q[7].abs().sum()) == 0 and float(s[7]) == 0.0
    # the row maximum maps to +-127 exactly
    assert bool((q.abs().amax(1)[s > 0] == 127).all())


def test_argmax_equal():
    g = torch.Generator().manual_seed(2)
    V, H, R = 1000, 64, 32
    w = torch.randn(V, H, generator=g).to(torch.bfloat16)
    h = torch.randn(R, H, generator=g).to(torch.bfloat16)
    # make one id per row win by a wide margin: h aligned with that row of w
    tgt = torch.randint(0, 900, (R,), generator=g)
    h = (h.float() * 0.1 + 3 * w[tgt].float()).to(torch.bfloat16)
    q, s = lp.quantise_rows(w)
    ref = lp.logits_ref(h, w, chunk=128)
    qnt = lp.logits_int8(h, q, s, chunk=128)
    assert torch.equal(ref.argmax(1), tgt) and torch.equal(qnt.argmax(1), tgt)
    # chunking does not change anything
    assert torch.equal(lp.logits_int8(h, q, s, chunk=V), qnt)


def test_kl_and_filter():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(16, 500, generator=g) * 3
    sp = dict(temp=1.0, top_k=20, top_p=0.95)
    ids, p = lp.filt(x, 400, **sp)
    kl, bad = lp.kl_filtered(ids, p, ids, p)
    assert bad.sum() == 0 and float(kl.abs().max()) == 0.0
    # engine order by hand for row 0: top-k by logit, softmax at T over the k, keep while
    # the cumulative mass BEFORE the token is < top_p (cut after the first prefix >= top_p)
    v, i = torch.topk(x[0, :400], 20)
    pr = torch.softmax(v.double() / sp["temp"], -1)
    cum, kept = 0.0, 20
    for j in range(20):
        cum += float(pr[j])
        if cum >= sp["top_p"]:
            kept = j + 1
            break
    got = int((p[0] > 0).sum())
    assert got == kept, (got, kept)
    assert torch.equal(ids[0, :kept], i[:kept])
    # a q that lacks one of p's kept ids is a support mismatch (KL = inf)
    q_ids = ids.clone()
    q_ids[0, 0] = 499
    kl2, bad2 = lp.kl_filtered(ids, p, q_ids, p)
    assert bool(bad2[0]) and not bool(bad2[1:].any())


def test_vocab_mask():
    x = torch.linspace(0, 1, 300).repeat(2, 1)  # distinct, increasing: the padded ids are the largest
    x[0, 3], x[1, 5] = 5.0, 5.0
    y = x.clone()
    y[:, 250:] = -100.0                         # q differs ONLY on the padded ids
    m = lp.row_metrics(x, y, vocab_used=250, samplings={"t1": dict(temp=1.0, top_k=20, top_p=0.95)})
    assert m["argmax_ref"].tolist() == [3, 5] and m["argmax_q"].tolist() == [3, 5]
    assert bool(m["top20_equal"].all())
    assert float(m["cos"].min()) > 1 - 1e-6     # cosine over vocab_used only
    assert float(m["kl.t1"].abs().max()) == 0.0
    ids, _ = lp.filt(x, 250, temp=1.0, top_k=20, top_p=0.95)
    assert int(ids.max()) < 250


def main() -> None:
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("ALL OK")


if __name__ == "__main__":
    main()
