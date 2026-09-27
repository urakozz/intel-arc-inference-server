#!/usr/bin/env python3
"""Checks for mtp_ref.py (plan 8a Task 1 Step 2). Head only, no main model: seconds.

    test_mtp_ref.py <snapshot>

1. the 15 `mtp.*` tensors: names, shapes and dtypes against docs/03-models.md,
   and their total size (0.849 GB);
2. a 16-token smoke run of a depth-3 chain: every logit finite;
3. the explicit attention in MtpHead.step against transformers' own decoder
   layer (eager, causal mask) on the same inputs: depth 1 over all rows, and
   one chained depth-2 row as the sequence [depth-1 rows 0..t, depth-2 row]
   at positions [0..t, t+1].
Run in the reference container (tools/oracle/run_in_container.sh).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import importlib.util  # noqa: E402

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("mtp_ref", os.path.join(_HERE, "mtp_ref.py"))
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)


def layer_ref(head, x_in, pos):
    """transformers' layer + mtp.norm on a [T, H] fc output, positions pos, causal."""
    T = x_in.shape[0]
    pid = pos.view(1, 1, T).expand(3, 1, T)
    pe = head.rotary(x_in, pid)
    mask = torch.full((T, T), float("-inf")).triu(1).to(x_in.dtype)[None, None]
    y = head.layer(x_in[None], position_embeddings=pe, attention_mask=mask)
    return head.norm(y[0])


def fc_in(head, ids, h):
    return torch.cat([head.pre_e(head.embed[ids]), head.pre_h(h)], -1) @ head.fc.t()


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(a @ b / (a.norm() * b.norm()))


@torch.no_grad()
def main() -> None:
    snap = sys.argv[1]
    t = ref.read_tensors(snap, list(ref.MTP_SHAPES))
    assert len(t) == 15, len(t)
    nbytes = 0
    for name, shape in ref.MTP_SHAPES.items():
        assert tuple(t[name].shape) == shape, (name, tuple(t[name].shape), shape)
        assert t[name].dtype == torch.bfloat16, (name, t[name].dtype)
        nbytes += t[name].numel() * 2
    print(f"PASS 15 mtp tensors, shapes and bf16 as doc 03, {nbytes / 1e9:.3f} GB")
    assert abs(nbytes / 1e9 - 0.849) < 0.001

    head = ref.MtpHead.from_snapshot(snap)
    g = torch.Generator().manual_seed(0)
    T = 16
    ids = torch.randint(0, 200000, (T,), generator=g)
    h = (torch.randn(T, 5120, generator=g) * 2).to(torch.bfloat16)
    pos = torch.arange(T)
    out = head.chain(h, ids, pos, 3)
    for d, (lg, y) in enumerate(out, 1):
        assert torch.isfinite(lg.float()).all() and torch.isfinite(y.float()).all(), d
    print(f"PASS 16-token depth-3 chain finite; depth-1 logits {tuple(out[0][0].shape)}")

    y1 = out[0][1]
    y1_ref = layer_ref(head, fc_in(head, ids, h), pos)
    c1 = cos(y1, y1_ref)
    print(f"depth-1 hidden vs transformers layer: cosine {c1:.6f}, "
          f"max abs {float((y1.float() - y1_ref.float()).abs().max()):.4f}")
    assert c1 > 0.9999, c1

    # depth 2, row r: the sequence [depth-1 inputs 0..r, depth-2 input of row r]
    feed = torch.randint(0, 200000, (T,), generator=g)
    out2 = head.chain(h, ids, pos, 2, feed_ids=[feed])
    for r in (0, 7, 15):
        x1 = fc_in(head, ids[: r + 1], h[: r + 1])
        x2 = fc_in(head, feed[r:r + 1], out2[0][1][r:r + 1])
        yr = layer_ref(head, torch.cat([x1, x2]), torch.cat([pos[: r + 1], pos[r:r + 1] + 1]))[-1]
        c2 = cos(out2[1][1][r], yr)
        print(f"depth-2 row {r:>2} vs transformers layer on the chained sequence: cosine {c2:.6f}")
        assert c2 > 0.9999, (r, c2)
    print("ALL PASS")


if __name__ == "__main__":
    main()
