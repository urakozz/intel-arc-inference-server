"""W4A8 accuracy study on real captured layer-63 linear tensors (CPU, fake-quant).

Every variant is Q(x') @ Q(W') with x' = x T^-1, W' = T W for an invertible
input-side transform T (identity, SmoothQuant diag, Hadamard, random orthogonal),
so y is mathematically unchanged before quantisation. Integer GEMM with a
per-token x scale and a per-channel / per-group W scale is exactly
(xq*xs) @ (wq*ws) up to float associativity, so fake-quant in fp32 is the same
arithmetic the kernel would do (int32 accumulation vs fp32: ~1e-7 relative).

Reference: fp32 x (from bf16) @ fp32 W (the checkpoint's own g64 dequant) --
i.e. "the model as quantised by AutoRound, run exactly". Bar: per-row cosine
0.999 (tests/golden/golden_common.h kBar).

usage: python w4a8_accuracy.py <calib-capture> <eval-capture>
  captures come from `probe_w4a8 --capture` (gate||up) or `--capture-down` (down);
  calibration (SmoothQuant scales) and evaluation should come from different prompts.
  Results: docs/probe-w4a8-2026-09-23.md section 14.
"""
import sys, time
import numpy as np

HDR = 36
M, K, N, G = 2048, 5120, 34816, 64   # set per capture by load()
KEEP = None                          # rows kept for evaluation


def load(path):
    global M, K, N, G, KEEP
    raw = np.fromfile(path, dtype=np.uint8)
    assert bytes(raw[:7]) == b"B70W4A8"
    ver, layer, m, k, n, g, pad = (int(v) for v in np.frombuffer(raw[8:HDR].tobytes(), dtype=np.uint32))
    M, K, N, G = m, k, n, g
    keep = np.ones(M, dtype=bool)
    if ver == 2:  # rows clobbered by step_head's final-norm write (probe_w4a8 --capture-down)
        last = ((M - 1) * 5120 + 5120 - 1) // K
        keep[pad:last + 1] = False
    KEEP = keep
    off = HDR
    act = np.frombuffer(raw[off:off + M * K * 2].tobytes(), dtype=np.uint16).reshape(M, K)
    off += M * K * 2
    qw = np.frombuffer(raw[off:off + (K // 8) * N * 4].tobytes(), dtype=np.uint32).reshape(K // 8, N)
    off += (K // 8) * N * 4
    sc = np.frombuffer(raw[off:off + (K // G) * N * 2].tobytes(), dtype=np.float16).reshape(K // G, N)
    x = (act.astype(np.uint32) << 16).view(np.float32)[keep]
    # GPTQ layout 0: nibble i of qweight[r][n] is k = 8r+i; symmetric, zero 8
    q = np.empty((K, N), dtype=np.int8)
    for i in range(8):
        q[i::8] = ((qw >> (4 * i)) & 0xF).astype(np.int8) - 8
    W = q.astype(np.float32) * np.repeat(sc.astype(np.float32), G, axis=0)
    return x, W, q, sc


def bf16_round(a):
    b = a.astype(np.float32).view(np.uint32)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.view(np.float32)


def q_act_token(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(x).max(axis=1, keepdims=True) / qmax
    s[s == 0] = 1
    return np.clip(np.rint(x / s), -qmax, qmax) * s


def q_w_channel(W, bits):
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(W).max(axis=0, keepdims=True) / qmax
    s[s == 0] = 1
    return (np.clip(np.rint(W / s), -qmax - 1, qmax) * s).astype(np.float32)


def q_w_group(W, bits, g):
    qmax = 2 ** (bits - 1) - 1
    Wg = W.reshape(K // g, g, N)
    s = np.abs(Wg).max(axis=1, keepdims=True) / qmax
    s[s == 0] = 1
    return (np.clip(np.rint(Wg / s), -qmax - 1, qmax) * s).reshape(K, N).astype(np.float32)


def hadamard(n):
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def block_rot(b, seed):
    """Randomised block-diagonal Hadamard, orthogonal: R = D * blockdiag(H_b)."""
    rng = np.random.default_rng(seed)
    d = rng.choice([-1.0, 1.0], size=K).astype(np.float32)
    H = hadamard(b)
    def fwd_x(x):  # x R
        y = (x * d).reshape(x.shape[0], K // b, b) @ H
        return y.reshape(x.shape[0], K)
    def fwd_w(W):  # R^T W
        Wb = (W * d[:, None]).reshape(K // b, b, N)
        return np.einsum("ij,gjn->gin", H.T, Wb, optimize=True).reshape(K, N)
    return fwd_x, fwd_w


def metrics(y, ref):
    diff = y - ref
    rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(ref))
    cos = (y * ref).sum(1) / (np.linalg.norm(y, axis=1) * np.linalg.norm(ref, axis=1))
    top1 = int((y.argmax(1) != ref.argmax(1)).sum())
    k = 8
    ti = np.argpartition(-y, k, axis=1)[:, :k]
    tr = np.argpartition(-ref, k, axis=1)[:, :k]
    ov = np.mean([len(set(a) & set(b)) / k for a, b in zip(ti, tr)])
    return rel_l2, float(cos.mean()), float(cos.min()), top1, float(ov)


def ratio_stats(x):
    a = np.abs(x)
    r = a.max(1) / np.median(a, 1)
    return float(np.median(r)), float(r.max())


def main():
    calib_path, eval_path = sys.argv[1], sys.argv[2]
    t0 = time.time()
    xc, Wc, _, _ = load(calib_path)
    x, W, q, sc = load(eval_path)
    assert np.array_equal(W, Wc), "captures must share the layer's weights"
    del Wc
    print(f"# loaded in {time.time()-t0:.1f}s; calib={calib_path} eval={eval_path}")
    print(f"# layer K={K} N={N}; eval rows {x.shape[0]}, calib rows {xc.shape[0]}")
    ref = x @ W
    print(f"# eval activation max|x|/median|x| per token: median {ratio_stats(x)[0]:.1f}, max {ratio_stats(x)[1]:.1f}")

    rows = []
    def run(name, xt, Wt, ratio_x=None):
        t = time.time()
        y = xt @ Wt
        m = metrics(y, ref)
        rs = ratio_stats(ratio_x) if ratio_x is not None else (float("nan"),) * 2
        rows.append((name,) + m + rs)
        ok = "PASS" if m[2] >= 0.999 else "fail"
        print(f"{name:52s} relL2 {m[0]*100:6.3f}%  cos mean {m[1]:.6f} worst {m[2]:.6f} {ok}  "
              f"top1 moved {m[3]:4d}/{y.shape[0]}  top8 {m[4]:.4f}  x ratio med {rs[0]:.1f}  ({time.time()-t:.1f}s)",
              flush=True)

    # 0. floor: production path, bf16 weights (the dequant slab rounds to bf16), bf16 x
    run("0  production: bf16(W g64) x bf16", x, bf16_round(W))
    # 1. reproduce the doc: g64 int4 W as-is, per-token int8 x
    xq = q_act_token(x)
    run("1  W4 g64 (ckpt) + A8 token", xq, W, x)
    # 2. weight-only effect of per-channel requant
    W8 = q_w_channel(W, 8)
    run("2  W8 per-channel requant, A16 (weight error only)", x, W8)
    run("3  W8 per-channel requant + A8 token", xq, W8, x)
    W4 = q_w_channel(W, 4)
    run("4  W4 per-channel RTN + A8 token", xq, W4, x)
    for g in (128, 256):
        run(f"4g W4 g{g} RTN + A8 token", xq, q_w_group(W, 4, g), x)

    # SmoothQuant, scales from the CALIBRATION capture's activations
    amax_c = np.abs(xc).max(0)
    wmax = np.abs(W).max(1)
    for alpha in (0.5, 0.8):
        s = (amax_c ** alpha) / np.maximum(wmax, 1e-8) ** (1 - alpha)
        s = np.clip(s, 1e-5, None).astype(np.float32)
        xs, Ws = x / s, W * s[:, None]
        xsq = q_act_token(xs)
        run(f"5  SmoothQuant a={alpha} + W8pc + A8", xsq, q_w_channel(Ws, 8), xs)
        run(f"5b SmoothQuant a={alpha} + W4pc RTN + A8", xsq, q_w_channel(Ws, 4), xs)

    # Hadamard (randomised, block-diagonal; 5120 = 5 x 1024)
    for b in (128, 1024):
        fx, fw = block_rot(b, seed=b)
        xr, Wr = fx(x), fw(W)
        xrq = q_act_token(xr)
        run(f"6  Hadamard b={b} + W8pc + A8", xrq, q_w_channel(Wr, 8), xr)
        run(f"6b Hadamard b={b} + W4pc RTN + A8", xrq, q_w_channel(Wr, 4), xr)

    # random orthogonal over all of K (ceiling for "any rotation"); too costly at K=17408
    if K > 8192:
        print(f"# total {time.time()-t0:.1f}s")
        return
    rng = np.random.default_rng(0)
    Qm, _ = np.linalg.qr(rng.standard_normal((K, K)).astype(np.float32))
    xr, Wr = x @ Qm, Qm.T @ W
    xrq = q_act_token(xr)
    run("7  random orthogonal 5120 + W8pc + A8", xrq, q_w_channel(Wr, 8), xr)
    run("7b random orthogonal 5120 + W4pc RTN + A8", xrq, q_w_channel(Wr, 4), xr)

    # SmoothQuant then Hadamard
    s = ((amax_c ** 0.5) / np.maximum(wmax, 1e-8) ** 0.5).clip(1e-5).astype(np.float32)
    fx, fw = block_rot(1024, seed=7)
    xr, Wr = fx(x / s), fw(W * s[:, None])
    xrq = q_act_token(xr)
    run("8  SQ a=0.5 + Hadamard 1024 + W8pc + A8", xrq, q_w_channel(Wr, 8), xr)
    run("8b SQ a=0.5 + Hadamard 1024 + W4pc RTN + A8", xrq, q_w_channel(Wr, 4), xr)
    print(f"# total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
