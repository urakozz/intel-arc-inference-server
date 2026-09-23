"""The residual-stream rotation for Qwen3.5 (model_type qwen3_5), shared by the
rotation writer and its check.

R = D * blockdiag(H_1024) / 32 over the 5120-wide residual stream: random signs
D, then a Sylvester Walsh-Hadamard per 1024-block, orthonormal. The same
transform the W8A8 probe applies to activations (tools/probe/probe_w8a8.cl,
pw8_quant_had2_*; docs/probe-w4a8-2026-09-23.md section 14). A rotated model
carries h' = h R in its residual stream instead of h:

    embed_tokens       E' = E R                         (rows)
    RMSNorm (1 + w)    folded into every linear that reads it, then w = 0,
                       because RMSNorm commutes with R only without its gain
    stream readers     W' = (W diag(1 + w)) R           q/k/v, in_proj_qkv/z/a/b,
                                                        gate/up, lm_head
    stream writers     W' = R^T W                       o_proj, out_proj, down_proj

x' W'^T = x W^T for every linear, so the rotated model computes the same
function as the original, up to rounding, and needs NO runtime operation:
it is a plain Hugging Face checkpoint that transformers, AutoRound and vLLM
run unmodified. What it buys is that every stream reader sees rotated,
outlier-spread activations (per-channel int8 activations become accurate;
the section 14 study).

Deliberately NOT rotated:
  * the down projection's own INPUT (silu(gate) * up): elementwise, so a
    rotation there cannot be folded into anything; it stays a runtime op in the
    engine (section 14's pw8_quant_had2_17 + rotating requant).
  * the MTP head (`mtp.*`): it shares embed_tokens and lm_head with the main
    model but has its own norms, so it cannot be made consistent with a folded,
    rotated lm_head. Left byte-identical; an engine that uses it must un-rotate
    its inputs (x R^T) and feed lm_head x_mtp_normed * (1 + w_mtp) / (1 + w_final),
    rotated. Recorded in the config, not solved here.
  * the vision tower.
"""
import torch

BLOCK = 1024
HIDDEN = 5120
SEED = 20260923
PREFIX = "model.language_model."


def signs(n: int = HIDDEN, seed: int = SEED) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.float32)


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalised Walsh-Hadamard per 1024-block of the last dim, ascending stages."""
    shp = x.shape
    n = shp[-1]
    assert n % BLOCK == 0, n
    y = x.reshape(-1, n // BLOCK, BLOCK)
    h = 1
    while h < BLOCK:
        y = y.reshape(-1, n // BLOCK, BLOCK // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2)
        h *= 2
    return y.reshape(shp)


def rot(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """x R, along the last dim."""
    return _fwht(x * d) * (1.0 / 32.0)


def unrot(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """x R^T, along the last dim (R^T = blockdiag(H) D / 32)."""
    return _fwht(x) * (1.0 / 32.0) * d


READERS_IN = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a",
              "linear_attn.in_proj_b", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
READERS_POST = ("mlp.gate_proj", "mlp.up_proj")
WRITERS = ("linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj")
NORMS = ("input_layernorm", "post_attention_layernorm")
ROW_CHUNK = 16384   # rows per step for the 248320-row embed / lm_head


def norm_keys(n_layers: int) -> list[str]:
    ks = [f"{PREFIX}layers.{i}.{n}.weight" for i in range(n_layers) for n in NORMS]
    return ks + [f"{PREFIX}norm.weight"]


def classify(key: str):
    """-> (kind, norm key or None). kind: embed | head | norm | read | write | copy."""
    if key == f"{PREFIX}embed_tokens.weight":
        return "embed", None
    if key == "lm_head.weight":
        return "head", f"{PREFIX}norm.weight"
    if key == f"{PREFIX}norm.weight":
        return "norm", None
    if key.startswith(f"{PREFIX}layers."):
        rest = key[len(f"{PREFIX}layers."):]
        idx, _, tail = rest.partition(".")
        base = f"{PREFIX}layers.{idx}."
        if tail in tuple(f"{n}.weight" for n in NORMS):
            return "norm", None
        if tail in tuple(f"{r}.weight" for r in READERS_IN):
            return "read", base + "input_layernorm.weight"
        if tail in tuple(f"{r}.weight" for r in READERS_POST):
            return "read", base + "post_attention_layernorm.weight"
        if tail in tuple(f"{w}.weight" for w in WRITERS):
            return "write", None
    return "copy", None


def _rows(fn, w: torch.Tensor) -> torch.Tensor:
    if w.shape[0] <= ROW_CHUNK:
        return fn(w)
    return torch.cat([fn(w[a:a + ROW_CHUNK]) for a in range(0, w.shape[0], ROW_CHUNK)])


def transform(key: str, w: torch.Tensor, norms: dict[str, torch.Tensor], d: torch.Tensor):
    """The rotated tensor for `key`, in fp32; `w` is the original (any dtype).

    Returns (kind, tensor). norms maps norm keys to their ORIGINAL weights.
    """
    kind, nk = classify(key)
    if kind == "copy":
        return kind, w
    w = w.to(torch.float32)
    if kind == "norm":
        return kind, torch.zeros_like(w)                       # (1 + 0) = 1
    if kind == "embed":
        return kind, _rows(lambda t: rot(t, d), w)             # E R
    if kind in ("read", "head"):
        assert w.shape[1] == HIDDEN, (key, w.shape)
        g = 1.0 + norms[nk].to(torch.float32)
        return kind, _rows(lambda t: rot(t * g, d), w)         # (W diag(g)) R
    if kind == "write":
        assert w.shape[0] == HIDDEN, (key, w.shape)
        return kind, rot(w.t().contiguous(), d).t().contiguous()   # R^T W = (W^T R)^T
    raise AssertionError(kind)
