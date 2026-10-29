#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec 25a: Bolt's remote modelling code, imported from the checkpoint snapshot at run time.

The language model of `nemotron_h_omni` is the remote code's `NemotronHPuzzleForCausalLM` (modeling_nemotron_h_puzzle.py
over modeling_nemotron_h.py = R). Nothing of it is vendored: `load_remote(snapshot)` imports the four text files
(configuration_nemotron_h{,_puzzle}.py, modeling_nemotron_h{,_puzzle}.py) from the snapshot under a private package
name, after checking each file's sha256 against the pinned revision (8932aa8d of our export; NVIDIA's 24GB repo ships
the same bytes). The omni wrapper and the vision tower are never imported.

R needs mamba_ssm at import (R:60-63: `rmsnorm_fn`, else ImportError) and imports the other kernels only on CUDA
(R:54-68; transformers' is_mamba_2_ssm_available / is_causal_conv1d_available require CUDA). `load_remote` installs
bolt_mamba_ops.stub_package() as those modules, then sets R's five kernel globals to the stub's functions and patches
ONE thing, the mixer's dispatch (R:729-738, the kernel path only when the weights are on CUDA):

    mixer._bolt_path == "kernel"  -> R's cuda_kernels_forward (the port; spec 25 Decision 1)
                                     + a prompt continuation (R's kernel path has none: with a cache that has a
                                       previous state it assumes ONE token, R:397-439) - `continue_forward` below
    mixer._bolt_path == "torch"   -> R's torch_forward, unchanged (its own conv, scan, roundings and dt clamp;
                                     rmsnorm_fn is the stub's either way - R imports no other)

    load_remote(snapshot) -> namespace (R, RP config module, the puzzle LM class, the cache class, the ops)
    build_lm(cfg_dict, dtype, path="kernel", seed=None) -> NemotronHPuzzleForCausalLM (eager attention)
    set_path(model, "kernel" | "torch");  fp32_cache(model, batch, dtype=torch.float32)
    llm_config(snapshot) -> the checkpoint's llm_config dict (special floats decoded)
"""
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import torch  # noqa: E402

PINNED_TRANSFORMERS = "5.19.0"
PKG = "bolt_remote_r"
# sha256 of the remote text files at our export's 8932aa8d (= NVIDIA's 24GB repo c6cc2e8f, byte-identical); the HF blob
# of modeling_nemotron_h.py is de290021078450c4d5042f66dd74b8eecf1d3eb3 (git sha1), the spec's `R:` file.
REMOTE_SHA256 = {
    "configuration_nemotron_h.py": "0fc818c10506c91bd02df5a605f49cb0704b5498954f46dbde2d63999ae36c3d",
    "configuration_nemotron_h_puzzle.py": "b13b837322dd0e5ec972227d180aa86eb671cab2594df0c0d51d2456599bae91",
    "modeling_nemotron_h.py": "e1cb5fc02e887983f0a445bf4c1a2604453b2cb2db4624c7004dcf663bbb1b6e",
    "modeling_nemotron_h_puzzle.py": "e81a7a97ce55896c6af898cf792cfbdff123eb476d11267aefc3f29bc5af3882",
}
REMOTE_BLOB = "de290021078450c4d5042f66dd74b8eecf1d3eb3"
SNAPSHOT_GLOBS = (
    "models--urakozz--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ/snapshots/8932aa8d1844f91056c5b9433448075a842b9156",
    "models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/snapshots/c6cc2e8ffa353403213d6b9b18aaf8eb684d705a",
)

_OPS = None
_NS = None


def _ops():
    global _OPS
    if _OPS is None and "bolt_mamba_ops" in sys.modules:          # one module object for the stubs and the callers
        _OPS = sys.modules["bolt_mamba_ops"]
    if _OPS is None:
        spec = importlib.util.spec_from_file_location("bolt_mamba_ops", os.path.join(_HERE, "bolt_mamba_ops.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["bolt_mamba_ops"] = mod
        spec.loader.exec_module(mod)
        _OPS = mod
    return _OPS


def require_version() -> None:
    import transformers
    if transformers.__version__ != PINNED_TRANSFORMERS:
        raise RuntimeError(f"the Bolt reference runs R through transformers {PINNED_TRANSFORMERS}; this is "
                           f"{transformers.__version__} ({os.path.dirname(transformers.__file__)}) - put the 5.19.0 "
                           f"site first on PYTHONPATH (tools/oracle/qwen4exp_env.sh)")


def find_snapshot() -> str | None:
    """The pinned snapshot in the HF cache ($BOLT_SNAPSHOT first; HF_HOME, /hf, ~/.cache/huggingface)."""
    env = os.environ.get("BOLT_SNAPSHOT", "")
    if env and os.path.exists(os.path.join(env, "config.json")):
        return env
    roots = [os.environ.get("HF_HOME", ""), "/hf", os.path.expanduser("~/.cache/huggingface")]
    for r in roots:
        for g in SNAPSHOT_GLOBS:
            d = os.path.join(r, "hub", g)
            if r and os.path.exists(os.path.join(d, "modeling_nemotron_h.py")):
                return d
    return None


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(path: str) -> str:
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def llm_config(snapshot: str) -> dict:
    """config.json's llm_config (nemotron_h_puzzle), `{"__float__": "Infinity"}` decoded as transformers does."""
    require_version()
    from transformers.configuration_utils import PreTrainedConfig
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        c = PreTrainedConfig._decode_special_floats(json.load(f))
    if c.get("model_type") != "nemotron_h_omni":
        raise RuntimeError(f"{snapshot}: model_type {c.get('model_type')!r}, expected nemotron_h_omni")
    lc = c["llm_config"]
    if lc.get("model_type") != "nemotron_h_puzzle":
        raise RuntimeError(f"{snapshot}: llm_config.model_type {lc.get('model_type')!r}, expected nemotron_h_puzzle")
    return lc


def install_stub() -> None:
    """mamba_ssm / causal_conv1d -> bolt_mamba_ops' stub modules (refused if a real package is already imported)."""
    for name, mod in _ops().stub_package().items():
        have = sys.modules.get(name)
        if have is not None and not getattr(have, "__bolt_stub__", False):
            raise RuntimeError(f"{name} is a real package here ({getattr(have, '__file__', '?')}): the Bolt reference "
                               f"runs the plain-torch ops, never the GPU kernels")
        sys.modules[name] = mod


def continue_forward(self, hidden_states, cache_params):
    """A prompt chunk of T > 1 rows after a cached prefix: R's kernel-path prefill (R:446-531) with the conv seeded
    from the cache's raw columns and the scan from its SSM state (initial_states), writing both back. R has no such
    path (its cached branch takes one token); this is the same five functions called the way the prefill calls them."""
    ops = _ops()
    Bt, T, _ = hidden_states.shape
    K = self.conv_kernel_size
    projected = self.in_proj(hidden_states)
    A = -torch.exp(self.A_log.float())
    gate, xBC, dt = torch.split(projected, [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
    st = cache_params.conv_states[self.layer_idx]                               # [B][conv_dim][K] raw columns
    xBC_t = xBC.transpose(1, 2)
    new_state = torch.cat([st, xBC_t.to(st.dtype)], -1)[..., -K:]
    conv = ops.conv_seq(xBC_t, self.conv1d.weight.squeeze(1), self.conv1d.bias, st[..., -(K - 1):]).transpose(1, 2)
    cache_params.conv_states[self.layer_idx].copy_(new_state)
    gts = self.n_groups * self.ssm_state_size
    x, B, C = torch.split(conv, [self.intermediate_size, gts, gts], dim=-1)
    lim = {} if self.time_step_limit is None else {"dt_limit": self.time_step_limit}
    out, ssm = ops.KERNEL_FUNCTIONS["mamba_chunk_scan_combined"](
        x.view(Bt, T, -1, self.head_dim), dt, A, B.view(Bt, T, self.n_groups, -1), C.view(Bt, T, self.n_groups, -1),
        chunk_size=self.chunk_size, D=self.D, z=None, seq_idx=None, return_final_states=True, dt_bias=self.dt_bias,
        dt_softplus=True, initial_states=cache_params.ssm_states[self.layer_idx], **lim)
    cache_params.ssm_states[self.layer_idx].copy_(ssm)
    out = self.norm(out.view(Bt, T, -1), gate)
    return self.out_proj(out)


def _dispatch(self, hidden_states, cache_params=None, attention_mask=None):
    """The patched R:729-738: the path is the instance's `_bolt_path`, not the device."""
    path = getattr(self, "_bolt_path", "torch")
    if path == "torch":
        return self.torch_forward(hidden_states, cache_params, attention_mask)
    if path != "kernel":
        raise ValueError(path)
    if cache_params is not None and cache_params.has_previous_state and hidden_states.shape[1] > 1:
        if attention_mask is not None and not torch.all(attention_mask == 1):
            raise NotImplementedError("a padded continuation")
        return continue_forward(self, hidden_states, cache_params)
    return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)


def load_remote(snapshot: str):
    """Import the remote code from `snapshot` (sha256-checked) with the stub; returns a namespace:
    R (modeling_nemotron_h), RC, RP (the two config modules), RM (modeling_nemotron_h_puzzle), LM (the puzzle class),
    Cache (NemotronHHybridDynamicCache), Config (NemotronHPuzzleConfig), ops (bolt_mamba_ops), snapshot."""
    global _NS
    require_version()
    if _NS is not None:
        if _NS.snapshot_sha != {f: sha256(os.path.join(snapshot, f)) for f in REMOTE_SHA256}:
            raise RuntimeError(f"{snapshot}: other remote files than the ones already imported")
        return _NS
    for fn, want in REMOTE_SHA256.items():
        p = os.path.join(snapshot, fn)
        if not os.path.exists(p):
            raise RuntimeError(f"{snapshot}: {fn} absent (the pinned snapshot's remote code)")
        got = sha256(p)
        if got != want:
            raise RuntimeError(f"{p}: sha256 {got}, the pinned revision's is {want} - another revision of the remote "
                               f"code is not the reference")
    if PKG in sys.modules:
        raise RuntimeError(f"the remote code is imported already (by another bolt_remote module object, "
                           f"{sys.modules.get('bolt_remote')}): load bolt_remote once per process via sys.modules")
    install_stub()
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [snapshot]
    pkg.__file__ = os.path.join(snapshot, "__init__.py")
    sys.modules[PKG] = pkg
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True                  # never write __pycache__ into the HF cache
    try:
        RC = importlib.import_module(f"{PKG}.configuration_nemotron_h")
        RP = importlib.import_module(f"{PKG}.configuration_nemotron_h_puzzle")
        R = importlib.import_module(f"{PKG}.modeling_nemotron_h")
        RM = importlib.import_module(f"{PKG}.modeling_nemotron_h_puzzle")
    finally:
        sys.dont_write_bytecode = old
    ops = _ops()
    for name, fn in ops.KERNEL_FUNCTIONS.items():
        if getattr(R, name, "absent") is not None:
            raise RuntimeError(f"R.{name} resolved at import ({getattr(R, name)!r}): expected None on a CPU")
        setattr(R, name, fn)
    if R.rmsnorm_fn is not ops._rmsnorm_fn:
        raise RuntimeError("R.rmsnorm_fn is not the stub's")
    R.NemotronHMamba2Mixer._r_forward = R.NemotronHMamba2Mixer.forward
    R.NemotronHMamba2Mixer.forward = _dispatch
    _NS = types.SimpleNamespace(R=R, RC=RC, RP=RP, RM=RM, LM=RM.NemotronHPuzzleForCausalLM,
                                Cache=R.NemotronHHybridDynamicCache, Config=RP.NemotronHPuzzleConfig, ops=ops,
                                snapshot=snapshot,
                                snapshot_sha={f: sha256(os.path.join(snapshot, f)) for f in REMOTE_SHA256})
    return _NS


def make_config(cfg_dict: dict, dtype: torch.dtype = torch.bfloat16):
    """NemotronHPuzzleConfig from an llm_config-shaped dict, eager attention."""
    ns = _require()
    d = dict(cfg_dict)
    for k in ("auto_map", "architectures", "_megatron_bridge_hf_prefix", "dtype", "torch_dtype"):
        d.pop(k, None)
    cfg = ns.Config(**d)
    cfg._attn_implementation = "eager"
    cfg.dtype = dtype
    return cfg


def _require():
    if _NS is None:
        snap = find_snapshot()
        if snap is None:
            raise RuntimeError("load_remote(snapshot) first (or put the pinned snapshot in the HF cache)")
        load_remote(snap)
    return _NS


def set_path(model, path: str) -> None:
    """Every Mamba mixer of `model` to the kernel path (the port) or R's torch fallback."""
    if path not in ("kernel", "torch"):
        raise ValueError(path)
    ns = _require()
    for m in model.modules():
        if isinstance(m, ns.R.NemotronHMamba2Mixer):
            m._bolt_path = path


def fix_dtypes(model, dtype: torch.dtype) -> None:
    """The model in `dtype`, except each router's `e_score_correction_bias`, kept fp32 (R:997, R:1054-1063 - the
    checkpoint stores it F32 and its values are not bf16 values)."""
    ns = _require()
    routers = [m for m in model.modules() if isinstance(m, ns.R.NemotronHTopkRouter)]
    keep = [m.e_score_correction_bias.detach().clone().float() for m in routers]
    model.to(dtype)
    for m, b in zip(routers, keep):
        m.e_score_correction_bias = b


def build_lm(cfg_dict: dict, dtype: torch.dtype = torch.bfloat16, path: str = "kernel", seed: int | None = None):
    """A random NemotronHPuzzleForCausalLM of `cfg_dict` (R's own init under `seed`), in `dtype` (the router bias fp32),
    the Mamba mixers on `path`."""
    ns = _require()
    if seed is not None:
        torch.manual_seed(seed)
    cfg = make_config(cfg_dict, dtype)
    model = ns.LM(cfg)
    fix_dtypes(model, dtype)
    set_path(model, path)
    return model.eval()


def fp32_cache(model, batch: int = 1, dtype: torch.dtype = torch.float32):
    """R's NemotronHHybridDynamicCache with its conv / SSM states in `dtype` (fp32: the reference, Decision 2; R:1664-1667
    would build it in the model's dtype)."""
    ns = _require()
    return ns.Cache(model.config, batch, dtype=dtype, device="cpu")
