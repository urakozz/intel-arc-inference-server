#!/usr/bin/env python3
"""Agnes 3.0 Flash as a Qwen3.5 reference (spec 14 §3.4), for tools/oracle/dump.py.

Agnes is Qwen3.5 with (a) 72 layers, (b) `delta_attn.` / `global_attn.` tensor
names for Qwen3.5's `linear_attn.` / `self_attn.`, and (c) a second, narrower
SwiGLU per layer, `mlp.parallel_ffn` (intermediate 2048), whose output is ADDED
to the MLP's - exactly what the operator's vLLM PR #57003 does
(`_AGNES_TO_QWEN3_5`, `AgnesMLP`) and what the checkpoint's own
`modeling_agnes.py` computes (`AgnesMLP.forward`: y = mlp(x) + parallel_ffn(x),
each branch a bf16 nn.Linear chain, the sum in bf16).

So the reference is transformers' `Qwen3_5ForCausalLM` built from a translated
text config, with each layer's `mlp` replaced by `AgnesMLP`, which adopts the
layer's own gate/up/down (state-dict names unchanged) and adds the branch.
Nothing here folds: the fold is the ENGINE's (spec 14 §2), and the reference
keeps the two branches so the golden sets grade the fold rather than share it.

The pure parts (name map, config translation, `AgnesMLP`) need only torch; the
model build needs transformers >= 5 (Qwen3_5) and runs in the oracle container.
"""
import copy

import torch
from torch import nn

# vLLM PR #57003's WeightsMapper.orig_to_new_substr, verbatim.
NAME_MAP = (("delta_attn.", "linear_attn."), ("global_attn.", "self_attn."))
# configuration_agnes.LAYER_DELTA / LAYER_GLOBAL -> Qwen3_5TextConfig's layer types.
LAYER_TYPE_MAP = {"agnes_delta_attention": "linear_attention",
                  "agnes_global_attention": "full_attention"}
# Agnes text-config keys Qwen3_5TextConfig does not have. `output_gate_type`
# ("swish") is a label only: modeling_agnes.py gates attention with
# torch.sigmoid, exactly as Qwen3.5 does (AgnesGlobalAttention.forward).
DROP_KEYS = ("parallel_ffn_intermediate_size", "global_attention_interval",
             "output_gate_type", "mtp_use_dedicated_embeddings", "model_type",
             "architectures", "auto_map", "transformers_version")


def is_agnes(config: dict) -> bool:
    return config.get("model_type") == "agnes"


def map_name(name: str) -> str:
    """Checkpoint name -> Qwen3.5 name: the first matching infix, as the PR's mapper."""
    for old, new in NAME_MAP:
        if old in name:
            return name.replace(old, new, 1)
    return name


def translate_text_config(config: dict) -> tuple[dict, int]:
    """Agnes config.json -> (Qwen3_5TextConfig kwargs, parallel FFN width)."""
    tc = dict(config["text_config"])
    parallel = int(tc.get("parallel_ffn_intermediate_size") or 0)
    if parallel <= 0:
        raise ValueError("an Agnes text_config without parallel_ffn_intermediate_size")
    types = tc.get("layer_types")
    if types is None:   # configuration_agnes.py's default plan
        period = int(tc.get("global_attention_interval", 4))
        types = ["agnes_global_attention" if (i + 1) % period == 0 else "agnes_delta_attention"
                 for i in range(tc["num_hidden_layers"])]
    unknown = sorted(set(types) - set(LAYER_TYPE_MAP))
    if unknown:
        raise ValueError(f"unknown Agnes layer types {unknown}")
    out = {k: v for k, v in tc.items() if k not in DROP_KEYS}
    out["layer_types"] = [LAYER_TYPE_MAP[t] for t in types]
    if len(out["layer_types"]) != out["num_hidden_layers"]:
        raise ValueError("layer_types and num_hidden_layers disagree")
    return out, parallel


class AgnesMLP(nn.Module):
    """A Qwen3.5 MLP with Agnes's parallel branch summed onto it.

    Adopts the layer's projections so their state-dict names stay
    `mlp.{gate,up,down}_proj`; the branch is `mlp.parallel_ffn.{gate,up,down}_proj`.
    The arithmetic is modeling_agnes.py's: both branches in the module dtype, then
    one add in that dtype.
    """

    def __init__(self, mlp: nn.Module, parallel_ffn: nn.Module) -> None:
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        self.act_fn = mlp.act_fn
        self.parallel_ffn = parallel_ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return y + self.parallel_ffn(x)


def attach_parallel_ffn(model: nn.Module, text_config, parallel: int) -> int:
    """Replace every decoder layer's `mlp` by an AgnesMLP; returns the layer count.

    The branch is the layer's own MLP class at intermediate `parallel`, on the
    device the model was built on. transformers 5.15's `Qwen3_5MLP(config,
    intermediate_size)` takes the width as an argument; a class that reads it from
    the config alone gets a config copy with `intermediate_size = parallel`.
    """
    import inspect
    pc = copy.copy(text_config)
    pc.intermediate_size = parallel
    n = 0
    for layer in model.model.layers:
        cls = type(layer.mlp)
        takes_width = "intermediate_size" in inspect.signature(cls.__init__).parameters
        branch = cls(text_config, parallel) if takes_width else cls(pc)
        layer.mlp = AgnesMLP(layer.mlp, branch)
        n += 1
    return n


def build_reference(config: dict, device: str = "meta"):
    """Qwen3_5ForCausalLM + the parallel FFN, from an Agnes config.json dict.

    Returns (model, text_config). Needs transformers >= 5 (Qwen3_5).
    """
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    kwargs, parallel = translate_text_config(config)
    tc = Qwen3_5TextConfig(**kwargs)
    with torch.device(device):
        model = Qwen3_5ForCausalLM(tc)
        attach_parallel_ffn(model, tc, parallel)
    return model, tc
