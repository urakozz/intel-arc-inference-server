# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the b70-inference-server authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations under the
# License.
#
# Source of the semantics: Aleph Alpha's vLLM plugin `aleph-alpha-inference`
# (https://github.com/Aleph-Alpha/aleph-alpha-inference, Apache-2.0, Copyright 2026 Aleph Alpha GmbH),
# `aleph_alpha_inference/config.py` @ 049a6a7b: `Kolibri1Config(Qwen3MoeConfig)` with
# `model_type = "kolibri1"`. This file is a standalone transformers (5.x) config with the same fields,
# written for spec 20a of b70-inference-server (no transformers release implements Kolibri 1).
"""Kolibri 1 configuration (transformers 5.x, trust_remote_code)."""

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


@strict
class Kolibri1Config(PreTrainedConfig):
    r"""Kolibri 1 (Aleph Alpha): a MoE decoder with sandwich norms, q/k norm, 4:1 sliding:full attention
    (RoPE in the sliding layers only), sigmoid routing on `logits + expert_bias` and an ungated shared
    expert. Field names follow the checkpoint's `config.json` (`Aleph-Alpha/Kolibri-1-BF16`).

    layer_types (`list[str]`): "sliding_attention" (window `sliding_window`, RoPE) or "full_attention"
        (no positional encoding), one per layer.
    sliding_window (`int`): keys visible to a query in a sliding layer, the query itself included
        (513 = 512 preceding tokens plus the current one).
    shared_expert_intermediate_size (`int`): the ungated shared expert's width.
    head_dtype (`str`): the dtype of the lm_head projection ("float32": bf16 hidden and weights, fp32
        accumulation and fp32 logits; vLLM's `ModelConfig.head_dtype`).
    """

    model_type = "kolibri1"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 128000
    hidden_size: int = 2560
    num_hidden_layers: int = 50
    num_attention_heads: int = 48
    num_key_value_heads: int = 4
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    use_sliding_window: bool = True
    sliding_window: int | None = 513
    layer_types: list[str] | None = None
    attention_dropout: float | int = 0.0
    num_experts: int = 384
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    norm_topk_prob: bool = False
    head_dtype: str | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = ["full_attention" if (i + 1) % 5 == 0 else "sliding_attention"
                                for i in range(self.num_hidden_layers)]
        if not self.use_sliding_window:
            self.sliding_window = None
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture() if hasattr(super(), "validate_architecture") else None
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(f"layer_types has {len(self.layer_types)} entries for {self.num_hidden_layers} layers")
        bad = set(self.layer_types) - {"sliding_attention", "full_attention"}
        if bad:
            raise ValueError(f"layer_types: unknown {sorted(bad)}")
        if "sliding_attention" in self.layer_types and not (self.sliding_window and self.sliding_window > 0):
            raise ValueError("sliding layers need a positive sliding_window (and use_sliding_window true)")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads % num_key_value_heads != 0")
        if self.hidden_act != "silu":
            raise ValueError(f"hidden_act {self.hidden_act!r}: Kolibri 1's experts are SiLU")
        if self.attention_bias:
            raise ValueError("attention_bias: Kolibri 1 has none")
        if self.norm_topk_prob:
            raise ValueError("norm_topk_prob: Kolibri 1 does not renormalise its routing weights")


__all__ = ["Kolibri1Config"]
