#!/usr/bin/env bash
# Checkpoint provenance - dense Qwen/Qwen3.8-27B quantized on the box (2026-08-25).
#
# Both runs: AutoRound from source (/home/user/auto-round, its .venv:
# python 3.12.13, torch 2.13.0+xpu), device 1 (second B70), HF cache model.
# Flags are load-bearing for b70-inference-server:
#   --group_size 64            the kernels/loader bake g64 (AutoRound default is 128)
#   sym                        default for W4A16 - dequant contract w = (q-8)*s
#   --quant_lm_head            the point: int4 lm_head (memo §5.1)
#   --format auto_round:auto_gptq
#                              passes the lm_head restriction AND packs v1 zeros
#                              (0x77777777) - byte-compatible with our loader
#
# Variant 1 - tuned (sign-SGD, hours; best ints). Started 2026-08-25:
#   source ./.venv/bin/activate
#   python -m auto_round Qwen/Qwen3.8-27B --scheme W4A16 --group_size 64 \
#     --quant_lm_head --format "auto_round:auto_gptq" --device 1 \
#     --output_dir ./qwen38-27b-w4g64-lmhead --low_gpu_mem_usage
#
# Variant 2 - RTN via the REGULAR flow (minutes; format-identical bytes, lower
# accuracy; used for format-performance bring-up).
# NOTE (measured 2026-08-25): --iters 0 --disable_opt_rtn WITHOUT
# --disable_model_free auto-routes to the model-free path, which on this
# wrapped arch (model.language_model.*) records lm_head int4 in
# extra_config but leaves lm_head.weight BF16 - upstream bug/limitation
# (auto_round/compressors/model_free.py + utils/model_free_utils.py).
# --disable_model_free forces the regular flow, whose lm_head path is the
# documented, supported one (auto_round/schemes.py set_layer_config).
#   source ./.venv/bin/activate
#   python -m auto_round Qwen/Qwen3.8-27B --scheme W4A16 --group_size 64 \
#     --quant_lm_head --format "auto_round:auto_gptq" --device 1 \
#     --output_dir ./qwen38-27b-w4g64-rtn --iters 0 --disable_opt_rtn \
#     --disable_model_free --low_gpu_mem_usage
#
# This script re-runs variant 2 (the reproducible-in-minutes one) when executed
# on the box from /home/user/auto-round.
set -euo pipefail
cd /home/user/auto-round
source ./.venv/bin/activate
python -m auto_round Qwen/Qwen3.8-27B --scheme W4A16 --group_size 64 \
  --quant_lm_head --format "auto_round:auto_gptq" --device "${DEVICE:-1}" \
  --output_dir "${OUT:-./qwen38-27b-w4g64-rtn}" --iters 0 --disable_opt_rtn \
  --disable_model_free --low_gpu_mem_usage
