#!/bin/bash
# DO NOT use GPTQ/AWQ model in FSDP+QLoRA

DISABLE_VERSION_CHECK=1 nohup accelerate launch --config_file ./examples/GRLM/fsdp_config.yaml src/train.py ./examples/GRLM/shopping_sft.yaml > ./logs/qwen35-9b_fsdp_qlora.out 2>&1 &
