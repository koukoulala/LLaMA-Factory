#!/bin/bash
# DO NOT use GPTQ/AWQ model in FSDP+QLoRA

CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch --config_file ./examples/ImageLPRelevance/fsdp_config.yaml src/train.py ./examples/ImageLPRelevance/qwen3vl_lora_sft.yaml > ./logs/qwen3vl-2b.out 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli export examples/ImageLPRelevance/qwen3vl_gptq.yaml