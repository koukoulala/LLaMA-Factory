#!/bin/bash
# DO NOT use GPTQ/AWQ model in FSDP+QLoRA

DISABLE_VERSION_CHECK=1 nohup accelerate launch --config_file ./examples/GRLM/fsdp_config.yaml src/train.py ./examples/GRLM/shopping_sft.yaml > ./logs/qwen35-9b_fsdp_qlora.out 2>&1 &

DISABLE_VERSION_CHECK=1 PYTHONPATH=src nohup python -u -m llamafactory.cli export examples/GRLM/shopping_merge_lora.yaml > ./logs/merge_lora.out 2>&1 &

nohup bash ./examples/GRLM/shopping_sft.sh > ./logs/shopping_sft_v2.log 2>&1 &
nohup bash ./examples/GRLM/shopping_sft_v100.sh > ./logs/shopping_sft_v2_v100.log 2>&1 &

nohup bash -c 'DATA="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/sft_data/combined_sft.jsonl"; while true; do if [ -f "$DATA" ] && [ $(wc -l < "$DATA" 2>/dev/null || echo 0) -ge 1000 ]; then echo "$(date): Data ready ($(wc -l < "$DATA") lines), starting training..."; cd /scratch/workspaceblobstore/users/xiaoyukou/LLaMA-Factory && bash ./examples/GRLM/shopping_sft.sh; break; fi; echo "$(date): Waiting for data..."; sleep 60; done' > ./logs/shopping_sft_v2.log 2>&1 &
