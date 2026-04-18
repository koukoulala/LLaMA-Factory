#!/bin/bash
# DO NOT use GPTQ/AWQ model in FSDP+QLoRA

FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file ./examples/GRLM/fsdp2_config.yaml src/train.py ./examples/GRLM/shopping_sft.yaml > ./logs/shopping_sft_full_v4.log 2>&1 &
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --config_file ./examples/GRLM/fsdp2_config.yaml src/train.py ./examples/GRLM/shopping_sft_journey_w_query.yaml > ./logs/shopping_sft_full_journey_w_query.log 2>&1 &

DISABLE_VERSION_CHECK=1 PYTHONPATH=src nohup python -u -m llamafactory.cli export examples/GRLM/shopping_merge_lora.yaml > ./logs/merge_lora.out 2>&1 &

nohup bash ./examples/GRLM/shopping_sft.sh > ./logs/shopping_sft_v2.log 2>&1 &
nohup bash ./examples/GRLM/shopping_sft_v100.sh > ./logs/shopping_sft_v2_v100.log 2>&1 &
nohup bash ./examples/GRLM/shopping_sft_continue.sh > ./logs/shopping_sft_v3_continue.log 2>&1 &
nohup bash ./examples/GRLM/shopping_sft_v4.sh > ./logs/shopping_sft_v4.log 2>&1 &
nohup bash ./examples/GRLM/shopping_sft_full.sh > ./logs/shopping_sft_full_v4.log 2>&1 &

CUDA_VISIBLE_DEVICES="" DISABLE_VERSION_CHECK=1 nohup python3 -u -m llamafactory.cli train examples/GRLM/shopping_preprocess_v4.yaml > ./logs/preprocess.out 2>&1 &

nohup bash -c 'DATA="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/sft_data/combined_sft.jsonl"; while true; do if [ -f "$DATA" ] && [ $(wc -l < "$DATA" 2>/dev/null || echo 0) -ge 1000 ]; then echo "$(date): Data ready ($(wc -l < "$DATA") lines), starting training..."; cd /scratch/workspaceblobstore/users/xiaoyukou/LLaMA-Factory && bash ./examples/GRLM/shopping_sft.sh; break; fi; echo "$(date): Waiting for data..."; sleep 60; done' > ./logs/shopping_sft_v2.log 2>&1 &
nohup bash -c 'while [ ! -f /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Results/qwen3-5-9b_lora_v2/checkpoint-8000/trainer_state.json ]; do sleep 10; done; sleep 300; pkill -f "llamafactory.cli train"; sleep 300; bash ./examples/GRLM/shopping_sft_continue.sh' > ./logs/shopping_sft_v2_continue_bs8.log 2>&1 &
