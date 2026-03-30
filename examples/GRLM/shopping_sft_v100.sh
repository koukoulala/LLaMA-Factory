MODEL_PATH=/scratch/workspaceblobstore/users/xiaoyukou/ckpts/Qwen3.5-4B
GPU_NUM=8
LORA_RANK=32
LEARNING_RATE=5e-5
CUTOFF_LEN=8192
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
SAVE_STEPS=1000
EVAL_STEPS=1000

# Activate conda environment
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda activate base
export LD_LIBRARY_PATH=$HOME/anaconda3/lib:$LD_LIBRARY_PATH
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
 
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m llamafactory.cli train \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --quantization_bit 4 \
    --lora_rank ${LORA_RANK} \
    --lora_target all \
    --dataset shopping_gen_rec_v2 \
    --template qwen3_5_nothink \
    --cutoff_len ${CUTOFF_LEN} \
    --max_samples 2000000 \
    --preprocessing_num_workers 40 \
    --dataloader_num_workers 8 \
    --tokenized_path /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Results/qwen3-5-4b_qlora_v2_v100/tokenized_dataset_8192 \
    --output_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Results/qwen3-5-4b_qlora_v2_v100/ \
    --logging_steps 20 \
    --save_steps ${SAVE_STEPS} \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs 2.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --fp16 \
    --ddp_timeout 180000000 \
    --val_size 0.001 \
    --per_device_eval_batch_size 2 \
    --eval_strategy steps \
    --eval_steps ${EVAL_STEPS} \
    --flash_attn sdpa \
    --packing true \
    --enable_liger_kernel true