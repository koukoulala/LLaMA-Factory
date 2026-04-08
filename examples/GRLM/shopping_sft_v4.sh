MODEL_PATH=/scratch/workspaceblobstore/users/xiaoyukou/ckpts/Qwen3.5-9B
GPU_NUM=4
LORA_RANK=64
LEARNING_RATE=5e-5
CUTOFF_LEN=20000
BATCH_SIZE=12
GRADIENT_ACCUMULATION_STEPS=4
SAVE_STEPS=50
 
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u -m llamafactory.cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank ${LORA_RANK} \
    --lora_target all \
    --dataset shopping_gen_rec_v4 \
    --template qwen3_5_nothink \
    --cutoff_len ${CUTOFF_LEN} \
    --max_samples 2000000 \
    --preprocessing_num_workers 40 \
    --dataloader_num_workers 8 \
    --tokenized_path /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Results/qwen3-5-9b_lora_v4/full_tokenized_dataset \
    --output_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Results/qwen3-5-9b_lora_v4/ \
    --resume_from_checkpoint /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Results/qwen3-5-9b_lora_v4/checkpoint-500 \
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
    --warmup_ratio 0.02 \
    --bf16 \
    --ddp_timeout 180000000 \
    --eval_strategy no \
    --packing true \
    --enable_liger_kernel true \
   