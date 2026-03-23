MODEL_PATH=/scratch/workspaceblobstore/users/xiaoyukou/ckpts/Qwen3.5-9B
GPU_NUM=4
LORA_RANK=16
LEARNING_RATE=5e-5
CUTOFF_LEN=4096
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
SAVE_STEPS=2000
EVAL_STEPS=1000
 
DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m llamafactory.cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank ${LORA_RANK} \
    --lora_target all \
    --dataset shopping_gen_rec \
    --template qwen3_5 \
    --cutoff_len ${CUTOFF_LEN} \
    --max_samples 2000000 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Results/qwen3-5-9b_lora_sft_lr${LEARNING_RATE}_batch${BATCH_SIZE}_packing \
    --logging_steps 10 \
    --save_steps ${SAVE_STEPS} \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to tensorboard \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs 2.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000 \
    --val_size 0.001 \
    --per_device_eval_batch_size 8 \
    --eval_strategy steps \
    --eval_steps ${EVAL_STEPS} \
    --packing true \
   