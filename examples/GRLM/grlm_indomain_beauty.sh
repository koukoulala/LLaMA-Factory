model_path=/data/xiaoyukou/ckpts/Qwen3-4B-Instruct-2507
output_dir=./saves/grlm/indomain_beauty

deepspeed --num_gpus 2 \
src/train.py \
--deepspeed examples/deepspeed/ds_z3_config.json \
--stage sft \
--model_name_or_path $model_path \
--do_train \
--dataset grlm_indomain_beauty \
--template qwen3 \
--finetuning_type full \
--output_dir  $output_dir \
--overwrite_cache \
--save_total_limit 1 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 200 \
--learning_rate 1e-4 \
--num_train_epochs 3.0 \
--plot_loss \
--bf16