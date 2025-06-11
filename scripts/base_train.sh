#!/bin/bash

num_epochs=6
learning_rate=5e-6
num_neg_sample=63
lr_scheduler_type="constant"
warmup_ratio=0
batch_size=16
start_date="20250516"
end_date="20250520"

log_dir="/mnt/data/LLM_emb_recall/ckpt/${end_date}"
mkdir -p "$log_dir"
log_file="${log_dir}/base_train.log"

torchrun --nproc_per_node=6 --nnode=1 --master_port=29500 /mnt/data/LLM_emb_recall/main.py \
--learning_rate="$learning_rate" \
--lr_scheduler_type="$lr_scheduler_type" \
--num_epochs="$num_epochs" \
--warmup_ratio="$warmup_ratio" \
--num_neg_sample="$num_neg_sample" \
--batch_size="$batch_size" \
--start_date="$start_date" \
--end_date="$end_date" >> "$log_file" 2>&1
