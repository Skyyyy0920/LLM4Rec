#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate base

################################################
######     Daily incremental training     ######
################################################
num_epochs=6
learning_rate=2e-6
num_neg_sample=31
lr_scheduler_type="constant"
warmup_ratio=0
batch_size=16

# current_date=$(date +%Y%m%d)
# yesterday=$(date -d "yesterday" +%Y%m%d)
current_date="20250525"
yesterday="20250524"
start_date=$current_date
end_date=$current_date

log_dir="/mnt/data/LLM_emb_recall/ckpt/${end_date}"
ckpt_dir="/mnt/data/LLM_emb_recall/ckpt/${yesterday}"
mkdir -p "$log_dir"
log_file="${log_dir}/incremental_train.log"

torchrun --nproc_per_node=6 --nnode=1 --master_port=29500 /mnt/data/LLM_emb_recall/main.py \
--learning_rate="$learning_rate" \
--lr_scheduler_type="$lr_scheduler_type" \
--num_epochs="$num_epochs" \
--warmup_ratio="$warmup_ratio" \
--num_neg_sample="$num_neg_sample" \
--batch_size="$batch_size" \
--start_date="$start_date" \
--end_date="$end_date" \
--incremental_training \
--ckpt_dir="$ckpt_dir" >> "$log_file" 2>&1