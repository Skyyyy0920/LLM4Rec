#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate base

# yesterday=$(date -d "yesterday" +%Y%m%d)
# data_date=$(date +%Y%m%d)
data_date="20250525"
yesterday="20250523"

# ========== 用户配置区域 ==========
script_path="/mnt/data/LLM_emb_recall/daily_eval.py"
model_dir="/mnt/data/LLM_emb_recall/ckpt/${yesterday}"
log_dir="/mnt/data/LLM_emb_recall/ckpt/all/${yesterday}"
mkdir -p "$log_dir"

all_devices="0 1 2 3 4 5"
parallel_limit=6

# 自动查找所有 checkpoint-* 目录并提取数字
mapfile -t checkpoints <<< "$(find "$model_dir" -maxdepth 1 -type d -name 'checkpoint-*' | \
    sed -E 's|.*/checkpoint-([0-9]+)$|\1|' | sort -n)"

# 转换为数组
devices=($all_devices)

current_jobs=0
total=${#checkpoints[@]}
for ((i=0; i<total; i++)); do
    device_idx=$((i % ${#devices[@]}))
    gpu_id=${devices[$device_idx]}
    
    log_file="${log_dir}/eval_ckpt_${checkpoints[$i]}_gpu${gpu_id}.log"
    
    # 执行评估命令
    CUDA_VISIBLE_DEVICES=$gpu_id \
    python $script_path \
        --ckpt=${checkpoints[$i]} \
        --model_dir=$model_dir \
        --data_date=$data_date > "$log_file" 2>&1 &
    
    ((current_jobs++))
    if [ $current_jobs -ge $parallel_limit ]; then
        wait -n
        ((current_jobs--))
    fi
done

wait
echo "所有评估任务已完成！"