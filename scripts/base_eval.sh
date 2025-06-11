#!/bin/bash

# ========== 用户配置区域 ==========
# 检查点参数
start_ckpt=1152      # 起始检查点基数
num_ckpts=16          # 生成的检查点总数
step_multiplier=1    # 步长乘数

# GPU配置（改为空格分隔的字符串）
all_devices="0 1 2 3 4 5 6 7"
parallel_limit=8

# 路径配置
script_path="/mnt/data/LLM4Rec/evaluate.py"
log_dir="/mnt/data/LLM4Rec/eval_logs_8epoch_2e-06_15_v4"
# ================================

# 预检查设备数量
if [ $(echo $all_devices | wc -w) -ne 8 ]; then
    echo "错误：设备数量不是8张"
    exit 1
fi

# 生成检查点列表
# ckpts=""
# for i in $(seq 1 $num_ckpts); do
#     ckpts="$ckpts $(( start_ckpt * step_multiplier * i ))"
# done
ckpts="1152 2304 3456 4608 5760 6912 8064 9208"

# 创建日志目录
mkdir -p "$log_dir"

# 转换为数组
devices=($all_devices)
checkpoints=($ckpts)

# 任务计数器
current_jobs=0

# 执行任务（使用数字索引）
total=${#checkpoints[@]}
for ((i=0; i<total; i++)); do  # 注意这里改为传统数字索引
    # 计算设备索引
    device_idx=$((i % ${#devices[@]}))
    gpu_id=${devices[$device_idx]}
    
    # 构造日志路径
    log_file="${log_dir}/ckpt_${checkpoints[$i]}_gpu${gpu_id}.log"
    
    # 执行命令
    CUDA_VISIBLE_DEVICES=$gpu_id \
    python $script_path --ckpt=${checkpoints[$i]} > "$log_file" 2>&1 &
    
    # 并发控制
    ((current_jobs++))
    if [ $current_jobs -ge $parallel_limit ]; then
        wait -n
        ((current_jobs--))
    fi
done

wait
echo "所有任务已完成！"