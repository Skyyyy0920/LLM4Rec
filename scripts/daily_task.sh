#!/bin/bash

cd /mnt/data/LLM_emb_recall/scripts || exit

# 记录开始时间
echo "[$(date)] 开始执行每日任务..." >> /mnt/data/LLM_emb_recall/scripts/daily_task.log

# 执行评估
echo "[$(date)] 正在运行评估..." >> /mnt/data/LLM_emb_recall/scripts/daily_task.log
./daily_eval.sh

if [ $? -eq 0 ]; then
    echo "[$(date)] 评估完成，开始增量训练..." >> /mnt/data/LLM_emb_recall/scripts/daily_task.log
else
    echo "[$(date)] 评估失败，开始增量训练..." >> /mnt/data/LLM_emb_recall/scripts/daily_task.log
fi
./daily_train.sh
if [ $? -eq 0 ]; then
    echo "[$(date)] 训练完成。" >> /mnt/data/LLM_emb_recall/scripts/daily_task.log
else
    echo "[$(date)] 训练失败！" >> /mnt/data/LLM_emb_recall/scripts/daily_task.log
fi