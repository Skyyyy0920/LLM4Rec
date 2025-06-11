import os
import re
import torch
import random
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path
from ast import literal_eval
from odps import ODPS
from odps.accounts import AliyunAccount
from transformers import Trainer, TrainingArguments
from modelscope import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset
from embedding import *
from utils import *

    
class RecallDataset(Dataset):
    def __init__(self, sample_df, item_df):
        self.sample_df = sample_df
        
    def __len__(self):
        return len(self.sample_df)
    
    def __getitem__(self, idx):
        sample = self.sample_df.iloc[idx]

        return {
            'sample': sample,
        }


class RecallDataCollator:
    def __init__(self, tokenizer, item_df, num_negatives):
        self.tokenizer = tokenizer
        self.item_df = item_df
        self.num_negatives = num_negatives
        self.all_item_ids = self.item_df.index.tolist()
        
    def __call__(self, features):        
        user_messages = [
            [
                {"role": "system", "content": SYSTEM_USER_PROMPT},
                {"role": "user", "content": build_user_prompt(feat['sample'])}
            ] for feat in features
        ]
        user_texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in user_messages]
        user_inputs = self.tokenizer(user_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        positive_item_messages = [
            [
                {"role": "system", "content": SYSTEM_ITEM_PROMPT},
                {"role": "user", "content": build_item_prompt(feat['sample'])}
            ] for feat in features
        ]
        positive_item_texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in positive_item_messages]
        positive_item_inputs = self.tokenizer(positive_item_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        
        positive_item_ids = [feat['sample']['item_id'] for feat in features]
        
        negative_item_ids = []
        available_items = [item_id for item_id in self.all_item_ids if item_id not in positive_item_ids]
        if len(available_items) >= self.num_negatives:
            negative_item_ids = random.sample(available_items, self.num_negatives)
        else:
            negative_item_ids = random.choices(available_items, k=self.num_negatives)
        
        negative_items = [self.item_df.loc[item_id] for item_id in negative_item_ids]
        
        negative_item_messages = [
            [
                {"role": "system", "content": SYSTEM_ITEM_PROMPT},
                {"role": "user", "content": build_item_prompt(item)}
            ] for item in negative_items
        ]
        negative_item_texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in negative_item_messages]
        negative_item_inputs = self.tokenizer(negative_item_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        
        return {
            'user_inputs': user_inputs,
            'positive_item_inputs': positive_item_inputs,
            'negative_item_inputs': negative_item_inputs,
        }


class RecallTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):      
        user_outputs = model(
            input_ids=inputs["user_inputs"]["input_ids"],
            attention_mask=inputs["user_inputs"]["attention_mask"],
            output_hidden_states=True
        )
        user_embs = user_outputs.hidden_states[-1].mean(dim=1)  # [B, D]

        positive_item_outputs = model(
            input_ids=inputs["positive_item_inputs"]["input_ids"],
            attention_mask=inputs["positive_item_inputs"]["attention_mask"],
            output_hidden_states=True
        )
        positive_item_embs = positive_item_outputs.hidden_states[-1].mean(dim=1)  # [B, D]

        negative_item_outputs = model(
            input_ids=inputs["negative_item_inputs"]["input_ids"],
            attention_mask=inputs["negative_item_inputs"]["attention_mask"],
            output_hidden_states=True
        )
        negative_item_embs = negative_item_outputs.hidden_states[-1].mean(dim=1)  # [N, D]    
        
        # all_item_embs = torch.cat([positive_item_embs.unsqueeze(1), negative_item_embs.unsqueeze(0).expand(batch_size, -1, -1)], dim=1)  # [B, 1+N, D]
        # expanded_user_embs = user_embs.unsqueeze(1).expand(-1, 1+num_negatives, -1)  # [B, 1+N, D]
        # logits = torch.sum(expanded_user_embs * all_item_embs, dim=2)  # [B, 1+N]
        positive_logits = (user_embs * positive_item_embs).sum(dim=1, keepdim=True)    # [B, 1]
        negative_logits = torch.matmul(user_embs, negative_item_embs.T)                # [B, N]
        logits = torch.cat([positive_logits, negative_logits], dim=1)                  # [B, 1+N]
        
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1.0
        
        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(labels.bool(), probs, 1-probs)
        
        gamma = 2
        alpha = 0.25
        alpha_t = torch.where(labels.bool(), alpha, 1-alpha)
        focal_weight = alpha_t * (1 - pt) ** gamma
        
        loss = (focal_weight * bce_loss).mean()
        
        outputs = (user_embs, positive_item_embs, negative_item_embs, logits)
        return (loss, outputs) if return_outputs else loss
    

def train_recall_model(train_df, item_df, user_pos_items, output_dir, args):
    if args.incremental_training:  # 增量更新
        checkpoints = [
            d for d in os.listdir(args.ckpt_dir)
            if re.match(r'checkpoint-\d+', d)
        ]
        latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r'\d+', x).group()))
        latest_ckpt_path = os.path.join(args.ckpt_dir, latest_checkpoint)

        print(f"Load from {latest_ckpt_path}, start incremental training...")
        model = AutoModelForCausalLM.from_pretrained(latest_ckpt_path)
        tokenizer = AutoTokenizer.from_pretrained(latest_ckpt_path)
    else:
        print(f"Load from Qwen/{args.model_name}, start base training...")
        model = AutoModelForCausalLM.from_pretrained(f"Qwen/{args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{args.model_name}")
    
    dataset = RecallDataset(train_df, item_df)
    collator = RecallDataCollator(tokenizer, item_df, args.num_neg_sample)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        bf16=True,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=20,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.grad_accum,
    )

    trainer = RecallTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    
    trainer.train()
    
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen3-0.6B")  # Qwen2.5-0.5B-instruct  Qwen3-0.6B
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_neg_sample", type=int, default=31)
    parser.add_argument("--start_date", type=str, default="20250515")
    parser.add_argument("--end_date", type=str, default="20250519")
    parser.add_argument("--incremental_training", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default="")
    args = parser.parse_args()
    print(args)
    
    #######################################################################################################################################
    ########################################                      Load data                        ########################################
    #######################################################################################################################################
    access_id = ''
    access_key = ''
    project_name = ''
    endpoint = 'http://service.odps.aliyun.com/api'

    account = AliyunAccount(access_id, access_key)
    o = ODPS(account=account, project=project_name, endpoint=endpoint)
    n_process = multiprocessing.cpu_count()
    
    t_item = o.get_table('rec_sln_demo_item_table_v1_filter')
    with t_item.open_reader(partition='ds='+args.end_date, arrow=True) as reader:
        item_df = reader.to_pandas(n_process=n_process)
    print(f"load item length: {len(item_df)}")
    
    t_sample = o.get_table('rec_sln_demo_sample_table_v1_filter')
    dfs = []
    dates = pd.date_range(start=args.start_date, end=args.end_date)
    partitions = [f'ds={date.strftime("%Y%m%d")}' for date in dates]
    for partition in partitions:
        with t_sample.open_reader(partition=partition, arrow=True) as reader:
            df_part = reader.to_pandas(n_process=n_process)
            dfs.append(df_part)
    sample_df = pd.concat(dfs, ignore_index=True)
    print(f"load sample length: {len(sample_df)}")
    
    #######################################################################################################################################
    ########################################                      Preprocess                       ########################################
    #######################################################################################################################################
    count = sample_df[~sample_df['item_id'].isin(item_df['item_id'])].shape[0]
    print(f"sample_df 中有 {count} 行的 item_id 不在 item_df 中")
    sample_df = sample_df[sample_df['item_id'].isin(item_df['item_id'])]  # 去掉不在item_df中的sample数据
    
    sample_df["city"] = sample_df["city"].fillna("未知城市")
    sample_df['click_50_seq__item_id'] = sample_df['click_50_seq__item_id'].fillna('').apply(lambda x: [item for item in x.split(';')] if x else [])
    sample_df['click_50_seq__category'] = sample_df['click_50_seq__category'].fillna('').apply(lambda x: [item for item in x.split(';')] if x else [])
    # sample_df['camp_id_list'] = sample_df['camp_id_list'].fillna('').apply(lambda x: [item for item in x.split(',')] if x else [])
    # sample_df['category_id_list'] = sample_df['category_id_list'].fillna('').apply(lambda x: [item for item in x.split(',')] if x else [])
    sample_df['category_name_list'] = sample_df['category_name_list'].fillna('').apply(lambda x: [item for item in x.split(',')] if x else [])
    
    def filter_chinese_digits(text):
        text = re.sub(r'<[^>]+>', '', text)
        return re.sub(r'[^\u4e00-\u9fa5]', '', text)
    for df in [sample_df, item_df]:
        df["content_title"] = df["content"].fillna('').apply(filter_chinese_digits)
        df["content_title"] = df["content_title"].str[:10]
        mask = df["title"].isna() | (df["title"].astype(str).str.strip() == "")
        df["title"] = np.where(mask, df["content_title"], df["title"])
    
    title_map = item_df.set_index('item_id')['title'].fillna('').to_dict()
    def convert_ids_to_titles(id_list):
        return [title_map.get(item_id, '') for item_id in id_list]
    sample_df['click_50_seq__item_title'] = sample_df['click_50_seq__item_id'].apply(convert_ids_to_titles)
    # sample_df['camp_title_list'] = sample_df['camp_id_list'].apply(convert_ids_to_titles)

    user_pos_items = sample_df.groupby('user_id')['item_id'].apply(set).to_dict()

    #######################################################################################################################################
    ########################################                      Training                         ########################################
    #######################################################################################################################################
    save_dir = f"/mnt/data/LLM_emb_recall/ckpt/{args.end_date}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    train_recall_model(sample_df, item_df, user_pos_items, save_dir, args)
    print("==================================  Finish Fine-tuning  ==================================")


if __name__ == "__main__":
    main()