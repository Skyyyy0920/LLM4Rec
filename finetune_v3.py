import os
import torch
import logging
import argparse
import random
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from ast import literal_eval
from transformers import Trainer, TrainingArguments
from modelscope import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from embedding import build_item_prompt, build_user_prompt, generate_item_embs, generate_user_embs, SYSTEM_USER_PROMPT, SYSTEM_ITEM_PROMPT
from utils import compute_hit_rate, build_faiss_index_from_embeddings, setup_logger


def init_model(args):
    model = AutoModelForCausalLM.from_pretrained(f"Qwen/{args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{args.model_name}")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[USER]", "[ITEM]"]})
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

    
class RecallDataset(Dataset):
    def __init__(self, sample_df, item_df):
        self.sample_df = sample_df
        self.item_df = item_df
        self.item_df = self.item_df.set_index('item_id')
        
    def __len__(self):
        return len(self.sample_df)
    
    def __getitem__(self, idx):
        sample = self.sample_df.iloc[idx]
        try:
            item = self.item_df.loc[sample['item_id']]
        except KeyError:
            item = self.item_df.iloc[0]

        return {
            'sample': sample,
            'item': item
        }


class RecallDataCollator:
    def __init__(self, tokenizer, item_df, num_negatives=7):
        self.tokenizer = tokenizer
        self.item_df = item_df
        self.num_negatives = num_negatives
        self.all_item_ids = self.item_df.index.tolist()
        
    def __call__(self, features):
        batch_size = len(features)
        
        user_messages = [
            [
                {"role": "system", "content": SYSTEM_USER_PROMPT},
                {"role": "user", "content": build_user_prompt(feat['sample'])}
            ] for feat in features
        ]
        user_texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in user_messages]
        user_inputs = self.tokenizer(user_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        positive_item_messages = [
            [
                {"role": "system", "content": SYSTEM_ITEM_PROMPT},
                {"role": "user", "content": build_item_prompt(feat['item'])}
            ] for feat in features
        ]
        positive_item_texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in positive_item_messages]
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
        negative_item_texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in negative_item_messages]
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
        user_embs = user_outputs.hidden_states[-1][:, -1, :]  # [B, D] 取序列最后一个token的embedding
        # user_embs = user_outputs.hidden_states[-1].mean(dim=1)  # [B, D]

        # 正样本物品表征提取
        positive_item_outputs = model(
            input_ids=inputs["positive_item_inputs"]["input_ids"],
            attention_mask=inputs["positive_item_inputs"]["attention_mask"],
            output_hidden_states=True
        )
        positive_item_embs = positive_item_outputs.hidden_states[-1][:, -1, :]  # [B, D]
        # positive_item_embs = positive_item_outputs.hidden_states[-1].mean(dim=1)  # [B, D]

        # 负样本物品表征提取
        negative_item_outputs = model(
            input_ids=inputs["negative_item_inputs"]["input_ids"],
            attention_mask=inputs["negative_item_inputs"]["attention_mask"],
            output_hidden_states=True
        )
        negative_item_embs = negative_item_outputs.hidden_states[-1][:, -1, :]  # [N, D]
        # negative_item_embs = negative_item_outputs.hidden_states[-1].mean(dim=1)  # [N, D]    
        
        # user_embs = F.normalize(user_embs, p=2, dim=-1)
        # positive_item_embs = F.normalize(positive_item_embs, p=2, dim=-1)
        # negative_item_embs = F.normalize(negative_item_embs, p=2, dim=-1)
        
        batch_size = user_embs.size(0)
        num_negatives = negative_item_embs.size(0)
        
        all_item_embs = torch.cat([positive_item_embs.unsqueeze(1), 
                                  negative_item_embs.unsqueeze(0).expand(batch_size, -1, -1)], dim=1)  # [B, 1+N, D]
        
        expanded_user_embs = user_embs.unsqueeze(1).expand(-1, 1+num_negatives, -1)  # [B, 1+N, D]
        
        logits = torch.sum(expanded_user_embs * all_item_embs, dim=2)  # [B, 1+N]
        
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
    model, tokenizer = init_model(args)
    
    dataset = RecallDataset(train_df, item_df)
    collator = RecallDataCollator(tokenizer, item_df)
    
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
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    data_dir = "/mnt/data/LLM4Rec/data"
    item_df = pd.read_csv(os.path.join(data_dir, 'item_processed.csv'))
    item_df = item_df.dropna(subset=['publish_source'])
    item_df['item_id'] = item_df['item_id'].astype('int64')
    print(f"Successfully load item data.",len(item_df))

    train_df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'), engine='pyarrow')
    test_df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'), engine='pyarrow')
    print(f"Successfully load train data and test data.",len(train_df), len(test_df))
    
    user_pos_items = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    save_dir = f"/mnt/data/LLM4Rec/model/fine-tune/{args.model_name}/{args.lr_scheduler_type}/{str(args.num_epochs)}epoch_{str(args.learning_rate)}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    trainer = train_recall_model(train_df, item_df, user_pos_items, save_dir, args)
    print("==================================  Finish Fine-tuning  ==================================")


if __name__ == "__main__":
    main()
