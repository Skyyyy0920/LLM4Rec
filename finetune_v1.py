import os
import torch
import logging
import argparse
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
from embedding import build_item_prompt, build_user_prompt, generate_item_embs, generate_user_embs
from utils import compute_hit_rate, build_faiss_index_from_embeddings, setup_logger


def init_model(args):
    model = AutoModelForCausalLM.from_pretrained(f"Qwen/{args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{args.model_name}")
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
            item = self.item_df.iloc[0]  # TODO

        return {
            'sample': sample,
            'item': item
        }


class RecallDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, features):
        messages_list = [
            [
                {"role": "system", "content": "在个性化内容推荐场景中，根据用户信息，生成个性化内容推荐。"},
                {"role": "user", "content": build_user_prompt(feat['sample'])}
            ] for feat in features
        ]
        texts = [self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]

        user_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        item_messages_list = [
            [
                {"role": "system", "content": "在个性化内容推荐场景中，根据内容信息推断潜在客户信息（如性别、内容偏好等），并向用户推荐。"},
                {"role": "user", "content": build_item_prompt(feat['item'])}
            ] for feat in features
        ]
        item_texts = [self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in item_messages_list]

        item_inputs = self.tokenizer(item_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)

        input_ids = item_inputs["input_ids"]  # [batch_size, seq_len]
        attention_mask = item_inputs["attention_mask"]  # [batch_size, seq_len]
        neg_inputs = []
        batch_size = len(features)
        for i in range(batch_size):
            other_indices = [j for j in range(batch_size) if j != i]
            neg_input_ids = input_ids[other_indices]  # [batch_size-1, seq_len]
            neg_attention_mask = attention_mask[other_indices]  # [batch_size-1, seq_len]
            
            neg_inputs.append({
                "input_ids": neg_input_ids,
                "attention_mask": neg_attention_mask
            })

        return {
            'user_inputs': user_inputs,        # shape: [batch_size, seq_len]
            'pos_item_inputs': item_inputs,    # shape: [batch_size, seq_len]
            'neg_item_inputs': neg_inputs,     # shape: [batch_size, batch_size-1, seq_len]
        }
    
class RecallTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        user_outputs = model(
            input_ids=inputs["user_inputs"]["input_ids"], 
            attention_mask=inputs["user_inputs"]["attention_mask"], 
            output_hidden_states=True
        )
        user_embs = user_outputs.hidden_states[-1].mean(dim=1)  # [batch_size, hidden_dim]
        # batch_emb = outputs.hidden_states[-1][:, -1, :]
        # batch_emb = batch_emb.float().cpu().numpy()  # [batch_size, hidden_dim]

        pos_item_outputs = model(
            input_ids=inputs["pos_item_inputs"]["input_ids"], 
            attention_mask=inputs["pos_item_inputs"]["attention_mask"],
            output_hidden_states=True
        )
        pos_item_embs = pos_item_outputs.hidden_states[-1].mean(dim=1)

        neg_input_ids = torch.cat([x["input_ids"] for x in inputs["neg_item_inputs"]], dim=0)
        neg_attention_mask = torch.cat([x["attention_mask"] for x in inputs["neg_item_inputs"]], dim=0)
        neg_item_outputs = model(
            input_ids=neg_input_ids, 
            attention_mask=neg_attention_mask,
            output_hidden_states=True
        )
        neg_item_embs = neg_item_outputs.hidden_states[-1].mean(dim=1)
        
        pos_scores = torch.sum(user_embs * pos_item_embs, dim=-1)  # [batch_size]
        neg_scores = torch.sum(user_embs.unsqueeze(1) * neg_item_embs, dim=-1)  # [batch_size, batch_size-1]
        # print("min/max", pos_scores.min(), pos_scores.max())
        # print("min/max", neg_scores.min(), neg_scores.max())
        
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [batch_size, batch_size]
        
        def focal_loss(logits, gamma=2.0):
            labels = torch.zeros_like(logits)
            labels[:, 0] = 1.0  # 第一个位置是正样本
            
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            
            probs = torch.sigmoid(logits)
            pt = torch.where(labels == 1, probs, 1 - probs)
            focal_weight = (1 - pt) ** gamma
            
            return (focal_weight * bce_loss).mean()
        
        loss = focal_loss(logits, gamma=2.0)
        
        if return_outputs:
            return loss, {
                "user_embs": user_embs,
                "pos_item_embs": pos_item_embs,
                "neg_item_embs": neg_item_embs,
                "scores": logits
            }
        return loss

def train_recall_model(test_df, item_df, output_dir, args):
    base_model, tokenizer = init_model(args)
    
    dataset = RecallDataset(test_df, item_df)
    collator = RecallDataCollator(tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        bf16=True,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        remove_unused_columns=False,
        save_safetensors=True,
        gradient_accumulation_steps=4,
    )

    trainer = RecallTrainer(
        model=base_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    
    trainer.train()
    
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3-0.6B", help="预训练模型名称")
    parser.add_argument("--batch_size", type=int, default=8, help="批量大小")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="学习率")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="预热比例")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地设备编号（默认-1表示非分布式模式）")
    args = parser.parse_args()
    
    data_dir = "/mnt/data/LLM4Rec/data"
    item_df = pd.read_csv(os.path.join(data_dir, 'item_processed.csv'))
    item_df = item_df.dropna(subset=['publish_source'])
    item_df['item_id'] = item_df['item_id'].astype('int64')
    print(f"Successfully load item data.",len(item_df))
    # item_df = item_df[:1024]

    train_df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'), engine='pyarrow')
    test_df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'), engine='pyarrow')
    print(f"Successfully load train data and test data.",len(train_df), len(test_df))
    
    save_dir = f"/mnt/data/LLM4Rec/model/fine-tune/{args.model_name}/{args.lr_scheduler_type}/{str(args.num_epochs)}epoch_{str(args.learning_rate)}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    trainer = train_recall_model(train_df, item_df, save_dir, args)
    print("==================================  Finish Fine-tuning  ==================================")
    

if __name__ == "__main__":
    main()
