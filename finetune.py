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
from transformers import Trainer, TrainingArguments
from modelscope import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from embedding import build_item_prompt, build_user_prompt
from utils import generate_item_embs, generate_user_embs, compute_hit_rate, build_faiss_index_from_embeddings, setup_logger


def init_model(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
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
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, features):
        user_messages = [
            [
                {"role": "system", "content": "在个性化内容推荐场景中，根据用户信息，生成个性化内容推荐。"},
                {"role": "user", "content": build_user_prompt(feat['sample'])}
            ] for feat in features
        ]
        user_texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in user_messages]
        user_inputs = self.tokenizer(user_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)

        item_messages = [
            [
                {"role": "system", "content": "在个性化内容推荐场景中，根据内容信息推断潜在客户信息（如性别、内容偏好等），并向用户推荐。"},
                {"role": "user", "content": build_item_prompt(feat['item'])}
            ] for feat in features
        ]
        item_texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in item_messages]
        item_inputs = self.tokenizer(item_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)

        return {
            'user_inputs': user_inputs,
            'pos_item_inputs': item_inputs
        }
    
class RecallTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        user_outputs = model(
            input_ids=inputs["user_inputs"]["input_ids"],
            attention_mask=inputs["user_inputs"]["attention_mask"],
            output_hidden_states=True
        )
        user_embs = user_outputs.hidden_states[-1].mean(dim=1)
        # outputs.hidden_states[-1][:, -1, :]  TODO

        pos_outputs = model(
            input_ids=inputs["pos_item_inputs"]["input_ids"],
            attention_mask=inputs["pos_item_inputs"]["attention_mask"],
            output_hidden_states=True
        )
        pos_embs = pos_outputs.hidden_states[-1].mean(dim=1)
        
        # user_embs = F.normalize(user_embs, p=2, dim=-1)  # [B, D]
        # pos_embs = F.normalize(pos_embs, p=2, dim=-1)    # [B, D]

        # 分布式收集所有正样本
        if dist.is_initialized():
            world_size = dist.get_world_size()
            gather_pos = [torch.zeros_like(pos_embs) for _ in range(world_size)]
            dist.all_gather(gather_pos, pos_embs)
            all_pos = torch.cat(gather_pos, dim=0)
            # print(f"Rank {dist.get_rank()}: all_pos.shape={all_pos.shape}")
        else:
            all_pos = pos_embs

        # 相似度计算
        pos_sim = torch.sum(user_embs * pos_embs, dim=-1)  # [B]
        neg_sim = torch.matmul(user_embs, all_pos.T)  # [B, N*B]
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1 + num_GPUs × batch_size]

        labels = torch.zeros_like(logits)
        labels[:, 0] = 1.0
        # 排除自身正样本
        batch_size = user_embs.size(0)
        if dist.is_initialized():
            rank = dist.get_rank()
            mask_idx = torch.arange(rank*batch_size, (rank+1)*batch_size, device=user_embs.device)
        else:
            mask_idx = torch.arange(batch_size, device=user_embs.device)
        labels[torch.arange(batch_size, device=mask_idx.device), mask_idx] = 1.0

        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(labels.bool(), probs, 1-probs)
        gamma = 2
        focal_weight = (1-pt) ** gamma
        loss = (focal_weight * bce_loss).mean()

        return (loss, (user_embs, pos_embs, all_pos, logits)) if return_outputs else loss


def train_recall_model(train_df, item_df, output_dir, args):
    model, tokenizer = init_model(args)
    
    dataset = RecallDataset(train_df, item_df)
    collator = RecallDataCollator(tokenizer)
    
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
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-instruct")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    data_dir = "/mnt/data/LLM4Rec/data"
    item_df = pd.read_csv(os.path.join(data_dir, 'item_test.csv'))
    item_df = item_df.dropna(subset=['publish_source'])
    item_df['item_id'] = item_df['item_id'].astype('int64')
    print(f"Successfully load item data.",len(item_df))

    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    print(f"Successfully load train data and test data.",len(train_df), len(test_df))
    
    save_dir = f"/mnt/data/LLM4Rec/model/fine-tune/Qwen2.5-0.5B/{args.lr_scheduler_type}/{str(args.num_epochs)}epoch_{str(args.learning_rate)}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    trainer = train_recall_model(train_df, item_df, save_dir, args)
    print("==================================  Finish Fine-tuning  ==================================")

    # 评估（需根据实际情况调整）
    if args.local_rank in [-1, 0]:  # 仅在主进程评估
        test_df = pd.read_csv(f"{data_dir}/test.csv")
        model = trainer.model.cuda()
        
        # 生成item embedding
        item_ids, item_embs = generate_item_embs(item_df, model, trainer.tokenizer, 'cuda', 256)
        index, id_map = build_faiss_index_from_embeddings(item_ids, item_embs)
        
        # 生成user embedding并评估
        user_ids, user_embs = generate_user_embs(test_df, model, trainer.tokenizer, 'cuda', 64)
        hit_rate, _ = compute_hit_rate(user_embs, user_ids, index, id_map, test_df['item_id'].values)
        print(f"Final Hit Rate @200: {hit_rate:.4f}")


if __name__ == "__main__":
    main()
