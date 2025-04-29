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
from torch.distributed.tensor import DTensor
from embedding import build_item_prompt, build_user_prompt, generate_item_embs, generate_user_embs
from utils import compute_hit_rate, build_faiss_index_from_embeddings, setup_logger


def init_model(args):
    model = AutoModelForCausalLM.from_pretrained(f"Qwen/{args.model_name}", torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{args.model_name}")
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
                {"role": "system", "content": "在个性化内容推荐场景中，根据用户画像，生成个性化内容推荐。"},
                {"role": "user", "content": build_user_prompt(feat['sample'])}
            ] for feat in features
        ]
        user_texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in user_messages]
        user_inputs = self.tokenizer(user_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)

        item_messages = [
            [
                {"role": "system", "content": "在个性化内容推荐场景中，根据内容信息推断潜在客户信息（如性别、内容偏好等），并向用户推荐。"},
                {"role": "user", "content": build_item_prompt(feat['item'])}
            ] for feat in features
        ]
        item_texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in item_messages]
        item_inputs = self.tokenizer(item_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)

        return {
            'user_inputs': user_inputs,
            'item_inputs': item_inputs,
            'user_ids': [feat['sample']['user_id'] for feat in features],
            'item_ids': [feat['sample']['item_id'] for feat in features],
        }
    
class RecallTrainer(Trainer):
    def __init__(self, user_pos_items, **kwargs):
        super().__init__(**kwargs)
        self.user_pos_items = user_pos_items
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        user_outputs = model(
            input_ids=inputs["user_inputs"]["input_ids"],
            attention_mask=inputs["user_inputs"]["attention_mask"],
            output_hidden_states=True
        )
        user_embs = user_outputs.hidden_states[-1].mean(dim=1)
        # outputs.hidden_states[-1][:, -1, :]  TODO

        item_outputs = model(
            input_ids=inputs["item_inputs"]["input_ids"],
            attention_mask=inputs["item_inputs"]["attention_mask"],
            output_hidden_states=True
        )
        item_embs = item_outputs.hidden_states[-1].mean(dim=1)
        
        # user_embs = F.normalize(user_embs, p=2, dim=-1)  # [B, D]
        # item_embs = F.normalize(item_embs, p=2, dim=-1)  # [B, D]
        
        if dist.is_initialized():
            world_size = dist.get_world_size()
            gather_pos = [torch.zeros_like(item_embs) for _ in range(world_size)]
            dist.all_gather(gather_pos, item_embs)
            all_embs = torch.cat(gather_pos, dim=0)
            
            item_ids_tensor = torch.tensor(inputs["item_ids"], dtype=torch.long, device=item_embs.device)
            gather_ids = [torch.zeros_like(item_ids_tensor) for _ in range(world_size)]
            dist.all_gather(gather_ids, item_ids_tensor)
            all_item_ids = torch.cat(gather_ids, dim=0)
        else:
            all_embs = item_embs
            all_item_ids = torch.tensor(inputs["item_ids"], dtype=torch.long, device=item_embs.device)

        user_ids = inputs["user_ids"]
        batch_size = len(user_ids)
        num_candidates = all_item_ids.size(0)

        labels = torch.zeros(batch_size, num_candidates, device=all_embs.device)
        for i, user_id in enumerate(user_ids):
            pos_items = self.user_pos_items.get(user_id, set())
            if len(pos_items) > 0:
                pos_tensor = torch.tensor(list(pos_items), device=all_item_ids.device)
                mask = torch.isin(all_item_ids, pos_tensor)
                labels[i, mask] = 1.0

        logits = torch.matmul(user_embs, all_embs.T)  # [batch_size, num_GPUs*batch_size]
        
        # if dist.is_initialized():
        #     rank = dist.get_rank()
        #     # 生成类别标签，每个样本的正样本索引为 rank*batch_size + i
        #     labels = torch.arange(logits.size(0), device=user_embs.device) + rank * logits.size(0)
        # else:
        #     labels = torch.arange(logits.size(0), device=user_embs.device)

        # # Cross Entropy Loss
        # loss = F.cross_entropy(logits, labels)

        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(labels.bool(), probs, 1-probs)
        
        gamma = 2
        alpha = 0.25
        alpha_t = torch.where(labels.bool(), alpha, 1-alpha)
        focal_weight = alpha_t * (1 - pt) ** gamma

        loss = (focal_weight * bce_loss).mean()

        return (loss, (user_embs, item_embs, all_embs, logits)) if return_outputs else loss


def train_recall_model(train_df, item_df, user_pos_items, output_dir, args):
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
        user_pos_items=user_pos_items,
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
    parser.add_argument("--model_name", default="Qwen2.5-0.5B-instruct")  # Qwen2.5-0.5B-instruct  Qwen3-0.6B
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

    # train_df = pd.read_csv(os.path.join(data_dir, 'train_title.csv'))
    # test_df = pd.read_csv(os.path.join(data_dir, 'test_title.csv'))
    # print(f"Successfully load train data and test data.",len(train_df), len(test_df))
    
    # train_df["city"] = train_df["city"].fillna("未知城市")
    # test_df["city"] = test_df["city"].fillna("未知城市")
    # # str_to_dict_cols = [
    # #     "camp_id_list", "category_id_list", "category_name_list", 
    # #     "follow_user_id_list", "comment_author_list",
    # #     "click_50_seq__item_id", "click_50_seq__category",
    # #     "click_50_seq__author", "click_50_seq__theme_id"
    # # ]
    # str_to_dict_cols = ['category_name_list', 'click_50_seq__category', 'click_50_seq__item_title']
    # for col in str_to_dict_cols:
    #     train_df[col] = train_df[col].apply(lambda x: literal_eval(x) if pd.notna(x) else {})
    #     test_df[col] = test_df[col].apply(lambda x: literal_eval(x) if pd.notna(x) else {})
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
