import os
import torch
import argparse
import numpy as np
import pandas as pd
from modelscope import AutoModelForCausalLM, AutoTokenizer
from embedding import build_item_prompt, build_user_prompt, generate_item_embs, generate_user_embs
from utils import compute_hit_rate, build_faiss_index_from_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="1151")
    args = parser.parse_args()
    print(args.ckpt)
    
    data_dir = "/mnt/data/LLM4Rec/data"
    item_df = pd.read_csv(os.path.join(data_dir, 'item_processed.csv'))
    item_df = item_df.dropna(subset=['publish_source'])
    item_df['item_id'] = item_df['item_id'].astype('int64')
    print(f"Successfully load item data.",len(item_df))

    test_df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'), engine='pyarrow')

    model = AutoModelForCausalLM.from_pretrained(f"/mnt/data/LLM4Rec/model/fine-tune/Qwen3-0.6B/constant/v4_1epoch_2e-06_15/checkpoint-{args.ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(f"/mnt/data/LLM4Rec/model/fine-tune/Qwen3-0.6B/constant/v4_1epoch_2e-06_15/checkpoint-{args.ckpt}")
    model.to('cuda')
    
    # model = AutoModelForCausalLM.from_pretrained(f"Qwen/Qwen3-0.6B")
    # tokenizer = AutoTokenizer.from_pretrained(f"Qwen/Qwen3-0.6B")
    # model.to('cuda')
    
    # model = AutoModelForCausalLM.from_pretrained(f"Qwen/Qwen2.5-0.5B-instruct")
    # tokenizer = AutoTokenizer.from_pretrained(f"Qwen/Qwen2.5-0.5B-instruct")
    # model.to('cuda')

    item_index_ids, item_embs = generate_item_embs(
        df=item_df,
        model=model,
        tokenizer=tokenizer,
        device='cuda',
        batch_size=32,
        use_amp=True
    )

    index, index_id_map = build_faiss_index_from_embeddings(
        index_id_map=item_index_ids,
        embeddings=item_embs,
    )
    index.nprobe = 800

    user_index_ids, user_embs = generate_user_embs(
        df=test_df,
        model=model,
        tokenizer=tokenizer,
        device='cuda',
        batch_size=16,
        use_amp=True
    )
    
    hit_rate, sampled_recalls = compute_hit_rate(
        user_embs=user_embs,
        user_index_ids=user_index_ids,
        index=index,
        index_id_map=index_id_map,
        gt_mapping=test_df['item_id'].to_numpy(),
        top_k=200,
        batch_size=512
    )
    print(hit_rate)
    

if __name__ == "__main__":
    main()