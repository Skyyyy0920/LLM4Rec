import os
import torch
import numpy as np
import pandas as pd
from modelscope import AutoModelForCausalLM, AutoTokenizer
from main import generate_item_embs, generate_user_embs, compute_hit_rate, build_faiss_index_from_embeddings


def main():
    data_dir = "/mnt/data/LLM4Rec/data"
    item_df = pd.read_csv(os.path.join(data_dir, 'item_test.csv'))
    item_df = item_df.dropna(subset=['publish_source'])
    item_df['item_id'] = item_df['item_id'].astype('int64')
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    model = AutoModelForCausalLM.from_pretrained("/mnt/data/LLM4Rec/model/fine-tune/Qwen2.5-0.5B/cosine/1epoch/checkpoint-9211")
    tokenizer = AutoTokenizer.from_pretrained("/mnt/data/LLM4Rec/model/fine-tune/Qwen2.5-0.5B/cosine/1epoch/checkpoint-9211")
    model.to('cuda')
    
    weights = torch.load(f"/mnt/data/LLM4Rec/model/fine-tune/Qwen2.5-0.5B/checkpoint-114514/pytorch_model.bin")
    fixed_weights = {k.replace("module.", ""): v for k, v in weights.items()}
    missing_keys, unexpected_keys = model.load_state_dict(fixed_weights, strict=False)

    item_index_ids, item_embs = generate_item_embs(
        df=item_df,
        model=model,
        tokenizer=tokenizer,
        device='cuda',
        batch_size=128,
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
        batch_size=32,
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

    # print("\n===== 召回结果分析 =====")
    for sample in sampled_recalls:
        user_id = sample["user_id"]
        print(f"\n用户 {user_id} 的召回结果：")
        
        # 获取召回商品的详细信息
        recalled_ids = sample["recalled_items"]
        recalled_items = item_df[item_df['item_id'].astype(str).isin(recalled_ids)]
        print("召回商品类别分布：")
        print(recalled_items['category'].value_counts())
        
        # 获取真实交互商品信息
        gt_ids = sample["ground_truth"]
        gt_ids = list(str(gt_ids))

        gt_items = item_df[item_df['item_id'].astype(str).isin(gt_ids)]
        print("\n真实交互商品类别：")
        print(gt_items['category'].unique())
    
    print(f"Hit Rate @200: {hit_rate:.4f}")

if __name__ == "__main__":
    main()
