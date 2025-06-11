import os
import re
import argparse
import multiprocessing
import numpy as np
from odps import ODPS
from odps.accounts import AliyunAccount
from modelscope import AutoModelForCausalLM, AutoTokenizer
from embedding import generate_item_embs, generate_user_embs
from utils import compute_hit_rate, build_faiss_index_from_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint编号，例如: 9208")
    parser.add_argument("--model_dir", type=str, required=True, help="模型所在目录，例如: /mnt/data/LLM_emb_recall/ckpt/20250516")
    parser.add_argument("--data_date", type=str, required=True, help="测试数据日期，格式YYYYMMDD，例如: 20250517")
    parser.add_argument("--top_k", type=int, default=200, help="召回Top-K个item")
    args = parser.parse_args()
    print(args)

    access_id = ''
    access_key = ''
    project_name = ''
    endpoint = 'http://service.odps.aliyun.com/api'

    account = AliyunAccount(access_id, access_key)
    o = ODPS(account=account, project=project_name, endpoint=endpoint)
    n_process = multiprocessing.cpu_count()
    
    t_item = o.get_table('rec_sln_demo_item_table_v1_filter')
    with t_item.open_reader(partition='ds='+args.data_date, arrow=True) as reader:
        item_df = reader.to_pandas(n_process=n_process)
    print(f"load item length: {len(item_df)}")
    
    t_sample = o.get_table('rec_sln_demo_sample_table_v1_filter')
    with t_sample.open_reader(partition='ds='+args.data_date, arrow=True) as reader:
        sample_df = reader.to_pandas(n_process=n_process)
    print(f"load sample length: {len(sample_df)}")
    
    count = sample_df[~sample_df['item_id'].isin(item_df['item_id'])].shape[0]
    print(f"sample_df 中有 {count} 行的 item_id 不在 item_df 中")
    sample_df = sample_df[sample_df['item_id'].isin(item_df['item_id'])]  # 去掉不在item_df中的sample数据
    
    sample_df["city"] = sample_df["city"].fillna("未知城市")
    sample_df['click_50_seq__item_id'] = sample_df['click_50_seq__item_id'].fillna('').apply(lambda x: [item for item in x.split(';')] if x else [])
    sample_df['click_50_seq__category'] = sample_df['click_50_seq__category'].fillna('').apply(lambda x: [item for item in x.split(';')] if x else [])
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

    ckpt_path = os.path.join(args.model_dir, f"checkpoint-{args.ckpt}")
    print(f"Load model: {ckpt_path}")
    model = AutoModelForCausalLM.from_pretrained(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model.to('cuda')

    print("Generate item embeddings...")
    item_index_ids, item_embs = generate_item_embs(
        df=item_df,
        model=model,
        tokenizer=tokenizer,
        device='cuda',
        batch_size=32,
        use_amp=True
    )

    print("Construct Faiss index...")
    index, index_id_map = build_faiss_index_from_embeddings(
        index_id_map=item_index_ids,
        embeddings=item_embs,
    )
    index.nprobe = 800

    print("Generate user embeddings...")
    user_index_ids, user_embs = generate_user_embs(
        df=sample_df,
        model=model,
        tokenizer=tokenizer,
        device='cuda',
        batch_size=16,
        use_amp=True
    )

    print(f"Calculate Hit Rate@{args.top_k}...")
    hit_rate, sampled_recalls = compute_hit_rate(
        user_embs=user_embs,
        user_index_ids=user_index_ids,
        index=index,
        index_id_map=index_id_map,
        gt_mapping=sample_df['item_id'].to_numpy(),
        top_k=args.top_k,
        batch_size=512
    )
    print("**************************************")
    print(f"Hit Rate@{args.top_k}: {hit_rate:.4f}")
    print("**************************************")


if __name__ == "__main__":
    main()