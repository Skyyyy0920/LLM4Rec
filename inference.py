import os
import torch
import faiss
import logging
# import deepspeed
import numpy as np
import pandas as pd
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer
from embedding import generate_item_embs, generate_user_embs, save_embeddings, load_embeddings


def build_faiss_index_from_embeddings(
    index_id_map: list,
    embeddings: np.ndarray,
    index_type: str = 'IVFFlatL2',
    ivf_nlist: int = 1000,
    hnsw_M: int = 32,
    hnsw_efConstruction: int = 200
):
    """
    从内存中的embedding直接构建FAISS索引
    
    Args:
        index_id_map: item ID列表
        embeddings: 已生成的embedding矩阵
        index_type: 索引类型
        ivf_nlist: IVF索引参数
        hnsw_M: HNSW索引参数
        hnsw_efConstruction: HNSW索引参数
    
    Returns:
        faiss.Index: 构建好的索引
        np.ndarray: 索引ID映射数组
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings should be 2D array, got {embeddings.ndim}D")
    
    embedding_dim = embeddings.shape[1]

    res = faiss.StandardGpuResources()
    
    if index_type.endswith('IP'):
        quantizer = faiss.IndexFlatIP(embedding_dim)
        metric_type = faiss.METRIC_INNER_PRODUCT
    elif index_type.endswith('L2'):
        quantizer = faiss.IndexFlatL2(embedding_dim)
        metric_type = faiss.METRIC_L2
    else:
        raise ValueError(f"Unsupported metric type in index: {index_type}")
    
    if index_type.startswith("IVFFlat"):
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, ivf_nlist, metric_type)
        index = faiss.index_cpu_to_gpu(res, 0, index)  # 转到GPU上，重要！
        if not index.is_trained:
            index.train(embeddings)
    elif index_type.startswith("HNSWFlat"):
        index = faiss.IndexHNSWFlat(embedding_dim, hnsw_M, metric_type)
        index = faiss.index_cpu_to_gpu(res, 0, index)
        index.hnsw.efConstruction = hnsw_efConstruction
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    index.add(embeddings)

    return index, np.array(index_id_map, dtype=str)


def save_faiss_index(index, index_id_map, save_dir, index_name):
    """
    保存FAISS索引和ID映射
    
    Args:
        index: FAISS索引
        index_id_map: ID映射数组
        save_dir: 保存目录
        index_name: 索引名称(不带后缀)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if faiss.get_num_gpus() > 0:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, f"{save_dir}/{index_name}.index")
    
    np.save(f"{save_dir}/{index_name}_ids.npy", index_id_map)


def load_faiss_index(save_dir, index_name, gpu_id=0):
    """
    加载FAISS索引和ID映射
    
    Args:
        save_dir: 保存目录
        index_name: 索引名称(不带后缀)
        gpu_id: 要使用的GPU ID
    
    Returns:
        faiss.Index: 加载的索引
        np.ndarray: ID映射数组
    """
    index = faiss.read_index(f"{save_dir}/{index_name}.index")
    
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)
    
    index_id_map = np.load(f"{save_dir}/{index_name}_ids.npy")
    
    return index, index_id_map


# def write_faiss_index(
#     index: faiss.Index, index_id_map: npt.NDArray, output_dir: str
# ) -> None:
#     """Write faiss index.

#     Args:
#         index (faiss.Index): faiss index.
#         index_id_map (NDArray): a list of embedding ids
#             for mapping continuous ids to origin id.
#         output_dir (str): index output dir.
#     """
#     # pyre-ignore [16]
#     faiss.write_index(index, os.path.join(output_dir, "faiss_index"))
#     with open(os.path.join(output_dir, "id_mapping"), "w") as f:
#         for eid in index_id_map:
#             f.write(f"{eid}\n")


def compute_hit_rate(
    user_embs: np.ndarray,          # 用户embedding矩阵 [num_users, emb_dim]
    user_index_ids: np.ndarray,     # 用户ID列表 [num_users]
    index: faiss.Index,             # 构建好的FAISS索引
    index_id_map: np.ndarray,       # 物品ID映射表 [num_items]
    gt_mapping: np.ndarray,        # 用户真实物品映射 {user_id: [item_id1, item_id2...]}
    top_k: int = 10,                # 召回数量
    batch_size: int = 1024          # 批处理大小
):
    """
    计算基于FAISS检索的命中率
    
    返回:
        hit_rate: 命中用户数 / 总用户数
        recall_rate: 命中物品数 / 总物品数
    """
    total_hits = 0
    total_gt_items = 0

    sampled_recalls = []  # 存储采样结果
    
    user_embs = np.ascontiguousarray(user_embs.astype('float32'))
    
    for i in tqdm(range(0, len(user_embs), batch_size)):
        batch_embs = user_embs[i:i+batch_size]
        batch_ids = user_index_ids[i:i+batch_size]
        gt_items = gt_mapping[i:i+batch_size]
        
        _, recall_indices = index.search(batch_embs, top_k)  # [batch_size, top_k]
        
        recalled_items = index_id_map[recall_indices]  # [batch_size, top_k]
        
        for user_id, recalls, gt_item in zip(batch_ids, recalled_items, gt_items):
            if len(sampled_recalls) < 3:
                sampled_recalls.append({
                    "user_id": user_id,
                    "recalled_items": recalls,
                    "ground_truth": gt_item
                })
            
            if not gt_item:
                continue
                
            hit_flags = np.isin(recalls, str(gt_item))
            
            total_gt_items += 1
            total_hits += np.sum(hit_flags)
                
    hit_rate = total_hits / total_gt_items if total_gt_items > 0 else 0
    
    return hit_rate, sampled_recalls


def init_model(size):
    model_name = f"Qwen/Qwen2.5-{size}B-instruct"
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model_name = "alibaba-pai/DistilQwen2.5-R1-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    ds_config = {
        "tensor_parallel": {"tp_size": 1},
        "dtype": "fp16",
        "replace_with_kernel_inject": True
    }
    
    ds_model = deepspeed.init_inference(
        model=model,
        config=ds_config,
    )
    device = next(model.parameters()).device
    
    return ds_model, tokenizer, device


def setup_logger(log_dir="/mnt/data/LLM4Rec/logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"run.log"
    
    logger = logging.getLogger("LLM4RecEvaluation")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


if __name__ == "__main__":
    data_dir = "/home/tianqiong/LLM4Rec/data"
    item_df = pd.read_csv(os.path.join(data_dir, 'item_test.csv'))
    item_df = item_df.dropna(subset=['publish_source'])
    item_df['item_id'] = item_df['item_id'].astype('int64')
    print(f"Successfully load item data.",len(item_df))
    # item_df = item_df[:1024]
    # train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    # print(f"Successfully load train data and test data.",len(train_df), len(test_df))
    # test_df = test_df[:1024]

    test_df['exists_in_item'] = test_df['item_id'].isin(item_df['item_id'])
    exists_count = test_df['exists_in_item'].sum()
    not_exists_count = len(test_df) - exists_count
    print(f"Total items in test_df: {len(test_df)}")
    print(f"Items existing in item_df: {exists_count}")
    print(f"Items NOT existing in item_df: {not_exists_count}")
    
    size="7"
    model, tokenizer, device = init_model(size)

    # gt_mapping = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    gt_mapping = test_df['item_id'].to_numpy()
    
    if_load = True
    if if_load:
        item_index_ids, item_embs = load_embeddings("/home/tianqiong/LLM4Rec/emb", f"item_{size}B")
    else:
        item_index_ids, item_embs = generate_item_embs(
            df=item_df,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=32,
            use_amp=True
        )
    if not if_load:
        save_embeddings(item_index_ids, item_embs, "/home/tianqiong/LLM4Rec/emb", f"item_{size}B")

    if if_load:
        index, index_id_map = load_faiss_index("/home/tianqiong/LLM4Rec/emb", f"item_index_{size}B")
    else:
        index, index_id_map = build_faiss_index_from_embeddings(
            index_id_map=item_index_ids,
            embeddings=item_embs,
        )
    index.nprobe = 800

    if not if_load:
        save_faiss_index(index, index_id_map, "/home/tianqiong/LLM4Rec/emb", f"item_index_{size}B")
    if_load = False
    if if_load:
        user_index_ids, user_embs = load_embeddings("/home/tianqiong/LLM4Rec/emb", f"user_{size}B")
    else:
        user_index_ids, user_embs = generate_user_embs(
            df=test_df,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=32,
            use_amp=True
        )
    if not if_load:
        save_embeddings(user_index_ids, user_embs, "/home/tianqiong/LLM4Rec/emb", f"user_{size}B")

    hit_rate, sampled_recalls = compute_hit_rate(
        user_embs=user_embs,
        user_index_ids=user_index_ids,
        index=index,
        index_id_map=index_id_map,
        gt_mapping=gt_mapping,
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
