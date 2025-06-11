import os
import faiss
import numpy as np
from tqdm import tqdm


CATEGORY_MAPPING = {
    'OTHER': '其他',
    'CHINASTUDIES': '国学',
    'YOGA': '瑜伽',
    'ZHONGYI': '中医',
    'Pilates': '普拉提',
    'SHORT_VIDEO': '短视频',
    'PHONE_PHOTOGRAPHY': '手机摄影',
    'SING': '唱歌',
    'CHUANDA': '穿搭',
    'EAT_THIN': '吃瘦',
    'FIVEFOWLPLAYS': '五禽戏',
    'Cameraphotography': '相机摄影',
    'QIXUE_TIAOLI': '气血调理',
    'LIFE': '生活',
    'GUQIHUOXUE': '古琴活学',
    'DANCE': '舞蹈',
    'ASTROLOGY': '占星',
    'YIJING': '易经',
    'ZMGJ': '正面管教',
    'Videoclip': '视频剪辑',
    'TAROT': '塔罗',
    'shequceshileimu': '社区测试类目',
    np.nan: '未知类别',
    None: '未知类别',
    '': '未知类别'
}


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
