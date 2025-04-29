import os
import torch
import pickle
import contextlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from utils import CATEGORY_MAPPING


def build_user_prompt(row):
    base_template = """用户是来自{city}的{gender}性，账户等级{growth_level}{new_user_tag}。
最近活跃于周{week_day}{day_h}时，历史点击{item_cnt}次内容。
用户关注{follow_cnt}人，拥有{follower_cnt}粉丝，收藏{favorite_cnt}内容。
已购买{buy_camp_cnt}门课程，{category_name_list}。
用户近期点击内容类别：{formatted_category_seq}
近期点击内容标题：{formatted_item_title_seq}"""

    def format_seq(seq):
        seq = [f"'{item}'" for item in seq if item.strip() != '']
        return ", ".join([f"{item}" for item in seq])
    
    def analyze_purchases(cat_list):
        counter = Counter(cat_list)
        total = len(cat_list)
        
        frequent = [f"{k}{v}门" for k,v in counter.items()]
        return "其中" + "，".join(frequent) if total > 0 else "暂无显著消费倾向"
    
    def format_category_seq(seq):
        counter = Counter(seq)
        total = len(seq)
        
        main_cats = {k:v for k,v in counter.items()}
        sorted_cats = sorted(main_cats.items(), key=lambda x: -x[1])
        
        parts = []
        for cat, cnt in sorted_cats:
            translated = CATEGORY_MAPPING.get(cat, cat)
            parts.append(f"{translated}{cnt}次")
            
        return "、".join(parts)

    week_day_cn = "一二三四五六七"
    week_day = week_day_cn[row["week_day"]-1]
    new_user = "是新用户，" if row['is_new_user'] == 1 else ""
    gender_map = ['男', '女', '未知']
    if row['gender'] == 0:
        gender = '男'
    elif row['gender'] == 1:
        gender = '女'
    else:
        gender = '未知'

    prompt = base_template.format(
        city=row["city"],
        gender=gender,
        day_h=row["day_h"],
        week_day=week_day,
        is_new_user=new_user,
        growth_level=f"{row["growth_level"]:.0f}",
        new_user_tag="(新用户)" if row['is_new_user'] == 1 else "",
        buy_camp_cnt=f"{row['buy_camp_cnt']:.0f}",
        item_cnt=f"{row['item_cnt']:.0f}",
        follow_cnt=f"{row['follow_cnt']:.0f}",
        follower_cnt=f"{row['follower_cnt']:.0f}",
        favorite_cnt=f"{row['favorite_cnt']:.0f}",
        category_name_list=analyze_purchases(row["category_name_list"]),
        formatted_category_seq=format_category_seq(row["click_50_seq__category"]),
        formatted_item_title_seq=format_seq(row["click_50_seq__item_title"]),
    )

    return prompt
    
# def build_user_prompt(row):
#     base_template = """用户是{city}市的{gender}性，{is_new_user}账户等级为{growth_level}。当前是周{week_day}的{day_h}时。
#     用户已点击内容数量为{item_cnt}，关注用户数量为{follow_cnt}，粉丝数量为{follower_cnt}，收藏内容数量为{favorite_cnt}，购买过的课程数量为{buy_camp_cnt}。
#     用户曾经购买过以下课程：{category_name_list}，用户的历史点击行为从远到近显示其对如下内容种类感兴趣: {category_seq}。"""

#     week_day_cn = "一二三四五六七"
#     week_day = week_day_cn[row["week_day"]-1]
#     category_seq = str(row["click_50_seq__category"])
#     new_user = "是新用户，" if row['is_new_user'] == 1 else ""
#     gender_map = ['男', '女', '未知']
#     if row['gender'] == 0:
#         gender = '男'
#     elif row['gender'] == 1:
#         gender = '女'
#     else:
#         gender = '未知'
    
#     return base_template.format(
#         # user_id=row["user_id"],
#         city=row["city"],
#         gender=gender,
#         day_h=row["day_h"],
#         week_day=week_day,
#         is_new_user=new_user,
#         growth_level=f"{row["growth_level"]:.0f}",
#         category_seq=category_seq,
#         buy_camp_cnt=f"{row['buy_camp_cnt']:.0f}",
#         item_cnt=f"{row['item_cnt']:.0f}",
#         follow_cnt=f"{row['follow_cnt']:.0f}",
#         follower_cnt=f"{row['follower_cnt']:.0f}",
#         favorite_cnt=f"{row['favorite_cnt']:.0f}",
#         category_name_list=row['category_name_list']
#     )


def build_item_prompt(row):
#     base_template = """内容类型为{type}，发布于{pub_time}，{status}被推荐。标题为：{title}，一级标签为{category}。
# 作者身份为{author_status}，所属话题为{theme_id}，活动id为{activity}，发布源为{publish}。
# 内容的点赞数为{praise}，评论数为{comment}，收藏数为{collect}，分享数为{share}。
# 内容{home_mark}首页精选，{club_mark}广场精选。"""
    base_template = """内容类型为{type}，发布于{pub_time}，{status}被推荐。标题为：{title}，一级标签为{category}。
作者身份为{author_status}，发布源为{publish}。
内容的点赞数为{praise}，评论数为{comment}，收藏数为{collect}，分享数为{share}。"""

    pub_time = datetime.fromtimestamp(row['pub_time']).strftime("%Y-%m-%d")
    status = "已" if row['status'] else "未"
    home_mark = "属于" if row['home_mark'] == 'Y' else "不属于"
    club_mark = "属于" if row['club_mark'] == 'Y' else "不属于"

    prompt = base_template.format(
        type=row['item_type'],
        pub_time=pub_time,
        status=status,
        title=row['title'],
        category=row['category'],
        author_status=row['author_status'],
        theme_id=f"{row['theme_id']:.0f}",
        praise=f"{row['praise_count']:.0f}",
        comment=f"{row['comment_count']:.0f}",
        collect=f"{row['collect_count']:.0f}",
        share=f"{row['share_count']:.0f}",
        publish=row['publish_source'],
        activity=f"{row['activity_id']:.0f}",
        home_mark=home_mark,
        club_mark=club_mark,
    )
    
    return prompt


class ItemDataset(Dataset):
    def __init__(self, df, id_field):
        self.df = df
        self.id_field = id_field
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return {
            "item_id": str(item[self.id_field]),
            "text": build_item_prompt(item)
        }
    

class UserDataset(Dataset):
    def __init__(self, df, id_field):
        self.df = df
        self.id_field = id_field
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        return {
            "user_id": str(sample[self.id_field]),
            "text": build_user_prompt(sample)
        }


@torch.inference_mode()
def generate_user_embs(
    df: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    batch_size: int = 256,
    use_amp: bool = True
):
    """
    加速优化的embedding生成函数
    
    Args:
        df: 输入的DataFrame
        model: 预加载的模型
        tokenizer: 预加载的分词器
        device: 计算设备
        batch_size: 批处理大小
        use_amp: 是否使用自动混合精度
        
    Returns:
        index_ids: 商品ID列表
        embeddings: 生成的embedding矩阵
    """
    dataset = UserDataset(df, id_field="user_id")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False
    )

    torch.backends.cudnn.benchmark = True
    
    index_ids = []
    embeddings = []
    
    # amp_ctx = torch.amp.autocast('cuda') if use_amp and device.type == "cuda" else contextlib.nullcontext()
    amp_ctx = torch.amp.autocast('cuda') if use_amp else contextlib.nullcontext()
    with amp_ctx:
        for batch in tqdm(dataloader, desc="Generating User Embeddings"):
            prompts = batch["text"]
            batch_ids = batch["user_id"]

            messages_list = [
                [
                    {"role": "system", "content": "在个性化内容推荐场景中，根据用户画像，生成个性化内容推荐。"},
                    {"role": "user", "content": prompt}
                ] for prompt in prompts
            ]
            texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]

            model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)

            # generated_ids = model.generate(**model_inputs, max_new_tokens=2048)
            # generated_ids = [
            #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            # ]
            # responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # for res in responses:
            #     print(res)

            outputs = model(**model_inputs, output_hidden_states=True)
            
            # batch_emb = outputs.hidden_states[-1][:, -1, :]
            batch_emb = outputs.hidden_states[-1].mean(dim=1)
            batch_emb = batch_emb.float().cpu().numpy()  # [batch_size, hidden_dim]
            
            index_ids.extend(batch_ids)
            embeddings.append(batch_emb)
    
    return index_ids, np.concatenate(embeddings, axis=0)


@torch.inference_mode()
def generate_item_embs(
    df: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    batch_size: int = 256,
    use_amp: bool = True
):
    dataset = ItemDataset(df, id_field="item_id")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=False
    )

    torch.backends.cudnn.benchmark = True
    
    index_ids = []
    embeddings = []
    
    # amp_ctx = torch.amp.autocast('cuda') if use_amp and device.type == "cuda" else contextlib.nullcontext()
    amp_ctx = torch.amp.autocast('cuda') if use_amp else contextlib.nullcontext()
    with amp_ctx:
        for batch in tqdm(dataloader, desc="Generating Item Embeddings"):
            prompts = batch["text"]
            batch_ids = batch["item_id"]

            messages_list = [
                [
                    {"role": "system", "content": "在个性化内容推荐场景中，根据内容信息推断潜在客户信息（如性别、内容偏好等），并向用户推荐。"},
                    {"role": "user", "content": prompt}
                ] for prompt in prompts
            ]
            texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]

            model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)

            # generated_ids = model.generate(**model_inputs, max_new_tokens=2048)
            # generated_ids = [
            #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            # ]
            # responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # for res in responses:
            #     print(res)

            outputs = model(**model_inputs, output_hidden_states=True)
            
            # batch_emb = outputs.hidden_states[-1][:, -1, :]
            batch_emb = outputs.hidden_states[-1].mean(dim=1)
            batch_emb = batch_emb.float().cpu().numpy()  # [batch_size, hidden_dim]
            
            index_ids.extend(batch_ids)
            embeddings.append(batch_emb)
    
    return index_ids, np.concatenate(embeddings, axis=0)


def save_embeddings(index_ids, embeddings, save_dir, prefix="item"):
    """
    保存生成的embedding和对应的ID
    
    Args:
        index_ids: 物品ID列表
        embeddings: embedding矩阵
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(f"{save_dir}/{prefix}_index_ids.pkl", "wb") as f:
        pickle.dump(index_ids, f)
    
    np.save(f"{save_dir}/{prefix}_embs.npy", embeddings)


def load_embeddings(save_dir, prefix="item"):
    """
    加载保存的embedding和ID
    
    Args:
        save_dir: 保存目录
        prefix: 文件名前缀
    
    Returns:
        list: 物品ID列表
        np.ndarray: embedding矩阵
    """
    with open(f"{save_dir}/{prefix}_index_ids.pkl", "rb") as f:
        index_ids = pickle.load(f)
    
    embeddings = np.load(f"{save_dir}/{prefix}_embs.npy")
    
    return index_ids, embeddings
