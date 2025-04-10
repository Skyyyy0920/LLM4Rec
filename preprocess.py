import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta


headers = pd.read_csv('/mnt/nas/tianqiong/item_header.csv', header=None)[0].tolist()[1:-1]
df = pd.read_csv('/mnt/nas/tianqiong/tdm_item_filter_more_col_dot.csv', names=headers, header=None, on_bad_lines='skip', sep=',')
df_result = df[df['item_id'].str.contains(r'[\u4e00-\u9fa5]', regex=True, na=False)]

# 打印结果
print("包含汉字的 item_id 行：")
print(df_result)
print(len(df))
# 统计 item_id 的缺失值
missing_count = df['item_id'].isnull().sum()
print(f"item_id 列的缺失值数量：{missing_count}")

# 输出缺失值所在的行
print("\n包含 item_id 缺失值的行：")
print(df[df['title'].isnull()])
missing_counts = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

# 输出结果
print("列名及缺失值统计：")
print(df.columns.tolist())
print("\n缺失值数量：")
print(missing_counts)
print("\n缺失值比例（%）：")
print(missing_percent)
df['item_id'] = df['item_id'].astype('int64')
df.to_csv('/home/tianqiong/LLM4Rec/data/item_more_col.csv', index=False)
exit()


start_date = "20250117"
# end_date = "20250301"
end_date = "20250127"

start = datetime.strptime(start_date, "%Y%m%d")
end = datetime.strptime(end_date, "%Y%m%d")

dates = []
current = start
while current <= end:
    dates.append(current.strftime("%Y%m%d"))
    current += timedelta(days=1)

headers = pd.read_csv('/mnt/nas/tianqiong/sample_header.csv', header=None)[0].tolist()[1:-1]  # 去掉第一个和最后一个不是字段名的

# 定义需要处理的字段和分隔符
split_fields = {
    'camp_id_list': '\x1d',
    'category_id_list': '\x1d',
    'category_name_list': '\x1d',
    'follow_user_id_list': '\x1d',
    'comment_author_list': '\x1d',
    'click_50_seq__item_id': ';',
    'click_50_seq__category': ';',
    'click_50_seq__author': ';',
    'click_50_seq__theme_id': ';'
}

# 读取所有文件并合并
df_list = []
for date in tqdm(dates):
    file_name = f"/mnt/nas/tianqiong/tdm_{date}.csv"
    try:
        df = pd.read_csv(file_name, names=headers, header=None, on_bad_lines='skip')  # 读取文件，跳过列数不匹配的行
        df['date'] = date  # 添加日期列
        df_list.append(df)
    except Exception as e:
        print(f"Error reading {file_name}: {e}")

big_df = pd.concat(df_list, ignore_index=True)

# 处理分隔符字段
for col, sep in split_fields.items():
    big_df[col] = big_df[col].fillna('')
    big_df[col] = big_df[col].str.split(sep)

# # 6. 处理缺失值（示例方法）
# for col in big_df.columns:
#     if col in split_fields:  # 已处理的列表字段跳过
#         continue
#     if big_df[col].dtype == 'object':
#         # 分类字段填充众数（默认第一个）
#         mode_val = big_df[col].mode()[0] if len(big_df[col].mode()) > 0 else ''
#         big_df[col].fillna(mode_val, inplace=True)
#     else:
#         # 数值字段填充均值
#         mean_val = big_df[col].mean()
#         big_df[col].fillna(mean_val, inplace=True)

# 划分训练集和测试集
dates_sorted = sorted(dates)
test_dates = dates_sorted[-3:]  # 最后三天
train_dates = dates_sorted[:-3]

train_df = big_df[big_df['date'].isin(train_dates)].drop(columns=['date'])
test_df = big_df[big_df['date'].isin(test_dates)].drop(columns=['date'])

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
