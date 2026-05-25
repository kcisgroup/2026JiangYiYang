import pandas as pd
import os

BASE_DIR = os.getcwd()

# --- 1. 参数设置 ---
file_path = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_tky_processed.csv')
cleaned_file_path = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_tky_cleaned.csv')
min_venue_checkins = 10  # 场馆最少签到次数阈值
min_user_checkins = 5    # 用户最少签到次数阈值

# --- 2. 加载数据 ---
print(f"正在从 '{file_path}' 加载数据...")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'。请确保文件路径正确，并且文件和脚本在同一目录下，或者提供完整路径。")
    exit()

print("\n--- 原始数据统计 ---")
print(f"总签到记录数: {len(df)}")
# 注意：场馆的唯一标识符是 geo_id，而不是 venue_category_id
print(f"独立用户数: {df['user_id'].nunique()}")
print(f"独立场馆数 (geo_id): {df['geo_id'].nunique()}")


# --- 3. 迭代过滤数据 ---
print("\n--- 开始迭代过滤数据 ---")
# 循环直到数据集的大小不再变化
while True:
    # 记录过滤前的数据行数
    rows_before_filter = len(df)
    
    # 步骤 A: 删除签到次数少于 min_venue_checkins 的场馆记录
    # 1. 计算每个场馆 (geo_id) 的签到次数
    venue_counts = df['geo_id'].value_counts()
    # 2. 找出签到次数 >= 阈值的场馆
    venues_to_keep = venue_counts[venue_counts >= min_venue_checkins].index
    # 3. 在 DataFrame 中只保留这些场馆的记录
    df = df[df['geo_id'].isin(venues_to_keep)]
    
    # 步骤 B: 删除签到次数少于 min_user_checkins 的用户记录
    # 1. 计算每个用户 (user_id) 的签到次数
    user_counts = df['user_id'].value_counts()
    # 2. 找出签到次数 >= 阈值的用户
    users_to_keep = user_counts[user_counts >= min_user_checkins].index
    # 3. 在 DataFrame 中只保留这些用户的记录
    df = df[df['user_id'].isin(users_to_keep)]
    
    # 记录过滤后的数据行数
    rows_after_filter = len(df)
    
    print(f"本轮过滤后剩余记录数: {rows_after_filter}")
    
    # 如果数据行数没有变化，说明已达到稳定状态，可以跳出循环
    if rows_before_filter == rows_after_filter:
        print("数据已稳定，过滤完成。")
        break

# --- 4. 显示最终结果并保存 ---
print("\n--- 清理后数据统计 ---")
print(f"剩余总签到记录数: {len(df)}")
print(f"剩余独立用户数: {df['user_id'].nunique()}")
print(f"剩余独立场馆数 (geo_id): {df['geo_id'].nunique()}")

# 将清理后的数据保存到新的 CSV 文件
df.to_csv(cleaned_file_path, index=False)
print(f"\n清理后的数据已保存到: '{cleaned_file_path}'")