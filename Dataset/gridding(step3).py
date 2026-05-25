import pandas as pd
import os
import numpy as np

# ==============================================================================
# 0. 配置项
# ==============================================================================
BASE_DIR = os.getcwd()

input_cleaned_path = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_nyc_cleaned.csv')

output_final_path = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_nyc_final.csv')

# --- 空间离散化 (Gridding) 参数 ---
GRID_SIZE_LAT = 0.005  # 纬度方向的网格大小
GRID_SIZE_LON = 0.005  # 经度方向的网格大小

# ==============================================================================
# 1. 加载已清理的数据
# ==============================================================================
print(f"正在从 '{input_cleaned_path}' 加载已清理的数据...")
try:
    df = pd.read_csv(input_cleaned_path)
except FileNotFoundError:
    print(f"错误：找不到文件 '{input_cleaned_path}'。请确保文件已存在。")
    exit()

print("\n--- 已清理数据预览 ---")
print(f"总记录数: {len(df)}")
print(f"独立用户数: {df['user_id'].nunique()}")
print(f"独立场馆数 (geo_id): {df['geo_id'].nunique()}")

# 检查进行空间离散化所必需的列
if 'latitude' not in df.columns or 'longitude' not in df.columns:
    print("错误：数据中缺少 'latitude' 或 'longitude' 列，无法进行空间离散化。")
    exit()

# ==============================================================================
# 2. 执行空间离散化 (Gridding)
# ==============================================================================
print("\n--- 开始空间离散化 (Gridding) ---")

# 1. 找到所有签到点的经纬度范围，以确定整个地图的边界
lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

print(f"地图边界: 纬度 [{lat_min:.4f}, {lat_max:.4f}], 经度 [{lon_min:.4f}, {lon_max:.4f}]")

# 2. 计算在经纬度方向上各有多少个网格
# 使用 np.ceil 确保整个区域都被覆盖
num_grids_y = int(np.ceil((lat_max - lat_min) / GRID_SIZE_LAT))
num_grids_x = int(np.ceil((lon_max - lon_min) / GRID_SIZE_LON))

# 避免 num_grids_x 为0的情况，如果经度范围很小
if num_grids_x == 0:
    num_grids_x = 1

print(f"网格尺寸: {num_grids_y} (纬度方向) x {num_grids_x} (经度方向) = {num_grids_y * num_grids_x} 个总网格")

# 3. 为每一条签到记录计算其所在的网格坐标 (grid_x, grid_y)
# (lat - lat_min) / GRID_SIZE_LAT 计算出该点在纬度方向上是第几个格子
df['grid_y'] = ((df['latitude'] - lat_min) / GRID_SIZE_LAT).astype(int)
df['grid_x'] = ((df['longitude'] - lon_min) / GRID_SIZE_LON).astype(int)

# 确保坐标不会超出边界
df['grid_y'] = df['grid_y'].clip(0, num_grids_y - 1)
df['grid_x'] = df['grid_x'].clip(0, num_grids_x - 1)

# 4. 创建一个唯一的 region_id
# 这是一个常用的方法，将二维的网格坐标 (x, y) 映射到一个一维的整数ID
# 类似于数组中元素的索引计算：index = row * num_columns + column
df['region_id'] = df['grid_y'] * num_grids_x + df['grid_x']

print("空间离散化完成，已添加 'region_id' 列。")


# ==============================================================================
# 3. 显示最终结果并保存
# ==============================================================================
print("\n--- 最终处理后数据统计 ---")
print(f"总记录数: {len(df)}")
print(f"独立用户数: {df['user_id'].nunique()}")
print(f"独立场馆数 (geo_id): {df['geo_id'].nunique()}")
print(f"生成的独立 Region ID 数量: {df['region_id'].nunique()}")

# 将最终的数据保存到新的 CSV 文件
df_final = df # 直接使用完整的DataFrame

# 按照 user_id 和 time 排序
df_final = df_final.sort_values(by=['user_id', 'time']).reset_index(drop=True)

df_final.to_csv(output_final_path, index=False)

print(f"\n最终处理后的数据已保存到: '{output_final_path}'")