import pandas as pd
import numpy as np
import os
import sys
from math import radians, cos, sin, asin, sqrt

# --- 1. 路径与环境设置 ---
BASE_DIR = "/workspace/Spatioal-temporal"
sys.path.append(os.path.join(BASE_DIR, "MyModel"))

# 注意这里：导入整个模块对象，而不是直接导入变量
import GSCAT.model_final as model_final 

input_cleaned_path = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_nyc_cleaned.csv')
output_dir = os.path.join(BASE_DIR, 'MyModel/dataset_process/mined_results/nyc')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def run_mining():
    print(f"读取数据源: {input_cleaned_path}")
    if not os.path.exists(input_cleaned_path):
        print("错误：找不到输入文件！")
        return

    # A. 运行点一的预处理函数
    # 这会填充 model_final 模块内部的全局变量 venue_map 和 category_map
    print("正在同步点一模型的 ID 映射表...")
    model_final.load_and_preprocess_data(input_cleaned_path)
    
    # 关键修改：通过模块对象获取最新的映射表
    v_map = model_final.venue_map
    c_map = model_final.category_map
    
    valid_poi_ids = set(v_map.keys())
    print(f"模型支持的 POI 数量: {len(valid_poi_ids)}")

    # B. 读取原始数据
    df = pd.read_csv(input_cleaned_path)
    
    # 强制将原始数据的 geo_id 转为字符串，因为 v_map 的 key 通常是字符串
    df['geo_id'] = df['geo_id'].astype(str)
    
    # 核心过滤：只保留模型里存在的 POI
    df = df[df['geo_id'].isin(valid_poi_ids)].copy()
    print(f"对齐后的有效签到数据行数: {len(df)}")

    if len(df) == 0:
        print("错误：对齐后数据为空！请检查原始数据 geo_id 的类型是否匹配。")
        return

    # 核心转换：将字符串 ID 转换为模型内部索引
    df['model_idx'] = df['geo_id'].map(v_map)
    
    df['time'] = pd.to_datetime(df['time'])
    
    # --- 2. 挖掘：POI停留时长 ---
    print("任务 1/4: 停留时长估计...")
    df = df.sort_values(['user_id', 'time'])
    df['time_diff'] = df.groupby('user_id')['time'].diff().dt.total_seconds() / 60
    valid_stays = df[(df['time_diff'] >= 15) & (df['time_diff'] <= 240)].copy()
    stay_time_by_cat = valid_stays.groupby('venue_category_name')['time_diff'].median().reset_index()
    stay_time_by_cat.columns = ['venue_category_name', 'avg_stay_minutes']
    
    # --- 3. 挖掘：软时间窗 ---
    print("任务 2/4: 软时间窗识别...")
    poi_time_windows = df.groupby('model_idx')['hour'].agg([
        lambda x: np.percentile(x, 5),
        lambda x: np.percentile(x, 95)
    ]).reset_index()
    poi_time_windows.columns = ['geo_id', 'open_hour', 'close_hour']
    
    # --- 4. 挖掘：时空热度 ---
    print("任务 3/4: 时空热度计算...")
    poi_heat = df.groupby(['model_idx', 'hour']).size().reset_index(name='raw_heat')
    poi_heat.columns = ['geo_id', 'hour', 'raw_heat']
    poi_heat['norm_heat'] = poi_heat.groupby('geo_id')['raw_heat'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-5)
    )
    
    # --- 5. 整合 POI 元数据 ---
    print("任务 4/4: 整合元数据...")
    poi_meta = df[['model_idx', 'venue_category_id', 'venue_category_name', 'longitude', 'latitude']].drop_duplicates('model_idx')
    poi_meta.columns = ['geo_id', 'venue_category_id', 'venue_category_name', 'longitude', 'latitude']
    
    poi_meta = poi_meta.merge(stay_time_by_cat, on='venue_category_name', how='left')
    global_median = stay_time_by_cat['avg_stay_minutes'].median() if not stay_time_by_cat.empty else 60
    poi_meta['avg_stay_minutes'] = poi_meta['avg_stay_minutes'].fillna(global_median)
    
    # --- 6. 保存结果 ---
    poi_meta.to_csv(os.path.join(output_dir, 'poi_meta_processed.csv'), index=False)
    poi_time_windows.to_csv(os.path.join(output_dir, 'poi_time_windows.csv'), index=False)
    poi_heat.to_csv(os.path.join(output_dir, 'poi_hourly_heat.csv'), index=False)
    
    print(f"\n挖掘任务全部完成！")
    print(f"对齐后的 POI 数量: {len(poi_meta)}")
    print(f"最大 POI 索引: {poi_meta['geo_id'].max()}")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    run_mining()