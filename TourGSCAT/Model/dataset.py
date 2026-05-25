import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from config import CONFIG
from utils import haversine

def load_and_extract_itineraries(data_path, mapping_path):
    """
    加载Stage 1的映射表，并将原始历史流水切分为一条条“一日游”的子行程（Itinerary）。
    """
    print(">>> 1. 加载 Stage 1 的 ID 映射与原始数据...")
    with open(mapping_path, 'rb') as f:
        maps = pickle.load(f)
    user_map, venue_map, cat_map = maps['user_map'], maps['venue_map'], maps['category_map']
    venue_pad_idx, cat_pad_idx = maps['venue_pad_idx'], maps['cat_pad_idx']
    
    num_users = len(user_map)
    num_venues_with_pad = venue_pad_idx + 1 
    
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    df['geo_id'] = df['geo_id'].astype(str)
    df['venue_category_id'] = df['venue_category_id'].fillna('UNK_CAT').astype(str)
    df = df.sort_values(by=['user_id', 'time'])
    
    # 提取全局 POI 信息
    poi_info = {}
    for _, row in df.drop_duplicates(subset=['geo_id']).iterrows():
        if row['geo_id'] in venue_map:
            poi_info[venue_map[row['geo_id']]] = {
                'lat': row['latitude'], 'lon': row['longitude'],
                'cat': cat_map.get(row['venue_category_id'], cat_pad_idx)
            }
            
    itineraries = []
    current_trip = []
    cat_dwell_times = defaultdict(list) # 记录每一类 POI 的平均逗留时间
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="切分一日游轨迹"):
        if row['user_id'] not in user_map or row['geo_id'] not in venue_map: continue
            
        u_idx, v_idx = user_map[row['user_id']], venue_map[row['geo_id']]
        c_idx = cat_map.get(row['venue_category_id'], cat_pad_idx)
        curr_time = row['time']
        
        # 用户发生切换，保存上一个有效Trip
        if not current_trip or current_trip[-1]['user'] != u_idx:
            if current_trip and CONFIG['min_trip_len'] <= len(current_trip) <= CONFIG['max_trip_len']:
                itineraries.append({'user_id': current_trip[0]['user'], 'trip': current_trip.copy()})
            current_trip = [{'user': u_idx, 'poi': v_idx, 'cat': c_idx, 'time': curr_time}]
            continue
            
        last_time = current_trip[-1]['time']
        diff_hours = (curr_time - last_time).total_seconds() / 3600.0
        
        # 统计逗留时间，忽略异常大或小的值（如过夜）
        if 10 < diff_hours * 60.0 < 240:
            cat_dwell_times[current_trip[-1]['cat']].append(diff_hours * 60.0 * 0.5)
            
        # 根据时间间隔截断行程
        if diff_hours > CONFIG['max_time_gap_hours']:
            if CONFIG['min_trip_len'] <= len(current_trip) <= CONFIG['max_trip_len']:
                itineraries.append({'user_id': u_idx, 'trip': current_trip.copy()})
            current_trip = [{'user': u_idx, 'poi': v_idx, 'cat': c_idx, 'time': curr_time}]
        else:
            current_trip.append({'user': u_idx, 'poi': v_idx, 'cat': c_idx, 'time': curr_time})
            
    if current_trip and CONFIG['min_trip_len'] <= len(current_trip) <= CONFIG['max_trip_len']:
        itineraries.append({'user_id': current_trip[0]['user'], 'trip': current_trip.copy()})
            
    # 计算每个类别的平均停留时间，缺乏数据的默认为45分钟
    avg_cat_dwell = {c: np.mean(times) if times else 45.0 for c, times in cat_dwell_times.items()}
    return itineraries, num_users, num_venues_with_pad, venue_pad_idx, poi_info, avg_cat_dwell

def build_global_spatiotemporal_tensors(poi_info, avg_cat_dwell, num_venues_with_pad, pad_idx, itineraries):
    """
    基于所有行程数据，构建物理与拓扑层面的全局约束张量：
    通行时间矩阵、逗留时间向量、共现状态转移矩阵、节点热度向量。
    """
    print(">>> 2. 构建全局时空约束张量与归一化物理先验矩阵...")
    transit_matrix = torch.full((num_venues_with_pad, num_venues_with_pad), fill_value=30.0)
    dwell_tensor = torch.full((num_venues_with_pad,), fill_value=45.0)
    transition_matrix = torch.zeros((num_venues_with_pad, num_venues_with_pad))
    pop_tensor = torch.zeros((num_venues_with_pad,))
    
    # 统计转移矩阵与热度
    for item in itineraries:
        seq = [step['poi'] for step in item['trip']]
        for i in range(len(seq)):
            pop_tensor[seq[i]] += 1
            if i < len(seq) - 1:
                transition_matrix[seq[i], seq[i+1]] += 1
                transition_matrix[seq[i+1], seq[i]] += 1 
                
    pop_tensor = torch.log1p(pop_tensor) 
    pop_tensor = pop_tensor / (pop_tensor.max() + 1e-9) 
    
    # 归一化转移矩阵
    row_sum = transition_matrix.sum(dim=1, keepdim=True)
    transition_matrix = transition_matrix / (row_sum + 1e-9)
    
    # 赋予每个POI逗留时间
    for v_idx, info in poi_info.items():
        dwell_tensor[v_idx] = avg_cat_dwell.get(info['cat'], 45.0)
        
    # 根据地理坐标计算通行时间矩阵
    poi_indices = list(poi_info.keys())
    for i in tqdm(range(len(poi_indices)), desc="计算物理交通矩阵"):
        v1 = poi_indices[i]
        for j in range(i+1, len(poi_indices)):
            v2 = poi_indices[j]
            dist_km = haversine(poi_info[v1]['lon'], poi_info[v1]['lat'], poi_info[v2]['lon'], poi_info[v2]['lat'])
            mins = (dist_km / CONFIG['avg_speed_kmh']) * 60.0
            transit_matrix[v1, v2] = mins
            transit_matrix[v2, v1] = mins
            
    transit_matrix.fill_diagonal_(0.0)
    # 处理 Padding 节点
    transit_matrix[pad_idx, :] = 0.0
    transit_matrix[:, pad_idx] = 0.0
    dwell_tensor[pad_idx] = 0.0
    
    norm_transit = transit_matrix / (transit_matrix.max() + 1e-9)
    return transit_matrix.float(), dwell_tensor.float(), transition_matrix.float(), pop_tensor.float(), norm_transit.float()

# ==============================================================================
# PyTorch Dataset
# ==============================================================================
class ItineraryDataset(Dataset):
    """
    组装供模型消费的张量数据集。提取起始点、时间预算、目标序列。
    """
    def __init__(self, itineraries, venue_pad_idx):
        self.data = []
        for item in itineraries:
            trip = item['trip']
            seq = [step['poi'] for step in trip]
            total_mins = (trip[-1]['time'] - trip[0]['time']).total_seconds() / 60.0
            total_mins = max(30.0, min(total_mins, 720.0)) # 截断异常长短的游览时长
            
            # 提取起点的绝对一天内分钟数
            start_time_min = trip[0]['time'].hour * 60 + trip[0]['time'].minute
            
            self.data.append({
                'user_id': item['user_id'],
                'start_poi': seq[0],
                'start_time_min': start_time_min,
                'time_budget': total_mins,
                'target_seq': seq[1:], 
                'seq_len': len(seq) - 1
            })
            
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch, pad_idx):
    """按 Batch 进行变长序列打平与对齐填充"""
    user_ids = torch.tensor([x['user_id'] for x in batch], dtype=torch.long)
    start_pois = torch.tensor([x['start_poi'] for x in batch], dtype=torch.long)
    start_times = torch.tensor([x['start_time_min'] for x in batch], dtype=torch.float)
    time_budgets = torch.tensor([x['time_budget'] for x in batch], dtype=torch.float)
    
    max_len = max(x['seq_len'] for x in batch)
    target_seqs = [x['target_seq'] + [pad_idx] * (max_len - x['seq_len']) for x in batch]
    target_seqs = torch.tensor(target_seqs, dtype=torch.long)
    
    return user_ids, start_pois, start_times, time_budgets, target_seqs