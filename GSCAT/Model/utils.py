import torch
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from collections import defaultdict
from torch.optim.lr_scheduler import LambdaLR
from config import TIME_SEGMENT_RULES, TIME_SEGMENT_CATEGORIES, TIME_SEGMENT_PAD_IDX, NUM_TIME_SEGMENTS

def set_seed(seed_value):
    """设置所有随机种子以确保结果可复现"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def generate_cyclical_encoding_table(max_val: int, embed_dim: int) -> torch.Tensor:
    """生成周期性特征（如一天中的小时）的正弦/余弦位置编码表"""
    if embed_dim == 0: 
        return torch.empty(max_val, 0)
    if embed_dim % 2 != 0: 
        embed_dim += 1
    position = torch.arange(max_val, dtype=torch.float).unsqueeze(1)
    num_timescales = embed_dim // 2
    div_term = torch.exp(torch.arange(0, num_timescales, dtype=torch.float) * (-math.log(10000.0) / num_timescales))
    encoding_table = torch.zeros(max_val, embed_dim)
    encoding_table[:, 0:num_timescales] = torch.sin(position * div_term)
    if embed_dim > num_timescales:
        encoding_table[:, num_timescales:2*num_timescales] = torch.cos(position * div_term)
    return encoding_table

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """带线性预热和余弦退火的学习率调度器"""
    def lr_lambda(current_step):
        if num_training_steps <= 0: return 0.0
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_time_segment_id(timestamp: pd.Timestamp) -> int:
    """根据规则将连续的时间戳映射为离散的语义时间段ID"""
    if pd.isna(timestamp): return TIME_SEGMENT_PAD_IDX
    weekday, hour = timestamp.weekday(), timestamp.hour
    for seg_name, rule_func in TIME_SEGMENT_RULES:
        if rule_func(weekday, hour): return TIME_SEGMENT_CATEGORIES[seg_name]
    return TIME_SEGMENT_CATEGORIES.get("LATE_NIGHT_DEEP", NUM_TIME_SEGMENTS - 1)

def bin_series_to_int(series, bins, labels, pad_value):
    """使用Pandas将序列中的连续值离散化为整数标签"""
    if not isinstance(series, pd.Series): series = pd.Series(series)
    binned_series = pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)
    return [int(x) if pd.notna(x) else pad_value for x in binned_series]

def bin_pairwise_time_diffs_torch(pairwise_diff_minutes_tensor, bins_list):
    """在PyTorch层面上高速执行张量分箱（适用于成对注意力机制的时间差）"""
    boundaries_tensor = torch.tensor(bins_list[1:-1], dtype=pairwise_diff_minutes_tensor.dtype, device=pairwise_diff_minutes_tensor.device)
    return torch.bucketize(pairwise_diff_minutes_tensor, boundaries_tensor, right=False)

def load_precomputed_distances(distance_file_path, venue_map):
    """加载预计算的POI之间距离，返回距离查找字典"""
    if not venue_map:
        print("错误: venue_map未传入，无法加载距离数据。")
        return None
    try: 
        dist_df = pd.read_csv(distance_file_path)
    except FileNotFoundError: 
        print(f"错误: 距离文件未找到 {distance_file_path}")
        return None
    
    dist_lookup = defaultdict(dict)
    max_venue_idx = max(venue_map.values()) if venue_map else -1
    for _, row in tqdm(dist_df.iterrows(), total=len(dist_df), desc="构建距离查找表"):
        try:
            v1_idx, v2_idx, dist = int(row['venue1']), int(row['venue2']), float(row['distance'])
            if 0 <= v1_idx <= max_venue_idx and 0 <= v2_idx <= max_venue_idx:
                dist_lookup[v1_idx][v2_idx] = dist
                dist_lookup[v2_idx][v1_idx] = dist
        except (ValueError, KeyError):
            continue
    return dist_lookup

def info_nce_loss(anchor_embeds, positive_embeds, negative_embeds, temperature=0.1):
    """计算InfoNCE对比学习损失 (使得anchor与positive接近，与negatives远离)"""
    positive_embeds = positive_embeds.unsqueeze(1) # [B, 1, D]
    all_candidates = torch.cat([positive_embeds, negative_embeds], dim=1) # [B, 1+N, D]
    logits = torch.bmm(anchor_embeds.unsqueeze(1), all_candidates.transpose(1, 2)).squeeze(1) # [B, 1+N]
    logits = logits / temperature
    labels = torch.zeros(anchor_embeds.size(0), dtype=torch.long, device=anchor_embeds.device)
    return F.cross_entropy(logits, labels)