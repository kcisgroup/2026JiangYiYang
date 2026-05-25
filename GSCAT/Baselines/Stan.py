import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- 1. 配置参数 ---
class Config:
    BASE_DIR = os.getcwd()
    DATA_PATH = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_nyc_cleaned.csv')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模型参数
    EMBED_SIZE = 50          # 论文建议 TKY 数据集设为 50
    MAX_SEQ_LEN = 100        # 论文设置的最大序列长度
    DROPOUT = 0.2
    
    # 训练参数
    BATCH_SIZE = 64          # STAN 显存占用较大，适当减小 Batch Size
    LEARNING_RATE = 0.003    # 论文推荐 0.003
    EPOCHS = 30
    WEIGHT_DECAY = 1e-5

    # 数据处理参数
    MIN_TRAJ_LEN = 5         # 最短轨迹长度限制

    # 评估参数
    TOP_K = [1, 5, 10]

# --- 2. 辅助函数 ---
def haversine_np(lon1, lat1, lon2, lat2):
    """
    使用 numpy 计算 Haversine 距离 (km)
    支持广播机制
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

# --- 3. 数据预处理 ---
def preprocess_data_for_stan(config):
    print("Step 1: Loading data...")
    df = pd.read_csv(config.DATA_PATH)
    df.rename(columns={'geo_id': 'poi_id'}, inplace=True)
    
    # ID 映射
    poi_map = {poi: i + 1 for i, poi in enumerate(df['poi_id'].unique())} # 0 for padding
    user_map = {user: i for i, user in enumerate(df['user_id'].unique())}
    n_pois = len(poi_map) + 1
    n_users = len(user_map)
    
    df['poi_id_mapped'] = df['poi_id'].map(poi_map)
    df['user_id_mapped'] = df['user_id'].map(user_map)
    
    # 处理时间：论文将一周分为 7*24 = 168 个小时段
    df['time'] = pd.to_datetime(df['time'])
    df['hour_of_week'] = df['time'].dt.dayofweek * 24 + df['time'].dt.hour
    
    # 保存所有 POI 的 GPS 信息，用于后续计算距离矩阵
    # 创建一个 Tensor: [n_pois, 2] -> (lat, lon)
    poi_gps = torch.zeros((n_pois, 2), dtype=torch.float32)
    # 填充 GPS (注意：这里假设 poi_id_mapped 是唯一的 key)
    temp_gps = df[['poi_id_mapped', 'latitude', 'longitude']].drop_duplicates('poi_id_mapped') # type: ignore
    for _, row in temp_gps.iterrows():
        pid = int(row['poi_id_mapped'])
        poi_gps[pid, 0] = row['latitude']
        poi_gps[pid, 1] = row['longitude']

    df.sort_values(by=['user_id_mapped', 'time'], inplace=True)

    print("Step 2: Generating samples and splitting (8:1:1)...")
    train_data, valid_data, test_data = [], [], []
    
    user_groups = df.groupby('user_id_mapped')
    for user_id, group in tqdm(user_groups, desc="Processing users"):
        traj = group['poi_id_mapped'].tolist()
        times = group['hour_of_week'].tolist()
        # 我们不需要在这里存储 GPS 序列，因为可以通过 POI ID 查表
        
        num_checkins = len(traj)
        if num_checkins < config.MIN_TRAJ_LEN:
            continue
            
        train_end_idx = int(num_checkins * 0.8)
        valid_end_idx = int(num_checkins * 0.9)
        
        # STAN 需要固定长度序列，我们切片
        for i in range(1, num_checkins):
            # 获取历史序列
            start_idx = max(0, i - config.MAX_SEQ_LEN)
            seq_pois = traj[start_idx:i]
            seq_times = times[start_idx:i]
            target_poi = traj[i]
            target_time = times[i]
            
            # 为了计算候选矩阵，我们需要目标的真实 GPS (评估时用到) 
            # 但 STAN 的特点是需要计算 历史->所有候选 的距离。
            # 我们只需要存 User ID 和 Input Sequence 即可，距离在 Model 或 Dataset 中算
            
            sample = {
                'user': user_id,
                'traj_pois': seq_pois,
                'traj_times': seq_times,
                'target_poi': target_poi,
                'target_time': target_time # 预测下一时刻去哪，需要下一时刻的时间戳计算时间差
            }
            
            if i < train_end_idx:
                train_data.append(sample)
            elif i < valid_end_idx:
                valid_data.append(sample)
            else:
                test_data.append(sample)
                
    return train_data, valid_data, test_data, n_users, n_pois, poi_gps

# --- 4. Dataset ---
class STANDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'user': torch.tensor(item['user'], dtype=torch.long),
            'traj_pois': torch.tensor(item['traj_pois'], dtype=torch.long),
            'traj_times': torch.tensor(item['traj_times'], dtype=torch.long),
            'target_poi': torch.tensor(item['target_poi'], dtype=torch.long),
            'target_time': torch.tensor(item['target_time'], dtype=torch.long)
        }

def collate_fn(batch):
    traj_pois = [item['traj_pois'] for item in batch]
    traj_times = [item['traj_times'] for item in batch]
    users = torch.stack([item['user'] for item in batch])
    targets = torch.stack([item['target_poi'] for item in batch])
    target_times = torch.stack([item['target_time'] for item in batch])
    
    # Padding
    padded_pois = pad_sequence(traj_pois, batch_first=True, padding_value=0)
    padded_times = pad_sequence(traj_times, batch_first=True, padding_value=0)
    
    # Mask: 1 for valid, 0 for padding (STAN 论文中的 Mask 定义)
    mask = (padded_pois != 0).float()
    
    return {
        'user': users,
        'traj_pois': padded_pois,
        'traj_times': padded_times,
        'mask': mask,
        'target_poi': targets,
        'target_time': target_times
    }

# --- 5. STAN 模型定义 ---
class STAN(nn.Module):
    def __init__(self, n_users, n_pois, poi_gps, config):
        super(STAN, self).__init__()
        self.config = config
        self.n_pois = n_pois
        self.emb_size = config.EMBED_SIZE
        
        # 注册 POI GPS 表 (Lat, Lon)
        self.register_buffer('poi_gps', poi_gps)
        
        # 1. Embeddings
        self.user_emb = nn.Embedding(n_users, self.emb_size)
        self.poi_emb = nn.Embedding(n_pois, self.emb_size, padding_idx=0)
        self.time_emb = nn.Embedding(169, self.emb_size, padding_idx=0) # 168 hours + 1 padding
        
        # Spatio-Temporal Embeddings (Unit embeddings)
        # 论文 Eq 3: e_delta_t * delta_t. 这里简化为学习离散的 bin embedding，更稳定
        self.st_time_emb = nn.Embedding(1000, self.emb_size) # 假设最大时间间隔 bucket
        self.st_dist_emb = nn.Embedding(1000, self.emb_size) # 假设最大距离 bucket
        
        # 2. Self-Attention Aggregation Layers
        self.query_mat = nn.Linear(self.emb_size, self.emb_size)
        self.key_mat = nn.Linear(self.emb_size, self.emb_size)
        self.value_mat = nn.Linear(self.emb_size, self.emb_size)
        
        # 3. Attention Matching Layers
        # 这里的参数用于匹配候选集
        self.match_linear = nn.Linear(self.emb_size, self.emb_size)
        
        self.dropout = nn.Dropout(config.DROPOUT)
        self.softmax = nn.Softmax(dim=-1)

    def _get_st_relation(self, times, pois, target_time=None, target_pois=None):
        """
        计算时空关系矩阵。
        如果 target_time/pois 为 None，计算轨迹内部的自相关 (Seq, Seq)。
        否则计算轨迹到候选集的相关 (Seq, L)。
        """
        batch_size, seq_len = times.shape
        
        # GPS Coordinates
        # pois: (B, S), poi_gps: (N, 2) -> (B, S, 2)
        coords = self.poi_gps[pois] 
        
        # --- 计算时间差 (Delta T) ---
        if target_time is None:
            # Trajectory Self-Relation: (B, S, S)
            # t_i - t_j
            t1 = times.unsqueeze(2) # (B, S, 1)
            t2 = times.unsqueeze(1) # (B, 1, S)
            delta_t = (t1 - t2).abs()
        else:
            # Candidate Relation: (B, S, L)
            # target_time (B), times (B, S)
            # 我们假设 target_time 是标量，要扩展到 L 个候选
            # 但实际上 STAN 比较的是 (Trajectory Point) vs (Candidate Point)
            # Candidate Time 就是 target_time (下一跳时间)
            t_traj = times.unsqueeze(2) # (B, S, 1)
            t_cand = target_time.view(batch_size, 1, 1) # (B, 1, 1) - 广播到 L
            delta_t = (t_traj - t_cand).abs() # (B, S, 1) -> 广播

        # --- 计算空间距离 (Delta S) ---
        if target_pois is None:
            # Self-Relation (B, S, S)
            # 为避免 OOM 和太慢，使用欧氏距离近似 Haversine，或者简化处理
            # 这里为了性能，对 lat/lon 进行简单的欧氏距离近似 (对于小范围 TKY 是可接受的)
            c1 = coords.unsqueeze(2) # (B, S, 1, 2)
            c2 = coords.unsqueeze(1) # (B, 1, S, 2)
            # 简单的 lat/lon 差值并缩放 (近似 km)
            diff = (c1 - c2).abs()
            delta_s = (diff * 111.0).sum(dim=-1) # 粗略 1度=111km
        else:
            # Candidate Relation (B, S, L)
            # target_pois (L, 2)
            c_traj = coords.unsqueeze(2) # (B, S, 1, 2)
            c_cand = self.poi_gps.unsqueeze(0).unsqueeze(0) # (1, 1, N, 2) -> 全局候选
            # 这里的 N 是所有 POI。显存可能不够。
            # 论文中 L 是所有 Locations。
            # 为了能在 GPU 跑，我们计算 (B, S, N) 会非常大。
            # 优化：只计算 diff，不展开全部
            # 实际上，match layer 是 E(l) matching S(u).
            # 让我们使用欧氏距离近似，或者 batch 处理
            pass 
            # 暂时用占位符，下面 forward 里单独处理
            delta_s = None

        return delta_t, delta_s

    def forward(self, batch):
        user = batch['user']
        seq_pois = batch['traj_pois']
        seq_times = batch['traj_times']
        mask = batch['mask'] # (B, S)
        target_time = batch['target_time'] # (B)
        
        batch_size, seq_len = seq_pois.shape
        
        # --- 1. Multimodal Embedding ---
        u_e = self.user_emb(user).unsqueeze(1) # (B, 1, D)
        l_e = self.poi_emb(seq_pois)           # (B, S, D)
        t_e = self.time_emb(seq_times)         # (B, S, D)
        
        # Trajectory Embedding E(u)
        e_u = u_e + l_e + t_e # (B, S, D)
        
        # --- 2. Spatio-Temporal Relation Matrix (Self) ---
        # 计算轨迹内部距离矩阵
        # (B, S, S)
        coords = self.poi_gps[seq_pois] # (B, S, 2)
        # 距离差 (简化计算: 欧氏距离 * 系数)
        d_s = torch.cdist(coords, coords) * 111.0 # (B, S, S)
        # 时间差
        t1 = seq_times.unsqueeze(2)
        t2 = seq_times.unsqueeze(1)
        d_t = (t1 - t2).abs() # (B, S, S)
        
        # 离散化并获取 Embedding
        # 限制最大值防止越界
        d_s = torch.clamp(d_s.long(), 0, 999)
        d_t = torch.clamp(d_t.long(), 0, 999)
        
        e_ds = self.st_dist_emb(d_s) # (B, S, S, D)
        e_dt = self.st_time_emb(d_t) # (B, S, S, D)
        e_delta = e_ds + e_dt        # (B, S, S, D) - 论文 Eq 5 E(Delta)
        # 聚合最后一维: 论文里是 Weighted Sum，这里简化为 Sum
        # 原论文: E(Delta) 是 N x N (即 S x S)。但在 Transformer 中我们需要加到 Attention Score 上。
        # Attention Score 是 (B, S, S)。所以我们需要把 (B, S, S, D) 变成 (B, S, S)?
        # 不，论文 Eq 7: (Q K^T + Delta) / sqrt(d). 这里的 Delta 是标量矩阵吗？
        # 论文 Eq 3 定义 e_delta 是向量。
        # 论文 Eq 7 中的 Delta 必须是 (S, S) 形状才能加到 QK^T 上。
        # 通常做法：将 e_delta (S, S, D) 通过线性层投影到 (S, S, 1) 或者直接取 mean/sum。
        # 这里我们取 sum(dim=-1) 作为 Attention Bias
        attn_bias = e_delta.sum(dim=-1) # (B, S, S)

        # --- 3. Self-Attention Aggregation ---
        Q = self.query_mat(e_u) # (B, S, D)
        K = self.key_mat(e_u)   # (B, S, D)
        V = self.value_mat(e_u) # (B, S, D)
        
        # Scaled Dot-Product with ST Bias
        scores = (torch.matmul(Q, K.transpose(-2, -1)) + attn_bias) / math.sqrt(self.emb_size) # (B, S, S)
        
        # Masking padding
        # mask (B, S) -> (B, 1, 1, S) or (B, 1, S)
        # scores (B, S, S)
        mask_expanded = mask.unsqueeze(1).expand(-1, seq_len, -1) # (B, S, S)
        scores = scores.masked_fill(mask_expanded == 0, -1e9)
        
        attn_weights = self.softmax(scores)
        S_u = torch.matmul(attn_weights, V) # (B, S, D) - Updated representation
        
        # --- 4. Attention Matching Layer ---
        # 论文 Eq 8: Matching(E(l), S(u), E(N))
        # 我们需要计算 History S(u) 和 All Candidates E(l) 的匹配度
        # E(l): (N_POI, D)
        cand_emb = self.poi_emb.weight # (N_POI, D)
        
        # 计算 Candidate Spatio-Temporal Relation Matrix E(N)
        # 这步计算量极大：(B, S) vs (N_POI). 
        # 为了高效，STAN 通常只对最后一个访问点或者做近似。
        # 但论文说是 sum-up representation。
        # 让我们简化：只计算最后一个有效历史点与所有候选点的距离嵌入。
        # 或者，为了完全复现，我们必须计算 (B, S, N_POI) 的距离矩阵，这在内存上是不可能的 (64*100*4000*4 bytes ~ 100MB 看起来还行?)
        # TKY 数据集 POI 数量不多 (约 8k)，显存够用。
        
        # 1. 扩展 S(u) -> (B, S, N_POI)? 不，太大了。
        # 让我们换个思路：Match Score = Sum_over_S ( Softmax( (q k^T + N) ) )
        # 这里 q 是 candidate (N_POI, D), k 是 history (B, S, D).
        # output (B, N_POI)
        
        # 计算 k^T * q -> (B, S, N_POI)
        # q: (N_POI, D), k: (B, S, D)
        # matmul(k, q.T) -> (B, S, N_POI)
        interaction = torch.matmul(S_u, cand_emb.t()) # (B, S, N_POI)
        
        # 计算 E(N): (B, S, N_POI) 的时空嵌入 Bias
        # 空间距离
        # coords (B, S, 2), all_coords (N_POI, 2)
        # cdist (B, S, 2) vs (1, N_POI, 2) -> (B, S, N_POI)
        dist_mat = torch.cdist(coords, self.poi_gps.unsqueeze(0)) * 111.0
        dist_mat = torch.clamp(dist_mat.long(), 0, 999)
        e_n_s = self.st_dist_emb(dist_mat).sum(dim=-1) # (B, S, N_POI)
        
        # 时间距离
        # times (B, S), target_time (B)
        # dt = |times - target_time|
        dt_mat = (seq_times.unsqueeze(2) - target_time.view(batch_size, 1, 1)).abs() # (B, S, 1) -> Broadcast to N_POI
        dt_mat = torch.clamp(dt_mat.long(), 0, 999)
        # 扩展到 N_POI 维度
        dt_mat = dt_mat.expand(-1, -1, self.n_pois)
        e_n_t = self.st_time_emb(dt_mat).sum(dim=-1) # (B, S, N_POI)
        
        st_bias_cand = e_n_s + e_n_t # (B, S, N_POI)
        
        # Matching Score
        # (B, S, N_POI)
        match_scores = (interaction + st_bias_cand) / math.sqrt(self.emb_size)
        
        # Mask padding in sequence dimension
        mask_cand = mask.unsqueeze(2) # (B, S, 1)
        match_scores = match_scores.masked_fill(mask_cand == 0, -1e9)
        
        # Softmax over Sequence dimension (论文 Eq 9: Sum(Softmax(...)))
        # 注意：这里是对 S 维度做 softmax 还是 Sum？
        # Eq 9: Sum(softmax(...)). 通常 Attention 是对 Key 维度 (S) 做 Softmax。
        # 这意味着：对于每个候选 POI，历史轨迹中哪一个点最相关？
        probs = self.softmax(match_scores) # Softmax over last dim (N_POI)?? No.
        # Attention matching should calculate weight for each history step.
        # 所以 softmax 应该在 dim=1 (Sequence length)
        # 含义：对于候选集中的每个点 c，它是历史中哪些点的延续？
        attn_cand = F.softmax(match_scores, dim=1) # (B, S, N_POI)
        
        # Sum-up (Eq 9)
        # 这里的 Sum 实际上是加权求和，但在 Eq 9 中似乎直接 Sum probabilities?
        # 或者是 Sum(Last Dimension of Embedding)?
        # 让我们仔细看 Eq 9: Sum(softmax(...)). 
        # 按照 STAN 逻辑，这是聚合了所有历史步骤对当前候选的“投票”。
        final_scores = torch.sum(attn_cand, dim=1) # (B, N_POI)
        
        # 因为前面已经做过 Softmax，这里的 Sum 结果可能大于 1。
        # 为了配合 CrossEntropyLoss (需要 Logits)，我们在外面再取 Log 或者直接输出。
        # 但标准的 CE 需要 Logits。
        # 如果我们把这里看作概率，我们可以加上一个 epsilon 然后取 log
        final_logits = torch.log(final_scores + 1e-8)
        
        return final_logits

# --- 6. 训练和评估函数 (复用标准流程) ---
def train_stan(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        for k, v in batch.items():
            batch[k] = v.to(device)

        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch['target_poi'])
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, top_k, device):
    model.eval()
    metrics = {f'acc@{k}': 0 for k in top_k}
    metrics['mrr'] = 0.0 # type: ignore
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            for k, v in batch.items():
                batch[k] = v.to(device)
                
            logits = model(batch)
            target = batch['target_poi']

            max_k = max(top_k)
            _, top_indices = torch.topk(logits, max_k, dim=1) # (batch_size, max_k)

            expanded_target = target.view(-1, 1).expand_as(top_indices)
            hits_matrix = (top_indices == expanded_target)

            for k in top_k:
                hits_in_k = hits_matrix[:, :k].any(dim=1).sum().item()
                metrics[f'acc@{k}'] += hits_in_k
            
            match_indices = torch.nonzero(hits_matrix, as_tuple=True)
            reciprocal_ranks = torch.zeros(target.size(0)).to(device)
            ranks = match_indices[1].float() + 1.0
            reciprocal_ranks.scatter_(0, match_indices[0], 1.0 / ranks)
            metrics['mrr'] += reciprocal_ranks.sum().item() # type: ignore
                
            total_samples += target.size(0)

    for k in top_k:
        metrics[f'acc@{k}'] /= total_samples # type: ignore
    metrics['mrr'] /= total_samples # type: ignore
        
    return metrics

# --- 7. 主程序 ---
if __name__ == '__main__':
    config = Config()
    print(f"Using device: {config.DEVICE}")

    # 1. 数据处理
    train_data, valid_data, test_data, n_users, n_pois, poi_gps = preprocess_data_for_stan(config)
    print(f"\nNumber of POIs: {n_pois}")
    print(f"Train samples: {len(train_data)}")
    print(f"Valid samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")

    # 2. DataLoader
    train_dataset = STANDataset(train_data)
    valid_dataset = STANDataset(valid_data)
    test_dataset = STANDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. 初始化模型
    # poi_gps 需要移动到 CPU tensor，模型初始化时会注册为 buffer 并自动随模型移动
    model = STAN(n_users, n_pois, poi_gps, config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 4. 训练循环
    best_val_mrr = 0.0
    best_model_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")
        avg_train_loss = train_stan(model, train_loader, optimizer, criterion, config.DEVICE)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        # 验证
        val_metrics = evaluate(model, valid_loader, config.TOP_K, config.DEVICE)
        print("--- Validation Metrics ---")
        for metric, value in val_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # 早停
        current_val_mrr = val_metrics['mrr']
        if current_val_mrr > best_val_mrr:
            best_val_mrr = current_val_mrr
            best_model_state = model.state_dict()
            patience_counter = 0
            print("Validation MRR improved! Saving model state.")
        else:
            patience_counter += 1
            print(f"No improvement in validation MRR for {patience_counter} epoch(s).")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs.")
            break

    # 5. 最终测试
    print("\n--- Training Finished ---")
    if best_model_state:
        print("Loading best model from validation phase for final testing...")
        model.load_state_dict(best_model_state)
    
    print("\n--- Final Test Metrics ---")
    test_metrics = evaluate(model, test_loader, config.TOP_K, config.DEVICE)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")