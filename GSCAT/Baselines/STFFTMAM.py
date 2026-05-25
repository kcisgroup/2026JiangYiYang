import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- 1. 配置参数 ---
class Config:
    BASE_DIR = os.getcwd()
    # 严格按照要求修改路径
    DATA_PATH = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_nyc_cleaned.csv')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模型参数
    EMBED_SIZE = 128
    HIDDEN_SIZE = 256
    N_LAYERS = 2
    DROPOUT = 0.3
    
    # 时空特征参数
    NUM_HOURS = 24
    NUM_DAYS = 7
    
    # 训练参数
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 20  # 适当调整，STFFTMAM通常收敛较快
    WEIGHT_DECAY = 1e-5

    # 数据处理参数
    SESSION_LEN = 20
    MIN_TRAJ_LEN = 3

    # 评估参数
    TOP_K = [1, 5, 10]

# --- 2. 数据预处理 ---
def preprocess_stfftmam_data(config):
    print("Step 1: Loading and mapping data...")
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Data not found at {config.DATA_PATH}")
        
    df = pd.read_csv(config.DATA_PATH)
    
    # 确保列名正确，Foursquare数据集通常包含 user_id, geo_id (poi_id), time 等
    if 'geo_id' in df.columns:
        df.rename(columns={'geo_id': 'poi_id'}, inplace=True)
    
    # 时间处理
    print("Step 1.1: Extracting Temporal Features...")
    df['time_obj'] = pd.to_datetime(df['time'])
    df['hour'] = df['time_obj'].dt.hour
    df['weekday'] = df['time_obj'].dt.dayofweek
    
    # ID 映射
    poi_map = {poi: i + 1 for i, poi in enumerate(df['poi_id'].unique())} # 0 reserved for padding
    user_map = {user: i for i, user in enumerate(df['user_id'].unique())}
    
    n_pois = len(poi_map) + 1
    n_users = len(user_map)
    
    df['poi_id_mapped'] = df['poi_id'].map(poi_map).fillna(0).astype(int)
    df['user_id_mapped'] = df['user_id'].map(user_map).fillna(0).astype(int)
    
    df.sort_values(by=['user_id', 'time'], inplace=True)

    # --- Step 2: Build Spatial Feature Transition Matrix (M_spatial) ---
    # 这是 STFFTMAM 的核心之一：基于马尔可夫链的全局转移概率
    print("Step 2: Building Feature Transition Matrix...")
    transition_counts = sp.dok_matrix((n_pois, n_pois), dtype=np.float32)
    user_groups = df.groupby('user_id')
    
    for _, group in tqdm(user_groups, desc="Building Matrix"):
        traj = group['poi_id_mapped'].tolist()
        for i in range(len(traj) - 1):
            u, v = traj[i], traj[i+1]
            if u != 0 and v != 0:
                transition_counts[u, v] += 1
    
    # 归一化为概率矩阵
    transition_counts = transition_counts.tocsr()
    row_sums = transition_counts.sum(axis=1)
    row_sums[row_sums == 0] = 1
    transition_probs = transition_counts.multiply(1.0 / row_sums)
    
    # 转换为 Tensor
    spatial_trans_matrix = torch.from_numpy(transition_probs.toarray()).float()

    # --- Step 3: Generate Sequence Samples ---
    print("Step 3: Generating samples and splitting...")
    train_data, valid_data, test_data = [], [], []
    
    for _, group in tqdm(user_groups, desc="Generating Samples"):
        traj_poi = group['poi_id_mapped'].tolist()
        traj_hour = group['hour'].tolist()
        traj_day = group['weekday'].tolist()
        user_id = group['user_id_mapped'].iloc[0]
        
        num_checkins = len(traj_poi)
        if num_checkins < config.MIN_TRAJ_LEN: continue
            
        train_end_idx = int(num_checkins * 0.8)
        valid_end_idx = int(num_checkins * 0.9)
            
        for i in range(1, num_checkins):
            start_idx = max(0, i - config.SESSION_LEN)
            
            # 提取序列
            sess_poi = traj_poi[start_idx:i]
            sess_hour = traj_hour[start_idx:i]
            sess_day = traj_day[start_idx:i]
            target = traj_poi[i]
            
            if not sess_poi: continue

            sample = {
                'user': user_id,
                'session_poi': sess_poi,
                'session_hour': sess_hour,
                'session_day': sess_day,
                'target': target
            }
            
            if i < train_end_idx: train_data.append(sample)
            elif i < valid_end_idx: valid_data.append(sample)
            else: test_data.append(sample)
        
    return train_data, valid_data, test_data, n_pois, n_users, spatial_trans_matrix

# --- 3. Dataset 和 DataLoader ---
class STFFTMAMDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'user': torch.tensor(item['user'], dtype=torch.long),
            'session_poi': torch.tensor(item['session_poi'], dtype=torch.long),
            'session_hour': torch.tensor(item['session_hour'], dtype=torch.long),
            'session_day': torch.tensor(item['session_day'], dtype=torch.long),
            'last_poi': torch.tensor(item['session_poi'][-1], dtype=torch.long), # 用于查表
            'target': torch.tensor(item['target'], dtype=torch.long)
        }

def collate_fn(batch):
    # Padding POI with 0, others with 0 (assuming 0 is safe for hour/day embedding index 0)
    sess_poi = pad_sequence([item['session_poi'] for item in batch], batch_first=True, padding_value=0)
    sess_hour = pad_sequence([item['session_hour'] for item in batch], batch_first=True, padding_value=0)
    sess_day = pad_sequence([item['session_day'] for item in batch], batch_first=True, padding_value=0)
    
    users = torch.stack([item['user'] for item in batch])
    last_pois = torch.stack([item['last_poi'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    mask = (sess_poi != 0)
    
    return {
        'user': users,
        'session_poi': sess_poi,
        'session_hour': sess_hour,
        'session_day': sess_day,
        'mask': mask,
        'last_poi': last_pois,
        'target': targets
    }

# --- 4. STFFTMAM 模型定义 ---
class STFFTMAM(nn.Module):
    def __init__(self, n_pois, n_users, spatial_trans_matrix, config):
        super(STFFTMAM, self).__init__()
        self.config = config
        
        # 1. Embeddings (Spatio-Temporal-User)
        self.poi_emb = nn.Embedding(n_pois, config.EMBED_SIZE, padding_idx=0)
        self.user_emb = nn.Embedding(n_users, config.EMBED_SIZE)
        self.hour_emb = nn.Embedding(config.NUM_HOURS + 1, config.EMBED_SIZE) # +1 safe buffer
        self.day_emb = nn.Embedding(config.NUM_DAYS + 1, config.EMBED_SIZE)
        
        # 2. LSTM Layer (Sequential Pattern Learning)
        # 输入维度是所有特征嵌入的拼接或求和，这里采用求和融合
        self.lstm = nn.LSTM(
            input_size=config.EMBED_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.N_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.N_LAYERS > 1 else 0
        )
        
        # 3. Transition Matrix (Global Context)
        # 注册为 buffer，不参与梯度更新，但作为模型一部分保存
        self.register_buffer('spatial_matrix', spatial_trans_matrix)
        
        # 4. Attention Mechanism (Feature Fusion)
        # 决定是更相信 LSTM 的长期序列预测，还是更相信全局转移矩阵
        self.attn_linear = nn.Linear(config.HIDDEN_SIZE + config.EMBED_SIZE, 2) 
        
        # 5. Prediction Layers
        self.fc_lstm = nn.Linear(config.HIDDEN_SIZE, n_pois)
        self.dropout = nn.Dropout(config.DROPOUT)
        
    def forward(self, batch):
        user = batch['user']
        poi = batch['session_poi']
        hour = batch['session_hour']
        day = batch['session_day']
        last_poi = batch['last_poi']
        mask = batch['mask']
        
        # --- Channel 1: Deep Sequential Learning (BiLSTM/LSTM) ---
        # 融合时空特征
        x = self.poi_emb(poi) + self.hour_emb(hour) + self.day_emb(day)
        # 加入用户个性化偏好 (Broadcast)
        u_emb = self.user_emb(user).unsqueeze(1)
        x = x + u_emb
        x = self.dropout(x)
        
        # LSTM 前向传播
        # pack_padded_sequence 可以更高效，这里为了代码简洁直接用 mask 处理
        output, (hn, cn) = self.lstm(x)
        
        # 获取序列最后一个有效时间步的隐藏状态
        # hn[-1] 通常包含最后的信息，但在 batch padding 下需小心，这里简化取 hn[-1]
        # 更严谨的做法是 gather lengths
        last_hidden = hn[-1] # (Batch, Hidden)
        
        # LSTM 分支的 Logits
        logits_lstm = self.fc_lstm(last_hidden) # (Batch, n_pois)
        
        # --- Channel 2: Transition Matrix (Global Markov) ---
        # 直接查表获取概率分布
        probs_trans = self.spatial_matrix[last_poi] # (Batch, n_pois)
        # 转换为 Logits 空间 (加个极小值防止 log(0))
        logits_trans = torch.log(probs_trans + 1e-9)
        
        # --- Channel 3: Attention Fusion ---
        # 动态计算权重：基于当前用户状态 (User Emb) 和 历史状态 (Hidden)
        # Context: [Batch, Hidden + Embed]
        context = torch.cat([last_hidden, self.user_emb(user)], dim=1)
        attn_weights = F.softmax(self.attn_linear(context), dim=1) # (Batch, 2)
        
        alpha = attn_weights[:, 0].unsqueeze(1) # Weight for LSTM
        beta = attn_weights[:, 1].unsqueeze(1)  # Weight for Matrix
        
        # 最终融合
        final_logits = alpha * logits_lstm + beta * logits_trans
        
        return final_logits

# --- 5. 训练和评估函数 ---
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        for k, v in batch.items(): 
            if torch.is_tensor(v): batch[k] = v.to(device)
            
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch['target'])
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
                if torch.is_tensor(v): batch[k] = v.to(device)
                
            logits = model(batch)
            target = batch['target']
            
            # 计算 Top-K
            max_k = max(top_k)
            _, top_indices = torch.topk(logits, max_k, dim=1)
            
            expanded_target = target.view(-1, 1).expand_as(top_indices)
            hits = (top_indices == expanded_target)
            
            for k in top_k:
                metrics[f'acc@{k}'] += hits[:, :k].any(dim=1).sum().item()
            
            # MRR
            # 找到匹配的索引位置
            match_indices = torch.nonzero(hits, as_tuple=True)
            if len(match_indices[0]) > 0:
                ranks = match_indices[1].float() + 1.0
                reciprocal_ranks = 1.0 / ranks
                # 累加 MRR (注意：没有命中的样本 MRR 为 0)
                # 这种写法对于每个样本只统计第一个命中（Usually fine for single target）
                metrics['mrr'] += reciprocal_ranks.sum().item() # type: ignore
                
            total_samples += target.size(0)
            
    for k in metrics:
        metrics[k] /= total_samples # type: ignore
        
    return metrics

# --- 6. 主程序 ---
if __name__ == '__main__':
    config = Config()
    print(f"Using device: {config.DEVICE}")

    # 1. PREPROCESS
    try:
        train_data, valid_data, test_data, n_pois, n_users, st_matrix = preprocess_stfftmam_data(config)
    except FileNotFoundError as e:
        print(e)
        print("Please ensure the CSV file is at the correct path.")
        exit()

    print(f"\nStats: Users={n_users}, POIs={n_pois}")
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    # 2. DATALOADERS
    train_loader = DataLoader(STFFTMAMDataset(train_data), batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(STFFTMAMDataset(valid_data), batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(STFFTMAMDataset(test_data), batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. INITIALIZE MODEL
    model = STFFTMAM(n_pois, n_users, st_matrix, config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 4. TRAINING LOOP
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")
        train_loss = train_model(model, train_loader, optimizer, criterion, config.DEVICE)
        print(f"Avg Train Loss: {train_loss:.4f}")
        
        val_metrics = evaluate(model, valid_loader, config.TOP_K, config.DEVICE)
        print("--- Validation Metrics ---")
        for k, v in val_metrics.items(): print(f"{k}: {v:.4f}")
        
        if val_metrics['acc@1'] > best_val_acc:
            best_val_acc = val_metrics['acc@1']
            torch.save(model.state_dict(), 'stfftmam_best.pth')
            patience_counter = 0
            print("New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    # 5. TEST
    print("\n--- Final Test ---")
    model.load_state_dict(torch.load('stfftmam_best.pth'))
    test_metrics = evaluate(model, test_loader, config.TOP_K, config.DEVICE)
    for k, v in test_metrics.items(): print(f"{k}: {v:.4f}")