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
    DATA_PATH = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_tky_cleaned.csv')
    DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    # 模型参数
    EMBED_SIZE = 128      # D_u, D_l, D_c
    TIME_EMBED_SIZE = 32  # D_t (通常时间嵌入维度小一点)
    HIDDEN_SIZE = 128     # LSTM 隐藏层维度
    DROPOUT = 0.3
    
    # 训练参数
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 30
    WEIGHT_DECAY = 1e-5

    # 数据处理参数
    SESSION_LEN = 20      # 短期序列长度
    MIN_TRAJ_LEN = 5      # 过滤过短的用户
    MAX_HIST_LEN = 200    # 长期历史的最大长度限制 (防止显存爆炸)

    # 评估参数
    TOP_K = [1, 5, 10]

# --- 2. 数据预处理 ---
def preprocess_data_for_lspl(config):
    print("Step 1: Loading and mapping data...")
    df = pd.read_csv(config.DATA_PATH)
    df.rename(columns={'geo_id': 'poi_id', 'venue_category_id': 'category_id'}, inplace=True)
    
    # ID 映射 (0 为 padding)
    poi_map = {poi: i + 1 for i, poi in enumerate(df['poi_id'].unique())}
    cat_map = {cat: i + 1 for i, cat in enumerate(df['category_id'].unique())}
    user_map = {user: i for i, user in enumerate(df['user_id'].unique())}
    
    n_pois = len(poi_map) + 1
    n_cats = len(cat_map) + 1
    n_users = len(user_map)
    
    df['poi_id'] = df['poi_id'].map(poi_map)
    df['category_id'] = df['category_id'].map(cat_map)
    df['user_id'] = df['user_id'].map(user_map)
    
    # 处理时间: 使用 Hour of Week (0-167)
    df['time'] = pd.to_datetime(df['time'])
    df['time_slot'] = df['time'].dt.dayofweek * 24 + df['time'].dt.hour
    df.sort_values(by=['user_id', 'time'], inplace=True)

    print("Step 2: Generating samples (8:1:1) & Building Long-term History...")
    train_data, valid_data, test_data = [], [], []
    
    # 用于存储每个用户的长期历史 (只包含训练集部分的数据)
    # 格式: user_id -> {'pois': [], 'cats': [], 'times': []}
    long_term_history = {}

    user_groups = df.groupby('user_id')
    
    for user_id, group in tqdm(user_groups, desc="Processing users"):
        pois = group['poi_id'].tolist()
        cats = group['category_id'].tolist()
        times = group['time_slot'].tolist()
        
        num_checkins = len(pois)
        if num_checkins < config.MIN_TRAJ_LEN:
            continue
            
        train_end_idx = int(num_checkins * 0.8)
        valid_end_idx = int(num_checkins * 0.9)
        
        # 1. 构建长期历史 (仅使用训练集部分)
        # 限制最大长度以提高效率
        hist_pois = pois[:train_end_idx][-config.MAX_HIST_LEN:]
        hist_cats = cats[:train_end_idx][-config.MAX_HIST_LEN:]
        hist_times = times[:train_end_idx][-config.MAX_HIST_LEN:]
        
        long_term_history[user_id] = {
            'pois': hist_pois,
            'cats': hist_cats,
            'times': hist_times
        }
        
        # 2. 生成滑动窗口样本 (短期序列)
        # 我们对整个轨迹进行滑动窗口，根据 target 的位置划分集合
        for i in range(1, num_checkins):
            # 短期序列窗口
            start_idx = max(0, i - config.SESSION_LEN)
            
            sample = {
                'user': user_id,
                'short_pois': pois[start_idx:i],
                'short_cats': cats[start_idx:i],
                'short_times': times[start_idx:i],
                'target': pois[i]
            }
            
            if i < train_end_idx:
                train_data.append(sample)
            elif i < valid_end_idx:
                valid_data.append(sample)
            else:
                test_data.append(sample)
                
    return train_data, valid_data, test_data, long_term_history, n_users, n_pois, n_cats

# --- 3. Dataset & Collate ---
class LSPLDataset(Dataset):
    def __init__(self, data, long_term_history):
        self.data = data
        self.long_term_history = long_term_history # 引用大字典
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        user_id = item['user']
        
        # 获取该用户的长期历史
        hist = self.long_term_history[user_id]
        
        return {
            'user': torch.tensor(user_id, dtype=torch.long),
            
            # Short-term data
            'short_pois': torch.tensor(item['short_pois'], dtype=torch.long),
            'short_cats': torch.tensor(item['short_cats'], dtype=torch.long),
            'short_times': torch.tensor(item['short_times'], dtype=torch.long),
            
            # Long-term data (来自于预处理的训练集部分)
            'hist_pois': torch.tensor(hist['pois'], dtype=torch.long),
            'hist_cats': torch.tensor(hist['cats'], dtype=torch.long),
            'hist_times': torch.tensor(hist['times'], dtype=torch.long),
            
            'target': torch.tensor(item['target'], dtype=torch.long)
        }

def collate_fn(batch):
    # 处理变长序列的 padding
    user = torch.stack([item['user'] for item in batch])
    target = torch.stack([item['target'] for item in batch])
    
    # Short-term padding
    short_pois = pad_sequence([item['short_pois'] for item in batch], batch_first=True, padding_value=0)
    short_cats = pad_sequence([item['short_cats'] for item in batch], batch_first=True, padding_value=0)
    short_times = pad_sequence([item['short_times'] for item in batch], batch_first=True, padding_value=0)
    
    # Long-term padding
    hist_pois = pad_sequence([item['hist_pois'] for item in batch], batch_first=True, padding_value=0)
    hist_cats = pad_sequence([item['hist_cats'] for item in batch], batch_first=True, padding_value=0)
    hist_times = pad_sequence([item['hist_times'] for item in batch], batch_first=True, padding_value=0)
    
    # Mask for long-term history (used in Attention)
    hist_mask = (hist_pois != 0)
    
    return {
        'user': user,
        'target': target,
        'short': {'pois': short_pois, 'cats': short_cats, 'times': short_times},
        'long': {'pois': hist_pois, 'cats': hist_cats, 'times': hist_times, 'mask': hist_mask}
    }

# --- 4. LSPL 模型定义 ---
class LSPL(nn.Module):
    def __init__(self, n_users, n_pois, n_cats, config):
        super(LSPL, self).__init__()
        self.config = config
        
        # 1. Embeddings
        self.user_emb = nn.Embedding(n_users, config.EMBED_SIZE)
        self.poi_emb = nn.Embedding(n_pois, config.EMBED_SIZE, padding_idx=0)
        self.cat_emb = nn.Embedding(n_cats, config.EMBED_SIZE, padding_idx=0)
        self.time_emb = nn.Embedding(168, config.TIME_EMBED_SIZE) # 0-167
        
        # 2. Long-term Module (Attention)
        # 组合特征变换层: W_l, W_c, W_t, b (Eq. 1)
        # Input dim = Embed + Embed + Time_Embed
        self.long_fc = nn.Linear(config.EMBED_SIZE * 2 + config.TIME_EMBED_SIZE, config.EMBED_SIZE)
        self.tanh = nn.Tanh()
        
        # Attention output mapping -> Logits
        self.long_output_layer = nn.Linear(config.EMBED_SIZE, n_pois)

        # 3. Short-term Module (Dual LSTM)
        # Location LSTM Inputs: User + POI + Time
        loc_input_dim = config.EMBED_SIZE * 2 + config.TIME_EMBED_SIZE
        self.lstm_loc = nn.LSTM(loc_input_dim, config.HIDDEN_SIZE, batch_first=True)
        self.short_loc_output = nn.Linear(config.HIDDEN_SIZE, n_pois)
        
        # Category LSTM Inputs: User + Cat + Time
        cat_input_dim = config.EMBED_SIZE * 2 + config.TIME_EMBED_SIZE
        self.lstm_cat = nn.LSTM(cat_input_dim, config.HIDDEN_SIZE, batch_first=True)
        self.short_cat_output = nn.Linear(config.HIDDEN_SIZE, n_pois)
        
        # 4. Fusion Weights (Eq. 4)
        # Learnable scalars
        self.alpha = nn.Parameter(torch.tensor(0.33))
        self.beta = nn.Parameter(torch.tensor(0.33))
        self.gamma = nn.Parameter(torch.tensor(0.33))
        
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, batch):
        user = batch['user']
        short_data = batch['short']
        long_data = batch['long']
        
        batch_size = user.size(0)
        
        # Common Embeddings
        u_e = self.user_emb(user) # (B, D)
        
        # ==========================
        # 1. Long-term Preference
        # ==========================
        # Hist Embeddings
        h_p_e = self.poi_emb(long_data['pois']) # (B, L, D)
        h_c_e = self.cat_emb(long_data['cats']) # (B, L, D)
        h_t_e = self.time_emb(long_data['times']) # (B, L, D_t)
        
        # Composite feature h_i (Eq. 1)
        # Concatenate and Linear transform
        h_concat = torch.cat([h_p_e, h_c_e, h_t_e], dim=-1) # (B, L, 2D+Dt)
        h_i = self.tanh(self.long_fc(h_concat)) # (B, L, D)
        
        # Attention (Eq. 2)
        # Query: User embedding u_e (B, D) -> (B, 1, D)
        # Key: h_i (B, L, D)
        scores = torch.bmm(h_i, u_e.unsqueeze(2)).squeeze(2) # (B, L)
        
        # Mask padding
        scores = scores.masked_fill(long_data['mask'] == 0, -1e9)
        attn_weights = F.softmax(scores, dim=1).unsqueeze(1) # (B, 1, L)
        
        # Weighted Sum (Eq. 3 u_long)
        # 注意: 论文 Eq 3 说是 weighted sum of concatenated embeddings [vl; vc; vt]
        # 但 h_i 已经是 transformed feature。为了简单和一致性，通常是对 h_i 加权。
        # 严格按照论文 Eq 3: Sum(a_i * [vl; vc; vt])
        # 我们用 h_concat 代表原始 concat 特征
        u_long = torch.bmm(attn_weights, h_concat).squeeze(1) # (B, 2D+Dt)
        # 但 u_long 维度变了，无法直接放入 output layer (Linear(D -> N))
        # 论文提到: "u_long is fed into a fully connected layer"
        # 这里的 long_output_layer 是 (D -> N)，所以我们最好对 h_i (dim=D) 进行加权
        u_long = torch.bmm(attn_weights, h_i).squeeze(1) # (B, D)
        
        P_long = self.long_output_layer(u_long) # (B, N_pois)
        
        # ==========================
        # 2. Short-term Preference
        # ==========================
        # Short Embeddings
        s_p_e = self.poi_emb(short_data['pois'])
        s_c_e = self.cat_emb(short_data['cats'])
        s_t_e = self.time_emb(short_data['times'])
        
        # User embedding repeated for sequence
        u_e_seq = u_e.unsqueeze(1).expand(-1, short_data['pois'].size(1), -1)
        
        # --- Location-based LSTM ---
        # Input: User + POI + Time
        loc_input = torch.cat([u_e_seq, s_p_e, s_t_e], dim=-1)
        _, (h_loc, _) = self.lstm_loc(loc_input)
        # h_loc: (1, B, Hidden)
        P_loc = self.short_loc_output(h_loc.squeeze(0)) # (B, N_pois)
        
        # --- Category-based LSTM ---
        # Input: User + Cat + Time
        cat_input = torch.cat([u_e_seq, s_c_e, s_t_e], dim=-1)
        _, (h_cat, _) = self.lstm_cat(cat_input)
        P_cat = self.short_cat_output(h_cat.squeeze(0)) # (B, N_pois)
        
        # ==========================
        # 3. Fusion (Eq. 4)
        # ==========================
        # Weighted Sum of Logits
        final_logits = self.alpha * P_long + self.beta * P_loc + self.gamma * P_cat
        
        return final_logits

# --- 5. 训练和评估函数 (标准复用) ---
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        # Move data to device
        batch['user'] = batch['user'].to(device)
        batch['target'] = batch['target'].to(device)
        for k in batch['short']: batch['short'][k] = batch['short'][k].to(device)
        for k in batch['long']: batch['long'][k] = batch['long'][k].to(device)

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
            batch['user'] = batch['user'].to(device)
            batch['target'] = batch['target'].to(device)
            for k in batch['short']: batch['short'][k] = batch['short'][k].to(device)
            for k in batch['long']: batch['long'][k] = batch['long'][k].to(device)
                
            logits = model(batch)
            target = batch['target']

            max_k = max(top_k)
            _, top_indices = torch.topk(logits, max_k, dim=1)

            expanded_target = target.view(-1, 1).expand_as(top_indices)
            hits_matrix = (top_indices == expanded_target)

            for k in top_k:
                metrics[f'acc@{k}'] += hits_matrix[:, :k].any(dim=1).sum().item()
            
            match_indices = torch.nonzero(hits_matrix, as_tuple=True)
            reciprocal_ranks = torch.zeros(target.size(0)).to(device)
            ranks = match_indices[1].float() + 1.0
            reciprocal_ranks.scatter_(0, match_indices[0], 1.0 / ranks)
            metrics['mrr'] += reciprocal_ranks.sum().item() # type: ignore
                
            total_samples += target.size(0)

    for k in top_k: metrics[f'acc@{k}'] /= total_samples # type: ignore
    metrics['mrr'] /= total_samples # type: ignore
    return metrics

# --- 6. 主程序 ---
if __name__ == '__main__':
    config = Config()
    print(f"Using device: {config.DEVICE}")

    # 1. PREPROCESS DATA
    train_data, valid_data, test_data, long_term_history, n_users, n_pois, n_cats = preprocess_data_for_lspl(config)
    
    print(f"\nNum Users: {n_users}, Num POIs: {n_pois}, Num Cats: {n_cats}")
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    # 2. DATALOADERS
    # Pass long_term_history to dataset
    train_dataset = LSPLDataset(train_data, long_term_history)
    valid_dataset = LSPLDataset(valid_data, long_term_history)
    test_dataset = LSPLDataset(test_data, long_term_history)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. MODEL
    model = LSPL(n_users, n_pois, n_cats, config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 4. TRAIN LOOP
    best_val_mrr = 0.0
    best_model_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")
        avg_loss = train_model(model, train_loader, optimizer, criterion, config.DEVICE)
        print(f"Avg Train Loss: {avg_loss:.4f}")
        
        # Check learned weights
        print(f"Weights: alpha={model.alpha.item():.2f}, beta={model.beta.item():.2f}, gamma={model.gamma.item():.2f}")
        
        val_metrics = evaluate(model, valid_loader, config.TOP_K, config.DEVICE)
        print("--- Validation Metrics ---")
        for k, v in val_metrics.items(): print(f"{k}: {v:.4f}")
        
        if val_metrics['mrr'] > best_val_mrr:
            best_val_mrr = val_metrics['mrr']
            best_model_state = model.state_dict()
            patience_counter = 0
            print("Validation MRR improved! Saving model.")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")
            
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 5. TEST
    print("\n--- Training Finished ---")
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print("\n--- Final Test Metrics ---")
    test_metrics = evaluate(model, test_loader, config.TOP_K, config.DEVICE)
    for k, v in test_metrics.items(): print(f"{k}: {v:.4f}")