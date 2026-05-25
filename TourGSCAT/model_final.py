import os
import sys
import time
import math
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. 全局配置与超参数
# ==============================================================================
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

BASE_DIR = os.getcwd()
sys.path.append(os.path.join(BASE_DIR, "MyModel"))
sys.path.append(BASE_DIR)

CONFIG = {
    'data_path': os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_nyc_cleaned.csv'),
    'stage1_dir': os.path.join(BASE_DIR, 'MyModel/Stage2_Features/NYC'),
    'seed': 42,
    'device': DEVICE,
    
    # 行程提取参数
    'max_time_gap_hours': 8,   
    'min_trip_len': 3,         
    'max_trip_len': 10,        
    'avg_speed_kmh': 15.0,     
    
    # 模型参数
    'lstm_hidden_dim': 256,    
    'dropout': 0.2,
    
    # 训练参数
    'batch_size': 32,
    'epochs': 200,             
    'patience': 10,            
    'lr': 1e-3,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==============================================================================
# 1. 数据预处理
# ==============================================================================
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians,[lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def load_and_extract_itineraries(data_path, mapping_path):
    print(">>> 1. 加载 Stage 1 的 ID 映射与原始数据...")
    with open(mapping_path, 'rb') as f:
        maps = pickle.load(f)
    user_map, venue_map, cat_map = maps['user_map'], maps['venue_map'], maps['category_map']
    venue_pad_idx, cat_pad_idx = maps['venue_pad_idx'], maps['cat_pad_idx']
    
    num_users = len(user_map)
    num_venues = len(venue_map)
    num_venues_with_pad = venue_pad_idx + 1 
    
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    df['geo_id'] = df['geo_id'].astype(str)
    df['venue_category_id'] = df['venue_category_id'].fillna('UNK_CAT').astype(str)
    df = df.sort_values(by=['user_id', 'time'])
    
    poi_info = {}
    for _, row in df.drop_duplicates(subset=['geo_id']).iterrows():
        if row['geo_id'] in venue_map:
            poi_info[venue_map[row['geo_id']]] = {
                'lat': row['latitude'], 'lon': row['longitude'],
                'cat': cat_map.get(row['venue_category_id'], cat_pad_idx)
            }
            
    itineraries =[]
    current_trip =[]
    cat_dwell_times = defaultdict(list)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="切分一日游轨迹"):
        if row['user_id'] not in user_map or row['geo_id'] not in venue_map: continue
            
        u_idx, v_idx = user_map[row['user_id']], venue_map[row['geo_id']]
        c_idx = cat_map.get(row['venue_category_id'], cat_pad_idx)
        curr_time = row['time']
        
        if not current_trip or current_trip[-1]['user'] != u_idx:
            if current_trip and CONFIG['min_trip_len'] <= len(current_trip) <= CONFIG['max_trip_len']:
                itineraries.append({'user_id': current_trip[0]['user'], 'trip': current_trip.copy()})
            current_trip =[{'user': u_idx, 'poi': v_idx, 'cat': c_idx, 'time': curr_time}]
            continue
            
        last_time = current_trip[-1]['time']
        diff_hours = (curr_time - last_time).total_seconds() / 3600.0
        
        if 10 < diff_hours * 60.0 < 240:
            cat_dwell_times[current_trip[-1]['cat']].append(diff_hours * 60.0 * 0.5)
            
        if diff_hours > CONFIG['max_time_gap_hours']:
            if CONFIG['min_trip_len'] <= len(current_trip) <= CONFIG['max_trip_len']:
                itineraries.append({'user_id': u_idx, 'trip': current_trip.copy()})
            current_trip =[{'user': u_idx, 'poi': v_idx, 'cat': c_idx, 'time': curr_time}]
        else:
            current_trip.append({'user': u_idx, 'poi': v_idx, 'cat': c_idx, 'time': curr_time})
            
    if current_trip and CONFIG['min_trip_len'] <= len(current_trip) <= CONFIG['max_trip_len']:
        itineraries.append({'user_id': current_trip[0]['user'], 'trip': current_trip.copy()})
            
    avg_cat_dwell = {c: np.mean(times) if times else 45.0 for c, times in cat_dwell_times.items()}
    return itineraries, num_users, num_venues_with_pad, venue_pad_idx, poi_info, avg_cat_dwell

def build_global_spatiotemporal_tensors(poi_info, avg_cat_dwell, num_venues_with_pad, pad_idx, itineraries):
    print(">>> 2. 构建全局时空约束张量与归一化物理先验矩阵...")
    transit_matrix = torch.full((num_venues_with_pad, num_venues_with_pad), fill_value=30.0)
    dwell_tensor = torch.full((num_venues_with_pad,), fill_value=45.0)
    transition_matrix = torch.zeros((num_venues_with_pad, num_venues_with_pad))
    pop_tensor = torch.zeros((num_venues_with_pad,))
    
    for item in itineraries:
        seq = [step['poi'] for step in item['trip']]
        for i in range(len(seq)):
            pop_tensor[seq[i]] += 1
            if i < len(seq) - 1:
                transition_matrix[seq[i], seq[i+1]] += 1
                transition_matrix[seq[i+1], seq[i]] += 1 
                
    pop_tensor = torch.log1p(pop_tensor) 
    pop_tensor = pop_tensor / (pop_tensor.max() + 1e-9) 
    row_sum = transition_matrix.sum(dim=1, keepdim=True)
    transition_matrix = transition_matrix / (row_sum + 1e-9)
    
    for v_idx, info in poi_info.items():
        dwell_tensor[v_idx] = avg_cat_dwell.get(info['cat'], 45.0)
        
    poi_indices = list(poi_info.keys())
    for i in tqdm(range(len(poi_indices)), desc="计算交通矩阵"):
        v1 = poi_indices[i]
        for j in range(i+1, len(poi_indices)):
            v2 = poi_indices[j]
            dist_km = haversine(poi_info[v1]['lon'], poi_info[v1]['lat'], poi_info[v2]['lon'], poi_info[v2]['lat'])
            mins = (dist_km / CONFIG['avg_speed_kmh']) * 60.0
            transit_matrix[v1, v2] = mins
            transit_matrix[v2, v1] = mins
            
    transit_matrix.fill_diagonal_(0.0)
    transit_matrix[pad_idx, :] = 0.0; transit_matrix[:, pad_idx] = 0.0; dwell_tensor[pad_idx] = 0.0
    norm_transit = transit_matrix / (transit_matrix.max() + 1e-9)
    
    return transit_matrix.float(), dwell_tensor.float(), transition_matrix.float(), pop_tensor.float(), norm_transit.float()

# ==============================================================================
# 2. Dataset 与 评估指标
# ==============================================================================
class ItineraryDataset(Dataset):
    def __init__(self, itineraries, venue_pad_idx):
        self.data =[]
        for item in itineraries:
            trip = item['trip']
            seq = [step['poi'] for step in trip]
            total_mins = (trip[-1]['time'] - trip[0]['time']).total_seconds() / 60.0
            total_mins = max(30.0, min(total_mins, 720.0)) 
            
            # 提取起点的绝对时间 (分钟数)
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
    user_ids = torch.tensor([x['user_id'] for x in batch], dtype=torch.long)
    start_pois = torch.tensor([x['start_poi'] for x in batch], dtype=torch.long)
    start_times = torch.tensor([x['start_time_min'] for x in batch], dtype=torch.float)
    time_budgets = torch.tensor([x['time_budget'] for x in batch], dtype=torch.float)
    
    max_len = max(x['seq_len'] for x in batch)
    target_seqs = [x['target_seq'] +[pad_idx] * (max_len - x['seq_len']) for x in batch]
    target_seqs = torch.tensor(target_seqs, dtype=torch.long)
    
    return user_ids, start_pois, start_times, time_budgets, target_seqs

def calculate_all_metrics(gt_list, pred_list, pred_times, target_budgets, poi_details, K_list=[3, 5, 10]):
    metrics = {}
    for k in K_list:
        precisions, recalls, f1s = [], [],[]
        for pred, truth in zip(pred_list, gt_list):
            pred_k = set(pred[:k])
            truth_set = set(truth)
            if not truth_set: continue
            hits = len(pred_k & truth_set)
            prec = hits / len(pred_k) if len(pred_k) > 0 else 0.0
            rec = hits / len(truth_set)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            precisions.append(prec); recalls.append(rec); f1s.append(f1)
        metrics[f'P@{k}'] = np.mean(precisions)
        metrics[f'R@{k}'] = np.mean(recalls)
        metrics[f'F1@{k}'] = np.mean(f1s)
        
    pairs_f1s, div_list, ttrs = [], [],[]
    for pred, truth, pt, budget in zip(pred_list, gt_list, pred_times, target_budgets):
        p_pairs = set((pred[i], pred[j]) for i in range(len(pred)) for j in range(i+1, len(pred)))
        t_pairs = set((truth[i], truth[j]) for i in range(len(truth)) for j in range(i+1, len(truth)))
        if t_pairs and p_pairs:
            hits = len(p_pairs & t_pairs)
            p = hits / len(p_pairs); r = hits / len(t_pairs)
            pairs_f1s.append(2*p*r/(p+r) if (p+r)>0 else 0.0)
        else:
            pairs_f1s.append(0.0)
            
        cats = set([poi_details[pid]['cat'] for pid in pred if pid in poi_details])
        if len(pred) > 0: div_list.append(len(cats) / len(pred))
        if budget > 0: ttrs.append(max(0.0, 1.0 - abs(pt - budget) / budget))
            
    metrics['Pairs-F1'] = np.mean(pairs_f1s) if pairs_f1s else 0.0
    metrics['Diversity'] = np.mean(div_list) if div_list else 0.0
    metrics['TTR'] = np.mean(ttrs) if ttrs else 0.0
    return metrics

# ==============================================================================
# 3. 核心模型 TourGSCAT (保留最强的 Time-Encoding + Adaptive-STC)
# ==============================================================================
class TourGSCAT(nn.Module):
    def __init__(self, num_users, num_venues_with_pad, pretrained_user_emb, pretrained_poi_emb, 
                 transit_matrix, dwell_tensor, transition_matrix, pop_tensor, norm_transit, 
                 pad_idx, lstm_hidden, dropout):
        super(TourGSCAT, self).__init__()
        self.pad_idx = pad_idx
        
        user_embed_dim = pretrained_user_emb.size(1) if pretrained_user_emb is not None else 128
        poi_embed_dim = pretrained_poi_emb.size(1) if pretrained_poi_emb is not None else 128
        
        self.user_emb = nn.Embedding(num_users, user_embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        if pretrained_user_emb is not None:
            if pretrained_user_emb.size(0) == num_users:
                self.user_emb.weight.data.copy_(pretrained_user_emb)
                
        self.poi_emb = nn.Embedding(num_venues_with_pad, poi_embed_dim, padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.poi_emb.weight)
        self.poi_emb.weight.data[pad_idx].zero_()
        if pretrained_poi_emb is not None:
            if pretrained_poi_emb.size(0) == num_venues_with_pad:
                self.poi_emb.weight.data.copy_(pretrained_poi_emb)
            elif pretrained_poi_emb.size(0) == num_venues_with_pad - 1:
                self.poi_emb.weight.data[:-1].copy_(pretrained_poi_emb)
                
        self.user_emb.weight.requires_grad = True
        self.poi_emb.weight.requires_grad = True
        
        # 【稳定且有效的特征】：绝对时间语义感知 (Time-of-Day Encoding)
        time_emb_dim = poi_embed_dim // 4
        self.time_emb = nn.Embedding(48, time_emb_dim) # 每天分48个半小时的bin
        
        budget_dim = poi_embed_dim // 2
        self.budget_enc = nn.Sequential(nn.Linear(1, budget_dim), nn.ReLU(), nn.Linear(budget_dim, budget_dim))
        
        # LSTM 输入: User + POI + Budget + Time
        lstm_input_dim = user_embed_dim + poi_embed_dim + budget_dim + time_emb_dim
        self.state_proj = nn.Linear(lstm_input_dim, lstm_hidden)
        self.lstm_cell = nn.LSTMCell(input_size=lstm_hidden, hidden_size=lstm_hidden)
        
        self.W_q = nn.Linear(lstm_hidden, poi_embed_dim)
        self.W_k = nn.Linear(poi_embed_dim, poi_embed_dim)
        
        # 自适应上下文门控
        self.adaptive_gate = nn.Sequential(
            nn.Linear(lstm_hidden + user_embed_dim, poi_embed_dim // 2), nn.ReLU(), nn.Linear(poi_embed_dim // 2, 1), nn.Sigmoid()
        )
        
        self.dist_alpha = nn.Parameter(torch.tensor([1.0]))  
        self.trans_beta = nn.Parameter(torch.tensor([1.0]))   
        self.pop_gamma = nn.Parameter(torch.tensor([1.0]))    
        
        self.register_buffer('transit_matrix', transit_matrix)  
        self.register_buffer('dwell_tensor', dwell_tensor)
        self.register_buffer('norm_transit', norm_transit)      
        self.register_buffer('transition_matrix', transition_matrix)
        self.register_buffer('pop_tensor', pop_tensor)
        
        self.dropout = nn.Dropout(dropout)

    def _compute_mask(self, current_pois, budget_remains, visited_mask):
        B = current_pois.size(0)
        num_v = self.dwell_tensor.size(0)
        transit_t = self.transit_matrix[current_pois] 
        dwell_t = self.dwell_tensor.unsqueeze(0).expand(B, -1) 
        req_time = transit_t + dwell_t
        time_mask = req_time > budget_remains.expand(-1, num_v)
        final_mask = time_mask | visited_mask
        final_mask[:, self.pad_idx] = False 
        return final_mask, req_time

    def _get_fused_scores(self, q, current_pois):
        B = q.size(0)
        semantic_scores = torch.matmul(q, self.W_k(self.poi_emb.weight).transpose(0, 1)) 
        physic_scores = - F.relu(self.dist_alpha) * self.norm_transit[current_pois] \
                        + F.relu(self.trans_beta) * self.transition_matrix[current_pois] \
                        + F.relu(self.pop_gamma) * self.pop_tensor.unsqueeze(0)
        return semantic_scores, physic_scores

    def forward(self, user_ids, start_pois, start_times, time_budgets, target_seqs, tf_ratio=0.5):
        B = user_ids.size(0)
        device = user_ids.device
        num_v = self.poi_emb.num_embeddings
        max_len = target_seqs.size(1)
        
        u_e = self.user_emb(user_ids)
        h_t = torch.zeros(B, self.lstm_cell.hidden_size, device=device)
        c_t = torch.zeros(B, self.lstm_cell.hidden_size, device=device)
        
        curr_pois = start_pois
        budgets = time_budgets.unsqueeze(1).float()
        curr_time_mins = start_times.unsqueeze(1).float()
        
        visited = torch.zeros(B, num_v, dtype=torch.bool, device=device)
        visited.scatter_(1, curr_pois.unsqueeze(1), True)
        
        logits_list =[]
        
        for t in range(max_len):
            p_e = self.poi_emb(curr_pois)
            b_e = self.budget_enc(budgets)
            
            # 加入当前时间的 Embedding
            time_bins = ((curr_time_mins % 1440) / 30).long()
            t_e = self.time_emb(time_bins.squeeze(1))
            
            lstm_in = self.dropout(F.relu(self.state_proj(torch.cat([u_e, p_e, b_e, t_e], dim=-1))))
            h_t, c_t = self.lstm_cell(lstm_in, (h_t, c_t))
            
            q = self.W_q(h_t)
            semantic_scores, physic_scores = self._get_fused_scores(q, curr_pois)
            gate = self.adaptive_gate(torch.cat([h_t, u_e], dim=-1)) 
            scores = gate * semantic_scores + (1 - gate) * physic_scores
            
            mask, req_time = self._compute_mask(curr_pois, budgets, visited)
            gt_next = target_seqs[:, t].unsqueeze(1)
            mask.scatter_(1, gt_next, False) 
            
            scores.masked_fill_(mask, -1e9)
            logits_list.append(scores.unsqueeze(1))
            
            if random.random() < tf_ratio:
                next_pois = target_seqs[:, t]
            else:
                next_pois = scores.argmax(dim=1)
                
            actual_time = req_time.gather(1, next_pois.unsqueeze(1))
            budgets = budgets - actual_time
            curr_time_mins = curr_time_mins + actual_time
            visited.scatter_(1, next_pois.unsqueeze(1), True)
            curr_pois = next_pois
            
        return torch.cat(logits_list, dim=1)

    def generate(self, user_ids, start_pois, start_times, time_budgets, max_len=10, temperature=0.8):
        self.eval()
        with torch.no_grad():
            B = user_ids.size(0)
            device = user_ids.device
            num_v = self.poi_emb.num_embeddings
            
            u_e = self.user_emb(user_ids)
            h_t = torch.zeros(B, self.lstm_cell.hidden_size, device=device)
            c_t = torch.zeros(B, self.lstm_cell.hidden_size, device=device)
            
            curr_pois = start_pois
            budgets = time_budgets.unsqueeze(1).float()
            curr_time_mins = start_times.unsqueeze(1).float()
            initial_budgets = budgets.clone()
            
            visited = torch.zeros(B, num_v, dtype=torch.bool, device=device)
            visited.scatter_(1, curr_pois.unsqueeze(1), True)
            
            gen_seqs = [[] for _ in range(B)]
            
            for t in range(max_len):
                p_e = self.poi_emb(curr_pois)
                b_e = self.budget_enc(budgets)
                time_bins = ((curr_time_mins % 1440) / 30).long()
                t_e = self.time_emb(time_bins.squeeze(1))
                
                lstm_in = F.relu(self.state_proj(torch.cat([u_e, p_e, b_e, t_e], dim=-1)))
                h_t, c_t = self.lstm_cell(lstm_in, (h_t, c_t))
                
                q = self.W_q(h_t)
                semantic_scores, physic_scores = self._get_fused_scores(q, curr_pois)
                gate = self.adaptive_gate(torch.cat([h_t, u_e], dim=-1))
                scores = gate * semantic_scores + (1 - gate) * physic_scores
                
                mask, req_time = self._compute_mask(curr_pois, budgets, visited)
                scores.masked_fill_(mask, -1e9)
                
                scaled_scores = scores / temperature
                next_pois = scaled_scores.argmax(dim=1)
                
                actual_time = req_time.gather(1, next_pois.unsqueeze(1))
                budgets = budgets - actual_time
                curr_time_mins = curr_time_mins + actual_time
                visited.scatter_(1, next_pois.unsqueeze(1), True)
                curr_pois = next_pois
                
                for i in range(B):
                    idx = next_pois[i].item()
                    if idx != self.pad_idx: gen_seqs[i].append(idx)
                        
            gen_times = (initial_budgets - budgets).squeeze(1).cpu().tolist()
        return gen_seqs, gen_times

# ==============================================================================
# 4. 训练与评估循环 (稳定的 MLE + 调度策略)
# ==============================================================================
def train_eval_loop(model, train_loader, val_loader, test_loader, num_venues_with_pad, pad_idx, poi_info):
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    device = CONFIG['device']
    
    best_f1 = 0.0
    epochs_no_improve = 0  
    
    print("\n" + "="*40)
    print("🚀 开始稳定版训练...")
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        total_loss = 0
        
        # 稳定的计划采样，下限控制在 0.3，不脱离太远
        current_tf_ratio = max(0.3, 0.9 - (epoch / CONFIG['epochs']) * 0.6)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for u, s, st, b, targets in pbar:
            u, s, st, b, targets = u.to(device), s.to(device), st.to(device), b.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(u, s, st, b, targets, tf_ratio=current_tf_ratio)
            loss = criterion(logits.view(-1, num_venues_with_pad), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | LR: {curr_lr:.2e} | TF: {current_tf_ratio:.2f} | Loss: {total_loss/len(train_loader):.4f}")
        
        metrics = evaluate(model, val_loader, device, pad_idx, poi_info)
        current_f1 = metrics['F1@5']
        print(f"[Validation] F1@5: {current_f1:.4f}, Pairs-F1: {metrics['Pairs-F1']:.4f}, TTR: {metrics['TTR']:.4f}")
        
        scheduler.step(current_f1)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            epochs_no_improve = 0  
            torch.save(model.state_dict(), 'tour_gscat_stable_best.pth')
            print(f"  [*] Best Model Saved! (F1@5 improved to {best_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  [!] No improvement for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= CONFIG['patience']:
            print(f"触发早停机制，停止训练。")
            break
            
    # --------- 最终评估 ---------
    print("\n🔥 最终测试集评估 (Final Testing)")
    model.load_state_dict(torch.load('tour_gscat_stable_best.pth', weights_only=True))
    test_metrics = evaluate(model, test_loader, device, pad_idx, poi_info)
    
    print("-" * 50)
    print(f"{'K':<6}| {'Recall':<9}| {'Prec':<9}| {'F1':<8}")
    print("-" * 50)
    for k in [3, 5, 10]:
        r = test_metrics.get(f'R@{k}', 0.0)
        p = test_metrics.get(f'P@{k}', 0.0)
        f1 = test_metrics.get(f'F1@{k}', 0.0)
        print(f"{k:<6}| {r:<9.4f}| {p:<9.4f}| {f1:<8.4f}")
    print(f"Pairs-F1  : {test_metrics.get('Pairs-F1', 0.0):.4f}")
    print(f"Diversity : {test_metrics.get('Diversity', 0.0):.4f}")
    print(f"TTR       : {test_metrics.get('TTR', 0.0):.4f}")

def evaluate(model, dataloader, device, pad_idx, poi_info):
    all_preds, all_truths, all_times, all_budgets = [], [], [],[]
    for u, s, st, b, targets in dataloader:
        u, s, st, b = u.to(device), s.to(device), st.to(device), b.to(device)
        
        gen_seqs, gen_times = model.generate(u, s, st, b, max_len=CONFIG['max_trip_len'], temperature=0.8)
        
        start_pois = s.cpu().numpy().tolist()
        targets_list = targets.cpu().numpy().tolist()
        
        for i in range(len(start_pois)):
            clean_truth = [start_pois[i]] +[x for x in targets_list[i] if x != pad_idx]
            full_pred =[start_pois[i]] + gen_seqs[i]
            all_truths.append(clean_truth)
            all_preds.append(full_pred)
            
        all_times.extend(gen_times)
        all_budgets.extend(b.cpu().tolist())
        
    return calculate_all_metrics(all_truths, all_preds, all_times, all_budgets, poi_info)

# ==============================================================================
# 5. 主函数 Main
# ==============================================================================
def main():
    set_seed(CONFIG['seed'])
    
    mapping_path = os.path.join(CONFIG['stage1_dir'], 'id_mappings.pkl')
    user_emb_path = os.path.join(CONFIG['stage1_dir'], 'pretrained_user_embs.pt')
    poi_emb_path = os.path.join(CONFIG['stage1_dir'], 'pretrained_poi_embs.pt')
    
    itineraries, num_u, num_v_pad, pad_idx, poi_info, avg_dwell = load_and_extract_itineraries(CONFIG['data_path'], mapping_path)
    
    print(">>> 加载 GSCAT 预训练表征...")
    pretrained_user_emb = torch.load(user_emb_path, map_location=DEVICE)
    pretrained_poi_emb = torch.load(poi_emb_path, map_location=DEVICE)
    
    trans_mat, dwell_ten, trans_prob_mat, pop_ten, norm_trans = build_global_spatiotemporal_tensors(poi_info, avg_dwell, num_v_pad, pad_idx, itineraries)
    
    random.shuffle(itineraries)
    t_end = int(len(itineraries) * 0.8)
    v_end = t_end + int(len(itineraries) * 0.1)
    
    train_dl = DataLoader(ItineraryDataset(itineraries[:t_end], pad_idx), batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=lambda x: collate_fn(x, pad_idx))
    val_dl = DataLoader(ItineraryDataset(itineraries[t_end:v_end], pad_idx), batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=lambda x: collate_fn(x, pad_idx))
    test_dl = DataLoader(ItineraryDataset(itineraries[v_end:], pad_idx), batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=lambda x: collate_fn(x, pad_idx))
    
    print("\n>>> 3. 初始化 TourGSCAT(Stable) 模型...")
    model = TourGSCAT(
        num_users=num_u, num_venues_with_pad=num_v_pad, 
        pretrained_user_emb=pretrained_user_emb, pretrained_poi_emb=pretrained_poi_emb,    
        transit_matrix=trans_mat, dwell_tensor=dwell_ten, transition_matrix=trans_prob_mat, 
        pop_tensor=pop_ten, norm_transit=norm_trans, pad_idx=pad_idx,
        lstm_hidden=CONFIG['lstm_hidden_dim'], dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    print("\n>>> 4. 启动训练...")
    train_eval_loop(model, train_dl, val_dl, test_dl, num_v_pad, pad_idx, poi_info)

if __name__ == "__main__":
    main()