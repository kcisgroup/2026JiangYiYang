import os
import time
import math
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置与参数
# ==========================================
class Config:
    def __init__(self):
        self.BASE_DIR = os.getcwd()
        self.DATA_PATH = os.path.join(self.BASE_DIR, 'MyModel/dataset_process/foursquare_tky_cleaned.csv')
        
        # 列名映射
        self.COL_TIME = 'time'
        self.COL_USER = 'user_id'
        self.COL_POI = 'geo_id'
        self.COL_CAT = 'venue_category_name'
        self.COL_LON = 'longitude'
        self.COL_LAT = 'latitude'
        
        # DLIR 模型参数
        self.EMBED_DIM = 128
        self.HIDDEN_DIM = 256
        self.NUM_HEADS = 4
        self.NUM_LAYERS = 2
        self.DROPOUT = 0.3
        self.LR = 0.001
        self.EPOCHS = 5
        self.BATCH_SIZE = 64
        self.MAX_SEQ_LEN = 20
        
        # 行程规划推断参数
        self.TOP_K_CANDIDATES = 20
        self.AVG_SPEED = 0.83     # 预估速度 km/min (约50km/h)
        self.VISIT_TIME = 45      # 默认游玩时间 45 min
        
        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

config = Config()

# ==========================================
# 2. 距离计算工具 (Haversine)
# ==========================================
def haversine(lat1, lon1, lat2, lon2):
    """计算两个经纬度之间的球面距离 (单位: km)"""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ==========================================
# 3. 核心评估指标逻辑 (还原包含起点，全新 TTR)
# ==========================================
def calculate_all_metrics(gt_list, pred_list, pred_times, target_budgets, poi_details, K_list=[3, 5, 10]):
    """
    计算所有的行程规划指标：Top-K, Pairs-F1, Diversity, 优化后的 TTR
    注：这里的 gt_list 和 pred_list 均包含 Start POI (还原 0.30 的 Recall 水平)
    """
    metrics = {}
    
    # --- 1. Top-K 指标 (Recall, Precision, F1) ---
    for k in K_list:
        rec_list, prec_list, f1_list = [], [],[]
        for g, p in zip(gt_list, pred_list):
            p_cut = p[:k] # 截取前K个预测结果
            if not p_cut: continue
            
            g_set = set(g)
            p_set = set(p_cut)
            
            hit = len(g_set & p_set)
            
            # Recall = Hit / 真实长度
            r = hit / len(g_set) if g_set else 0.0
            # Precision = Hit / 预测长度
            p_val = hit / len(p_set) if p_set else 0.0
            # F1-score
            f = 2 * p_val * r / (p_val + r) if (p_val + r) > 0 else 0.0
            
            rec_list.append(r)
            prec_list.append(p_val)
            f1_list.append(f)
            
        metrics[f'R@{k}'] = np.mean(rec_list)
        metrics[f'P@{k}'] = np.mean(prec_list)
        metrics[f'F1@{k}'] = np.mean(f1_list)
        
    # --- 2. Global 指标 (Pairs-F1, Diversity, TTR) ---
    pairs_f1_list, div_list, ttr_list = [], [],[]
    for g, p, p_time, budget in zip(gt_list, pred_list, pred_times, target_budgets):
        if not p: continue
        
        # 2.1 Pairs-F1 (顺序正确性)
        g_pairs = set([(g[i], g[j]) for i in range(len(g)) for j in range(i+1, len(g))])
        p_pairs = set([(p[i], p[j]) for i in range(len(p)) for j in range(i+1, len(p))])
        
        if g_pairs and p_pairs:
            hit = len(g_pairs & p_pairs)
            pr = hit / len(g_pairs)
            pp = hit / len(p_pairs)
            pf = 2 * pp * pr / (pp + pr) if (pp + pr) > 0 else 0.0
            pairs_f1_list.append(pf)
        else:
            pairs_f1_list.append(0.0)
            
        # 2.2 Diversity (多样性：唯一类别数 / 序列长度)
        cats = set()
        for pid in p:
            if pid in poi_details:
                cats.add(poi_details[pid]['cat'])
        div_list.append(len(cats) / len(p))
        
        # 2.3 TTR (时间合理性：你提供的全新公式)
        if budget > 0:
            # TTR = 1 - (|预测总耗时 - 目标预算| / 目标预算)
            # 使用 max(0, x) 防止严重超时导致出现负数
            ttr_val = max(0.0, 1.0 - abs(p_time - budget) / budget)
            ttr_list.append(ttr_val)
            
    metrics['Pairs-F1'] = np.mean(pairs_f1_list) if pairs_f1_list else 0.0
    metrics['Diversity'] = np.mean(div_list) if div_list else 0.0
    metrics['TTR'] = np.mean(ttr_list) if ttr_list else 0.0
    
    return metrics

# ==========================================
# 4. 数据处理 (加入真实 Time Budget 提取)
# ==========================================
class DataProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.poi_encoder = LabelEncoder()
        self.user_encoder = LabelEncoder()
        self.cat_encoder = LabelEncoder()
        self.poi_details = {} 
        self.num_pois = 0
        self.num_users = 0
        
    def load_data(self):
        print(f"Loading data from {self.cfg.DATA_PATH}...")
        try:
            df = pd.read_csv(self.cfg.DATA_PATH)
        except FileNotFoundError:
            print(">>> ⚠️ Data file not found! Generating mock data for baseline testing...")
            mock_data = {
                'time': pd.date_range(start='2012-04-01', periods=2000, freq='40min'), # freq='40T' 替代方案
                'user_id': np.random.randint(0, 50, 2000),
                'geo_id': np.random.randint(0, 100, 2000),
                'venue_category_name': np.random.randint(0, 20, 2000),
                'latitude': np.random.uniform(40.6, 40.8, 2000),
                'longitude': np.random.uniform(-74.0, -73.8, 2000)
            }
            df = pd.DataFrame(mock_data)
        
        df[self.cfg.COL_TIME] = pd.to_datetime(df[self.cfg.COL_TIME])
        df = df.sort_values([self.cfg.COL_USER, self.cfg.COL_TIME])
        
        # ID 编码，POI ID 整体 +1，预留 0 给 Padding
        df['user_idx'] = self.user_encoder.fit_transform(df[self.cfg.COL_USER].astype(str))
        df['poi_idx'] = self.poi_encoder.fit_transform(df[self.cfg.COL_POI].astype(str)) + 1 # type: ignore
        df['cat_idx'] = self.cat_encoder.fit_transform(df[self.cfg.COL_CAT].astype(str))
        
        # 记录 POI 的坐标、类别、热门度
        for _, row in df.iterrows():
            pid = row['poi_idx']
            if pid not in self.poi_details:
                self.poi_details[pid] = {
                    'cat': row['cat_idx'],
                    'coords': (row[self.cfg.COL_LAT], row[self.cfg.COL_LON]),
                    'popularity': 0
                }
            self.poi_details[pid]['popularity'] += 1
            
        self.num_pois = len(self.poi_encoder.classes_) + 1 
        self.num_users = len(self.user_encoder.classes_)
        return df

    def generate_sessions(self, df):
        """生成会话，并附带该序列的真实 Time Budget (单位: 分钟)"""
        sessions =[]
        user_groups = df.groupby('user_idx')
        for uid, group in user_groups:
            group = group.sort_values(self.cfg.COL_TIME)
            temp_seq = []
            temp_times =[]
            
            for _, row in group.iterrows():
                curr_time = row[self.cfg.COL_TIME]
                poi = row['poi_idx']
                
                if not temp_seq:
                    temp_seq.append(poi)
                    temp_times.append(curr_time)
                else:
                    diff_hours = (curr_time - temp_times[-1]).total_seconds() / 3600.0
                    # 如果两次签到超过8小时，切割行程
                    if diff_hours > 8: 
                        if len(temp_seq) >= 3: # 至少保留长度为3的行程
                            budget = (temp_times[-1] - temp_times[0]).total_seconds() / 60.0
                            budget = max(30.0, min(budget, 720.0)) # 限制在 30分钟 ~ 12小时
                            sessions.append((uid, temp_seq, budget))
                        temp_seq = [poi]
                        temp_times = [curr_time]
                    else:
                        temp_seq.append(poi)
                        temp_times.append(curr_time)
                        
            # 收尾最后一个序列
            if len(temp_seq) >= 3:
                budget = (temp_times[-1] - temp_times[0]).total_seconds() / 60.0
                budget = max(30.0, min(budget, 720.0))
                sessions.append((uid, temp_seq, budget))
                
        return sessions

    def build_covisiting_matrix(self, sessions):
        """基于用户行程构建 GCN 共现矩阵"""
        adj = np.zeros((self.num_pois, self.num_pois))
        for _, seq, _ in sessions: # 拆包时包含 budget
            for i in range(len(seq)-1):
                u, v = seq[i], seq[i+1]
                adj[u][v] += 1
                adj[v][u] += 1
                
        for i in range(1, self.num_pois):
            adj[i][i] = 1 # 自连接
        
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return torch.FloatTensor(d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).to(self.cfg.device)

# ==========================================
# 5. DLIR 模型定义
# ==========================================
class DatasetWrapper(Dataset):
    def __init__(self, sess, max_len):
        self.sess = sess
        self.max_len = max_len
        
    def __getitem__(self, i):
        u, seq, _ = self.sess[i] # 训练阶段不需要 budget
        if len(seq) > self.max_len + 1: 
            seq = seq[-(self.max_len+1):]
        src, trg = seq[:-1], seq[1:]
        s_pad, t_pad = np.zeros(self.max_len, int), np.zeros(self.max_len, int)
        s_pad[:len(src)], t_pad[:len(trg)] = src, trg
        return torch.tensor(u), torch.tensor(s_pad), torch.tensor(t_pad)
        
    def __len__(self): 
        return len(self.sess)

class GCNLayer(nn.Module):
    def __init__(self, dim, adj):
        super().__init__()
        self.adj = adj
        self.w = nn.Parameter(torch.FloatTensor(dim, dim))
        nn.init.xavier_uniform_(self.w)
        
    def forward(self, x): 
        return torch.mm(self.adj, torch.mm(x, self.w))

class DLIR(nn.Module):
    def __init__(self, n_users, n_pois, cfg, adj):
        super().__init__()
        self.cfg = cfg
        self.u_emb = nn.Embedding(n_users, cfg.EMBED_DIM)
        self.p_emb = nn.Embedding(n_pois, cfg.EMBED_DIM, padding_idx=0)
        self.pos = nn.Embedding(cfg.MAX_SEQ_LEN, cfg.EMBED_DIM)
        self.gcn = GCNLayer(cfg.EMBED_DIM, adj)
        
        enc = nn.TransformerEncoderLayer(cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.HIDDEN_DIM, cfg.DROPOUT, batch_first=True)
        self.trans = nn.TransformerEncoder(enc, cfg.NUM_LAYERS)
        self.fc = nn.Linear(cfg.EMBED_DIM, n_pois)
        
    def forward(self, uid, seq):
        bs, sl = seq.size()
        x = self.p_emb(seq) + self.u_emb(uid).unsqueeze(1) + \
            self.pos(torch.arange(sl, device=self.cfg.device)).unsqueeze(0)
        
        gcn_feat = self.gcn(self.p_emb.weight).index_select(0, seq.view(-1)).view(bs, sl, -1)
        return self.fc(self.trans(x + gcn_feat, src_key_padding_mask=(seq==0)))

# ==========================================
# 6. 行程生成 (带 Budget 约束)
# ==========================================
def generate_itinerary(model, uid, start_poi, target_budget, poi_details, config):
    """根据输入的真实 Budget 生成行程，并返回花费的时间"""
    model.eval()
    itinerary = [start_poi]
    budget = target_budget  # 使用实际会话的 Budget
    curr_seq = torch.tensor([[start_poi]]).to(config.device)
    visited = {start_poi}
    total_time_spent = 0.0  # 用于计算 TTR
    
    with torch.no_grad():
        while budget > 0:
            logits = model(torch.tensor([uid]).to(config.device), curr_seq)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            vals, inds = torch.topk(probs, config.TOP_K_CANDIDATES)
            
            best_poi, best_score, best_cost = None, -float('inf'), 0
            
            for pid, prob in zip(inds.cpu().numpy()[0], vals.cpu().numpy()[0]):
                if pid == 0 or pid in visited: continue
                
                c1 = poi_details[itinerary[-1]]['coords']
                c2 = poi_details[pid]['coords']
                # 计算交通时间
                travel = haversine(c1[0], c1[1], c2[0], c2[1]) / config.AVG_SPEED * 60
                pop = poi_details[pid]['popularity']
                queue = math.log(pop + 1) * 5
                
                # 总消耗时间 = 交通 + 游玩 + 排队
                cost = travel + config.VISIT_TIME + queue
                
                if cost <= budget:
                    score = (prob * math.log(pop + 1)) / max(queue, 1.0)
                    if score > best_score:
                        best_score, best_poi, best_cost = score, pid, cost
            
            if best_poi:
                itinerary.append(best_poi)
                visited.add(best_poi)
                budget -= best_cost
                total_time_spent += best_cost # 累加耗时
                
                curr_seq = torch.cat([curr_seq, torch.tensor([[best_poi]]).to(config.device)], 1)
                if curr_seq.size(1) > config.MAX_SEQ_LEN:
                    curr_seq = curr_seq[:, -config.MAX_SEQ_LEN:]
            else:
                break
                
    return itinerary, total_time_spent

# ==========================================
# 7. 主流程 (训练与评估)
# ==========================================
def main():
    dp = DataProcessor(config)
    df = dp.load_data()
    if df is None: return
    
    sessions = dp.generate_sessions(df)
    
    # --- 统一按照 8:1:1 划分 ---
    total_len = len(sessions)
    train_end = int(total_len * 0.8)
    val_end = train_end + int(total_len * 0.1)
    
    train_sess = sessions[:train_end]
    val_sess = sessions[train_end:val_end]
    test_sess = sessions[val_end:]
    
    print(f"Data Split -> Train: {len(train_sess)}, Val: {len(val_sess)}, Test: {len(test_sess)}")
    
    adj = dp.build_covisiting_matrix(train_sess)
    loader = DataLoader(DatasetWrapper(train_sess, config.MAX_SEQ_LEN), batch_size=config.BATCH_SIZE, shuffle=True)
    model = DLIR(dp.num_users, dp.num_pois, config, adj).to(config.device)
    opt = optim.Adam(model.parameters(), lr=config.LR)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    
    print("\n>>> Training Baseline (DLIR)...")
    for e in range(config.EPOCHS):
        model.train()
        tl = 0
        pbar = tqdm(loader, desc=f"Epoch {e+1}/{config.EPOCHS}", leave=False)
        for uid, src, trg in pbar:
            uid, src, trg = uid.to(config.device), src.to(config.device), trg.to(config.device)
            opt.zero_grad()
            out = model(uid, src)
            loss = crit(out.view(-1, dp.num_pois), trg.view(-1))
            loss.backward()
            opt.step()
            tl += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        print(f"Epoch {e+1}: Train Loss {tl/len(loader):.4f}")
        
    print("\n>>> Evaluating Baseline (Final Testing)...")
    all_gt, all_pred, all_times, all_budgets = [], [], [],[]
    
    # 在测试集上进行自回归生成验证
    for uid, gt, budget in tqdm(test_sess, desc="Testing", leave=False):
        if len(gt) < 2: continue
        # 传入实际 budget，返回预测序列和预测花费时间
        pred, p_time = generate_itinerary(model, uid, gt[0], budget, dp.poi_details, config)
        
        # 记录用于评价（包含起点）
        all_gt.append(gt)
        all_pred.append(pred)
        all_times.append(p_time)
        all_budgets.append(budget)
            
    # 计算所有指标
    test_metrics = calculate_all_metrics(all_gt, all_pred, all_times, all_budgets, dp.poi_details, K_list=[3, 5, 10])
    
    # ==========================================
    # 按照精美格式打印结果
    # ==========================================
    print("\n" + "-" * 50)
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

if __name__ == '__main__':
    main()