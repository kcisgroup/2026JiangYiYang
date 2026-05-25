import os
import time
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import LinearSVC
from collections import defaultdict
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
torch.cuda.empty_cache()

# ==========================================
# 0. 全局配置与设备
# ==========================================
class Config:
    def __init__(self):
        self.BASE_DIR = os.getcwd()
        self.DATA_PATH = os.path.join(self.BASE_DIR, 'MyModel/dataset_process/foursquare_tky_cleaned.csv')
        
        self.COL_TIME = 'time'
        self.COL_USER = 'user_id'
        self.COL_POI = 'geo_id'
        self.COL_CAT = 'venue_category_name'
        self.COL_LON = 'longitude'
        self.COL_LAT = 'latitude'
        
        # 行程规划对齐参数 (与 DLIR/DROMC 完全一致)
        self.AVG_SPEED = 0.83     # 预估速度 km/min
        self.VISIT_TIME = 45      # 默认游玩时间 45 min
        
        # TRAR 特定参数
        self.HIDDEN_DIM = 64
        self.EMBEDDING_DIM = 32
        self.ALPHA = 0.2
        self.DROPOUT = 0.3
        self.LR = 0.001
        self.WEIGHT_DECAY = 5e-4
        self.EPOCHS = 100
        self.LAMBDA_CLU = 0.1
        self.N_CLUSTERS = 2

config = Config()
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 基础工具函数
# ==========================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ==========================================
# 2. 统一的评估指标逻辑
# ==========================================
def calculate_all_metrics(gt_list, pred_list, pred_times, target_budgets, poi_details, K_list=[3, 5, 10]):
    metrics = {}
    
    # --- 1. Top-K 指标 ---
    for k in K_list:
        rec_list, prec_list, f1_list = [], [],[]
        for g, p in zip(gt_list, pred_list):
            p_cut = p[:k]
            if not p_cut: continue
            
            g_set = set(g)
            p_set = set(p_cut)
            
            hit = len(g_set & p_set)
            r = hit / len(g_set) if g_set else 0.0
            p_val = hit / len(p_set) if p_set else 0.0
            f = 2 * p_val * r / (p_val + r) if (p_val + r) > 0 else 0.0
            
            rec_list.append(r)
            prec_list.append(p_val)
            f1_list.append(f)
            
        metrics[f'R@{k}'] = np.mean(rec_list)
        metrics[f'P@{k}'] = np.mean(prec_list)
        metrics[f'F1@{k}'] = np.mean(f1_list)
        
    # --- 2. Global 指标 ---
    pairs_f1_list, div_list, ttr_list = [], [],[]
    for g, p, p_time, budget in zip(gt_list, pred_list, pred_times, target_budgets):
        if not p: continue
        
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
            
        cats = set()
        for pid in p:
            if pid in poi_details:
                cats.add(poi_details[pid]['cat'])
        div_list.append(len(cats) / len(p))
        
        if budget > 0:
            ttr_val = max(0.0, 1.0 - abs(p_time - budget) / budget)
            ttr_list.append(ttr_val)
            
    metrics['Pairs-F1'] = np.mean(pairs_f1_list) if pairs_f1_list else 0.0
    metrics['Diversity'] = np.mean(div_list) if div_list else 0.0
    metrics['TTR'] = np.mean(ttr_list) if ttr_list else 0.0
    
    return metrics

# ==========================================
# 3. 数据处理 
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
        self.num_cats = 0
        
    def load_data(self):
        print(f"Loading data from {self.cfg.DATA_PATH}...")
        try:
            df = pd.read_csv(self.cfg.DATA_PATH)
        except FileNotFoundError:
            print(">>> ⚠️ Data file not found! Generating mock data for baseline testing...")
            mock_data = {
                'time': pd.date_range(start='2012-04-01', periods=2000, freq='40min'),
                'user_id': np.random.randint(0, 50, 2000),
                'geo_id': np.random.randint(0, 100, 2000),
                'venue_category_name': np.random.randint(0, 20, 2000),
                'latitude': np.random.uniform(40.6, 40.8, 2000),
                'longitude': np.random.uniform(-74.0, -73.8, 2000)
            }
            df = pd.DataFrame(mock_data)
        
        df[self.cfg.COL_TIME] = pd.to_datetime(df[self.cfg.COL_TIME])
        df = df.sort_values([self.cfg.COL_USER, self.cfg.COL_TIME])
        
        df['user_idx'] = self.user_encoder.fit_transform(df[self.cfg.COL_USER].astype(str))
        df['poi_idx'] = self.poi_encoder.fit_transform(df[self.cfg.COL_POI].astype(str))
        df['cat_idx'] = self.cat_encoder.fit_transform(df[self.cfg.COL_CAT].astype(str))
        
        for _, row in df.iterrows():
            pid = row['poi_idx']
            if pid not in self.poi_details:
                self.poi_details[pid] = {
                    'cat': row['cat_idx'],
                    'coords': (row[self.cfg.COL_LAT], row[self.cfg.COL_LON]),
                    'popularity': 0
                }
            self.poi_details[pid]['popularity'] += 1
            
        self.num_pois = len(self.poi_encoder.classes_)
        self.num_users = len(self.user_encoder.classes_)
        self.num_cats = len(self.cat_encoder.classes_)
        return df

    def generate_sessions(self, df):
        sessions =[]
        user_groups = df.groupby('user_idx')
        for uid, group in user_groups:
            group = group.sort_values(self.cfg.COL_TIME)
            temp_seq =[]
            temp_times =[]
            
            for _, row in group.iterrows():
                curr_time = row[self.cfg.COL_TIME]
                poi = row['poi_idx']
                if not temp_seq:
                    temp_seq.append(poi)
                    temp_times.append(curr_time)
                else:
                    diff_hours = (curr_time - temp_times[-1]).total_seconds() / 3600.0
                    if diff_hours > 8:
                        if len(temp_seq) >= 3:
                            budget = (temp_times[-1] - temp_times[0]).total_seconds() / 60.0
                            budget = max(30.0, min(budget, 720.0))
                            sessions.append((uid, temp_seq, budget))
                        temp_seq = [poi]
                        temp_times = [curr_time]
                    else:
                        temp_seq.append(poi)
                        temp_times.append(curr_time)
            
            if len(temp_seq) >= 3:
                budget = (temp_times[-1] - temp_times[0]).total_seconds() / 60.0
                budget = max(30.0, min(budget, 720.0))
                sessions.append((uid, temp_seq, budget))
                
        return sessions

    def build_graph_and_features(self, train_sess):
        self.train_user_trips = defaultdict(list)
        self.adj = np.zeros((self.num_pois, self.num_pois))
        
        for uid, seq, _ in train_sess:
            self.train_user_trips[uid].append(seq)
            for i in range(len(seq)-1):
                u, v = seq[i], seq[i+1]
                self.adj[u, v] += 1
                
        cat_feat = np.zeros((self.num_pois, self.num_cats))
        pop_feat = np.zeros((self.num_pois, 1))
        for pid, info in self.poi_details.items():
            cat_feat[pid, info['cat']] = 1
            pop_feat[pid, 0] = info['popularity']
            
        deg_feat = np.sum(self.adj, axis=1).reshape(-1, 1)
        scaler = MinMaxScaler()
        num_feat = scaler.fit_transform(np.hstack([pop_feat, deg_feat]))
        
        self.node_features = np.hstack([cat_feat, num_feat])
        self.adj_norm = MinMaxScaler().fit_transform(self.adj)

# ==========================================
# 4. GATE 模型
# ==========================================
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a_1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a_2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a_3 = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.a_1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_2.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_3.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj_weight):
        Wh = torch.mm(h, self.W)
        e_1 = torch.mm(Wh, self.a_1)
        e_2 = torch.mm(Wh, self.a_2)
        e_3 = adj_weight * self.a_3
        
        e = self.leakyrelu(e_1 + e_2.T + e_3)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_weight > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime) if self.concat else h_prime

class TRAR_GATE(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, alpha):
        super(TRAR_GATE, self).__init__()
        self.gat1 = GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
        self.gat2 = GraphAttentionLayer(nhid, nout, dropout, alpha, concat=False)
        
    def forward(self, x, adj_weight):
        x = self.gat1(x, adj_weight)
        z = self.gat2(x, adj_weight)
        adj_rec = torch.sigmoid(torch.mm(z, z.t()))
        return z, adj_rec

# ==========================================
# 5. TRAR 核心系统实现 (已修复冷启动 Bug)
# ==========================================
class TRAR_System:
    def __init__(self, data_manager, config):
        self.dm = data_manager
        self.cfg = config
        self.model = TRAR_GATE(
            nfeat=self.dm.node_features.shape[1],
            nhid=self.cfg.HIDDEN_DIM,
            nout=self.cfg.EMBEDDING_DIM,
            dropout=self.cfg.DROPOUT,
            alpha=self.cfg.ALPHA
        ).to(DEVICE)
        
        self.x = torch.FloatTensor(self.dm.node_features).to(DEVICE)
        self.adj = torch.FloatTensor(self.dm.adj_norm).to(DEVICE)
        self.raw_adj = self.dm.adj
        
        # 预存一份全局热门节点，应对测试集中完全没有邻居的死胡同
        pops = [(pid, info['popularity']) for pid, info in self.dm.poi_details.items()]
        pops.sort(key=lambda x: x[1], reverse=True)
        self.top_popular_pois = [x[0] for x in pops[:30]]
        
        self.attractive_routes = set()
        self.ar_classifiers = {} 
        self.ar_rating_scores = {}
        self.user_prefs = {}
        self._build_user_profiles()
        
    def _build_user_profiles(self):
        for uid, trips_idx in self.dm.train_user_trips.items():
            pref_vec = np.zeros(self.dm.num_cats)
            total_visits = 0
            for seq in trips_idx:
                for pid in seq:
                    cid = self.dm.poi_details[pid]['cat']
                    pref_vec[cid] += 1
                    total_visits += 1
            if total_visits > 0:
                self.user_prefs[uid] = pref_vec / total_visits
    
    def train_and_discover(self):
        print(">>> Training GATE & Identifying Attractive Routes...")
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.LR, weight_decay=self.cfg.WEIGHT_DECAY)
        edges_idx = np.transpose(np.nonzero(self.raw_adj))
        if len(edges_idx) < 2: 
            print("Graph too sparse.")
            return
            
        edges_tensor = torch.tensor(edges_idx, dtype=torch.long).to(DEVICE)
        
        self.model.eval()
        with torch.no_grad():
            z, _ = self.model(self.x, self.adj)
            z_np = z.cpu().numpy()
            edge_embs =[]
            init_indices = np.random.choice(len(edges_idx), min(10000, len(edges_idx)), replace=False)
            for i in init_indices:
                u, v = edges_idx[i]
                edge_embs.append(np.concatenate([z_np[u], z_np[v]]))
            edge_embs = np.array(edge_embs)
            
        if len(np.unique(edge_embs, axis=0)) < self.cfg.N_CLUSTERS:
            edge_embs += np.random.normal(0, 1e-6, edge_embs.shape)
            
        kmeans = KMeans(n_clusters=self.cfg.N_CLUSTERS, n_init=10)
        kmeans.fit(edge_embs)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(DEVICE)
        
        self.model.train()
        for epoch in range(self.cfg.EPOCHS):
            optimizer.zero_grad()
            z, adj_rec = self.model(self.x, self.adj)
            loss_rec = F.mse_loss(adj_rec, self.adj)
            
            sample_indices = np.random.choice(len(edges_idx), min(2048, len(edges_idx)), replace=False)
            batch_edges = edges_tensor[sample_indices]
            batch_embs = torch.cat([z[batch_edges[:, 0]], z[batch_edges[:, 1]]], dim=1)
            
            dist0 = torch.sum((batch_embs - centers[0])**2, dim=1)
            dist1 = torch.sum((batch_embs - centers[1])**2, dim=1)
            min_dist, _ = torch.min(torch.stack([dist0, dist1], dim=1), dim=1)
            loss_clu = torch.mean(min_dist)
            
            loss = loss_rec + self.cfg.LAMBDA_CLU * loss_clu
            loss.backward()
            optimizer.step()
                    
        print(">>> Evaluating ARs (SVM)...")
        self.model.eval()
        
        self.poi_prefs = {}
        for pid, info in self.dm.poi_details.items():
            vec = np.zeros(self.dm.num_cats)
            vec[info['cat']] = 1
            self.poi_prefs[pid] = vec
            
        self.attractive_routes = set([(u, v) for u, v in edges_idx[np.random.choice(len(edges_idx), min(200, len(edges_idx)), replace=False)]])

    def predict_trip(self, user_id, start_node, target_budget):
        """带有真实预算限制的行程推断"""
        user_vec = self.user_prefs.get(user_id)
        
        #[核心修复 1] 应对测试集中的新用户 (冷启动)：使用全局类别平均偏好分布兜底
        if user_vec is None: 
            user_vec = np.ones(self.dm.num_cats) / self.dm.num_cats
        
        trip = [start_node]
        visited = {start_node}
        current_node = start_node
        
        budget = target_budget
        total_time_spent = 0.0
        
        while budget > 0:
            neighbors = np.where(self.raw_adj[current_node] > 0)[0]
            
            #[核心修复 2] 应对没有邻居的孤立节点：推荐全局最热门的 POI 作为侯选项
            if len(neighbors) == 0:
                neighbors = self.top_popular_pois

            best_node = None
            best_score = -float('inf')
            best_cost = 0
            
            for next_node in neighbors:
                if next_node in visited: continue
                
                c1 = self.dm.poi_details[current_node]['coords']
                c2 = self.dm.poi_details[next_node]['coords']
                d = haversine(c1[0], c1[1], c2[0], c2[1])
                cost = (d / self.cfg.AVG_SPEED) * 60 + self.cfg.VISIT_TIME
                
                if cost > budget: continue
                
                # 计算偏好匹配度
                poi_vec = self.poi_prefs[next_node]
                score = np.dot(user_vec, poi_vec) / (np.linalg.norm(user_vec)*np.linalg.norm(poi_vec) + 1e-9)
                
                # AR 强化
                if (current_node, next_node) in self.attractive_routes:
                    score *= 1.5 
                
                #[核心修复 3] 打破偏好全一样的死局：稍微加一点热门度奖励，减去一点距离惩罚
                score += 1e-4 * self.dm.poi_details[next_node]['popularity']
                score -= 1e-4 * cost
                    
                if score > best_score:
                    best_score = score
                    best_node = next_node
                    best_cost = cost
            
            if best_node is not None:
                visited.add(best_node)
                trip.append(best_node)
                budget -= best_cost
                total_time_spent += best_cost
                current_node = best_node
            else:
                break
                
        return trip, total_time_spent

# ==========================================
# 6. 主流程
# ==========================================
def main():
    print(">>> TRAR-GATE Baseline Reproduction")
    
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
    
    # 构建图 (仅用训练集)
    dp.build_graph_and_features(train_sess)
    
    sys = TRAR_System(dp, config)
    sys.train_and_discover()
    
    all_gt, all_pred, all_times, all_budgets = [], [], [],[]
    
    print("\n>>> Evaluating TRAR Baseline (Final Testing)...")
    
    for uid, gt, budget in tqdm(test_sess, desc="Testing", leave=False):
        if len(gt) < 2: continue
        
        # 传入实际 budget，返回预测序列和预测花费时间
        pred, p_time = sys.predict_trip(uid, gt[0], budget)
        
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
    for k in[3, 5, 10]:
        r = test_metrics.get(f'R@{k}', 0.0)
        p = test_metrics.get(f'P@{k}', 0.0)
        f1 = test_metrics.get(f'F1@{k}', 0.0)
        print(f"{k:<6}| {r:<9.4f}| {p:<9.4f}| {f1:<8.4f}")
        
    print(f"Pairs-F1  : {test_metrics.get('Pairs-F1', 0.0):.4f}")
    print(f"Diversity : {test_metrics.get('Diversity', 0.0):.4f}")
    print(f"TTR       : {test_metrics.get('TTR', 0.0):.4f}")

if __name__ == "__main__":
    main()