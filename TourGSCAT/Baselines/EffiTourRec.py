import os
import time
import math
import random
import warnings
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
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
        
        # 行程规划对齐参数 (与其余模型保持绝对一致)
        self.AVG_SPEED = 0.25      # 预估速度 km/min (15km/h)
        self.MAX_SEQ_LEN = 20
        
        # EffiTourRec (MCTS) 特定参数
        self.MAX_LOOP = 200        # MCTS 迭代次数 (论文中上限可达1000，200为兼顾速度与精度的平衡点)
        self.C_P = 0.707           # UCT 中的探索系数 (1/sqrt(2))
        self.SAMPLE_SIZE = 50      # 每次扩展时采样的候选POI数量 (加速大图搜索)

config = Config()

# ==========================================
# 2. 距离计算工具
# ==========================================
def haversine(lat1, lon1, lat2, lon2):
    """计算两点间地球表面距离 (km)"""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ==========================================
# 3. 统一的评估指标逻辑
# ==========================================
def calculate_all_metrics(gt_list, pred_list, pred_times, target_budgets, poi_details, K_list=[3, 5, 10]):
    """
    计算所有的行程规划指标：包含起点的 Top-K, Pairs-F1, Diversity, 优化后的 TTR
    """
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
        
        # Pairs-F1
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
            
        # Diversity
        cats = set()
        for pid in p:
            if pid in poi_details:
                cats.add(poi_details[pid]['cat'])
        div_list.append(len(cats) / len(p))
        
        # TTR
        if budget > 0:
            ttr_val = max(0.0, 1.0 - abs(p_time - budget) / budget)
            ttr_list.append(ttr_val)
            
    metrics['Pairs-F1'] = np.mean(pairs_f1_list) if pairs_f1_list else 0.0
    metrics['Diversity'] = np.mean(div_list) if div_list else 0.0
    metrics['TTR'] = np.mean(ttr_list) if ttr_list else 0.0
    
    return metrics

# ==========================================
# 4. 数据处理与画像构建
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
        return df

    def generate_sessions(self, df):
        """生成会话，并提取时间预算"""
        sessions =[]
        cat_dwell_times = defaultdict(list)
        user_groups = df.groupby('user_idx')
        
        for uid, group in user_groups:
            group = group.sort_values(self.cfg.COL_TIME)
            temp_seq = []
            temp_times =[]
            
            for _, row in group.iterrows():
                curr_time = row[self.cfg.COL_TIME]
                poi = row['poi_idx']
                cat = row['cat_idx']
                
                if not temp_seq:
                    temp_seq.append(poi)
                    temp_times.append(curr_time)
                else:
                    diff_hours = (curr_time - temp_times[-1]).total_seconds() / 3600.0
                    if 10 < diff_hours * 60.0 < 240:
                        cat_dwell_times[self.poi_details[temp_seq[-1]]['cat']].append(diff_hours * 60.0 * 0.5)
                        
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
                
        # 计算 VDoP (Visit Duration of POI)
        avg_cat_dwell = {c: np.mean(times) if times else 45.0 for c, times in cat_dwell_times.items()}
        for pid in self.poi_details:
            self.poi_details[pid]['vdop'] = avg_cat_dwell.get(self.poi_details[pid]['cat'], 45.0)
            
        return sessions

    def build_user_profiles(self, train_sess):
        """【重现论文公式 2】：提取用户个人的兴趣画像 (IoP)"""
        self.user_profiles = defaultdict(lambda: defaultdict(int))
        self.global_cat_dist = defaultdict(int)
        total_visits = 0
        
        for uid, seq, _ in train_sess:
            for pid in seq:
                cat = self.poi_details[pid]['cat']
                self.user_profiles[uid][cat] += 1
                self.global_cat_dist[cat] += 1
                total_visits += 1
                
        # 归一化为概率分布 [0, 1]
        for uid in self.user_profiles:
            total_u = sum(self.user_profiles[uid].values())
            for cat in self.user_profiles[uid]:
                self.user_profiles[uid][cat] /= total_u # type: ignore
                
        for cat in self.global_cat_dist:
            self.global_cat_dist[cat] /= total_visits # type: ignore

# ==========================================
# 5. EffiTourRec 核心算法 (MCTS + Pruning)
# ==========================================
class EffiTourRec_Solver:
    def __init__(self, poi_details, user_profiles, global_cat_dist, config):
        self.poi_details = poi_details
        self.user_profiles = user_profiles
        self.global_cat_dist = global_cat_dist
        self.cfg = config
        
        # 预计算归一化的特征，以保证 MCTS UCT 乘法项稳定
        max_pop = max([info['popularity'] for info in poi_details.values()])
        self.norm_pop = {pid: info['popularity'] / max_pop for pid, info in poi_details.items()}
        
        max_vdop = max([info['vdop'] for info in poi_details.values()])
        self.norm_vdop = {pid: info['vdop'] / max_vdop for pid, info in poi_details.items()}
        
        # Queue Time: Foursquare 无队列数据，利用论文概念，用 popularity 模拟拥挤产生的排队
        self.queue_time = {pid: math.log(info['popularity'] + 1) * 5.0 for pid, info in poi_details.items()}
        
    def get_iop(self, uid, cid):
        """获取用户对类别的兴趣度 (IoP)"""
        if uid in self.user_profiles and cid in self.user_profiles[uid]:
            return self.user_profiles[uid][cid]
        return self.global_cat_dist.get(cid, 0.0)

    def solve(self, uid, start_poi, target_budget):
        """【复刻论文 Algorithm 1 & 2】：基于剪枝的 MCTS 行程推荐"""
        T_visits = {}   # 记录节点访问次数 key: (depth, poi), val: count
        T_reward = {}   # 记录累积Reward key: (depth, poi), val: total_reward
        T_prune = {}    # MCTS Pruning Tree key: (depth, poi), val: max_Prune_Factor
        
        I_list =[]
        
        for iteration in range(self.cfg.MAX_LOOP):
            path = [start_poi]
            time_spent = 0.0
            temp_reward = 0.0
            visited = {start_poi}
            
            nodes_visited_in_iter =[]
            
            while time_spent < target_budget:
                curr_poi = path[-1]
                depth = len(path)
                
                best_poi = None
                best_uct = -float('inf')
                best_cost = 0.0
                best_step_reward = 0.0
                
                visit_i = T_visits.get((depth-1, curr_poi), 1)
                
                # 采样候选集 (加速版展开)
                candidates = random.sample(list(self.poi_details.keys()), min(self.cfg.SAMPLE_SIZE, len(self.poi_details)))
                
                for p_j in candidates:
                    if p_j in visited: continue
                    
                    # 1. 计算时间成本
                    c1 = self.poi_details[curr_poi]['coords']
                    c2 = self.poi_details[p_j]['coords']
                    dist = haversine(c1[0], c1[1], c2[0], c2[1])
                    travel_t = (dist / self.cfg.AVG_SPEED) * 60.0
                    
                    cost = travel_t + self.poi_details[p_j]['vdop'] + self.queue_time[p_j]
                    if time_spent + cost > target_budget:
                        continue
                        
                    # 2. 【复现公式 12】：计算 Heuristic (Potential POI)
                    pop_val = self.norm_pop[p_j]
                    iop_val = self.get_iop(uid, self.poi_details[p_j]['cat'])
                    vdop_val = self.norm_vdop[p_j]
                    q_val = self.queue_time[p_j] + 1.0 # +1 防除零
                    
                    heuristic = (pop_val * iop_val * vdop_val) / (travel_t + q_val)
                    
                    # 3. 计算 UCT (加入探索与开发)
                    visit_j = T_visits.get((depth, p_j), 0)
                    if visit_j == 0:
                        # 对于未访问节点，加上一个常数，优先根据启发式排序探索
                        uct = heuristic + 1000.0 
                    else:
                        reward_j = T_reward.get((depth, p_j), 0.0)
                        exploitation = heuristic + (reward_j / visit_j)
                        exploration = 2 * self.cfg.C_P * math.sqrt(2 * math.log(visit_i) / visit_j)
                        uct = exploitation + exploration
                        
                    if uct > best_uct:
                        best_uct = uct
                        best_poi = p_j
                        best_cost = cost
                        # 【复现公式 13】：单步 Reward
                        best_step_reward = (pop_val * iop_val * vdop_val) / q_val
                        
                if best_poi is None:
                    break # 时间耗尽或无点可去
                    
                path.append(best_poi)
                visited.add(best_poi)
                time_spent += best_cost
                temp_reward += best_step_reward
                nodes_visited_in_iter.append((depth, best_poi, time_spent, temp_reward))
                
                # 4. 【复现公式 14, 15】：MCTS 剪枝机制 (Pruning Technique)
                PF = temp_reward / time_spent if time_spent > 0 else 0
                existing_PF = T_prune.get((depth, best_poi), -1.0)
                if existing_PF != -1.0 and PF <= existing_PF:
                    # 如果当前生成的子路径性价比不如已经探索过的，提前终止！
                    break 
                    
            # --- 反向传播 (Back-propagation) ---
            if len(path) > 1:
                I_list.append({
                    'path': path,
                    'reward': temp_reward,
                    'time': time_spent
                })
                
                for d, p, ct, cr in nodes_visited_in_iter:
                    T_visits[(d, p)] = T_visits.get((d, p), 0) + 1
                    T_reward[(d, p)] = T_reward.get((d, p), 0.0) + temp_reward # 累加整条路径的总收益
                    
                    # 更新剪枝树的 Prune Factor
                    pf = cr / ct if ct > 0 else 0
                    if pf > T_prune.get((d, p), -1.0):
                        T_prune[(d, p)] = pf
        
        if not I_list:
            return [start_poi], 0.0
            
        # 选择所有迭代中 Reward 最高的合法路线
        best_itinerary = max(I_list, key=lambda x: x['reward'])
        return best_itinerary['path'], best_itinerary['time']

# ==========================================
# 6. 主流程
# ==========================================
def main():
    print(">>> EffiTourRec (Adaptive MCTS + Pruning) Baseline Reproduction")
    
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
    
    # 仅使用训练集构建用户兴趣偏好 (避免数据穿越)
    dp.build_user_profiles(train_sess)
    
    solver = EffiTourRec_Solver(dp.poi_details, dp.user_profiles, dp.global_cat_dist, config)
    
    all_gt, all_pred, all_times, all_budgets = [], [], [],[]
    
    print(f"\n>>> Evaluating EffiTourRec Baseline (MaxLoop={config.MAX_LOOP})...")
    
    for uid, gt, budget in tqdm(test_sess, desc="MCTS Testing", leave=False):
        if len(gt) < 2: continue
        
        # 传入实际 budget，MCTS 会在预算内寻找最优解
        pred_path, p_time = solver.solve(uid, gt[0], budget)
        
        all_gt.append(gt)
        all_pred.append(pred_path)
        all_times.append(p_time)
        all_budgets.append(budget)
            
    # 计算所有指标
    test_metrics = calculate_all_metrics(all_gt, all_pred, all_times, all_budgets, dp.poi_details, K_list=[3, 5, 10])
    
    # ==========================================
    # 精美表格输出
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