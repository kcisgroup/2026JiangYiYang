import os
import time
import math
import random
import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
        
        # 数据集列名
        self.COL_TIME = 'time'
        self.COL_USER = 'user_id'
        self.COL_POI = 'geo_id'
        self.COL_CAT = 'venue_category_name'
        self.COL_LON = 'longitude'
        self.COL_LAT = 'latitude'
        
        # DROMC 特定参数
        self.BASE_SPEED = 0.83     # 50km/h ≈ 0.83 km/min
        self.MIN_SPEED = 0.33      # 拥堵时 20km/h ≈ 0.33 km/min
        self.STAY_TIME = 45        # 平均停留时间统一设为 45 min (对齐之前基线)
        
        # 营业时间模拟 (开始, 结束) - 单位: 一天中的分钟数
        self.OPEN_TIME_RANGE = (8*60, 10*60)   # 8:00 - 10:00 开门
        self.CLOSE_TIME_RANGE = (20*60, 24*60) # 20:00 - 24:00 关门

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# ==========================================
# 2. 基础工具函数
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

def get_dynamic_speed(current_time_min, config):
    """
    模拟时间依赖路网：根据一天中的时间返回速度
    input: current_time_min (0-1440)
    """
    t = current_time_min % 1440
    # 早高峰 8:00-9:00 (480-540)
    if 480 <= t <= 540:
        return config.MIN_SPEED
    # 晚高峰 17:00-19:00 (1020-1140)
    elif 1020 <= t <= 1140:
        return config.MIN_SPEED
    else:
        return config.BASE_SPEED

# ==========================================
# 3. 统一的评估指标逻辑 (与 DLIR 完全对齐)
# ==========================================
def calculate_all_metrics(gt_list, pred_list, pred_times, target_budgets, poi_details, K_list=[3, 5, 10]):
    """
    整合计算所有的指标：包含起点的 Top-K, Pairs-F1, Diversity, 以及最新的 TTR
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
        
        # TTR (最新公式)
        if budget > 0:
            ttr_val = max(0.0, 1.0 - abs(p_time - budget) / budget)
            ttr_list.append(ttr_val)
            
    metrics['Pairs-F1'] = np.mean(pairs_f1_list) if pairs_f1_list else 0.0
    metrics['Diversity'] = np.mean(div_list) if div_list else 0.0
    metrics['TTR'] = np.mean(ttr_list) if ttr_list else 0.0
    
    return metrics

# ==========================================
# 4. 数据处理 (添加预算提取)
# ==========================================
class DataProcessor:
    def __init__(self, config):
        self.cfg = config
        self.poi_encoder = LabelEncoder()
        self.user_encoder = LabelEncoder()
        self.cat_encoder = LabelEncoder()
        self.poi_details = {}
        self.category_inverted_index = defaultdict(list)
        
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
        
        # 编码
        df['user_idx'] = self.user_encoder.fit_transform(df[self.cfg.COL_USER].astype(str))
        df['poi_idx'] = self.poi_encoder.fit_transform(df[self.cfg.COL_POI].astype(str)) + 1 # type: ignore
        df['cat_idx'] = self.cat_encoder.fit_transform(df[self.cfg.COL_CAT].astype(str))
        
        np.random.seed(42) 
        for _, row in df.iterrows():
            pid = row['poi_idx']
            cid = row['cat_idx']
            
            if pid not in self.poi_details:
                open_t = np.random.randint(self.cfg.OPEN_TIME_RANGE[0], self.cfg.OPEN_TIME_RANGE[1])
                close_t = np.random.randint(self.cfg.CLOSE_TIME_RANGE[0], self.cfg.CLOSE_TIME_RANGE[1])
                self.poi_details[pid] = {
                    'cat': cid,
                    'coords': (row[self.cfg.COL_LAT], row[self.cfg.COL_LON]),
                    'popularity': 0,
                    'open_time': open_t,
                    'close_time': close_t
                }
                self.category_inverted_index[cid].append(pid)
                
            self.poi_details[pid]['popularity'] += 1
            
        self.num_pois = len(self.poi_encoder.classes_) + 1
        print(f"Loaded: {self.num_pois-1} POIs, {len(self.cat_encoder.classes_)} Categories.")
        return df

    def generate_sessions(self, df):
        """生成会话，提取开始时间和真实预算"""
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
                            start_min = temp_times[0].hour * 60 + temp_times[0].minute
                            budget = (temp_times[-1] - temp_times[0]).total_seconds() / 60.0
                            budget = max(30.0, min(budget, 720.0))
                            sessions.append((uid, temp_seq, start_min, budget))
                        temp_seq = [poi]
                        temp_times =[curr_time]
                    else:
                        temp_seq.append(poi)
                        temp_times.append(curr_time)
                        
            if len(temp_seq) >= 3:
                start_min = temp_times[0].hour * 60 + temp_times[0].minute
                budget = (temp_times[-1] - temp_times[0]).total_seconds() / 60.0
                budget = max(30.0, min(budget, 720.0))
                sessions.append((uid, temp_seq, start_min, budget))
                
        return sessions

# ==========================================
# 5. DROMC 核心算法 (约束与消耗统一)
# ==========================================
class DROMC_Solver:
    def __init__(self, poi_details, cat_index, config):
        self.poi_details = poi_details
        self.cat_index = cat_index
        self.cfg = config
        
    def check_is_open(self, poi_id, current_time):
        """检查POI在当前时间是否营业"""
        info = self.poi_details[poi_id]
        time_of_day = current_time % 1440
        return info['open_time'] <= time_of_day <= info['close_time']

    def solve(self, start_node, end_node, required_cats, start_time, target_budget):
        """
        DROMC 规划器: 动态追踪耗时并返回
        """
        path =[start_node]
        current_node = start_node
        current_time = start_time
        remaining_cats = set(required_cats)
        visited_pois = {start_node}
        
        total_time_spent = 0.0 # 记录总耗时
        end_time_limit = start_time + target_budget # 使用真实预算约束
        
        while remaining_cats and current_time < end_time_limit:
            best_next_poi = None
            min_cost = float('inf')
            satisfied_cat = None
            
            speed = get_dynamic_speed(current_time, self.cfg)
            c1 = self.poi_details[current_node]['coords']
            
            for cat in remaining_cats:
                cat_pois = self.cat_index[cat]
                sample_pois = cat_pois if len(cat_pois) < 20 else random.sample(cat_pois, 20)
                
                for cand_id in sample_pois:
                    if cand_id in visited_pois: continue
                    
                    c2 = self.poi_details[cand_id]['coords']
                    dist = haversine(c1[0], c1[1], c2[0], c2[1])
                    travel_time = (dist / speed) * 60
                    arrival_time = current_time + travel_time
                    
                    if not self.check_is_open(cand_id, arrival_time):
                        continue
                        
                    c_end = self.poi_details[end_node]['coords']
                    dist_to_end = haversine(c2[0], c2[1], c_end[0], c_end[1])
                    heuristic = (dist_to_end / self.cfg.BASE_SPEED) * 60
                    
                    total_cost = travel_time + heuristic
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_next_poi = cand_id
                        satisfied_cat = cat
            
            if best_next_poi:
                path.append(best_next_poi)
                visited_pois.add(best_next_poi)
                remaining_cats.remove(satisfied_cat)
                
                c2 = self.poi_details[best_next_poi]['coords']
                dist = haversine(c1[0], c1[1], c2[0], c2[1])
                travel_time = (dist / speed) * 60
                
                # 更新时间和耗时
                step_cost = travel_time + self.cfg.STAY_TIME
                current_time += step_cost
                total_time_spent += step_cost
                current_node = best_next_poi
            else:
                break
        
        # 最后去终点
        if current_node != end_node:
            c1 = self.poi_details[current_node]['coords']
            c2 = self.poi_details[end_node]['coords']
            dist = haversine(c1[0], c1[1], c2[0], c2[1])
            travel_time = (dist / get_dynamic_speed(current_time, self.cfg)) * 60
            
            step_cost = travel_time + self.cfg.STAY_TIME
            total_time_spent += step_cost
            path.append(end_node)
            
        return path, total_time_spent

# ==========================================
# 6. 主程序
# ==========================================
def main():
    print(">>> DROMC Reproduction (Time-Dependent & Multi-Category Constraints)")
    
    dp = DataProcessor(config)
    df = dp.load_data()
    if df is None: return
    
    sessions = dp.generate_sessions(df)
    
    # --- 统一按照 8:1:1 划分 (严格对齐标准) ---
    total_len = len(sessions)
    train_end = int(total_len * 0.8)
    val_end = train_end + int(total_len * 0.1)
    
    train_sess = sessions[:train_end]
    val_sess = sessions[train_end:val_end]
    test_sess = sessions[val_end:]
    
    print(f"Data Split -> Train: {len(train_sess)}, Val: {len(val_sess)}, Test: {len(test_sess)}")
    
    solver = DROMC_Solver(dp.poi_details, dp.category_inverted_index, config)
    
    all_gt, all_pred, all_times, all_budgets = [], [], [],[]
    
    print("\n>>> Evaluating DROMC Baseline (Final Testing)...")
    
    # 使用 test_sess 保证测试集数据口径一致
    for uid, seq, start_time_min, budget in tqdm(test_sess, desc="Testing", leave=False):
        start_node = seq[0]
        end_node = seq[-1]
        
        middle_pois = seq[1:-1]
        required_cats = set([dp.poi_details[p]['cat'] for p in middle_pois])
        
        if not required_cats:
            # 只有起点终点，补齐默认耗时
            pred_path =[start_node, end_node]
            c1, c2 = dp.poi_details[start_node]['coords'], dp.poi_details[end_node]['coords']
            dist = haversine(c1[0], c1[1], c2[0], c2[1])
            p_time = (dist / config.BASE_SPEED) * 60 + config.STAY_TIME
        else:
            # 执行带有真实预算约束的 DROMC 求解
            pred_path, p_time = solver.solve(start_node, end_node, required_cats, start_time_min, budget)
        
        # 记录用于评价（包含起点）
        all_gt.append(seq)
        all_pred.append(pred_path)
        all_times.append(p_time)
        all_budgets.append(budget)
            
    # 计算所有指标
    test_metrics = calculate_all_metrics(all_gt, all_pred, all_times, all_budgets, dp.poi_details, K_list=[3, 5, 10])
    
    # ==========================================
    # 精美的最终打印格式
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