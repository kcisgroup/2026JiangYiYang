import os
import time
import math
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置与参数 (与所有基线绝对对齐)
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
        
        # 启发式推断参数 (统一的时空物理常识)
        self.AVG_SPEED = 0.25     # 预估速度 km/min (即 15km/h)
        self.VISIT_TIME = 45      # 默认游玩时间 45 min
        self.MAX_SEQ_LEN = 10     # 行程最大长度截断

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
# 3. 统一的评估指标逻辑 (包含起点、全新 TTR)
# ==========================================
def calculate_all_metrics(gt_list, pred_list, pred_times, target_budgets, poi_details, K_list=[3, 5, 10]):
    metrics = {}
    
    # 1. Top-K 指标
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
        
    # 2. Global 指标 (Pairs-F1, Diversity, TTR)
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
        cats = set([poi_details[pid]['cat'] for pid in p if pid in poi_details])
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
# 4. 数据处理 (严格保持 8:1:1 切分逻辑)
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
        df['poi_idx'] = self.poi_encoder.fit_transform(df[self.cfg.COL_POI].astype(str)) + 1 # type: ignore
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
            
        self.num_pois = len(self.poi_encoder.classes_) + 1 
        self.num_users = len(self.user_encoder.classes_)
        return df

    def generate_sessions(self, df):
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

# ==========================================
# 5. Heuristic Baseline 1: PopRank
# ==========================================
class PopRank_Solver:
    """
    无脑推爆款逻辑：把城市里最热门的 POI 排序，每次推荐最热门且时间预算够的 POI
    """
    def __init__(self, poi_details, config):
        self.poi_details = poi_details
        self.cfg = config
        # 将所有 POI 按照 Popularity 从大到小排序
        self.popular_pois = sorted(poi_details.keys(), key=lambda x: poi_details[x]['popularity'], reverse=True)

    def solve(self, start_poi, target_budget):
        path = [start_poi]
        budget = target_budget
        visited = {start_poi}
        total_time_spent = 0.0
        curr_poi = start_poi

        for cand_poi in self.popular_pois:
            # 停止条件：预算耗尽 或 达到最大序列长度
            if budget <= 0 or len(path) >= self.cfg.MAX_SEQ_LEN:
                break
                
            if cand_poi in visited: 
                continue

            c1 = self.poi_details[curr_poi]['coords']
            c2 = self.poi_details[cand_poi]['coords']
            dist = haversine(c1[0], c1[1], c2[0], c2[1])
            cost = (dist / self.cfg.AVG_SPEED) * 60.0 + self.cfg.VISIT_TIME

            if cost <= budget:
                path.append(cand_poi)
                visited.add(cand_poi)
                budget -= cost
                total_time_spent += cost
                curr_poi = cand_poi
                
        return path, total_time_spent

# ==========================================
# 6. Heuristic Baseline 2: NearestRank
# ==========================================
class NearestRank_Solver:
    """
    纯就近逻辑：从当前点出发，每次选择距离最近且时间预算够的未知 POI
    """
    def __init__(self, poi_details, config):
        self.poi_details = poi_details
        self.cfg = config
        self.all_pois = list(poi_details.keys())

    def solve(self, start_poi, target_budget):
        path = [start_poi]
        budget = target_budget
        visited = {start_poi}
        total_time_spent = 0.0
        curr_poi = start_poi

        while budget > 0 and len(path) < self.cfg.MAX_SEQ_LEN:
            best_cand = None
            best_cost = float('inf')

            c1 = self.poi_details[curr_poi]['coords']

            # 遍历寻找最近的未访问 POI
            for cand_poi in self.all_pois:
                if cand_poi in visited: 
                    continue

                c2 = self.poi_details[cand_poi]['coords']
                dist = haversine(c1[0], c1[1], c2[0], c2[1])
                cost = (dist / self.cfg.AVG_SPEED) * 60.0 + self.cfg.VISIT_TIME

                if cost <= budget and cost < best_cost:
                    best_cost = cost
                    best_cand = cand_poi

            if best_cand is not None:
                path.append(best_cand)
                visited.add(best_cand)
                budget -= best_cost
                total_time_spent += best_cost
                curr_poi = best_cand
            else:
                # 剩下的预算连最近的 POI 都去不了了，直接结束
                break

        return path, total_time_spent

# ==========================================
# 7. 评估与打印辅助函数
# ==========================================
def evaluate_and_print(name, solver, test_sess, dp, config):
    all_gt, all_pred, all_times, all_budgets = [],[], [],[]
    
    print(f"\n>>> Evaluating {name} Baseline...")
    for uid, gt, budget in tqdm(test_sess, desc=f"Testing {name}", leave=False):
        if len(gt) < 2: continue
        
        # 启发式规则不区分用户 (uid)，只关心起点和预算
        pred, p_time = solver.solve(gt[0], budget)
        
        all_gt.append(gt)
        all_pred.append(pred)
        all_times.append(p_time)
        all_budgets.append(budget)
            
    test_metrics = calculate_all_metrics(all_gt, all_pred, all_times, all_budgets, dp.poi_details, K_list=[3, 5, 10])
    
    print(f"\n{name}")
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

# ==========================================
# 8. 主流程
# ==========================================
def main():
    print(">>> Generating Heuristic Baselines (PopRank & NearestRank)")
    
    dp = DataProcessor(config)
    df = dp.load_data()
    if df is None: return
    
    sessions = dp.generate_sessions(df)
    
    # --- 统一按照 8:1:1 划分 ---
    total_len = len(sessions)
    train_end = int(total_len * 0.8)
    val_end = train_end + int(total_len * 0.1)
    
    test_sess = sessions[val_end:]
    print(f"Total Test Sessions: {len(test_sess)}")
    
    # 实例化两个推断器
    pop_solver = PopRank_Solver(dp.poi_details, config)
    nearest_solver = NearestRank_Solver(dp.poi_details, config)
    
    # 分别评测并打印
    evaluate_and_print("PopRank", pop_solver, test_sess, dp, config)
    evaluate_and_print("NearestRank", nearest_solver, test_sess, dp, config)

if __name__ == '__main__':
    main()