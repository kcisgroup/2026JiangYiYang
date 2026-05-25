import math
import random
import numpy as np
import torch

def set_seed(seed):
    """设置随机种子，保证实验的可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def haversine(lon1, lat1, lon2, lat2):
    """
    计算地球上两点之间的球面距离 (单位: km)
    """
    R = 6371.0 # 地球半径
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def calculate_all_metrics(gt_list, pred_list, pred_times, target_budgets, poi_details, K_list=[3, 5, 10]):
    """
    评估生成的行程质量。
    包括: Precision@K, Recall@K, F1@K, Pairs-F1(成对F1), Diversity(类别多样性), TTR(时间履约率)
    """
    metrics = {}
    
    # 1. 计算 P@K, R@K, F1@K
    for k in K_list:
        precisions, recalls, f1s = [], [], []
        for pred, truth in zip(pred_list, gt_list):
            pred_k = set(pred[:k])
            truth_set = set(truth)
            if not truth_set: continue
            
            hits = len(pred_k & truth_set)
            prec = hits / len(pred_k) if len(pred_k) > 0 else 0.0
            rec = hits / len(truth_set)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            
        metrics[f'P@{k}'] = np.mean(precisions)
        metrics[f'R@{k}'] = np.mean(recalls)
        metrics[f'F1@{k}'] = np.mean(f1s)
        
    # 2. 计算 Pairs-F1, 多样性 和 时间履约率 (TTR)
    pairs_f1s, div_list, ttrs = [], [], []
    for pred, truth, pt, budget in zip(pred_list, gt_list, pred_times, target_budgets):
        # 成对准确率计算 (考虑行程内POI的相对关系)
        p_pairs = set((pred[i], pred[j]) for i in range(len(pred)) for j in range(i+1, len(pred)))
        t_pairs = set((truth[i], truth[j]) for i in range(len(truth)) for j in range(i+1, len(truth)))
        if t_pairs and p_pairs:
            hits = len(p_pairs & t_pairs)
            p = hits / len(p_pairs); r = hits / len(t_pairs)
            pairs_f1s.append(2*p*r/(p+r) if (p+r)>0 else 0.0)
        else:
            pairs_f1s.append(0.0)
            
        # 多样性计算 (行程中唯一类别的占比)
        cats = set([poi_details[pid]['cat'] for pid in pred if pid in poi_details])
        if len(pred) > 0: 
            div_list.append(len(cats) / len(pred))
            
        # 时间履约率计算 (预测消耗时间与时间预算的偏差)
        if budget > 0: 
            ttrs.append(max(0.0, 1.0 - abs(pt - budget) / budget))
            
    metrics['Pairs-F1'] = np.mean(pairs_f1s) if pairs_f1s else 0.0
    metrics['Diversity'] = np.mean(div_list) if div_list else 0.0
    metrics['TTR'] = np.mean(ttrs) if ttrs else 0.0
    return metrics