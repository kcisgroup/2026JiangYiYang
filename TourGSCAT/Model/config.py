import os
import torch
import warnings

# 忽略警告以保持控制台整洁
warnings.filterwarnings("ignore")

# ==============================================================================
# 0. 硬件与基础配置
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.getcwd()

# ==============================================================================
# 1. 全局配置字典
# ==============================================================================
CONFIG = {
    # ---------------- 路径配置 ----------------
    'data_path': os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_nyc_cleaned.csv'),
    'stage1_dir': os.path.join(BASE_DIR, 'MyModel/Stage2_Features/NYC'),
    'best_model_path': os.path.join(BASE_DIR, 'tour_gscat_stable_best.pth'), # 最佳模型保存路径
    
    # ---------------- 运行环境 ----------------
    'seed': 42,
    'device': DEVICE,
    
    # ---------------- 行程提取规则 ----------------
    'max_time_gap_hours': 8,   # 两个POI之间超过多少小时被截断为不同行程
    'min_trip_len': 3,         # 最小行程长度
    'max_trip_len': 10,        # 最大行程长度
    'avg_speed_kmh': 15.0,     # 物理先验：假设的平均通行速度 (km/h)
    
    # ---------------- 模型架构参数 ----------------
    'lstm_hidden_dim': 256,    # LSTM 隐藏层维度
    'dropout': 0.2,            # 随机失活概率
    
    # ---------------- 训练参数 ----------------
    'batch_size': 32,          # 批次大小
    'epochs': 200,             # 最大训练轮数
    'patience': 10,            # 早停耐心值
    'lr': 1e-3,                # 初始学习率
    'tf_start_ratio': 0.9,     # 初始 Teacher Forcing 比例
    'tf_end_ratio': 0.3,       # 最终 Teacher Forcing 比例
}