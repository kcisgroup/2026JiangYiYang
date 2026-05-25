import os
import numpy as np
import pandas as pd

# ==============================================================================
# 0. 基础路径配置
# ==============================================================================
BASE_DIR = os.getcwd()  # 当前工作目录
DATA_PATH = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_tky_cleaned.csv')  # 清洗后的签到数据
DISTANCE_DATA_PATH = os.path.join(BASE_DIR, 'MyModel/dataset_process/distance_df_tky.csv')  # POI间距离数据
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'MyModel/GSCAT/TKY/OptunaRuns')  # Optuna及模型保存目录
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'transformer_poi_cate_model_best_optuna.pth') 

PRETRAINED_EMBED_DIR = os.path.join(BASE_DIR, 'MyModel/GSCAT/TKY/PretrainedEmbeddings')  # 预训练嵌入目录
os.makedirs(PRETRAINED_EMBED_DIR, exist_ok=True)
POI_EMBED_PATH = os.path.join(PRETRAINED_EMBED_DIR, 'poi_embeddings_transe.pt')
CAT_EMBED_PATH = os.path.join(PRETRAINED_EMBED_DIR, 'cat_embeddings_transe.pt')
USER_EMBED_PATH = os.path.join(PRETRAINED_EMBED_DIR, 'user_embeddings_transe.pt')

# ==============================================================================
# 1. TransE 预训练配置
# ==============================================================================
DO_TRANSE_PRETRAINING = True  # 是否执行或加载TransE预训练
TRANSE_EMBED_DIM = 128        # TransE嵌入维度
TRANSE_EPOCHS = 50            # 训练轮数
TRANSE_BATCH_SIZE = 4096      # 批次大小
TRANSE_LR = 0.001             # 学习率
TRANSE_MARGIN = 1.0           # 损失函数的margin
TRANSE_NEG_RATIO = 10         # 正负样本比例

# ==============================================================================
# 2. 多关系图构建配置
# ==============================================================================
GLOBAL_GRAPH_DISTANCE_THRESHOLD = 2.5      # (km) 地理关系边的距离阈值
GLOBAL_GRAPH_COOCCUR_MIN_USERS = 3         # 共现关系边的最少共现用户数
GLOBAL_GRAPH_TOP_POPULAR_RATIO = 0.1       # 热门关系的POI比例
GLOBAL_GRAPH_SAME_CATEGORY_ENABLED = True  # 是否构建同类别边

RELATION_TYPES = {
    'GEOGRAPHICAL': 0,    # 地理相邻
    'SAME_CATEGORY': 1,   # 同类别
    'CO_OCCURRENCE': 2,   # 共现
    'POPULARITY': 3       # 热门POI连接
}
NUM_RELATION_TYPES = len(RELATION_TYPES)

# ==============================================================================
# 3. 序列数据与分箱参数
# ==============================================================================
MAX_SEQ_LENGTH = 50          # 序列最大长度
VAL_USER_RATIO = 0.1         # 验证集比例
TEST_USER_RATIO = 0.1        # 测试集比例
DATA_AUG_MASKING_RATIO = 0.1 # 序列随机Masking比例 (数据增强)

# 时间段规则映射 (将连续时间戳映射为离散的语义时间类别)
TIME_SEGMENT_RULES = [
    ("WORKDAY_MORNING_PEAK", lambda wd,h:0<=wd<=4 and 7<=h<=9),      
    ("WORKDAY_DAYTIME",      lambda wd,h:0<=wd<=4 and 10<=h<=16),     
    ("WORKDAY_EVENING_PEAK", lambda wd,h:0<=wd<=4 and 17<=h<=19),     
    ("WORKDAY_NIGHT",        lambda wd,h:0<=wd<=4 and (20<=h<=23 or 0<=h<=1)),
    ("WEEKEND_DAYTIME",      lambda wd,h:5<=wd<=6 and 9<=h<=17),      
    ("WEEKEND_NIGHT",        lambda wd,h:5<=wd<=6 and (18<=h<=23 or 0<=h<=1)),
    ("LATE_NIGHT_DEEP",      lambda wd,h:2<=h<=6)                       
]
TIME_SEGMENT_CATEGORIES = {name: i for i,(name,_) in enumerate(TIME_SEGMENT_RULES)}
NUM_TIME_SEGMENTS = len(TIME_SEGMENT_CATEGORIES)
TIME_SEGMENT_PAD_IDX = NUM_TIME_SEGMENTS
NUM_TIME_SEGMENTS_W_PAD = NUM_TIME_SEGMENTS + 1

# GAT使用的边属性：时间差分箱（分钟）
EDGE_TIME_DIFF_BINS_GAT = [-1, 0, 5, 30, 120, 360, 1440, 1440*3, 1440*7, np.inf]
NUM_EDGE_TIME_BINS_GAT = len(EDGE_TIME_DIFF_BINS_GAT) - 1
EDGE_TIME_BIN_PAD_IDX_GAT = NUM_EDGE_TIME_BINS_GAT
NUM_EDGE_TIME_BINS_W_PAD_GAT = NUM_EDGE_TIME_BINS_GAT + 1

# GAT使用的边属性：地理距离分箱（km）
EDGE_DIST_BINS_GAT = [0, 0.1, 0.5, 1, 2, 5, 10, 20, np.inf]
NUM_EDGE_DIST_BINS_GAT = len(EDGE_DIST_BINS_GAT) - 1
EDGE_DIST_BIN_PAD_IDX_GAT = NUM_EDGE_DIST_BINS_GAT
NUM_EDGE_DIST_BINS_W_PAD_GAT = NUM_EDGE_DIST_BINS_GAT + 1

# Transformer内部：成对时间差分箱
PAIRWISE_TIME_DIFF_BINS = [-float('inf'), -1440*7, -1440, -360, -120, -30, -5, 0, 5, 30, 120, 360, 1440, 1440*7, float('inf')]
NUM_PAIRWISE_TIME_DIFF_BINS = len(PAIRWISE_TIME_DIFF_BINS) - 1

# ==============================================================================
# 4. 训练与模型默认参数 (当不用Optuna时生效)
# ==============================================================================
BATCH_SIZE = 128
EPOCHS_PER_TRIAL = 20
OPTUNA_N_TRIALS = 0  # 为0则只使用Best Config跑一遍
PATIENCE = 5
WARMUP_STEPS = 1000
SEED = 42

DEFAULT_GAT_NODE_ID_EMBED_DIM = 32
DEFAULT_GAT_NODE_CAT_EMBED_DIM = 16
DEFAULT_GAT_NODE_LOC_EMBED_DIM = 16
DEFAULT_EDGE_TIME_EMBED_DIM_GAT = 8
DEFAULT_EDGE_DIST_EMBED_DIM_GAT = 8
DEFAULT_GAT_HIDDEN_DIMS = [64] 
DEFAULT_GAT_NUM_HEADS_LIST = [4]
DEFAULT_GAT_OUTPUT_DIM = 64 
DEFAULT_GAT_DROPOUT = 0.1

DEFAULT_USER_REP_TF_D_MODEL = 64
DEFAULT_USER_REP_FUSION_TYPE = 'concat_mlp'

DEFAULT_GLOBAL_GRAPH_HIDDEN_DIM = 128
DEFAULT_GLOBAL_GRAPH_NUM_LAYERS = 2
DEFAULT_GLOBAL_GRAPH_NUM_HEADS = 4
DEFAULT_GLOBAL_GRAPH_DROPOUT = 0.1