import os
import torch
import optuna
import pickle
from functools import partial
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import defaultdict
import random

from config import *
from utils import set_seed, load_precomputed_distances, get_cosine_schedule_with_warmup
from dataset import load_and_preprocess_data, split_users, TrajectoryDataset, main_task_collate_fn, UserFullHistoryDataset, user_full_history_gat_collate_fn, GlobalPOIGraphBuilder
from transe import prepare_transe_data, run_transe_pretraining, extract_and_save_embeddings
from models import POIRecommender
from trainer import GlobalStateCache, update_user_representation_store, train_epoch, evaluate

DEVICE = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

def create_optimizer_with_distinct_param_groups(model, lr_trial, user_rep_lr_scale_trial, global_graph_lr_scale, wd_trial):
    """根据模块名创建差异化学习率的Optimizer参数组"""
    all_params, param_groups = set(), []
    
    if hasattr(model, 'global_poi_encoder') and model.global_poi_encoder:
        gp = [p for p in model.global_poi_encoder.parameters() if p.requires_grad]
        all_params.update(gp)
        param_groups.append({'params': gp, 'lr': lr_trial * global_graph_lr_scale, 'weight_decay': wd_trial})
        
    if hasattr(model, 'user_rep_module') and model.user_rep_module:
        up = [p for p in model.user_rep_module.parameters() if p.requires_grad and p not in all_params]
        all_params.update(up)
        param_groups.append({'params': up, 'lr': lr_trial * user_rep_lr_scale_trial, 'weight_decay': wd_trial})

    other = [p for p in model.parameters() if p.requires_grad and p not in all_params]
    param_groups.append({'params': other, 'lr': lr_trial, 'weight_decay': wd_trial})
    return torch.optim.AdamW(param_groups)

def main():
    set_seed(SEED)
    print(f"使用设备: {DEVICE}")

    # 1. 基础预处理
    df_proc, user_map, venue_map, category_map, counts = load_and_preprocess_data(DATA_PATH)
    dist_lookup = load_precomputed_distances(DISTANCE_DATA_PATH, venue_map)
    train_ids, val_ids, test_ids = split_users(df_proc, VAL_USER_RATIO, TEST_USER_RATIO)
    
    # 2. 初始化缓存状态类
    state = GlobalStateCache()
    state.train_user_ids_list = list(train_ids)

    df_tr = df_proc[df_proc['user_id'].isin(train_ids)]
    train_ds = TrajectoryDataset(df_tr.groupby('user_id'), venue_map, MAX_SEQ_LENGTH, counts['venue_pad_idx'], counts['cat_pad_idx'], counts['num_categories'], DATA_AUG_MASKING_RATIO, True)
    val_ds = TrajectoryDataset(df_proc[df_proc['user_id'].isin(val_ids)].groupby('user_id'), venue_map, MAX_SEQ_LENGTH, counts['venue_pad_idx'], counts['cat_pad_idx'], counts['num_categories'], 0.0, False)
    
    collate_fn_main = partial(main_task_collate_fn, venue_pad_idx_g=counts['venue_pad_idx'], cat_pad_idx_g=counts['cat_pad_idx'], time_segment_pad_idx_g=TIME_SEGMENT_PAD_IDX)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_main, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_main, num_workers=4)

    top_k_poi_indices = torch.tensor([venue_map[p] for p in df_tr['geo_id'].value_counts().index if p in venue_map], dtype=torch.long)

    # 3. 提取特征图与TransE
    graph_builder = GlobalPOIGraphBuilder(venue_map, category_map, dist_lookup, df_tr)
    global_graph = graph_builder.build_global_graph()
    
    cl_samples = []
    cat_to_pois = defaultdict(list)
    for i in range(global_graph.num_nodes):  # type: ignore
        c = int(global_graph.x[i, 1].item()) # type: ignore
        if c != counts['cat_pad_idx']: cat_to_pois[c].append(i)
    for anchor in range(global_graph.num_nodes): # type: ignore
        ac = int(global_graph.x[anchor, 1].item()) # type: ignore
        if ac == counts['cat_pad_idx'] or len(cat_to_pois.get(ac,[])) < 2: continue
        pos = random.choice([x for x in cat_to_pois[ac] if x != anchor] or [anchor])
        negs = random.choices(range(global_graph.num_nodes), k=TRANSE_NEG_RATIO)  # type: ignore
        cl_samples.append({'anchor': anchor, 'positive': pos, 'negatives': negs})

    poi_embeds_pt, cat_embeds_pt = None, None
    if DO_TRANSE_PRETRAINING and not (os.path.exists(POI_EMBED_PATH) and os.path.exists(CAT_EMBED_PATH)):
        triplets, _, _, num_ent, num_rel = prepare_transe_data(df_tr, user_map, venue_map, category_map)
        transe_embs = run_transe_pretraining(triplets, num_ent, num_rel, TRANSE_EMBED_DIM, DEVICE, TRANSE_EPOCHS, TRANSE_BATCH_SIZE, TRANSE_LR, TRANSE_MARGIN, TRANSE_NEG_RATIO)
        _, poi_embeds_pt, cat_embeds_pt = extract_and_save_embeddings(transe_embs, user_map, venue_map, category_map, len(user_map), counts['num_venues_with_pad'], counts['num_categories_with_pad'], TRANSE_EMBED_DIM, USER_EMBED_PATH, POI_EMBED_PATH, CAT_EMBED_PATH)
    elif os.path.exists(POI_EMBED_PATH):
        poi_embeds_pt = torch.load(POI_EMBED_PATH, map_location=DEVICE, weights_only=True)
        cat_embeds_pt = torch.load(CAT_EMBED_PATH, map_location=DEVICE, weights_only=True)

    # 4. 模拟 Optuna / 最终训练逻辑
    fixed_params = {
        'num_venues_w_pad': counts['num_venues_with_pad'], 'venue_pad_idx': counts['venue_pad_idx'],
        'num_cats_w_pad': counts['num_categories_with_pad'], 'cat_pad_idx': counts['cat_pad_idx'], 'num_categories': counts['num_categories'],
        'num_time_segments_w_pad': NUM_TIME_SEGMENTS_W_PAD, 'time_segment_pad_idx': TIME_SEGMENT_PAD_IDX,
        'num_pairwise_time_bins': NUM_PAIRWISE_TIME_DIFF_BINS, 'max_seq_len': MAX_SEQ_LENGTH,
        'num_edge_time_bins_gat_w_pad': NUM_EDGE_TIME_BINS_W_PAD_GAT, 'edge_time_pad_idx_gat': EDGE_TIME_BIN_PAD_IDX_GAT,
        'num_edge_dist_bins_gat_w_pad': NUM_EDGE_DIST_BINS_W_PAD_GAT, 'edge_dist_pad_idx_gat': EDGE_DIST_BIN_PAD_IDX_GAT,
        'top_k_poi_indices_for_memory': top_k_poi_indices, 'contrastive_samples': cl_samples, 'num_cl_negatives': TRANSE_NEG_RATIO
    }

    print("\n--- 构建最终模型进行训练 ---")
    best_config = {'use_gat': True, 'use_global_graph': True, 'learning_rate': 1e-4, 'weight_decay': 1e-5} # 这里填Optuna跑出的最佳配置字典
    model = POIRecommender(best_config, **fixed_params, poi_embeds_pt=poi_embeds_pt, cat_embeds_pt=cat_embeds_pt).to(DEVICE)
    
    user_rep_ds = UserFullHistoryDataset(df_tr.groupby('user_id'), venue_map, category_map, MAX_SEQ_LENGTH, counts['venue_pad_idx'], counts['cat_pad_idx'], EDGE_TIME_DIFF_BINS_GAT, EDGE_DIST_BINS_GAT, dist_lookup, True, 0.0)
    user_rep_dl = DataLoader(user_rep_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=partial(user_full_history_gat_collate_fn, venue_pad_idx_g=counts['venue_pad_idx'], cat_pad_idx_g=counts['cat_pad_idx'], time_segment_pad_idx_g=TIME_SEGMENT_PAD_IDX), num_workers=4)

    update_user_representation_store(model.user_rep_module, user_rep_dl, DEVICE, global_graph, state)

    optimizer = create_optimizer_with_distinct_param_groups(model, 1e-4, 0.5, 0.5, 1e-5)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, len(train_loader) * 50)
    crit_main = nn.CrossEntropyLoss().to(DEVICE)
    aux_c_crit = nn.CrossEntropyLoss(ignore_index=counts['cat_pad_idx']).to(DEVICE)

    for ep in range(1, 51):
        tr_met = train_epoch(model, optimizer, scheduler, train_loader, crit_main, aux_c_crit, best_config, fixed_params, DEVICE, global_graph, state)
        val_loss, val_met = evaluate(model, val_loader, crit_main, DEVICE, (1, 5, 10), global_graph, best_config, state)
        print(f"Ep {ep}: TrL {tr_met['main']:.3f} | VaL {val_loss:.3f} | Acc@5 {val_met.get('Acc@5', 0)*100:.2f}%")

if __name__ == "__main__":
    main()