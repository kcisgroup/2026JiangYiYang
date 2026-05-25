import time
import torch
import copy
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import random

from utils import info_nce_loss
from torch_geometric.data import Data

class GlobalStateCache:
    """用对象封装状态缓存，替代传统的 global 变量，保证多文件之间数据安全透传"""
    def __init__(self):
        self.all_train_user_reps_tensor = None
        self.train_user_id_to_idx_map = None
        self.train_user_ids_list = []

def update_user_representation_store(user_rep_model, dataloader, device, global_graph_data, state: GlobalStateCache):
    """提取训练集中所有用户的静态画像缓存，放入 state 以供后续读取"""
    if not state.train_user_ids_list: return
    user_rep_model.eval()
    if state.train_user_id_to_idx_map is None: state.train_user_id_to_idx_map = {uid: i for i, uid in enumerate(state.train_user_ids_list)} # type: ignore
    
    local_reps = torch.zeros(len(state.train_user_ids_list), user_rep_model.user_rep_final_dim, device=device)
    static_poi_reps = user_rep_model.global_poi_encoder(global_graph_data.to(device)) if user_rep_model.global_poi_encoder else None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="缓存用户画像", leave=False):
            if batch is None: continue
            for k in batch: 
                if isinstance(batch[k], (torch.Tensor, Data)): batch[k] = batch[k].to(device)
            enhanced_reps = user_rep_model(batch, global_poi_representations=static_poi_reps)
            for i, uid in enumerate(batch['user_ids']):
                if uid in state.train_user_id_to_idx_map: local_reps[state.train_user_id_to_idx_map[uid]] = enhanced_reps[i].detach() # type: ignore
    state.all_train_user_reps_tensor = local_reps # type: ignore

def train_epoch(model, optimizer, scheduler, dataloader_main, criterion_main, aux_cat_crit, config, fixed_params, device, global_graph_data, state: GlobalStateCache):
    """单个 Epoch 训练循环"""
    model.train()
    scaler = GradScaler('cuda', enabled=True) # type: ignore
    total_loss, total_m_loss, total_cl_loss, total_samples = 0.0, 0.0, 0.0, 0
    
    global_poi_reps = model(batch=None, mode='compute_global_reps', global_graph_data=global_graph_data, config=config) if model.use_global_graph else None
    cl_samples_pool = fixed_params.get('contrastive_samples', [])
    cl_weight = config.get('cl_loss_weight', 0.0)

    for batch in tqdm(dataloader_main, desc="训练中", leave=False):
        optimizer.zero_grad(set_to_none=True)
        for k in batch: 
            if isinstance(batch[k], torch.Tensor): batch[k] = batch[k].to(device, non_blocking=True)
        
        with autocast('cuda', enabled=True): # type: ignore
            loss_cl = torch.tensor(0.0, device=device)
            if cl_weight > 0 and cl_samples_pool and global_poi_reps is not None:
                cl_batch = random.sample(cl_samples_pool, k=min(2048, len(cl_samples_pool)))
                anchors = F.embedding(torch.tensor([i['anchor'] for i in cl_batch], device=device), global_poi_reps)
                positives = F.embedding(torch.tensor([i['positive'] for i in cl_batch], device=device), global_poi_reps)
                negatives = F.embedding(torch.tensor([neg for i in cl_batch for neg in i['negatives']], device=device), global_poi_reps).view(len(cl_batch), fixed_params['num_cl_negatives'], -1)
                loss_cl = info_nce_loss(anchors, positives, negatives)

            main_logits, aux_cat_logits, _ = model(batch, mode='trajectory', all_train_user_reps=state.all_train_user_reps_tensor, train_user_id_to_idx=state.train_user_id_to_idx_map, precomputed_global_reps=global_poi_reps)
            loss_m = criterion_main(main_logits, batch['target'])
            
            ac_mask = (batch['aux_cats'].view(-1) != fixed_params['cat_pad_idx'])
            loss_ac = aux_cat_crit(aux_cat_logits.reshape(-1, fixed_params['num_cats_w_pad'])[ac_mask], batch['aux_cats'].view(-1)[ac_mask]) if ac_mask.any() else torch.tensor(0.0, device=device)
            
            total = loss_m + config.get('aux_cat_loss_weight', 0.2) * loss_ac + cl_weight * loss_cl

        if not torch.isfinite(total): optimizer.zero_grad(set_to_none=True); continue

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip_norm', 1.0))
        scaler.step(optimizer); scaler.update(); scheduler.step()

        bs = batch['target'].size(0)
        total_samples += bs; total_loss += total.item() * bs; total_m_loss += loss_m.item() * bs; total_cl_loss += loss_cl.item() * bs
        
    return {'total': total_loss/total_samples, 'main': total_m_loss/total_samples, 'cl': total_cl_loss/total_samples}

def evaluate(model, dataloader, criterion, device, top_k, global_graph_data, config, state: GlobalStateCache):
    """在验证/测试集上计算损失及各 Acc@K 和 MRR"""
    model.eval()
    total_loss, total_samples, total_mrr = 0.0, 0, 0.0
    corrects = {k: 0 for k in top_k}
    global_poi_reps = model(batch=None, mode='compute_global_reps', global_graph_data=global_graph_data, config=config) if model.use_global_graph else None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中", leave=False):
            if batch is None: continue
            for k in batch: 
                if isinstance(batch[k], torch.Tensor): batch[k] = batch[k].to(device)
            
            with autocast('cuda', enabled=True): # type: ignore
                logits, _, _ = model(batch, mode='trajectory', all_train_user_reps=state.all_train_user_reps_tensor, train_user_id_to_idx=state.train_user_id_to_idx_map, precomputed_global_reps=global_poi_reps)
                loss = criterion(logits, batch['target'])
                
            bs = batch['target'].size(0)
            total_samples += bs; total_loss += loss.item() * bs
            
            _, top_idx = torch.topk(logits.float(), max(top_k), dim=1)
            for k_val in top_k: corrects[k_val] += (top_idx == batch['target'].unsqueeze(1))[:, :k_val].any(dim=1).sum().item()
            
            sorted_idx = torch.argsort(logits.float(), dim=1, descending=True)
            for i in range(bs):
                rank = (sorted_idx[i] == batch['target'][i]).nonzero()
                if rank.numel() > 0: total_mrr += 1.0 / (rank.item() + 1.0)
                
    return total_loss / total_samples, {f"Acc@{k}": v/total_samples for k,v in corrects.items()} | {'MRR': total_mrr/total_samples}