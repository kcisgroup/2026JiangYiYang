import os
import random
import torch
from torch.utils.data import DataLoader

from config import CONFIG
from utils import set_seed
from dataset import load_and_extract_itineraries, build_global_spatiotemporal_tensors, ItineraryDataset, collate_fn
from models import TourGSCAT
from trainer import train_eval_loop

def main():
    # 1. 基础环境设置
    set_seed(CONFIG['seed'])
    print(f"使用设备: {CONFIG['device']}")
    
    mapping_path = os.path.join(CONFIG['stage1_dir'], 'id_mappings.pkl')
    user_emb_path = os.path.join(CONFIG['stage1_dir'], 'pretrained_user_embs.pt')
    poi_emb_path = os.path.join(CONFIG['stage1_dir'], 'pretrained_poi_embs.pt')
    
    # 2. 数据抽取与时空属性建构
    itineraries, num_u, num_v_pad, pad_idx, poi_info, avg_dwell = load_and_extract_itineraries(
        CONFIG['data_path'], mapping_path
    )
    
    trans_mat, dwell_ten, trans_prob_mat, pop_ten, norm_trans = build_global_spatiotemporal_tensors(
        poi_info, avg_dwell, num_v_pad, pad_idx, itineraries
    )
    
    # 3. 按照 8:1:1 划分行程数据集
    random.shuffle(itineraries)
    t_end = int(len(itineraries) * 0.8)
    v_end = t_end + int(len(itineraries) * 0.1)
    
    train_ds = ItineraryDataset(itineraries[:t_end], pad_idx)
    val_ds = ItineraryDataset(itineraries[t_end:v_end], pad_idx)
    test_ds = ItineraryDataset(itineraries[v_end:], pad_idx)
    
    # 借助 lambda 函数动态传入 pad_idx 处理对齐
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                          collate_fn=lambda x: collate_fn(x, pad_idx))
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                        collate_fn=lambda x: collate_fn(x, pad_idx))
    test_dl = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                         collate_fn=lambda x: collate_fn(x, pad_idx))
    
    # 4. 读入 GSCAT 的预训练参数并初始化模型
    print("\n>>> 3. 加载 GSCAT 预训练表征与初始化 TourGSCAT(Stable) 模型...")
    pretrained_user_emb = torch.load(user_emb_path, map_location=CONFIG['device'])
    pretrained_poi_emb = torch.load(poi_emb_path, map_location=CONFIG['device'])
    
    model = TourGSCAT(
        num_users=num_u, 
        num_venues_with_pad=num_v_pad, 
        pretrained_user_emb=pretrained_user_emb, 
        pretrained_poi_emb=pretrained_poi_emb,    
        transit_matrix=trans_mat, 
        dwell_tensor=dwell_ten, 
        transition_matrix=trans_prob_mat, 
        pop_tensor=pop_ten, 
        norm_transit=norm_trans, 
        pad_idx=pad_idx,
        lstm_hidden=CONFIG['lstm_hidden_dim'], 
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    # 5. 执行循环
    print("\n>>> 4. 启动训练...")
    train_eval_loop(model, train_dl, val_dl, test_dl, num_v_pad, pad_idx, poi_info, CONFIG)

if __name__ == "__main__":
    main()