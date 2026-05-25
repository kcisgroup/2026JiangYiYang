import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import os

class TransEDataset(Dataset):
    """为TransE生成正负三元组"""
    def __init__(self, triplets, num_entities, neg_ratio=1):
        self.triplets = triplets
        self.num_entities = num_entities
        self.neg_ratio = neg_ratio
        self.entity_list = list(range(num_entities))

    def __len__(self): return len(self.triplets)

    def __getitem__(self, idx):
        h, r, t = self.triplets[idx]
        neg_triplets = []
        for _ in range(self.neg_ratio):
            neg_t = random.choice(self.entity_list)
            while neg_t == t: neg_t = random.choice(self.entity_list)
            neg_triplets.append((h, r, neg_t))
        return torch.LongTensor([h, r, t]), torch.LongTensor(neg_triplets)

class TransE(nn.Module):
    """基于平移距离的知识图谱嵌入模型"""
    def __init__(self, num_entities, num_relations, embed_dim, margin=1.0, norm=2):
        super().__init__()
        self.entity_embedding = nn.Embedding(num_entities, embed_dim)
        self.relation_embedding = nn.Embedding(num_relations, embed_dim)
        self.margin = margin
        self.norm = norm
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

    def forward(self, pos_triplets, neg_triplets):
        h_pos = self.entity_embedding(pos_triplets[:, 0])
        r_pos = self.relation_embedding(pos_triplets[:, 1])
        t_pos = self.entity_embedding(pos_triplets[:, 2])
        pos_scores = torch.norm(h_pos + r_pos - t_pos, p=self.norm, dim=1)

        h_neg = self.entity_embedding(neg_triplets[..., 0])
        r_neg = self.relation_embedding(neg_triplets[..., 1])
        t_neg = self.entity_embedding(neg_triplets[..., 2])
        neg_scores = torch.norm(h_neg + r_neg - t_neg, p=self.norm, dim=-1)

        loss = F.relu(self.margin + pos_scores.unsqueeze(1) - neg_scores).mean()
        return loss

def prepare_transe_data(df, user_map, venue_map, category_map):
    """构建统一实体映射并提取 User-POI 和 POI-Cat 的三元组关系"""
    num_users, num_pois, num_cats = len(user_map), len(venue_map), len(category_map)
    entity2id = {}
    for uid, u_int in user_map.items(): entity2id[f"u_{uid}"] = u_int
    for pid, p_int in venue_map.items(): entity2id[f"p_{pid}"] = num_users + p_int
    for cid, c_int in category_map.items(): entity2id[f"c_{cid}"] = num_users + num_pois + c_int
    num_entities = len(entity2id)
    
    relation2id = {'user_attends_poi': 0, 'poi_is_category': 1}
    triplets = []
    
    for _, row in tqdm(df[['user_id', 'geo_id']].drop_duplicates().iterrows(), desc="User-POI三元组"):
        h, t = entity2id.get(f"u_{row['user_id']}"), entity2id.get(f"p_{row['geo_id']}")
        if h is not None and t is not None: triplets.append((h, 0, t))
        
    for _, row in tqdm(df[['geo_id', 'venue_category_id']].drop_duplicates().iterrows(), desc="POI-Cat三元组"):
        h, t = entity2id.get(f"p_{row['geo_id']}"), entity2id.get(f"c_{row['venue_category_id']}")
        if h is not None and t is not None: triplets.append((h, 1, t))
        
    return triplets, entity2id, relation2id, num_entities, len(relation2id)

def run_transe_pretraining(triplets, num_entities, num_relations, embed_dim, device, epochs, batch_size, lr, margin, neg_ratio):
    dataset = TransEDataset(triplets, num_entities, neg_ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = TransE(num_entities, num_relations, embed_dim, margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for ep in range(1, epochs + 1):
        total_loss = 0
        for pos, neg in tqdm(dataloader, desc=f"TransE Ep {ep}/{epochs}", leave=False):
            optimizer.zero_grad()
            loss = model(pos.to(device), neg.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"TransE Ep {ep} Loss: {total_loss/len(dataloader):.4f}")
    return model.entity_embedding.weight.data.cpu()

def extract_and_save_embeddings(pretrained_embs, user_map, venue_map, category_map, num_users, num_venues_w_pad, num_cats_w_pad, dim, user_path, poi_path, cat_path):
    u_embs = torch.zeros(len(user_map), dim)
    for _, u_idx in user_map.items(): u_embs[u_idx] = pretrained_embs[u_idx]
    
    p_embs = torch.zeros(num_venues_w_pad, dim)
    for _, p_idx in venue_map.items(): p_embs[p_idx] = pretrained_embs[num_users + p_idx]
    
    c_embs = torch.zeros(num_cats_w_pad, dim)
    for _, c_idx in category_map.items(): c_embs[c_idx] = pretrained_embs[num_users + len(venue_map) + c_idx]
    
    torch.save(u_embs, user_path); torch.save(p_embs, poi_path); torch.save(c_embs, cat_path)
    return u_embs, p_embs, c_embs