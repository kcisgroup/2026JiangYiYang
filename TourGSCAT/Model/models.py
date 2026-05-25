import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class TourGSCAT(nn.Module):
    """
    基于时间感知与自适应时空约束的轨迹生成模型 (Time-Encoding + Adaptive-STC)
    """
    def __init__(self, num_users, num_venues_with_pad, pretrained_user_emb, pretrained_poi_emb, 
                 transit_matrix, dwell_tensor, transition_matrix, pop_tensor, norm_transit, 
                 pad_idx, lstm_hidden, dropout):
        super(TourGSCAT, self).__init__()
        self.pad_idx = pad_idx
        
        # 嵌入维度识别
        user_embed_dim = pretrained_user_emb.size(1) if pretrained_user_emb is not None else 128
        poi_embed_dim = pretrained_poi_emb.size(1) if pretrained_poi_emb is not None else 128
        
        # ----------- 嵌入层 -----------
        self.user_emb = nn.Embedding(num_users, user_embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        if pretrained_user_emb is not None:
            if pretrained_user_emb.size(0) == num_users:
                self.user_emb.weight.data.copy_(pretrained_user_emb)
                
        self.poi_emb = nn.Embedding(num_venues_with_pad, poi_embed_dim, padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.poi_emb.weight)
        self.poi_emb.weight.data[pad_idx].zero_()
        if pretrained_poi_emb is not None:
            if pretrained_poi_emb.size(0) == num_venues_with_pad:
                self.poi_emb.weight.data.copy_(pretrained_poi_emb)
            elif pretrained_poi_emb.size(0) == num_venues_with_pad - 1:
                self.poi_emb.weight.data[:-1].copy_(pretrained_poi_emb)
                
        # 允许微调
        self.user_emb.weight.requires_grad = True
        self.poi_emb.weight.requires_grad = True
        
        # ----------- 上下文编码器 -----------
        time_emb_dim = poi_embed_dim // 4
        self.time_emb = nn.Embedding(48, time_emb_dim) # 一天分成48个时间槽(半小时一个)
        
        budget_dim = poi_embed_dim // 2
        self.budget_enc = nn.Sequential(
            nn.Linear(1, budget_dim), nn.ReLU(), nn.Linear(budget_dim, budget_dim)
        )
        
        # ----------- 循环网络核心 -----------
        lstm_input_dim = user_embed_dim + poi_embed_dim + budget_dim + time_emb_dim
        self.state_proj = nn.Linear(lstm_input_dim, lstm_hidden)
        self.lstm_cell = nn.LSTMCell(input_size=lstm_hidden, hidden_size=lstm_hidden)
        
        self.W_q = nn.Linear(lstm_hidden, poi_embed_dim)
        self.W_k = nn.Linear(poi_embed_dim, poi_embed_dim)
        
        # 自适应上下文门控，用于平衡语义意图与物理阻力
        self.adaptive_gate = nn.Sequential(
            nn.Linear(lstm_hidden + user_embed_dim, poi_embed_dim // 2), 
            nn.ReLU(), 
            nn.Linear(poi_embed_dim // 2, 1), 
            nn.Sigmoid()
        )
        
        # 可学习的物理先验系数
        self.dist_alpha = nn.Parameter(torch.tensor([1.0]))  
        self.trans_beta = nn.Parameter(torch.tensor([1.0]))   
        self.pop_gamma = nn.Parameter(torch.tensor([1.0]))    
        
        # 注册不可导的张量为 Buffer
        self.register_buffer('transit_matrix', transit_matrix)  
        self.register_buffer('dwell_tensor', dwell_tensor)
        self.register_buffer('norm_transit', norm_transit)      
        self.register_buffer('transition_matrix', transition_matrix)
        self.register_buffer('pop_tensor', pop_tensor)
        
        self.dropout = nn.Dropout(dropout)

    def _compute_mask(self, current_pois, budget_remains, visited_mask):
        """计算时间超限的 Mask 和总耗时矩阵"""
        B = current_pois.size(0)
        num_v = self.dwell_tensor.size(0)
        transit_t = self.transit_matrix[current_pois] 
        dwell_t = self.dwell_tensor.unsqueeze(0).expand(B, -1) 
        req_time = transit_t + dwell_t
        
        time_mask = req_time > budget_remains.expand(-1, num_v)
        final_mask = time_mask | visited_mask
        final_mask[:, self.pad_idx] = False 
        return final_mask, req_time

    def _get_fused_scores(self, q, current_pois):
        """合并模型学习到的语义注意力与环境给定的物理先验"""
        # 1. 计算语义得分
        semantic_scores = torch.matmul(q, self.W_k(self.poi_emb.weight).transpose(0, 1)) 
        
        # 2. 计算物理先验约束得分 (距离惩罚 + 共现奖励 + 热度奖励)
        physic_scores = - F.relu(self.dist_alpha) * self.norm_transit[current_pois] \
                        + F.relu(self.trans_beta) * self.transition_matrix[current_pois] \
                        + F.relu(self.pop_gamma) * self.pop_tensor.unsqueeze(0)
        return semantic_scores, physic_scores

    def forward(self, user_ids, start_pois, start_times, time_budgets, target_seqs, tf_ratio=0.5):
        """训练前向传播，包含 Teacher Forcing 逻辑"""
        B = user_ids.size(0)
        device = user_ids.device
        num_v = self.poi_emb.num_embeddings
        max_len = target_seqs.size(1)
        
        u_e = self.user_emb(user_ids)
        h_t = torch.zeros(B, self.lstm_cell.hidden_size, device=device)
        c_t = torch.zeros(B, self.lstm_cell.hidden_size, device=device)
        
        curr_pois = start_pois
        budgets = time_budgets.unsqueeze(1).float()
        curr_time_mins = start_times.unsqueeze(1).float()
        
        visited = torch.zeros(B, num_v, dtype=torch.bool, device=device)
        visited.scatter_(1, curr_pois.unsqueeze(1), True)
        
        logits_list = []
        
        for t in range(max_len):
            p_e = self.poi_emb(curr_pois)
            b_e = self.budget_enc(budgets)
            
            time_bins = ((curr_time_mins % 1440) / 30).long()
            t_e = self.time_emb(time_bins.squeeze(1))
            
            lstm_in = self.dropout(F.relu(self.state_proj(torch.cat([u_e, p_e, b_e, t_e], dim=-1))))
            h_t, c_t = self.lstm_cell(lstm_in, (h_t, c_t))
            
            q = self.W_q(h_t)
            semantic_scores, physic_scores = self._get_fused_scores(q, curr_pois)
            
            # 动态融合门控
            gate = self.adaptive_gate(torch.cat([h_t, u_e], dim=-1)) 
            scores = gate * semantic_scores + (1 - gate) * physic_scores
            
            mask, req_time = self._compute_mask(curr_pois, budgets, visited)
            
            # Ground truth的节点必须保持不被Mask掉，否则Loss算无穷大
            gt_next = target_seqs[:, t].unsqueeze(1)
            mask.scatter_(1, gt_next, False) 
            
            scores.masked_fill_(mask, -1e9)
            logits_list.append(scores.unsqueeze(1))
            
            # Teacher Forcing 判断
            if random.random() < tf_ratio:
                next_pois = target_seqs[:, t]
            else:
                next_pois = scores.argmax(dim=1)
                
            actual_time = req_time.gather(1, next_pois.unsqueeze(1))
            budgets = budgets - actual_time
            curr_time_mins = curr_time_mins + actual_time
            visited.scatter_(1, next_pois.unsqueeze(1), True)
            curr_pois = next_pois
            
        return torch.cat(logits_list, dim=1)

    def generate(self, user_ids, start_pois, start_times, time_budgets, max_len=10, temperature=0.8):
        """推理阶段生成，自回归采样直到预算耗尽或达到长度限制"""
        self.eval()
        with torch.no_grad():
            B = user_ids.size(0)
            device = user_ids.device
            num_v = self.poi_emb.num_embeddings
            
            u_e = self.user_emb(user_ids)
            h_t = torch.zeros(B, self.lstm_cell.hidden_size, device=device)
            c_t = torch.zeros(B, self.lstm_cell.hidden_size, device=device)
            
            curr_pois = start_pois
            budgets = time_budgets.unsqueeze(1).float()
            curr_time_mins = start_times.unsqueeze(1).float()
            initial_budgets = budgets.clone()
            
            visited = torch.zeros(B, num_v, dtype=torch.bool, device=device)
            visited.scatter_(1, curr_pois.unsqueeze(1), True)
            
            gen_seqs = [[] for _ in range(B)]
            
            for t in range(max_len):
                p_e = self.poi_emb(curr_pois)
                b_e = self.budget_enc(budgets)
                time_bins = ((curr_time_mins % 1440) / 30).long()
                t_e = self.time_emb(time_bins.squeeze(1))
                
                lstm_in = F.relu(self.state_proj(torch.cat([u_e, p_e, b_e, t_e], dim=-1)))
                h_t, c_t = self.lstm_cell(lstm_in, (h_t, c_t))
                
                q = self.W_q(h_t)
                semantic_scores, physic_scores = self._get_fused_scores(q, curr_pois)
                gate = self.adaptive_gate(torch.cat([h_t, u_e], dim=-1))
                scores = gate * semantic_scores + (1 - gate) * physic_scores
                
                mask, req_time = self._compute_mask(curr_pois, budgets, visited)
                scores.masked_fill_(mask, -1e9)
                
                scaled_scores = scores / temperature
                next_pois = scaled_scores.argmax(dim=1)
                
                actual_time = req_time.gather(1, next_pois.unsqueeze(1))
                budgets = budgets - actual_time
                curr_time_mins = curr_time_mins + actual_time
                visited.scatter_(1, next_pois.unsqueeze(1), True)
                curr_pois = next_pois
                
                for i in range(B):
                    idx = next_pois[i].item()
                    if idx != self.pad_idx: gen_seqs[i].append(idx)
                        
            gen_times = (initial_budgets - budgets).squeeze(1).cpu().tolist()
        return gen_seqs, gen_times