import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from config import *
from layers import *
from utils import bin_pairwise_time_diffs_torch

class MultiRelationalPOIEncoder(nn.Module):
    """全局多关系POI图的编码器"""
    def __init__(self, num_pois, num_categories_w_pad, cat_pad_idx, poi_embed_dim, cat_embed_dim, hidden_dim, num_layers, num_heads, dropout, poi_embeddings_pretrained=None, cat_embeddings_pretrained=None):
        super().__init__()
        self.num_pois, self.hidden_dim, self.num_relations = num_pois, hidden_dim, NUM_RELATION_TYPES
        self.poi_embedding = nn.Embedding(num_pois, poi_embed_dim)
        self.cat_embedding = nn.Embedding(num_categories_w_pad, cat_embed_dim, padding_idx=cat_pad_idx)
        
        if poi_embeddings_pretrained is not None: self.poi_embedding.weight.data[:num_pois] = poi_embeddings_pretrained[:num_pois]
        if cat_embeddings_pretrained is not None: self.cat_embedding.weight.data.copy_(cat_embeddings_pretrained)
            
        self.node_feature_encoder = nn.Sequential(
            nn.Linear(poi_embed_dim + cat_embed_dim + 3, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        
        self.relation_gats = nn.ModuleList()
        for _ in range(self.num_relations):
            rel_gats, current_dim = nn.ModuleList(), hidden_dim
            for layer_idx in range(num_layers):
                is_last = (layer_idx == num_layers - 1)
                heads = 1 if is_last else num_heads
                rel_gats.append(GATv2Conv(current_dim, hidden_dim, heads=heads, dropout=dropout, concat=not is_last, add_self_loops=True))
                current_dim = hidden_dim * heads if not is_last else hidden_dim
            self.relation_gats.append(rel_gats)
        
        self.relation_fusion = nn.Sequential(nn.Linear(hidden_dim * self.num_relations, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.final_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, global_graph_data, target_node_indices=None):
        if global_graph_data is None or global_graph_data.x.size(0) == 0:
            return torch.zeros(self.num_pois if target_node_indices is None else len(target_node_indices), self.hidden_dim, device=next(self.parameters()).device)
        
        node_features = global_graph_data.x
        poi_ids, cat_ids, coords_pop = node_features[:, 0].long(), node_features[:, 1].long(), node_features[:, 2:]
        node_h = self.node_feature_encoder(torch.cat([self.poi_embedding(poi_ids), self.cat_embedding(cat_ids), coords_pop], dim=-1))
        
        relation_outputs = []
        for rel_id in range(self.num_relations):
            rel_mask = (global_graph_data.edge_attr == rel_id)
            if rel_mask.sum() == 0:
                relation_outputs.append(torch.zeros_like(node_h))
                continue
            rel_edge_index = global_graph_data.edge_index[:, rel_mask]
            rel_h = node_h
            for gat_layer in self.relation_gats[rel_id]: rel_h = F.elu(gat_layer(rel_h, rel_edge_index)) # type: ignore
            relation_outputs.append(rel_h)
            
        final_h = self.final_norm(node_h + self.relation_fusion(torch.cat(relation_outputs, dim=-1))) if relation_outputs else node_h
        
        if target_node_indices is None:
            if global_graph_data.num_nodes >= self.num_pois - 1: 
                full_reps = torch.zeros(self.num_pois, self.hidden_dim, device=final_h.device)
                full_reps[global_graph_data.x[:, 0].long()] = final_h
                return full_reps
            return final_h
        return final_h.index_select(0, target_node_indices)

class SequenceEncoder(nn.Module):
    """序列特征提取器"""
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, num_venues_w_pad, venue_pad_idx, v_orig_dim, num_cats_w_pad, cat_pad_idx, c_orig_dim, tc_orig_dim, num_time_segments_w_pad, time_segment_pad_idx, ts_type_orig_dim, loc_embed_dim, pop_embed_dim, max_len, use_learnable_pos_enc, num_pairwise_time_bins, attn_gate_activation, attn_context_gate_input_type, poi_embeddings_pretrained=None, cat_embeddings_pretrained=None, global_poi_encoder=None, enable_cross_attention=False, max_cross_attn_memory_size=4096, top_k_poi_indices_for_memory=None):
        super().__init__()
        self.d_model, self.venue_pad_idx, self.cat_pad_idx = d_model, venue_pad_idx, cat_pad_idx
        self.use_learnable_pos_enc, self.use_global_graph = use_learnable_pos_enc, (global_poi_encoder is not None)
        self.enable_cross_attention, self.max_cross_attn_memory_size = enable_cross_attention, max_cross_attn_memory_size
        self.register_buffer('top_k_poi_indices', top_k_poi_indices_for_memory) if top_k_poi_indices_for_memory is not None else setattr(self, 'top_k_poi_indices', None)

        self.venue_embedding = nn.Embedding(num_venues_w_pad, v_orig_dim, padding_idx=venue_pad_idx)
        self.cat_embedding = nn.Embedding(num_cats_w_pad, c_orig_dim, padding_idx=cat_pad_idx)
        if poi_embeddings_pretrained is not None: self.venue_embedding.weight.data.copy_(poi_embeddings_pretrained)
        if cat_embeddings_pretrained is not None: self.cat_embedding.weight.data.copy_(cat_embeddings_pretrained)

        self.hour_encoding = nn.Embedding.from_pretrained(generate_cyclical_encoding_table(24, tc_orig_dim), freeze=False) # type: ignore
        self.time_segment_embedding = nn.Embedding(num_time_segments_w_pad, ts_type_orig_dim, padding_idx=time_segment_pad_idx)
        self.location_encoder = nn.Sequential(nn.Linear(2, loc_embed_dim * 2), nn.ReLU(), nn.Linear(loc_embed_dim * 2, loc_embed_dim))
        self.popularity_encoder = nn.Sequential(nn.Linear(1, pop_embed_dim * 2), nn.ReLU(), nn.Linear(pop_embed_dim * 2, pop_embed_dim))
        
        if self.use_global_graph: self.global_graph_proj = nn.Linear(global_poi_encoder.hidden_dim, d_model) # type: ignore
        self.venue_proj, self.cat_proj = nn.Linear(v_orig_dim, d_model), nn.Linear(c_orig_dim, d_model)
        self.hour_proj, self.ts_type_proj = nn.Linear(tc_orig_dim, d_model), nn.Linear(ts_type_orig_dim, d_model)
        self.loc_proj, self.pop_proj = nn.Linear(loc_embed_dim, d_model), nn.Linear(pop_embed_dim, d_model)
        
        self.w_venue, self.w_cat, self.w_hour, self.w_ts_type, self.w_loc, self.w_pop = [nn.Parameter(torch.ones(1)) for _ in range(6)]
        if self.use_global_graph: self.w_global = nn.Parameter(torch.ones(1))
        self.fusion_layernorm = nn.LayerNorm(d_model)

        self.pos_embedding = nn.Embedding(max_len, d_model) if use_learnable_pos_enc else PositionalEncoding(d_model, dropout, max_len)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.transformer_encoder_layers = nn.ModuleList([TimeAwareTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, 'relu', num_pairwise_time_bins, attn_gate_activation, attn_context_gate_input_type, enable_cross_attention) for _ in range(num_encoder_layers)]) if num_encoder_layers > 0 else nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(d_model) if num_encoder_layers > 0 else None

    def forward(self, venues, hours, time_segment_types, cats, lats, lons, popularities, raw_timestamps, padding_mask, global_poi_representations=None):
        B, L = venues.shape
        fused_terms = [
            self.w_venue * F.relu(self.venue_proj(self.venue_embedding(venues))),
            self.w_cat * F.relu(self.cat_proj(self.cat_embedding(cats))),
            self.w_hour * F.relu(self.hour_proj(self.hour_encoding(hours))),
            self.w_ts_type * F.relu(self.ts_type_proj(self.time_segment_embedding(time_segment_types))),
            self.w_loc * F.relu(self.loc_proj(self.location_encoder(torch.stack([lats, lons], dim=-1)))),
            self.w_pop * F.relu(self.pop_proj(self.popularity_encoder(popularities.unsqueeze(-1))))
        ]
        if self.use_global_graph and global_poi_representations is not None:
            fused_terms.append(self.w_global * F.relu(self.global_graph_proj(global_poi_representations.to(venues.device)[venues])))
        
        fused_features = self.fusion_layernorm(sum(fused_terms))
        tf_input = self.dropout_layer(fused_features + self.pos_embedding(torch.arange(L, device=venues.device).unsqueeze(0).expand(B, -1))) if self.use_learnable_pos_enc else self.pos_encoder_fixed(fused_features)
        
        binned_pairwise_tdiffs = bin_pairwise_time_diffs_torch((raw_timestamps.unsqueeze(2) - raw_timestamps.unsqueeze(1)) / 60.0, PAIRWISE_TIME_DIFF_BINS)
        
        cross_attn_memory = None
        if self.enable_cross_attention and self.use_global_graph and global_poi_representations is not None and self.top_k_poi_indices is not None:
            cross_attn_memory = self.global_graph_proj(global_poi_representations.to(venues.device))[self.top_k_poi_indices[:self.max_cross_attn_memory_size]].unsqueeze(0).expand(B, -1, -1)
            
        current_src = tf_input
        for layer in self.transformer_encoder_layers: current_src = layer(current_src, binned_pairwise_tdiffs, src_key_padding_mask=padding_mask, cross_attention_memory=cross_attn_memory)
        return self.encoder_norm(current_src) if self.encoder_norm else current_src

class UserRepresentationModule(nn.Module):
    """联合序列与图提取用户级画像特征"""
    def __init__(self, sequence_encoder, num_total_venues, venue_embed_dim_gat_node_id, num_total_categories_w_pad, cat_embed_dim_gat_node_cat, cat_pad_idx_gat_node, loc_input_dim_gat_node, loc_embed_dim_gat_node, pop_input_dim_gat_node, pop_embed_dim_gat_node, transe_embed_dim, num_edge_time_bins_gat_w_pad, edge_time_embed_dim_gat, edge_time_pad_idx_gat, num_edge_dist_bins_gat_w_pad, edge_dist_embed_dim_gat, edge_dist_pad_idx_gat, gat_hidden_dims_list, gat_num_heads_list, gat_output_dim, gat_dropout, use_gat, user_rep_fusion_type, user_rep_final_dim, poi_embeddings_pretrained=None, cat_embeddings_pretrained=None, global_poi_encoder=None):
        super().__init__()
        self.sequence_encoder, self.global_poi_encoder = sequence_encoder, global_poi_encoder
        self.transformer_d_model = sequence_encoder.d_model
        self.use_gat, self.gat_output_dim_internal = use_gat, max(0, gat_output_dim) if use_gat else 0
        
        if self.use_gat:
            self.gat_node_poi_id_embedding = nn.Embedding(num_total_venues, venue_embed_dim_gat_node_id)
            self.gat_node_poi_cat_embedding = nn.Embedding(num_total_categories_w_pad, cat_embed_dim_gat_node_cat, padding_idx=cat_pad_idx_gat_node)
            if cat_embeddings_pretrained is not None: self.gat_node_poi_cat_embedding.weight.data.copy_(cat_embeddings_pretrained)
            self.gat_node_poi_loc_encoder = nn.Sequential(nn.Linear(loc_input_dim_gat_node, max(1,loc_embed_dim_gat_node*2)), nn.ReLU(), nn.Linear(max(1,loc_embed_dim_gat_node*2), loc_embed_dim_gat_node))
            self.gat_node_poi_pop_encoder = nn.Sequential(nn.Linear(pop_input_dim_gat_node, max(1, pop_embed_dim_gat_node*2)), nn.ReLU(), nn.Linear(max(1, pop_embed_dim_gat_node*2), pop_embed_dim_gat_node))
            
            init_dim = venue_embed_dim_gat_node_id + transe_embed_dim + cat_embed_dim_gat_node_cat + loc_embed_dim_gat_node + pop_embed_dim_gat_node
            self.gat_initial_node_proj = nn.Sequential(nn.Linear(init_dim, init_dim), nn.ReLU(), nn.LayerNorm(init_dim))
            
            self.gat_edge_time_embedding = nn.Embedding(num_edge_time_bins_gat_w_pad, edge_time_embed_dim_gat, padding_idx=edge_time_pad_idx_gat)
            self.gat_edge_dist_embedding = nn.Embedding(num_edge_dist_bins_gat_w_pad, edge_dist_embed_dim_gat, padding_idx=edge_dist_pad_idx_gat)
            self.gat_edge_feature_dim_used = edge_time_embed_dim_gat + edge_dist_embed_dim_gat
            
            self.gat_layers, self.gat_norms = nn.ModuleList(), nn.ModuleList()
            current_dim = init_dim
            for i, (h_dim, n_h) in enumerate(zip(gat_hidden_dims_list, gat_num_heads_list)):
                is_last = (i == len(gat_hidden_dims_list) - 1)
                self.gat_layers.append(GATv2Conv(current_dim, h_dim // n_h, heads=n_h, dropout=gat_dropout, concat=(not is_last), add_self_loops=True, edge_dim=self.gat_edge_feature_dim_used))
                current_dim = h_dim
                self.gat_norms.append(nn.LayerNorm(current_dim))
            self.gat_output_projection = nn.Linear(current_dim, self.gat_output_dim_internal) if current_dim != self.gat_output_dim_internal else nn.Identity()

        self.user_rep_fusion_type, self.user_rep_final_dim = user_rep_fusion_type, user_rep_final_dim or self.transformer_d_model
        fuse_in_dim = self.transformer_d_model + self.gat_output_dim_internal
        
        if not self.use_gat or self.user_rep_fusion_type == 'seq_only': self.user_rep_fusion_layer = nn.Linear(self.transformer_d_model, self.user_rep_final_dim)
        elif self.user_rep_fusion_type == 'add':
            self.proj_seq_for_add = nn.Linear(self.transformer_d_model, self.user_rep_final_dim)
            self.proj_graph_for_add = nn.Linear(self.gat_output_dim_internal, self.user_rep_final_dim)
        elif self.user_rep_fusion_type == 'concat_mlp':
            self.user_rep_fusion_layer = nn.Sequential(nn.Linear(fuse_in_dim, fuse_in_dim // 2), nn.ReLU(), nn.Dropout(sequence_encoder.dropout_layer.p), nn.Linear(fuse_in_dim // 2, self.user_rep_final_dim))
        elif self.user_rep_fusion_type == 'gated':
            self.gate_fc_user_rep = nn.Linear(fuse_in_dim, self.user_rep_final_dim)
            self.proj_seq_for_gated = nn.Linear(self.transformer_d_model, self.user_rep_final_dim)
            self.proj_graph_for_gated = nn.Linear(self.gat_output_dim_internal, self.user_rep_final_dim)

    def forward(self, batch_dict, global_poi_representations=None):
        tf_out = self.sequence_encoder(batch_dict['venues_seq'], batch_dict['hours_seq'], batch_dict['time_segment_types_seq'], batch_dict['cats_seq'], batch_dict['lats_seq'], batch_dict['lons_seq'], batch_dict['popularities_seq'], batch_dict['raw_timestamps_seq'], batch_dict['padding_mask_seq'], global_poi_representations)
        valid_lens = batch_dict['seq_lens'].float().clamp(min=1).unsqueeze(1).to(tf_out.device)
        user_rep_seq = (tf_out * (~batch_dict['padding_mask_seq']).unsqueeze(-1).float()).sum(dim=1) / valid_lens
        
        user_rep_graph = torch.zeros(tf_out.size(0), self.gat_output_dim_internal, device=user_rep_seq.device)
        if self.use_gat and 'gat_batch' in batch_dict and batch_dict['gat_batch'].num_graphs > 0:
            gb = batch_dict['gat_batch']
            init_h = self.gat_initial_node_proj(torch.cat([self.gat_node_poi_id_embedding(gb.x_node_venue_ids), self.sequence_encoder.venue_embedding(gb.x_node_venue_ids), self.gat_node_poi_cat_embedding(gb.x_node_cat_ids), self.gat_node_poi_loc_encoder(gb.x_node_locs), self.gat_node_poi_pop_encoder(gb.x_node_popularity)], dim=-1))
            edge_attr = torch.cat([self.gat_edge_time_embedding(gb.edge_attr[:, 0]), self.gat_edge_dist_embedding(gb.edge_attr[:, 1])], dim=-1) if gb.edge_attr.numel() > 0 else None
            h_in = init_h
            for i, layer in enumerate(self.gat_layers): h_in = self.gat_norms[i](F.elu((h_in + layer(h_in, gb.edge_index, edge_attr=edge_attr)) if h_in.shape == layer(h_in, gb.edge_index, edge_attr=edge_attr).shape else layer(h_in, gb.edge_index, edge_attr=edge_attr)))
            user_rep_graph = global_mean_pool(self.gat_output_projection(h_in), gb.batch)

        if not self.use_gat or self.user_rep_fusion_type == 'seq_only': return self.user_rep_fusion_layer(user_rep_seq)
        elif self.user_rep_fusion_type == 'add': return self.proj_seq_for_add(user_rep_seq) + self.proj_graph_for_add(user_rep_graph)
        elif self.user_rep_fusion_type == 'concat_mlp': return self.user_rep_fusion_layer(torch.cat([user_rep_seq, user_rep_graph], dim=-1))
        elif self.user_rep_fusion_type == 'gated':
            gate = torch.sigmoid(self.gate_fc_user_rep(torch.cat([user_rep_seq, user_rep_graph], dim=-1)))
            return gate * self.proj_seq_for_gated(user_rep_seq) + (1 - gate) * self.proj_graph_for_gated(user_rep_graph)

class TrajectoryTransformer(nn.Module):
    """主任务：融合当前轨迹上下文、用户偏好、社交相似用户画像进行预测"""
    def __init__(self, sequence_encoder, user_rep_dim, num_categories_g, pooling_strategy, dropout, use_similar_user_fusion, num_similar_users_k, similar_user_rep_dim_scale, similar_user_aggregation_temp, similar_user_fusion_gate_type):
        super().__init__()
        self.sequence_encoder, self.pooling_strategy = sequence_encoder, pooling_strategy
        self.d_model = sequence_encoder.d_model
        self.attention_pooling = AttentionPooling(self.d_model) if pooling_strategy == 'attention' else None
        
        self.use_similar_user_fusion, self.num_similar_users_k, self.similar_user_aggregation_temp = use_similar_user_fusion, num_similar_users_k, similar_user_aggregation_temp
        self.similar_user_fusion_gate_type = similar_user_fusion_gate_type
        
        final_in_dim = self.d_model
        if use_similar_user_fusion:
            self.similar_user_fusion_dim = max(1, int(self.d_model * similar_user_rep_dim_scale))
            self.fusion_with_similar_users_mlp = nn.Sequential(nn.Linear(self.d_model + user_rep_dim, self.d_model * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(self.d_model * 2, self.similar_user_fusion_dim))
            if similar_user_fusion_gate_type == 'simple_sigmoid': self.similar_user_gate_fc = nn.Linear(self.d_model, 1)
            elif similar_user_fusion_gate_type == 'contextual_mlp': self.similar_user_gate_mlp = nn.Sequential(nn.Linear(self.d_model + user_rep_dim, self.d_model), nn.ReLU(), nn.Linear(self.d_model, 2))
            elif similar_user_fusion_gate_type == 'final_concat_gate': self.final_concat_gate_fc = nn.Linear(self.d_model + self.similar_user_fusion_dim, 1)
            final_in_dim += self.similar_user_fusion_dim

        self.output_layer = nn.Linear(final_in_dim, num_categories_g)
        self.aux_cat_output_layer = nn.Linear(self.d_model, sequence_encoder.cat_embedding.num_embeddings)

    def forward(self, venues, hours, time_segment_types, cats, lats, lons, popularities, raw_timestamps, padding_mask, seq_lens, current_user_ids_batch=None, all_train_user_reps_tensor=None, train_user_id_to_idx_map=None, global_poi_representations=None):
        tf_out = self.sequence_encoder(venues, hours, time_segment_types, cats, lats, lons, popularities, raw_timestamps, padding_mask, global_poi_representations)
        aux_cat_logits = self.aux_cat_output_layer(tf_out)
        
        if self.pooling_strategy == 'last': pooled_output = tf_out.gather(1, (seq_lens - 1).view(-1, 1, 1).expand(-1, -1, self.d_model).clamp(min=0)).squeeze(1)
        elif self.pooling_strategy == 'max': pooled_output = tf_out.masked_fill(padding_mask.unsqueeze(-1), -1e9).max(dim=1)[0].masked_fill_(tf_out.masked_fill(padding_mask.unsqueeze(-1), -1e9).max(dim=1)[0] == -1e9, 0.0)
        elif self.pooling_strategy == 'attention' and self.attention_pooling: pooled_output = self.attention_pooling(tf_out, padding_mask)
        else: pooled_output = (tf_out * (~padding_mask).unsqueeze(-1).float()).sum(dim=1) / seq_lens.float().clamp(min=1).unsqueeze(1).to(tf_out.device)
        
        final_rep = pooled_output
        if self.use_similar_user_fusion and current_user_ids_batch is not None and all_train_user_reps_tensor is not None and train_user_id_to_idx_map is not None:
            agg_sim_reps = []
            for i, uid in enumerate(current_user_ids_batch):
                if uid not in train_user_id_to_idx_map: agg_sim_reps.append(torch.zeros(self.d_model, device=pooled_output.device)); continue
                u_idx = train_user_id_to_idx_map[uid]
                u_rep = all_train_user_reps_tensor[u_idx]
                sims = F.cosine_similarity(u_rep.unsqueeze(0), all_train_user_reps_tensor, dim=1)
                sims[u_idx] = -float('inf')
                top_k = min(self.num_similar_users_k, len(sims)-1)
                top_scores, top_idx = torch.topk(sims, k=top_k)
                weights = F.softmax(top_scores / self.similar_user_aggregation_temp, dim=0)
                agg_sim_reps.append(torch.sum(all_train_user_reps_tensor[top_idx] * weights.unsqueeze(1), dim=0))
            
            agg_tensor = torch.stack(agg_sim_reps)
            fused_info = self.fusion_with_similar_users_mlp(torch.cat([pooled_output, agg_tensor], dim=1))
            
            if self.similar_user_fusion_gate_type == 'simple_sigmoid': fused_info *= torch.sigmoid(self.similar_user_gate_fc(pooled_output))
            elif self.similar_user_fusion_gate_type == 'contextual_mlp': 
                logits2 = self.similar_user_gate_mlp(torch.cat([pooled_output, agg_tensor], dim=1))
                fused_info *= torch.sigmoid(logits2[:, 1:2])
            elif self.similar_user_fusion_gate_type == 'final_concat_gate': fused_info *= torch.sigmoid(self.final_concat_gate_fc(torch.cat([pooled_output, fused_info], dim=1).detach()))
            
            final_rep = torch.cat([pooled_output, fused_info], dim=1)
            if self.similar_user_fusion_gate_type == 'contextual_mlp': final_rep[:, :self.d_model] *= torch.sigmoid(logits2[:, 0:1]) # type: ignore

        return self.output_layer(final_rep), aux_cat_logits, None

class POIRecommender(nn.Module):
    """总封装模型：根据参数选择性调用 GNN Encoder / User Representation / Trajectory Transformer"""
    def __init__(self, config, num_venues_w_pad, venue_pad_idx, num_cats_w_pad, cat_pad_idx, num_categories, num_time_segments_w_pad, time_segment_pad_idx, num_pairwise_time_bins, max_seq_len, num_edge_time_bins_gat_w_pad, edge_time_pad_idx_gat, num_edge_dist_bins_gat_w_pad, edge_dist_pad_idx_gat, poi_embeds_pt=None, cat_embeds_pt=None, top_k_poi_indices_for_memory=None):
        super().__init__()
        self.use_gat = config.get("use_gat", True)
        self.use_global_graph = config.get("use_global_graph", True)

        self.global_poi_encoder = MultiRelationalPOIEncoder(num_venues_w_pad, num_cats_w_pad, cat_pad_idx, TRANSE_EMBED_DIM, TRANSE_EMBED_DIM, config.get("global_graph_hidden_dim", DEFAULT_GLOBAL_GRAPH_HIDDEN_DIM), config.get("global_graph_num_layers", DEFAULT_GLOBAL_GRAPH_NUM_LAYERS), config.get("global_graph_num_heads", DEFAULT_GLOBAL_GRAPH_NUM_HEADS), config.get("global_graph_dropout", DEFAULT_GLOBAL_GRAPH_DROPOUT), poi_embeds_pt, cat_embeds_pt) if self.use_global_graph else None
        
        self.user_rep_sequence_encoder = SequenceEncoder(config.get("user_rep_tf_d_model", DEFAULT_USER_REP_TF_D_MODEL), config.get("user_rep_tf_nhead", 4), config.get("user_rep_tf_layers", 2), config.get("user_rep_tf_d_model", DEFAULT_USER_REP_TF_D_MODEL) * 2, config.get("user_rep_tf_dropout", 0.1), num_venues_w_pad, venue_pad_idx, TRANSE_EMBED_DIM, num_cats_w_pad, cat_pad_idx, TRANSE_EMBED_DIM, 32, num_time_segments_w_pad, time_segment_pad_idx, 8, 16, 8, max_seq_len, True, num_pairwise_time_bins, 'sigmoid', 'qk', poi_embeds_pt, cat_embeds_pt, self.global_poi_encoder)
        
        self.user_rep_module = UserRepresentationModule(self.user_rep_sequence_encoder, num_venues_w_pad, config.get("gat_node_id_embed_dim", 32), num_cats_w_pad, config.get("gat_node_cat_embed_dim", 16), cat_pad_idx, 2, 16, 1, 16, TRANSE_EMBED_DIM, num_edge_time_bins_gat_w_pad, 8, edge_time_pad_idx_gat, num_edge_dist_bins_gat_w_pad, 8, edge_dist_pad_idx_gat, config.get("gat_h_dim_list", [64]), config.get("gat_n_heads_list", [4]), 64, 0.1, self.use_gat, config.get('user_rep_fusion_type', 'add'), config.get("traj_tf_d_model", 128), poi_embeds_pt, cat_embeds_pt, self.global_poi_encoder)
        
        self.traj_tf_sequence_encoder = SequenceEncoder(config.get("traj_tf_d_model", 128), config.get("traj_tf_nhead", 4), config.get("traj_tf_layers", 2), config.get("traj_tf_d_model", 128) * 2, config.get("dropout_shared", 0.15), num_venues_w_pad, venue_pad_idx, TRANSE_EMBED_DIM, num_cats_w_pad, cat_pad_idx, TRANSE_EMBED_DIM, 32, num_time_segments_w_pad, time_segment_pad_idx, 16, 16, 8, max_seq_len, True, num_pairwise_time_bins, 'sigmoid', 'qk', poi_embeds_pt, cat_embeds_pt, self.global_poi_encoder, config.get('traj_tf_enable_cross_attention', False), 4096, top_k_poi_indices_for_memory)
        
        self.traj_transformer = TrajectoryTransformer(self.traj_tf_sequence_encoder, self.user_rep_module.user_rep_final_dim, num_categories, config.get('traj_tf_pooling_strategy', 'attention'), config.get("dropout_shared", 0.15), config.get("traj_tf_use_similar_user", True), config.get("traj_tf_num_similar_k", 10), 0.5, 1.0, 'contextual_mlp')

    def forward(self, batch, mode, all_train_user_reps=None, train_user_id_to_idx=None, global_graph_data=None, precomputed_global_reps=None, config=None):
        if mode == 'compute_global_reps':
            from torch_geometric.utils import k_hop_subgraph
            from torch_geometric.data import Data
            from tqdm import tqdm
            if not self.use_global_graph: return None
            all_reps = torch.zeros(self.global_poi_encoder.num_pois, self.global_poi_encoder.hidden_dim, device=next(self.parameters()).device) # type: ignore
            graph_cpu = global_graph_data.to('cpu') # type: ignore
            for i in tqdm(range(0, graph_cpu.num_nodes, 2048), desc="预计算全图节点表征", leave=False):
                nodes = torch.arange(graph_cpu.num_nodes)[i:i+2048]
                sub_n, sub_e, map_idx, e_mask = k_hop_subgraph(nodes, self.global_poi_encoder.num_layers, graph_cpu.edge_index, relabel_nodes=True, num_nodes=graph_cpu.num_nodes) # type: ignore
                sub_data = Data(x=graph_cpu.x[sub_n], edge_index=sub_e, edge_attr=graph_cpu.edge_attr[e_mask]).to(next(self.parameters()).device) # type: ignore
                all_reps[nodes] = self.global_poi_encoder(sub_data, target_node_indices=map_idx.to(next(self.parameters()).device)) # type: ignore
            return all_reps
        elif mode == 'user_rep': return self.user_rep_module(batch, global_poi_representations=precomputed_global_reps)
        elif mode == 'trajectory': return self.traj_transformer(batch['venues'], batch['hours'], batch['time_segment_types'], batch['cats'], batch['lats'], batch['lons'], batch['popularities'], batch['raw_timestamps'], batch['padding_mask'], batch['seq_lens'], batch.get('user_ids'), all_train_user_reps, train_user_id_to_idx, precomputed_global_reps)