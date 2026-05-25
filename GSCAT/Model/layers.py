import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """经典的正弦/余弦位置编码"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        num_timescales = d_model // 2
        div_term = torch.exp(torch.arange(0, num_timescales, dtype=torch.float) * (-math.log(10000.0) / num_timescales))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0:2*num_timescales:2] = torch.sin(position * div_term)
        pe[0, :, 1:2*num_timescales:2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.dropout(x + self.pe[:, :x.size(1), :])

class AttentionPooling(nn.Module):
    """注意力池化操作，提取变长序列的固定维度表示"""
    def __init__(self, input_dim):
        super().__init__()
        self.attention_net = nn.Sequential(nn.Linear(input_dim, input_dim//2), nn.Tanh(), nn.Linear(input_dim//2, 1))
    def forward(self, x, mask):
        attn_logits = self.attention_net(x).squeeze(-1)
        attn_logits.masked_fill_(mask, -float('inf'))
        attn_weights = F.softmax(attn_logits, dim=1)
        return torch.sum(x * (attn_weights.unsqueeze(-1) + 1e-9), dim=1)

class TimeAwareScaledDotProductAttention(nn.Module):
    """融入时间差偏置的点积注意力"""
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, time_bias=None, key_padding_mask=None, attn_mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if time_bias is not None: scores = scores + time_bias
        if key_padding_mask is not None: scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        if attn_mask is not None:
            mask_exp = attn_mask.unsqueeze(0).unsqueeze(0) if attn_mask.dim()==2 else attn_mask.unsqueeze(1) if attn_mask.dim()==3 else attn_mask
            if mask_exp.dtype==torch.bool: scores = scores.masked_fill(mask_exp, float('-inf'))
            else: scores = scores + mask_exp
        p_attn = F.softmax(scores, dim=-1)
        p_attn = torch.nan_to_num(p_attn, nan=0.0)
        return torch.matmul(self.dropout(p_attn), v), p_attn

class TimeAwareMultiheadAttention_ContextGated(nn.Module):
    """带有上下文门控机制的时间感知多头自注意力层"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, num_pairwise_time_bins=None, bias_in_linear=True, gate_activation='sigmoid', context_gate_input_type='qk'):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * self.num_heads != self.embed_dim: self.embed_dim = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias_in_linear)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias_in_linear)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias_in_linear)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias_in_linear)
        self.scaled_attn = TimeAwareScaledDotProductAttention(dropout=dropout)
        
        self.time_diff_bias_embedding = nn.Embedding(num_pairwise_time_bins, num_heads) # type: ignore
        nn.init.zeros_(self.time_diff_bias_embedding.weight)
        self.time_bias_layernorm = nn.LayerNorm(num_heads)
        
        self.gate_activation_type, self.context_gate_input_type = gate_activation, context_gate_input_type
        gate_dim = self.head_dim * 2 + (num_heads if context_gate_input_type=='qkt' else 0)
        self.contextual_gate_fc_sigmoid = nn.Linear(max(1, gate_dim), num_heads)
        nn.init.xavier_uniform_(self.contextual_gate_fc_sigmoid.weight, gain=nn.init.calculate_gain('sigmoid'))
        if self.gate_activation_type == 'tanh_sigmoid':
            self.contextual_gate_fc_tanh = nn.Linear(max(1, gate_dim), num_heads)
            nn.init.xavier_uniform_(self.contextual_gate_fc_tanh.weight, gain=nn.init.calculate_gain('tanh'))

    def forward(self, query, key, value, binned_pairwise_time_diffs, key_padding_mask=None, attn_mask=None):
        B, L_q, E = query.shape; _, L_k, _ = key.shape
        q = self.q_proj(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)

        tb_raw = self.time_diff_bias_embedding(binned_pairwise_time_diffs)
        tb_norm = self.time_bias_layernorm(tb_raw)

        q_exp = q.permute(0,2,1,3).unsqueeze(2).expand(-1,-1,L_k,-1,-1)
        k_exp = k.permute(0,2,1,3).unsqueeze(1).expand(-1,L_q,-1,-1,-1)
        gate_in = torch.cat([q_exp, k_exp], dim=-1)
        if self.context_gate_input_type == 'qkt':
            tb_exp = tb_raw.unsqueeze(3).expand(-1,-1,-1,self.num_heads,-1)
            gate_in = torch.cat([gate_in, tb_exp], dim=-1)
            
        mean_feat = gate_in.mean(dim=3)
        logits_sig = self.contextual_gate_fc_sigmoid(mean_feat)
        gate_vals = torch.sigmoid(logits_sig)
        if self.gate_activation_type == 'tanh_sigmoid': gate_vals *= torch.tanh(self.contextual_gate_fc_tanh(mean_feat))
        elif self.gate_activation_type == 'identity': gate_vals = logits_sig

        time_bias = (tb_norm * gate_vals).permute(0, 3, 1, 2)
        attn_out, attn_w = self.scaled_attn(q, k, v, time_bias, key_padding_mask, attn_mask)
        return self.out_proj(attn_out.transpose(1,2).contiguous().view(B, L_q, E)), attn_w

class TimeAwareTransformerEncoderLayer(nn.Module):
    """Transformer Encoder的修改版，集成时间感知和跨注意力机制"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation_fn="relu", num_pairwise_time_bins=None, mha_gate_activation='sigmoid', mha_context_gate_input_type='qk', has_cross_attention=False):
        super().__init__()
        self.d_model, self.has_cross_attention = d_model, has_cross_attention
        self.self_attn = TimeAwareMultiheadAttention_ContextGated(d_model, nhead, dropout, num_pairwise_time_bins, gate_activation=mha_gate_activation, context_gate_input_type=mha_context_gate_input_type)
        self.norm1, self.dropout1 = nn.LayerNorm(d_model), nn.Dropout(dropout)

        if self.has_cross_attention:
            self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
            self.norm_cross, self.dropout_cross = nn.LayerNorm(d_model), nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2, self.dropout2 = nn.LayerNorm(d_model), nn.Dropout(dropout)
        self.activation = F.relu if activation_fn == "relu" else F.gelu

    def forward(self, src, binned_pairwise_time_diffs, src_mask=None, src_key_padding_mask=None, cross_attention_memory=None, memory_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, binned_pairwise_time_diffs, key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        src = self.norm1(src + self.dropout1(src2))

        if self.has_cross_attention and cross_attention_memory is not None:
            src3, _ = self.cross_attn(query=src, key=cross_attention_memory, value=cross_attention_memory, key_padding_mask=memory_key_padding_mask)
            src = self.norm_cross(src + self.dropout_cross(src3))

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))