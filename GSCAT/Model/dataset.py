import torch
import pandas as pd
import numpy as np
import random
import copy
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from config import *
from utils import get_time_segment_id, bin_series_to_int

# ==============================================================================
# 1. 基础数据加载与划分
# ==============================================================================
def load_and_preprocess_data(data_path: str):
    """加载原始CSV，解析时间、计算热度、归一化经纬度，并返回所需的映射表"""
    print("加载并预处理数据...")
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df.dropna(subset=['time'], inplace=True)
    df['hour'] = df['time'].dt.hour
    df['weekday'] = df['time'].dt.weekday
    df = df.sort_values(by=['user_id', 'time'])
    
    # POI热度对数平滑
    poi_visit_counts = df['geo_id'].value_counts()
    df['poi_popularity_log'] = np.log1p(df['geo_id'].map(poi_visit_counts).fillna(0))
    
    # 经纬度归一化
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max() 
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    df['latitude_norm'] = (df['latitude']-lat_min)/(lat_max-lat_min) if lat_max>lat_min else 0.0
    df['longitude_norm'] = (df['longitude']-lon_min)/(lon_max-lon_min) if lon_max>lon_min else 0.0
    
    # 类别映射
    df['venue_category_id'] = df['venue_category_id'].fillna('UNK_CAT').astype(str)
    category_map = {cid: i for i, cid in enumerate(df['venue_category_id'].unique())}
    df['integer_venue_category_id'] = df['venue_category_id'].map(category_map)
    
    # POI/用户映射
    df['geo_id'] = df['geo_id'].astype(str)
    venue_map = {vid: i for i, vid in enumerate(df['geo_id'].unique())}
    user_map = {uid: i for i, uid in enumerate(df['user_id'].unique())}

    # 将原有global变量转化为字典返回
    counts = {
        'num_categories': len(category_map),
        'cat_pad_idx': len(category_map),
        'num_categories_with_pad': len(category_map) + 1,
        'num_venues': len(venue_map),
        'venue_pad_idx': len(venue_map),
        'num_venues_with_pad': len(venue_map) + 1,
    }
    print(f"数据预处理完成。用户:{len(user_map)}, 类别:{counts['num_categories']}, POI:{counts['num_venues']}")
    return df, user_map, venue_map, category_map, counts

def split_users(df: pd.DataFrame, val_ratio: float, test_ratio: float):
    """按用户粒度进行训练集、验证集、测试集的划分"""
    all_user_ids = df['user_id'].unique()
    np.random.shuffle(all_user_ids)
    n_users = len(all_user_ids)
    n_val = int(n_users * val_ratio)
    n_test = int(n_users * test_ratio)
    
    val_ids = set(all_user_ids[:n_val])
    test_ids = set(all_user_ids[n_val:n_val+n_test])
    train_ids = set(all_user_ids[n_val+n_test:])
    if not train_ids and n_users > 0: 
        train_ids = val_ids if not val_ids else set(all_user_ids)
    return train_ids, val_ids, test_ids

# ==============================================================================
# 2. PyTorch Dataset 与 Collate Functions
# ==============================================================================
class TrajectoryDataset(Dataset):
    """用于主任务（预测下一个POI）的序列数据集"""
    def __init__(self, data_grouped, venue_map_g, max_len, venue_pad_idx_g, cat_pad_idx_g, num_categories_g, masking_ratio=0.0, is_train=False):
        self.data = []
        self.max_len = max_len
        self.venue_map = venue_map_g
        self.venue_pad_idx = venue_pad_idx_g
        self.cat_pad_idx = cat_pad_idx_g
        self.num_categories = num_categories_g
        self.masking_ratio = masking_ratio
        self.is_train = is_train

        for user_id, group in tqdm(data_grouped, desc="创建主任务序列"):
            group = group.sort_values('time')
            if len(group) < 2: continue 
                
            venues_ids = [self.venue_map.get(str(v), self.venue_pad_idx) for v in group['geo_id'].tolist()]
            hours = group['hour'].tolist()
            time_segment_type_ids = [get_time_segment_id(ts) for ts in group['time']]
            cats = [int(c) if pd.notna(c) and 0<=int(c)<self.num_categories else self.cat_pad_idx for c in group['integer_venue_category_id'].tolist()]
            norm_lats = group['latitude_norm'].tolist()
            norm_lons = group['longitude_norm'].tolist()
            popularities = group['poi_popularity_log'].tolist()
            raw_timestamps_utc = group['time'].apply(lambda x:x.timestamp() if pd.notna(x) else 0.0).tolist()
            
            for i in range(1, len(venues_ids)):
                input_end_idx = i
                start_idx = max(0, input_end_idx - self.max_len)
                
                seq_v_orig = venues_ids[start_idx:input_end_idx]
                target_cat = cats[i]
                
                if not seq_v_orig or target_cat == self.cat_pad_idx: continue

                seq_v = list(seq_v_orig) 
                seq_c = list(cats[start_idx:input_end_idx])
                seq_h = list(hours[start_idx:input_end_idx])
                seq_ts_type = list(time_segment_type_ids[start_idx:input_end_idx])
                seq_nl = list(norm_lats[start_idx:input_end_idx])
                seq_nlo = list(norm_lons[start_idx:input_end_idx])
                seq_pop = list(popularities[start_idx:input_end_idx])
                seq_raw_ts = list(raw_timestamps_utc[start_idx:input_end_idx])
                
                # 随机Masking数据增强
                if self.is_train and self.masking_ratio > 0:
                    seq_len = len(seq_v)
                    num_to_mask = int(seq_len * self.masking_ratio)
                    if num_to_mask > 0:
                        mask_indices = random.sample(range(seq_len), k=num_to_mask)
                        for mask_idx in mask_indices:
                            seq_v[mask_idx] = self.venue_pad_idx
                            seq_c[mask_idx] = self.cat_pad_idx
                            seq_h[mask_idx] = 0
                            seq_ts_type[mask_idx] = TIME_SEGMENT_PAD_IDX
                            seq_nl[mask_idx] = 0.0
                            seq_nlo[mask_idx] = 0.0
                            seq_pop[mask_idx] = 0.0
                            seq_raw_ts[mask_idx] = 0.0

                self.data.append({
                    'user_id': user_id, 'venues': seq_v, 'hours': seq_h,
                    'time_segment_types': seq_ts_type, 'cats': seq_c,
                    'lats': seq_nl, 'lons': seq_nlo, 'popularities': seq_pop,
                    'raw_timestamps': seq_raw_ts, 'target': target_cat,
                    'aux_cats': copy.deepcopy(cats[start_idx:input_end_idx]), 
                    'aux_venues': copy.deepcopy(seq_v_orig)
                })

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def main_task_collate_fn(batch, venue_pad_idx_g, cat_pad_idx_g, time_segment_pad_idx_g):
    """序列张量的补齐 (Padding) 与合并"""
    keys = batch[0].keys() if batch else []
    batch_data = {k:[] for k in keys}
    seq_lens = []
    
    valid_batch = [item for item in batch if item and 'venues' in item and item['venues']]
    if not valid_batch: return None
    
    max_len_in_batch = max(len(item['venues']) for item in valid_batch)
    if max_len_in_batch == 0: return None
    
    for item in valid_batch:
        seq_len = len(item['venues'])
        seq_lens.append(seq_len)
        pad_len = max_len_in_batch - seq_len
        
        for k in ['venues','cats','aux_cats','aux_venues']:
            pad_val = venue_pad_idx_g if 'venue' in k else cat_pad_idx_g
            batch_data[k].append(torch.tensor(item[k] + [pad_val]*pad_len, dtype=torch.long))
        
        batch_data['hours'].append(torch.tensor(item['hours'] + [0]*pad_len, dtype=torch.long))
        batch_data['time_segment_types'].append(torch.tensor(item['time_segment_types'] + [time_segment_pad_idx_g]*pad_len, dtype=torch.long))
        
        for k in ['lats','lons','popularities','raw_timestamps']:
            dtype = torch.double if k=='raw_timestamps' else torch.float
            batch_data[k].append(torch.tensor(item[k] + [0.0]*pad_len, dtype=dtype))
        
        for k in ['user_id','target']: batch_data[k].append(item[k])
    
    stacked_data={}
    tensor_keys=['venues','hours','time_segment_types','cats','lats','lons','popularities','raw_timestamps','aux_cats','aux_venues']
    
    for k in tensor_keys:
        if k in batch_data and batch_data[k]: 
            stacked_data[k] = torch.stack(batch_data[k])
    
    stacked_data['user_ids'] = batch_data['user_id']
    stacked_data['target'] = torch.tensor(batch_data['target'], dtype=torch.long)
    stacked_data['seq_lens'] = torch.tensor(seq_lens,dtype=torch.long)
    stacked_data['padding_mask'] = (stacked_data['venues']==venue_pad_idx_g)
    
    return stacked_data

class UserFullHistoryDataset(Dataset):
    """为用户画像模块构建数据集，既提取时间序列，也构建局部动态图(供GAT使用)"""
    def __init__(self, df_user_histories_grouped, venue_map_g, category_map_g, max_len_traj, venue_pad_idx_g, cat_pad_idx_g, edge_time_bins_list, edge_dist_bins_list, distance_lookup_g, is_train=False, edge_dropout_rate=0.0):
        self.user_data_list = []
        self.max_len_traj = max_len_traj
        self.venue_map = venue_map_g
        self.category_map = category_map_g
        self.venue_pad_idx = venue_pad_idx_g
        self.cat_pad_idx = cat_pad_idx_g
        self.edge_time_bins_list = edge_time_bins_list
        self.edge_dist_bins_list = edge_dist_bins_list
        self.edge_time_labels = list(range(len(edge_time_bins_list)-1)) if edge_time_bins_list else []
        self.edge_dist_labels = list(range(len(edge_dist_bins_list)-1)) if edge_dist_bins_list else []
        self.distance_lookup = distance_lookup_g if distance_lookup_g is not None else {}
        self.is_train = is_train
        self.edge_dropout_rate = edge_dropout_rate

        for user_id, group in tqdm(df_user_histories_grouped, desc="构建用户图结构与序列"):
            group = group.sort_values('time').tail(self.max_len_traj)
            if len(group) < 2: continue

            # 1. 序列特征
            seq_venues_ids = [self.venue_map.get(str(v), self.venue_pad_idx) for v in group['geo_id'].tolist()]
            hours_seq = group['hour'].tolist()
            time_segment_types_seq = [get_time_segment_id(ts) for ts in group['time']]
            cats_seq = [int(c) if pd.notna(c) and 0 <= c < len(self.category_map) else self.cat_pad_idx for c in group['integer_venue_category_id'].tolist()]
            lats_seq = group['latitude_norm'].tolist()
            lons_seq = group['longitude_norm'].tolist()
            popularities_seq = group['poi_popularity_log'].tolist()
            raw_timestamps_seq = group['time'].apply(lambda x: x.timestamp() if pd.notna(x) else 0.0).tolist()
            raw_times_pd_seq = group['time'].tolist()

            # 2. 图节点
            unique_pois_df_in_group = group.drop_duplicates(subset=['geo_id'])
            node_global_venue_ids, node_global_cat_ids = [], []
            node_norm_lats, node_norm_lons, node_popularities = [], [], []
            map_global_venue_idx_to_local = {}
            current_local_node_idx = 0
            
            for _, poi_row in unique_pois_df_in_group.iterrows():
                global_venue_idx = self.venue_map.get(str(poi_row['geo_id']), self.venue_pad_idx)
                if global_venue_idx == self.venue_pad_idx: continue 

                if global_venue_idx not in map_global_venue_idx_to_local:
                    map_global_venue_idx_to_local[global_venue_idx] = current_local_node_idx
                    node_global_venue_ids.append(global_venue_idx)
                    cat_val_int = poi_row['integer_venue_category_id']
                    node_global_cat_ids.append(int(cat_val_int) if pd.notna(cat_val_int) and 0 <= cat_val_int < len(self.category_map) else self.cat_pad_idx)
                    node_norm_lats.append(poi_row['latitude_norm'])
                    node_norm_lons.append(poi_row['longitude_norm'])
                    node_popularities.append(poi_row.get('poi_popularity_log', 0.0))
                    current_local_node_idx += 1
            if not node_global_venue_ids: continue

            # 3. 图边属性
            edge_list_src_local_idx, edge_list_dst_local_idx = [], []
            edge_time_attr_list, edge_dist_attr_list = [], []

            for i in range(len(group) - 1):
                row_u, row_v = group.iloc[i], group.iloc[i+1]
                u_gid = self.venue_map.get(str(row_u['geo_id']), self.venue_pad_idx)
                v_gid = self.venue_map.get(str(row_v['geo_id']), self.venue_pad_idx)

                if u_gid != self.venue_pad_idx and v_gid != self.venue_pad_idx and u_gid != v_gid:
                    if u_gid in map_global_venue_idx_to_local and v_gid in map_global_venue_idx_to_local:
                        u_local = map_global_venue_idx_to_local[u_gid]
                        v_local = map_global_venue_idx_to_local[v_gid]
                        edge_list_src_local_idx.append(u_local)
                        edge_list_dst_local_idx.append(v_local)

                        time_diff_min = (raw_times_pd_seq[i+1] - raw_times_pd_seq[i]).total_seconds() / 60.0
                        time_bin = bin_series_to_int(pd.Series([time_diff_min]), self.edge_time_bins_list, self.edge_time_labels, EDGE_TIME_BIN_PAD_IDX_GAT)[0]
                        edge_time_attr_list.append(time_bin)

                        dist_km = -1.0 
                        if self.distance_lookup: 
                            if u_gid in self.distance_lookup and v_gid in self.distance_lookup[u_gid]:
                                dist_km = self.distance_lookup[u_gid][v_gid]
                            elif v_gid in self.distance_lookup and u_gid in self.distance_lookup[v_gid]: 
                                dist_km = self.distance_lookup[v_gid][u_gid]
                        dist_bin = bin_series_to_int(pd.Series([dist_km if dist_km >=0 else -1.0]), self.edge_dist_bins_list, self.edge_dist_labels, EDGE_DIST_BIN_PAD_IDX_GAT)[0]
                        edge_dist_attr_list.append(dist_bin)
            
            if self.is_train and self.edge_dropout_rate > 0 and edge_list_src_local_idx:
                num_edges = len(edge_list_src_local_idx)
                perm = np.random.permutation(num_edges)
                num_edges_to_keep = int(num_edges * (1.0 - self.edge_dropout_rate))
                indices_to_keep = perm[:num_edges_to_keep]
                
                edge_list_src_local_idx = [edge_list_src_local_idx[i] for i in indices_to_keep]
                edge_list_dst_local_idx = [edge_list_dst_local_idx[i] for i in indices_to_keep]
                edge_time_attr_list = [edge_time_attr_list[i] for i in indices_to_keep]
                edge_dist_attr_list = [edge_dist_attr_list[i] for i in indices_to_keep]
            
            gat_data_obj = Data(
                x_node_venue_ids=torch.tensor(node_global_venue_ids, dtype=torch.long),
                x_node_cat_ids=torch.tensor(node_global_cat_ids, dtype=torch.long),
                x_node_locs=torch.tensor(list(zip(node_norm_lats, node_norm_lons)), dtype=torch.float),
                x_node_popularity=torch.tensor(node_popularities, dtype=torch.float).unsqueeze(1),
                edge_index=torch.tensor([edge_list_src_local_idx, edge_list_dst_local_idx], dtype=torch.long) if edge_list_src_local_idx else torch.empty(2,0,dtype=torch.long),
                edge_attr=torch.tensor(list(zip(edge_time_attr_list, edge_dist_attr_list)), dtype=torch.long) if edge_list_src_local_idx else torch.empty(0,2,dtype=torch.long)
            )

            self.user_data_list.append({
                'user_id': user_id,
                'venues_seq': seq_venues_ids, 'hours_seq': hours_seq, 'time_segment_types_seq': time_segment_types_seq, 
                'cats_seq': cats_seq, 'lats_seq': lats_seq, 'lons_seq': lons_seq, 'popularities_seq': popularities_seq,
                'raw_timestamps_seq': raw_timestamps_seq,
                'gat_data': gat_data_obj
            })

    def __len__(self): return len(self.user_data_list)
    def __getitem__(self, idx): return self.user_data_list[idx]

def user_full_history_gat_collate_fn(batch, venue_pad_idx_g, cat_pad_idx_g, time_segment_pad_idx_g):
    """序列+PyG Graph联合拼装"""
    seq_keys = ['venues_seq', 'hours_seq', 'time_segment_types_seq', 'cats_seq', 'lats_seq', 'lons_seq', 'popularities_seq', 'raw_timestamps_seq']
    batch_seq_data = {k: [] for k in seq_keys}
    batch_seq_data['user_ids'] = []
    seq_lens_list = []
    
    valid_items = [item for item in batch if item is not None]
    if not valid_items: return None
    
    max_seq_len = max(len(item.get('venues_seq',[])) for item in valid_items)
    gat_data_list = []

    for item in valid_items:
        seq_len = len(item.get('venues_seq', []))
        seq_lens_list.append(seq_len)
        pad_len = max_seq_len - seq_len
        
        batch_seq_data['user_ids'].append(item.get('user_id', -1))

        for k in ['venues_seq', 'cats_seq']:
            pad_val = venue_pad_idx_g if 'venue' in k else cat_pad_idx_g
            batch_seq_data[k].append(torch.tensor(item.get(k, []) + [pad_val] * pad_len, dtype=torch.long))
        
        batch_seq_data['hours_seq'].append(torch.tensor(item.get('hours_seq', []) + [0] * pad_len, dtype=torch.long))
        batch_seq_data['time_segment_types_seq'].append(torch.tensor(item.get('time_segment_types_seq', []) + [time_segment_pad_idx_g] * pad_len, dtype=torch.long))
        
        for k in ['lats_seq', 'lons_seq', 'popularities_seq', 'raw_timestamps_seq']:
            dtype_val = torch.double if 'timestamp' in k else torch.float
            batch_seq_data[k].append(torch.tensor(item.get(k, []) + [0.0] * pad_len, dtype=dtype_val))

        current_gat = item.get('gat_data')
        empty_gat = Data(x_node_venue_ids=torch.empty(0, dtype=torch.long), x_node_cat_ids=torch.empty(0, dtype=torch.long), x_node_locs=torch.empty(0, 2, dtype=torch.float), x_node_popularity=torch.empty(0, 1, dtype=torch.float), edge_index=torch.empty(2, 0, dtype=torch.long), edge_attr=torch.empty(0, 2, dtype=torch.long))

        if current_gat is None or current_gat.x_node_venue_ids.numel() == 0:
            data_pyg = empty_gat.clone()
        else:
            data_pyg = current_gat.clone()
            for k, v in empty_gat.to_dict().items():
                if getattr(data_pyg, k, None) is None: setattr(data_pyg, k, v.clone())
            if data_pyg.edge_index.size(1) > 0 and (not hasattr(data_pyg, 'edge_attr') or data_pyg.edge_attr.numel() == 0):
                setattr(data_pyg, 'edge_attr', torch.full((data_pyg.edge_index.size(1), 2), fill_value=EDGE_TIME_BIN_PAD_IDX_GAT, dtype=torch.long))
        gat_data_list.append(data_pyg)

    stacked_data = {}
    for k in seq_keys:
        if k in batch_seq_data and batch_seq_data[k]:
            stacked_data[k] = torch.stack(batch_seq_data[k])
    
    stacked_data['user_ids'] = batch_seq_data['user_ids']
    stacked_data['seq_lens'] = torch.tensor(seq_lens_list, dtype=torch.long)
    stacked_data['padding_mask_seq'] = (stacked_data['venues_seq'] == venue_pad_idx_g)
    
    stacked_data['gat_batch'] = Batch.from_data_list(gat_data_list) if gat_data_list else Batch()
    return stacked_data

# ==============================================================================
# 3. 全局 POI 图构建器
# ==============================================================================
class GlobalPOIGraphBuilder:
    """构建多关系全局POI图（地理、类别、共现、热门度）"""
    def __init__(self, venue_map, category_map, distance_lookup, df):
        self.venue_map = venue_map
        self.category_map = category_map 
        self.distance_lookup = distance_lookup if distance_lookup else {}
        self.df = df
        self.num_pois = len(venue_map)
        self.poi_to_category, self.poi_to_coords, self.poi_popularity = {}, {}, {}
        self._build_poi_info()
        
    def _build_poi_info(self):
        for _, row in self.df[['geo_id', 'integer_venue_category_id']].drop_duplicates().iterrows():
            poi_str = str(row['geo_id'])
            if poi_str in self.venue_map: self.poi_to_category[self.venue_map[poi_str]] = int(row['integer_venue_category_id'])
        for _, row in self.df[['geo_id', 'latitude_norm', 'longitude_norm']].drop_duplicates().iterrows():
            poi_str = str(row['geo_id'])
            if poi_str in self.venue_map: self.poi_to_coords[self.venue_map[poi_str]] = (row['latitude_norm'], row['longitude_norm'])
        for poi_str, count in self.df['geo_id'].value_counts().items():
            if str(poi_str) in self.venue_map: self.poi_popularity[self.venue_map[str(poi_str)]] = count
    
    def build_geographical_edges(self, distance_threshold=GLOBAL_GRAPH_DISTANCE_THRESHOLD):
        edges = []
        for p1 in self.distance_lookup:
            for p2, dist in self.distance_lookup[p1].items():
                if dist <= distance_threshold and p1 != p2:
                    edges.append((p1, p2, RELATION_TYPES['GEOGRAPHICAL']))
        return edges
    
    def build_same_category_edges(self, max_total=1000000, max_per_cat=5000):
        edges, count = [], 0
        cat_to_pois = defaultdict(list)
        for p, c in self.poi_to_category.items(): cat_to_pois[c].append(p)
        for _, pois in sorted(cat_to_pois.items(), key=lambda x: len(x[1])):
            if len(pois) < 2: continue
            potentials = len(pois) * (len(pois) - 1)
            if potentials <= max_per_cat and count + potentials <= max_total:
                for i in range(len(pois)):
                    for j in range(i+1, len(pois)):
                        edges.append((pois[i], pois[j], RELATION_TYPES['SAME_CATEGORY']))
                        edges.append((pois[j], pois[i], RELATION_TYPES['SAME_CATEGORY']))
                count += potentials
            else:
                to_sample = min(max_per_cat, max_total - count)
                if to_sample <= 0: break
                edge_set = set()
                att, max_att = 0, to_sample * 5
                while len(edge_set) < to_sample // 2 and att < max_att:
                    try:
                        p1, p2 = random.sample(range(len(pois)), 2)
                        edge_set.add(tuple(sorted((pois[p1], pois[p2]))))
                    except ValueError: break
                    att += 1
                for p1, p2 in edge_set:
                    edges.append((p1, p2, RELATION_TYPES['SAME_CATEGORY']))
                    edges.append((p2, p1, RELATION_TYPES['SAME_CATEGORY']))
                count += len(edge_set) * 2
        return edges

    def build_cooccurrence_edges(self, min_users=GLOBAL_GRAPH_COOCCUR_MIN_USERS):
        edges = []
        user_pois = defaultdict(set)
        for _, row in self.df.iterrows():
            p_str = str(row['geo_id'])
            if p_str in self.venue_map: user_pois[row['user_id']].add(self.venue_map[p_str])
        
        pair_counts = defaultdict(int)
        for p_set in user_pois.values():
            p_list = list(p_set)
            for i in range(len(p_list)):
                for j in range(i+1, len(p_list)):
                    pair_counts[tuple(sorted((p_list[i], p_list[j])))] += 1
                    
        for (p1, p2), c in pair_counts.items():
            if c >= min_users:
                edges.append((p1, p2, RELATION_TYPES['CO_OCCURRENCE']))
                edges.append((p2, p1, RELATION_TYPES['CO_OCCURRENCE']))
        return edges

    def build_popularity_edges(self, top_ratio=0.1, max_edges=100000):
        if not self.poi_popularity: return []
        sorted_pois = sorted(self.poi_popularity.items(), key=lambda x: x[1], reverse=True)
        top_k = max(1, int(len(sorted_pois) * top_ratio))
        nodes, weights = [p for p,_ in sorted_pois[:top_k]], [w for _,w in sorted_pois[:top_k]]
        if len(nodes) < 2: return []
        
        edges, edge_set, count, att = [], set(), 0, 0
        while count < max_edges and att < max_edges * 5:
            try: p1, p2 = random.choices(nodes, weights=weights, k=2)
            except IndexError: break
            if p1 == p2: att+=1; continue
            pair = tuple(sorted((p1, p2)))
            if pair not in edge_set:
                edge_set.add(pair)
                edges.extend([(p1, p2, RELATION_TYPES['POPULARITY']), (p2, p1, RELATION_TYPES['POPULARITY'])])
                count += 2
            att += 1
        return edges

    def build_global_graph(self):
        all_edges = self.build_geographical_edges()
        if GLOBAL_GRAPH_SAME_CATEGORY_ENABLED: all_edges.extend(self.build_same_category_edges())
        all_edges.extend(self.build_cooccurrence_edges())
        all_edges.extend(self.build_popularity_edges())
        
        unique_edges = set()
        final_idx, final_attr = [], []
        for src, dst, rel in all_edges:
            if (src, dst, rel) not in unique_edges:
                unique_edges.add((src, dst, rel))
                final_idx.append((src, dst))
                final_attr.append(rel)
        
        edge_index = torch.tensor(final_idx, dtype=torch.long).t().contiguous() if final_idx else torch.empty(2,0,dtype=torch.long)
        edge_attr = torch.tensor(final_attr, dtype=torch.long) if final_attr else torch.empty(0,dtype=torch.long)
        
        node_features = []
        for i in range(self.num_pois):
            cat = self.poi_to_category.get(i, NUM_TIME_SEGMENTS_W_PAD) # fallback PAD
            c_lat, c_lon = self.poi_to_coords.get(i, (0.0, 0.0))
            pop = np.log1p(self.poi_popularity.get(i, 0))
            node_features.append([i, cat, c_lat, c_lon, pop])
        node_x = torch.tensor(node_features, dtype=torch.float)
        
        return Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=self.num_pois)