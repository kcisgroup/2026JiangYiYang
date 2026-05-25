import torch
import torch.nn as nn
from tqdm import tqdm
from utils import calculate_all_metrics

def evaluate(model, dataloader, device, pad_idx, poi_info, max_trip_len):
    """验证集/测试集评估逻辑"""
    all_preds, all_truths, all_times, all_budgets = [], [], [], []
    
    for u, s, st, b, targets in dataloader:
        u, s, st, b = u.to(device), s.to(device), st.to(device), b.to(device)
        
        # 调用模型的自回归推理方法
        gen_seqs, gen_times = model.generate(u, s, st, b, max_len=max_trip_len, temperature=0.8)
        
        start_pois = s.cpu().numpy().tolist()
        targets_list = targets.cpu().numpy().tolist()
        
        # 去掉 Padding，合并起点的 Ground Truth 与 Pred 轨迹
        for i in range(len(start_pois)):
            clean_truth = [start_pois[i]] + [x for x in targets_list[i] if x != pad_idx]
            full_pred = [start_pois[i]] + gen_seqs[i]
            all_truths.append(clean_truth)
            all_preds.append(full_pred)
            
        all_times.extend(gen_times)
        all_budgets.extend(b.cpu().tolist())
        
    # 计算详细的验证指标
    return calculate_all_metrics(all_truths, all_preds, all_times, all_budgets, poi_info)

def train_eval_loop(model, train_loader, val_loader, test_loader, num_venues_with_pad, pad_idx, poi_info, config):
    """完整的训练管线设计"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # 当指标不上升时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    device = config['device']
    
    best_f1 = 0.0
    epochs_no_improve = 0  
    
    print("\n" + "="*40)
    print("🚀 开始稳定版训练...")
    
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0
        
        # 稳定的计划采样（Teacher Forcing 比例线性衰减机制）
        # 范围大致在 config 配置的起始到结束值之间
        ratio_span = config['tf_start_ratio'] - config['tf_end_ratio']
        current_tf_ratio = max(config['tf_end_ratio'], config['tf_start_ratio'] - (epoch / config['epochs']) * ratio_span)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for u, s, st, b, targets in pbar:
            u, s, st, b, targets = u.to(device), s.to(device), st.to(device), b.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(u, s, st, b, targets, tf_ratio=current_tf_ratio)
            loss = criterion(logits.view(-1, num_venues_with_pad), targets.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | LR: {curr_lr:.2e} | TF Ratio: {current_tf_ratio:.2f} | Loss: {total_loss/len(train_loader):.4f}")
        
        # 每轮结束进行验证集评估
        metrics = evaluate(model, val_loader, device, pad_idx, poi_info, config['max_trip_len'])
        current_f1 = metrics['F1@5']
        print(f"[Validation] F1@5: {current_f1:.4f}, Pairs-F1: {metrics['Pairs-F1']:.4f}, TTR: {metrics['TTR']:.4f}")
        
        # 调度器基于 F1 值步进
        scheduler.step(current_f1)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            epochs_no_improve = 0  
            torch.save(model.state_dict(), config['best_model_path'])
            print(f"  [*] Best Model Saved! (F1@5 improved to {best_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  [!] No improvement for {epochs_no_improve} epoch(s).")
            
        # 早停判断
        if epochs_no_improve >= config['patience']:
            print(f"⚠️ 触发早停机制，停止训练。")
            break
            
    # --------- 最终评估 ---------
    print("\n🔥 最终测试集评估 (Final Testing)")
    model.load_state_dict(torch.load(config['best_model_path'], weights_only=True))
    test_metrics = evaluate(model, test_loader, device, pad_idx, poi_info, config['max_trip_len'])
    
    print("-" * 50)
    print(f"{'K':<6}| {'Recall':<9}| {'Prec':<9}| {'F1':<8}")
    print("-" * 50)
    for k in [3, 5, 10]:
        r = test_metrics.get(f'R@{k}', 0.0)
        p = test_metrics.get(f'P@{k}', 0.0)
        f1 = test_metrics.get(f'F1@{k}', 0.0)
        print(f"{k:<6}| {r:<9.4f}| {p:<9.4f}| {f1:<8.4f}")
    print(f"Pairs-F1  : {test_metrics.get('Pairs-F1', 0.0):.4f}")
    print(f"Diversity : {test_metrics.get('Diversity', 0.0):.4f}")
    print(f"TTR       : {test_metrics.get('TTR', 0.0):.4f}")