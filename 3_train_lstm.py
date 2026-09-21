# -*- coding: utf-8 -*-
"""
3_train_lstm.py — LSTM 时空代理模型训练
============================================
多任务学习:
  - 主任务: 回归三场拼接的潜变量 Z (N, T, k_total), 等价于预测全场。
  - 辅助任务: 回归关键标量指标 [最大位移, 最大应力, 最大损伤] (N, T, 3)。

输入: 施工事件特征 + 材料参数 (N, T, IN_DIM)
输出: 潜变量 Z (N, T, k_total) + 标量指标 (N, T, 3)

数据划分按 case (train/val/test 已独立), 时间步内不跨 case, 避免泄漏。

运行:
  python 3_train_lstm.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import config as C
from model import Surrogate


# ============================================================================
# 数据加载与标准化
# ============================================================================

def load_split(split):
    d = np.load(os.path.join(C.DATA_DIR, split, 'latent.npz'))
    return d['inputs'], d['Z'], d['scalars']


def standardize_inputs(inputs_tr, inputs_va, inputs_te):
    """用 train 的均值/方差对输入特征做 z-score 标准化。"""
    flat = inputs_tr.reshape(-1, inputs_tr.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    return ((inputs_tr - mean) / std,
            (inputs_va - mean) / std,
            (inputs_te - mean) / std), (mean, std)


# ============================================================================
# 训练
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("LSTM 时空代理模型训练")
    print("=" * 60)
    print(f"  设备: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # ── 数据 ──
    inputs_tr, Z_tr, scalars_tr = load_split('train')
    inputs_va, Z_va, scalars_va = load_split('val')
    inputs_te, Z_te, scalars_te = load_split('test')

    (inputs_tr, inputs_va, inputs_te), (x_mean, x_std) = standardize_inputs(
        inputs_tr, inputs_va, inputs_te)

    k_total = Z_tr.shape[-1]
    print(f"  训练样本: {inputs_tr.shape[0]} case × {inputs_tr.shape[1]} 步, "
          f"输入 {inputs_tr.shape[2]} 维 → 潜变量 {k_total} 维")

    def to_tensor(*arrs):
        return [torch.from_numpy(a.astype(np.float32)).to(device) for a in arrs]

    X_tr, Y_tr, S_tr = to_tensor(inputs_tr, Z_tr, scalars_tr)
    X_va, Y_va, S_va = to_tensor(inputs_va, Z_va, scalars_va)
    X_te, Y_te, S_te = to_tensor(inputs_te, Z_te, scalars_te)

    loader = DataLoader(TensorDataset(X_tr, Y_tr, S_tr),
                        batch_size=C.BATCH_SIZE, shuffle=True)

    # ── 模型 ──
    model = Surrogate(C.IN_DIM, k_total, C.HIDDEN, C.LAYERS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    loss_fn = nn.MSELoss()

    # 标量指标的相对权重 (与潜变量损失平衡)
    scalar_weight = 1.0

    best_val = float('inf')
    os.makedirs(C.MODEL_DIR, exist_ok=True)

    print(f"\n  超参: epochs={C.EPOCHS}, batch={C.BATCH_SIZE}, lr={C.LR}, "
          f"hidden={C.HIDDEN}, layers={C.LAYERS}\n")

    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb, sb in loader:
            z_pred, s_pred = model(xb)
            loss = loss_fn(z_pred, yb) + scalar_weight * loss_fn(s_pred, sb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(loader.dataset)

        # 验证
        model.eval()
        with torch.no_grad():
            z_va, s_va = model(X_va)
            val_loss = (loss_fn(z_va, Y_va) + scalar_weight * loss_fn(s_va, S_va)).item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model': model.state_dict(),
                'x_mean': x_mean, 'x_std': x_std,
                'k_total': k_total,
            }, os.path.join(C.MODEL_DIR, 'lstm.pt'))

        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{C.EPOCHS}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    print(f"\n训练完成! 最佳 val_loss = {best_val:.6f}")
    print(f"模型已保存: {C.MODEL_DIR}/lstm.pt")


if __name__ == '__main__':
    main()
