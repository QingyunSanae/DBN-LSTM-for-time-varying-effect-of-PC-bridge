# -*- coding: utf-8 -*-
"""
4_infer.py — 推理与评估
============================================
在 test 集上评估代理模型:
  - 还原三个物理场 (U/S/DAMAGE) 的相对误差
  - 关键标量指标 (最大位移/最大应力/最大损伤) 的误差, 两种来源对比:
      a) 从预测场提取极值
      b) LSTM 标量 head 直接回归

运行:
  python 4_infer.py
"""

import os
import numpy as np
import torch
import joblib
import config as C
from model import Surrogate


def rel_error(pred, true, axis=None):
    """相对误差 = ||pred - true|| / ||true||。"""
    num = np.linalg.norm(pred - true, axis=axis)
    den = np.linalg.norm(true, axis=axis)
    return num / (den + 1e-8)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 加载模型与 PCA ──
    ckpt = torch.load(os.path.join(C.MODEL_DIR, 'lstm.pt'), map_location=device, weights_only=False)
    pca_dict = joblib.load(os.path.join(C.MODEL_DIR, 'pca.joblib'))
    pca_U, pca_S, pca_D = pca_dict['pca_U'], pca_dict['pca_S'], pca_dict['pca_D']
    k_total = ckpt['k_total']

    model = Surrogate(C.IN_DIM, k_total, C.HIDDEN, C.LAYERS).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # ── 加载 test 数据 ──
    data = np.load(os.path.join(C.DATA_DIR, 'test', 'data.npz'))
    latent = np.load(os.path.join(C.DATA_DIR, 'test', 'latent.npz'))
    inputs = (latent['inputs'] - ckpt['x_mean']) / ckpt['x_std']
    U_true, S_true, D_true = data['U'], data['S'], data['D']
    scalars_true = data['scalars']                    # (N, T, 3)

    X = torch.from_numpy(inputs.astype(np.float32)).to(device)

    print("=" * 60)
    print("代理模型推理评估 (test 集)")
    print("=" * 60)
    print(f"  设备: {device}, 测试样本: {X.shape[0]} case × {X.shape[1]} 步\n")

    # ── 前向推理 ──
    with torch.no_grad():
        z_pred, s_pred = model(X)
    z_pred = z_pred.cpu().numpy()                      # (N, T, k_total)
    s_pred = s_pred.cpu().numpy()                      # (N, T, 3)

    # ── 拆出三场潜变量, PCA 逆变换还原物理场 ──
    zU, zS, zD = np.split(z_pred, [C.K_U, C.K_U + C.K_S], axis=-1)
    N, T, _ = zU.shape

    U_pred = pca_U.inverse_transform(zU.reshape(-1, C.K_U)).reshape(N, T, C.D_FIELD)
    S_pred = pca_S.inverse_transform(zS.reshape(-1, C.K_S)).reshape(N, T, C.D_FIELD)
    D_pred = pca_D.inverse_transform(zD.reshape(-1, C.K_D)).reshape(N, T, C.D_FIELD)

    # ── 物理场相对误差 ──
    print("【物理场相对误差】 (逐 case 平均)")
    for name, pred, true in [('U 位移场', U_pred, U_true),
                             ('S 应力场', S_pred, S_true),
                             ('D 损伤场', D_pred, D_true)]:
        err = rel_error(pred, true, axis=(1, 2))       # 每个 case 一个误差
        print(f"  {name}: 均值 {err.mean() * 100:.2f}%  中位 {np.median(err) * 100:.2f}%  "
              f"最大 {err.max() * 100:.2f}%")

    # ── 标量指标误差 ──
    print("\n【关键标量指标相对误差】 (逐 case 平均)")
    scalar_names = ['最大位移 max|U|', '最大应力 max S', '最大损伤 max D']

    # a) 从预测场提取极值
    scalar_from_field = np.stack([
        np.abs(U_pred).max(axis=2),
        S_pred.max(axis=2),
        D_pred.max(axis=2),
    ], axis=2)                                         # (N, T, 3)

    print("  (a) 从预测场提取极值:")
    for i, name in enumerate(scalar_names):
        err = rel_error(scalar_from_field[..., i], scalars_true[..., i], axis=1)
        print(f"      {name}: 均值 {err.mean() * 100:.2f}%")

    # b) LSTM 标量 head 直接回归
    print("  (b) LSTM 标量 head 直接回归:")
    for i, name in enumerate(scalar_names):
        err = rel_error(s_pred[..., i], scalars_true[..., i], axis=1)
        print(f"      {name}: 均值 {err.mean() * 100:.2f}%")

    # ── 抽查一个 case 的数值对比 ──
    print("\n【抽查 case 0 末时刻标量对比】")
    t = T - 1
    for i, name in enumerate(scalar_names):
        print(f"  {name}: 真实 {scalars_true[0, t, i]:.4f}  "
              f"| 场提取 {scalar_from_field[0, t, i]:.4f}  "
              f"| head {s_pred[0, t, i]:.4f}")

    print("\n评估完成。")


if __name__ == '__main__':
    main()
