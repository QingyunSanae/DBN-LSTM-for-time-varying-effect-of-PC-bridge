# -*- coding: utf-8 -*-
"""
2_build_pca.py — 多场 PCA 空间降维
============================================
对 U / S / DAMAGE 三场各自独立做「全局 PCA」(所有 train 工况所有时间步一起拟合)，
把高维空间场压成少量潜变量，得到 LSTM 的训练标签 Z。

设计要点:
  - 三场独立降维: 量级、平滑度差异大, 混做会让小量级场被淹没。
  - PCA 只在 train 上拟合, val/test 用同一套基做 transform, 避免数据泄漏。
  - PCA 正交 → 潜变量 MSE 等价于物理场加权 MSE, 故 LSTM 直接在潜变量空间训练。

运行:
  python 2_build_pca.py
"""

import os
import numpy as np
import joblib
from sklearn.decomposition import PCA
import config as C


def load_split(split):
    d = np.load(os.path.join(C.DATA_DIR, split, 'data.npz'))
    return d['inputs'], d['U'], d['S'], d['D'], d['scalars']


def fit_field_pca(field, k, name):
    """对单个场做全局 PCA, 返回 (pca, 该场潜变量 Z)。"""
    N, T, D = field.shape
    flat = field.reshape(-1, D)
    pca = PCA(n_components=k).fit(flat)
    var = pca.explained_variance_ratio_.sum()
    print(f"  {name}: 主成分 {k} 个, 累计方差率 {var * 100:.2f}%")
    return pca


def to_latent(pca, field, k):
    """把场变换为潜变量 (N, T, k)。"""
    N, T, D = field.shape
    return pca.transform(field.reshape(-1, D)).reshape(N, T, k)


def main():
    os.makedirs(C.MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("多场 PCA 降维")
    print("=" * 60)

    # ── 1. 仅在 train 上拟合 PCA ──
    _, U_tr, S_tr, D_tr, _ = load_split('train')
    print("拟合全局 PCA (仅 train):")
    pca_U = fit_field_pca(U_tr, C.K_U, 'U 位移场')
    pca_S = fit_field_pca(S_tr, C.K_S, 'S 应力场')
    pca_D = fit_field_pca(D_tr, C.K_D, 'D 损伤场')

    k_total = C.K_U + C.K_S + C.K_D
    print(f"  潜变量总维度 k_total = {C.K_U}+{C.K_S}+{C.K_D} = {k_total}")

    # ── 2. 保存 PCA 模型 ──
    joblib.dump(
        {'pca_U': pca_U, 'pca_S': pca_S, 'pca_D': pca_D,
         'K_U': C.K_U, 'K_S': C.K_S, 'K_D': C.K_D},
        os.path.join(C.MODEL_DIR, 'pca.joblib'),
    )
    print(f"  PCA 模型已保存: {C.MODEL_DIR}/pca.joblib")

    # ── 3. 对全部分割 transform, 生成潜变量标签 latent.npz ──
    for split in ['train', 'val', 'test']:
        inputs, U, S, D, scalars = load_split(split)
        Z = np.concatenate([
            to_latent(pca_U, U, C.K_U),
            to_latent(pca_S, S, C.K_S),
            to_latent(pca_D, D, C.K_D),
        ], axis=-1).astype(np.float32)                  # (N, T, k_total)

        out = os.path.join(C.DATA_DIR, split, 'latent.npz')
        np.savez_compressed(out, inputs=inputs, Z=Z, scalars=scalars)
        print(f"  [{split}] 潜变量已保存: latent.npz (Z 形状 {Z.shape})")

    print("\n完成! 潜变量标签可用于 LSTM 训练。")


if __name__ == '__main__':
    main()
