# -*- coding: utf-8 -*-
"""
5_visualize.py — 场对比可视化
============================================
选取一个 test case 的末时刻，画三场 (U/S/DAMAGE) 的
「真实 | 预测 | 误差」3×3 对比图，直观检验代理模型还原质量。

配色遵循科学可视化规范:
  - 场量用感知均匀的单一色相 sequential colormap (viridis/magma)，色盲安全, 非 rainbow。
  - 误差用 diverging colormap (RdBu_r)，中性白中点，正负可辨。

运行:
  python 5_visualize.py
"""

import os
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use('Agg')                     # 无头环境保存图片
import matplotlib.pyplot as plt
import config as C
from model import Surrogate


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

    # ── 推理, 还原预测场 ──
    X = torch.from_numpy(inputs.astype(np.float32)).to(device)
    with torch.no_grad():
        z_pred, _ = model(X)
    z_pred = z_pred.cpu().numpy()
    zU, zS, zD = np.split(z_pred, [C.K_U, C.K_U + C.K_S], axis=-1)

    N, T, _ = zU.shape
    U_pred = pca_U.inverse_transform(zU.reshape(-1, C.K_U)).reshape(N, T, C.D_FIELD)
    S_pred = pca_S.inverse_transform(zS.reshape(-1, C.K_S)).reshape(N, T, C.D_FIELD)
    D_pred = pca_D.inverse_transform(zD.reshape(-1, C.K_D)).reshape(N, T, C.D_FIELD)

    # ── 选取 case 0 的末时刻 ──
    case, t = 0, T - 1
    fields = [
        ('U displacement', U_true[case, t], U_pred[case, t], 'viridis'),
        ('S stress',       S_true[case, t], S_pred[case, t], 'viridis'),
        ('DAMAGE',         D_true[case, t], D_pred[case, t], 'magma'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    col_titles = ['True', 'Predicted', 'Error (pred - true)']

    for i, (name, true, pred, cmap) in enumerate(fields):
        t_img = true.reshape(C.NX, C.NY)
        p_img = pred.reshape(C.NX, C.NY)
        e_img = p_img - t_img
        vmin, vmax = t_img.min(), t_img.max()
        emax = np.abs(e_img).max()

        im0 = axes[i, 0].imshow(t_img, cmap=cmap, vmin=vmin, vmax=vmax)
        im1 = axes[i, 1].imshow(p_img, cmap=cmap, vmin=vmin, vmax=vmax)
        im2 = axes[i, 2].imshow(e_img, cmap='RdBu_r', vmin=-emax, vmax=emax)

        axes[i, 0].set_ylabel(name, fontsize=12)
        for j, im in enumerate([im0, im1, im2]):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13)

    fig.suptitle(f'Surrogate field reconstruction — test case {case}, step {t}',
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(C.ROOT, 'field_comparison.png')
    fig.savefig(out_path, dpi=150)
    print(f"图片已保存: {out_path}")


if __name__ == '__main__':
    main()
