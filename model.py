# -*- coding: utf-8 -*-
"""
model.py — 代理模型定义 (训练与推理共用)
============================================
LSTM 时空代理模型, 多任务:
  - head_latent: 回归三场拼接潜变量 Z (N, T, k_total) → 全场
  - head_scalar: 回归关键标量指标 [max|U|, max S, max D] (N, T, 3)
"""

import torch.nn as nn


class Surrogate(nn.Module):
    def __init__(self, in_dim, k_total, hidden, layers):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True)
        self.head_latent = nn.Linear(hidden, k_total)   # 潜变量 (全场)
        self.head_scalar = nn.Linear(hidden, 3)         # 标量指标

    def forward(self, x):
        h, _ = self.lstm(x)                             # (B, T, hidden)
        z = self.head_latent(h)                         # (B, T, k_total)
        s = self.head_scalar(h)                         # (B, T, 3)
        return z, s
