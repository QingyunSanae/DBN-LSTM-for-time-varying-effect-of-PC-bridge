# -*- coding: utf-8 -*-
"""
config.py — PCA-LSTM 代理模型管线共享配置
============================================
合成数据版：用合成场数据替代 Abaqus 真实场，跑通「数据 → 降维 → 训练 → 推理」整条链路。

真实场景替换点：
  - 本文件的 NX/NY 空间网格，真实场景换成 .odb 里提取的节点自由度 (约 5 万节点 × 3 分量)。
  - 场的物理公式，真实场景换成 Abaqus 求解结果。
  其余 (PCA / LSTM / 训练 / 推理) 逻辑完全不变。
"""

import os

# ── 路径 ──
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
MODEL_DIR = os.path.join(ROOT, 'models')

# ── 空间网格 (合成场用 2D 网格，D = NX*NY 维) ──
NX = 48
NY = 48
D_FIELD = NX * NY                 # 2304，每个场的自由度

# ── 时间轴 (施工过程) ──
N_SEG = 33                        # 施工节段数 S0..S32，对齐真实模型
STEPS_PER_SEG = 3                 # 每节段 C(浇筑) / T(张拉) / R(拆模)
T_STEPS = N_SEG * STEPS_PER_SEG   # 99 个时间步

# ── 材料不确定性参数 (对齐 B4021_nonlinear.for 的 5 个 THETA) ──
N_THETA = 5
THETA_NAMES = ['theta_cr', 'theta_sh', 'theta_fc', 'theta_pre', 'theta_h']
THETA_RANGE = (0.8, 1.3)          # 参数采样范围

# ── 输入特征维度 ──
# 每步: 步类型 onehot(3) + 当前步时长(1) + 累计时间(1) + 节段进度(1) + 材料参数(5)
IN_DIM = 3 + 1 + 1 + 1 + N_THETA  # = 11

# ── 数据规模 (case 数，按 case 划分 train/val/test) ──
N_TRAIN = 200
N_VAL = 20
N_TEST = 20

# ── PCA 主成分数 (每场独立) ──
K_U = 20                          # 位移场平滑，收敛快
K_S = 40                          # 应力场梯度大，多给
K_D = 20                          # 损伤场局部化，需要更多主成分

# ── LSTM 训练超参 ──
EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
HIDDEN = 256
LAYERS = 2

# ── 可复现种子 ──
SEED = 42
