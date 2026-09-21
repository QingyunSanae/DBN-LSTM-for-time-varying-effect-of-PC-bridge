# -*- coding: utf-8 -*-
"""
1_generate_synthetic.py — 合成场数据生成器
============================================
模拟「施工节奏 + 材料参数 → U/S/DAMAGE 场演化」，替代 Abaqus 求解结果，
生成 N 个工况的数据文件，供 PCA 降维和 LSTM 训练。

关键设计原则 (真实场景同样适用):
  场的变异必须「由输入参数 (材料 theta + 施工节奏) 决定」，代理模型才学得会。
  因此这里不引入任何独立于输入的随机场/随机系数，只保留少量高斯噪声
  作为「不可约误差」(对应真实有限元的数值/测量噪声)。

场物理逻辑 (占位, 不追求精确, 只求结构合理):
  - 位移 U: 固定挠曲模态 (sin 形状), 幅度随进度增长, 徐变放大, 抗压强度折减, 收缩附加
  - 应力 S: 自重 + 预应力, 受材料不均匀性 (由 theta 调制) 影响
  - 损伤 D: 应力超阈值后非线性出现, 在材料弱区局部化

运行:
  python 1_generate_synthetic.py
"""

import os
import json
import numpy as np
import config as C


# ============================================================================
# 空间与确定性基函数
# ============================================================================

def spatial_grid():
    """返回展平后的 (X, Y)，各 (D_FIELD,) 维，范围 [0,1]。"""
    x = np.linspace(0.0, 1.0, C.NX)
    y = np.linspace(0.0, 1.0, C.NY)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X.ravel().astype(np.float32), Y.ravel().astype(np.float32)


def deflection_shape(X):
    """固定挠曲模态: 前 4 个 sin 模态组合, 代表桥梁固有挠曲形状。"""
    shape = np.zeros_like(X)
    for k in range(1, 5):
        shape += (1.0 / k) * np.sin(k * np.pi * X)
    return shape.astype(np.float32)


def material_pattern(X, Y):
    """固定材料不均匀空间模式, 幅度由 theta_h / theta_fc 调制 (可预测)。"""
    p = np.sin(3 * np.pi * X) * np.cos(2 * np.pi * Y) + 0.5 * np.sin(5 * np.pi * X)
    return p.astype(np.float32)


def sample_theta(rng):
    """采样 5 个材料不确定性参数，范围见 config。"""
    lo, hi = C.THETA_RANGE
    return rng.uniform(lo, hi, size=C.N_THETA).astype(np.float32)


def sample_schedule(rng):
    """采样施工节奏: 每节段 (C, T, R) 时长 (天)。"""
    C_d = rng.uniform(3.0, 8.0, size=C.N_SEG)     # 浇筑 3~8 天
    T_d = rng.uniform(0.5, 2.0, size=C.N_SEG)     # 张拉 0.5~2 天
    R_d = rng.uniform(0.5, 2.0, size=C.N_SEG)     # 拆模 0.5~2 天
    return C_d.astype(np.float32), T_d.astype(np.float32), R_d.astype(np.float32)


# ============================================================================
# 场演化 (全部由 theta + 施工进度决定, 加少量噪声)
# ============================================================================

def compute_fields(X, Y, shape, pattern, theta, dur_kind, cum_time, progress, rng):
    """计算单个时间步的三个场, 返回 (U, S, DAMAGE)。"""
    theta_cr, theta_sh, theta_fc, theta_pre, theta_h = theta
    t = max(cum_time, 1e-3)

    # ── 位移场 U ──
    creep = 1.0 + theta_cr * 0.3 * np.log1p(t / 10.0)       # 徐变放大
    stiffness = 1.0 / theta_fc                              # 抗压强度 → 刚度
    shrink = theta_sh * 0.05 * (1.0 - np.exp(-t / 40.0)) * X
    U = (0.4 * progress ** 2 * (1.0 + 0.2 * Y) * shape * creep * stiffness
         + shrink)
    U = U + 0.004 * rng.standard_normal(U.shape, dtype=np.float32)

    # ── 应力场 S ──
    self_weight = progress * (1.0 - Y)                      # 自重 (顶压底拉)
    tensioned = (dur_kind >= 1)                             # T 步起预应力生效
    prestress = -theta_pre * (0.6 + 0.4 * np.exp(-((X - 0.5) ** 2) / 0.1)) * tensioned
    S = (self_weight + prestress) * (1.0 + 0.1 * theta_h * pattern)
    S = S + 0.004 * rng.standard_normal(S.shape, dtype=np.float32)

    # ── 损伤场 DAMAGE ──
    strength = theta_fc * (1.2 - 0.3 * Y)                   # 抗压强度随高度
    S_eff = np.abs(S) * (1.0 + 0.2 * np.maximum(pattern, 0.0))  # 不均匀导致应力集中
    overstress = S_eff - strength * (0.8 + 0.4 * progress)  # 阈值随进度提高
    D = 1.0 - np.exp(-np.maximum(overstress, 0.0) / 0.15)
    D = D * (1.0 + 0.2 * (theta_h - 1.0))                   # 湿度影响耐久
    D = D + 0.002 * rng.standard_normal(D.shape, dtype=np.float32)

    return (U.astype(np.float32), S.astype(np.float32), D.astype(np.float32))


# ============================================================================
# 单个工况生成
# ============================================================================

def make_case(rng):
    """生成一个完整工况: 输入特征序列 + 三个场序列 + 标量指标。"""
    theta = sample_theta(rng)
    C_d, T_d, R_d = sample_schedule(rng)

    X, Y = spatial_grid()
    shape = deflection_shape(X)
    pattern = material_pattern(X, Y)

    inputs = []
    U_list, S_list, D_list = [], [], []

    cum_time = 0.0
    step_types = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]          # C / T / R onehot
    dur_arr = [C_d, T_d, R_d]

    for seg in range(C.N_SEG):
        progress = (seg + 1) / C.N_SEG
        for kind in range(C.STEPS_PER_SEG):
            dur = dur_arr[kind][seg]
            cum_time += dur

            feat = np.concatenate([
                np.array(step_types[kind], dtype=np.float32),
                np.array([dur, cum_time, progress], dtype=np.float32),
                theta,
            ]).astype(np.float32)                            # (IN_DIM,)
            inputs.append(feat)

            U, S, D = compute_fields(X, Y, shape, pattern, theta, kind,
                                     cum_time, progress, rng)
            U_list.append(U); S_list.append(S); D_list.append(D)

    inputs = np.stack(inputs)                                # (T, IN_DIM)
    U = np.stack(U_list)                                     # (T, D)
    S = np.stack(S_list)
    D = np.stack(D_list)

    # 标量指标: 从场取全局极值 (工程关心的最大位移/应力/损伤)
    scalars = np.stack([
        np.abs(U).max(axis=1),                               # 最大位移
        S.max(axis=1),                                       # 最大应力
        D.max(axis=1),                                       # 最大损伤
    ], axis=1).astype(np.float32)                            # (T, 3)

    return inputs, U, S, D, scalars, theta


# ============================================================================
# 批量生成与存储
# ============================================================================

def generate_and_save(n_cases, split, rng):
    """生成 n_cases 个工况并保存为 .npz 到 data/<split>/。"""
    out_dir = os.path.join(C.DATA_DIR, split)
    os.makedirs(out_dir, exist_ok=True)

    all_inputs = np.zeros((n_cases, C.T_STEPS, C.IN_DIM), dtype=np.float32)
    all_U = np.zeros((n_cases, C.T_STEPS, C.D_FIELD), dtype=np.float32)
    all_S = np.zeros((n_cases, C.T_STEPS, C.D_FIELD), dtype=np.float32)
    all_D = np.zeros((n_cases, C.T_STEPS, C.D_FIELD), dtype=np.float32)
    all_scalars = np.zeros((n_cases, C.T_STEPS, 3), dtype=np.float32)

    for i in range(n_cases):
        inputs, U, S, D, scalars, theta = make_case(rng)
        all_inputs[i] = inputs
        all_U[i] = U
        all_S[i] = S
        all_D[i] = D
        all_scalars[i] = scalars
        if (i + 1) % 50 == 0:
            print(f"    [{split}] {i + 1}/{n_cases}")

    np.savez_compressed(
        os.path.join(out_dir, 'data.npz'),
        inputs=all_inputs, U=all_U, S=all_S, D=all_D, scalars=all_scalars,
    )
    print(f"  [{split}] 已保存: {out_dir}/data.npz "
          f"({os.path.getsize(os.path.join(out_dir, 'data.npz')) / 1024 / 1024:.1f} MB)")


def main():
    rng = np.random.default_rng(C.SEED)
    os.makedirs(C.DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("合成场数据生成")
    print("=" * 60)
    print(f"  空间网格: {C.NX}×{C.NY} = {C.D_FIELD} 自由度/场")
    print(f"  时间步:   {C.T_STEPS} (节段 {C.N_SEG} × {C.STEPS_PER_SEG} 步)")
    print(f"  输入维度: {C.IN_DIM}")
    print(f"  数据划分: train={C.N_TRAIN} / val={C.N_VAL} / test={C.N_TEST}")
    print()

    generate_and_save(C.N_TRAIN, 'train', rng)
    generate_and_save(C.N_VAL, 'val', rng)
    generate_and_save(C.N_TEST, 'test', rng)

    # 写元信息
    meta = {
        'NX': C.NX, 'NY': C.NY, 'D_FIELD': C.D_FIELD,
        'N_SEG': C.N_SEG, 'T_STEPS': C.T_STEPS, 'IN_DIM': C.IN_DIM,
        'N_TRAIN': C.N_TRAIN, 'N_VAL': C.N_VAL, 'N_TEST': C.N_TEST,
        'THETA_NAMES': C.THETA_NAMES,
        'SEED': C.SEED,
    }
    with open(os.path.join(C.DATA_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n完成! 数据已输出到 {C.DATA_DIR}/")


if __name__ == '__main__':
    main()
