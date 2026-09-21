# PCA-LSTM 时空代理模型管线（合成数据版）

虎门大桥施工过程代理模型的**端到端训练管线**。用合成场数据替代 Abaqus 求解结果，
跑通「数据 → PCA 降维 → LSTM 训练 → 推理还原」整条链路。

## 为什么用合成数据

真实数据需要研究版 Abaqus（支持 `B4021_nonlinear.for` 用户子程序）且模型 5 万+ 节点，
当前环境（16G 内存 + 学习版 Abaqus）跑不了。合成数据规模可控，用于**验证和调优整套算法**，
等研究版 license 到位后，只需替换数据来源（见文末「接入真实数据」）。

## 运行环境

独立 conda 环境 `ml310`，**不再依赖 ComfyUI 的 venv**：

- Python: `C:\Users\admin\.conda\envs\ml310\python.exe`（3.10.21）
- torch 2.5.1+cu121（GPU，RTX 4060 Laptop，cuda 可用）
- numpy 2.2.6 / scikit-learn 1.7.2

> 备用的 ComfyUI venv（`C:\Users\admin\ComfyUI\venv\`）仍可用，但别在里面装包。

## 文件结构

```
surrogate_pipeline/
├── config.py                 # 共享配置（规模、超参、路径）
├── model.py                  # Surrogate 模型定义（训练/推理共用）
├── 1_generate_synthetic.py   # 合成场数据生成
├── 2_build_pca.py            # 多场 PCA 降维
├── 3_train_lstm.py           # LSTM 训练
├── 4_infer.py                # 推理与评估
├── 5_visualize.py            # 场对比可视化
├── data/                     # 生成的数据（train/val/test）
└── models/                   # 训练产物（pca.joblib, lstm.pt）
```

## 使用方法（按顺序执行）

```bash
cd C:\Users\admin\Desktop\project\surrogate_pipeline

# 1. 生成合成数据
"C:\Users\admin\.conda\envs\ml310\python.exe" 1_generate_synthetic.py

# 2. PCA 降维
"C:\Users\admin\.conda\envs\ml310\python.exe" 2_build_pca.py

# 3. 训练 LSTM
"C:\Users\admin\.conda\envs\ml310\python.exe" 3_train_lstm.py

# 4. 推理评估
"C:\Users\admin\.conda\envs\ml310\python.exe" 4_infer.py

# 5. 可视化场对比（可选）
"C:\Users\admin\.conda\envs\ml310\python.exe" 5_visualize.py
```

> 这里写绝对路径而不是裸 `python`：本机裸 `python` 解析到 Windows Store 的 **Python 3.13**
> （torch 2.7.1+**cpu**，仅 CPU），照抄会静默落到 CPU 上跑，慢很多且依赖版本对不上。

## 数据流

```
输入 (每个工况): 施工节奏(每节段 C/T/R 时长) + 材料参数(5 个 THETA)
        │
        ▼
  合成场演化 U/S/DAMAGE: (T=99 步, D=2304 自由度)
        │  全局 PCA (三场独立)
        ▼
  潜变量 Z: (T, k_total=80)
        │  LSTM 监督学习: 输入 → Z + 标量指标
        ▼
  推理: 新施工方案 → LSTM → Ẑ → PCA 逆变换 → 预测物理场
```

## 模型结构

- **LSTM**: 2 层，hidden 256，输入 11 维（步类型 onehot + 时长 + 累计时间 + 进度 + 5 材料参数）
- **多任务 head**: `head_latent` 回归 80 维潜变量（全场），`head_scalar` 回归 3 维标量指标

## 当前结果（test 集，20 case）

| 指标 | 误差 |
|---|---|
| U 位移场相对误差 | 2.27% |
| S 应力场相对误差 | 1.52% |
| D 损伤场相对误差 | 20.97%（中位 7.93%）|
| 最大位移 / 最大应力 / 最大损伤 | 1.78% / 2.71% / 8.60% |

损伤场误差偏大是稀疏局部化场（大量位置 D≈0）的相对误差特性 + 线性 PCA 对非线性场的局限，
真实研究中的已知难点，后续可用 VAE 非线性降维替代 PCA。

## 接入真实数据（拿到研究版 Abaqus 后）

1. **阶段 0**: 用 `generate_cases.py` 生成 `--N 20 --M 50` 工况，批量提交 Abaqus 求解。
2. **阶段 1**: 从 `.odb` 提取 U/S/DAMAGE 场，替代 `1_generate_synthetic.py` 的输出格式
   （每个 case 存成 `(T, D)` 的场矩阵，D = 节点数 × 分量数）。
3. 其余（PCA、训练、推理）**逻辑完全不变**，只需调整 `config.py` 里的
   `D_FIELD`（换成真实自由度，约 5 万 × 3）和 `K_U/K_S/K_D`（按方差率拐点选）。
