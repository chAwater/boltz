# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

Boltz 是一个生物分子相互作用预测模型家族（Boltz-1 和 Boltz-2）。Boltz-2 是最新版本，能同时预测复合物结构和结合亲和力，是第一个接近物理方法（FEP）精度的深度学习模型。

## 安装

```bash
# GPU 版本（推荐）
pip install -e ".[cuda]"

# CPU 版本
pip install -e .

# 安装 lint 工具
pip install -e ".[lint]"

# 安装测试工具
pip install -e ".[test]"
```

## 常用命令

### 推理预测

```bash
# 基本预测（自动生成 MSA）
boltz predict input.yaml --use_msa_server

# 带结合亲和力预测
boltz predict input.yaml --use_msa_server

# 使用推理时势函数改善结构质量
boltz predict input.yaml --use_msa_server --use_potentials

# 更多样本（更慢但更准）
boltz predict input.yaml --use_msa_server --recycling_steps 10 --diffusion_samples 25

# 旧 GPU 兼容模式
boltz predict input.yaml --no_kernels
```

### 训练

```bash
# 调试模式（单设备，无 DDP，无 wandb）
python scripts/train/train.py scripts/train/configs/structure.yaml debug=1

# 完整训练
python scripts/train/train.py scripts/train/configs/structure.yaml

# 训练 confidence 模型
python scripts/train/train.py scripts/train/configs/confidence.yaml
```

### 代码质量

```bash
# Lint 检查
ruff check src/

# 格式化
ruff format src/

# 运行测试（跳过慢速测试）
pytest -m "not slow"

# 运行全部测试
pytest
```

## 代码架构

### 整体结构

```
src/boltz/
├── main.py              # CLI 入口 (boltz predict 命令)
├── data/                # 数据处理流水线
│   ├── parse/           # 输入解析器（yaml, fasta, mmcif, pdb, a3m, csv）
│   ├── tokenize/        # 分词器（boltz.py=v1, boltz2.py=v2）
│   ├── feature/         # 特征化器
│   ├── crop/            # 裁剪策略
│   ├── filter/          # 数据过滤（static=预处理阶段, dynamic=训练阶段）
│   ├── module/          # PyTorch Lightning DataModule（inference/training，各有 v2 版本）
│   ├── sample/          # 采样策略（cluster, random, distillation）
│   ├── write/           # 输出写入器（mmcif, pdb）
│   ├── msa/             # MSA 生成（mmseqs2 API）
│   ├── types.py         # 核心数据类型定义
│   └── const.py         # 全局常量
└── model/
    ├── models/
    │   ├── boltz1.py    # Boltz-1 模型（LightningModule）
    │   └── boltz2.py    # Boltz-2 模型（LightningModule）
    ├── modules/         # 模型子模块
    │   ├── trunk.py / trunkv2.py        # InputEmbedder, MSAModule, TemplateModule, DistogramModule
    │   ├── diffusion.py / diffusionv2.py # 扩散过程（结构生成）
    │   ├── confidence.py / confidencev2.py # 置信度预测头
    │   ├── affinity.py                  # 结合亲和力预测头（仅 v2）
    │   └── encoders.py / encodersv2.py  # 原子编码器、位置编码器
    ├── layers/          # 基础层实现（pairformer, triangular attention/mult, attention）
    ├── loss/            # 损失函数（diffusion, confidence, distogram, bfactor）
    └── optim/           # 优化器工具（EMA, AlphaFold LR Scheduler）
```

### 版本命名规律

代码库同时维护 v1（Boltz-1）和 v2（Boltz-2）实现。v2 组件均带 `v2` 后缀（如 `trunkv2.py`, `featurizerv2.py`, `boltz2.py`）。默认命令运行最新的 Boltz-2 模型。

### 核心数据流（推理）

1. **解析**：`data/parse/yaml.py` 解析输入 YAML → 内部数据类型（`data/types.py`）
2. **预处理**：分词（`tokenize/`）→ 特征化（`feature/`）→ 填充（`pad.py`）
3. **推理**：DataModule（`data/module/inferencev2.py`）→ Boltz2 模型（`model/models/boltz2.py`）
4. **输出**：Writer（`data/write/writer.py`）写出 CIF/PDB 和 JSON 置信度/亲和力文件

### 核心数据流（训练）

数据预处理（`scripts/process/`）→ 训练数据集（`data/module/training.py`）→ Trainer（PyTorch Lightning）

### 输入格式

YAML 输入格式支持：
- **sequences**：`protein`/`dna`/`rna`（需提供 sequence 和 msa）、`ligand`（提供 smiles 或 ccd）
- **constraints**：`bond`（共价键）、`pocket`（结合口袋）、`contact`（接触约束）
- **templates**：结构模板（CIF/PDB）
- **properties**：`affinity`（结合亲和力，指定配体链 ID）

### 输出格式

```
out_dir/predictions/[input_name]/
├── [name]_model_0.cif              # 预测结构（mmcif 格式）
├── confidence_[name]_model_0.json  # 置信度分数（ptm, iptm, plddt 等）
├── affinity_[name].json            # 亲和力分数（affinity_pred_value, affinity_probability_binary）
├── pae_*.npz / pde_*.npz / plddt_*.npz  # 详细分数矩阵
```

### 框架与依赖

- **PyTorch Lightning**：模型训练和推理的主框架
- **Hydra**：配置管理（训练配置文件在 `scripts/train/configs/`）
- **RDKit**：小分子处理
- **cuEquivariance**：NVIDIA GPU 加速（可选，旧 GPU 用 `--no_kernels` 跳过）
- **wandb**：训练日志记录

### 模型权重缓存

默认缓存路径为 `~/.boltz`（可通过 `--cache` 或环境变量 `BOLTZ_CACHE` 修改）。首次运行时自动从 HuggingFace 或 model-gateway.boltz.bio 下载权重。
