# arch-scaling-lab

架构消融与 scaling 外推实验。核心问题：**小规模（124M–360M）上胜出的注意力/架构变体，能否外推到更大尺寸？**

完整研究设计见最初讨论（compute-matched 对比、同数据同 tokenizer、自训 dense 对照、3 seeds、1B gate、曲线拟合外推）。

## Phase 0 — 124M baseline 复现（Colab T4）

目标：打通 from-scratch 预训练全链路，并给出 SmolLM2-135M 量级的参考基线。

- 模型：GPT-2 small 拓扑（12L/12H/768d, ctx 1024, vocab 50304, tied embeddings, dropout 0），≈124M 非全量参数
- 数据：FineWeb-Edu `sample/10BT`（与 SmolLM2 同源数据族），GPT-2 BPE，uint16 bins
- 训练：nanoGPT 风格单文件，fp16 autocast + GradScaler（T4 无 bf16），SDPA attention，无 compile
  - batch：32 micro × 1024 ctx × 8 accum = 262,144 tokens/step
  - AdamW (0.9, 0.95), wd 0.1 (2D only), grad clip 1.0
  - LR：8e-4 peak，40 step warmup，cosine → 6e-5
- 预算：600 步 ≈ 157M tokens（T4 上 ~4h）；`--max-minutes` 熔断保护免费会话

### 文件

| 文件 | 用途 |
|---|---|
| `prep_data.py` | 下载 2 个 10BT shard（~1B tokens）→ tokenize → `/content/data/{train,val}.bin` |
| `train_124m.py` | 训练 + 周期 eval + sample 生成 + checkpoint + loss 曲线 |
| `notebooks/` | Colab 会话导出的 notebook（含完整执行输出） |
| `results/` | loss 曲线、summary.json |

### 结果

（训练进行中，完成后回填）

### 已知偏差说明

- 这是 pipeline smoke run，157M tokens 远低于 Chinchilla 最优（124M ≈ 2.5B tokens，T4 需 ~2 天，免费会话不现实）
- 参考系对照：SmolLM2-135M 用同族数据训练 ~11T tokens（1.9T 主 + decay），规模不可比，只用于 sanity check 数据/代码正确性

## Roadmap

- Phase 1（16GB 单卡）：124M 架构消融初筛（KV 结构 / 线性注意力混合 / 稀疏模式 / 位置编码 / ±QK-Norm），3 seeds 复跑 top-3
- Phase 2（单卡）：360M 确认实验
- Phase 3（集群）：1B gate → scaling curve 拟合 → 7B 验证外推
