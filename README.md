# arch-scaling-lab

架构消融 + scaling 外推实验室。核心研究问题：**小规模（单卡 16GB）上胜出的架构变体，其优势能否外推到更大尺度？**

## 研究设计

1. **Phase 0（单卡 T4 16GB）**：固定数据（FineWeb-Edu sample-10BT）+ tokenizer（GPT-2 BPE）+ 优化器（AdamW）+ token 预算，from-scratch 训练 GPT-2-124M baseline 与各架构变体，按 compute-matched 协议比较
2. **Phase 1**：入围变体 × 3 seeds 复跑，360M 尺度确认
3. **Phase 2（外部集群）**：1B gate → scaling curve 拟合 → 7B 验证外推

对照原则：**自己训 dense baseline 做科学对照**，公开 checkpoint（SmolLM2 系列，同数据系）只做 sanity check。

## Phase 0 当前状态

| 项 | 值 |
|---|---|
| 模型 | GPT-2 124M（163M 含 wpe），12L/12H/768d/1024 ctx |
| 数据 | FineWeb-Edu sample-10BT 前 2 shards → train 1.50B tokens + val 2.2M tokens（GPT-2 BPE, uint16 bins） |
| 训练 | fp16 + GradScaler（T4 无 bf16）、SDPA、fused AdamW、global batch 32768 tokens、cosine LR 6e-4、10000 iters ≈ 328M tokens |
| 吞吐 | ~15.3k tok/s（T4, GPU 100%, 峰值 9.9GB） |

## 文件

- `phase0-124m/prep_data.py` — 流式 tokenize（12GB RAM 安全，逐 batch 读 parquet，2 vCPU × tiktoken）
- `phase0-124m/train.py` — nanoGPT 风格单文件训练器，带 loss-vs-GPU-hour CSV 日志（log.csv：iter/loss/val_loss/tok_s/vram/elapsed）、best-val 与周期 checkpoint、resume

## 已知坑（Colab free T4）

- VM 只有 12GB RAM：禁止整 shard `to_pylist()`，必须流式
- `colab exec` 的 stdin 是 **IPython kernel 不是 shell**：跑 shell 命令用 `subprocess.run(['bash','-lc',...])` 包装
- 长任务一律 tmux，断连不死；checkpoint 每 1000 iter 落盘支持 resume
