# arch-scaling-lab

架构消融 + scaling 外推实验室。核心研究问题：**小规模（单卡 16GB）上胜出的架构变体，其优势能否外推到更大尺度？**

## 研究设计

1. **Phase 0（单卡 16GB：T4 → 4060 Ti）**：固定数据（FineWeb-Edu sample-10BT）+ tokenizer（GPT-2 BPE）+ token 预算，from-scratch 训练 GPT-2-124M baseline 与各架构变体；优化器矩阵级统一（AdamW 或 Muon，见对比口径）
2. **Phase 1**：入围变体 × 3 seeds 复跑，360M 尺度确认
3. **Phase 2（外部集群）**：1B gate → scaling curve 拟合 → 7B 验证外推

对照原则：**自己训 dense baseline 做科学对照**，公开 checkpoint（SmolLM2 系列，同数据系）只做 sanity check。

## 对比口径（2026-09-27 修订）

- **主对比轴 = 推理 compute**：结论以「质量 vs 推理 profile」呈现——非嵌入 params、FLOPs/token@1k 与 @32k、KV cache bytes@32k、4060 Ti bf16 实测 prefill/decode tok/s（harness 待建）。KV 压缩类变体（GQA/MLA）的赢面直接体现在 profile 上，不必等 KV 容量压力出现。
- **训练侧只要求「配方统一」，不要求训练 FLOPs 匹配**：全矩阵同数据、同 token 数、同调度、同优化器（矩阵级统一，禁止部分变体单独换）。同 tokens ≠ 同训练 FLOPs（32k 下 full attention 训练成本远高于线性/SWA）——按部署框架各架构自付训练成本，但 log 已记录 tok/s 与时长，两种读法都成立。
- **优化器轴（独立消融格）**：baseline（AdamW）完成后用 `--optimizer muon` 同预算复训一格对照；Muon 在 124M/328M tokens 的收益是开放问题（文献 ~2× 证据在 compute-optimal 大规模）。矩阵最终优化器由该格决定；切换必须矩阵级 + baseline 复训，禁止中途热换（优化器状态不通用）。
- **undertrained 风险 gate**：328M tokens（≈2.6 tok/param）深处排序可能随预算翻转；Phase 1 入围变体须在 ≥1B tokens 复验后才能下架构结论——推理轴的干净不能替质量轴背书。

## Phase 0 当前状态

| 项 | 值 |
|---|---|
| 模型 | GPT-2 124M（163M 含 wpe），12L/12H/768d/1024 ctx |
| 数据 | FineWeb-Edu sample-10BT 前 2 shards → train 1.50B tokens + val 2.2M tokens（GPT-2 BPE, uint16 bins） |
| 训练 | bf16（4060 Ti Ada 原生）/ fp16+GradScaler（T4）、SDPA、fused AdamW、global batch 32768 tokens、warmup 200 + cosine LR 6e-4→0.1x、10000 iters ≈ 328M tokens |
| 吞吐 | ~29k tok/s（4060 Ti, bf16, GPU 100%, 峰值 9.9GB）；~15.3k tok/s（T4 参照） |
| **baseline 结果（2026-09-27 完成）** | 10000/10000 iters（328M tokens）；**best val loss 3.7472 @iter 8999**（终值 3.7528）；val 曲线：250→6.32, 1000→5.18, 3000→4.28, 5000→4.01, 7500→3.81, 9999→3.75 |
| 总 GPU 时长 / 吞吐 | ≈3.4h（含 2 次中断续训）；全程平均 ~27.5k tok/s（续训段 ~27k，属 resume 前段 29k 正常衰减区间） |
| 产物 | `phase0-124m/runs/124m-baseline/`：ckpt_best/ckpt_last/ckpt_final.pt、log.csv（iter≤1499）、log_resume_1500_10000.csv、stdout.log（旧 resume 前 ckpt 留 .pre_resume.bak） |

## Phase 0b：长上下文扩展（32k）

目的：1024 ctx 看不出各 attention 变体的长程检索（大海捞针）差异。协议见 `phase0-124m/lc_protocol.json`。

课程化三阶段（每阶段 328M tokens，global batch 恒 32768 tok/step）：

1. **rope-1024-parity**：RoPE 替换 wpe，同预算复训，验证与 wpe baseline 的 loss parity（gate：终值差 ~0.03 内）——长上下文支线全部走 RoPE 主干
2. **lc-4k**：YaRN factor 4 续训（init-from parity ckpt，LR 1e-4）
3. **lc-32k**：YaRN factor 32 续训（LR 5e-5），分块 CE + 梯度检查点在此启用

消融格：`wpe-32k-ext`（wpe 表尾块平铺扩到 32k，同预算）= 位置编码轴对照；可选 `lc-32k-scratch`（原生 32k 从零训）分离续训贡献。

评估：`eval_lengths.csv` 记录 1k/2k/8k/32k 多窗口 val loss（loss-vs-context 曲线）；NIAH 谜题独立 harness（待建）。注意 MHA/GQA/MLA 的 KV 差异在 32k/16GB 下不设 KV 容量压力，预期不分高下，该轴需 128k+ 或等 KV 预算 + 驱逐。

工程要点：train.bin 是扁平 token 流，**seq_len 是纯视图参数，4k/32k 不需要重新打包数据**；32k 时 logits 若整体物化需 3.3GB bf16（autograd 下 ×2-3），train_lc.py 用分块 + checkpoint 的 CE 规避，micro-bs 1 显存 ~7GB。

## 文件

- `phase0-124m/prep_data.py` — 流式 tokenize（12GB RAM 安全，逐 batch 读 parquet，2 vCPU × tiktoken）
- `phase0-124m/train.py` — nanoGPT 风格单文件训练器，带 loss-vs-GPU-hour CSV 日志（log.csv：iter/loss/val_loss/tok_s/vram/elapsed）、best-val 与周期 checkpoint、resume、`--optimizer adamw|muon`（checkpoint 内 opt state 存为 list，新旧格式互兼容读取）
- `phase0-124m/muon.py` — Muon 优化器（Newton-Schulz 正交化，混合约定：2D hidden 矩阵走 Muon，embedding/head/1D 走 AdamW）
- `phase0-124m/train_lc.py` — 长上下文版训练器：RoPE/YaRN（NTK-by-parts + attention 温度）、wpe 可选 + `--extend-wpe` 尾块平铺扩表、分块+checkpoint 交叉熵、块级激活检查点（阈值可配）、多长度 val eval（eval_lengths.csv）、init-from 支持跨位置编码/长度热启动；seq_len 为视图参数，数据无需重打包
- `phase0-124m/prep_data_lc.py` — 长上下文数据就绪校验（token id 范围、EOT 密度、各 seq_len 窗口统计）→ data_manifest.json
- `phase0-124m/lc_protocol.json` — 32k 扩展协议：三阶段命令/预算/ETA + 消融格 + 评估协议

## 已知坑（Colab free T4）

- VM 只有 12GB RAM：禁止整 shard `to_pylist()`，必须流式
- `colab exec` 的 stdin 是 **IPython kernel 不是 shell**：跑 shell 命令用 `subprocess.run(['bash','-lc',...])` 包装
- 长任务一律 tmux，断连不死；checkpoint 每 1000 iter 落盘支持 resume
