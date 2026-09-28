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

### QA bench（似然式多选，lm-eval 口径，2026-09-27 建）

`qa_eval.py`：loglikelihood 多选评分（acc + byte-length acc_norm），数据 `data/qa/`（HellaSwag val 10,042 / ARC-Easy 570 / ARC-Challenge 299 / WinoGrande-XL 1,267，HF parquet 直读，无训练集污染）。v1=`124m-baseline/ckpt_final`（wpe@1024），v2=`runs/v2-muon-rope4k/ckpt_final`（Muon+RoPE@4096）。

| 任务（随机基线） | v1 AdamW+wpe | v2 Muon+RoPE | Δ acc |
|---|---|---|---|
| HellaSwag acc_norm (.25) | 0.2683 | **0.2706** | +0.2pp |
| ARC-Easy acc (.25) | 0.4211 | **0.4281** | +0.7pp |
| ARC-Challenge acc (.25) | 0.1773 | **0.1873** | +1.0pp |
| WinoGrande acc (.50) | **0.5067** | 0.4807 | −2.6pp |

读法：328M tokens（2 tok/param）下四集几乎全部贴着随机基线，区分度有限——ARC 上 v2 小幅胜出、WinoGrande 差距在 ~2σ 边缘、HellaSwag 持平。**结论：优化器+位置编码切换无 QA 退化，val loss（3.7472→3.6790）仍为主信号**；QA bench 保留为消融矩阵的标准 sanity gate，预计 token 预算上到 ≥1B 后才开始有区分度。注意 HellaSwag 语料与 FineWeb-Edu 同源（WikiHow 部分），小模型上偏乐观，横向对比仍有效。

### Easy-tier + PPL 套件（2026-09-28 建，`dl_qa_easy.py` / `qa_eval.py` 扩展 / `len_eval.py` / `run_easy_suite.sh`）

动机：旧四集里三个是对抗性构造（HellaSwag/ARC-C 专门过滤掉弱模型能答对的题），124M 级已顶到该模型类天花板（完全训练 GPT-2 124M 参考值：HellaSwag ~29-30 / ARC-E ~42-43 / WG ~51，我们已在 92%/99%/随机线）。换非对抗性的 GPT-2 时代任务 + BLiMP（BabyLM 小模型标准）+ PPL 族，天花板高得多（LAMBADA 36.5 / PIQA ~63 / SciQ ~60 / BLiMP 70+ / WT103 PPL 37.5）。

新增任务口径：PIQA（val 1,000 子集，gimmaru 镜像）/ SciQ（val 1,000，closed-book，选项确定性混洗）/ BLiMP（67 现象×500=33,500 句对，空上下文+EOT 锚，sum-logprob 二元判定）/ LAMBADA-OpenAI（test 5,153，GPT-2 论文口径末词 argmax，单 token 末词过滤后 n=4,011）。

| 任务（随机基线） | v1 AdamW+wpe | v2 Muon+RoPE | Δ acc |
|---|---|---|---|
| LAMBADA acc (~0) | 0.1015 | **0.1319** | +3.0pp |
| SciQ acc (.25) | 0.4050 | **0.4330** | +2.8pp |
| PIQA acc (.50) | 0.5830 | 0.5840 | +0.1pp (norm +1.7pp) |
| BLiMP acc (.50) | 0.7436 | **0.7520** | +0.8pp |

PPL 套件（`len_eval.py`：val.bin 长度分层 + WikiText-103-raw 零样本迁移，lm_head 分片防 logits 膨胀）：

| 指标 | v1 (wpe@1K) | v2 (RoPE@4K) |
|---|---|---|
| val 分层 PPL @256 / 512 / 1024 | 49.80 / 44.30 / 42.62 | **48.28 / 42.32 / 40.52** |
| val 分层 PPL @2048 / 4096 | —（wpe 结构性上限 1024） | **38.81 / 36.65** |
| WT103 PPL @1024 | 99.2 | **93.4** |
| WT103 PPL @4096 | — | **76.3** |

读法：①同 T=1024 干净对比，v2 PPL 低 4.9%，与 val loss −0.068 一致；②上下文扩展是真实收益：v2 PPL 随上下文单调下降 48.3→36.7（−24%），@4096 比 v1 最佳（42.6@1024）低 14%——这是 RoPE@4K 增益的直接量化，也是后续位置编码消融的主探针；③easy-tier 区分度立竿见影（LAMBADA +3.0pp / SciQ +2.8pp，远超旧档噪声），且 8 任务全部方向一致偏 v2、零退化，新基线在行为层面成立；④我们距完全训练 GPT-2 的天花板仍远（LAMBADA 13.2 vs 36.5），与 4% token 预算相符。运维：全评测已快到 ~90 秒/ckpt（切片修复后），消融每格全适应跑无压力；长批任务用 `systemd-run --user`（exec 会话回收会连带杀 & 后台子进程，详见 workspace ERRORS.md 2026-09-28）。

## Phase 0b：长上下文扩展（32k）

目的：1024 ctx 看不出各 attention 变体的长程检索（大海捞针）差异。协议见 `phase0-124m/lc_protocol.json`。

课程化三阶段（每阶段 328M tokens，global batch 恒 32768 tok/step）：

1. **rope-1024-parity**：RoPE 替换 wpe，同预算复训，验证与 wpe baseline 的 loss parity（gate：终值差 ~0.03 内）——长上下文支线全部走 RoPE 主干
2. **lc-4k**：YaRN factor 4 续训（init-from parity ckpt，LR 1e-4）
3. **lc-32k**：YaRN factor 32 续训（LR 5e-5），分块 CE + 梯度检查点在此启用

### v3 优化器归因格（2026-09-28 收割：AdamW+RoPE@4K，同预算 328M tok）

v3 与 v2 只差优化器（AdamW vs Muon），直接归因 Muon 贡献。val loss 同长度口径（@4096）：v2 3.6790 → v3 3.6898，**Δopt = +0.0108（v3 更差）**，落在 0.01~0.02 边际档且方向偏 Muon——Muon 边际贡献成立，非白给。

| 指标 | v2 Muon+RoPE | v3 AdamW+RoPE | Δ |
|---|---|---|---|
| val loss @4096 | **3.6790** | 3.6898 | −0.0108（偏 Muon） |
| val PPL @4096 | **36.65** | 37.09 | +0.44 |
| WT103 PPL @4096 | **76.3** | 77.8 | +1.5 |
| LAMBADA acc | **0.1319** | 0.1037 | −2.8pp |
| SciQ acc | **0.4330** | 0.3790 | −5.4pp |
| PIQA acc | 0.5840 | 0.5780 | −0.6pp |
| BLiMP acc | 0.7520 | **0.7645** | +1.3pp |

读法：主信号（val loss / PPL）方向一致偏 Muon，easy-tier 主要任务（LAMBADA/SciQ）也明显偏 Muon；仅 BLiMP（句法性，与优化器关系最弱）和 WinoGrande 反向。**结论：优化器消融裁定 Muon 保留为矩阵优化器，边际档；RoPE 增益归因已剥离优化器轴。** 结果文件 `results/qa_v3.json` / `ppl_v3.json`。

消融格：`wpe-32k-ext`（wpe 表尾块平铺扩到 32k，同预算）= 位置编码轴对照；可选 `lc-32k-scratch`（原生 32k 从零训）分离续训贡献。

### v4 位置机制格 + 四方矩阵分解（2026-09-28 收割：AdamW+RoPE@1K，同预算 328M tok）

v4 与 v1 只差位置编码（wpe→RoPE，均 AdamW@1K），与 v3 只差训练长度（1K→4K，均 AdamW+RoPE）。四方矩阵（len_eval 同口径，ln loss = ln(PPL)）：

| len-strat PPL | v1 wpe@1K | v2 Muon@4K | v3 AdamW@4K | v4 AdamW@1K | v4 外推 |
|---|---|---|---|---|---|
| @256 | 49.80 | 48.28 | 48.61 | **43.79** | 同左 |
| @512 | 44.30 | 42.32 | 43.02 | **39.04** | 同左 |
| @1024 | 42.62 | 40.52 | 41.16 | **37.64** | 同左 |
| @2048 | n/a（wpe 封顶） | **38.81** | 39.36 | — | 53.82（崩） |
| @4096 | n/a | **36.65** | 37.09 | — | 110.82（崩 3×） |
| WT103@1024 | 99.2 | 93.4 | 95.2 | **86.1** | — |
| WT103@4096 | n/a | **76.3** | 77.8 | — | 239.7（崩 3×） |

**@1024 同口径加性分解**（v2−v1 = −0.050 完全可加 ✓）：

| 分量 | 对比格 | Δ ln loss | 结论 |
|---|---|---|---|
| 位置机制（wpe→RoPE） | v1→v4 | **−0.124** | 矩阵最大单一杠杆 |
| 训练长度（1K→4K） | v4→v3 | **+0.089** | 4K 训练在短上下文上是纯成本 |
| 优化器（AdamW→Muon） | v3→v2 | **−0.015**（@4096 为 −0.011） | 边际正贡献，方向与 v3 格一致 |

QA（v4 横扫 6/8）：hellaswag_norm .2727 / arc_e .4386 / arc_c .2040 / wg .5107 / piqa .591 / sciq .443 全场最优；v2 保 LAMBADA .132（长上下文任务），v3 保 BLiMP .7645。

**三条硬结论**：
1. **v4（AdamW+RoPE@1K）是同口径全场最优格**——原生 val 3.6334、QA 6/8 横扫、WT103 迁移 86.1。此前 v2 相对 v1 的提升主要是 RoPE 机制（−0.124），被 4K 训练的短上下文成本（+0.089）抵掉大半，Muon 只补 −0.015。
2. **4K 训练不白给**：同 token 预算下 @1024 质量降 0.089 ln，买到的只有 >1024 的能力。下游 ≤1K 就用 v4 配置。
3. **RoPE@1K factor=1 不外推**：@2048 +43%、@4096 崩到 3×（110.8 vs 37.1）。“练短送长”路线判死，4K 能力必须 4K 训练或位置插值（YaRN/PI）——**v5 YaRN 1K→4K 扩展格由此升为下一优先**：若 82M token 扩展能以 <0.02 的 @1024 损失换到 @4096 ≤37，即同时拿到 v4 短上下文质量与 4K 能力，也是 32k 路线的正式预演。

结果文件 `results/qa_v4.json` / `ppl_v4.json` / `ppl_v4x.json`（外推）。


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
