# ASL 暂停状态（2026-09-28 21:30Z）

**原因**: 用户需要使用电脑，手动暂停 v7 训练，GPU 让出。

## 冻结时状态

| 项 | 值 |
|---|---|
| v7 (MLA+AdamW+RoPE@4K) | 已停在 **it 5249 / 10000**（151min，19.0k tok/s，10.6GB） |
| val@4096 轨迹 | it 999: 5.815 → 2999: 4.777 → 4999: 4.270 → 5249: **4.2467**（当前 best） |
| checkpoint | `runs/v7-mla-rope4k/ckpt_last.pt` = it 4999；`ckpt_best.pt` = it 5249（val 4.2467） |
| systemd | `asl-v7-train` 已 stop + **disable**（WSL 重启不会自动开跑）；v6 unit enable 位也已摘 |
| GPU | 已释放（背景水位 7% / 1.4GB） |
| cron | `asl-v7-mid`(22:46) 与 `asl-v7-harvest`(01:36) 均已撤除，队列无 ASL 任务 |

注：续跑会从 ckpt_last (it 4999) 重放 ~250 步到 5249，无损。

## 恢复手册（用户一句话即可触发）

```bash
# 1. 续跑（自动从 it 4999 接力）
systemctl --user enable --now asl-v7-train
# 2. 观察
journalctl --user -u asl-v7-train -f
```

- 剩余 ~5000 it ≈ **2.4-2.5h**（19.0k tok/s），ETA 恢复时刻 +2.5h
- 训完自动进 len-eval（run_v7.sh 内嵌），终值预期 val@4096 ~3.95-4.00（见 18:33 半程外推）

## 完成后收割清单（勿忘）

1. bench：`phase0-124m/.venv` 解释器跑 qa_eval + len_eval（照抄 v6 收割命令，gqa→mla 命名空间）
2. README：三架构终表（MHA v3 3.6898 / GQA v6 3.7061 / MLA v7 待补）+ KV/参数量/吞吐三轴
3. git add/commit/push
4. WhatsApp 终报（显式 target `+8618913914281`）：三架构记分牌 + 32k 线选型建议
5. 32k 线决策点：MLA 若 PPL 落后 >0.2 → GQA 主力，MLA 挂"≥1B token 再议"

## 背景一句话

矩阵已完成：v1-v4（四方分解）+ v6 GQA（质量平价，KV 3×↓，QA 7/8 反超）。attention kernel bake-off 已完成（flash pin 入 train_lc.py）。v7 MLA 是架构轴最后一格。
