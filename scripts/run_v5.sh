#!/usr/bin/env bash
# v5 cell: YaRN 1K->4K context extension (position-mechanism payoff test)
# Inits weights-only from v4 (AdamW+RoPE@1K, val 3.6334); NO optimizer state (fresh).
# YaRN: rope-factor 4.0 (4096/1024), NTK-by-parts + attention temp 1/(0.1*ln s + 1).
# LR 2e-4 = 1/3 of pretrain peak (extension convention: preserve converged state).
# 2500 iters x 32768 tok = 82M tokens (~75 min). flash pinned (bake-off 6c99798).
# Pre-registered gates (README): @1024 val regression <0.02 (i.e. <=3.6534)
#   AND @4096 len-strat PPL <= 37 (v3 native 37.1).
# systemd unit: asl-v5-train (manual start; 30m watchdog automation guards it).
cd /home/openclaw/arch-scaling-lab
OUT=runs/v5-yarn-1k-to-4k
RESUME=""
[ -f "/home/openclaw/arch-scaling-lab/$OUT/ckpt_last.pt" ] && RESUME="--resume /home/openclaw/arch-scaling-lab/$OUT/ckpt_last.pt"
# note: --resume takes precedence over --init-from in train_lc.py; RESUME points at v5's
# own ckpt_last (iter 0-2499) so watchdog restarts continue, never re-run from v4 weights.
exec phase0-124m/.venv/bin/python phase0-124m/train_lc.py \
  --seq-len 4096 --pos rope --rope-base 10000 \
  --rope-factor 4.0 --rope-orig-len 1024 \
  --init-from /home/openclaw/arch-scaling-lab/runs/v4-adamw-rope1k/ckpt_final.pt \
  --attn FLASH_ATTENTION \
  --optimizer adamw --lr 2e-4 \
  --micro-bs 2 --grad-accum 4 \
  --max-iters 2500 --warmup 100 --weight-decay 0.1 --grad-clip 1.0 \
  --eval-interval 250 --len-eval-interval 1000 \
  --data-dir /home/openclaw/arch-scaling-lab/phase0-124m/data \
  --out-dir /home/openclaw/arch-scaling-lab/$OUT $RESUME
