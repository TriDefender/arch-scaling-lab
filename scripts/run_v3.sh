#!/usr/bin/env bash
# v3 ablation cell: AdamW + RoPE @ 4K (optimizer attribution vs v2)
# Isolates Muon's contribution: identical to run_v2.sh except --optimizer adamw.
# Tokens-matched: 10000 iters x 32768 tok = 328M, same LR schedule & eval protocol.
# systemd unit: asl-v3-train (Restart=on-failure); auto-resumes from ckpt_last after crash
cd /home/openclaw/arch-scaling-lab
OUT=runs/v3-adamw-rope4k
RESUME=""
[ -f "$OUT/ckpt_last.pt" ] && RESUME="--resume $OUT/ckpt_last.pt"
exec phase0-124m/.venv/bin/python phase0-124m/train_lc.py \
  --seq-len 4096 --pos rope --rope-base 10000 \
  --optimizer adamw --lr 6e-4 \
  --micro-bs 2 --grad-accum 4 \
  --max-iters 10000 --warmup 200 --weight-decay 0.1 --grad-clip 1.0 \
  --eval-interval 250 --len-eval-interval 1000 \
  --data-dir /home/openclaw/arch-scaling-lab/phase0-124m/data \
  --out-dir /home/openclaw/arch-scaling-lab/$OUT $RESUME
