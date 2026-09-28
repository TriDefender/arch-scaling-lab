#!/usr/bin/env bash
# v4 ablation cell: AdamW + RoPE @ 1K (position-mechanism attribution)
# v1->v4 isolates wpe vs RoPE mechanism delta @1K; v4->v3 isolates pure length delta (1K->4K).
# Tokens-matched: 328M total (micro-bs/grad-accum auto to 32768 tok/step).
# systemd unit: asl-v4-train — NOT boot-enabled; launched by v3 harvest job after v3 completes
# (enabling early risks v3+v4 co-start OOM after a reboot).
cd /home/openclaw/arch-scaling-lab
OUT=runs/v4-adamw-rope1k
RESUME=""
[ -f "$OUT/ckpt_last.pt" ] && RESUME="--resume $OUT/ckpt_last.pt"
exec phase0-124m/.venv/bin/python phase0-124m/train_lc.py \
  --seq-len 1024 --pos rope --rope-base 10000 \
  --optimizer adamw --lr 6e-4 \
  --max-iters 10000 --warmup 200 --weight-decay 0.1 --grad-clip 1.0 \
  --eval-interval 250 --len-eval-interval 1000 \
  --data-dir /home/openclaw/arch-scaling-lab/phase0-124m/data \
  --out-dir /home/openclaw/arch-scaling-lab/$OUT $RESUME
