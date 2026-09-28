#!/usr/bin/env bash
# v6 ablation cell: GQA-4KV + AdamW + RoPE @ 4K (attention architecture vs v3 MHA)
# Isolates attention arch: identical to run_v3.sh except --attn-arch gqa --n-kv-heads 4.
# Params 152.8M (MHA 162.3M; KV-proj delta is arch-inherent, confound noted in README).
# KV cache/token/layer: 1024 B vs MHA 3072 B (3x reduction) - the inference-side payoff.
# Tokens-matched: 10000 iters x 32768 tok = 328M, same LR schedule & eval protocol.
# systemd unit: asl-v6-train (Restart=on-failure); auto-resumes from ckpt_last after crash
cd /home/openclaw/arch-scaling-lab
OUT=runs/v6-gqa4-rope4k
RESUME=""
[ -f "$OUT/ckpt_last.pt" ] && RESUME="--resume $OUT/ckpt_last.pt"
exec phase0-124m/.venv/bin/python phase0-124m/train_lc.py \
  --seq-len 4096 --pos rope --rope-base 10000 \
  --attn-arch gqa --n-kv-heads 4 --attn FLASH_ATTENTION \
  --optimizer adamw --lr 6e-4 \
  --micro-bs 2 --grad-accum 4 \
  --max-iters 10000 --warmup 200 --weight-decay 0.1 --grad-clip 1.0 \
  --eval-interval 250 --len-eval-interval 1000 \
  --data-dir /home/openclaw/arch-scaling-lab/phase0-124m/data \
  --out-dir /home/openclaw/arch-scaling-lab/$OUT $RESUME
