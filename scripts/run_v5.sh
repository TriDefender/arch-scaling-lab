#!/usr/bin/env bash
# v5 (NEXT CELL, pending owner go): YaRN-style 1K->4K extension rehearsal on top of v4.
# Muon verdict 2026-09-28: marginal (-0.015 vs AdamW) -> optimizer axis defaults to AdamW (>=1B re-check).
# Attention pinned FLASH_ATTENTION per attn_bench.py 2026-09-28 (math fallback = 16x slower at 4K).
# Accept criteria (pre-registered): @1024 val regression <0.02 vs v4 AND @4096 PPL <= v3's 37.1.
# init-from v4 final weights; RoPE re-anchored orig=1024, factor=4; YaRN temp on by default.
# Budget: 2500 iters x 32768 = 82M tokens (~1.5h @4K) — extension-cost class, NOT tokens-matched.
# Harvest agent may calibrate LR/iters before launch; do not enable at boot.
cd /home/openclaw/arch-scaling-lab
OUT=runs/v5-yarn-1k-to-4k
RESUME=""
[ -f "$OUT/ckpt_last.pt" ] && RESUME="--resume $OUT/ckpt_last.pt"
exec phase0-124m/.venv/bin/python phase0-124m/train_lc.py \
  --seq-len 4096 --pos rope --rope-base 10000 --attn FLASH_ATTENTION \
  --rope-orig-len 1024 --rope-factor 4 \
  --init-from runs/v4-adamw-rope1k/ckpt_final.pt \
  --optimizer adamw --lr 1e-4 --warmup 100 \
  --micro-bs 2 --grad-accum 4 \
  --max-iters 2500 --weight-decay 0.1 --grad-clip 1.0 \
  --eval-interval 100 --len-eval-interval 250 \
  --data-dir /home/openclaw/arch-scaling-lab/phase0-124m/data \
  --out-dir /home/openclaw/arch-scaling-lab/$OUT $RESUME
