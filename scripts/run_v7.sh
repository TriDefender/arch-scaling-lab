#!/usr/bin/env bash
# v7 ablation cell: MLA + AdamW + RoPE @ 4K (attention architecture vs v3 MHA / v6 GQA)
# Isolates attention arch: identical to run_v3.sh except --attn-arch mla.
# DeepSeek-V2-style simplified MLA (no query latent compression): latent 256 (~d/3), rope 32.
# Params 155.7M (MHA 162.3M). KV cache/token/layer: 576 B vs MHA 3072 B (5.3x reduction).
# SDPA qk/v dim constraint handled by v-pad (exact, see MLAttention in train_lc.py).
# Tokens-matched: 10000 iters x 32768 tok = 328M, same LR schedule & eval protocol.
# systemd unit: asl-v7-train (kept disabled; fired by v6 harvest after GPU frees)
cd /home/openclaw/arch-scaling-lab
OUT=runs/v7-mla-rope4k
RESUME=""
[ -f "$OUT/ckpt_last.pt" ] && RESUME="--resume $OUT/ckpt_last.pt"
exec phase0-124m/.venv/bin/python phase0-124m/train_lc.py \
  --seq-len 4096 --pos rope --rope-base 10000 \
  --attn-arch mla --mla-latent 256 --mla-rope-dim 32 --attn FLASH_ATTENTION \
  --optimizer adamw --lr 6e-4 \
  --micro-bs 2 --grad-accum 4 \
  --max-iters 10000 --warmup 200 --weight-decay 0.1 --grad-clip 1.0 \
  --eval-interval 250 --len-eval-interval 1000 \
  --data-dir /home/openclaw/arch-scaling-lab/phase0-124m/data \
  --out-dir /home/openclaw/arch-scaling-lab/$OUT $RESUME
