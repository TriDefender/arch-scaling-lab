"""Long-context variant of the Phase 0 124M baseline: RoPE/YaRN, chunked CE, grad checkpointing.

Self-contained (nanoGPT style), mirrors train.py optimizer recipe. Key differences:
  - Position: RoPE (default, zero params) or learnable wpe; YaRN NTK-by-parts factor for extension
  - seq_len is a pure view parameter over the flat token stream -- NO data repack needed
  - Chunked + checkpointed cross-entropy (never materializes full T x VOCAB logits:
    32768 x 50257 would be 3.3 GB bf16, x2-3 with autograd)
  - Per-block activation checkpointing (auto when seq_len > --grad-ckpt-thresh)
  - Multi-length val eval -> eval_lengths.csv (loss-vs-context-length curve)

Usage:
  parity:  python train_lc.py --seq-len 1024 --max-iters 10000 --out-dir runs/rope-1024
  ext 4k:  python train_lc.py --seq-len 4096 --rope-factor 4 --lr 1e-4 --warmup 100 \
             --init-from runs/rope-1024/ckpt_final.pt --out-dir runs/lc-4k
  ext 32k: python train_lc.py --seq-len 32768 --rope-factor 32 --lr 5e-5 --warmup 100 --micro-bs 1 \
             --init-from runs/lc-4k/ckpt_final.pt --out-dir runs/lc-32k
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

VOCAB = 50257
DROPOUT = 0.0

# ---------------- args ----------------
p = argparse.ArgumentParser()
p.add_argument("--seq-len", type=int, default=1024)
p.add_argument("--pos", choices=["rope", "wpe"], default="rope")
p.add_argument("--rope-base", type=float, default=10000.0)
p.add_argument("--rope-factor", type=float, default=1.0, help="YaRN scale s = target_ctx / base_ctx")
p.add_argument("--rope-orig-len", type=int, default=1024, help="context the base model was trained at (YaRN anchors)")
p.add_argument("--no-yarn-temp", action="store_true", help="disable YaRN attention temperature 1/(0.1*ln s + 1)")
p.add_argument("--n-layer", type=int, default=12)
p.add_argument("--n-head", type=int, default=12)
p.add_argument("--n-embd", type=int, default=768)
p.add_argument("--micro-bs", type=int, default=0, help="0 = auto by seq length")
p.add_argument("--grad-accum", type=int, default=0, help="0 = auto to --global-tokens")
p.add_argument("--global-tokens", type=int, default=32768)
p.add_argument("--max-iters", type=int, default=10000)
p.add_argument("--lr", type=float, default=6e-4)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--weight-decay", type=float, default=0.1)
p.add_argument("--grad-clip", type=float, default=1.0)
p.add_argument("--eval-interval", type=int, default=250)
p.add_argument("--eval-iters", type=int, default=0, help="full-length val crops; 0 = auto")
p.add_argument("--len-eval-interval", type=int, default=1000)
p.add_argument("--ce-chunk", type=int, default=0, help="tokens per CE chunk; 0 = auto")
p.add_argument("--grad-ckpt-thresh", type=int, default=4096)
p.add_argument("--no-grad-ckpt", action="store_true")
p.add_argument("--data-dir", default=None)
p.add_argument("--out-dir", default=None)
p.add_argument("--resume", default=None, help="full-state resume (model+opt+iter)")
p.add_argument("--init-from", default=None, help="weights-only init; allows pos/len changes")
p.add_argument("--extend-wpe", action="store_true", help="with --init-from: tile old wpe table to new seq_len")
args = p.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = args.data_dir or os.path.join(SCRIPT_DIR, "data")
OUT = args.out_dir or os.environ.get("ASL_OUT", SCRIPT_DIR)
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "log.csv")
LENLOG = os.path.join(OUT, "eval_lengths.csv")
T = args.seq_len
HEAD = args.n_embd // args.n_head
MICRO_BS = args.micro_bs or (8 if T <= 2048 else 4 if T <= 4096 else 2 if T <= 8192 else 1)
GRAD_ACCUM = args.grad_accum or max(1, round(args.global_tokens / (MICRO_BS * T)))
CE_CHUNK = args.ce_chunk or min(8192, MICRO_BS * T)
USE_CKPT = (not args.no_grad_ckpt) and T > args.grad_ckpt_thresh
EVAL_ITERS = args.eval_iters or max(8, min(40, 2 ** 18 // T))

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
assert torch.cuda.is_available()

print(
    f"config: seq={T} micro={MICRO_BS} accum={GRAD_ACCUM} "
    f"global={MICRO_BS * T * GRAD_ACCUM} tok/step | pos={args.pos} factor={args.rope_factor} | "
    f"ce_chunk={CE_CHUNK} block_ckpt={USE_CKPT} eval_iters={EVAL_ITERS}",
    flush=True,
)

# ---------------- rope ----------------
def build_rope(head_dim, max_len, base, factor, orig_len):
    """cos/sin tables (max_len, head_dim//2). factor>1 -> YaRN NTK-by-parts on wavelengths."""
    inv = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    if factor > 1.0:
        lam = 2 * math.pi / inv                      # per-frequency wavelength
        hi, lo = orig_len / 2.0, max_len / 2.0       # ramp anchors on wavelength
        m = 1.0 + (factor - 1.0) * torch.clamp((lam - hi) / (lo - hi), 0.0, 1.0)
        inv = inv / m                                # high-freq unchanged, low-freq fully interpolated
    t = torch.arange(max_len, dtype=torch.float32)
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):  # x: (B, H, T, D)
    Tn = x.size(-2)
    c = cos[:Tn].view(1, 1, Tn, -1)
    s = sin[:Tn].view(1, 1, Tn, -1)
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).to(x.dtype)

# ---------------- model ----------------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(args.n_embd, 3 * args.n_embd)
        self.proj = nn.Linear(args.n_embd, args.n_embd)
        self.attn_dropout = DROPOUT
        scale = 1.0 / math.sqrt(HEAD)
        if args.pos == "rope" and args.rope_factor > 1.0 and not args.no_yarn_temp:
            scale /= 0.1 * math.log(args.rope_factor) + 1.0  # YaRN attention temperature
        self.sdpa_scale = scale

    def forward(self, x, cos, sin):
        B, Tn, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, Tn, args.n_head, HEAD).transpose(1, 2)
        k = k.view(B, Tn, args.n_head, HEAD).transpose(1, 2)
        v = v.view(B, Tn, args.n_head, HEAD).transpose(1, 2)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.sdpa_scale,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, Tn, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(args.n_embd, 4 * args.n_embd)
        self.proj = nn.Linear(4 * args.n_embd, args.n_embd)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(args.n_embd), nn.LayerNorm(args.n_embd)
        self.attn, self.mlp = CausalSelfAttention(), MLP()

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_mode = args.pos
        self.wte = nn.Embedding(VOCAB, args.n_embd)
        if args.pos == "wpe":
            self.wpe = nn.Embedding(T, args.n_embd)
        else:
            cos, sin = build_rope(HEAD, T, args.rope_base, args.rope_factor, args.rope_orig_len)
            self.register_buffer("rope_cos", cos, persistent=False)  # not in state_dict
            self.register_buffer("rope_sin", sin, persistent=False)
        self.blocks = nn.ModuleList([Block() for _ in range(args.n_layer)])
        self.ln_f = nn.LayerNorm(args.n_embd)
        self.lm_head = nn.Linear(args.n_embd, VOCAB, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, Tn = idx.shape
        x = self.wte(idx)
        if self.pos_mode == "wpe":
            x = x + self.wpe(torch.arange(Tn, device=idx.device))
        cos = sin = None
        if self.pos_mode == "rope":
            cos, sin = self.rope_cos, self.rope_sin
        for blk in self.blocks:
            if self.training and USE_CKPT:
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        x = self.ln_f(x)
        if targets is None:
            logits = self.lm_head(x)
            return logits, None
        # chunked CE: no full (B*T, VOCAB) logits ever materialized for backward
        hf = x.reshape(-1, x.size(-1))
        tf = targets.reshape(-1)
        total = hf.new_zeros((), dtype=torch.float32)
        for i in range(0, hf.size(0), CE_CHUNK):
            hl, tl = hf[i : i + CE_CHUNK], tf[i : i + CE_CHUNK]

            def _ce(hl, tl):
                return F.cross_entropy(self.lm_head(hl).float(), tl, reduction="sum")

            total = total + checkpoint(_ce, hl, tl, use_reentrant=False)
        return None, total / hf.size(0)


model = GPT().to(dev)
nparams = sum(q.numel() for q in model.parameters())
print(f"params: {nparams/1e6:.1f}M", flush=True)

decay, no_decay = [], []
for n, q in model.named_parameters():
    (no_decay if (q.ndim < 2 or "ln" in n) else decay).append(q)
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True,
)
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))
print(f"amp dtype: {AMP_DTYPE}", flush=True)

# ---------------- data (flat stream; seq_len = view parameter) ----------------
train_data = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(DATA, "val.bin"), dtype=np.uint16, mode="r")
print(f"train tokens: {train_data.size/1e6:.1f}M, val tokens: {val_data.size/1e6:.1f}M", flush=True)


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - T - 1, (MICRO_BS,))
    x = torch.stack([torch.from_numpy(data[i : i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + T].astype(np.int64)) for i in ix])
    return x.pin_memory().to(dev, non_blocking=True), y.pin_memory().to(dev, non_blocking=True)


LEN_EVAL = sorted({l for l in (1024, T // 16, T // 4, T) if l <= T})


@torch.no_grad()
def eval_at(L, K):
    """Fixed-seed val loss at window length L (same crops every call -> comparable)."""
    model.eval()
    g = torch.Generator().manual_seed(42 + L)
    ix = torch.randint(0, len(val_data) - L - 1, (K,), generator=g)
    tot = 0.0
    for i in ix:
        x = torch.from_numpy(val_data[i : i + L].astype(np.int64))[None].pin_memory().to(dev, non_blocking=True)
        y = torch.from_numpy(val_data[i + 1 : i + 1 + L].astype(np.int64))[None].pin_memory().to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / len(ix)


def lr_at(it):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    frac = (it - args.warmup) / max(1, args.max_iters - args.warmup)
    return 0.1 * args.lr + 0.5 * args.lr * (1 + math.cos(math.pi * frac)) * 0.9

# ---------------- resume / init-from ----------------
start_iter, best_val = 0, float("inf")
if args.resume and os.path.exists(args.resume):
    ck = torch.load(args.resume, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    scaler.load_state_dict(ck["scaler"])
    start_iter, best_val = ck["iter"] + 1, ck["best_val"]
    print(f"resumed at iter {start_iter} (ckpt args: {ck.get('args')})", flush=True)
elif args.init_from and os.path.exists(args.init_from):
    ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    if args.pos == "wpe" and args.extend_wpe and "wpe.weight" in sd:
        old = sd["wpe.weight"]
        n = T - old.size(0)
        if n > 0:
            sd["wpe.weight"] = torch.cat([old, old[-1:].repeat(n, 1)], 0)
            print(f"wpe extended: {old.size(0)} -> {T} rows (tail-tiled)", flush=True)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"init-from {args.init_from} (iter {ck.get('iter')}): missing={list(miss)} unexpected={list(unexp)}", flush=True)

if start_iter == 0:
    with open(LOG, "w") as f:
        f.write("iter,loss,val_loss,tok_s,vram_gb,elapsed_s\n")

# ---------------- train ----------------
model.train()
t0 = time.time()
latest = start_iter
tok_per_step = MICRO_BS * T * GRAD_ACCUM
for it in range(start_iter, args.max_iters):
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    for _ in range(GRAD_ACCUM):
        x, y = get_batch("train")
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            _, loss = model(x, y)
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    latest = it

    if it % 50 == 0:
        torch.cuda.synchronize()
        el = time.time() - t0
        tok_s = (it - start_iter + 1) * tok_per_step / el
        print(f"it {it} | loss {loss.item()*GRAD_ACCUM:.3f} | {tok_s:.0f} tok/s | "
              f"{torch.cuda.max_memory_allocated()/2**30:.1f}GB | {el/60:.0f}min", flush=True)

    if (it + 1) % args.eval_interval == 0:
        vl = eval_at(T, EVAL_ITERS)
        el = time.time() - t0
        tok_s = (it - start_iter + 1) * tok_per_step / el
        with open(LOG, "a") as f:
            f.write(f"{it},{loss.item()*GRAD_ACCUM:.4f},{vl:.4f},{tok_s:.0f},"
                    f"{torch.cuda.max_memory_allocated()/2**30:.2f},{el:.0f}\n")
        print(f"== eval it {it}: val@{T} {vl:.4f} ({el/60:.0f}min) ==", flush=True)
        if vl < best_val:
            best_val = vl
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                 "iter": it, "best_val": best_val, "args": vars(args)},
                os.path.join(OUT, "ckpt_best.pt"),
            )

    if (it + 1) % args.len_eval_interval == 0:
        row = {L: eval_at(L, max(2, min(64, 2 ** 18 // L))) for L in LEN_EVAL}
        if not os.path.exists(LENLOG):
            with open(LENLOG, "w") as f:
                f.write("iter," + ",".join(f"val@{l}" for l in LEN_EVAL) + "\n")
        with open(LENLOG, "a") as f:
            f.write(f"{it}," + ",".join(f"{row[l]:.4f}" for l in LEN_EVAL) + "\n")
        print(f"~~ len-eval it {it}: " + " ".join(f"@{l}={row[l]:.3f}" for l in LEN_EVAL), flush=True)

    if (it + 1) % 1000 == 0:
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
             "iter": it, "best_val": best_val, "args": vars(args)},
            os.path.join(OUT, "ckpt_last.pt"),
        )

torch.save({"model": model.state_dict(), "iter": latest, "best_val": best_val, "args": vars(args)},
           os.path.join(OUT, "ckpt_final.pt"))
print(f"DONE iters {start_iter}..{latest}, best val {best_val:.4f}, total {(time.time()-t0)/60:.0f}min", flush=True)
