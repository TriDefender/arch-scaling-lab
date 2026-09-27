"""Phase 0: GPT-2 (124M) from-scratch baseline on FineWeb-Edu.

nanoGPT-style single file. Auto dtype: bf16 (Ada+) or fp16+GradScaler (Turing), SDPA attention.
Logs step, loss, val loss, tok/s, peak VRAM, elapsed -> <script_dir>/log.csv (loss-vs-GPU-hour data).
Resume: --resume ckpt_last.pt
Usage: python train.py [--max-iters N] [--resume CKPT]
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- config ----------------
N_LAYER, N_HEAD, N_EMBD, BLOCK = 12, 12, 768, 1024
VOCAB = 50257
DROPOUT = 0.0
MICRO_BS = 8
GRAD_ACCUM = 4  # global batch = 32768 tokens
LR = 6e-4
WARMUP = 200
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_INTERVAL = 250
EVAL_ITERS = 40
CKPT_EVERY = 1000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.environ.get("ASL_OUT", SCRIPT_DIR)
LOG = os.path.join(CKPT_DIR, "log.csv")
DATA = os.path.join(SCRIPT_DIR, "data")

parser = argparse.ArgumentParser()
parser.add_argument("--max-iters", type=int, default=10000)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                    help="muon = hybrid Muon(2D hidden)+AdamW(rest), Moonlight-style")
parser.add_argument("--tb", choices=["on", "off"], default="on",
                    help="write TensorBoard events under <out>/tb/ (default on)")
parser.add_argument("--muon-lr", type=float, default=0.02)
args = parser.parse_args()

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
assert torch.cuda.is_available()

# ---------------- model ----------------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.attn_dropout = DROPOUT

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        k = k.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        v = v.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.proj = nn.Linear(4 * N_EMBD, N_EMBD)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.attn, self.mlp = CausalSelfAttention(), MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, N_EMBD)
        self.wpe = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss


model = GPT().to(dev)
nparams = sum(p.numel() for p in model.parameters())
print(f"params: {nparams/1e6:.1f}M", flush=True)

# ---- TensorBoard: scalars mirror log.csv; events under <out>/tb/run-<ts> ----
_WRITER = None

def tb_add(tag, val, step):
    global _WRITER
    if args.tb != "on":
        return
    if _WRITER is None:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(CKPT_DIR, "tb", time.strftime("run-%Y%m%d-%H%M%S"))
        _WRITER = SummaryWriter(log_dir=tb_dir)
        print(f"tensorboard: events -> {tb_dir}", flush=True)
    _WRITER.add_scalar(tag, val, step)

def make_optimizers():
    """Returns [(optimizer, base_lr)]. adamw: single fused AdamW.
    muon: Muon on 2D hidden matrices + fused AdamW on embeddings/head/1D params."""
    if args.optimizer == "adamw":
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            (no_decay if (p.ndim < 2 or "ln" in n) else decay).append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
            lr=LR, betas=(0.9, 0.95), eps=1e-8, fused=True,
        )
        return [(opt, LR)]
    from muon import Muon
    hidden2d, decay, no_decay = [], [], []
    for n, p in model.named_parameters():
        if p.ndim == 2 and not any(k in n for k in ("wte", "wpe", "lm_head")):
            hidden2d.append(p)
        else:
            (no_decay if (p.ndim < 2 or "ln" in n) else decay).append(p)
    muon_opt = Muon(hidden2d, lr=args.muon_lr, weight_decay=WEIGHT_DECAY)
    adamw_opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    n_muon = sum(p.numel() for p in hidden2d)
    print(f"optimizer: Muon lr={args.muon_lr} on {len(hidden2d)} matrices ({n_muon/1e6:.1f}M) + AdamW lr={LR} on rest", flush=True)
    return [(muon_opt, args.muon_lr), (adamw_opt, LR)]


optims = make_optimizers()
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))
print(f"amp dtype: {AMP_DTYPE}", flush=True)

# ---------------- data ----------------
train_data = np.memmap(f"{DATA}/train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap(f"{DATA}/val.bin", dtype=np.uint16, mode="r")
ntok = train_data.size
print(f"train tokens: {ntok/1e6:.1f}M, val tokens: {val_data.size/1e6:.1f}M", flush=True)


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK - 1, (MICRO_BS,))
    x = torch.stack([torch.from_numpy(data[i : i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x.pin_memory().to(dev, non_blocking=True), y.pin_memory().to(dev, non_blocking=True)


@torch.no_grad()
def eval_val():
    model.eval()
    losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
        _, loss = model(*get_batch("val"))
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def lr_at(it):
    if it < WARMUP:
        return LR * (it + 1) / WARMUP
    p = (it - WARMUP) / max(1, args.max_iters - WARMUP)
    return 0.1 * LR + 0.5 * LR * (1 + math.cos(math.pi * p)) * 0.9


# ---------------- resume ----------------
start_iter, best_val = 0, float("inf")
if args.resume and os.path.exists(args.resume):
    ck = torch.load(args.resume, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    opt_states = ck["opt"] if isinstance(ck["opt"], list) else [ck["opt"]]
    if len(opt_states) != len(optims):
        print(f"WARN: ckpt has {len(opt_states)} optimizer state(s), current config has {len(optims)}; starting optimizer state fresh", flush=True)
    else:
        for o, s in zip((o for o, _ in optims), opt_states):
            o.load_state_dict(s)
    scaler.load_state_dict(ck["scaler"])
    start_iter, best_val = ck["iter"] + 1, ck["best_val"]
    print(f"resumed at iter {start_iter}", flush=True)

if start_iter == 0:
    with open(LOG, "w") as f:
        f.write("iter,loss,val_loss,tok_s,vram_gb,elapsed_s\n")

# ---------------- train ----------------
model.train()
t0 = time.time()
latest = start_iter
for it in range(start_iter, args.max_iters):
    lrf = lr_at(it) / LR
    for o, base in optims:
        for g in o.param_groups:
            g["lr"] = base * lrf

    for micro in range(GRAD_ACCUM):
        x, y = get_batch("train")
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            _, loss = model(x, y)
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
    for o, _ in optims:
        scaler.unscale_(o)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    for o, _ in optims:
        scaler.step(o)
    scaler.update()
    for o, _ in optims:
        o.zero_grad(set_to_none=True)
    latest = it

    if it % 50 == 0:
        torch.cuda.synchronize()
        el = time.time() - t0
        tok_s = (it - start_iter + 1) * MICRO_BS * GRAD_ACCUM * BLOCK / el
        vram = torch.cuda.max_memory_allocated() / 2**30
        print(f"it {it} | loss {loss.item()*GRAD_ACCUM:.3f} | {tok_s:.0f} tok/s | {vram:.1f}GB | {el/60:.0f}min", flush=True)
        tb_add("train/loss", loss.item() * GRAD_ACCUM, it)
        tb_add("perf/tok_s", tok_s, it)
        tb_add("perf/vram_gb", vram, it)

    if (it + 1) % EVAL_INTERVAL == 0:
        vl = eval_val()
        el = time.time() - t0
        tok_s = (it - start_iter + 1) * MICRO_BS * GRAD_ACCUM * BLOCK / el
        with open(LOG, "a") as f:
            f.write(f"{it},{loss.item()*GRAD_ACCUM:.4f},{vl:.4f},{tok_s:.0f},{torch.cuda.max_memory_allocated()/2**30:.2f},{el:.0f}\n")
        tb_add("val/loss", vl, it)
        tb_add("perf/tok_s", tok_s, it)
        print(f"== eval it {it}: val {vl:.4f} ({el/60:.0f}min) ==", flush=True)
        if vl < best_val:
            best_val = vl
            torch.save(
                {"model": model.state_dict(), "opt": [o.state_dict() for o, _ in optims], "scaler": scaler.state_dict(),
                 "iter": it, "best_val": best_val, "args": vars(args)},
                os.path.join(CKPT_DIR, "ckpt_best.pt"),
            )

    if (it + 1) % CKPT_EVERY == 0:
        torch.save(
            {"model": model.state_dict(), "opt": [o.state_dict() for o, _ in optims], "scaler": scaler.state_dict(),
             "iter": it, "best_val": best_val, "args": vars(args)},
            os.path.join(CKPT_DIR, "ckpt_last.pt"),
        )

torch.save({"model": model.state_dict(), "iter": latest, "best_val": best_val, "args": vars(args)}, os.path.join(CKPT_DIR, "ckpt_final.pt"))
print(f"DONE iters {start_iter}..{latest}, best val {best_val:.4f}, total {(time.time()-t0)/60:.0f}min", flush=True)
if _WRITER is not None:
    _WRITER.close()
