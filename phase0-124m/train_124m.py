"""Phase 0: GPT-2 124M from scratch on FineWeb-Edu, single T4.

nanoGPT-style, fp16 autocast (Turing has no bf16), SDPA attention, no compile.
Config: 12L/12H/768d, ctx 1024, vocab 50304, tied embeddings, dropout 0.
Total batch = 32 micro x 1024 ctx x 8 accum = 262,144 tokens/step.
Default budget: 600 steps (~157M tokens) or MAX_MINUTES wall clock, whichever first.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- args ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/content/data")
ap.add_argument("--out", default="/content/out")
ap.add_argument("--max-steps", type=int, default=600)
ap.add_argument("--max-minutes", type=float, default=210.0)
ap.add_argument("--micro-bsz", type=int, default=32)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=8e-4)
ap.add_argument("--min-lr", type=float, default=6e-5)
ap.add_argument("--warmup", type=int, default=40)
ap.add_argument("--eval-every", type=int, default=50)
ap.add_argument("--ckpt-every", type=int, default=100)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
torch.manual_seed(1337)
np.random.seed(1337)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------- data ----------------
train = np.memmap(f"{args.data}/train.bin", dtype=np.uint16, mode="r")
val = np.memmap(f"{args.data}/val.bin", dtype=np.uint16, mode="r")
CTX = 1024


def get_batch(split, bsz):
    src = train if split == "train" else val
    ix = torch.randint(len(src) - CTX - 1, (bsz,))
    x = torch.stack([torch.from_numpy(src[i : i + CTX].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(src[i + 1 : i + 1 + CTX].astype(np.int64)) for i in ix])
    return x.pin_memory().to(dev, non_blocking=True), y.pin_memory().to(dev, non_blocking=True)


# ---------------- model ----------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d, h, ctx):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 4 * d)
        self.proj = nn.Linear(4 * d, d)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, d, h, ctx):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, h, ctx)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = MLP(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, n_layer=12, n_head=12, n_embd=768, ctx=CTX, vocab=50304):
        super().__init__()
        self.ctx = ctx
        self.wte = nn.Embedding(vocab, n_embd)
        self.wpe = nn.Embedding(ctx, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, ctx) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.wte.weight = self.head.weight  # weight tying
        self.apply(self._init)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for b in self.blocks:
            x = b(x)
        x = self.lnf(x)
        if targets is None:
            return self.head(x), None
        logits = self.head(x)
        return None, F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )


model = GPT().to(dev)
nparams = sum(p.numel() for p in model.parameters())
print(f"params: {nparams/1e6:.1f}M", flush=True)

# ---------------- optimizer ----------------
decay, no_decay = [], []
for n, p in model.named_parameters():
    (decay if p.dim() >= 2 else no_decay).append(p)
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr,
    betas=(0.9, 0.95),
    eps=1e-8,
)
scaler = torch.amp.GradScaler("cuda")


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if step >= args.max_steps:
        return args.min_lr
    r = (step - args.warmup) / (args.max_steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * r))


# ---------------- eval ----------------
@torch.no_grad()
def evaluate(iters=64):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch("val", args.micro_bsz)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def sample(prompt="\n", n=200):
    model.eval()
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode_ordinary(prompt)], device=dev)
    for _ in range(n):
        logits, _ = model(idx[:, -CTX :])
        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
    model.train()
    return enc.decode(idx[0].tolist())


# ---------------- train ----------------
history = {"train": [], "val": [], "tok_s": []}
t0 = time.time()
best_val = float("inf")
tokens_per_step = args.micro_bsz * CTX * args.accum

x, y = get_batch("train", args.micro_bsz)
for step in range(args.max_steps):
    if (time.time() - t0) / 60 > args.max_minutes:
        print(f"time budget {args.max_minutes} min reached, stopping", flush=True)
        break
    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr

    model.train()
    ts = time.time()
    for _ in range(args.accum):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _, loss = model(x, y)
        scaler.scale(loss / args.accum).backward()
        x, y = get_batch("train", args.micro_bsz)
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    dt = time.time() - ts

    if step % 10 == 0 or step == args.max_steps - 1:
        tps = tokens_per_step / dt
        history["train"].append((step, loss.item()))
        history["tok_s"].append(tps)
        el = (time.time() - t0) / 60
        print(
            f"step {step:4d}/{args.max_steps} | loss {loss.item():.4f} | "
            f"lr {lr:.2e} | {tps/1e3:.1f}k tok/s | {dt:.1f}s/step | {el:.1f} min",
            flush=True,
        )

    if (step + 1) % args.eval_every == 0 or step == args.max_steps - 1:
        vl = evaluate()
        history["val"].append((step, vl))
        print(f"[eval] step {step}: val loss {vl:.4f}", flush=True)
        s = sample("\nHello")
        print("SAMPLE >>>", s.replace("\n", " ")[:300], flush=True)
        if vl < best_val:
            best_val = vl
            torch.save(
                {"model": model.state_dict(), "step": step, "val": vl, "args": vars(args)},
                f"{args.out}/ckpt_best.pt",
            )

    if (step + 1) % args.ckpt_every == 0:
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
             "history": history, "args": vars(args)},
            f"{args.out}/ckpt_latest.pt",
        )

# ---------------- finish ----------------
torch.save({"model": model.state_dict(), "step": step, "history": history, "args": vars(args)},
           f"{args.out}/ckpt_final.pt")
mean_tps = float(np.mean(history["tok_s"])) if history["tok_s"] else 0
summary = {
    "params_M": round(nparams / 1e6, 2),
    "steps_done": step + 1,
    "tokens_seen": int((step + 1) * tokens_per_step),
    "final_train_loss": history["train"][-1][1],
    "best_val_loss": best_val,
    "mean_tok_s": round(mean_tps),
    "gpu": torch.cuda.get_device_name(0),
    "minutes": round((time.time() - t0) / 60, 1),
}
print("SUMMARY " + json.dumps(summary), flush=True)
with open(f"{args.out}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
if history["train"]:
    s, l = zip(*history["train"])
    ax[0].plot(s, l)
    ax[0].set_title("train loss")
    ax[0].set_xlabel("step")
if history["val"]:
    s, l = zip(*history["val"])
    ax[1].plot(s, l, "o-")
    ax[1].set_title("val loss")
    ax[1].set_xlabel("step")
fig.tight_layout()
fig.savefig(f"{args.out}/loss_curve.png", dpi=120)
print("DONE", flush=True)
