"""CPU smoke test for muon.py: tiny 2-block transformer on a deterministic cyclic task.
Pass criteria: loss drops from ~ln(V) to <1.0 in 80 iters, no NaN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from muon import Muon

torch.manual_seed(0)
D, V, T, B, H = 64, 64, 128, 8, 4


class Blk(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D)
        self.aproj = nn.Linear(D, D)
        self.fc = nn.Linear(D, 4 * D)
        self.mproj = nn.Linear(4 * D, D)
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)

    def forward(self, x):
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        q = q.view(B, T, H, D // H).transpose(1, 2)
        k = k.view(B, T, H, D // H).transpose(1, 2)
        v = v.view(B, T, H, D // H).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, D)
        x = x + self.aproj(y)
        x = x + self.mproj(F.gelu(self.fc(self.ln2(x))))
        return x


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(V, D)
        self.wpe = nn.Embedding(T, D)
        self.blocks = nn.ModuleList([Blk() for _ in range(2)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1)))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


model = Tiny()
hidden2d, rest = [], []
for n, p in model.named_parameters():
    if p.ndim == 2 and not any(s in n for s in ("wte", "wpe", "head")):
        hidden2d.append(p)
    else:
        rest.append(p)
print(f"muon params: {len(hidden2d)}, adamw params: {len(rest)}")
muon = Muon(hidden2d, lr=0.02, weight_decay=0.0)
adamw = torch.optim.AdamW(rest, lr=1e-3)

# deterministic cyclic data: next token = current + 1 mod V (fully predictable)
idx = (torch.arange(T) % V).unsqueeze(0).repeat(B, 1)
y = torch.roll(idx, shifts=-1, dims=1)

first = None
for it in range(81):
    logits = model(idx)
    loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    muon.step()
    adamw.step()
    muon.zero_grad(set_to_none=True)
    adamw.zero_grad(set_to_none=True)
    if it in (0, 40, 80):
        print(f"it {it}: loss {loss.item():.4f}", flush=True)
    assert not torch.isnan(loss), "NaN loss"
    if it == 0:
        first = loss.item()

final = loss.item()
assert final < 1.0, f"FAIL: loss did not converge ({first:.3f} -> {final:.3f})"
print(f"PASS: {first:.3f} -> {final:.3f}")
