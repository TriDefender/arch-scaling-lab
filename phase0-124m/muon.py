"""Muon optimizer (Keller Jordan 2024) + quintic Newton-Schulz orthogonalization.

Hybrid convention (Moonlight, arXiv 2502.16982): Muon on 2D hidden matrices
(attention/MLP weights), AdamW on embeddings, lm_head and all 1D params.
~2x compute efficiency vs AdamW reported at compute-optimal scale; benefit at
124M / 300M-token budget is an open question -> run as its own ablation cell.

Refs: https://github.com/KellerJordan/Muon , modded-nanogpt speedrun.
"""
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate orthogonalization (UV^T) of G via quintic Newton-Schulz in bf16.

    Coefficients tuned to maximize shrinkage slope at zero; 5 steps suffice.
    """
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                     ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "mom_buf" not in st:
                    st["mom_buf"] = torch.zeros_like(g)
                buf = st["mom_buf"]
                buf.lerp_(g, 1 - mom)
                upd = g.lerp(buf, mom) if group["nesterov"] else buf
                u = zeropower_via_newtonschulz5(upd, steps=group["ns_steps"])
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(u.to(p.dtype), alpha=-lr * scale)
        return loss
