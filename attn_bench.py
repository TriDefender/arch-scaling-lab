#!/usr/bin/env python
"""Attention backend bake-off @ 4K RoPE — GPT-2 124M shape (d=768, H=12, hd=64).

Backends: auto (SDPA default pick), eager (manual, fp32 softmax), SDPA flash /
cudnn / mem_efficient / math, FlexAttention (compiled, block-mask causal).

For each: fwd+bwd step time, peak VRAM, numerics vs fp32 eager reference
(output max-abs-err + dQ max-abs-err). Mirrors train_lc.py rope (fp32 math,
half-split) and scale=1/sqrt(hd). B=2 is the 4K training operating point
(micro-bs 2 x grad-accum 4 = 32768 global batch).
"""
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

p = argparse.ArgumentParser()
p.add_argument("--L", type=int, default=4096)
p.add_argument("--B", type=int, default=2)
p.add_argument("--H", type=int, default=12)
p.add_argument("--hd", type=int, default=64)
p.add_argument("--iters", type=int, default=10)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--sweep-bs", type=str, default="",
               help="comma list of batch sizes to sweep with the fastest backend")
p.add_argument("--out", type=str, default="phase0-124m/results/attn_bench.json")
a = p.parse_args()

DEV = "cuda"
DTYPE = torch.bfloat16


def rope_cache(L, hd, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=DEV).float() / hd))
    t = torch.arange(L, device=DEV).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):  # mirrors train_lc.py: fp32 math, half split
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).to(x.dtype)


def eager_attn(q, k, v):
    L = q.size(-2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
    mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), 1)
    att = att.masked_fill(mask, float("-inf"))
    att = F.softmax(att.float(), dim=-1).to(q.dtype)
    return att @ v


def build_backends(L):
    bk = {"auto": lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
          "eager": eager_attn}
    for name, b in [("sdpa-flash", SDPBackend.FLASH_ATTENTION),
                    ("sdpa-cudnn", SDPBackend.CUDNN_ATTENTION),
                    ("sdpa-mem_eff", SDPBackend.EFFICIENT_ATTENTION),
                    ("sdpa-math", SDPBackend.MATH)]:
        def fn(q, k, v, _b=b):
            with sdpa_kernel(_b):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        bk[name] = fn
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        def mask_mod(b, h, qi, ki):
            return qi >= ki

        bm = create_block_mask(mask_mod, None, None, L, L, device=DEV)
        flexc = torch.compile(flex_attention)
        bk["flex"] = lambda q, k, v: flexc(q, k, v, block_mask=bm)
    except Exception as e:
        print(f"[flex unavailable] {type(e).__name__}: {e}")
    return bk


def bench(fn, B, iters=None, warmup=None):
    iters = iters or a.iters
    warmup = warmup or a.warmup
    q = torch.randn(B, a.H, a.L, a.hd, device=DEV, dtype=DTYPE, requires_grad=True)
    k = torch.randn(B, a.H, a.L, a.hd, device=DEV, dtype=DTYPE, requires_grad=True)
    v = torch.randn(B, a.H, a.L, a.hd, device=DEV, dtype=DTYPE, requires_grad=True)
    cos, sin = rope_cache(a.L, a.hd)

    def step():
        qq, kk = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = fn(qq, kk, v)
        out.sum().backward()
        q.grad = k.grad = v.grad = None

    try:
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) / iters * 1e3
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        return dt_ms, peak, None
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, "OOM"
    except Exception as e:
        torch.cuda.empty_cache()
        return None, None, f"{type(e).__name__}: {e}"
    finally:
        del q, k, v
        torch.cuda.empty_cache()


def numerics(fn):
    torch.manual_seed(0)
    base = torch.randn(3, a.B, a.H, a.L, a.hd, device=DEV, dtype=DTYPE)
    cos, sin = rope_cache(a.L, a.hd)
    q, k, v = [t.clone().requires_grad_(True) for t in base]
    out = fn(apply_rope(q, cos, sin), apply_rope(k, cos, sin), v)
    out.sum().backward()
    dq_b = q.grad.detach().float()
    q32, k32, v32 = [t.detach().float().requires_grad_(True) for t in base]
    ref = eager_attn(apply_rope(q32, cos.float(), sin.float()),
                     apply_rope(k32, cos.float(), sin.float()), v32)
    ref.sum().backward()
    out_err = (out.detach().float() - ref.detach()).abs().max().item()
    dq_err = (dq_b - q32.grad.detach()).abs().max().item()
    return out_err, dq_err


def main():
    results = {"L": a.L, "B": a.B, "H": a.H, "hd": a.hd, "rows": [], "sweep": []}
    bk = build_backends(a.L)

    print(f"\n=== attention bake-off  L={a.L} B={a.B} H={a.H} hd={a.hd} "
          f"({DTYPE}, fwd+bwd, {a.iters} iters) ===")
    print(f"{'backend':<12} {'step ms':>9} {'tok/s':>10} {'peak MiB':>9} "
          f"{'out err':>9} {'dQ err':>9}  note")
    for name, fn in bk.items():
        out_err = dq_err = float("nan")
        try:
            out_err, dq_err = numerics(fn)
        except Exception as e:
            print(f"[numerics {name} failed] {type(e).__name__}: {e}")
        dt, peak, err = bench(fn, a.B)
        if err:
            print(f"{name:<12} {'-':>9} {'-':>10} {'-':>9} {out_err:>9.5f} {dq_err:>9.5f}  {err}")
            results["rows"].append({"name": name, "err": err})
            continue
        tps = a.B * a.L / (dt / 1e3)
        print(f"{name:<12} {dt:>9.2f} {tps:>10,.0f} {peak:>9.0f} "
              f"{out_err:>9.5f} {dq_err:>9.5f}")
        results["rows"].append({"name": name, "step_ms": dt, "tok_s": tps,
                                "peak_mib": peak, "out_err": out_err, "dq_err": dq_err})

    ok = [r for r in results["rows"] if r.get("step_ms")]
    if a.sweep_bs and ok:
        fastest = min(ok, key=lambda r: r["step_ms"])["name"]
        print(f"\n=== micro-bs sweep with fastest backend: {fastest} @ L={a.L} ===")
        for B in [int(x) for x in a.sweep_bs.split(",")]:
            dt, peak, err = bench(bk[fastest], B)
            if err:
                print(f"B={B}: {err}")
                results["sweep"].append({"B": B, "err": err})
                continue
            tps = B * a.L / (dt / 1e3)
            print(f"B={B}: {dt:8.2f} ms/step  {tps:>10,.0f} tok/s  peak {peak:6.0f} MiB")
            results["sweep"].append({"B": B, "step_ms": dt, "tok_s": tps, "peak_mib": peak})

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
