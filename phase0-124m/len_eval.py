#!/usr/bin/env python
"""PPL suite: (a) length-stratified PPL on the shared val.bin stream,
(b) WikiText-103 (raw) zero-shot transfer PPL.
The length sweep is the direct probe for position-encoding ablations:
wpe models are structurally capped at their trained length; RoPE models are not.
Usage:
  len_eval.py --ckpt phase0-124m/runs/124m-baseline/ckpt_final.pt --tag v1 --pos wpe --seq-len 1024
  len_eval.py --ckpt runs/v2-muon-rope4k/ckpt_final.pt --tag v2 --pos rope --seq-len 4096
"""
import argparse, json, math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
enc = tiktoken.get_encoding("gpt2")

# model classes verbatim from qa_eval.py (same state_dict layout).
# GPT resolves n_embd/pos/... from the qa_eval MODULE globals, so config is injected there.
import qa_eval  # noqa: E402


@torch.no_grad()
def chunk_nll(model, ids_t):
    """mean-NLL over one [T] token tensor; full-vocab logits computed in 512-pos slices."""
    x = ids_t[None].to(DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, _ = model(x[:, :-1])
        tgt = x[0, 1:]
        tot = 0.0
        for s in range(0, h.size(1), 512):
            logits = model.lm_head(h[0, s:s + 512])
            tot += F.cross_entropy(logits.float(), tgt[s:s + 512], reduction="sum").item()
    return tot, x.size(1) - 1


@torch.no_grad()
def strat_ppl(model, val_np, T, n=48):
    """PPL over n strided, non-overlapping T-token chunks spread across the val stream."""
    stride = (len(val_np) - T) // n
    tot_nll = tot_tok = 0
    for i in range(n):
        s = i * stride
        nll, tok = chunk_nll(model, torch.from_numpy(val_np[s:s + T].astype(np.int64)))
        tot_nll += nll
        tot_tok += tok
    return math.exp(tot_nll / tot_tok), tot_tok


def load_wikitext_ids():
    lines = []
    for f in sorted((ROOT / "data/qa").glob("wikitext_val_*.parquet")):
        lines += [r["text"] for r in pq.read_table(f).to_pylist()]
    text = "\n".join(l for l in lines if l.strip())
    return np.array(enc.encode(text, allowed_special=set()), dtype=np.uint16)


def main():
    global args, HEAD, VOCAB, T, DROPOUT, USE_CKPT, CE_CHUNK, DEV, dev
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--pos", choices=["rope", "wpe"], required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--attn-arch", choices=["mha", "gqa", "mla"], default="mha")
    p.add_argument("--n-kv-heads", type=int, default=0)
    p.add_argument("--mla-latent", type=int, default=256)
    p.add_argument("--mla-rope-dim", type=int, default=32)
    a = p.parse_args()

    args = argparse.Namespace(
        pos=a.pos, rope_base=10000.0, rope_factor=1.0, rope_orig_len=1024, no_yarn_temp=False,
        n_layer=12, n_head=12, n_embd=768,
        attn_arch=a.attn_arch, n_kv_heads=a.n_kv_heads, mla_latent=a.mla_latent, mla_rope_dim=a.mla_rope_dim,
    )
    qa_eval.args = args
    qa_eval.HEAD = 768 // 12  # head_dim = n_embd // n_head; n_head=12 was the bug (crashed attention view)
    qa_eval.VOCAB, qa_eval.T = 50257, a.seq_len
    qa_eval.DROPOUT, qa_eval.USE_CKPT, qa_eval.CE_CHUNK = 0.0, False, 1024
    HEAD, VOCAB, T = 12, 50257, a.seq_len
    DROPOUT, USE_CKPT, CE_CHUNK = 0.0, False, 1024
    DEV = dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else {k: v for k, v in ck.items() if k not in ("opt", "scaler", "iter", "best_val", "args")}
    model = qa_eval.GPT().to(dev)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"loaded {a.ckpt} iter={ck.get('iter')}", flush=True)

    out = {"tag": a.tag, "ckpt": a.ckpt, "seq_len": a.seq_len, "len_strat": {}, "wikitext103_ppl": {}}

    val_np = np.memmap(ROOT / "data/val.bin", dtype=np.uint16, mode="r")
    print(f"val.bin: {len(val_np):,} tokens", flush=True)
    for bt in (256, 512, 1024, 2048, 4096):
        if bt > a.seq_len:
            out["len_strat"][str(bt)] = None
            continue
        ppl, ntok = strat_ppl(model, val_np, bt)
        out["len_strat"][str(bt)] = round(ppl, 3)
        print(f"  len={bt:5d} ppl={ppl:.3f} ({ntok} tok)", flush=True)

    wt = load_wikitext_ids()
    print(f"wikitext-103 val: {len(wt):,} tokens", flush=True)
    for cl in (1024, 4096):
        if cl > a.seq_len:
            out["wikitext103_ppl"][str(cl)] = None
            continue
        ppl, ntok = strat_ppl(model, wt, cl, n=min(64, len(wt) // cl))
        out["wikitext103_ppl"][str(cl)] = round(ppl, 3)
        print(f"  wikitext@{cl} ppl={ppl:.3f} ({ntok} tok)", flush=True)

    outd = ROOT / a.out
    outd.mkdir(exist_ok=True)
    (outd / f"ppl_{a.tag}.json").write_text(json.dumps(out, indent=2))
    print("saved", outd / f"ppl_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
