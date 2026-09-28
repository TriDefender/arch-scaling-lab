#!/usr/bin/env python
"""Likelihood-style QA benchmark for ASL checkpoints.

Protocol matches lm-eval-harness: loglikelihood multiple-choice, acc + acc_norm (byte-length).
  hellaswag: acc_norm primary | arc_easy/challenge: acc_norm | winogrande_xl: acc
Usage:
  qa_eval.py --ckpt runs/v2-muon-rope4k/ckpt_final.pt --tag v2 --pos rope --seq-len 4096
  qa_eval.py --ckpt phase0-124m/runs/124m-baseline/ckpt_final.pt --tag v1 --pos wpe --seq-len 1024
Model classes copied verbatim from train_lc.py (state_dict layout identical).
"""
import argparse, json, math, random, re, time
from pathlib import Path

import pyarrow.parquet as pq
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

QA = Path(__file__).resolve().parent / "data/qa"

# ---------------- model (verbatim from train_lc.py) ----------------
import math


def build_rope(head_dim, max_len, base, factor, orig_len):
    inv = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    if factor > 1.0:
        lam = 2 * math.pi / inv
        hi, lo = orig_len / 2.0, max_len / 2.0
        m = 1.0 + (factor - 1.0) * torch.clamp((lam - hi) / (lo - hi), 0.0, 1.0)
        inv = inv / m
    t = torch.arange(max_len, dtype=torch.float32)
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    Tn = x.size(-2)
    c = cos[:Tn].view(1, 1, Tn, -1)
    s = sin[:Tn].view(1, 1, Tn, -1)
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(args.n_embd, 3 * args.n_embd)
        self.proj = nn.Linear(args.n_embd, args.n_embd)
        self.attn_dropout = DROPOUT
        scale = 1.0 / math.sqrt(HEAD)
        if args.pos == "rope" and args.rope_factor > 1.0 and not args.no_yarn_temp:
            scale /= 0.1 * math.log(args.rope_factor) + 1.0
        self.sdpa_scale = scale

    def forward(self, x, cos, sin):
        B, Tn, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, Tn, args.n_head, HEAD).transpose(1, 2)
        k = k.view(B, Tn, args.n_head, HEAD).transpose(1, 2)
        v = v.view(B, Tn, args.n_head, HEAD).transpose(1, 2)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.sdpa_scale)
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
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.blocks = nn.ModuleList([Block() for _ in range(args.n_layer)])
        self.ln_f = nn.LayerNorm(args.n_embd)
        self.lm_head = nn.Linear(args.n_embd, VOCAB, bias=False)

    def forward(self, idx, targets=None):
        B, Tn = idx.shape
        x = self.wte(idx)
        if self.pos_mode == "wpe":
            x = x + self.wpe(torch.arange(Tn, device=idx.device))
        cos = sin = None
        if self.pos_mode == "rope":
            cos, sin = self.rope_cos, self.rope_sin
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        if targets is None:
            return x, None  # hidden states; score_batch projects only needed positions
        logits = self.lm_head(x)
        return logits, None


# ---------------- data loaders (lm-eval protocols) ----------------
def hs_preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def load_hellaswag():
    rows = pq.read_table(QA / "hellaswag_val.parquet").to_pylist()
    docs = []
    for d in rows:
        ctx = hs_preprocess(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize())
        choices = [hs_preprocess(e) for e in d["endings"]]
        docs.append({"ctx": ctx, "choices": choices, "gold": int(d["label"]), "norm": True})
    return "hellaswag", docs


def load_arc(name):
    rows = pq.read_table(QA / f"{name}_val.parquet").to_pylist()
    docs = []
    for d in rows:
        q = "Question: " + d["question"] + "\nAnswer:"
        labels = list(d["choices"]["label"])
        texts = list(d["choices"]["text"])
        gold = labels.index(d["answerKey"])
        docs.append({"ctx": q, "choices": texts, "gold": gold, "norm": True})
    return name, docs


def load_winogrande():
    rows = pq.read_table(QA / "winogrande_xl_val.parquet").to_pylist()
    docs = []
    for d in rows:
        s = d["sentence"]
        i = s.index("_")
        docs.append({
            "ctx": s[:i],
            "choices": [o + s[i + 1:] for o in (d["option1"], d["option2"])],
            "gold": int(d["answer"]) - 1,
            "norm": False,
        })
    return "winogrande", docs


def load_piqa():
    rows = pq.read_table(QA / "piqa_val.parquet").to_pylist()
    docs = []
    for d in rows:
        docs.append({
            "ctx": "Question: " + d["goal"] + "\nAnswer:",
            "choices": [d["sol1"], d["sol2"]],
            "gold": int(d["label"]),
            "norm": True,
        })
    return "piqa", docs


def load_sciq():
    rows = pq.read_table(QA / "sciq_val.parquet").to_pylist()
    docs = []
    for i, d in enumerate(rows):
        opts = [d["correct_answer"], d["distractor1"], d["distractor2"], d["distractor3"]]
        order = list(range(4))
        random.Random(i).shuffle(order)  # deterministic per-doc shuffle
        docs.append({
            "ctx": "Question: " + d["question"] + "\nAnswer:",
            "choices": [opts[j] for j in order],
            "gold": order.index(0),
            "norm": True,
        })
    return "sciq", docs


def load_blimp(per_cfg=500):
    files = sorted(QA.glob("blimp_*.parquet"))
    docs = []
    for f in files:
        rows = pq.read_table(f).to_pylist()[:per_cfg]
        for d in rows:
            docs.append({
                "ctx": "",  # full-sentence acceptability: sum-logprob binary
                "choices": [d["sentence_good"], d["sentence_bad"]],
                "gold": 0,
                "norm": False,
            })
    return "blimp", docs


def load_lambada():
    files = sorted(QA.glob("lambada_*.parquet"))
    rows = pq.read_table(files[0]).to_pylist()
    return "lambada", [{"text": d["text"].strip()} for d in rows if d.get("text", "").strip()]


# ---------------- scoring ----------------
enc = tiktoken.get_encoding("gpt2")


def tok(s):
    return enc.encode(s, allowed_special=set())


@torch.no_grad()
def score_batch(model, seqs, bs=16):
    """seqs: list of (ctx_ids, cont_ids, nbytes). returns [(logprob, ntok, nbytes)]"""
    out = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    for b in range(0, len(order), bs):
        idxs = order[b:b + bs]
        if (b // bs) % 100 == 0:
            print(f"  batch {b//bs}/{(len(order)+bs-1)//bs}", flush=True)
        toks = [seqs[i][0] + seqs[i][1] for i in idxs]
        L = max(len(t) for t in toks)
        ids = torch.zeros(len(idxs), L, dtype=torch.long)
        for r, t in enumerate(toks):
            ids[r, :len(t)] = torch.tensor(t)
        ids = ids.to(dev)
        # gather continuation positions BEFORE lm_head: logits shrink from B*(L-1)*V to P*V
        pos_flat, tgts, spans = [], [], []
        for r, i in enumerate(idxs):
            _, cont, nb = seqs[i]
            n = len(cont)
            end = len(toks[r])
            pos_flat.append(torch.arange(end - n - 1, end - 1) + r * (L - 1))
            tgts.append(ids[r, end - n:end])
            spans.append((i, n, nb))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, _ = model(ids[:, :-1])
            hsel = h.reshape(-1, h.size(-1))[torch.cat(pos_flat)]
            logits = model.lm_head(hsel)
        logprobs = F.log_softmax(logits.float(), -1)
        lp_all = logprobs.gather(1, torch.cat(tgts).to(dev)[:, None]).sum(-1)
        off = 0
        for r, (i, n, nb) in enumerate(spans):
            out[i] = (lp_all[off:off + n].sum().item(), n, nb)
            off += n
    return out


def eval_task(model, docs, maxctx, bs=16):
    # build all (ctx, cont) pairs
    flat = []
    spans = []  # (doc_idx, choice_idx)
    for di, d in enumerate(docs):
        c = tok(d["ctx"])
        if not c:
            c = [50256]  # BOS/EOT anchor for empty-context tasks (blimp)
        if len(c) + 1 > maxctx:
            c = c[-(maxctx - 1):]
        for ci, ch in enumerate(d["choices"]):
            t = tok(" " + ch)
            flat.append((c, t, len(ch.encode("utf-8"))))
            spans.append((di, ci))
    res = score_batch(model, flat, bs=bs)
    per_doc = [[] for _ in docs]
    for (di, ci), r in zip(spans, res):
        per_doc[di].append(r)
    acc = accn = 0
    for d, rs in zip(docs, per_doc):
        lps = [r[0] for r in rs]
        lpn = [r[0] / max(r[2], 1) for r in rs]
        acc += int(max(range(len(lps)), key=lambda i: lps[i]) == d["gold"])
        if d["norm"]:
            accn += int(max(range(len(lpn)), key=lambda i: lpn[i]) == d["gold"])
    n = len(docs)
    accn_val = accn / n if docs[0]["norm"] else None
    return acc / n, accn_val


@torch.no_grad()
def eval_lambada(model, docs, maxctx, bs=32):
    """GPT-2 paper protocol: predict the final word (argmax over vocab).
    Keeps rows whose final word is exactly one BPE token."""
    rows = []
    for d in docs:
        ids = tok(d["text"])
        if len(ids) < 4 or len(ids) > maxctx:
            continue
        last_word = d["text"].split()[-1]
        if enc.decode([ids[-1]]).strip() != last_word.strip():
            continue
        rows.append(ids)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    hit = 0
    for b in range(0, len(order), bs):
        idxs = order[b:b + bs]
        L = max(len(rows[i]) for i in idxs)
        ids = torch.zeros(len(idxs), L, dtype=torch.long)
        for r, i in enumerate(idxs):
            ids[r, :len(rows[i])] = torch.tensor(rows[i])
        ids = ids.to(dev)
        lens = torch.tensor([len(rows[i]) for i in idxs], device=dev)
        pos = lens - 2 + torch.arange(len(idxs), device=dev) * (L - 1)
        tgts = ids[torch.arange(len(idxs), device=dev), lens - 1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, _ = model(ids[:, :-1])
            logits = model.lm_head(h.reshape(-1, h.size(-1))[pos])
        pred = logits.float().argmax(-1)
        hit += int((pred == tgts).sum().item())
    return hit / len(rows), len(rows)


def main():
    global args, HEAD, VOCAB, T, DROPOUT, USE_CKPT, CE_CHUNK, dev
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--pos", choices=["rope", "wpe"], required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--limit", type=int, default=0, help="debug: cap docs per task")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    args = argparse.Namespace(
        pos=a.pos, rope_base=10000.0, rope_factor=1.0, rope_orig_len=1024, no_yarn_temp=False,
        n_layer=12, n_head=12, n_embd=768,
    )
    HEAD = args.n_embd // args.n_head
    VOCAB = 50257
    T = a.seq_len
    DROPOUT = 0.0
    USE_CKPT = False
    CE_CHUNK = 1024
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else {k: v for k, v in ck.items() if k not in ("opt", "scaler", "iter", "best_val", "args")}
    model = GPT().to(dev)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"loaded {a.ckpt} iter={ck.get('iter')} best_val={ck.get('best_val')}", flush=True)

    tasks = [load_hellaswag(), load_arc("arc_easy"), load_arc("arc_challenge"), load_winogrande()]
    bs_map = {"hellaswag": 16, "arc_easy": 16, "arc_challenge": 16, "winogrande": 16}
    optional = [("piqa", load_piqa, 16), ("sciq", load_sciq, 16), ("blimp", load_blimp, 64), ("lambada", load_lambada, 32)]
    for nm, fn, tbs in optional:
        bs_map[nm] = tbs
        try:
            tasks.append(fn())
        except (FileNotFoundError, IndexError) as e:
            print(f"skip {nm}: {e}", flush=True)
            tasks.append((nm, []))
    results = {}
    for name, docs in tasks:
        torch.cuda.empty_cache()  # drop allocator high-water between tasks
        if a.limit:
            docs = docs[: a.limit]
        if not docs:
            results[name] = None
            continue
        t0 = time.time()
        if name == "lambada":
            acc, n_used = eval_lambada(model, docs, a.seq_len)
            accn = None
            n = n_used
        else:
            acc, accn = eval_task(model, docs, a.seq_len, bs=bs_map.get(name, 16))
            n = len(docs)
        dt = time.time() - t0
        entry = {"n": n, "acc": round(acc, 4), "acc_norm": round(accn, 4) if accn is not None else None}
        results[name] = entry
        print(f"{name:16s} n={entry['n']:5d} acc={entry['acc']:.4f} acc_norm={entry['acc_norm']} ({dt:.0f}s)", flush=True)

    outd = Path(__file__).resolve().parent / a.out
    outd.mkdir(exist_ok=True)
    (outd / f"qa_{a.tag}.json").write_text(json.dumps(
        {"tag": a.tag, "ckpt": a.ckpt, "iter": ck.get("iter"), "val_loss": ck.get("best_val"), "results": results}, indent=2))
    print("saved", outd / f"qa_{a.tag}.json")


if __name__ == "__main__":
    main()
