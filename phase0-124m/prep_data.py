"""Phase 0 data prep: FineWeb-Edu sample-10BT -> GPT-2 BPE uint16 bins.

Downloads 2 parquet shards (~1B GPT-2 tokens), tokenizes, writes:
  /content/data/val.bin   (held-out, first 2000 docs of shard 0)
  /content/data/train.bin (the rest)
Run on the Colab VM. Takes ~10-20 min (2 vCPU).
"""
import os
import numpy as np

DATA_DIR = "/content/data"
RAW_DIR = "/content/raw"
os.makedirs(DATA_DIR, exist_ok=True)

SHARDS = [
    "sample/10BT/000_00000.parquet",
    "sample/10BT/001_00000.parquet",
]
VAL_DOCS = 2000

print("downloading shards...", flush=True)
from huggingface_hub import snapshot_download

snapshot_download(
    "HuggingFaceFW/fineweb-edu",
    repo_type="dataset",
    allow_patterns=SHARDS,
    local_dir=RAW_DIR,
    max_workers=4,
)
for s in SHARDS:
    p = os.path.join(RAW_DIR, s)
    print(f"  {s}: {os.path.getsize(p)/1e9:.2f} GB", flush=True)

import tiktoken

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token  # 50256

import pyarrow.parquet as pq
from multiprocessing import Pool


def encode_doc(text: str) -> list:
    return enc.encode_ordinary(text)


def write_docs(docs, mmap, pos, sort_by_len=False):
    """Tokenize docs with a small pool, append to uint16 memmap, return new pos."""
    with Pool(2) as pool:
        for toks in pool.imap(encode_doc, docs, chunksize=256):
            toks = toks + [eot]
            if len(toks) > 0 and pos + len(toks) <= len(mmap):
                mmap[pos : pos + len(toks)] = np.asarray(toks, dtype=np.uint16)
                pos += len(toks)
    return pos


# ---- shard 0: split into val head + train rest ----
p0 = os.path.join(RAW_DIR, SHARDS[0])
t = pq.read_table(p0, columns=["text"])
all_docs = t.column("text").to_pylist()
del t
print(f"shard0: {len(all_docs)} docs", flush=True)

val_docs = all_docs[:VAL_DOCS]
train_docs0 = all_docs[VAL_DOCS:]
del all_docs

# pass 1: count tokens to size the memmaps (tokenize once, cache in tmp files)
import tempfile

cache = tempfile.mkdtemp()

def cache_encode(i_doc):
    i, doc = i_doc
    toks = np.asarray(encode_doc(doc) + [eot], dtype=np.uint16)
    np.save(os.path.join(cache, f"{i}.npy"), toks)
    return len(toks)

def tokenize_to_memmap(docs, out_path, tag):
    sizes = []
    with Pool(2) as pool:
        for n in pool.imap(cache_encode, enumerate(docs), chunksize=128):
            sizes.append(n)
    total = int(sum(sizes))
    print(f"{tag}: {len(docs)} docs, {total/1e6:.1f}M tokens", flush=True)
    m = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total,))
    pos = 0
    for i in range(len(docs)):
        toks = np.load(os.path.join(cache, f"{i}.npy"))
        m[pos : pos + len(toks)] = toks
        pos += len(toks)
        if i % 2000 == 0:
            m.flush()
    m.flush()
    for i in range(len(docs)):
        os.remove(os.path.join(cache, f"{i}.npy"))
    print(f"{tag}: wrote {out_path}", flush=True)
    return total

n_val = tokenize_to_memmap(val_docs, f"{DATA_DIR}/val.bin", "val")
n_train0 = tokenize_to_memmap(train_docs0, f"{DATA_DIR}/train_s0.bin", "train_s0")
del val_docs, train_docs0

# ---- shard 1: straight to train ----
p1 = os.path.join(RAW_DIR, SHARDS[1])
t = pq.read_table(p1, columns=["text"])
docs1 = t.column("text").to_pylist()
del t
n_train1 = tokenize_to_memmap(docs1, f"{DATA_DIR}/train_s1.bin", "train_s1")
del docs1

os.system(f'cat "{DATA_DIR}/train_s0.bin" "{DATA_DIR}/train_s1.bin" > "{DATA_DIR}/train.bin"')
os.remove(f"{DATA_DIR}/train_s0.bin")
os.remove(f"{DATA_DIR}/train_s1.bin")
os.system(f'rm -rf {RAW_DIR} {cache}')

nt = os.path.getsize(f"{DATA_DIR}/train.bin") // 2
nv = os.path.getsize(f"{DATA_DIR}/val.bin") // 2
print(f"DONE train={nt/1e6:.1f}M tokens, val={nv/1e6:.1f}M tokens", flush=True)
