"""Phase 0 data prep (streaming, 12GB-RAM safe): FineWeb-Edu -> GPT-2 BPE uint16 bins.

v2: never loads a whole shard into RAM; iterates parquet in 5k-doc batches.
Outputs /content/data/{train.bin,val.bin}. Reuses already-downloaded shards.
"""
import os
import numpy as np

DATA_DIR = "/content/data"
RAW_DIR = "/content/raw"
os.makedirs(DATA_DIR, exist_ok=True)

SHARDS = ["sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet"]
VAL_DOCS = 2000
TRAIN_CAP = 700_000_000  # oversized mmap; truncated to real size at the end

import pyarrow.parquet as pq

if not all(os.path.exists(os.path.join(RAW_DIR, s)) for s in SHARDS):
    print("downloading shards...", flush=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        "HuggingFaceFW/fineweb-edu",
        repo_type="dataset",
        allow_patterns=SHARDS,
        local_dir=RAW_DIR,
        max_workers=4,
    )

import tiktoken
from multiprocessing import Pool

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


def encode_doc(text):
    return enc.encode_ordinary(text)


def iter_docs(path, batch=5000):
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=batch, columns=["text"]):
        yield from rb.column("text").to_pylist()


def stream_encode(path, mmap_path, val_out=None):
    m = np.memmap(mmap_path, dtype=np.uint16, mode="w+", shape=(TRAIN_CAP,))
    vm = [] if val_out else None
    pos = ndocs = 0
    pool = Pool(2)

    def absorb(tok_lists):
        nonlocal pos, ndocs
        for toks in tok_lists:
            t = np.asarray(toks + [EOT], dtype=np.uint16)
            if vm is not None and ndocs < VAL_DOCS:
                vm.append(t)
            else:
                m[pos : pos + len(t)] = t
                pos += len(t)
            ndocs += 1

    buf = []
    for doc in iter_docs(path):
        buf.append(doc)
        if len(buf) >= 2000:
            absorb(pool.map(encode_doc, buf))
            buf.clear()
            print(f"  {ndocs} docs | {pos/1e6:.0f}M train tokens", flush=True)
    absorb(pool.map(encode_doc, buf))
    pool.close()

    if vm:
        v = np.concatenate(vm)
        v.tofile(val_out)
        print(f"val: {len(v)/1e6:.1f}M tokens -> {val_out}", flush=True)
    del m
    os.truncate(mmap_path, pos * 2)
    print(f"{os.path.basename(mmap_path)}: {pos/1e6:.1f}M tokens from {ndocs} docs", flush=True)
    return pos


print("tokenizing shard 0 (val head + train part)...", flush=True)
stream_encode(os.path.join(RAW_DIR, SHARDS[0]), f"{DATA_DIR}/train0.bin", val_out=f"{DATA_DIR}/val.bin")
print("tokenizing shard 1...", flush=True)
stream_encode(os.path.join(RAW_DIR, SHARDS[1]), f"{DATA_DIR}/train1.bin")

with open(f"{DATA_DIR}/train.bin", "wb") as out:
    for p in [f"{DATA_DIR}/train0.bin", f"{DATA_DIR}/train1.bin"]:
        with open(p, "rb") as f:
            while True:
                chunk = f.read(1 << 24)
                if not chunk:
                    break
                out.write(chunk)
        os.remove(p)

import shutil

shutil.rmtree(RAW_DIR, ignore_errors=True)
nt = os.path.getsize(f"{DATA_DIR}/train.bin") // 2
nv = os.path.getsize(f"{DATA_DIR}/val.bin") // 2
print(f"DONE train={nt/1e6:.1f}M tokens, val={nv/1e6:.1f}M tokens", flush=True)
