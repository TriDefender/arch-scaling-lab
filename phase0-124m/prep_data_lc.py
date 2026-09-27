"""Verify long-context readiness of the token stream (no repack needed).

train.bin / val.bin are flat uint16 token streams with documents joined by EOT.
Any seq_len is a pure view parameter via random crops (train_lc.get_batch), so
4k/16k/32k training reads the SAME files -- this script validates integrity and
reports window statistics, then writes data_manifest.json for reproducibility.

Usage: python prep_data_lc.py [--data-dir data] [--seq-lens 1024,4096,32768]
"""
import argparse
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=None)
p.add_argument("--seq-lens", default="1024,4096,32768")
args = p.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = args.data_dir or os.path.join(SCRIPT_DIR, "data")
EOT = 50256

manifest = {"data_dir": os.path.abspath(DATA), "files": {}}
ok = True
for name in ("train.bin", "val.bin"):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print(f"MISSING {path}")
        ok = False
        continue
    toks = np.memmap(path, dtype=np.uint16, mode="r")
    n = toks.size
    # integrity: max token id must be < VOCAB; sample EOT density on 10M-token probe
    probe = toks[: min(n, 10_000_000)]
    mx = int(probe.max())
    eot_frac = float((probe == EOT).mean())
    n_docs_probe = int((probe == EOT).sum())
    est_docs = int(n * eot_frac)
    entry = {
        "tokens": int(n),
        "bytes": int(os.path.getsize(path)),
        "max_token_id": mx,
        "eot_fraction_probe": round(eot_frac, 5),
        "est_documents": est_docs,
        "avg_doc_len_tokens": int(1 / eot_frac) if eot_frac > 0 else None,
    }
    if mx >= 50257:
        print(f"FAIL {name}: max token id {mx} >= 50257")
        ok = False
    # per-seq-len window stats
    entry["windows"] = {}
    for L in (int(x) for x in args.seq_lens.split(",")):
        full, rem = divmod(n, L)
        entry["windows"][L] = {
            "non_overlapping": full,
            "eval_windows_256k": max(1, 262144 // L),
            "epoch_tokens_frac": round(262144 / n, 6),
        }
    manifest["files"][name] = entry
    print(
        f"{name}: {n/1e6:.1f}M tokens | max_id={mx} | eot={eot_frac*100:.2f}% "
        f"~{est_docs/1e3:.0f}k docs (avg ~{entry['avg_doc_len_tokens']} tok) | "
        + " ".join(f"{L}:{v['non_overlapping']}x{L}" for L, v in entry["windows"].items())
    )

out = os.path.join(DATA, "data_manifest.json")
with open(out, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"manifest -> {out}")
print("OK" if ok else "PROBLEMS FOUND")
