#!/usr/bin/env python
"""Download easy-tier benchmarks (piqa/sciq/lambada/blimp/wikitext103) as parquet.
Sources: HF datasets-server parquet -> HF api/parquet fallback. Validates via pyarrow."""
import json, shutil, urllib.request
from pathlib import Path

import pyarrow.parquet as pq

QA = Path(__file__).resolve().parent / "data/qa"
QA.mkdir(exist_ok=True)
UA = {"User-Agent": "asl-qa-eval/1.0 (local research)"}


def get_json(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))


def dl(url, dest):
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(dest)


def server_urls(ds):
    try:
        d = get_json(f"https://datasets-server.huggingface.co/parquet?dataset={ds}")
        return [(pf["config"], pf["split"], pf["url"]) for pf in d.get("parquet_files", [])]
    except Exception as e:
        print(f"  datasets-server failed: {e}")
        return []


def api_urls(ds, config, split):
    try:
        d = get_json(f"https://huggingface.co/api/datasets/{ds}/parquet")
        node = d.get(config or "default", {})
        return [(config or "default", split, u) for u in node.get(split, [])]
    except Exception as e:
        print(f"  api/parquet failed: {e}")
        return []


def save(us, name_fn, max_mb=100):
    n = 0
    for cfg, sp, u in us:
        dest = QA / name_fn(cfg, sp)
        if dest.exists():
            n += 1
            continue
        try:
            dl(u, dest)
            md = pq.read_metadata(dest)
            if md.num_rows == 0:
                raise ValueError("empty")
            if dest.stat().st_size > max_mb << 20:
                raise ValueError("too big")
            n += 1
            print(f"  ok {dest.name} rows={md.num_rows}")
        except Exception as e:
            print(f"  FAIL {dest.name}: {e}")
            dest.unlink(missing_ok=True)
    return n


def job(title, fetch):
    print(f"== {title}")
    n = fetch()
    print(f"  -> {n} file(s)")
    return n


def main():
    ok = {}

    # --- piqa: single 'default' validation shard ---
    def piqa():
        us = server_urls("ybisk/piqa") or api_urls("ybisk/piqa", "default", "validation")
        us = [x for x in us if x[1] == "validation"]
        return save(us, lambda c, s: "piqa_val.parquet")
    ok["piqa"] = job("piqa", piqa)

    # --- sciq ---
    def sciq():
        us = server_urls("allenai/sciq") or api_urls("allenai/sciq", "default", "validation")
        us = [x for x in us if x[1] == "validation"]
        return save(us, lambda c, s: "sciq_val.parquet")
    ok["sciq"] = job("sciq", sciq)

    # --- lambada_openai: prefer test split ---
    def lambada():
        us = server_urls("lambada_openai") or api_urls("lambada_openai", "default", "test")
        pref = [x for x in us if x[1] == "test"] or [x for x in us if x[1] == "validation"]
        sp = pref[0][1] if pref else "test"
        return save(pref, lambda c, s: f"lambada_{sp}.parquet")
    ok["lambada"] = job("lambada", lambada)

    # --- blimp: all 67 configs, prefer validation split ---
    def blimp():
        us = server_urls("nyu-mll/blimp")
        bycfg = {}
        for c, s, u in us:
            bycfg.setdefault(c, []).append((s, u))
        flat = []
        for c, items in sorted(bycfg.items()):
            pick = [x for x in items if x[0] == "validation"] or [x for x in items if x[0] == "train"] or items[:1]
            flat.append((c, pick[0][0], pick[0][1]))
        return save(flat, lambda c, s: f"blimp_{c}.parquet")
    ok["blimp"] = job("blimp (67 configs)", blimp)

    # --- wikitext-103-raw validation (may be several shards) ---
    def wikitext():
        us = server_urls("Salesforce/wikitext")
        us = [x for x in us if x[0] == "wikitext-103-raw-v1" and x[1] == "validation"]
        if not us:
            us = api_urls("Salesforce/wikitext", "wikitext-103-raw-v1", "validation")
        return save(us, lambda c, s: f"wikitext_val_{s}.parquet")
    ok["wikitext"] = job("wikitext-103 val", wikitext)

    print("\n== manifest")
    for f in sorted(QA.glob("*.parquet")):
        t = pq.read_table(f)
        print(f"{f.name:45s} rows={t.num_rows:6d} cols={t.column_names}")


if __name__ == "__main__":
    main()
