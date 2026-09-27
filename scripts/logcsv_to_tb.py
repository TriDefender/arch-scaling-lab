"""Convert a train.py/train_lc.py log.csv into TensorBoard event files.

Usage: python logcsv_to_tb.py <log.csv> <out_event_dir>
Re-runnable: snapshots current rows; re-run after training ends to append finals.
"""
import csv
import sys

from torch.utils.tensorboard import SummaryWriter

src, dst = sys.argv[1], sys.argv[2]
w = SummaryWriter(log_dir=dst)
n = 0
with open(src) as f:
    for r in csv.DictReader(f):
        it = int(r["iter"])
        for col, tag in (("loss", "train/loss"), ("val_loss", "val/loss"),
                         ("tok_s", "perf/tok_s"), ("vram_gb", "perf/vram_gb")):
            v = r.get(col)
            if v:
                w.add_scalar(tag, float(v), it)
                n += 1
w.close()
print(f"logcsv_to_tb: {n} scalars -> {dst}")
