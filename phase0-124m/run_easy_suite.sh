#!/bin/bash
cd "$(dirname "$0")"
P=.venv/bin/python
echo "=== len v2 ==="; $P -u len_eval.py --ckpt ../runs/v2-muon-rope4k/ckpt_final.pt --tag v2 --pos rope --seq-len 4096 || echo "LEN_V2_FAIL"
echo "=== len v1 ==="; $P -u len_eval.py --ckpt runs/124m-baseline/ckpt_final.pt --tag v1 --pos wpe --seq-len 1024 || echo "LEN_V1_FAIL"
echo "=== qa v2 ===";  $P -u qa_eval.py  --ckpt ../runs/v2-muon-rope4k/ckpt_final.pt --tag v2 --pos rope --seq-len 4096 || echo "QA_V2_FAIL"
echo "=== qa v1 ===";  $P -u qa_eval.py  --ckpt runs/124m-baseline/ckpt_final.pt --tag v1 --pos wpe --seq-len 1024 || echo "QA_V1_FAIL"
echo ALL_DONE
