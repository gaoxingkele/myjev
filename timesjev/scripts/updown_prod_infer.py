#!/usr/bin/env python3
"""定版推理入口：加载 prod（或指定 checkpoint）+ 表格双闸门弃权。

若 F:/aicoding/myjev/models/decider-2b-updown-prod 不存在，则读
runs/updown_prod/promotion.json 中的实验权重，并打印 ARCHIVE 提示。
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from timesjev.models import build_model

WORK = ROOT / "runs" / "updown_prod"
PROD = Path(r"F:/aicoding/myjev/models/decider-2b-updown-prod")
SEL_KEY = "updown-prod-mm-L100"


def resolve_ckpt():
    promo = WORK / "promotion.json"
    if PROD.is_dir() and (PROD / "checkpoint").is_dir() and (PROD / "PROMOTION.md").is_file():
        meta = json.loads((PROD / "promotion.json").read_text(encoding="utf-8")) if (PROD / "promotion.json").is_file() else {}
        return str(PROD / "checkpoint"), "production", meta
    if promo.is_file():
        meta = json.loads(promo.read_text(encoding="utf-8"))
        win = meta.get("winner") or {}
        return win.get("checkpoint"), meta.get("decision", "archive"), meta
    raise SystemExit("no prod and no promotion.json — run updown_prod_pipeline.py first")


def main():
    ckpt, status, meta = resolve_ckpt()
    print(json.dumps({"status": status, "checkpoint": ckpt, "meets": (meta.get("winner") or {}).get("meets")}, ensure_ascii=False, indent=2))
    if status != "production" and status != "promote":
        print("NOTE: Decider not promoted; use tabular dual-gate milestone as primary.")
        return
    # smoke-load model
    state = f"selv_noarr:{SEL_KEY}"
    thr = (meta.get("winner") or {}).get("threshold") or 0.65
    model = build_model(
        {
            "backend": "decider_ts",
            "checkpoint": ckpt,
            "state": state,
            "revision": None,
            "temperature": None,
            "use_graphs": False,
        }
    )
    samples = pickle.loads((WORK / "samples_all.pkl").read_bytes())[:2]
    preds = model.predict(samples)
    print("smoke_preds", [{"id": p.id, "answer": p.answer, "probs": p.probs} for p in preds])
    print("dual_gate_threshold", thr)


if __name__ == "__main__":
    main()
