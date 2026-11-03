#!/usr/bin/env python3
"""用固化的 timesjev-onset-q-v1 权重做优质起涨点推理。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from quality_onset_train_pipeline import LEFT, handcrafted, load_bar  # noqa: E402

DEFAULT_BUNDLE = ROOT / "models" / "timesjev-onset-q-v1" / "bundle_full.joblib"
BAR_CACHE = ROOT / "runs" / "quality_onset_ashare" / "bar_cache"


def predict_one(bundle, feats: list[float]) -> dict:
    x = np.asarray(feats, float).reshape(1, -1)
    assert x.shape[1] == bundle["feature_spec"]["dim"], x.shape
    ea, eb, gate, thr = bundle["expert_a"], bundle["expert_b"], bundle["gate"], bundle["threshold"]
    Xa, Xb = x[:, :15], x[:, 15:]
    p_gate = gate.predict_proba(np.hstack([ea.predict_proba(Xa)[:, 1:], eb.predict_proba(Xb)[:, 1:]]))[0]
    p_vote = np.mean([ea.predict_proba(Xa)[0], eb.predict_proba(Xb)[0]], axis=0)
    pred_g, pred_v = int(p_gate.argmax()), int(p_vote.argmax())
    conf_g, conf_v = float(p_gate.max()), float(p_vote.max())
    select = pred_g == pred_v and conf_g >= thr and conf_v >= thr
    return {
        "pred": pred_g,
        "label": "quality_onset" if pred_g == 1 else "not_onset",
        "p_gate": p_gate.tolist(),
        "p_vote": p_vote.tolist(),
        "conf_gate": conf_g,
        "conf_vote": conf_v,
        "threshold": float(thr),
        "select": bool(select),
        "action": "ENTER_WATCH" if (select and pred_g == 1) else "ABSTAIN",
    }


def feats_from_code(code: str, cache: Path) -> tuple[list[float], str]:
    df = load_bar(code.zfill(6), cache)
    if df is None or len(df) < LEFT:
        raise SystemExit(f"no bars for {code} in {cache}")
    closes = df["close"].to_numpy(float)[-LEFT:]
    vols = df["vol"].to_numpy(float)[-LEFT:] if "vol" in df.columns else np.ones(LEFT)
    return handcrafted(closes, vols), str(df["date"].iloc[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--code", type=str, help="6-digit A-share code; uses bar_cache last LEFT bars")
    ap.add_argument("--cache", type=Path, default=BAR_CACHE)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)
    if not args.code:
        print(json.dumps({"version": bundle.get("version"), "threshold": bundle["threshold"], "bundle": str(args.bundle)}, indent=2))
        return
    feats, asof = feats_from_code(args.code, args.cache)
    out = predict_one(bundle, feats)
    out.update({"code": args.code.zfill(6), "asof": asof, "bundle": str(args.bundle)})
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
