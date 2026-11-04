#!/usr/bin/env python3
"""timesjev-onset-q-v1.1：去重 + 难负样本 + 更严阈值 + 截面相对特征，并落盘权重。

相对 v1:
  1) 正样本事件去重：同股一次起涨只留首触发日（间隔 ≥20 交易日）
  2) 难负样本：失败突破 / 假起涨 / 中等涨后回撤；保留部分 easy NEG
  3) 校准集搜 thr：在 ≥80% 下最大化覆盖（不再默认 0.5）
  4) 同日截面分位：ret5/ret20/vol/位置 等相对全池排名
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from quality_onset_train_pipeline import (  # noqa: E402
    ASHARE,
    LEFT,
    SEED,
    TARGET_ACC,
    TARGET_COV,
    MIN_N,
    _tail,
    dual_search,
    apply_dual,
    handcrafted,
    load_bar,
)

VERSION = "timesjev-onset-q-v1.1"
OUT = ROOT / "runs" / "quality_onset_v11"
MODEL_DIRS = [
    ROOT / "models" / VERSION,
    Path(r"F:/aicoding/myjev/models") / VERSION,
]
DEDUP_GAP = 20
HARD_NEG_RATIO = 3
EASY_NEG_RATIO = 1
SEED_FOLDS = (7, 11, 17, 23, 29)


def make_clf(seed=SEED):
    return CalibratedClassifierCV(
        HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.07, max_iter=300, l2_regularization=0.25, random_state=seed
        ),
        method="isotonic",
        cv=3,
    )


def dedup_positives(pos: pd.DataFrame, gap: int = DEDUP_GAP) -> pd.DataFrame:
    """同股正样本按日期排序，间隔 <gap 的后续触发丢弃。"""
    keep = []
    for code, g in pos.sort_values(["code", "date"]).groupby("code"):
        last_i = -10**9
        # use positional index in this group's date order
        dates = g["date"].tolist()
        # map date order via integer rank in this stock's timeline later; use row order as proxy
        # Better: convert to ordinal and require calendar-ish gap via list position in full bar — use date diff days
        last_ord = None
        for _, r in g.iterrows():
            ord_ = pd.Timestamp(r["date"]).toordinal()
            if last_ord is None or (ord_ - last_ord) >= gap:
                keep.append(r)
                last_ord = ord_
    return pd.DataFrame(keep).reset_index(drop=True)


def path_stats_left(df: pd.DataFrame, i: int) -> dict | None:
    if i < LEFT - 1:
        return None
    c = df["close"].to_numpy(float)[i - LEFT + 1 : i + 1]
    v = df["vol"].to_numpy(float)[i - LEFT + 1 : i + 1] if "vol" in df.columns else np.ones(LEFT)
    if len(c) < LEFT:
        return None
    rets = np.diff(c) / np.clip(c[:-1], 1e-9, None)
    return {
        "ret1": float(c[-1] / c[-2] - 1),
        "ret5": float(c[-1] / c[-6] - 1) if len(c) > 5 else 0.0,
        "ret20": float(c[-1] / c[-21] - 1) if len(c) > 20 else 0.0,
        "vol20": float(np.std(rets[-20:])) if len(rets) >= 20 else float(np.std(rets) or 0),
        "vol_ratio": float(np.mean(v[-5:]) / (np.median(v) + 1e-9)),
        "pos_range": float((c[-1] - c.min()) / (c.max() - c.min() + 1e-9)),
    }


def build_cross_section_maps(bar_cache: Path, dates_needed: set[str]) -> dict[str, dict[str, dict]]:
    """date -> code -> path_stats；仅对 dates_needed 建索引。"""
    by_date: dict[str, dict[str, dict]] = defaultdict(dict)
    files = list(bar_cache.glob("*.parquet"))
    print(f"building CS maps for {len(dates_needed)} dates over {len(files)} bars...", flush=True)
    for fi, fp in enumerate(files):
        code = fp.name.split("_")[0]
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        date_to_i = {str(d): i for i, d in enumerate(df["date"].astype(str).tolist())}
        for d in dates_needed:
            i = date_to_i.get(d)
            if i is None:
                continue
            st = path_stats_left(df, i)
            if st:
                by_date[d][code] = st
        if (fi + 1) % 50 == 0:
            print(f"  bars {fi+1}/{len(files)}", flush=True)
    return by_date


def cs_ranks(stats_by_code: dict[str, dict], code: str) -> list[float]:
    if code not in stats_by_code or len(stats_by_code) < 5:
        return [0.5] * 12
    keys = ["ret1", "ret5", "ret20", "vol20", "vol_ratio", "pos_range"]
    mats = {k: np.asarray([s[k] for s in stats_by_code.values()], float) for k in keys}
    codes = list(stats_by_code.keys())
    i = codes.index(code)
    out = []
    for k in keys:
        x = mats[k]
        out.append(float((x < x[i]).mean() + 0.5 * (x == x[i]).mean()))
    for k in keys:
        med = float(np.median(mats[k]))
        out.append(float(mats[k][i] - med))
    return out


def is_hard_negative(row: pd.Series) -> bool:
    """假起涨/中等上涨未达 Q：clear 或 R20 但非 Q，或非 NEG 的中性失败。"""
    q = int(row.get("Q2", 0) or 0) or int(row.get("Q1_20", 0) or 0)
    if q:
        return False
    if int(row.get("NEG_break", 0) or 0):
        return False
    # clear_UP or R20 without becoming Q → 看起来像机会但不是顶级起涨
    if int(row.get("clear_UP", 0) or 0) or int(row.get("R20", 0) or 0):
        return True
    return False


def assemble_dataset(onset: pd.DataFrame, bar_cache: Path):
    pos_raw = onset[(onset["Q2"] == 1) | (onset["Q1_20"] == 1)].copy()
    pos = dedup_positives(pos_raw, DEDUP_GAP)
    easy = onset[onset["NEG_break"] == 1].copy()
    hard = onset[onset.apply(is_hard_negative, axis=1)].copy()

    n_pos = len(pos)
    n_hard = min(len(hard), n_pos * HARD_NEG_RATIO)
    n_easy = min(len(easy), n_pos * EASY_NEG_RATIO)
    hard_s = hard.sample(n=n_hard, random_state=SEED) if len(hard) > n_hard else hard
    easy_s = easy.sample(n=n_easy, random_state=SEED + 1) if len(easy) > n_easy else easy

    pos = pos.assign(y=1, tier="Q")
    hard_s = hard_s.assign(y=0, tier="HARD_NEG")
    easy_s = easy_s.assign(y=0, tier="EASY_NEG")
    data = pd.concat([pos, hard_s, easy_s], ignore_index=True)

    dates_needed = set(data["date"].astype(str).tolist())
    cs_map = build_cross_section_maps(bar_cache, dates_needed)

    X, y, codes, dates, tiers = [], [], [], [], []
    skipped = 0
    for _, r in data.iterrows():
        code = str(r["code"]).zfill(6)
        d = str(r["date"])
        df = load_bar(code, bar_cache)
        if df is None:
            skipped += 1
            continue
        idx = df.index[df["date"].astype(str) == d]
        if len(idx) == 0:
            skipped += 1
            continue
        i = int(idx[0])
        if i < LEFT - 1:
            skipped += 1
            continue
        sl = slice(i - LEFT + 1, i + 1)
        closes = df["close"].to_numpy(float)[sl]
        vols = df["vol"].to_numpy(float)[sl] if "vol" in df.columns else np.ones(LEFT)
        if len(closes) < LEFT:
            skipped += 1
            continue
        feats = handcrafted(closes, vols)
        feats += cs_ranks(cs_map.get(d, {}), code)
        X.append(feats)
        y.append(int(r["y"]))
        codes.append(code)
        dates.append(d)
        tiers.append(r["tier"])

    meta = {
        "n_pos_raw": int(len(pos_raw)),
        "n_pos_dedup": int(n_pos),
        "n_hard_neg": int(len(hard_s)),
        "n_easy_neg": int(len(easy_s)),
        "skipped": skipped,
        "n_final": len(y),
        "feat_dim": int(len(X[0])) if X else 0,
    }
    return np.asarray(X, float), np.asarray(y), np.asarray(codes), np.asarray(dates), np.asarray(tiers), meta


def pack_proba(ea, eb, gate, Xa, Xb):
    pa = gate.predict_proba(np.hstack([ea.predict_proba(Xa)[:, 1:], eb.predict_proba(Xb)[:, 1:]]))
    pb = np.mean([ea.predict_proba(Xa), eb.predict_proba(Xb)], axis=0)
    return pa, pb


def fit_bundle(Xa, Xb, y, seed=SEED):
    ea, eb, gate = make_clf(seed), make_clf(seed + 1), make_clf(seed + 2)
    ea.fit(Xa, y)
    eb.fit(Xb, y)
    gate.fit(np.hstack([ea.predict_proba(Xa)[:, 1:], eb.predict_proba(Xb)[:, 1:]]), y)
    return ea, eb, gate


def eval_multi(X, y, codes):
    # split feats: price 30 + cs 12 = 42; expert A price[:15], B price[15:]+cs
    dim = X.shape[1]
    split_a = 15
    results = []
    for s in SEED_FOLDS:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=s)
        trval, te = next(gss1.split(X, y, codes))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=s + 4)
        tri, cai = next(gss2.split(X[trval], y[trval], codes[trval]))
        tr, ca = trval[tri], trval[cai]
        Xa, Xb = X[:, :split_a], X[:, split_a:]
        ea, eb, gate = fit_bundle(Xa[tr], Xb[tr], y[tr], seed=s)
        hit80 = dual_search(y[ca], *pack_proba(ea, eb, gate, Xa[ca], Xb[ca]), 0.80, MIN_N)
        hit75 = dual_search(y[ca], *pack_proba(ea, eb, gate, Xa[ca], Xb[ca]), 0.75, MIN_N)
        thr = (hit80 or hit75 or {"threshold": 0.60})["threshold"]
        test = apply_dual(y[te], *pack_proba(ea, eb, gate, Xa[te], Xb[te]), thr)
        results.append({"seed": s, "thr": thr, "calib80": hit80, "calib75": hit75, "test": test})
    sels = [r["test"]["acc"] for r in results if r["test"]["acc"] is not None]
    covs = [r["test"]["coverage"] for r in results]
    fulls = [r["test"]["full_acc"] for r in results]
    aucs = [r["test"]["full_auc"] for r in results if r["test"]["full_auc"] is not None]
    thrs = [r["thr"] for r in results]
    return {
        "folds": results,
        "agg": {
            "mean_full": float(np.mean(fulls)),
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "mean_sel": float(np.mean(sels)) if sels else None,
            "mean_cov": float(np.mean(covs)),
            "mean_thr": float(np.mean(thrs)),
            "n_meet": int(sum(1 for r in results if r["test"]["meets"])),
            "n_folds": len(results),
            "feat_dim": dim,
        },
    }


def export_weights(X, y, codes, meta, eval_report):
    Xa, Xb = X[:, :15], X[:, 15:]
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    trval, te = next(gss1.split(X, y, codes))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 4)
    tri, cai = next(gss2.split(X[trval], y[trval], codes[trval]))
    tr, ca = trval[tri], trval[cai]

    ea0, eb0, gate0 = fit_bundle(Xa[tr], Xb[tr], y[tr])
    hit80 = dual_search(y[ca], *pack_proba(ea0, eb0, gate0, Xa[ca], Xb[ca]), 0.80, MIN_N)
    hit75 = dual_search(y[ca], *pack_proba(ea0, eb0, gate0, Xa[ca], Xb[ca]), 0.75, MIN_N)
    # 更严：优先 80；若 thr 仍很低，抬到至少 0.55 并记录
    thr = (hit80 or hit75 or {"threshold": 0.60})["threshold"]
    thr = float(max(thr, 0.55)) if hit80 or hit75 else 0.60
    holdout = apply_dual(y[te], *pack_proba(ea0, eb0, gate0, Xa[te], Xb[te]), thr)
    # also report natural thr without floor
    thr_nat = (hit80 or hit75 or {"threshold": 0.60})["threshold"]
    holdout_nat = apply_dual(y[te], *pack_proba(ea0, eb0, gate0, Xa[te], Xb[te]), thr_nat)

    ea_full, eb_full, gate_full = fit_bundle(Xa, Xb, y)
    feature_spec = {
        "left_bars": LEFT,
        "price_dim": 30,
        "cs_dim": 12,
        "dim": int(X.shape[1]),
        "expert_a": "price[0:15]",
        "expert_b": "price[15:30]+cs[12]",
        "dedup_gap_days": DEDUP_GAP,
        "neg_mix": {"hard_ratio": HARD_NEG_RATIO, "easy_ratio": EASY_NEG_RATIO},
    }
    manifest = {
        "version": VERSION,
        "parent": "timesjev-onset-q-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "changes": [
            "positive event dedup gap>=20d",
            "hard negatives (clear/R20 but not Q) + easy NEG",
            "cross-section ranks/excess",
            "threshold floor >=0.55 when calibrated",
        ],
        "sample_meta": meta,
        "multi_seed_eval": eval_report["agg"],
        "threshold": thr,
        "threshold_natural": float(thr_nat),
        "calib_dual80": hit80,
        "calib_dual75": hit75,
        "holdout_test": holdout,
        "holdout_natural_thr": holdout_nat,
        "meets_prod_bar": bool(holdout.get("meets")),
        "feature_spec": feature_spec,
        "recommend_bundle": "bundle_full.joblib",
    }
    bundle = {
        "version": VERSION,
        "expert_a": ea_full,
        "expert_b": eb_full,
        "gate": gate_full,
        "threshold": thr,
        "threshold_natural": float(thr_nat),
        "feature_spec": feature_spec,
        "holdout_test": holdout,
        "fit_on": "all_after_lock_thr",
    }
    readme = f"""# {VERSION}

相对 v1：正样本去重、难负样本、截面特征、更严阈值。

## Holdout（thr={thr}）
- selective={holdout.get('acc')} coverage={holdout.get('coverage')} meets={holdout.get('meets')}
- natural thr={thr_nat}: sel={holdout_nat.get('acc')} cov={holdout_nat.get('coverage')}

## Multi-seed
{json.dumps(eval_report['agg'], ensure_ascii=False, indent=2)}

推理: `python scripts/quality_onset_infer.py --bundle models/{VERSION}/bundle_full.joblib --code 000001`
（v1.1 特征维={feature_spec['dim']}，infer 脚本需用 v11 特征构建；见 quality_onset_live_week.py）
"""
    for d in MODEL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, d / "bundle_full.joblib", compress=3)
        joblib.dump(bundle, d / "bundle.joblib", compress=3)
        (d / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "README.md").write_text(readme, encoding="utf-8")
        print("wrote", d)
    return manifest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    onset = pd.read_parquet(ASHARE / "onset_candidates.parquet")
    cache = ASHARE / "bar_cache"
    print("assembling v1.1 dataset...", flush=True)
    X, y, codes, dates, tiers, meta = assemble_dataset(onset, cache)
    print("meta", meta, "pos", int(y.sum()), "neg", int((1 - y).sum()), flush=True)
    (OUT / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "xy_codes.npz", X=X, y=y, codes=codes, dates=dates.astype("U"), tiers=tiers.astype("U"))

    print("multi-seed eval...", flush=True)
    report = eval_multi(X, y, codes)
    print("AGG", json.dumps(report["agg"], ensure_ascii=False), flush=True)
    (OUT / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("export weights...", flush=True)
    manifest = export_weights(X, y, codes, meta, report)
    (OUT / "REPORT.md").write_text(
        "\n".join(
            [
                f"# {VERSION}",
                "",
                f"- dataset: {meta}",
                f"- multi-seed: {json.dumps(report['agg'], ensure_ascii=False)}",
                f"- holdout: {json.dumps(manifest['holdout_test'], ensure_ascii=False)}",
                f"- thr: {manifest['threshold']} (natural {manifest['threshold_natural']})",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("done", OUT)


if __name__ == "__main__":
    main()
