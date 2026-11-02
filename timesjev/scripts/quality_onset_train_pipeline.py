#!/usr/bin/env python3
"""优质起涨点训练管线（A股加大训练 → 美股+BTC 同定义融合）。

正样本: Q2 ∪ Q1_20（+50% 不破起点 / 连续上行）
负样本: NEG_break（下采样）
可选 medium: R20（仅 A 股消融）

阶段:
  ashare   - 用已筛 onset_candidates + bar_cache 训 dual-gate
  expand   - 若正样本少，用 R20 作 soft-pos 分层权重消融
  fuse     - 拉取美股/BTC，同 Q 定义筛样后与 A股融合训练
  report   - 汇总成效
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
ASHARE = ROOT / "runs" / "quality_onset_ashare"
OUT = ROOT / "runs" / "quality_onset_train"
LEFT = 60
SEED = 7
TARGET_ACC, TARGET_COV, MIN_N = 0.80, 0.30, 20
NEG_RATIO = 4  # neg:pos


def make_clf():
    return CalibratedClassifierCV(
        HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.07, max_iter=280, l2_regularization=0.25, random_state=SEED
        ),
        method="isotonic",
        cv=3,
    )


def dual_search(y, p_a, p_b, target=0.80, min_n=MIN_N):
    best = None
    for thr in np.linspace(0.50, 0.92, 43):
        pa, pb = p_a.argmax(1), p_b.argmax(1)
        m = (pa == pb) & (p_a.max(1) >= thr) & (p_b.max(1) >= thr)
        n = int(m.sum())
        if n < min_n:
            continue
        acc = float(accuracy_score(y[m], pa[m]))
        if acc >= target:
            cov = float(m.mean())
            if best is None or cov > best["coverage"]:
                best = {"threshold": float(thr), "acc": acc, "n": n, "coverage": cov, "target": target}
    return best


def apply_dual(y, p_a, p_b, thr):
    pa, pb = p_a.argmax(1), p_b.argmax(1)
    m = (pa == pb) & (p_a.max(1) >= thr) & (p_b.max(1) >= thr)
    out = {
        "threshold": float(thr),
        "full_acc": float(accuracy_score(y, pa)),
        "full_auc": float(roc_auc_score(y, p_a[:, 1])) if len(np.unique(y)) > 1 else None,
        "n": int(m.sum()),
        "coverage": float(m.mean()),
        "acc": float(accuracy_score(y[m], pa[m])) if m.any() else None,
    }
    out["meets"] = bool(
        out["n"] >= MIN_N and out["acc"] is not None and out["acc"] >= TARGET_ACC and out["coverage"] >= TARGET_COV
    )
    return out


def _tail(a, n):
    x = np.asarray(a, float)
    return x[-n:] if len(x) >= n else x


def handcrafted(close, vol, high=None, low=None):
    c = np.asarray(close, float)
    v = np.asarray(vol if vol is not None else np.ones_like(c), float)
    feats = []
    for n in (20, 40, 60):
        cc, vv = _tail(c, n), _tail(v, n)
        rets = np.diff(cc) / np.clip(cc[:-1], 1e-9, None)

        def ret(k):
            if len(cc) <= k or cc[-k - 1] == 0:
                return 0.0
            return float(cc[-1] / cc[-k - 1] - 1)

        rng = float(cc.max() - cc.min()) + 1e-9
        feats += [
            ret(1),
            ret(min(5, len(cc) - 1)),
            ret(min(10, len(cc) - 1)),
            float((cc[-1] - cc.min()) / rng),
            float(np.std(rets)) if len(rets) else 0.0,
            float(np.mean(vv[-min(5, len(vv)) :]) / (np.median(vv) + 1e-9)),
            float(cc.max() / cc[0] - 1),
            float(cc.min() / cc[0] - 1),
            float(np.mean(rets[-5:])) if len(rets) >= 5 else 0.0,
            float((cc[-1] / (np.mean(cc) + 1e-9) - 1)),
        ]
    return feats


def load_bar(code: str, cache: Path) -> pd.DataFrame | None:
    files = sorted(cache.glob(f"{code}_*.parquet"))
    if not files:
        return None
    # prefer longest / latest cache file
    best = max(files, key=lambda p: p.stat().st_size)
    try:
        return pd.read_parquet(best)
    except Exception:
        return None


def build_xy_from_onset(onset: pd.DataFrame, bar_cache: Path, mode="hard"):
    """mode=hard: Q pos vs NEG; soft: also include R20 as pos with care via separate tag."""
    pos_m = (onset["Q2"] == 1) | (onset["Q1_20"] == 1)
    neg_m = onset["NEG_break"] == 1
    pos = onset[pos_m].copy()
    neg = onset[neg_m].copy()
    # drop rows that somehow both (should be 0)
    pos = pos[~((pos["NEG_break"] == 1))]
    n_pos = len(pos)
    n_neg_keep = min(len(neg), max(n_pos * NEG_RATIO, n_pos))
    neg = neg.sample(n=n_neg_keep, random_state=SEED) if len(neg) > n_neg_keep else neg
    parts = [pos.assign(y=1, tier="Q"), neg.assign(y=0, tier="NEG")]
    if mode == "soft_r20":
        # R20 但非 Q、非 NEG 作 medium — 训练时仍标 y=1，但报告分层
        mid = onset[(onset["R20"] == 1) & (~pos_m) & (~neg_m)].copy()
        mid = mid.sample(n=min(len(mid), n_pos * 2), random_state=SEED + 1) if len(mid) else mid
        if len(mid):
            parts.append(mid.assign(y=1, tier="R20"))
    data = pd.concat(parts, ignore_index=True)

    X, y, codes, dates, tiers = [], [], [], [], []
    skipped = 0
    for _, r in data.iterrows():
        df = load_bar(str(r["code"]).zfill(6), bar_cache)
        if df is None:
            skipped += 1
            continue
        # locate decision date
        d = str(r["date"])
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
        X.append(handcrafted(closes, vols))
        y.append(int(r["y"]))
        codes.append(str(r["code"]).zfill(6))
        dates.append(d)
        tiers.append(r["tier"])
    return (
        np.asarray(X, float),
        np.asarray(y),
        np.asarray(codes),
        np.asarray(dates),
        np.asarray(tiers),
        {"n_pos_src": int(n_pos), "n_neg_kept": int(n_neg_keep), "skipped": skipped, "n_final": len(y)},
    )


def eval_dual(X, y, codes, tag, seed=SEED, seeds=(7, 11, 17, 23, 29)):
    results = []
    Xa, Xb = X[:, :15], X[:, 15:]
    for s in seeds:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=s)
        trval, te = next(gss1.split(X, y, codes))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=s + 4)
        tri, cai = next(gss2.split(X[trval], y[trval], codes[trval]))
        tr, ca = trval[tri], trval[cai]
        if len(np.unique(y[tr])) < 2 or len(te) < 30:
            continue
        ea, eb, gate = make_clf(), make_clf(), make_clf()
        ea.fit(Xa[tr], y[tr])
        eb.fit(Xb[tr], y[tr])
        gate.fit(np.hstack([ea.predict_proba(Xa[tr])[:, 1:], eb.predict_proba(Xb[tr])[:, 1:]]), y[tr])

        def pack(idx):
            pa = gate.predict_proba(np.hstack([ea.predict_proba(Xa[idx])[:, 1:], eb.predict_proba(Xb[idx])[:, 1:]]))
            pb = np.mean([ea.predict_proba(Xa[idx]), eb.predict_proba(Xb[idx])], axis=0)
            return pa, pb

        hit80 = dual_search(y[ca], *pack(ca), 0.80, MIN_N)
        hit75 = dual_search(y[ca], *pack(ca), 0.75, MIN_N)
        thr = (hit80 or hit75 or {}).get("threshold", 0.60)
        test = apply_dual(y[te], *pack(te), thr)
        results.append(
            {
                "tag": tag,
                "seed": s,
                "n_train": int(len(tr)),
                "n_calib": int(len(ca)),
                "n_test": int(len(te)),
                "pos_rate_test": float(y[te].mean()),
                "calib80": hit80,
                "calib75": hit75,
                "test": test,
                "oracle80": dual_search(y[te], *pack(te), 0.80, max(12, MIN_N // 2)),
            }
        )
    if not results:
        return {"tag": tag, "error": "no valid folds"}
    sels = [r["test"]["acc"] for r in results if r["test"]["acc"] is not None]
    covs = [r["test"]["coverage"] for r in results]
    fulls = [r["test"]["full_acc"] for r in results]
    aucs = [r["test"]["full_auc"] for r in results if r["test"]["full_auc"] is not None]
    return {
        "tag": tag,
        "folds": results,
        "agg": {
            "mean_full": float(np.mean(fulls)),
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "mean_sel": float(np.mean(sels)) if sels else None,
            "mean_cov": float(np.mean(covs)),
            "n_meet_80_30": int(sum(1 for r in results if r["test"]["meets"])),
            "n_folds": len(results),
        },
    }


# ───── US / BTC Q screen (reuse definitions) ─────
def label_q2(entry, highs, lows, closes, horizon=60, gain=0.50):
    n = min(horizon, len(closes))
    if n < 5 or entry <= 0:
        return False
    h = np.asarray(highs[:n], float)
    l = np.asarray(lows[:n], float)
    for t in range(n):
        if l[t] < entry:
            return False
        if h[t] / entry - 1 >= gain:
            return True
    return False


def label_q1(entry, highs, lows, closes, min_days=20, gain=0.50):
    max_look = min(len(closes), max(min_days + 40, 80))
    if max_look < min_days or entry <= 0:
        return False
    c = np.asarray(closes[:max_look], float)
    l = np.asarray(lows[:max_look], float)
    h = np.asarray(highs[:max_look], float)
    for t in range(min_days, max_look + 1):
        if l[:t].min() < entry:
            return False
        if float(h[:t].max() / entry - 1) < gain:
            continue
        seg = c[:t]
        x = np.arange(t)
        if np.std(seg) < 1e-12:
            continue
        corr = float(np.corrcoef(x, seg)[0, 1])
        up_frac = float((np.diff(seg) >= 0).mean()) if t > 1 else 0.0
        if corr >= 0.60 and up_frac >= 0.55:
            return True
    return False


def label_neg(entry, highs, lows, closes, horizon=60):
    n = min(horizon, len(closes))
    if n < 10 or entry <= 0:
        return False
    l = np.asarray(lows[:n], float)
    h = np.asarray(highs[:n], float)
    early = min(20, n)
    return float(l[:early].min() / entry - 1) <= -0.08 and float(h.max() / entry - 1) < 0.15


def scan_bars_df(code, df, asset):
    opens = df["open"].to_numpy(float)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    vols = df["vol"].to_numpy(float) if "vol" in df.columns else np.ones(len(df))
    dates = df["date"].astype(str).tolist()
    rows = []
    last = len(df) - 62
    for i in range(LEFT - 1, max(LEFT, last)):
        entry = float(opens[i + 1])
        fut_h, fut_l, fut_c = highs[i + 1 :], lows[i + 1 :], closes[i + 1 :]
        q2 = label_q2(entry, fut_h, fut_l, fut_c)
        q1 = label_q1(entry, fut_h, fut_l, fut_c, 20)
        neg = label_neg(entry, fut_h, fut_l, fut_c)
        if not (q2 or q1 or neg):
            continue
        sl = slice(i - LEFT + 1, i + 1)
        feats = handcrafted(closes[sl], vols[sl])
        y = 1 if (q2 or q1) else 0
        if q2 or q1:
            tier = "Q"
        else:
            tier = "NEG"
        # skip ambiguous if somehow both
        if (q2 or q1) and neg:
            continue
        rows.append(
            {
                "code": code,
                "date": dates[i],
                "asset": asset,
                "y": y,
                "tier": tier,
                "Q2": int(q2),
                "Q1_20": int(q1),
                "feats": feats,
            }
        )
    return rows


def fetch_us_yahoo(symbol, range_="5y"):
    import json
    import urllib.request

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={range_}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        raw = json.loads(r.read().decode())
    res = raw["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(ts):
        o, h, l, c, v = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]
        if None in (o, h, l, c) or c <= 0:
            continue
        rows.append(
            {
                "date": time.strftime("%Y-%m-%d", time.gmtime(t)),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "vol": float(v or 0),
            }
        )
    return pd.DataFrame(rows).sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_btc():
    import json
    import urllib.request

    limit, end, rows = 1000, None, []
    while len(rows) < 1600:
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000"
        if end is not None:
            url += f"&endTime={end}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=40) as r:
            batch = json.loads(r.read().decode())
        if not batch:
            break
        rows = batch + rows
        end = int(batch[0][0]) - 1
        if len(batch) < limit:
            break
        time.sleep(0.1)
    out, seen = [], set()
    for b in rows:
        t = int(b[0])
        if t in seen:
            continue
        seen.add(t)
        out.append(
            {
                "date": time.strftime("%Y-%m-%d", time.gmtime(t / 1000)),
                "open": float(b[1]),
                "high": float(b[2]),
                "low": float(b[3]),
                "close": float(b[4]),
                "vol": float(b[5]),
            }
        )
    return pd.DataFrame(out).sort_values("date").drop_duplicates("date").reset_index(drop=True)


US_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "NFLX", "AMD",
    "JPM", "XOM", "UNH", "JNJ", "V", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "PEP", "COST", "WMT", "BAC", "CRM", "ADBE", "ORCL", "CSCO", "ACN",
    "MCD", "DIS", "INTC", "QCOM", "TXN", "AMAT", "CAT", "GE", "BA", "LIN",
    "SPGI", "BLK", "GS", "NOW", "ISRG", "BKNG", "TMO", "NEE", "UPS", "RTX",
]


def build_fuse_matrix(ashare_onset, ashare_cache):
    """Return X,y,codes,assets for fused set."""
    # A-share hard set features
    X_a, y_a, codes_a, dates_a, tiers_a, meta_a = build_xy_from_onset(ashare_onset, ashare_cache, "hard")
    rows = []
    for i in range(len(y_a)):
        rows.append(
            {
                "code": codes_a[i],
                "date": dates_a[i],
                "asset": "ashare",
                "y": int(y_a[i]),
                "tier": tiers_a[i],
                "feats": X_a[i].tolist(),
            }
        )

    # US
    us_pos = us_neg = 0
    for sym in US_UNIVERSE:
        print(f"US {sym}", flush=True)
        try:
            df = fetch_us_yahoo(sym)
        except Exception as e:
            print("  fail", e, flush=True)
            continue
        scanned = scan_bars_df(sym, df, "us")
        # balance per symbol: keep all Q, sample NEG
        q = [r for r in scanned if r["y"] == 1]
        n = [r for r in scanned if r["y"] == 0]
        keep_n = n[: max(len(q) * NEG_RATIO, len(q))] if q else n[:20]
        for r in q + keep_n:
            rows.append(r)
            if r["y"] == 1:
                us_pos += 1
            else:
                us_neg += 1
        time.sleep(0.12)

    print("BTC", flush=True)
    btc = fetch_btc()
    scanned = scan_bars_df("BTCUSDT", btc, "btc")
    q = [r for r in scanned if r["y"] == 1]
    n = [r for r in scanned if r["y"] == 0]
    keep_n = n[: max(len(q) * NEG_RATIO, len(q))] if q else n[:30]
    btc_pos = btc_neg = 0
    for r in q + keep_n:
        rows.append(r)
        if r["y"] == 1:
            btc_pos += 1
        else:
            btc_neg += 1

    X = np.asarray([r["feats"] for r in rows], float)
    y = np.asarray([r["y"] for r in rows])
    codes = np.asarray([r["code"] for r in rows])
    assets = np.asarray([r["asset"] for r in rows])
    meta = {
        "ashare": meta_a,
        "us_pos": us_pos,
        "us_neg": us_neg,
        "btc_pos": btc_pos,
        "btc_neg": btc_neg,
        "n_total": int(len(y)),
        "pos_total": int(y.sum()),
    }
    return X, y, codes, assets, meta, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    onset = pd.read_parquet(ASHARE / "onset_candidates.parquet")
    cache = ASHARE / "bar_cache"

    # 1) A-share hard
    print("=== ASHARE hard Q vs NEG ===", flush=True)
    X, y, codes, dates, tiers, meta = build_xy_from_onset(onset, cache, "hard")
    print(meta, "pos", int(y.sum()), "neg", int((1 - y).sum()), flush=True)
    ashare_hard = eval_dual(X, y, codes, "ashare_hard_Q")
    print("ASHARE_HARD", json.dumps(ashare_hard.get("agg"), ensure_ascii=False), flush=True)

    # 2) A-share soft with R20 medium (加大正类训练量)
    print("=== ASHARE soft + R20 ===", flush=True)
    Xs, ys, cs, ds, ts, metas = build_xy_from_onset(onset, cache, "soft_r20")
    print(metas, "pos", int(ys.sum()), flush=True)
    ashare_soft = eval_dual(Xs, ys, cs, "ashare_soft_QR20")
    print("ASHARE_SOFT", json.dumps(ashare_soft.get("agg"), ensure_ascii=False), flush=True)

    # 3) Fuse US+BTC+Ashare
    print("=== FUSE ashare+US+BTC ===", flush=True)
    Xf, yf, cf, af, metaf, rows = build_fuse_matrix(onset, cache)
    print(metaf, flush=True)
    (OUT / "fuse_meta.json").write_text(json.dumps(metaf, ensure_ascii=False, indent=2), encoding="utf-8")
    # save lightweight fuse sample index
    pd.DataFrame(
        [{k: r[k] for k in ("code", "date", "asset", "y", "tier", "Q2", "Q1_20") if k in r} for r in rows]
    ).to_csv(OUT / "fuse_samples.csv", index=False, encoding="utf-8-sig")

    fuse_all = eval_dual(Xf, yf, cf, "fuse_all")
    print("FUSE_ALL", json.dumps(fuse_all.get("agg"), ensure_ascii=False), flush=True)

    # asset-holdout style: train on ashare+us, test groups include btc if present
    fuse_no_btc = None
    m = af != "btc"
    if m.sum() > 200 and (af == "btc").any():
        fuse_no_btc = eval_dual(Xf[m], yf[m], cf[m], "fuse_ashare_us")
        print("FUSE_NO_BTC", json.dumps(fuse_no_btc.get("agg"), ensure_ascii=False), flush=True)

    # BTC-only purged if enough
    btc_only = None
    mb = af == "btc"
    if mb.sum() >= 80 and yf[mb].sum() >= 10:
        # temporal split on BTC
        order = np.argsort([r["date"] for r in rows if r["asset"] == "btc"])
        Xb, yb = Xf[mb][order], yf[mb][order]
        n = len(yb)
        i1, i2 = int(0.6 * n), int(0.75 * n)
        tr, ca, te = np.arange(0, i1), np.arange(i1, i2), np.arange(i2, n)
        if len(np.unique(yb[tr])) > 1 and len(te) >= 15:
            Xa, Xbb = Xb[:, :15], Xb[:, 15:]
            ea, eb, gate = make_clf(), make_clf(), make_clf()
            ea.fit(Xa[tr], yb[tr])
            eb.fit(Xbb[tr], yb[tr])
            gate.fit(np.hstack([ea.predict_proba(Xa[tr])[:, 1:], eb.predict_proba(Xbb[tr])[:, 1:]]), yb[tr])

            def pack(idx):
                pa = gate.predict_proba(np.hstack([ea.predict_proba(Xa[idx])[:, 1:], eb.predict_proba(Xbb[idx])[:, 1:]]))
                pb = np.mean([ea.predict_proba(Xa[idx]), eb.predict_proba(Xbb[idx])], axis=0)
                return pa, pb

            hit = dual_search(yb[ca], *pack(ca), 0.75, max(8, MIN_N // 2))
            thr = (hit or {}).get("threshold", 0.6)
            btc_only = {"tag": "btc_purged", "calib": hit, "test": apply_dual(yb[te], *pack(te), thr), "n": int(n)}
            print("BTC_ONLY", json.dumps(btc_only, ensure_ascii=False), flush=True)

    summary = {
        "objective": "quality onset Q1/Q2 training: ashare scale-up then US+BTC fuse",
        "bar": "selective>=80% @ coverage>=30%",
        "ashare_hard": ashare_hard,
        "ashare_soft_r20": ashare_soft,
        "fuse_all": fuse_all,
        "fuse_ashare_us": fuse_no_btc,
        "btc_purged": btc_only,
        "sample_meta": {"ashare_hard": meta, "ashare_soft": metas, "fuse": metaf},
    }
    (OUT / "train_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(block):
        if not block or "agg" not in block:
            return str(block)
        a = block["agg"]
        return f"full={a['mean_full']:.3f} auc={a.get('mean_auc')} sel={a.get('mean_sel')} cov={a['mean_cov']:.3f} meet={a['n_meet_80_30']}/{a['n_folds']}"

    md = [
        "# 优质起涨点训练成效报告",
        "",
        "## 设定",
        "- 正样本: Q2 ∪ Q1_20（+50% 且不破起点 / 连续上行）",
        "- 负样本: NEG_break（4× 下采样）",
        "- soft 消融: 加入 R20 作 medium 正样本加大训练量",
        "- 模型: 双专家 HGB + dual-gate（与既有里程碑同构）",
        "",
        "## 结果（5 seed 平均）",
        "",
        f"- **A股 hard Q**: {fmt(ashare_hard)}",
        f"- **A股 soft+R20**: {fmt(ashare_soft)}",
        f"- **融合 A+美股+BTC**: {fmt(fuse_all)}",
    ]
    if fuse_no_btc:
        md.append(f"- **融合 A+美股**: {fmt(fuse_no_btc)}")
    if btc_only:
        t = btc_only["test"]
        md.append(
            f"- **BTC purged**: full={t['full_acc']:.3f} sel={t['acc']} cov={t['coverage']:.3f} meets={t['meets']}"
        )
    md += [
        "",
        "## 样本规模",
        f"- A股 hard: {meta}",
        f"- A股 soft: {metas}",
        f"- Fuse: {metaf}",
        "",
        "## 结论",
    ]
    ah = ashare_hard.get("agg", {})
    fu = fuse_all.get("agg", {})
    best = max(
        [
            ("ashare_hard", ah.get("mean_sel") or 0, ah.get("mean_cov") or 0, ah.get("n_meet_80_30") or 0),
            ("fuse_all", fu.get("mean_sel") or 0, fu.get("mean_cov") or 0, fu.get("n_meet_80_30") or 0),
        ],
        key=lambda x: (x[3], x[1], x[2]),
    )
    md.append(
        f"- 当前最优配置倾向 **{best[0]}**（meet折数优先）；生产条仍以 selective≥80%@≥30% 为准。"
    )
    md.append("- Q 标签与 R20/S20 无硬冲突（见 quality_onset_ashare/REPORT.md）；训练以 Q 为主、R20 扩量。")
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("SUMMARY written", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
