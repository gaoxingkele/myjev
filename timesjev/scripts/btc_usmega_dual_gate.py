#!/usr/bin/env python3
"""BTC + 美股科技巨头 dual-gate 对照（不含山寨币）。

池子: BTCUSDT + AAPL/MSFT/NVDA/GOOGL/AMZN/META/TSLA/AVGO/NFLX/AMD
方法: 与 A 股 tabular 里程碑同构（清晰涨跌标签 / 双专家+gate / 双闸门）
切分: GroupShuffle by symbol；另报 BTC-only purged、美股-only
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[1] / "runs" / "btc_usmega_dual_gate"
LEFT, HORIZON, EMBARGO = 100, 20, 20
TARGET_ACC, TARGET_COV, MIN_N = 0.80, 0.30, 20
SEED = 7

# 统一用偏股票口径的清晰涨跌（BTC 波动大，仍会贡献足够事件）
UP_GAIN, UP_CLOSE, UP_DD = 0.08, 0.04, -0.08
DOWN_GAIN_MAX, DOWN_CLOSE, DOWN_DD = 0.03, -0.04, -0.08

US_MEGA = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "NFLX", "AMD"]


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read()


def fetch_binance_daily(symbol="BTCUSDT", n_bars=1600):
    limit, end, rows = 1000, None, []
    while len(rows) < n_bars:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
        if end is not None:
            url += f"&endTime={end}"
        batch = json.loads(_get(url).decode())
        if not batch:
            break
        rows = batch + rows
        end = int(batch[0][0]) - 1
        if len(batch) < limit:
            break
        time.sleep(0.12)
    seen, uniq = set(), []
    for b in rows:
        t = int(b[0])
        if t in seen:
            continue
        seen.add(t)
        uniq.append(
            {
                "date": time.strftime("%Y-%m-%d", time.gmtime(t / 1000)),
                "open": float(b[1]),
                "high": float(b[2]),
                "low": float(b[3]),
                "close": float(b[4]),
                "vol": float(b[5]),
            }
        )
    uniq.sort(key=lambda x: x["date"])
    return uniq[-n_bars:]


def fetch_yahoo_daily(symbol: str, range_="5y"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={range_}"
    raw = json.loads(_get(url).decode())
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
                "vol": float(v or 0.0),
            }
        )
    rows.sort(key=lambda x: x["date"])
    return rows


def rsi(closes, period=14):
    c = np.asarray(closes, float)
    d = np.diff(c)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    out = np.full(len(c), 50.0)
    if len(c) <= period:
        return out.tolist()
    au, ad = up[:period].mean(), dn[:period].mean()
    out[period] = 100.0 if ad == 0 else 100 - 100 / (1 + au / (ad + 1e-12))
    for i in range(period, len(d)):
        au = (au * (period - 1) + up[i]) / period
        ad = (ad * (period - 1) + dn[i]) / period
        out[i + 1] = 100.0 if ad == 0 else 100 - 100 / (1 + au / (ad + 1e-12))
    return out.tolist()


def ema(x, n):
    a = 2 / (n + 1)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]
    return y


def macd_hist(closes):
    c = np.asarray(closes, float)
    m = ema(c, 12) - ema(c, 26)
    return (m - ema(m, 9)).tolist()


def ma(closes, n):
    c = np.asarray(closes, float)
    out = np.full(len(c), np.nan)
    for i in range(n - 1, len(c)):
        out[i] = c[i - n + 1 : i + 1].mean()
    first = np.nanmean(out)
    return np.where(np.isnan(out), first, out).tolist()


def label_clear(entry, highs, lows, closes):
    h = np.asarray(highs[:HORIZON], float)
    l = np.asarray(lows[:HORIZON], float)
    c = np.asarray(closes[:HORIZON], float)
    if len(c) < HORIZON:
        return None
    mg = float(h.max() / entry - 1)
    md = float(l.min() / entry - 1)
    rc = float(c[-1] / entry - 1)
    if mg >= UP_GAIN and rc >= UP_CLOSE and md >= UP_DD:
        return 1
    if mg <= DOWN_GAIN_MAX and (rc <= DOWN_CLOSE or md <= DOWN_DD):
        return 0
    return None


def _tail(a, n):
    x = np.asarray(a, float)
    return x[-n:] if len(x) >= n else x


def handcrafted(close, vol, macd_h, rsi_v, ma5, ma20):
    feats = []
    for n in (20, 50, 100):
        c, v, m, r = _tail(close, n), _tail(vol, n), _tail(macd_h, n), _tail(rsi_v, n)
        a5, a20 = _tail(ma5, n), _tail(ma20, n)

        def ret(k):
            if len(c) <= k or c[-k - 1] == 0:
                return 0.0
            return float(c[-1] / c[-k - 1] - 1)

        rng = float(c.max() - c.min()) + 1e-9
        rets = np.diff(c) / np.clip(c[:-1], 1e-9, None)
        feats += [
            ret(1),
            ret(min(5, len(c) - 1)),
            float((c[-1] - c.min()) / rng),
            float(np.std(rets)) if len(rets) else 0.0,
            float(np.mean(v[-min(5, len(v)) :]) / (np.median(v) + 1e-9)),
            float(a5[-1] / (a20[-1] + 1e-9) - 1) if len(a5) and len(a20) else 0.0,
            float(r[-1]) / 100.0,
            float(m[-1]),
            float(c.max() / c[0] - 1),
            float(c.min() / c[0] - 1),
        ]
    return feats


def bars_to_events(code: str, bars: list[dict]):
    if len(bars) < LEFT + HORIZON + 50:
        return []
    closes = [b["close"] for b in bars]
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    vols = [b["vol"] for b in bars]
    rsi_v, macd_h = rsi(closes), macd_hist(closes)
    ma5, ma20 = ma(closes, 5), ma(closes, 20)
    out = []
    for i in range(LEFT - 1, len(bars) - HORIZON - 1):
        entry = bars[i + 1]["open"]
        lab = label_clear(entry, highs[i + 1 :], lows[i + 1 :], closes[i + 1 :])
        if lab is None:
            continue
        sl = slice(i - LEFT + 1, i + 1)
        out.append(
            {
                "code": code,
                "date": bars[i]["date"],
                "label": lab,
                "feats": handcrafted(closes[sl], vols[sl], macd_h[sl], rsi_v[sl], ma5[sl], ma20[sl]),
                "asset": "btc" if code == "BTCUSDT" else "us",
            }
        )
    return out


def make_clf(kind="logreg"):
    if kind == "hgb":
        return CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.07, max_iter=220, l2_regularization=0.2, random_state=7
            ),
            method="isotonic",
            cv=3,
        )
    return CalibratedClassifierCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.7)),
        method="sigmoid",
        cv=3,
    )


def dual_search(y, p_a, p_b, target=TARGET_ACC, min_n=MIN_N):
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


def eval_group(events, tag, kind="hgb", seed=SEED):
    X = np.asarray([e["feats"] for e in events], float)
    y = np.asarray([e["label"] for e in events])
    codes = np.asarray([e["code"] for e in events])
    Xa, Xb = X[:, :15], X[:, 15:]
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    trval, te = next(gss1.split(X, y, codes))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 4)
    tri, cai = next(gss2.split(X[trval], y[trval], codes[trval]))
    tr, ca = trval[tri], trval[cai]

    ea, eb = make_clf(kind), make_clf(kind)
    ea.fit(Xa[tr], y[tr])
    eb.fit(Xb[tr], y[tr])
    gate = make_clf(kind)
    gate.fit(np.hstack([ea.predict_proba(Xa[tr])[:, 1:], eb.predict_proba(Xb[tr])[:, 1:]]), y[tr])

    def pack(idx):
        pa = gate.predict_proba(np.hstack([ea.predict_proba(Xa[idx])[:, 1:], eb.predict_proba(Xb[idx])[:, 1:]]))
        pb = np.mean([ea.predict_proba(Xa[idx]), eb.predict_proba(Xb[idx])], axis=0)
        return pa, pb

    hit80 = dual_search(y[ca], *pack(ca), 0.80, MIN_N)
    hit75 = dual_search(y[ca], *pack(ca), 0.75, MIN_N)
    thr = (hit80 or hit75 or {}).get("threshold", 0.60)
    p_te_a, p_te_b = pack(te)
    test = apply_dual(y[te], p_te_a, p_te_b, thr)

    # BTC slice within test if present
    te_codes = codes[te]
    btc_m = te_codes == "BTCUSDT"
    btc_slice = None
    if btc_m.any() and btc_m.sum() >= 15:
        pa, pb = p_te_a[btc_m], p_te_b[btc_m]
        pred = pa.argmax(1)
        agree = (pred == pb.argmax(1)) & (pa.max(1) >= thr) & (pb.max(1) >= thr)
        yy = y[te][btc_m]
        btc_slice = {
            "n": int(btc_m.sum()),
            "full_acc": float(accuracy_score(yy, pred)),
            "sel_n": int(agree.sum()),
            "sel_cov": float(agree.mean()),
            "sel_acc": float(accuracy_score(yy[agree], pred[agree])) if agree.any() else None,
        }

    return {
        "tag": tag,
        "clf": kind,
        "n": int(len(y)),
        "up": int(y.sum()),
        "down": int((1 - y).sum()),
        "n_symbols": int(len(set(codes.tolist()))),
        "train_n": int(len(tr)),
        "calib_n": int(len(ca)),
        "test_n": int(len(te)),
        "test_symbols": sorted(set(codes[te].tolist())),
        "calib_dual80": hit80,
        "calib_dual75": hit75,
        "test_dual": test,
        "test_oracle_dual80": dual_search(y[te], p_te_a, p_te_b, 0.80, max(12, MIN_N // 2)),
        "btc_in_test": btc_slice,
    }


def eval_btc_purged(events, kind="hgb"):
    btc = sorted([e for e in events if e["code"] == "BTCUSDT"], key=lambda e: e["date"])
    if len(btc) < 200:
        return {"error": "too few BTC", "n": len(btc)}
    X = np.asarray([e["feats"] for e in btc], float)
    y = np.asarray([e["label"] for e in btc])
    dates = np.asarray([e["date"] for e in btc])
    Xa, Xb = X[:, :15], X[:, 15:]
    n = len(y)
    i1, i2 = int(0.60 * n), int(0.75 * n)
    tr = np.arange(0, max(0, i1 - EMBARGO))
    ca = np.arange(i1, max(i1, i2 - EMBARGO))
    te = np.arange(i2, n)
    ea, eb = make_clf(kind), make_clf(kind)
    ea.fit(Xa[tr], y[tr])
    eb.fit(Xb[tr], y[tr])
    gate = make_clf(kind)
    gate.fit(np.hstack([ea.predict_proba(Xa[tr])[:, 1:], eb.predict_proba(Xb[tr])[:, 1:]]), y[tr])

    def pack(idx):
        pa = gate.predict_proba(np.hstack([ea.predict_proba(Xa[idx])[:, 1:], eb.predict_proba(Xb[idx])[:, 1:]]))
        pb = np.mean([ea.predict_proba(Xa[idx]), eb.predict_proba(Xb[idx])], axis=0)
        return pa, pb

    hit80 = dual_search(y[ca], *pack(ca), 0.80, MIN_N)
    hit75 = dual_search(y[ca], *pack(ca), 0.75, MIN_N)
    thr = (hit80 or hit75 or {}).get("threshold", 0.60)
    return {
        "tag": "BTC_purged",
        "clf": kind,
        "n": n,
        "date_test": f"{dates[te[0]]}->{dates[te[-1]]}",
        "calib_dual80": hit80,
        "calib_dual75": hit75,
        "test_dual": apply_dual(y[te], *pack(te), thr),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    events = []

    print("fetch BTCUSDT...", flush=True)
    btc_bars = fetch_binance_daily("BTCUSDT", 1600)
    ev = bars_to_events("BTCUSDT", btc_bars)
    print(f"  BTC events={len(ev)} bars={len(btc_bars)}", flush=True)
    events.extend(ev)

    for sym in US_MEGA:
        print(f"fetch {sym}...", flush=True)
        try:
            bars = fetch_yahoo_daily(sym)
            ev = bars_to_events(sym, bars)
            print(f"  {sym} events={len(ev)} bars={len(bars)}", flush=True)
            events.extend(ev)
        except Exception as e:
            print(f"  FAIL {sym}: {e}", flush=True)
        time.sleep(0.25)

    meta = {
        "n_events": len(events),
        "by_symbol": {s: sum(1 for e in events if e["code"] == s) for s in sorted(set(e["code"] for e in events))},
        "labels": {"UP": [UP_GAIN, UP_CLOSE, UP_DD], "DOWN": [DOWN_GAIN_MAX, DOWN_CLOSE, DOWN_DD]},
        "universe": ["BTCUSDT"] + US_MEGA,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("total", len(events), meta["by_symbol"], flush=True)

    us_only = [e for e in events if e["asset"] == "us"]
    pool = [e for e in events]

    results = {
        "us_only_hgb": eval_group(us_only, "us_mega_only", "hgb"),
        "us_only_logreg": eval_group(us_only, "us_mega_only", "logreg"),
        "btc_us_hgb": eval_group(pool, "btc_plus_us_mega", "hgb"),
        "btc_us_logreg": eval_group(pool, "btc_plus_us_mega", "logreg"),
        "btc_purged_hgb": eval_btc_purged(events, "hgb"),
    }
    for k, v in results.items():
        print(k, json.dumps(v, ensure_ascii=False), flush=True)

    def ok(r):
        return bool(r.get("test_dual", {}).get("meets"))

    summary = {
        "question": "BTC + US mega-tech (no alts) dual-gate vs A-share milestone",
        "bar": "selective >=80% @ coverage >=30%",
        "meta": meta,
        "results": results,
        "verdict": {
            "us_only_hgb": "PASS" if ok(results["us_only_hgb"]) else "FAIL",
            "btc_us_hgb": "PASS" if ok(results["btc_us_hgb"]) else "FAIL",
            "btc_purged_hgb": "PASS" if ok(results["btc_purged_hgb"]) else "FAIL",
        },
    }
    (OUT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary["verdict"], ensure_ascii=False, indent=2))
    print("saved", OUT / "report.json")


if __name__ == "__main__":
    main()
