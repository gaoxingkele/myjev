#!/usr/bin/env python3
"""加密货币 dual-gate 对照实验（方法同 A 股 tabular 里程碑）。

两档:
  A) 单标的 BTCUSDT — purged walk-forward（压力测试）
  B) 多币种池 — GroupShuffle by symbol（与 A 股多股票设定对等）

数据: 币安公开日K（免 Key）
标签: 清晰涨/跌（剔除 chop），门槛按币圈波动标定
模型: 双专家 + meta gate + 双闸门弃权
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[1] / "runs" / "btc_dual_gate_smoke"
LEFT = 100
HORIZON = 20
EMBARGO = 20
TARGET_ACC, TARGET_COV, MIN_N = 0.80, 0.30, 20
SEED = 7
# BTC/山寨日波动高于 A 股
UP_GAIN, UP_CLOSE, UP_DD = 0.12, 0.06, -0.10
DOWN_GAIN_MAX, DOWN_CLOSE, DOWN_DD = 0.04, -0.06, -0.12
# 流动性较好的 USDT 永续对应现货对（日K）
UNIVERSE = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "UNIUSDT",
    "APTUSDT",
]


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode())


def fetch_binance_daily(symbol="BTCUSDT", n_bars=1600):
    limit = 1000
    end = None
    rows = []
    while len(rows) < n_bars:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
        if end is not None:
            url += f"&endTime={end}"
        try:
            batch = _get(url)
        except Exception as e:
            print(f"  skip {symbol}: {e}", flush=True)
            return []
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
                "ts": t,
                "date": time.strftime("%Y-%m-%d", time.gmtime(t / 1000)),
                "open": float(b[1]),
                "high": float(b[2]),
                "low": float(b[3]),
                "close": float(b[4]),
                "vol": float(b[5]),
            }
        )
    uniq.sort(key=lambda x: x["ts"])
    return uniq[-n_bars:]


def rsi(closes, period=14):
    c = np.asarray(closes, float)
    d = np.diff(c)
    up = np.clip(d, 0, None)
    dn = np.clip(-d, 0, None)
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


def macd_hist(closes, fast=12, slow=26, signal=9):
    c = np.asarray(closes, float)

    def ema(x, n):
        a = 2 / (n + 1)
        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = a * x[i] + (1 - a) * y[i - 1]
        return y

    macd = ema(c, fast) - ema(c, slow)
    return (macd - ema(macd, signal)).tolist()


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
        c = _tail(close, n)
        v = _tail(vol, n)
        m = _tail(macd_h, n)
        r = _tail(rsi_v, n)
        a5 = _tail(ma5, n)
        a20 = _tail(ma20, n)

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


def make_clf():
    return CalibratedClassifierCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.7)),
        method="sigmoid",
        cv=3,
    )


def dual_search(y, p_a, p_b, target=TARGET_ACC, min_n=MIN_N):
    best = None
    for thr in np.linspace(0.50, 0.90, 41):
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


def build_events(symbols, n_bars=1600):
    events = []
    for sym in symbols:
        print(f"fetch {sym}...", flush=True)
        bars = fetch_binance_daily(sym, n_bars=n_bars)
        if len(bars) < LEFT + HORIZON + 50:
            print(f"  too short: {len(bars)}", flush=True)
            continue
        closes = [b["close"] for b in bars]
        highs = [b["high"] for b in bars]
        lows = [b["low"] for b in bars]
        vols = [b["vol"] for b in bars]
        rsi_v = rsi(closes)
        macd_h = macd_hist(closes)
        ma5 = ma(closes, 5)
        ma20 = ma(closes, 20)
        n = 0
        for i in range(LEFT - 1, len(bars) - HORIZON - 1):
            entry = bars[i + 1]["open"]
            lab = label_clear(entry, highs[i + 1 :], lows[i + 1 :], closes[i + 1 :])
            if lab is None:
                continue
            sl = slice(i - LEFT + 1, i + 1)
            feats = handcrafted(closes[sl], vols[sl], macd_h[sl], rsi_v[sl], ma5[sl], ma20[sl])
            events.append(
                {
                    "code": sym,
                    "date": bars[i]["date"],
                    "label": lab,
                    "feats": feats,
                }
            )
            n += 1
        print(f"  clear events={n}", flush=True)
        time.sleep(0.2)
    return events


def eval_group_shuffle(events, tag):
    """与 A 股 prod splits 同构：按 code GroupShuffle。"""
    X = np.asarray([e["feats"] for e in events], float)
    y = np.asarray([e["label"] for e in events])
    codes = np.asarray([e["code"] for e in events])
    Xa, Xb = X[:, :15], X[:, 15:]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    trval, te = next(gss1.split(X, y, codes))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 4)
    tr_i, ca_i = next(gss2.split(X[trval], y[trval], codes[trval]))
    tr, ca = trval[tr_i], trval[ca_i]

    ea, eb = make_clf(), make_clf()
    ea.fit(Xa[tr], y[tr])
    eb.fit(Xb[tr], y[tr])
    gate = make_clf()
    gate.fit(np.hstack([ea.predict_proba(Xa[tr])[:, 1:], eb.predict_proba(Xb[tr])[:, 1:]]), y[tr])

    def pack(idx):
        pa = gate.predict_proba(np.hstack([ea.predict_proba(Xa[idx])[:, 1:], eb.predict_proba(Xb[idx])[:, 1:]]))
        pb = np.mean([ea.predict_proba(Xa[idx]), eb.predict_proba(Xb[idx])], axis=0)
        return pa, pb

    hit80 = dual_search(y[ca], *pack(ca), 0.80, MIN_N)
    hit75 = dual_search(y[ca], *pack(ca), 0.75, MIN_N)
    thr = (hit80 or hit75 or {}).get("threshold", 0.60)
    test = apply_dual(y[te], *pack(te), thr)
    return {
        "tag": tag,
        "n": int(len(y)),
        "up": int(y.sum()),
        "down": int((1 - y).sum()),
        "n_symbols": int(len(set(codes.tolist()))),
        "train_n": int(len(tr)),
        "calib_n": int(len(ca)),
        "test_n": int(len(te)),
        "calib_dual80": hit80,
        "calib_dual75": hit75,
        "test_dual": test,
        "test_oracle_dual80": dual_search(y[te], *pack(te), 0.80, max(12, MIN_N // 2)),
        "test_symbols": sorted(set(codes[te].tolist())),
    }


def eval_btc_walkforward(events):
    btc = [e for e in events if e["code"] == "BTCUSDT"]
    btc = sorted(btc, key=lambda e: e["date"])
    X = np.asarray([e["feats"] for e in btc], float)
    y = np.asarray([e["label"] for e in btc])
    dates = np.asarray([e["date"] for e in btc])
    Xa, Xb = X[:, :15], X[:, 15:]
    n = len(y)
    i1, i2 = int(0.60 * n), int(0.75 * n)
    tr = np.arange(0, max(0, i1 - EMBARGO))
    ca = np.arange(i1, max(i1, i2 - EMBARGO))
    te = np.arange(i2, n)
    if len(tr) < 100 or len(ca) < 30 or len(te) < 30:
        return {"error": "insufficient BTC samples", "n": n}

    ea, eb = make_clf(), make_clf()
    ea.fit(Xa[tr], y[tr])
    eb.fit(Xb[tr], y[tr])
    gate = make_clf()
    gate.fit(np.hstack([ea.predict_proba(Xa[tr])[:, 1:], eb.predict_proba(Xb[tr])[:, 1:]]), y[tr])

    def pack(idx):
        pa = gate.predict_proba(np.hstack([ea.predict_proba(Xa[idx])[:, 1:], eb.predict_proba(Xb[idx])[:, 1:]]))
        pb = np.mean([ea.predict_proba(Xa[idx]), eb.predict_proba(Xb[idx])], axis=0)
        return pa, pb

    hit80 = dual_search(y[ca], *pack(ca), 0.80, MIN_N)
    hit75 = dual_search(y[ca], *pack(ca), 0.75, MIN_N)
    thr = (hit80 or hit75 or {}).get("threshold", 0.60)
    test = apply_dual(y[te], *pack(te), thr)
    return {
        "tag": "BTCUSDT_purged_holdout",
        "n": n,
        "up": int(y.sum()),
        "down": int((1 - y).sum()),
        "date_test": f"{dates[te[0]]}→{dates[te[-1]]}",
        "calib_dual80": hit80,
        "calib_dual75": hit75,
        "test_dual": test,
        "test_oracle_dual80": dual_search(y[te], *pack(te), 0.80, max(12, MIN_N // 2)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    events = build_events(UNIVERSE, n_bars=1600)
    (OUT / "crypto_events_meta.json").write_text(
        json.dumps(
            {
                "n_events": len(events),
                "by_symbol": {
                    s: sum(1 for e in events if e["code"] == s) for s in sorted(set(e["code"] for e in events))
                },
                "labels": {
                    "UP": {"max_gain>=": UP_GAIN, "close>=": UP_CLOSE, "max_dd>=": UP_DD},
                    "DOWN": {"max_gain<=": DOWN_GAIN_MAX, "close<=": DOWN_CLOSE, "or_dd": DOWN_DD},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"total clear events={len(events)}", flush=True)

    btc_only = eval_btc_walkforward(events)
    print("BTC_ONLY", json.dumps(btc_only, ensure_ascii=False), flush=True)

    pool = eval_group_shuffle(events, "crypto_pool_group_shuffle")
    print("POOL", json.dumps(pool, ensure_ascii=False), flush=True)

    # BTC+ETH+BNB 小池再跑一遍，看稳健性
    core = [e for e in events if e["code"] in {"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"}]
    core_res = eval_group_shuffle(core, "core5_group_shuffle")
    print("CORE5", json.dumps(core_res, ensure_ascii=False), flush=True)

    def ok(r):
        return bool(r.get("test_dual", {}).get("meets"))

    summary = {
        "method": "tabular dual-gate (same as A-share milestone)",
        "bar": f"selective ≥{TARGET_ACC:.0%} @ coverage ≥{TARGET_COV:.0%}",
        "btc_only_purged": btc_only,
        "crypto_pool": pool,
        "core5": core_res,
        "verdict": {
            "btc_only": "PASS" if ok(btc_only) else "FAIL",
            "crypto_pool": "PASS" if ok(pool) else "FAIL",
            "core5": "PASS" if ok(core_res) else "FAIL",
            "note": (
                "A股成功依赖多股票截面；单BTC时序外推失败不否定方法，"
                "应以多币种 GroupShuffle 作为对等对照。"
            ),
        },
    }
    (OUT / "btc_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved", OUT / "btc_report.json")


if __name__ == "__main__":
    main()
