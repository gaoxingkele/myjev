#!/usr/bin/env python3
"""扩大美股成分股 + 基本面/重大事件 的 dual-gate 实验。

相对 btc_usmega:
  - 股票池: 标普500按 GICS 分层抽样 ~100 只（跨行业，不含山寨币；可选拼 BTC）
  - 事件: Nasdaq 财报 surprise、美联储 FOMC、美国宏观日历(CPI/NFP/PCE/GDP 等)、FRED 利率/CPI 变化
  - 特征: 原 K 线手工特征 + 事件距离/窗口/surprise
  - 评估: GroupShuffle dual-gate（HGB），多 seed
"""
from __future__ import annotations

import csv
import io
import json
import re
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "usbroad_events_dual_gate"
CACHE = OUT / "cache"
LEFT, HORIZON, EMBARGO = 100, 20, 20
TARGET_ACC, TARGET_COV, MIN_N = 0.80, 0.30, 20
SEED = 7
PER_SECTOR = 9  # ~99 names across 11 sectors
UP_GAIN, UP_CLOSE, UP_DD = 0.08, 0.04, -0.08
DOWN_GAIN_MAX, DOWN_CLOSE, DOWN_DD = 0.03, -0.04, -0.08

# 公开 FOMC 决议日（近似，用于距离特征）
FOMC_DATES = [
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]
MACRO_KEYWORDS = (
    "cpi", "consumer price", "payroll", "nonfarm", "nfp", "pce", "gdp", "fomc", "fed interest",
    "federal funds", "jobless claims", "unemployment rate", "retail sales", "ism manufacturing",
)


def _get(url: str, timeout=60, retries=3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/csv,*/*"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def yahoo_symbol(sym: str) -> str:
    return sym.replace(".", "-")


def load_sp500_universe(per_sector=PER_SECTOR) -> list[str]:
    text = _get("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv").decode()
    rows = list(csv.DictReader(io.StringIO(text)))
    # 优先纳入巨头
    priority = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "JPM", "XOM",
        "UNH", "JNJ", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "COST", "WMT",
        "BAC", "CRM", "AMD", "NFLX", "ADBE", "ORCL", "CSCO", "LIN", "ACN", "MCD", "DIS",
        "INTC", "QCOM", "TXN", "AMAT", "CAT", "GE", "BA", "RTX", "SPGI", "BLK", "GS",
    ]
    by = defaultdict(list)
    for r in rows:
        by[r["GICS Sector"]].append(yahoo_symbol(r["Symbol"]))
    chosen, seen = [], set()
    for s in priority:
        if any(s in vals for vals in by.values()) and s not in seen:
            chosen.append(s)
            seen.add(s)
    for sector, vals in sorted(by.items()):
        n_have = sum(1 for c in chosen if c in vals)
        for s in vals:
            if n_have >= per_sector:
                break
            if s in seen or s in ("GOOG",):  # 避免双类股重复
                continue
            chosen.append(s)
            seen.add(s)
            n_have += 1
    return chosen


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


def fetch_binance_btc(n_bars=1600):
    limit, end, rows = 1000, None, []
    while len(rows) < n_bars:
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit={limit}"
        if end is not None:
            url += f"&endTime={end}"
        batch = json.loads(_get(url).decode())
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
    out.sort(key=lambda x: x["date"])
    return out[-n_bars:]


def fetch_fred_csv(series_id: str):
    text = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}").decode()
    rows = []
    for line in text.strip().splitlines()[1:]:
        d, v = line.split(",")[:2]
        if v in ("", "."):
            continue
        rows.append((d, float(v)))
    return rows


def daterange(start: str, end: str):
    a = datetime.strptime(start, "%Y-%m-%d")
    b = datetime.strptime(end, "%Y-%m-%d")
    cur = a
    while cur <= b:
        if cur.weekday() < 5:
            yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)


def _first_fridays(start: str, end: str):
    """NFP 近似：每月第一个周五。"""
    a = datetime.strptime(start[:7] + "-01", "%Y-%m-%d")
    b = datetime.strptime(end, "%Y-%m-%d")
    out = []
    y, m = a.year, a.month
    while datetime(y, m, 1) <= b:
        d0 = datetime(y, m, 1)
        # weekday: Mon=0 ... Fri=4
        delta = (4 - d0.weekday()) % 7
        ff = d0 + timedelta(days=delta)
        if start <= ff.strftime("%Y-%m-%d") <= end:
            out.append({"date": ff.strftime("%Y-%m-%d"), "name": "NFP proxy (1st Friday)"})
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def _cpi_proxies(start: str, end: str):
    """CPI 近似：每月第二个周三（常见发布窗口附近）。"""
    a = datetime.strptime(start[:7] + "-01", "%Y-%m-%d")
    b = datetime.strptime(end, "%Y-%m-%d")
    out = []
    y, m = a.year, a.month
    while datetime(y, m, 1) <= b:
        d0 = datetime(y, m, 1)
        # first Wednesday
        delta = (2 - d0.weekday()) % 7
        second_wed = d0 + timedelta(days=delta + 7)
        ds = second_wed.strftime("%Y-%m-%d")
        if start <= ds <= end:
            out.append({"date": ds, "name": "CPI proxy (2nd Wednesday)"})
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def _fetch_earnings_day(d: str):
    try:
        raw = json.loads(_get(f"https://api.nasdaq.com/api/calendar/earnings?date={d}", timeout=25, retries=2).decode())
        rows = []
        for row in (raw.get("data") or {}).get("rows") or []:
            sym = yahoo_symbol((row.get("symbol") or "").upper())
            if not sym:
                continue
            sur = row.get("surprise")
            try:
                sur_f = float(str(sur).replace("%", "").replace(",", "")) if sur not in (None, "N/A", "") else 0.0
            except Exception:
                sur_f = 0.0
            rows.append((sym, {"date": d, "surprise": sur_f}))
        return rows
    except Exception:
        return []


def load_or_build_calendar(start="2021-01-01", end=None):
    """拉取 Nasdaq 日度财报（并行）+ FOMC/NFP/CPI 宏观日程，磁盘缓存。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    end = end or time.strftime("%Y-%m-%d", time.gmtime())
    path = CACHE / f"nasdaq_calendar_{start}_{end}.json"
    if path.exists():
        print("calendar cache hit", path.name, flush=True)
        return json.loads(path.read_text(encoding="utf-8"))

    CACHE.mkdir(parents=True, exist_ok=True)
    earnings_by_sym = defaultdict(list)
    # 财报季更集中在周二到周四，仍覆盖全部工作日但并行加速
    dates = list(daterange(start, end))
    print(f"fetching nasdaq earnings days={len(dates)} (parallel)...", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_fetch_earnings_day, d): d for d in dates}
        for fut in as_completed(futs):
            for sym, rec in fut.result():
                earnings_by_sym[sym].append(rec)
            done += 1
            if done % 150 == 0:
                print(f"  earnings {done}/{len(dates)} syms={len(earnings_by_sym)}", flush=True)
                path.write_text(
                    json.dumps({"earnings": dict(earnings_by_sym), "macro": []}, ensure_ascii=False),
                    encoding="utf-8",
                )

    macro_dates = [{"date": d, "name": "FOMC"} for d in FOMC_DATES if start <= d <= end]
    macro_dates += _first_fridays(start, end)
    macro_dates += _cpi_proxies(start, end)
    # 稀疏校验：抽若干天拉真实 US 宏观事件，合并进去
    sample_days = [d for d in dates if d.endswith(("-15", "-01")) ][:40]
    for d in sample_days:
        try:
            raw = json.loads(_get(f"https://api.nasdaq.com/api/calendar/economicevents?date={d}", timeout=20, retries=1).decode())
            for row in (raw.get("data") or {}).get("rows") or []:
                country = (row.get("country") or "").lower()
                name = row.get("eventName") or ""
                if "united states" not in country and country not in ("usa", "us"):
                    continue
                if any(k in name.lower() for k in MACRO_KEYWORDS):
                    macro_dates.append({"date": d, "name": name})
        except Exception:
            pass

    payload = {"earnings": dict(earnings_by_sym), "macro": macro_dates}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print("calendar saved", path, "macro", len(macro_dates), flush=True)
    return payload


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


def handcrafted_price(close, vol, macd_h, rsi_v, ma5, ma20):
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


def to_ord(d: str) -> int:
    return datetime.strptime(d, "%Y-%m-%d").toordinal()


def nearest_dist(date: str, sorted_ords: np.ndarray):
    """返回 (days_since_prev, days_to_next)，若无则大数。"""
    if len(sorted_ords) == 0:
        return 365.0, 365.0
    o = to_ord(date)
    idx = int(np.searchsorted(sorted_ords, o))
    if idx == 0:
        return 365.0, float(sorted_ords[0] - o)
    if idx >= len(sorted_ords):
        return float(o - sorted_ords[-1]), 365.0
    return float(o - sorted_ords[idx - 1]), float(sorted_ords[idx] - o)


def event_features(date: str, sym: str, earn_map, macro_ords, fomc_ords, fred_feat):
    earns = earn_map.get(sym, [])
    e_ords = np.asarray([to_ord(x["date"]) for x in earns], int) if earns else np.asarray([], int)
    since_e, to_e = nearest_dist(date, e_ords)
    # last surprise
    o = to_ord(date)
    last_sur = 0.0
    for e in reversed(earns):
        if to_ord(e["date"]) <= o:
            last_sur = float(e["surprise"])
            break
    since_m, to_m = nearest_dist(date, macro_ords)
    since_f, to_f = nearest_dist(date, fomc_ords)
    in_earn = 1.0 if min(since_e, to_e) <= 5 else 0.0
    in_fomc = 1.0 if min(since_f, to_f) <= 3 else 0.0
    in_macro = 1.0 if min(since_m, to_m) <= 2 else 0.0
    # FRED as-of (lagged monthly) — 用 date 之前最近观测
    ff_chg, cpi_chg = fred_feat.get(date, (0.0, 0.0))
    return [
        min(since_e, 120.0) / 120.0,
        min(to_e, 120.0) / 120.0,
        float(np.tanh(last_sur / 20.0)),
        in_earn,
        min(since_f, 90.0) / 90.0,
        min(to_f, 90.0) / 90.0,
        in_fomc,
        min(since_m, 60.0) / 60.0,
        min(to_m, 60.0) / 60.0,
        in_macro,
        float(ff_chg),
        float(cpi_chg),
    ]


def build_fred_asof_map():
    try:
        ff = fetch_fred_csv("FEDFUNDS")
        cpi = fetch_fred_csv("CPIAUCSL")
    except Exception as e:
        print(f"FRED unavailable ({e}); using zeros", flush=True)
        return {}

    def chg_series(rows):
        out = []
        for i, (d, v) in enumerate(rows):
            if i == 0:
                out.append((d, 0.0))
            else:
                prev = rows[i - 1][1]
                out.append((d, float((v - prev) / (abs(prev) + 1e-9))))
        return out

    ff_c, cpi_c = chg_series(ff), chg_series(cpi)
    all_dates = list(daterange("2020-01-01", time.strftime("%Y-%m-%d", time.gmtime())))
    ff_i = cpi_i = 0
    cur_ff = cur_cpi = 0.0
    mp = {}
    for d in all_dates:
        while ff_i < len(ff_c) and ff_c[ff_i][0] <= d:
            cur_ff = ff_c[ff_i][1]
            ff_i += 1
        while cpi_i < len(cpi_c) and cpi_c[cpi_i][0] <= d:
            cur_cpi = cpi_c[cpi_i][1]
            cpi_i += 1
        mp[d] = (cur_ff, cur_cpi)
    return mp


def bars_to_events(code, bars, earn_map, macro_ords, fomc_ords, fred_feat, asset="us"):
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
        date = bars[i]["date"]
        feats = handcrafted_price(closes[sl], vols[sl], macd_h[sl], rsi_v[sl], ma5[sl], ma20[sl])
        feats += event_features(date, code, earn_map, macro_ords, fomc_ords, fred_feat)
        out.append({"code": code, "date": date, "label": lab, "feats": feats, "asset": asset})
    return out


def make_clf():
    return CalibratedClassifierCV(
        HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.07, max_iter=250, l2_regularization=0.2, random_state=7
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


def eval_group(events, tag, seed=SEED, use_events=True):
    X = np.asarray([e["feats"] for e in events], float)
    if not use_events:
        X = X[:, :30]  # price-only ablation
    y = np.asarray([e["label"] for e in events])
    codes = np.asarray([e["code"] for e in events])
    # 专家切分：价格半 / 价格后半+事件
    mid = 15
    Xa, Xb = X[:, :mid], X[:, mid:]
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    trval, te = next(gss1.split(X, y, codes))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 4)
    tri, cai = next(gss2.split(X[trval], y[trval], codes[trval]))
    tr, ca = trval[tri], trval[cai]

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
    return {
        "tag": tag,
        "seed": seed,
        "use_events": use_events,
        "n": int(len(y)),
        "n_symbols": int(len(set(codes.tolist()))),
        "up": int(y.sum()),
        "down": int((1 - y).sum()),
        "test_symbols_n": int(len(set(codes[te].tolist()))),
        "calib_dual80": hit80,
        "calib_dual75": hit75,
        "test_dual": test,
        "test_oracle_dual80": dual_search(y[te], *pack(te), 0.80, max(12, MIN_N // 2)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    universe = load_sp500_universe(PER_SECTOR)
    (OUT / "universe.json").write_text(json.dumps(universe, indent=2), encoding="utf-8")
    print(f"universe n={len(universe)}", flush=True)

    print("building FRED asof...", flush=True)
    fred_feat = build_fred_asof_map()

    cal = load_or_build_calendar("2022-01-01")
    earn_map = cal["earnings"]
    macro_ords = np.asarray(sorted({to_ord(x["date"]) for x in cal["macro"]}), int)
    fomc_ords = np.asarray(sorted(to_ord(d) for d in FOMC_DATES), int)
    print(f"earnings symbols={len(earn_map)} macro_days={len(macro_ords)}", flush=True)

    # prices
    price_cache = CACHE / "prices.json"
    prices = json.loads(price_cache.read_text(encoding="utf-8")) if price_cache.exists() else {}
    for i, sym in enumerate(universe):
        if sym in prices and len(prices[sym]) > 400:
            continue
        print(f"price {i+1}/{len(universe)} {sym}", flush=True)
        try:
            prices[sym] = fetch_yahoo_daily(sym)
        except Exception as e:
            print("  fail", e, flush=True)
            prices[sym] = []
        if (i + 1) % 10 == 0:
            price_cache.write_text(json.dumps(prices), encoding="utf-8")
        time.sleep(0.15)
    # BTC optional
    print("price BTCUSDT", flush=True)
    prices["BTCUSDT"] = fetch_binance_btc()
    price_cache.write_text(json.dumps(prices), encoding="utf-8")

    events_us = []
    for sym in universe:
        ev = bars_to_events(sym, prices.get(sym, []), earn_map, macro_ords, fomc_ords, fred_feat, "us")
        print(f"  events {sym}={len(ev)}", flush=True)
        events_us.extend(ev)
    events_btc = bars_to_events("BTCUSDT", prices.get("BTCUSDT", []), earn_map, macro_ords, fomc_ords, fred_feat, "btc")
    pool = events_us + events_btc
    print(f"total us={len(events_us)} btc={len(events_btc)}", flush=True)

    meta = {
        "n_universe": len(universe),
        "n_us_events": len(events_us),
        "n_btc_events": len(events_btc),
        "earnings_symbols_hit": sum(1 for s in universe if s in earn_map),
        "macro_days": int(len(macro_ords)),
        "labels": {"UP": [UP_GAIN, UP_CLOSE, UP_DD], "DOWN": [DOWN_GAIN_MAX, DOWN_CLOSE, DOWN_DD]},
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    seeds = [7, 11, 17, 23, 29, 37]
    rows = []
    for seed in seeds:
        for tag, ev, use_ev in [
            ("us_price_only", events_us, False),
            ("us_price_plus_events", events_us, True),
            ("btc_us_plus_events", pool, True),
        ]:
            if len(ev) < 500:
                continue
            r = eval_group(ev, tag, seed=seed, use_events=use_ev)
            rows.append(r)
            td = r["test_dual"]
            print(
                f"seed={seed} {tag} full={td['full_acc']:.3f} sel={td['acc']} cov={td['coverage']:.3f} meets={td['meets']}",
                flush=True,
            )

    def agg(tag):
        xs = [r for r in rows if r["tag"] == tag]
        sels = [r["test_dual"]["acc"] for r in xs if r["test_dual"]["acc"] is not None]
        covs = [r["test_dual"]["coverage"] for r in xs]
        fulls = [r["test_dual"]["full_acc"] for r in xs]
        return {
            "n": len(xs),
            "mean_full": float(np.mean(fulls)) if fulls else None,
            "mean_sel": float(np.mean(sels)) if sels else None,
            "mean_cov": float(np.mean(covs)) if covs else None,
            "n_meet": int(sum(1 for r in xs if r["test_dual"]["meets"])),
        }

    summary = {
        "meta": meta,
        "bar": "selective>=80% @ coverage>=30%",
        "runs": rows,
        "aggregate": {
            "us_price_only": agg("us_price_only"),
            "us_price_plus_events": agg("us_price_plus_events"),
            "btc_us_plus_events": agg("btc_us_plus_events"),
        },
        "compare_note": {
            "ashare_tabular": "~80% @ ~50%",
            "us_mega10_prev": "~69% @ ~37%",
            "crypto15_prev": "~66% @ ~24%",
        },
    }
    (OUT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("AGG", json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print("saved", OUT / "report.json")


if __name__ == "__main__":
    main()
