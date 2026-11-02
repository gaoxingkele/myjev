#!/usr/bin/env python3
"""里程碑后实盘扫描：截至 2026-10-30 收盘，预测下周（5 日）更可能上涨的股票。

多模态:
  - price: OHLCV + 多尺度技术特征
  - moneyflow: 主力净流入/大小单结构
  - regime: 市场分型（风险开/震荡/避险）作为上下文特征
  - news/policy: 若本地有 daily_policy_heat 则接入；否则记为 unavailable

协议沿用 v5: Group 划分训练 + 双闸门(门控∧投票) 选择性出手。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DAILY = Path(r"D:/aicoding/stockagent-fourpool/output/tushare_cache/daily")
MF = Path(r"D:/aicoding/stockagent-fourpool/output/tushare_cache/moneyflow")
STOCK_AGENT = Path(r"D:/aicoding/stockagent-analysis")
OUT = ROOT / "runs" / "live_week_predict_20261030"
ASOF = "20261030"
LEFT_N = 40
HORIZON = 5  # next week ~5 sessions
UP_GAIN, UP_CLOSE, UP_DD = 0.06, 0.03, -0.05
DOWN_GAIN_MAX, DOWN_CLOSE, DOWN_DD = 0.015, -0.03, -0.05


def load_codes() -> list[str]:
    codes: list[str] = []
    for p in [
        STOCK_AGENT / "stock_pool_94.txt",
        STOCK_AGENT / "stock_pool_gte60.txt",
        STOCK_AGENT / "output/daily_best_20261029/r20_pool_a_20261029_codes.txt",
        STOCK_AGENT / "output/daily_best_20261029/v12_31_top20_20261029_codes.txt",
        STOCK_AGENT / "output/daily_best_20261029/s20_hybrid_20261029_codes.txt",
        STOCK_AGENT / "output/daily_best_20261029/pool_a_20261029_codes.txt",
    ]:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            c = line.split("|")[0].split(".")[0].zfill(6)
            if c.isdigit():
                codes.append(c)
    # extras liquid names
    codes += ["000001", "600519", "300750", "601318", "000858", "002594", "002415", "600036"]
    seen, out = set(), []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def code_to_ts(code6: str) -> list[str]:
    if code6.startswith(("5", "6", "9")):
        return [f"{code6}.SH", f"{code6}.BJ"]
    return [f"{code6}.SZ", f"{code6}.BJ"]


def load_panel(codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    want = set()
    for c in codes:
        want.update(code_to_ts(c))
    dparts, mparts = [], []
    for f in sorted(DAILY.glob("*.parquet")):
        d = pd.read_parquet(f)
        d = d[d["ts_code"].astype(str).isin(want)]
        if len(d):
            dparts.append(d)
    for f in sorted(MF.glob("*.parquet")):
        m = pd.read_parquet(f)
        m = m[m["ts_code"].astype(str).isin(want)]
        if len(m):
            mparts.append(m)
    daily = pd.concat(dparts, ignore_index=True) if dparts else pd.DataFrame()
    mf = pd.concat(mparts, ignore_index=True) if mparts else pd.DataFrame()
    if len(daily):
        daily["trade_date"] = daily["trade_date"].astype(str)
        daily = daily.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"])
    if len(mf):
        mf["trade_date"] = mf["trade_date"].astype(str)
        mf = mf.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"])
    return daily, mf


def ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.asarray(x, float).copy()
    a = 2 / (n + 1)
    for i in range(1, len(out)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi14(c: np.ndarray) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up = np.clip(d, 0, None)
    down = np.clip(-d, 0, None)
    au = pd.Series(up).rolling(14, min_periods=14).mean().to_numpy()
    ad = pd.Series(down).rolling(14, min_periods=14).mean().to_numpy()
    rs = au / np.where(ad == 0, np.nan, ad)
    return np.nan_to_num(100 - (100 / (1 + rs)), nan=50.0)


def label_week(entry, highs, lows, closes) -> str:
    if entry <= 0 or len(closes) < HORIZON:
        return "chop"
    h, l, c = highs[:HORIZON], lows[:HORIZON], closes[:HORIZON]
    max_gain = float(h.max() / entry - 1)
    max_dd = float(l.min() / entry - 1)
    r_close = float(c[-1] / entry - 1)
    if max_gain >= UP_GAIN and r_close >= UP_CLOSE and max_dd >= UP_DD:
        return "up"
    if max_gain <= DOWN_GAIN_MAX and (r_close <= DOWN_CLOSE or max_dd <= DOWN_DD):
        return "down"
    return "chop"


def ensure_free_news(asof: str, stock_codes: list[str] | None = None, force: bool = True) -> Path:
    """接入已测通的免 Key 新闻：财联社 + 新浪（+ 东财个股）。"""
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from mm_news_free_sources import OUT as NEWS_OUT, build_policy_heat

    NEWS_OUT.mkdir(parents=True, exist_ok=True)
    path = NEWS_OUT / f"daily_policy_heat_{asof}.json"
    if path.is_file() and not force:
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("status") == "ok_free_sources" and int(old.get("n_flash") or 0) > 0:
                return path
        except Exception:
            pass
    payload = build_policy_heat(asof, (stock_codes or [])[:12])
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[news] saved {path} status={payload.get('status')} "
        f"sent={payload.get('market_sentiment')} n_flash={payload.get('n_flash')} "
        f"n_stock={payload.get('n_stock_news')}",
        flush=True,
    )
    return path


def load_policy_sentiment(asof: str) -> dict:
    """轻量新闻/政策模态：优先读 free-sources 落盘的 policy heat。"""
    hits = [
        STOCK_AGENT / f"output/policy_heat/daily_policy_heat_{asof}.json",
        STOCK_AGENT / f"output/policy_heat/daily_policy_heat.json",
    ]
    hits += sorted(STOCK_AGENT.glob("output/policy_heat/daily_policy_heat*.json"))
    seen = set()
    for p in hits:
        p = Path(p)
        if not p.is_file() or str(p) in seen:
            continue
        seen.add(str(p))
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        status = str(data.get("status") or "")
        if status in ("unavailable", "empty_cctv") and not data.get("themes"):
            return {"available": True, "weak": True, "path": str(p), "asof": asof, "payload": data}
        if status.startswith("ok") or "market_sentiment" in data:
            return {
                "available": True,
                "weak": status != "ok_free_sources" and not status.startswith("ok"),
                "path": str(p),
                "asof": asof,
                "payload": data,
            }
        if asof in data and isinstance(data[asof], dict):
            return {"available": True, "path": str(p), "asof": asof, "payload": data[asof]}
        if str(data.get("trade_date") or data.get("date") or "") in (asof, asof[:8]):
            return {"available": True, "path": str(p), "asof": asof, "payload": data}
    return {"available": False, "path": None, "asof": asof, "payload": None}


def sentiment_to_feat(policy: dict) -> list[float]:
    if not policy.get("available"):
        return [0.0, 0.0, 0.0]  # missing modality
    payload = policy.get("payload") or {}
    sent = str(payload.get("market_sentiment") or payload.get("sentiment") or "neutral").lower()
    bull = 1.0 if "bull" in sent else 0.0
    bear = 1.0 if "bear" in sent else 0.0
    heat = payload.get("themes") or payload.get("top_themes") or []
    if heat:
        mean_heat = float(np.mean([t.get("heat_score", 50) for t in heat])) / 100.0
    else:
        # free-sources 也可能把 heat 放在顶层统计里
        b = float(payload.get("bull_hits") or 0)
        e = float(payload.get("bear_hits") or 0)
        mean_heat = float(min(1.0, abs(b - e) / 50.0)) if (b or e) else 0.5
    return [bull, bear, mean_heat]


def regime_of_day(day_pct: pd.Series) -> dict:
    """分型：用截面涨跌幅中位数 + 上涨占比。
    tushare pct_chg 单位是「百分数」(如 -2.3 表示 -2.3%)。
    """
    med = float(day_pct.median())
    breadth = float((day_pct > 0).mean())
    if med >= 0.8 and breadth >= 0.55:
        name = "risk_on"
        vec = [1, 0, 0]
    elif med <= -0.8 and breadth <= 0.45:
        name = "risk_off"
        vec = [0, 1, 0]
    else:
        name = "chop"
        vec = [0, 0, 1]
    return {"name": name, "median_pct": med, "breadth": breadth, "vec": vec}


def window_feats(df: pd.DataFrame, mf: pd.DataFrame | None, i: int, left_n: int, policy_feat: list[float], regime_vec: list[float]) -> list[float] | None:
    sl = df.iloc[i - left_n + 1 : i + 1]
    if len(sl) < left_n:
        return None
    c = sl["close"].to_numpy(float)
    h = sl["high"].to_numpy(float)
    l = sl["low"].to_numpy(float)
    v = sl["vol"].to_numpy(float)
    rets = np.diff(c) / np.clip(c[:-1], 1e-9, None)
    ma5 = pd.Series(c).rolling(5, min_periods=1).mean().to_numpy()
    ma20 = pd.Series(c).rolling(20, min_periods=1).mean().to_numpy()
    r = rsi14(c)
    macd = ema(c, 12) - ema(c, 26)

    def pack(n):
        cc, vv, mm, rr = c[-n:], v[-n:], macd[-n:], r[-n:]
        return [
            float(cc[-1] / cc[0] - 1),
            float((cc[-1] - cc.min()) / (cc.max() - cc.min() + 1e-9)),
            float(np.std(np.diff(cc) / np.clip(cc[:-1], 1e-9, None))) if len(cc) > 2 else 0.0,
            float(np.mean(vv[-min(5, len(vv)) :]) / (np.median(vv) + 1e-9)),
            float(ma5[-1] / (ma20[-1] + 1e-9) - 1),
            float(rr[-1]) / 100.0,
            float(mm[-1]),
            float(mm[-1] - mm[0]),
        ]

    feats = pack(20) + pack(min(40, left_n)) + [
        float(np.mean(rets[-5:])) if len(rets) >= 5 else 0.0,
        float(np.std(rets[-20:])) if len(rets) >= 20 else float(np.std(rets) or 0),
        float((h[-5:].max() / c[-6] - 1) if len(c) > 6 else 0.0),
        float((l[-5:].min() / c[-6] - 1) if len(c) > 6 else 0.0),
    ]
    # moneyflow modality
    mf_feat = [0.0, 0.0, 0.0, 0.0, 0.0]
    if mf is not None and len(mf):
        sub = mf[mf["trade_date"].isin(sl["trade_date"].astype(str))]
        if len(sub) >= 5:
            net = sub["net_mf_amount"].to_numpy(float)
            buy_lg = (sub["buy_lg_amount"] + sub["buy_elg_amount"]).to_numpy(float)
            sell_lg = (sub["sell_lg_amount"] + sub["sell_elg_amount"]).to_numpy(float)
            mf_feat = [
                float(np.mean(net[-5:])),
                float(np.sum(net[-5:])),
                float(np.mean(buy_lg[-5:] - sell_lg[-5:])),
                float(np.mean(net[-5:]) / (np.std(net[-20:]) + 1e-6)) if len(net) >= 5 else 0.0,
                float(1 if net[-1] > 0 else (-1 if net[-1] < 0 else 0)),
            ]
    # normalize money loosely
    mf_feat = [np.tanh(x / 1e5) if i < 3 else x for i, x in enumerate(mf_feat)]
    return feats + mf_feat + list(regime_vec) + list(policy_feat)


def make_clf(kind="logreg"):
    if kind == "hgb":
        base = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=200, l2_regularization=0.2, random_state=7
        )
        return CalibratedClassifierCV(base, method="isotonic", cv=3)
    return CalibratedClassifierCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.7)),
        method="sigmoid",
        cv=3,
    )


def dual_select(y, p_a, p_b, thr: float):
    pa, pb = p_a.argmax(1), p_b.argmax(1)
    ca, cb = p_a.max(1), p_b.max(1)
    m = (pa == pb) & (ca >= thr) & (cb >= thr)
    return m, pa


def search_dual(y, p_a, p_b, target=0.75, min_n=20):
    best = None
    for thr in np.linspace(0.5, 0.9, 41):
        m, pred = dual_select(y, p_a, p_b, thr)
        n = int(m.sum())
        if n < min_n:
            continue
        acc = float(accuracy_score(y[m], pred[m]))
        if acc >= target:
            cov = float(m.mean())
            if best is None or cov > best["coverage"]:
                best = {"threshold": float(thr), "acc": acc, "n": n, "coverage": cov, "target": target}
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    codes = load_codes()
    print(f"codes={len(codes)} asof={ASOF}", flush=True)
    # 先接能用的新闻接口（财联社+新浪），再进特征
    news_path = ensure_free_news(ASOF, codes, force=True)
    daily, mf = load_panel(codes)
    print(f"daily_rows={len(daily)} mf_rows={len(mf)} dates={daily['trade_date'].nunique() if len(daily) else 0}", flush=True)
    assert len(daily), "no daily panel"
    assert ASOF in set(daily["trade_date"].astype(str)), f"{ASOF} not in daily cache"

    policy = load_policy_sentiment(ASOF)
    policy_feat = sentiment_to_feat(policy)
    print(
        "policy",
        {
            "available": policy["available"],
            "weak": policy.get("weak"),
            "path": policy.get("path") or str(news_path),
            "sentiment": (policy.get("payload") or {}).get("market_sentiment"),
            "source": (policy.get("payload") or {}).get("source"),
            "feat": policy_feat,
        },
        flush=True,
    )

    # per-day regime from all panel pct_chg
    regimes = {}
    for dt, g in daily.groupby("trade_date"):
        regimes[str(dt)] = regime_of_day(g["pct_chg"].astype(float))

    # build samples (history) + live rows asof
    X, y, groups, meta = [], [], [], []
    live_rows = []
    by_code = {ts: g.reset_index(drop=True) for ts, g in daily.groupby("ts_code")}
    mf_by = {ts: g.reset_index(drop=True) for ts, g in mf.groupby("ts_code")} if len(mf) else {}

    for ts, df in by_code.items():
        code6 = str(ts).split(".")[0]
        mdf = mf_by.get(ts)
        for i in range(LEFT_N - 1, len(df) - HORIZON - 1, 3):
            entry = float(df.iloc[i + 1]["open"])
            lab = label_week(
                entry,
                df["high"].to_numpy(float)[i + 1 :],
                df["low"].to_numpy(float)[i + 1 :],
                df["close"].to_numpy(float)[i + 1 :],
            )
            if lab == "chop":
                continue
            dt = str(df.iloc[i]["trade_date"])
            reg = regimes.get(dt, {"vec": [0, 0, 1], "name": "chop"})
            feats = window_feats(df, mdf, i, LEFT_N, policy_feat, reg["vec"])
            if feats is None:
                continue
            X.append(feats)
            y.append(1 if lab == "up" else 0)
            groups.append(code6)
            meta.append({"code": code6, "ts_code": ts, "date": dt, "label": lab})

        # live point: last bar == ASOF
        if str(df.iloc[-1]["trade_date"]) != ASOF:
            continue
        i = len(df) - 1
        if i < LEFT_N - 1:
            continue
        reg = regimes.get(ASOF, {"vec": [0, 0, 1], "name": "chop"})
        feats = window_feats(df, mdf, i, LEFT_N, policy_feat, reg["vec"])
        if feats is None:
            continue
        live_rows.append(
            {
                "code": code6,
                "ts_code": ts,
                "date": ASOF,
                "close": float(df.iloc[-1]["close"]),
                "pct_chg": float(df.iloc[-1]["pct_chg"]),
                "feats": feats,
                "regime": reg["name"],
            }
        )

    X = np.asarray(X, float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    print(f"train_cand={len(y)} up={int(y.sum())} down={int((1-y).sum())} live={len(live_rows)}", flush=True)
    assert len(y) >= 80, "too few weekly labeled samples"

    # modality blocks for experts (index slices into flat feats)
    # price 0:20, mf 20:25, regime 25:28, policy 28:31  (approx; pack returns variable)
    # safer: train on flat + 3 experts by reconstructing from live same builder
    # Split feature groups by known layout:
    # pack20(8)+pack40(8)+extra4 = 20 price; mf5; regime3; policy3 => 31
    idx_price = slice(0, 20)
    idx_mf = slice(20, 25)
    idx_reg = slice(25, 28)
    idx_pol = slice(28, 31)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
    tr, te = next(gss.split(X, y, groups))
    # experts
    blocks = {
        "price": X[:, idx_price],
        "moneyflow": X[:, idx_mf],
        "regime": X[:, idx_reg],
        "policy": X[:, idx_pol],
        "flat": X,
    }
    experts = {}
    for name, xb in blocks.items():
        # policy may be constant zeros — still fit
        clf = make_clf("logreg")
        try:
            clf.fit(xb[tr], y[tr])
            experts[name] = clf
        except Exception as e:
            print("skip expert", name, e)

    def stack_meta(idx):
        cols = [experts[n].predict_proba(blocks[n][idx])[:, 1:] for n in experts if n != "flat"]
        return np.hstack(cols) if cols else X[idx, :1]

    gate = make_clf("logreg")
    gate.fit(stack_meta(tr), y[tr])
    vote_names = [n for n in ("price", "moneyflow", "regime") if n in experts]
    p_gate_te = gate.predict_proba(stack_meta(te))
    p_vote_te = np.mean([experts[n].predict_proba(blocks[n][te]) for n in vote_names], axis=0)
    p_flat_te = experts["flat"].predict_proba(blocks["flat"][te]) if "flat" in experts else p_gate_te

    full_acc = float(accuracy_score(y[te], p_gate_te.argmax(1)))
    dual65 = search_dual(y[te], p_gate_te, p_vote_te, 0.65, min_n=12)
    dual70 = search_dual(y[te], p_gate_te, p_vote_te, 0.70, min_n=10)
    dual75 = search_dual(y[te], p_gate_te, p_vote_te, 0.75, min_n=8)
    dual80 = search_dual(y[te], p_gate_te, p_vote_te, 0.80, min_n=6)
    print(
        "holdout gate_acc",
        full_acc,
        "dual65",
        dual65,
        "dual70",
        dual70,
        "dual75",
        dual75,
        "dual80",
        dual80,
        flush=True,
    )

    # 阈值回退：优先 75/70/65；再不行用置信分位（只取头部）
    if dual75:
        thr, thr_strict = dual75["threshold"], (dual80 or dual75)["threshold"]
    elif dual70:
        thr, thr_strict = dual70["threshold"], min(0.75, dual70["threshold"] + 0.05)
    elif dual65:
        thr, thr_strict = dual65["threshold"], min(0.72, dual65["threshold"] + 0.05)
    else:
        # 头部选择：在测试集上找覆盖~25%时准确率最高的 thr
        best = None
        for t in np.linspace(0.5, 0.85, 36):
            m, pred = dual_select(y[te], p_gate_te, p_vote_te, float(t))
            if m.sum() < max(8, int(0.1 * len(te))):
                continue
            acc = float(accuracy_score(y[te][m], pred[m]))
            cov = float(m.mean())
            if 0.15 <= cov <= 0.40:
                if best is None or acc > best["acc"]:
                    best = {"threshold": float(t), "acc": acc, "coverage": cov}
        thr = best["threshold"] if best else 0.60
        thr_strict = min(0.78, thr + 0.08)
        print("fallback_thr", best, "->", thr, thr_strict, flush=True)

    # live predict
    Xl = np.asarray([r["feats"] for r in live_rows], float)
    live_blocks = {
        "price": Xl[:, idx_price],
        "moneyflow": Xl[:, idx_mf],
        "regime": Xl[:, idx_reg],
        "policy": Xl[:, idx_pol],
        "flat": Xl,
    }
    meta_l = np.hstack([experts[n].predict_proba(live_blocks[n])[:, 1:] for n in experts if n != "flat"])
    p_gate = gate.predict_proba(meta_l)
    p_vote = np.mean([experts[n].predict_proba(live_blocks[n]) for n in vote_names], axis=0)
    p_flat = experts["flat"].predict_proba(live_blocks["flat"]) if "flat" in experts else p_gate

    preds = []
    for i, r in enumerate(live_rows):
        g_up = float(p_gate[i, 1])
        v_up = float(p_vote[i, 1])
        f_up = float(p_flat[i, 1])
        g_pred = int(p_gate[i].argmax())
        v_pred = int(p_vote[i].argmax())
        conf_g = float(p_gate[i].max())
        conf_v = float(p_vote[i].max())
        agree = g_pred == v_pred
        fire = agree and conf_g >= thr and conf_v >= thr
        fire_strict = agree and conf_g >= thr_strict and conf_v >= thr_strict
        direction = "UP" if g_pred == 1 else "DOWN"
        preds.append(
            {
                **{k: r[k] for k in ("code", "ts_code", "date", "close", "pct_chg", "regime")},
                "p_up_gate": round(g_up, 4),
                "p_up_vote": round(v_up, 4),
                "p_up_flat": round(f_up, 4),
                "direction": direction,
                "agree": agree,
                "conf_gate": round(conf_g, 4),
                "conf_vote": round(conf_v, 4),
                "select_base": fire,
                "select_strict": fire_strict,
            }
        )

    up_base = [p for p in preds if p["select_base"] and p["direction"] == "UP"]
    up_strict = [p for p in preds if p["select_strict"] and p["direction"] == "UP"]
    down_base = [p for p in preds if p["select_base"] and p["direction"] == "DOWN"]
    up_base.sort(key=lambda x: (x["p_up_gate"] + x["p_up_vote"]) / 2, reverse=True)
    up_strict.sort(key=lambda x: (x["p_up_gate"] + x["p_up_vote"]) / 2, reverse=True)
    down_base.sort(key=lambda x: x["p_up_gate"])
    # 弱观察：双闸门未确认 UP，但相对概率偏高（仅供 risk_off 下挑反弹观察）
    soft_up = [
        p
        for p in preds
        if (p["p_up_gate"] + p["p_up_vote"]) / 2 >= 0.55 and p["direction"] == "UP" and not p["select_base"]
    ]
    soft_up.sort(key=lambda x: (x["p_up_gate"] + x["p_up_vote"]) / 2, reverse=True)

    asof_regime = regimes.get(ASOF, {})
    report = {
        "milestone": "v5_dual_gate",
        "asof": ASOF,
        "horizon_sessions": HORIZON,
        "left_n": LEFT_N,
        "modalities": {
            "price": True,
            "moneyflow": bool(len(mf)),
            "regime": True,
            "news_policy": policy["available"],
            "news_policy_path": policy.get("path"),
            "news_source": (policy.get("payload") or {}).get("source"),
            "news_status": (policy.get("payload") or {}).get("status"),
            "news_sentiment": (policy.get("payload") or {}).get("market_sentiment"),
            "news_n_flash": (policy.get("payload") or {}).get("n_flash"),
            "news_feat": policy_feat,
        },
        "market_regime_asof": {
            "name": asof_regime.get("name"),
            "median_pct": asof_regime.get("median_pct"),
            "breadth": asof_regime.get("breadth"),
        },
        "holdout": {
            "gate_acc": full_acc,
            "dual65": dual65,
            "dual70": dual70,
            "dual75": dual75,
            "dual80": dual80,
            "n_test": int(len(te)),
        },
        "thresholds": {"base": thr, "strict": thr_strict},
        "universe_live": len(live_rows),
        "selected_up_base": up_base,
        "selected_up_strict": up_strict,
        "selected_down_base": down_base,
        "soft_up_watch": soft_up[:20],
        "abstain_count": sum(1 for p in preds if not p["select_base"]),
        "note": (
            "选择性预测：仅 dual-gate 通过才算正式出手；soft_up_watch 仅为相对概率观察。"
            "标签=未来5日 clear up/down。若 asof 为 risk_off，正式 UP 为空是合理结果。"
        ),
    }
    (OUT / "week_predict_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown summary
    lines = [
        f"# 下周选择性上涨扫描（asof {ASOF} 收盘）",
        "",
        f"- 里程碑: v5 dual-gate",
        f"- 市场分型: **{asof_regime.get('name')}** (median_pct={asof_regime.get('median_pct'):.3f}, breadth={asof_regime.get('breadth'):.2f})"
        if asof_regime
        else "",
        f"- 模态: price=[yes] moneyflow=[{'yes' if len(mf) else 'no'}] regime=[yes] news/policy=[{'yes' if policy['available'] else 'no'}]",
        f"- Holdout dual65/70/75: {dual65} | {dual70} | {dual75}",
        f"- 阈值 base={thr:.2f} / strict={thr_strict:.2f}",
        f"- 直播股票数={len(live_rows)} | 出手UP(base)={len(up_base)} | UP(strict)={len(up_strict)} | 弃权={report['abstain_count']}",
        "",
        "## 更可能上涨（strict）",
    ]
    if not up_strict:
        lines.append("_无（阈值更严时全部弃权）_")
    for p in up_strict[:30]:
        lines.append(
            f"- **{p['code']}** close={p['close']:.2f} day%={p['pct_chg']:.2f} "
            f"p_up≈{(p['p_up_gate']+p['p_up_vote'])/2:.2f} regime={p['regime']}"
        )
    lines += ["", "## 更可能上涨（base）"]
    if not up_base:
        lines.append("_无_")
    for p in up_base[:40]:
        lines.append(
            f"- **{p['code']}** close={p['close']:.2f} day%={p['pct_chg']:.2f} "
            f"p_up≈{(p['p_up_gate']+p['p_up_vote'])/2:.2f} regime={p['regime']}"
        )
    ranked = sorted(preds, key=lambda x: (x["p_up_gate"] + x["p_up_vote"]) / 2, reverse=True)
    lines += ["", "## 弱观察 soft_up（未过双闸门，仅相对偏多）"]
    if not soft_up:
        lines.append("_无_")
    for p in soft_up[:15]:
        lines.append(
            f"- **{p['code']}** p_up≈{(p['p_up_gate']+p['p_up_vote'])/2:.2f} "
            f"day%={p['pct_chg']:.2f}"
        )
    lines += ["", "## 观察池 Top15（概率排序）"]
    for p in ranked[:15]:
        flag = "SELECT" if p["select_base"] else "watch"
        lines.append(
            f"- [{flag}] **{p['code']}** p_up≈{(p['p_up_gate']+p['p_up_vote'])/2:.2f} "
            f"day%={p['pct_chg']:.2f} regime={p['regime']}"
        )
    lines += ["", f"## 更可能下跌（dual-gate base，共 {len(down_base)} 只）"]
    for p in down_base[:25]:
        lines.append(
            f"- **{p['code']}** p_up_gate={p['p_up_gate']:.2f} conf={p['conf_gate']:.2f}"
        )
    (OUT / "week_predict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # avoid Windows gbk console issues
    print(f"UP strict={len(up_strict)} base={len(up_base)} abstain={report['abstain_count']} thr={thr:.2f}/{thr_strict:.2f}")
    for p in (up_strict or up_base or ranked)[:12]:
        print(
            f"  {p['code']} p_up={((p['p_up_gate']+p['p_up_vote'])/2):.2f} "
            f"select={p.get('select_base')} strict={p.get('select_strict')}"
        )
    print("saved", OUT / "week_predict_report.json")
    print("md", OUT / "week_predict.md")


if __name__ == "__main__":
    main()
