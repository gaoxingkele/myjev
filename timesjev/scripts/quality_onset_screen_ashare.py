#!/usr/bin/env python3
"""A股优质起涨点样本筛选（步骤1）：Q1/Q2 与 R20 / clear-UP / S20 概念对照。

优质起涨定义（用户）:
  Q1 长窗连续大涨: 买点后持续上行 ≥20/30 日，涨幅 ≥50%，且不跌破起涨点
  Q2 短窗暴涨不破: 60 日内涨幅 ≥50%，且触及目标前低点不跌破起涨点

对照既有标签:
  R20: 20 日内 (收盘≥+25% 或 最大涨幅≥+25%) 且 最大回撤 ≥ -15%
  clear_UP (TimesJev v2): 20 日 max_gain≥15% 且 close≥8% 且 dd≥-10%
  S20: 工程上是 20 日持有排序/安全池，不是同一绝对收益标签；此处用「R20 失败的中等上涨」近似对照质量带

输出: runs/quality_onset_ashare/
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "quality_onset_ashare"
CACHE = OUT / "bar_cache"
STOCK_AGENT = Path(r"D:/aicoding/stockagent-analysis")
STOCK_S20 = Path(r"D:/aicoding/stockagent-s20")
LEFT_MIN = 60  # 起涨点左侧至少有多少根用于后续训练

# 标签阈值
Q_GAIN = 0.50
Q1_MIN_DAYS = (20, 30)
Q2_HORIZON = 60
R20_GAIN, R20_DD = 0.25, -0.15
CLEAR_GAIN, CLEAR_CLOSE, CLEAR_DD = 0.15, 0.08, -0.10


def load_universe(max_n: int = 400) -> list[str]:
    codes: list[str] = []
    paths = [
        STOCK_S20 / "output/experiments/daily_best/r20_pool_a_20261029_codes.txt",
        STOCK_S20 / "output/experiments/daily_best/s20_hybrid_20261029_codes.txt",
        STOCK_AGENT / "output/daily_best_20261029/r20_pool_a_20261029_codes.txt",
        STOCK_AGENT / "output/daily_best_20261029/s20_hybrid_20261029_codes.txt",
    ]
    for p in paths:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            c = line.strip().split(".")[0].zfill(6)
            if c.isdigit():
                codes.append(c)
    qdir = STOCK_AGENT / "output/quality_uptrend_charts"
    if qdir.is_dir():
        for d in qdir.iterdir():
            if d.is_dir() and not d.name.startswith("_") and "_" in d.name:
                codes.append(d.name.split("_")[0].zfill(6))
    # TimesJev events_cache codes
    evp = ROOT / "runs/smoke_updown_multimodal_v2/events_cache.pkl"
    if evp.is_file():
        import pickle

        raw = pickle.loads(evp.read_bytes())
        rows = raw[100] if 100 in raw else raw[list(raw.keys())[0]]
        for e in rows:
            c = (e["code"] if isinstance(e, dict) else e.code).split(".")[0].zfill(6)
            codes.append(c)
    # 再补沪深300成分（加大训练量）
    try:
        import akshare as ak

        hs = ak.index_stock_cons_csindex(symbol="000300")
        col = "成分券代码" if "成分券代码" in hs.columns else hs.columns[0]
        for c in hs[col].astype(str).str.zfill(6).tolist():
            codes.append(c)
    except Exception as e:
        print("hs300 skip", e, flush=True)

    seen, out = set(), []
    for c in codes:
        if c in seen or len(c) != 6:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= max_n:
            break
    return out


def fetch_bars(code6: str, start="20210101", end=None) -> pd.DataFrame | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    end = end or time.strftime("%Y%m%d")
    fp = CACHE / f"{code6}_{start}_{end}.parquet"
    if fp.exists():
        try:
            return pd.read_parquet(fp)
        except Exception:
            pass
    try:
        import akshare as ak

        df = ak.stock_zh_a_hist(
            symbol=code6, period="daily", start_date=start, end_date=end, adjust="qfq"
        )
    except Exception as e:
        print(f"  fetch fail {code6}: {e}", flush=True)
        return None
    if df is None or len(df) < LEFT_MIN + Q2_HORIZON + 5:
        return None
    # normalize columns (akshare CN headers)
    rename = {}
    for c in df.columns:
        cl = str(c)
        if "日期" in cl or cl.lower() == "date":
            rename[c] = "date"
        elif cl in ("开盘", "open"):
            rename[c] = "open"
        elif cl in ("收盘", "close"):
            rename[c] = "close"
        elif cl in ("最高", "high"):
            rename[c] = "high"
        elif cl in ("最低", "low"):
            rename[c] = "low"
        elif "成交量" in cl or cl.lower() == "volume":
            rename[c] = "vol"
    df = df.rename(columns=rename)
    need = ["date", "open", "high", "low", "close"]
    if any(c not in df.columns for c in need):
        return None
    df = df[need + (["vol"] if "vol" in df.columns else [])].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df.to_parquet(fp, index=False)
    return df


def _path_stats(entry: float, highs, lows, closes, horizon: int):
    h = np.asarray(highs[:horizon], float)
    l = np.asarray(lows[:horizon], float)
    c = np.asarray(closes[:horizon], float)
    if len(c) < horizon or entry <= 0:
        return None
    return {
        "max_gain": float(h.max() / entry - 1),
        "max_dd": float(l.min() / entry - 1),
        "r_close": float(c[-1] / entry - 1),
        "min_low": float(l.min()),
        "highs": h,
        "lows": l,
        "closes": c,
    }


def label_r20(entry, highs, lows, closes) -> bool:
    s = _path_stats(entry, highs, lows, closes, 20)
    if not s:
        return False
    return (s["r_close"] >= R20_GAIN or s["max_gain"] >= R20_GAIN) and s["max_dd"] >= R20_DD


def label_clear_up(entry, highs, lows, closes) -> bool:
    s = _path_stats(entry, highs, lows, closes, 20)
    if not s:
        return False
    return s["max_gain"] >= CLEAR_GAIN and s["r_close"] >= CLEAR_CLOSE and s["max_dd"] >= CLEAR_DD


def label_q2(entry, highs, lows, closes, horizon=Q2_HORIZON, gain=Q_GAIN):
    """60 日内涨幅≥50%，且触及目标前不跌破起涨点。"""
    n = min(horizon, len(closes))
    if n < 5 or entry <= 0:
        return False, None
    h = np.asarray(highs[:n], float)
    l = np.asarray(lows[:n], float)
    hit = None
    for t in range(n):
        if l[t] < entry:
            return False, None  # 跌破起涨点
        if h[t] / entry - 1 >= gain:
            hit = t + 1
            break
    if hit is None:
        return False, None
    return True, {"hit_day": hit, "max_gain": float(h[:hit].max() / entry - 1)}


def label_q1(entry, highs, lows, closes, min_days: int, gain=Q_GAIN, soft=True):
    """连续上行 ≥min_days 且涨幅≥50%，全程不跌破起涨点。

    soft=True: 允许少量回调日，但要求 (1) 不破起点 (2) 第 min_days 日及之后某日收盘≥+50%
               (3) 收盘序列对时间的 Spearman 相关 > 0.6（整体向上）
    soft=False: 严格每日收盘 ≥ 前收（几乎仅极限连板）
    """
    max_look = min(len(closes), max(min_days + 40, 80))
    if max_look < min_days or entry <= 0:
        return False, None
    c = np.asarray(closes[:max_look], float)
    l = np.asarray(lows[:max_look], float)
    h = np.asarray(highs[:max_look], float)
    for t in range(min_days, max_look + 1):
        if l[:t].min() < entry:
            return False, None
        if soft:
            seg = c[:t]
            if float(seg[-1] / entry - 1) < gain and float(h[:t].max() / entry - 1) < gain:
                continue
            # 整体向上
            x = np.arange(t)
            if np.std(seg) < 1e-12:
                continue
            corr = float(np.corrcoef(x, seg)[0, 1])
            up_frac = float((np.diff(seg) >= 0).mean()) if t > 1 else 0.0
            if corr >= 0.60 and up_frac >= 0.55 and float(h[:t].max() / entry - 1) >= gain:
                return True, {"days": t, "max_gain": float(h[:t].max() / entry - 1), "corr": corr, "up_frac": up_frac}
        else:
            if t >= 2 and np.any(np.diff(c[:t]) < 0):
                continue
            if float(c[t - 1] / entry - 1) >= gain:
                return True, {"days": t, "max_gain": float(c[t - 1] / entry - 1)}
    return False, None


def label_neg_break(entry, highs, lows, closes, horizon=60):
    """优质起涨对立面：较快跌破起点，且窗口内冲高有限。"""
    n = min(horizon, len(closes))
    if n < 10 or entry <= 0:
        return False
    l = np.asarray(lows[:n], float)
    h = np.asarray(highs[:n], float)
    # 20 日内跌破起点 8%+，且 60 日最大涨幅 < 15%
    early = min(20, n)
    broke = float(l[:early].min() / entry - 1) <= -0.08
    weak = float(h.max() / entry - 1) < 0.15
    return bool(broke and weak)


def scan_code(code: str, df: pd.DataFrame) -> list[dict]:
    opens = df["open"].to_numpy(float)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    dates = df["date"].tolist()
    rows = []
    # 决策日 i：用次日开盘作 entry（与既有约定一致）
    last = len(df) - Q2_HORIZON - 2
    for i in range(LEFT_MIN - 1, max(LEFT_MIN, last)):
        entry = float(opens[i + 1])
        if entry <= 0:
            continue
        fut_h, fut_l, fut_c = highs[i + 1 :], lows[i + 1 :], closes[i + 1 :]
        r20 = label_r20(entry, fut_h, fut_l, fut_c)
        clear = label_clear_up(entry, fut_h, fut_l, fut_c)
        q1_20, m120 = label_q1(entry, fut_h, fut_l, fut_c, 20, soft=True)
        q1_30, m130 = label_q1(entry, fut_h, fut_l, fut_c, 30, soft=True)
        q1_20s, _ = label_q1(entry, fut_h, fut_l, fut_c, 20, soft=False)
        q2, m2 = label_q2(entry, fut_h, fut_l, fut_c)
        neg = label_neg_break(entry, fut_h, fut_l, fut_c)
        if not (r20 or clear or q1_20 or q1_30 or q2 or neg):
            continue
        rows.append(
            {
                "code": code,
                "date": dates[i],
                "entry_date": dates[i + 1],
                "entry": entry,
                "R20": int(r20),
                "clear_UP": int(clear),
                "Q1_20": int(q1_20),
                "Q1_30": int(q1_30),
                "Q1_20_strict": int(q1_20s),
                "Q2": int(q2),
                "NEG_break": int(neg),
                "q1_20_meta": m120,
                "q1_30_meta": m130,
                "q2_meta": m2,
            }
        )
    return rows


def overlap_report(df: pd.DataFrame) -> dict:
    def rate(a, b):
        # P(b|a)
        m = df[a] == 1
        if m.sum() == 0:
            return None
        return float((df.loc[m, b] == 1).mean())

    flags = ["R20", "clear_UP", "Q1_20", "Q1_30", "Q1_20_strict", "Q2", "NEG_break"]
    counts = {k: int((df[k] == 1).sum()) for k in flags}
    # 互斥/包含
    matrix = {}
    for a in flags:
        matrix[a] = {b: rate(a, b) for b in flags}
    # 冲突：Q 正与 NEG 同时
    conflict = {
        "Q2_and_NEG": int(((df["Q2"] == 1) & (df["NEG_break"] == 1)).sum()),
        "Q1_20_and_NEG": int(((df["Q1_20"] == 1) & (df["NEG_break"] == 1)).sum()),
        "Q2_not_R20": int(((df["Q2"] == 1) & (df["R20"] == 0)).sum()),
        "Q2_and_R20": int(((df["Q2"] == 1) & (df["R20"] == 1)).sum()),
        "Q2_and_clear": int(((df["Q2"] == 1) & (df["clear_UP"] == 1)).sum()),
        "R20_not_Q2": int(((df["R20"] == 1) & (df["Q2"] == 0)).sum()),
        "clear_not_Q2": int(((df["clear_UP"] == 1) & (df["Q2"] == 0)).sum()),
    }
    return {"counts": counts, "P_B_given_A": matrix, "conflicts": conflict}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    universe = load_universe(420)
    (OUT / "universe.json").write_text(json.dumps(universe, indent=2), encoding="utf-8")
    print(f"universe={len(universe)}", flush=True)

    all_rows = []
    ok = 0
    for i, code in enumerate(universe):
        print(f"[{i+1}/{len(universe)}] {code}", flush=True)
        df = fetch_bars(code)
        if df is None:
            continue
        ok += 1
        rows = scan_code(code, df)
        all_rows.extend(rows)
        if (i + 1) % 20 == 0:
            print(f"  partial events={len(all_rows)} ok_codes={ok}", flush=True)
        time.sleep(0.12)

    if not all_rows:
        print("NO ROWS", flush=True)
        return

    df = pd.DataFrame(all_rows)
    # 只保留至少带一个关注标签的行已在 scan 完成
    df.to_parquet(OUT / "onset_candidates.parquet", index=False)
    df.to_csv(OUT / "onset_candidates.csv", index=False, encoding="utf-8-sig")

    # 纯正样本表
    pos = df[(df["Q1_20"] == 1) | (df["Q1_30"] == 1) | (df["Q2"] == 1)].copy()
    pos.to_csv(OUT / "quality_positives.csv", index=False, encoding="utf-8-sig")
    neg = df[df["NEG_break"] == 1].copy()
    neg.to_csv(OUT / "quality_negatives.csv", index=False, encoding="utf-8-sig")

    rep = overlap_report(df)
    # 整合建议
    integration = {
        "concept_map": {
            "R20": "20 日绝对目标约 +25%（可忍受 -15% 回撤）——中等优质",
            "S20": "20 日持有组合的排序/安全筛选层，不是单一绝对涨幅标签；与 Q 不直接等价",
            "clear_UP": "TimesJev dual-gate 用的清晰上涨（约 +15% 路径）——比 R20 更宽",
            "Q1": "长窗连续大涨 +50%（比 R20 严很多）——R20/clear 的严格子集倾向",
            "Q2": "60 日暴涨 +50% 且不破起点——强趋势突破；通常蕴含较强 R20/clear，但不必然",
            "NEG_break": "快速跌破起点且冲高弱——Q 的对立面，适合做负样本",
        },
        "quality_ladder": [
            "NEG_break (负)",
            "既非清晰也非失败的 chop（本脚本未全量落盘）",
            "clear_UP",
            "R20",
            "Q1_20 / Q2（顶级起涨）",
        ],
        "conflict_summary": rep["conflicts"],
        "recommendation": {
            "train_positive": "优先 Q2 ∪ Q1_20；若样本不足，分层加入 R20（medium pos）",
            "train_negative": "NEG_break；可加 R20 明确失败样本（另脚本）",
            "no_hard_conflict_with_R20": "Q 是更严的正类，与 R20 是包含/加强关系而非矛盾；S20 仍作选股层而非标签层",
            "next": "加大 A 股宇宙与年份后，按 Q 标签做 dual-gate/起涨点专用训练",
        },
    }
    summary = {
        "n_universe_requested": len(universe),
        "n_codes_with_bars": ok,
        "n_candidate_rows": int(len(df)),
        "n_quality_pos": int(len(pos)),
        "n_quality_neg": int(len(neg)),
        "label_counts": rep["counts"],
        "overlap": rep,
        "integration": integration,
        "date_range": {"min": str(df["date"].min()), "max": str(df["date"].max())},
        "top_codes_Q2": df[df["Q2"] == 1]["code"].value_counts().head(15).to_dict(),
    }
    (OUT / "screen_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append("# A股优质起涨点筛选报告（步骤1）\n")
    md.append(f"- 宇宙请求 {len(universe)} 只，成功拉到日线 {ok} 只")
    md.append(f"- 候选事件行 {len(df)}（至少命中某一标签）")
    md.append(f"- 日期 {df['date'].min()} → {df['date'].max()}\n")
    md.append("## 标签计数\n")
    for k, v in rep["counts"].items():
        md.append(f"- **{k}**: {v}")
    md.append("\n## 与 R20 / clear_UP 关系\n")
    md.append(f"- Q2 ∩ R20 = {rep['conflicts']['Q2_and_R20']}；Q2 \\ R20 = {rep['conflicts']['Q2_not_R20']}")
    md.append(f"- R20 \\ Q2 = {rep['conflicts']['R20_not_Q2']}（R20 更宽）")
    md.append(f"- Q2 ∩ clear_UP = {rep['conflicts']['Q2_and_clear']}")
    md.append(f"- Q2 ∩ NEG = {rep['conflicts']['Q2_and_NEG']}（应为 0）\n")
    md.append("## 概念整合\n")
    for k, v in integration["concept_map"].items():
        md.append(f"- **{k}**: {v}")
    md.append("\n## 训练建议\n")
    for k, v in integration["recommendation"].items():
        md.append(f"- **{k}**: {v}")
    md.append("\n## 质量阶梯\n")
    md.append(" → ".join(integration["quality_ladder"]))
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary["label_counts"], ensure_ascii=False, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
