#!/usr/bin/env python3
"""v3 改进：防泄漏划分 + 更富人特征 + 更强表格模型 + 校准置信度。

相对 v2 的改进点:
  1) 按股票 Group 划分（同股相邻窗不再漏进测试集）——此前随机划分会虚高
  2) 在 TimesJev 统计外，加入收益/均线差/RSI/量能/MACD 等手工人特征
  3) HistGradientBoosting + 概率校准，专门冲「高置信 ≥75%」
  4) 只主攻 L=100（v2 已证明 L=200 更弱）；多通道 vs 仅收盘对照保留
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from timesjev.data.pool import selected
from timesjev.schema import Sample

V2_OUT = ROOT / "runs" / "smoke_updown_multimodal_v2"
CACHE = V2_OUT / "events_cache.pkl"
OUT = ROOT / "runs" / "smoke_updown_improve_v3"


@dataclass
class Event:
    code: str
    date: str
    label: int
    left_n: int
    close: list[float]
    vol: list[float]
    macd_h: list[float]
    rsi: list[float]
    ma5: list[float]
    ma20: list[float]


def load_events(left_n: int = 100) -> list[Event]:
    raw = pickle.loads(CACHE.read_bytes())
    rows = raw[left_n] if left_n in raw else raw[str(left_n)]
    return [Event(**e) if isinstance(e, dict) else e for e in rows]


def handcrafted(ev: Event) -> list[float]:
    c = np.asarray(ev.close, float)
    v = np.asarray(ev.vol, float)
    m = np.asarray(ev.macd_h, float)
    r = np.asarray(ev.rsi, float)
    ma5 = np.asarray(ev.ma5, float)
    ma20 = np.asarray(ev.ma20, float)

    def ret(n: int) -> float:
        if len(c) <= n or c[-n - 1] == 0:
            return 0.0
        return float(c[-1] / c[-n - 1] - 1)

    # 左窗内相对位置 / 波动
    rng = float(c.max() - c.min()) + 1e-9
    pos = float((c[-1] - c.min()) / rng)
    vol_win = float(np.std(np.diff(c) / np.clip(c[:-1], 1e-9, None)))
    # 量能：近5 vs 全窗中位
    vol_ratio = float(np.mean(v[-5:]) / (np.median(v) + 1e-9))
    # 均线结构
    ma_spread = float(ma5[-1] / (ma20[-1] + 1e-9) - 1)
    ma_slope5 = float(ma5[-1] / (ma5[-6] + 1e-9) - 1) if len(ma5) > 6 else 0.0
    ma_slope20 = float(ma20[-1] / (ma20[-11] + 1e-9) - 1) if len(ma20) > 11 else 0.0
    # RSI / MACD
    rsi_last, rsi_mid = float(r[-1]), float(r[len(r) // 2])
    rsi_delta = rsi_last - rsi_mid
    macd_last = float(m[-1])
    macd_slope = float(m[-1] - m[-6]) if len(m) > 6 else 0.0
    # 连续涨跌天数
    d = np.diff(c)
    streak = 0
    for x in d[::-1]:
        if (x > 0 and streak >= 0) or (x < 0 and streak <= 0) or streak == 0:
            streak += 1 if x > 0 else (-1 if x < 0 else 0)
            if streak == 0:
                break
        else:
            break
    # 左窗自身「已实现」强弱（不是未来标签）
    left_gain = float(c.max() / c[0] - 1)
    left_dd = float(c.min() / c[0] - 1)
    return [
        ret(1), ret(5), ret(10), ret(20), ret(60) if len(c) > 60 else ret(20),
        pos, vol_win, vol_ratio,
        ma_spread, ma_slope5, ma_slope20,
        rsi_last / 100.0, rsi_delta / 100.0,
        macd_last, macd_slope,
        float(streak), left_gain, left_dd,
        float(np.mean(c[-5:]) / (np.mean(c[-20:]) + 1e-9) - 1),
        float((c[-1] / (ma20[-1] + 1e-9)) - 1),
    ]


def to_sample_close(ev: Event) -> Sample:
    return Sample(
        id=f"L{ev.left_n}/close/{ev.code}/{ev.date}/{ev.label}",
        task="gate_updown",
        split="train",
        context=f"{ev.code} L={ev.left_n}",
        question="up or down?",
        options={"A": "Yes — clear UP", "B": "No — clear DOWN"},
        answer="A" if ev.label == 1 else "B",
        series=[ev.close],
        domain="stock",
        source="v3",
        raw_problem="updown",
    )


def to_sample_mm(ev: Event) -> Sample:
    inter = []
    for a, b, c in zip(ev.close, ev.vol, ev.macd_h):
        inter.extend([a, b, c])
    return Sample(
        id=f"L{ev.left_n}/mm/{ev.code}/{ev.date}/{ev.label}",
        task="tsqa_classification",
        split="train",
        context=f"{ev.code} multimodal L={ev.left_n}",
        question="up or down?",
        options={"A": "Yes — clear UP", "B": "No — clear DOWN"},
        answer="A" if ev.label == 1 else "B",
        series=[inter],
        domain="stock",
        source="v3",
        raw_problem="updown",
    )


def write_selection(key: str, multimodal: bool) -> str:
    if multimodal:
        names = []
        for ch in ("X", "Y", "Z"):
            names += [f"{ch} std", f"{ch} range", f"{ch} slope", f"{ch} acf lag1", f"{ch} zero-crossing rate"]
        names += [
            "X-Y xcorr lag0", "X-Y best corr", "X-Z xcorr lag0", "X-Z best corr",
            "Y-Z xcorr lag0", "Y-Z best corr",
            "X high_fluctuation", "Y high_fluctuation", "X periodicity", "Z periodicity",
        ]
    else:
        names = [
            "std", "range", "IQR", "mean abs step", "std abs step", "max abs step",
            "acf lag1", "acf lag2", "slope", "argmax position", "zero-crossing rate",
            "dominant period", "periodicity", "high_fluctuation", "stats3 median", "c24 SD",
        ]
    path = ROOT / "configs" / "selections" / f"{key}.json"
    path.write_text(json.dumps({"names": names}, indent=2), encoding="utf-8")
    return key


def build_xy(events: list[Event], multimodal: bool, sel_key: str, use_tj: bool, use_hc: bool):
    X, y, groups, meta = [], [], [], []
    for ev in events:
        s = to_sample_mm(ev) if multimodal else to_sample_close(ev)
        feats = []
        if use_tj:
            vals, _ = selected(s, sel_key)
            if not vals or any(v != v or abs(v) == float("inf") for v in vals):
                continue
            feats.extend(vals)
        if use_hc:
            feats.extend(handcrafted(ev))
        if not feats:
            continue
        X.append(feats)
        y.append(ev.label)
        groups.append(ev.code)
        meta.append({"code": ev.code, "date": ev.date, "label": ev.label})
    return np.asarray(X, float), np.asarray(y), np.asarray(groups), meta


def conf_table(y_true, proba, thresholds=(0.55, 0.60, 0.65, 0.70, 0.75, 0.80)) -> dict:
    pred = proba.argmax(1)
    conf = proba.max(1)
    out = {
        "n": int(len(y_true)),
        "acc": float(accuracy_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, proba[:, 1])) if len(np.unique(y_true)) > 1 else None,
    }
    for thr in thresholds:
        m = conf >= thr
        out[f"acc>={thr}"] = float(accuracy_score(y_true[m], pred[m])) if m.any() else None
        out[f"n>={thr}"] = int(m.sum())
        out[f"cov>={thr}"] = float(m.mean())
    return out


def run_one(tag: str, X, y, groups, model_kind: str) -> dict:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
    tr, te = next(gss.split(X, y, groups))
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    # 测试集股票与训练集不重叠
    leak = len(set(groups[tr]) & set(groups[te]))
    if model_kind == "logreg":
        base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    elif model_kind == "hgb":
        base = HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.08, max_iter=250, l2_regularization=0.1, random_state=7
        )
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    else:
        raise ValueError(model_kind)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    tab = conf_table(yte, proba)
    # 选「acc≥0.75 且覆盖最大」的阈值
    best = None
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        a, n, cov = tab.get(f"acc>={thr}"), tab.get(f"n>={thr}"), tab.get(f"cov>={thr}")
        if a is not None and n >= 12 and a >= 0.75:
            if best is None or cov > best["coverage"]:
                best = {"threshold": thr, "acc": a, "n": n, "coverage": cov}
    return {
        "tag": tag,
        "model": model_kind,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_train_codes": int(len(set(groups[tr]))),
        "n_test_codes": int(len(set(groups[te]))),
        "code_leak_overlap": leak,
        "metrics": tab,
        "best_ge75": best,
        "beats_75_full": bool(tab["acc"] >= 0.75),
        "beats_75_conf": best is not None,
    }


def main():
    assert CACHE.is_file(), f"missing cache {CACHE}; run v2 scan first"
    OUT.mkdir(parents=True, exist_ok=True)
    events = load_events(100)
    print(f"events L=100: {len(events)} up={sum(e.label for e in events)}", flush=True)

    combos = []
    # 特征消融 × 模型
    specs = [
        ("close_tj", False, True, False),
        ("close_hc", False, False, True),
        ("close_tj+hc", False, True, True),
        ("mm_tj", True, True, False),
        ("mm_hc", True, False, True),
        ("mm_tj+hc", True, True, True),
    ]
    results = []
    for tag, mm, use_tj, use_hc in specs:
        sel = write_selection(f"updown-v3-{'mm' if mm else 'close'}", multimodal=mm) if use_tj else "none"
        X, y, groups, _ = build_xy(events, mm, sel if use_tj else "updown-v3-close", use_tj, use_hc)
        print(f"\n=== {tag} X={X.shape} ===", flush=True)
        for mk in ("logreg", "hgb"):
            r = run_one(f"{tag}/{mk}", X, y, groups, mk)
            results.append(r)
            b = r["best_ge75"]
            print(
                f"  {mk}: full={r['metrics']['acc']:.3f} auc={r['metrics']['auc']:.3f} "
                f"best75={b} codes_te={r['n_test_codes']} leak={r['code_leak_overlap']}",
                flush=True,
            )
            combos.append(r)

    # 对照：旧式随机划分（有泄漏）用最强特征，量化虚高多少
    sel = write_selection("updown-v3-mm", multimodal=True)
    X, y, groups, _ = build_xy(events, True, sel, True, True)
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, random_state=7, stratify=y)
    clf = CalibratedClassifierCV(
        HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.08, max_iter=250, l2_regularization=0.1, random_state=7
        ),
        method="isotonic",
        cv=3,
    )
    clf.fit(X[tr], y[tr])
    leak_codes = len(set(groups[tr]) & set(groups[te]))
    leak_tab = conf_table(y[te], clf.predict_proba(X[te]))
    leak_cmp = {
        "tag": "mm_tj+hc/hgb/RANDOM_SPLIT_LEAKY",
        "code_overlap": leak_codes,
        "metrics": leak_tab,
        "note": "同股窗可同时出现在 train/test，数字会虚高；仅作对照",
    }
    print("\n=== LEAKY random split (control) ===", json.dumps(leak_cmp, indent=2), flush=True)

    any75 = any(r.get("beats_75_conf") or r.get("beats_75_full") for r in results)
    best_full = max(results, key=lambda r: r["metrics"]["acc"])
    best_cov75 = max(
        (r for r in results if r.get("best_ge75")),
        key=lambda r: r["best_ge75"]["coverage"],
        default=None,
    )
    report = {
        "goal": "v3 improve: group-split (no code leak) + handcrafted features + HGB + calibration",
        "improvements_vs_v2": [
            "GroupShuffleSplit by stock code (honest generalization)",
            "handcrafted return/MA/RSI/vol/MACD features",
            "HistGradientBoosting + probability calibration",
            "ablation: TJ-only / HC-only / TJ+HC × close / multimodal",
            "leaky random-split control to quantify inflation",
        ],
        "n_events": len(events),
        "results": results,
        "leaky_control": leak_cmp,
        "best_full_acc": {"tag": best_full["tag"], "acc": best_full["metrics"]["acc"], "auc": best_full["metrics"]["auc"]},
        "best_coverage_at_ge75": best_cov75,
        "any_beats_75_conf_group_split": any75,
        "recommend": (
            "线上：Group 训练的最强模型 + 置信阈值；只对 best_ge75 阈值以上出手，其余当震荡弃权。"
            if best_cov75
            else "Group 划分下暂未稳定 ≥75%；先加严标签或扩股票池后再试。"
        ),
    }
    (OUT / "improve_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print("best full:", report["best_full_acc"])
    print("best cov@≥75%:", best_cov75)
    print("any≥75% (group split):", any75)
    print("saved", OUT / "improve_report.json")


if __name__ == "__main__":
    main()
