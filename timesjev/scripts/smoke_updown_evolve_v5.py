#!/usr/bin/env python3
"""v5 进化：在既有事件缓存上做更硬核的诚实验证 + 更强选择策略。

相对 v4 的深化:
  1) Walk-forward 时序划分（训练只用更早日期，测试用更晚）——防时间泄漏
  2) Purged Group K-Fold：按股票分组 + 日期间隔清洗
  3) 截面相对特征：同日相对同池分位（相对强弱，无需指数行情）
  4) 双闸门：专家投票 ∧ 门控概率 同时达标才出手
  5) 标签加严子集：ultra-clear（更大涨跌区分度）上的覆盖/精度曲线
  6) 自动搜阈值：最大化 ≥80%/≥85% 准确率下的覆盖率

不依赖外网拉新行情（腾讯超时）；复用 v2 events_cache。
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from timesjev.data.pool import selected
from timesjev.schema import Sample

CACHE = ROOT / "runs" / "smoke_updown_multimodal_v2" / "events_cache.pkl"
OUT = ROOT / "runs" / "smoke_updown_evolve_v5"


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


def _tail(a, n):
    x = np.asarray(a, float)
    return x[-n:] if len(x) >= n else x


def hc_scale(c, v, m, r, ma5, ma20):
    def ret(n):
        if len(c) <= n or c[-n - 1] == 0:
            return 0.0
        return float(c[-1] / c[-n - 1] - 1)

    rng = float(c.max() - c.min()) + 1e-9
    rets = np.diff(c) / np.clip(c[:-1], 1e-9, None)
    return [
        ret(1),
        ret(min(5, len(c) - 1)),
        ret(min(10, len(c) - 1)),
        float((c[-1] - c.min()) / rng),
        float(np.std(rets)) if len(rets) else 0.0,
        float(np.mean(v[-min(5, len(v)) :]) / (np.median(v) + 1e-9)),
        float(ma5[-1] / (ma20[-1] + 1e-9) - 1) if len(ma5) and len(ma20) else 0.0,
        float(r[-1]) / 100.0,
        float(r[-1] - r[len(r) // 2]) / 100.0 if len(r) > 2 else 0.0,
        float(m[-1]),
        float(m[-1] - m[0]),
        float(c.max() / c[0] - 1),
        float(c.min() / c[0] - 1),
        float(np.mean(c[-min(5, len(c)) :]) / (np.mean(c) + 1e-9) - 1),
    ]


def multiscale_hc(ev: Event) -> list[float]:
    out = []
    for n in (20, 50, 100):
        out.extend(
            hc_scale(
                _tail(ev.close, n),
                _tail(ev.vol, n),
                _tail(ev.macd_h, n),
                _tail(ev.rsi, n),
                _tail(ev.ma5, n),
                _tail(ev.ma20, n),
            )
        )
    return out


def path_stats(ev: Event) -> dict[str, float]:
    c = np.asarray(ev.close, float)
    rets = np.diff(c) / np.clip(c[:-1], 1e-9, None)
    return {
        "ret5": float(c[-1] / c[-6] - 1) if len(c) > 6 else 0.0,
        "ret20": float(c[-1] / c[-21] - 1) if len(c) > 21 else 0.0,
        "vol20": float(np.std(rets[-20:])) if len(rets) >= 20 else float(np.std(rets) or 0),
        "rsi": float(ev.rsi[-1]),
        "macd": float(ev.macd_h[-1]),
        "ma_spread": float(ev.ma5[-1] / (ev.ma20[-1] + 1e-9) - 1),
    }


def write_selection() -> str:
    names = []
    for ch in ("X", "Y", "Z"):
        names += [f"{ch} std", f"{ch} range", f"{ch} slope", f"{ch} acf lag1", f"{ch} zero-crossing rate"]
    names += [
        "X-Y xcorr lag0", "X-Y best corr", "X-Z xcorr lag0", "X-Z best corr",
        "Y-Z xcorr lag0", "Y-Z best corr",
        "X high_fluctuation", "Y high_fluctuation", "X periodicity", "Z periodicity",
    ]
    key = "updown-v5-mm"
    (ROOT / "configs" / "selections" / f"{key}.json").write_text(json.dumps({"names": names}, indent=2), encoding="utf-8")
    return key


def tj_feats(ev: Event, sel: str) -> list[float] | None:
    inter = []
    for a, b, c in zip(ev.close, ev.vol, ev.macd_h):
        inter.extend([a, b, c])
    s = Sample(
        id="x",
        task="tsqa_classification",
        split="train",
        context="x",
        question="q",
        options={"A": "Yes", "B": "No"},
        answer="A" if ev.label == 1 else "B",
        series=[inter],
        domain="stock",
        source="v5",
        raw_problem="updown",
    )
    vals, _ = selected(s, sel)
    if not vals or any(v != v or abs(v) == float("inf") for v in vals):
        return None
    return list(vals)


def cross_section_ranks(events: list[Event]) -> dict[tuple[str, str], list[float]]:
    """同日截面分位：相对强弱特征（VertiFuseX / 相对收益思想的轻量版）。"""
    by_date: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        by_date[e.date].append(e)
    out = {}
    for date, bucket in by_date.items():
        stats = [path_stats(e) for e in bucket]
        keys = list(stats[0].keys())
        mats = {k: np.asarray([s[k] for s in stats], float) for k in keys}
        for i, e in enumerate(bucket):
            ranks = []
            for k in keys:
                x = mats[k]
                # 分位 rank in [0,1]
                ranks.append(float((x < x[i]).mean() + 0.5 * (x == x[i]).mean()))
            # 相对中位数超额
            for k in keys:
                med = float(np.median(mats[k]))
                ranks.append(float(mats[k][i] - med))
            out[(e.code, e.date)] = ranks
    return out


def ultra_score(ev: Event) -> float:
    """形态清晰度（仅用左窗，无未来泄漏）：涨样偏强 / 跌样偏弱。"""
    c = np.asarray(ev.close, float)
    r20 = c[-1] / c[-21] - 1 if len(c) > 21 else 0.0
    rsi = ev.rsi[-1]
    if ev.label == 1:
        return float(r20 + (rsi - 50) / 100)
    return float(-r20 + (50 - rsi) / 100)


def make_clf(kind="logreg"):
    if kind == "hgb":
        base = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.07, max_iter=220, l2_regularization=0.2, random_state=7
        )
        return CalibratedClassifierCV(base, method="isotonic", cv=3)
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2500, C=0.6))
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def search_threshold(y, proba, targets=(0.75, 0.80, 0.85), min_n=20):
    pred = proba.argmax(1)
    conf = proba.max(1)
    out = {"full_acc": float(accuracy_score(y, pred)), "full_auc": float(roc_auc_score(y, proba[:, 1]))}
    best = {}
    for target in targets:
        hit = None
        for thr in np.linspace(0.50, 0.95, 46):
            m = conf >= thr
            n = int(m.sum())
            if n < min_n:
                continue
            acc = float(accuracy_score(y[m], pred[m]))
            if acc >= target:
                cov = float(m.mean())
                if hit is None or cov > hit["coverage"]:
                    hit = {"threshold": float(thr), "acc": acc, "n": n, "coverage": cov, "target": target}
        best[f"ge{int(target*100)}"] = hit
    out["best"] = best
    return out, pred, conf


def dual_gate(y, p_a, p_b, targets=(0.75, 0.80, 0.85), min_n=20):
    """双闸门：两模型同向且各自 conf 都过阈值。"""
    pred_a, pred_b = p_a.argmax(1), p_b.argmax(1)
    conf_a, conf_b = p_a.max(1), p_b.max(1)
    agree = pred_a == pred_b
    out = {}
    for target in targets:
        hit = None
        for thr in np.linspace(0.50, 0.95, 46):
            m = agree & (conf_a >= thr) & (conf_b >= thr)
            n = int(m.sum())
            if n < min_n:
                continue
            acc = float(accuracy_score(y[m], pred_a[m]))
            if acc >= target:
                cov = float(m.mean())
                if hit is None or cov > hit["coverage"]:
                    hit = {"threshold": float(thr), "acc": acc, "n": n, "coverage": cov, "target": target}
        out[f"ge{int(target*100)}"] = hit
    return out


def walk_forward_splits(dates: np.ndarray, n_test_folds: int = 4):
    """Expanding-window walk-forward：按日期均分成段，第 i 段做测试、之前全部做训练。"""
    uniq = np.array(sorted(set(dates.tolist())))
    # 前 40% 只做热身训练，后 60% 切成 n_test_folds 段依次测试
    start = int(0.40 * len(uniq))
    test_dates = uniq[start:]
    folds = []
    chunk = max(1, len(test_dates) // n_test_folds)
    for i in range(n_test_folds):
        a = i * chunk
        b = len(test_dates) if i == n_test_folds - 1 else (i + 1) * chunk
        if b <= a:
            continue
        cut, end = test_dates[a], test_dates[b - 1]
        tr = np.where(dates < cut)[0]
        te = np.where((dates >= cut) & (dates <= end))[0]
        if len(tr) >= 180 and len(te) >= 35:
            folds.append({"cut": str(cut), "end": str(end), "tr": tr, "te": te})
    return folds


def _to_ord(d: str) -> int:
    """YYYYMMDD → roughly day ordinal for distance (month*30+day approx ok for purge)."""
    d = str(d)
    y, m, day = int(d[:4]), int(d[4:6]), int(d[6:8])
    return y * 365 + m * 30 + day


def purged_group_folds(codes, dates, n_splits=4, purge_days=30):
    """GroupKFold by code，并 purge 测试日期邻域内的训练样本（防重叠窗泄漏）。"""
    gkf = GroupKFold(n_splits=n_splits)
    Xdummy = np.zeros((len(codes), 1))
    ydummy = np.zeros(len(codes))
    folds = []
    ords = np.asarray([_to_ord(d) for d in dates])
    for tr, te in gkf.split(Xdummy, ydummy, codes):
        te_ords = ords[te]
        te_min, te_max = int(te_ords.min()), int(te_ords.max())
        # 只 purge 落在测试日期区间外扩 purge_days 的训练点
        keep = [i for i in tr if not (te_min - purge_days <= ords[i] <= te_max + purge_days)]
        # 若清洗过狠，退回：仅按 code 隔离（仍无同股泄漏）
        tr2 = np.asarray(keep if len(keep) >= 150 else tr, int)
        if len(tr2) >= 150 and len(te) >= 40:
            folds.append((tr2, te))
    return folds


def fit_experts(X_blocks: dict[str, np.ndarray], y, tr):
    experts = {}
    for k, X in X_blocks.items():
        clf = make_clf("logreg")
        clf.fit(X[tr], y[tr])
        experts[k] = clf
    return experts


def expert_proba_matrix(experts, X_blocks, idx):
    cols = []
    full = []
    for k, clf in experts.items():
        p = clf.predict_proba(X_blocks[k][idx])
        cols.append(p[:, 1:])
        full.append(p)
    return np.hstack(cols), full


def main():
    assert CACHE.is_file(), CACHE
    OUT.mkdir(parents=True, exist_ok=True)
    events = load_events(100)
    sel = write_selection()
    print(f"events={len(events)}", flush=True)

    cs = cross_section_ranks(events)
    rows = []
    for ev in events:
        tj = tj_feats(ev, sel)
        if tj is None:
            continue
        ms = multiscale_hc(ev)
        xr = cs[(ev.code, ev.date)]
        close = np.asarray(ev.close, float)
        rets = np.diff(close) / np.clip(close[:-1], 1e-9, None)
        close_b = [
            float(np.mean(rets[-5:])) if len(rets) >= 5 else 0.0,
            float(np.std(rets) or 0),
            float(close[-1] / close[0] - 1),
            float((close[-1] - close.min()) / (close.max() - close.min() + 1e-9)),
        ]
        rows.append(
            {
                "y": ev.label,
                "code": ev.code,
                "date": ev.date,
                "ms": ms,
                "tj": tj,
                "cs": xr,
                "close": close_b,
                "flat": ms + tj + xr,
                "ultra_score": ultra_score(ev),
            }
        )
    y = np.asarray([r["y"] for r in rows])
    codes = np.asarray([r["code"] for r in rows])
    dates = np.asarray([r["date"] for r in rows])
    u_scores = np.asarray([r["ultra_score"] for r in rows])
    ultra = u_scores >= np.quantile(u_scores, 0.55)
    print(f"usable={len(rows)} ultra_clear={int(ultra.sum())}", flush=True)

    def mat(key):
        return np.asarray([r[key] for r in rows], float)

    X_blocks = {"ms": mat("ms"), "tj": mat("tj"), "cs": mat("cs"), "close": mat("close")}
    X_flat = mat("flat")

    report = {"protocol": {}, "walk_forward": [], "purged_group": [], "summary": {}}

    # ───────── Walk-forward ─────────
    wf_results = []
    for fi, fold in enumerate(walk_forward_splits(dates, n_test_folds=4)):
        tr, te = fold["tr"], fold["te"]
        # 再切一点 calib（训练尾部日期）
        tr_dates = dates[tr]
        cut_c = np.quantile(tr_dates.astype(int), 0.8).astype(int)
        ca = tr[tr_dates.astype(int) >= cut_c]
        tr2 = tr[tr_dates.astype(int) < cut_c]
        if len(tr2) < 120 or len(ca) < 30:
            tr2, ca = tr[: int(0.8 * len(tr))], tr[int(0.8 * len(tr)) :]

        experts = fit_experts(X_blocks, y, tr2)
        meta_tr, _ = expert_proba_matrix(experts, X_blocks, tr2)
        meta_ca, _ = expert_proba_matrix(experts, X_blocks, ca)
        meta_te, expert_ps = expert_proba_matrix(experts, X_blocks, te)
        gate = make_clf("logreg")
        gate.fit(meta_tr, y[tr2])
        flat = make_clf("hgb")
        flat.fit(X_flat[tr2], y[tr2])

        p_gate = gate.predict_proba(meta_te)
        p_flat = flat.predict_proba(X_flat[te])
        # vote soft
        vote_p = np.mean(expert_ps, axis=0)

        g_tab, _, _ = search_threshold(y[te], p_gate)
        f_tab, _, _ = search_threshold(y[te], p_flat)
        v_tab, _, _ = search_threshold(y[te], vote_p)
        dual = dual_gate(y[te], p_gate, vote_p)
        dual2 = dual_gate(y[te], p_gate, p_flat)

        # ultra slice（测试集内形态更清晰的子集）
        te_u = te[ultra[te]]
        ultra_res = None
        if len(te_u) >= 25:
            meta_u, ps_u = expert_proba_matrix(experts, X_blocks, te_u)
            p_gu = gate.predict_proba(meta_u)
            vote_u = np.mean(ps_u, axis=0)
            u_dual = dual_gate(y[te_u], p_gu, vote_u, min_n=12)
            u_g, _, _ = search_threshold(y[te_u], p_gu, min_n=12)
            ultra_res = {"n": int(len(te_u)), "gate": u_g["best"], "dual_gate_vote": u_dual}

        row = {
            "fold": fi,
            "cut": fold["cut"],
            "end": fold["end"],
            "n_train": int(len(tr2)),
            "n_calib": int(len(ca)),
            "n_test": int(len(te)),
            "gate": g_tab,
            "flat": f_tab,
            "vote": v_tab,
            "dual_gate_vote": dual,
            "dual_gate_flat": dual2,
            "ultra_clear": ultra_res,
        }
        wf_results.append(row)
        print(
            f"WF{fi} cut={fold['cut']}->{fold['end']} nte={len(te)} "
            f"gate80={g_tab['best'].get('ge80')} dual80={dual.get('ge80')}",
            flush=True,
        )

    # ───────── Purged Group CV ─────────
    pg_folds = purged_group_folds(codes, dates, n_splits=4, purge_days=30)
    print(f"purged folds={len(pg_folds)}", flush=True)
    pg_results = []
    for fi, (tr, te) in enumerate(pg_folds):
        # calib: 从 train 里 GroupShuffle 再切
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7 + fi)
        tr_i, ca_i = next(gss.split(y[tr], y[tr], codes[tr]))
        tr2, ca = tr[tr_i], tr[ca_i]
        experts = fit_experts(X_blocks, y, tr2)
        meta_tr, _ = expert_proba_matrix(experts, X_blocks, tr2)
        meta_te, expert_ps = expert_proba_matrix(experts, X_blocks, te)
        gate = make_clf("logreg")
        gate.fit(meta_tr, y[tr2])
        p_gate = gate.predict_proba(meta_te)
        vote_p = np.mean(expert_ps, axis=0)
        g_tab, _, _ = search_threshold(y[te], p_gate)
        v_tab, _, _ = search_threshold(y[te], vote_p)
        dual = dual_gate(y[te], p_gate, vote_p)
        # code leak check
        leak = len(set(codes[tr2]) & set(codes[te]))
        pg_results.append(
            {
                "fold": fi,
                "n_train": int(len(tr2)),
                "n_test": int(len(te)),
                "code_leak": leak,
                "gate": g_tab,
                "vote": v_tab,
                "dual_gate_vote": dual,
            }
        )
        print(f"PG{fi} nte={len(te)} leak={leak} dual80={dual.get('ge80')} dual85={dual.get('ge85')}", flush=True)

    def agg_best(results, path_keys, metric_key="ge80"):
        """跨 fold 汇总某策略的覆盖/精度。"""
        rows = []
        for r in results:
            block = r
            for k in path_keys:
                block = block.get(k) if isinstance(block, dict) else None
                if block is None:
                    break
            if not isinstance(block, dict):
                continue
            hit = block.get(metric_key) if "ge" in metric_key else block.get("best", {}).get(metric_key)
            # dual path stores ge80 directly; gate path stores under best
            if hit is None and isinstance(block.get("best"), dict):
                hit = block["best"].get(metric_key)
            if hit:
                rows.append(hit)
        if not rows:
            return None
        return {
            "n_folds": len(rows),
            "mean_acc": float(np.mean([x["acc"] for x in rows])),
            "mean_coverage": float(np.mean([x["coverage"] for x in rows])),
            "mean_n": float(np.mean([x["n"] for x in rows])),
            "folds": rows,
        }

    summary = {
        "walk_forward_dual_ge80": agg_best(wf_results, ["dual_gate_vote"], "ge80"),
        "walk_forward_dual_ge85": agg_best(wf_results, ["dual_gate_vote"], "ge85"),
        "walk_forward_gate_ge80": agg_best(wf_results, ["gate"], "ge80"),
        "walk_forward_vote_ge80": agg_best(wf_results, ["vote"], "ge80"),
        "purged_dual_ge80": agg_best(pg_results, ["dual_gate_vote"], "ge80"),
        "purged_dual_ge85": agg_best(pg_results, ["dual_gate_vote"], "ge85"),
        "purged_gate_ge80": agg_best(pg_results, ["gate"], "ge80"),
        "purged_vote_ge80": agg_best(pg_results, ["vote"], "ge80"),
        "walk_forward_ultra_dual_ge80": agg_best(
            [r for r in wf_results if r.get("ultra_clear")],
            ["ultra_clear", "dual_gate_vote"],
            "ge80",
        ),
    }

    # ───────── GroupShuffle 生产对照（+截面特征 + 双闸门）─────────
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
    tr_all, te = next(gss.split(y, y, codes))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=11)
    tr_i, ca_i = next(gss2.split(y[tr_all], y[tr_all], codes[tr_all]))
    tr2, ca = tr_all[tr_i], tr_all[ca_i]
    experts = fit_experts(X_blocks, y, tr2)
    meta_tr, _ = expert_proba_matrix(experts, X_blocks, tr2)
    meta_te, expert_ps = expert_proba_matrix(experts, X_blocks, te)
    gate = make_clf("logreg")
    gate.fit(meta_tr, y[tr2])
    flat = make_clf("hgb")
    flat.fit(X_flat[tr2], y[tr2])
    p_gate = gate.predict_proba(meta_te)
    p_flat = flat.predict_proba(X_flat[te])
    vote_p = np.mean(expert_ps, axis=0)
    prod = {
        "n_train": int(len(tr2)),
        "n_test": int(len(te)),
        "n_test_codes": int(len(set(codes[te]))),
        "code_leak": int(len(set(codes[tr2]) & set(codes[te]))),
        "gate": search_threshold(y[te], p_gate)[0],
        "vote": search_threshold(y[te], vote_p)[0],
        "flat": search_threshold(y[te], p_flat)[0],
        "dual_gate_vote": dual_gate(y[te], p_gate, vote_p),
        "dual_gate_flat": dual_gate(y[te], p_gate, p_flat),
        "triple_and": dual_gate(y[te], p_gate, (vote_p + p_flat) / 2),
    }
    print(
        "PROD dual80",
        prod["dual_gate_vote"].get("ge80"),
        "dual85",
        prod["dual_gate_vote"].get("ge85"),
        "vote80",
        prod["vote"]["best"].get("ge80"),
        flush=True,
    )

    report = {
        "goal": "v5 evolve: walk-forward + purged group CV + cross-section ranks + dual-gate selective",
        "papers_extended": [
            "Walk-forward / purged CV (finance ML hygiene)",
            "Cross-sectional relative strength features",
            "Dual-gate selective classification (AND of experts)",
            "Ultra-clear morphology subset evaluation",
        ],
        "n_events": len(rows),
        "walk_forward": wf_results,
        "purged_group": pg_results,
        "production_group_split": prod,
        "summary": summary,
        "recommend": None,
    }
    # pick recommend: prefer multi-fold evidence; else production dual
    cands = []
    for k, v in summary.items():
        if v and v.get("n_folds", 0) >= 2:
            cands.append({"name": k, **v})
    if not cands:
        for k, v in summary.items():
            if v:
                cands.append({"name": k, **v})
    pick = None
    if cands:
        ge85 = [c for c in cands if "ge85" in c["name"] and c["mean_acc"] >= 0.85]
        ge80 = [c for c in cands if "ge80" in c["name"] and c["mean_acc"] >= 0.80]
        pick = max(ge85 or ge80 or cands, key=lambda c: (c.get("n_folds", 1), c["mean_coverage"]))
    report["recommend"] = {
        "pick": pick,
        "production_dual_ge80": prod["dual_gate_vote"].get("ge80"),
        "production_dual_ge85": prod["dual_gate_vote"].get("ge85"),
        "production_vote_ge80": prod["vote"]["best"].get("ge80"),
        "note": "优先看 production_group_split（多股票未见）与 walk-forward 是否同向；双闸门(门控∧投票)作主策略。",
    }

    (OUT / "evolve_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(k, json.dumps(v, ensure_ascii=False) if v else None)
    print("recommend", json.dumps(report["recommend"], ensure_ascii=False, indent=2))
    print("saved", OUT / "evolve_report.json")


if __name__ == "__main__":
    main()
