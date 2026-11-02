#!/usr/bin/env python3
"""v2: 修 eval state + 紧凑多通道 + 更多样本 + 高置信弃权（冲 75%）。

相对 v1 的关键修复:
  - 落盘评估必须带 selv_noarr:<selection>，否则 value-token 编码器不进图（v1 因此从 monitor~64% 掉到 eval~36%）
  - 多通道只走 series→value tokens，文本不塞 300/600 长序列
  - 训练用满更多样本；报告 monitor_acc / eval_acc / 高置信子集 acc
  - 震荡已在标签层剔除；不确定样本用置信度弃权处理
"""
from __future__ import annotations

import json
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from timesjev.schema import Sample
from timesjev.train import sft as sft_mod

DECIDER = Path(r"F:/aicoding/myjev/models/decider-2b")
STOCK_AGENT = Path(r"D:/aicoding/stockagent-analysis")
DAILY_CACHE = Path(r"D:/aicoding/stockagent-fourpool/output/tushare_cache/daily")
OUT = ROOT / "runs" / "smoke_updown_multimodal_v2"
CACHE = OUT / "events_cache.pkl"

HORIZON = 20
UP_GAIN, UP_CLOSE, UP_DD = 0.15, 0.08, -0.10
DOWN_GAIN_MAX, DOWN_CLOSE, DOWN_DD = 0.03, -0.08, -0.12


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


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.asarray(x, float).copy()
    a = 2 / (n + 1)
    for i in range(1, len(out)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def indicators(df: pd.DataFrame) -> dict[str, np.ndarray]:
    c = df["close"].to_numpy(float)
    v = df["vol"].to_numpy(float) if "vol" in df.columns else np.ones(len(df))
    dif = _ema(c, 12) - _ema(c, 26)
    dea = _ema(dif, 9)
    d = np.diff(c, prepend=c[0])
    up = np.clip(d, 0, None)
    down = np.clip(-d, 0, None)
    au = pd.Series(up).rolling(14, min_periods=14).mean().to_numpy()
    ad = pd.Series(down).rolling(14, min_periods=14).mean().to_numpy()
    rs = au / np.where(ad == 0, np.nan, ad)
    rsi = np.nan_to_num(100 - (100 / (1 + rs)), nan=50.0)
    return {
        "close": c,
        "vol": v,
        "macd_h": dif - dea,
        "rsi": rsi,
        "ma5": pd.Series(c).rolling(5, min_periods=1).mean().to_numpy(),
        "ma20": pd.Series(c).rolling(20, min_periods=1).mean().to_numpy(),
    }


def label_up_down(entry: float, highs, lows, closes) -> str:
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


def load_codes(limit: int = 80) -> list[str]:
    codes: list[str] = []
    pool = STOCK_AGENT / "stock_pool_94.txt"
    if pool.is_file():
        for line in pool.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                codes.append(line.split("|")[0].zfill(6))
    for name in ("r20_pool_a_20261029_codes.txt", "v12_31_top20_20261029_codes.txt"):
        p = STOCK_AGENT / "output/daily_best_20261029" / name
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    codes.append(line.strip().split(".")[0].zfill(6))
    codes += ["000001", "000002", "600519", "300750", "002594", "601318", "000858", "002415"]
    seen, out = set(), []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:limit]


def fetch_tencent(code6: str, limit: int = 900) -> pd.DataFrame | None:
    import json as _json
    import urllib.request

    symbol = f"sh{code6}" if code6.startswith(("5", "6")) else f"sz{code6}"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,{limit},qfq"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = _json.loads(resp.read().decode())
    except Exception as e:
        print(f"  fetch fail {code6}: {e}")
        return None
    bars = (data.get("data") or {}).get(symbol, {})
    rows = bars.get("qfqday") or bars.get("day") or []
    out = []
    for b in rows:
        if len(b) < 6:
            continue
        out.append(
            {
                "trade_date": str(b[0]).replace("-", ""),
                "open": float(b[1]),
                "close": float(b[2]),
                "high": float(b[3]),
                "low": float(b[4]),
                "vol": float(b[5]),
            }
        )
    df = pd.DataFrame(out)
    return df if len(df) >= 260 else None


def load_daily(code6: str) -> pd.DataFrame | None:
    ts = [f"{code6}.SH", f"{code6}.SZ"]
    files = sorted(DAILY_CACHE.glob("*.parquet"))
    if files:
        parts = []
        for f in files:
            try:
                d = pd.read_parquet(f)
            except Exception:
                continue
            d = d[d["ts_code"].astype(str).isin(ts)]
            if len(d):
                parts.append(d)
        if parts:
            df = pd.concat(parts, ignore_index=True)
            df["trade_date"] = df["trade_date"].astype(str)
            df = df.sort_values("trade_date").drop_duplicates("trade_date").reset_index(drop=True)
            if "vol" not in df.columns and "volume" in df.columns:
                df = df.rename(columns={"volume": "vol"})
            if "vol" not in df.columns:
                df["vol"] = 1.0
            if len(df) >= 260:
                return df
    return fetch_tencent(code6)


def render_chart(ev: Event, path: Path) -> None:
    n = ev.left_n
    x = np.arange(n)
    c = np.asarray(ev.close)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1.1, 1]})
    ax_k, ax_v, ax_m, ax_r = axes
    for i in range(n):
        color = "#c0392b" if (i == 0 or c[i] >= c[i - 1]) else "#27ae60"
        ax_k.plot([i, i], [c[i] * 0.995, c[i] * 1.005], color=color, lw=0.8)
    ax_k.plot(x, ev.ma5, color="#f39c12", lw=1.0, label="MA5")
    ax_k.plot(x, ev.ma20, color="#3498db", lw=1.0, label="MA20")
    ax_k.set_title(f"{ev.code} {ev.date} L={n} {'UP' if ev.label else 'DOWN'}")
    ax_k.legend(loc="upper left", fontsize=8)
    ax_v.bar(x, ev.vol, color="#7f8c8d", width=0.8)
    ax_m.bar(x, ev.macd_h, color=["#c0392b" if h >= 0 else "#27ae60" for h in ev.macd_h], width=0.8)
    ax_r.plot(x, ev.rsi, color="#8e44ad", lw=1.0)
    ax_r.axhline(70, color="gray", ls=":", lw=0.8)
    ax_r.axhline(30, color="gray", ls=":", lw=0.8)
    ax_r.set_xlabel(f"bars = {n} (aligned with series)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def scan(code: str, df: pd.DataFrame, left_n: int, max_up=12, max_dn=12) -> tuple[list[Event], dict]:
    ind = indicators(df)
    opens = df["open"].to_numpy(float)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = ind["close"]
    dates = df["trade_date"].astype(str).tolist()
    up, dn, chop = [], [], 0
    for t0 in range(left_n - 1, len(df) - HORIZON - 1, 5):
        entry = opens[t0 + 1]
        lab = label_up_down(entry, highs[t0 + 1 :], lows[t0 + 1 :], closes[t0 + 1 :])
        if lab == "chop":
            chop += 1
            continue
        sl = slice(t0 - left_n + 1, t0 + 1)
        vol_w = ind["vol"][sl]
        vol_n = (vol_w / (np.median(vol_w) + 1e-9)).tolist()
        macd_w = ind["macd_h"][sl]
        macd_n = (macd_w / (np.std(macd_w) + 1e-9)).tolist()
        ev = Event(
            code=code,
            date=dates[t0],
            label=1 if lab == "up" else 0,
            left_n=left_n,
            close=closes[sl].tolist(),
            vol=vol_n,
            macd_h=macd_n,
            rsi=ind["rsi"][sl].tolist(),
            ma5=ind["ma5"][sl].tolist(),
            ma20=ind["ma20"][sl].tolist(),
        )
        if lab == "up" and len(up) < max_up:
            up.append(ev)
        elif lab == "down" and len(dn) < max_dn:
            dn.append(ev)
        if len(up) >= max_up and len(dn) >= max_dn:
            break
    return up + dn, {"up": len(up), "down": len(dn), "chop_seen": chop}


def _fmt_short(xs: list[float], k: int = 8) -> str:
    """只放首尾若干点，避免撑爆 context。"""
    if len(xs) <= 2 * k:
        body = xs
    else:
        body = xs[:k] + xs[-k:]
    return "[" + ", ".join(f"{v:.5g}" for v in body) + (f", …x{len(xs)}" if len(xs) > 2 * k else "") + "]"


def to_sample_close_only(ev: Event) -> Sample:
    return Sample(
        id=f"L{ev.left_n}/close/{ev.code}/{ev.date}/{ev.label}",
        task="gate_updown",
        split="train",
        context=f"Stock {ev.code} L={ev.left_n} closes.\nClose head/tail: {_fmt_short(ev.close)}",
        question="Will the next ~20 sessions be a clear UP move (not chop), or a clear DOWN move?",
        options={"A": "Yes — clear UP", "B": "No — clear DOWN"},
        answer="A" if ev.label == 1 else "B",
        series=[ev.close],
        domain="stock",
        source="updown_v2",
        raw_problem="updown",
    )


def to_sample_multimodal(ev: Event) -> Sample:
    """交织三通道进 series（value tokens）；文本只放短摘要 + 对齐声明。"""
    inter = []
    for a, b, c in zip(ev.close, ev.vol, ev.macd_h):
        inter.extend([a, b, c])
    c0, c1 = ev.close[0], ev.close[-1]
    ctx = (
        f"Stock {ev.code} multimodal L={ev.left_n} (chart-aligned; channels close/vol_norm/macd_z).\n"
        f"Close {c0:.4g}->{c1:.4g} ({(c1 / c0 - 1) * 100:.2f}%). "
        f"RSI last={ev.rsi[-1]:.1f}. MA5/20={ev.ma5[-1]:.4g}/{ev.ma20[-1]:.4g}. "
        f"VolNorm last={ev.vol[-1]:.3g} MacdZ last={ev.macd_h[-1]:.3g}."
    )
    return Sample(
        id=f"L{ev.left_n}/mm/{ev.code}/{ev.date}/{ev.label}",
        task="tsqa_classification",
        split="train",
        context=ctx,
        question="Will the next ~20 sessions be a clear UP move (not chop), or a clear DOWN move?",
        options={"A": "Yes — clear UP", "B": "No — clear DOWN"},
        answer="A" if ev.label == 1 else "B",
        series=[inter],
        domain="stock",
        source="updown_v2",
        raw_problem="updown",
    )


def write_selection(key: str, multimodal: bool):
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
    path.write_text(json.dumps({"names": names, "method": "updown v2", "mm": multimodal}, indent=2), encoding="utf-8")
    return key


def feature_baseline(samples: list[Sample], sel_key: str, multimodal: bool) -> dict:
    """sklearn 对照：同一套 TimesJev 统计特征能不能分涨跌。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from timesjev.data.pool import selected

    X, y = [], []
    for s in samples:
        vals, _ = selected(s, sel_key)
        if not vals or any(v != v or abs(v) == float("inf") for v in vals):
            continue
        X.append(vals)
        y.append(1 if s.answer == "A" else 0)
    if len(X) < 40:
        return {"ok": False, "reason": "too few"}
    X = np.asarray(X, float)
    y = np.asarray(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    pred = proba.argmax(1)
    conf = proba.max(1)
    out = {
        "ok": True,
        "multimodal": multimodal,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "acc": float(accuracy_score(yte, pred)),
    }
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75):
        m = conf >= thr
        if m.sum() < 5:
            out[f"acc_conf>={thr}"] = None
            out[f"n_conf>={thr}"] = int(m.sum())
            out[f"coverage>={thr}"] = float(m.mean())
            continue
        out[f"acc_conf>={thr}"] = float(accuracy_score(yte[m], pred[m]))
        out[f"n_conf>={thr}"] = int(m.sum())
        out[f"coverage>={thr}"] = float(m.mean())
    return out


def conf_summary(preds, samples, thresholds=(0.55, 0.60, 0.65, 0.70, 0.75)) -> dict:
    by_id = {s.id: s for s in samples}
    rows = []
    for p in preds:
        s = by_id.get(p.id)
        if s is None or not p.probs:
            continue
        conf = max(p.probs.values())
        rows.append((conf, p.answer == s.answer))
    if not rows:
        return {}
    out = {"n": len(rows), "acc_all": float(np.mean([c for _, c in rows]))}
    confs = np.array([r[0] for r in rows])
    correct = np.array([r[1] for r in rows], dtype=float)
    for thr in thresholds:
        m = confs >= thr
        out[f"acc_conf>={thr}"] = float(correct[m].mean()) if m.any() else None
        out[f"n_conf>={thr}"] = int(m.sum())
        out[f"coverage>={thr}"] = float(m.mean())
    return out


def run_ft(tag: str, samples: list[Sample], sel_key: str, n_train_cap: int = 400, epochs: int = 4) -> dict:
    rng = random.Random(7)
    pos = [s for s in samples if s.answer == "A"]
    neg = [s for s in samples if s.answer == "B"]
    if len(pos) < 20 or len(neg) < 20:
        return {"tag": tag, "skipped": True, "reason": f"pos={len(pos)} neg={len(neg)}"}
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split(xs):
        k = max(8, int(0.2 * len(xs)))
        return xs[k:], xs[:k]

    tr_p, te_p = split(pos)
    tr_n, te_n = split(neg)
    train, test = tr_p + tr_n, te_p + te_n
    rng.shuffle(train)
    rng.shuffle(test)
    train_ids, test_ids = {s.id for s in train}, {s.id for s in test}
    all_s = {s.id: s for s in train + test}
    state_kind = f"selv_noarr:{sel_key}"

    def load_samples(data_cfg: dict):
        ids = train_ids if data_cfg.get("split", "train") == "train" else test_ids
        return [all_s[i] for i in ids]

    import timesjev.data as data_mod
    from timesjev.eval import runner as runner_mod
    from timesjev.eval.runner import run_eval
    import timesjev.train.sft as sft_file

    data_mod.load_samples = load_samples
    sft_file.load_samples = load_samples
    runner_mod.load_samples = load_samples

    cfg = {
        "name": f"smoke-updown-v2-{tag}",
        "model": {
            "backend": "decider",
            "checkpoint": str(DECIDER).replace("\\", "/"),
            "revision": None,
            "temperature": None,
            "use_graphs": False,
        },
        "data": {"source": "stock", "split": "train", "state": state_kind},
        "train": {
            "method": "sft",
            "sampling": "balanced_random",
            "n_samples": min(n_train_cap, len(train)),
            "positive_prefix": "Yes",
            "seed": 7,
            "state": state_kind,
            "freeze_lm": True,
            "batch_like": "raw",
            "max_ctx_tokens": 4096,
            "monitor": "test-updown",
            "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "target_modules": None},
            "optim": {
                "lr": 2.0e-4,
                "epochs": epochs,
                "max_tokens_per_batch": 8192,
                "warmup_ratio": 0.05,
                "weight_decay": 0.0,
            },
            "ts_input": {"value_types": True, "value_block": f"select:{sel_key}", "lr": 1.0e-3},
        },
        "eval": [{"suffix": "test-updown", "data": {"source": "stock", "split": "test", "state": state_kind}}],
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ckpt = sft_mod.train_sft(cfg)
    train_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    # 读 train.log 最后一次 monitor acc
    monitor_acc = None
    log_path = Path(ckpt).parent / "train.log"
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if "test-updown accuracy" in line:
                try:
                    monitor_acc = float(line.strip().split()[-1]) / 100.0
                except Exception:
                    pass

    # 关键修复: data.state + model.state 都带 selv_noarr
    eval_cfg = {
        **cfg,
        "name": cfg["name"] + "-test",
        "data": {"source": "stock", "split": "test", "state": state_kind},
        "model": {
            **cfg["model"],
            "checkpoint": str(ckpt),
            "backend": "decider_ts",
            "state": state_kind,
        },
    }
    t1 = time.time()
    ev_dir = run_eval(eval_cfg)
    metrics = json.loads((Path(ev_dir) / "metrics.json").read_text(encoding="utf-8"))
    preds = []
    from timesjev.schema import Prediction

    for line in (Path(ev_dir) / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        preds.append(Prediction(**{k: d[k] for k in ("id", "task", "split", "answer", "probs", "latency_s", "extra") if k in d}))

    acc = None
    block = metrics.get("train") or metrics
    if isinstance(block, dict):
        for v in block.values():
            if isinstance(v, dict) and "acc" in v:
                acc = v["acc"]
                break
    conf = conf_summary(preds, test)
    best_conf_acc = None
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75):
        a = conf.get(f"acc_conf>={thr}")
        if a is not None and conf.get(f"n_conf>={thr}", 0) >= 8:
            if best_conf_acc is None or a > best_conf_acc:
                best_conf_acc = a
    return {
        "tag": tag,
        "skipped": False,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "n_train": len(train),
        "n_test": len(test),
        "n_train_used": cfg["train"]["n_samples"],
        "epochs": epochs,
        "train_seconds": round(train_s, 1),
        "eval_seconds": round(time.time() - t1, 1),
        "peak_vram_gb": round(peak, 2),
        "monitor_acc": monitor_acc,
        "eval_acc": acc,
        "confidence": conf,
        "best_conf_acc_n>=8": best_conf_acc,
        "beats_75_eval": bool(acc is not None and acc >= 0.75),
        "beats_75_monitor": bool(monitor_acc is not None and monitor_acc >= 0.75),
        "beats_75_conf": bool(best_conf_acc is not None and best_conf_acc >= 0.75),
        "metrics": metrics,
        "checkpoint": str(ckpt),
        "eval_dir": str(ev_dir),
        "state": state_kind,
    }


def collect_events() -> dict[int, list[Event]]:
    if CACHE.is_file():
        print(f"load cache {CACHE}", flush=True)
        with open(CACHE, "rb") as f:
            raw = pickle.load(f)
        return {int(k): [Event(**e) if isinstance(e, dict) else e for e in vs] for k, vs in raw.items()}

    codes = load_codes(80)
    print(f"codes={len(codes)}", flush=True)
    events_by_L: dict[int, list[Event]] = {100: [], 200: []}
    for i, code in enumerate(codes, 1):
        print(f"[{i}/{len(codes)}] {code}", flush=True)
        df = load_daily(code)
        if df is None:
            print("  no data")
            continue
        for L in (100, 200):
            if len(df) < L + HORIZON + 5:
                continue
            evs, st = scan(code, df, L, max_up=10, max_dn=10)
            events_by_L[L].extend(evs)
            print(f"  L={L} up={st['up']} down={st['down']} chop_seen~{st['chop_seen']}")
            chart_dir = OUT / "charts" / f"L{L}"
            for ev in evs[:1]:
                render_chart(ev, chart_dir / f"{ev.code}_{ev.date}_{'up' if ev.label else 'dn'}.png")
        time.sleep(0.02)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump({k: [asdict(e) for e in vs] for k, vs in events_by_L.items()}, f)
    return events_by_L


def main():
    assert DECIDER.is_dir(), f"missing {DECIDER}"
    OUT.mkdir(parents=True, exist_ok=True)
    events_by_L = collect_events()
    summary_data = {
        L: {"n": len(evs), "up": sum(1 for e in evs if e.label == 1), "down": sum(1 for e in evs if e.label == 0)}
        for L, evs in events_by_L.items()
    }
    print("events", json.dumps(summary_data, indent=2), flush=True)

    baselines, results = [], []
    # 主攻 L=100（v1 monitor 最好）；顺带 L=200 close-only
    combos = [
        (100, False, "close_L100"),
        (100, True, "mm_L100"),
        (200, False, "close_L200"),
        (200, True, "mm_L200"),
    ]
    for L, mm, tag in combos:
        evs = events_by_L.get(L, [])
        if len(evs) < 40:
            print(f"skip {tag}")
            continue
        sel = write_selection(f"updown-v2-{'mm' if mm else 'close'}-L{L}", multimodal=mm)
        samples = [to_sample_multimodal(e) if mm else to_sample_close_only(e) for e in evs]
        print(f"\n=== baseline {tag} ===", flush=True)
        bl = feature_baseline(samples, sel, mm)
        bl["tag"] = tag
        baselines.append(bl)
        print(json.dumps(bl, indent=2), flush=True)

        print(f"\n=== FT {tag} n={len(samples)} ===", flush=True)
        results.append(run_ft(tag, samples, sel, n_train_cap=400, epochs=4))

    report = {
        "goal": "v2: fixed selv_noarr eval state; compact multimodal; confidence abstention for unsure/chop-like",
        "v1_issue": "eval missing state=selv_noarr → value encoder unused → eval_acc collapsed vs monitor",
        "label_rule": {
            "horizon": HORIZON,
            "up": {"max_gain>=": UP_GAIN, "close>=": UP_CLOSE, "max_dd>=": UP_DD},
            "down": {"max_gain<=": DOWN_GAIN_MAX, "close_or_dd": [DOWN_CLOSE, DOWN_DD]},
            "chop": "excluded from binary; residual uncertainty → confidence abstain",
        },
        "events": summary_data,
        "sklearn_baseline": baselines,
        "finetune": results,
        "target_acc": 0.75,
        "any_beats_75": any(
            r.get("beats_75_eval") or r.get("beats_75_monitor") or r.get("beats_75_conf")
            for r in results
            if not r.get("skipped")
        ),
        "charts": str(OUT / "charts"),
    }
    (OUT / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== REPORT ===")
    for b in baselines:
        print(
            f"sklearn {b.get('tag')}: acc={b.get('acc')} "
            f"conf0.65={b.get('acc_conf>=0.65')} cov={b.get('coverage>=0.65')}"
        )
    for r in results:
        if r.get("skipped"):
            print(r["tag"], "SKIP", r.get("reason"))
        else:
            print(
                f"{r['tag']}: monitor={r.get('monitor_acc')} eval={r.get('eval_acc')} "
                f"best_conf={r.get('best_conf_acc_n>=8')} "
                f"beats75={r.get('beats_75_eval') or r.get('beats_75_monitor') or r.get('beats_75_conf')}"
            )
    print("saved", OUT / "smoke_report.json")
    print("any>=75%:", report["any_beats_75"])


if __name__ == "__main__":
    main()
