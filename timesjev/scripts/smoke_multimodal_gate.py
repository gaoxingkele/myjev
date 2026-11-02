#!/usr/bin/env python3
"""多模态门控冒烟：按池子分层正样本 + 多通道K线出图 + Decider-2B 少样本微调。

背景:
  - 托管 Jev 只看左图 ≈ 随机；右图识别/左右一致性有信号
  - TimesJev 数值特征 + 冻结 Decider 在 3090 可训
  - R20 / V12.31 正样本定义不同，必须分层标签

本冒烟验证:
  1) 能否按池规则筛出 pos/neg 并画出时间对齐多通道图（主图+量+MACD+RSI，左|右）
  2) 仅用左窗数值特征微调 value-token encoder，看判决准确率是否高于机会水平
  3) 左窗特征近邻的历史案例，未来收益是否系统性更好（相似→上涨假设）

用法:
  D:/aicoding/TimesJev/.venv/Scripts/python.exe scripts/smoke_multimodal_gate.py
"""
from __future__ import annotations

import json
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
import yaml

from timesjev.schema import Sample
from timesjev.train import sft as sft_mod

DECIDER = Path(r"F:/aicoding/myjev/models/decider-2b")
STOCK_AGENT = Path(r"D:/aicoding/stockagent-analysis")
DAILY_CACHE = Path(r"D:/aicoding/stockagent-fourpool/output/tushare_cache/daily")
OUT = ROOT / "runs" / "smoke_multimodal_gate"
SEL_KEY = "gate-stg-16"
LEFT_N, RIGHT_N = 150, 40


# ───────── 池子分层标签（正样本定义不同） ─────────
@dataclass
class LabelRule:
    pool: str
    horizon: int
    # R20: 20 日绝对目标；V12.31: 5 日启动质量
    mode: str  # "r20_target" | "v12_pump"


RULES = {
    "R20": LabelRule("R20", 20, "r20_target"),
    "V12.31": LabelRule("V12.31", 5, "v12_pump"),
}


def is_positive(rule: LabelRule, entry: float, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, past_r5: float) -> bool:
    """entry=次日开盘；highs/lows/closes 为未来 horizon 根（含次日）。"""
    if entry <= 0 or len(closes) < rule.horizon:
        return False
    h = highs[: rule.horizon]
    l = lows[: rule.horizon]
    c = closes[: rule.horizon]
    max_gain = float(h.max() / entry - 1)
    max_dd = float(l.min() / entry - 1)
    r_close = float(c[-1] / entry - 1)
    if rule.mode == "r20_target":
        # 与池A契约一致：涨幅路径 OR 收盘路径，且回撤可接受
        return (r_close >= 0.25 or max_gain >= 0.25) and max_dd >= -0.15
    if rule.mode == "v12_pump":
        # 对齐 pump v3c：5 日 max_gain≥10% 且 max_dd≥-5%，且过去5日未大涨
        return max_gain >= 0.10 and max_dd >= -0.05 and past_r5 <= 0.08
    return False


def is_negative(rule: LabelRule, entry: float, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, past_r5: float) -> bool:
    if entry <= 0 or len(closes) < rule.horizon:
        return False
    h = highs[: rule.horizon]
    l = lows[: rule.horizon]
    c = closes[: rule.horizon]
    max_gain = float(h.max() / entry - 1)
    max_dd = float(l.min() / entry - 1)
    r_close = float(c[-1] / entry - 1)
    if rule.mode == "r20_target":
        # 明确失败：涨不动且回撤深，或收盘大负
        return max_gain < 0.08 and (max_dd < -0.12 or r_close < -0.05)
    if rule.mode == "v12_pump":
        # 启动失败：几乎无上冲或跌破
        return max_gain < 0.04 or max_dd < -0.08
    return False


# ───────── 数据 ─────────
def load_codes() -> list[str]:
    codes: list[str] = []
    for name in ("r20_pool_a_20261029_codes.txt", "v12_31_top20_20261029_codes.txt", "s20_hybrid_20261029_codes.txt"):
        p = STOCK_AGENT / "output" / "daily_best_20261029" / name
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            c = line.strip()
            if not c:
                continue
            codes.append(c.split(".")[0].zfill(6))
    # 再补优质起涨图里出现过的代码，丰富历史正例
    qdir = STOCK_AGENT / "output" / "quality_uptrend_charts"
    if qdir.is_dir():
        for d in qdir.iterdir():
            if d.is_dir() and not d.name.startswith("_") and "_" in d.name:
                codes.append(d.name.split("_")[0].zfill(6))
    # unique preserve order
    seen, out = set(), []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:40] or ["000155", "000546", "000792", "300606", "688047"]


def load_daily_from_cache(code6: str) -> pd.DataFrame | None:
    """从 fourpool 日线缓存拼单股；不够则返回 None。"""
    ts_candidates = [f"{code6}.SH", f"{code6}.SZ", f"{code6}.BJ"]
    files = sorted(DAILY_CACHE.glob("*.parquet"))
    if not files:
        return None
    parts = []
    for f in files:
        try:
            d = pd.read_parquet(f)
        except Exception:
            continue
        d = d[d["ts_code"].astype(str).isin(ts_candidates)]
        if len(d):
            parts.append(d)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date").drop_duplicates("trade_date").reset_index(drop=True)
    return df if len(df) >= LEFT_N + RIGHT_N + 5 else None


def fetch_daily_tencent(code6: str, limit: int = 800) -> pd.DataFrame | None:
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
                "ts_code": f"{code6}.SH" if code6.startswith(("5", "6")) else f"{code6}.SZ",
            }
        )
    df = pd.DataFrame(out)
    return df if len(df) >= LEFT_N + RIGHT_N + 5 else None


def get_daily(code6: str) -> pd.DataFrame | None:
    df = load_daily_from_cache(code6)
    if df is not None and len(df) >= LEFT_N + 60:
        return df
    return fetch_daily_tencent(code6)


# ───────── 多通道出图（时间轴对齐：左历史 | 右未来） ─────────
def _ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.copy(x).astype(float)
    alpha = 2 / (n + 1)
    for i in range(1, len(out)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _macd(close: np.ndarray):
    ema12, ema26 = _ema(close, 12), _ema(close, 26)
    dif = ema12 - ema26
    dea = _ema(dif, 9)
    hist = dif - dea
    return dif, dea, hist


def _rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = np.clip(d, 0, None)
    down = np.clip(-d, 0, None)
    # wilder-ish SMA
    au = pd.Series(up).rolling(n, min_periods=n).mean().to_numpy()
    ad = pd.Series(down).rolling(n, min_periods=n).mean().to_numpy()
    rs = au / np.where(ad == 0, np.nan, ad)
    return 100 - (100 / (1 + rs))


def render_multichannel(df: pd.DataFrame, t0: int, out_path: Path, title: str) -> None:
    """t0=信号日索引；左 [t0-LEFT_N+1, t0]，右 (t0, t0+RIGHT_N]。"""
    left = df.iloc[t0 - LEFT_N + 1 : t0 + 1].copy()
    right = df.iloc[t0 + 1 : t0 + 1 + RIGHT_N].copy()
    both = pd.concat([left, right], ignore_index=True)
    nL, nR = len(left), len(right)
    close = both["close"].to_numpy(float)
    high = both["high"].to_numpy(float)
    low = both["low"].to_numpy(float)
    open_ = both["open"].to_numpy(float)
    vol = both["vol"].to_numpy(float) if "vol" in both.columns else np.ones(len(both))
    dif, dea, hist = _macd(close)
    rsi = _rsi(close)

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.0, 1.2, 1.0]}
    )
    ax_k, ax_v, ax_m, ax_r = axes
    x = np.arange(len(both))
    # candles
    for i in range(len(both)):
        color = "#c0392b" if close[i] >= open_[i] else "#27ae60"
        ax_k.vlines(i, low[i], high[i], color=color, linewidth=0.8)
        ax_k.add_patch(
            plt.Rectangle(
                (i - 0.3, min(open_[i], close[i])),
                0.6,
                max(abs(close[i] - open_[i]), 1e-6),
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
            )
        )
    for w, c in ((5, "#f39c12"), (20, "#3498db"), (60, "#8e44ad")):
        if len(close) >= w:
            ma = pd.Series(close).rolling(w).mean().to_numpy()
            ax_k.plot(x, ma, color=c, linewidth=1.0, label=f"MA{w}")
    ax_k.axvline(nL - 0.5, color="black", linestyle="--", linewidth=1.2)
    ax_k.set_ylabel("Price")
    ax_k.set_title(title)
    ax_k.legend(loc="upper left", fontsize=8, ncol=3)
    ax_k.text(nL / 2, ax_k.get_ylim()[1], "LEFT past", ha="center", va="top", fontsize=9, alpha=0.7)
    if nR:
        ax_k.text(nL + nR / 2, ax_k.get_ylim()[1], "RIGHT future", ha="center", va="top", fontsize=9, alpha=0.7)

    colors_v = ["#c0392b" if close[i] >= open_[i] else "#27ae60" for i in range(len(both))]
    ax_v.bar(x, vol, color=colors_v, width=0.8, alpha=0.8)
    ax_v.axvline(nL - 0.5, color="black", linestyle="--", linewidth=1.0)
    ax_v.set_ylabel("Vol")

    ax_m.plot(x, dif, color="#2980b9", linewidth=1.0, label="DIF")
    ax_m.plot(x, dea, color="#e67e22", linewidth=1.0, label="DEA")
    ax_m.bar(x, hist, color=["#c0392b" if h >= 0 else "#27ae60" for h in hist], width=0.8, alpha=0.7)
    ax_m.axvline(nL - 0.5, color="black", linestyle="--", linewidth=1.0)
    ax_m.set_ylabel("MACD")
    ax_m.legend(loc="upper left", fontsize=7)

    ax_r.plot(x, rsi, color="#8e44ad", linewidth=1.0)
    ax_r.axhline(70, color="gray", linestyle=":", linewidth=0.8)
    ax_r.axhline(30, color="gray", linestyle=":", linewidth=0.8)
    ax_r.axvline(nL - 0.5, color="black", linestyle="--", linewidth=1.0)
    ax_r.set_ylabel("RSI14")
    ax_r.set_ylim(0, 100)
    ax_r.set_xlabel("bars (time-aligned; cut = signal day)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ───────── 样本构建 ─────────
@dataclass
class GateEvent:
    pool: str
    code: str
    date: str
    label: int  # 1 pos / 0 neg
    t0: int
    closes_left: list[float]
    chart: str


def scan_events(code6: str, df: pd.DataFrame, rule: LabelRule, max_pos=8, max_neg=8) -> list[GateEvent]:
    closes = df["close"].to_numpy(float)
    opens = df["open"].to_numpy(float)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    dates = df["trade_date"].astype(str).tolist()
    pos, neg = [], []
    # 步长放大，减少时间相关泄漏
    for t0 in range(LEFT_N - 1, len(df) - rule.horizon - 1, 7):
        entry = opens[t0 + 1]
        fut_h = highs[t0 + 1 : t0 + 1 + rule.horizon]
        fut_l = lows[t0 + 1 : t0 + 1 + rule.horizon]
        fut_c = closes[t0 + 1 : t0 + 1 + rule.horizon]
        past_r5 = float(closes[t0] / closes[t0 - 5] - 1) if t0 >= 5 else 0.0
        left_closes = closes[t0 - LEFT_N + 1 : t0 + 1].tolist()
        if is_positive(rule, entry, fut_h, fut_l, fut_c, past_r5) and len(pos) < max_pos:
            pos.append((t0, dates[t0], left_closes, 1))
        elif is_negative(rule, entry, fut_h, fut_l, fut_c, past_r5) and len(neg) < max_neg:
            neg.append((t0, dates[t0], left_closes, 0))
        if len(pos) >= max_pos and len(neg) >= max_neg:
            break
    events = []
    chart_dir = OUT / "charts" / rule.pool.replace(".", "_")
    for t0, date, left_closes, lab in pos + neg:
        tag = "pos" if lab else "neg"
        path = chart_dir / f"{code6}_{date}_{tag}.png"
        if not path.exists():
            render_multichannel(df, t0, path, f"{rule.pool} {code6} {date} {'POS' if lab else 'NEG'}")
        events.append(
            GateEvent(rule.pool, code6, date, lab, t0, left_closes, str(path))
        )
    return events


def event_to_sample(ev: GateEvent) -> Sample:
    arr = "[" + ", ".join(f"{v:.4f}" for v in ev.closes_left) + "]"
    if ev.pool == "R20":
        q = "Given only this past close series at a candidate entry, will R20 absolute-target succeed (≈+25% in 20d with acceptable drawdown)?"
        options = {
            "A": "Yes, likely meets R20 target path",
            "B": "No, unlikely to meet R20 target",
        }
    else:
        q = "Given only this past close series, will a V12.31-style 5-day quality pump follow (gain≥10% without deep DD)?"
        options = {
            "A": "Yes, likely a quality short-horizon pump",
            "B": "No, unlikely a quality pump",
        }
    return Sample(
        id=f"{ev.pool}/{ev.code}/{ev.date}/{ev.label}",
        task=f"gate_{ev.pool.replace('.', '_')}",
        split="train",
        context=f"Pool={ev.pool} Stock={ev.code} signal_date={ev.date}\nClose series (left window only): {arr}",
        question=q,
        options=options,
        answer="A" if ev.label == 1 else "B",
        series=[ev.closes_left],
        domain="stock",
        source="gate_smoke",
        raw_problem=q,
    )


def write_selection():
    names = [
        "std", "range", "IQR", "mean abs step", "std abs step", "max abs step",
        "acf lag1", "acf lag2", "slope", "argmax position", "zero-crossing rate",
        "dominant period", "periodicity", "high_fluctuation", "stats3 median", "c24 SD",
    ]
    path = ROOT / "configs" / "selections" / f"{SEL_KEY}.json"
    path.write_text(
        json.dumps({"names": names, "method": "gate smoke", "dataset": "multimodal_gate"}, indent=2),
        encoding="utf-8",
    )


def similarity_probe(events: list[GateEvent]) -> dict:
    """左窗收益率路径的近邻：相似历史是否未来更好？用未来窗口收益代理。"""
    # 需要重新加载日线太重；这里用左窗末端相对收益结构作指纹，未来标签作 proxy
    from sklearn.neighbors import NearestNeighbors

    pos = [e for e in events if e.label == 1]
    neg = [e for e in events if e.label == 0]
    if len(pos) < 6 or len(neg) < 6:
        return {"ok": False, "reason": "too few"}

    def feat(e: GateEvent) -> np.ndarray:
        x = np.asarray(e.closes_left, dtype=float)
        r = np.diff(x) / np.clip(x[:-1], 1e-6, None)
        # 固定长度统计指纹
        return np.array(
            [
                r.mean(),
                r.std(),
                np.percentile(r, 10),
                np.percentile(r, 90),
                (x[-1] / x[0] - 1),
                (x[-1] / x[max(0, len(x) - 6)] - 1),
                (x[-1] / x[max(0, len(x) - 21)] - 1) if len(x) > 21 else 0.0,
            ],
            dtype=float,
        )

    X = np.stack([feat(e) for e in events])
    y = np.array([e.label for e in events])
    nn = NearestNeighbors(n_neighbors=min(6, len(events)), metric="euclidean").fit(X)
    # leave-one-out: 邻居里正例占比 vs 自身是否正例
    dists, idx = nn.kneighbors(X)
    scores = []
    for i, neigh in enumerate(idx):
        others = [j for j in neigh if j != i][:5]
        if not others:
            continue
        frac = float(y[others].mean())
        scores.append((frac, y[i]))
    # 高相似正例占比 → 自身为正的概率
    high = [lab for frac, lab in scores if frac >= 0.6]
    low = [lab for frac, lab in scores if frac <= 0.2]
    return {
        "ok": True,
        "n": len(scores),
        "pos_rate_when_neighbors_pos_ge_0.6": float(np.mean(high)) if high else None,
        "n_high": len(high),
        "pos_rate_when_neighbors_pos_le_0.2": float(np.mean(low)) if low else None,
        "n_low": len(low),
    }


def train_pool_smoke(pool: str, samples: list[Sample]) -> dict:
    write_selection()
    rng = random.Random(42 + hash(pool) % 1000)
    pos = [s for s in samples if s.answer == "A"]
    neg = [s for s in samples if s.answer == "B"]
    if len(pos) < 8 or len(neg) < 8:
        return {"pool": pool, "skipped": True, "reason": f"pos={len(pos)} neg={len(neg)}"}
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split(xs):
        k = max(3, int(0.25 * len(xs)))
        return xs[k:], xs[:k]

    tr_pos, te_pos = split(pos)
    tr_neg, te_neg = split(neg)
    train, test = tr_pos + tr_neg, te_pos + te_neg
    rng.shuffle(train)
    rng.shuffle(test)
    train_ids = {s.id for s in train}
    test_ids = {s.id for s in test}
    all_samples = {s.id: s for s in train + test}

    def load_samples(data_cfg: dict):
        ids = train_ids if data_cfg.get("split", "train") == "train" else test_ids
        return [all_samples[i] for i in ids]

    import timesjev.data as data_mod
    import timesjev.train.sft as sft_file
    from timesjev.eval import runner as runner_mod
    from timesjev.eval.runner import run_eval

    data_mod.load_samples = load_samples
    sft_file.load_samples = load_samples
    runner_mod.load_samples = load_samples

    cfg = {
        "name": f"smoke-gate-{pool.replace('.', '_')}",
        "model": {
            "backend": "decider",
            "checkpoint": str(DECIDER).replace("\\", "/"),
            "revision": None,
            "temperature": None,
            "use_graphs": False,
        },
        "data": {"source": "stock", "split": "train"},
        "train": {
            "method": "sft",
            "sampling": "balanced_random",
            "n_samples": min(48, len(train)),
            "positive_prefix": "Yes",
            "seed": 42,
            "state": f"selv_noarr:{SEL_KEY}",
            "freeze_lm": True,
            "batch_like": "raw",
            "max_ctx_tokens": 4096,
            "monitor": "test-gate",
            "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "target_modules": None},
            "optim": {
                "lr": 2.0e-4,
                "epochs": 2,
                "max_tokens_per_batch": 8192,
                "warmup_ratio": 0.05,
                "weight_decay": 0.0,
            },
            "ts_input": {
                "value_types": True,
                "value_block": f"select:{SEL_KEY}",
                "lr": 1.0e-3,
            },
        },
        "eval": [{"suffix": "test-gate", "data": {"source": "stock", "split": "test"}}],
    }
    t0 = time.time()
    ckpt = sft_mod.train_sft(cfg)
    train_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    eval_cfg = {
        **cfg,
        "name": cfg["name"] + "-test",
        "data": {"source": "stock", "split": "test"},
        "model": {**cfg["model"], "checkpoint": str(ckpt), "backend": "decider_ts"},
    }
    t1 = time.time()
    ev_dir = run_eval(eval_cfg)
    metrics = json.loads((Path(ev_dir) / "metrics.json").read_text(encoding="utf-8"))
    # pick first metric block
    acc = None
    for _, block in metrics.get("train", metrics).items() if isinstance(metrics.get("train"), dict) else [("x", metrics)]:
        if isinstance(block, dict) and "acc" in block:
            acc = block["acc"]
            break
    if acc is None and isinstance(metrics, dict):
        # flatten search
        for v in metrics.values():
            if isinstance(v, dict) and "acc" in v:
                acc = v["acc"]
                break
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, dict) and "acc" in vv:
                        acc = vv["acc"]
                        break
    return {
        "pool": pool,
        "skipped": False,
        "n_train": len(train),
        "n_test": len(test),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "train_seconds": round(train_s, 1),
        "eval_seconds": round(time.time() - t1, 1),
        "peak_vram_gb": round(peak, 2),
        "acc": acc,
        "metrics": metrics,
        "checkpoint": str(ckpt),
        "chance_baseline": 0.5,
        "beats_chance": (acc is not None and acc > 0.55),
    }


def main():
    assert DECIDER.is_dir(), f"missing {DECIDER}"
    OUT.mkdir(parents=True, exist_ok=True)
    codes = load_codes()
    print(f"codes={len(codes)}: {codes[:12]}...", flush=True)

    all_events: list[GateEvent] = []
    by_pool: dict[str, list[Sample]] = {k: [] for k in RULES}
    for i, code in enumerate(codes, 1):
        print(f"[{i}/{len(codes)}] {code}", flush=True)
        df = get_daily(code)
        if df is None:
            print("  skip no data")
            continue
        # normalize vol column
        if "vol" not in df.columns and "volume" in df.columns:
            df = df.rename(columns={"volume": "vol"})
        if "vol" not in df.columns:
            df["vol"] = 1.0
        for pool, rule in RULES.items():
            evs = scan_events(code, df, rule, max_pos=4, max_neg=4)
            all_events.extend(evs)
            for e in evs:
                by_pool[pool].append(event_to_sample(e))
            print(f"  {pool}: +{sum(1 for e in evs if e.label==1)} -{sum(1 for e in evs if e.label==0)} charts")
        time.sleep(0.05)

    # 分层统计
    taxonomy = {}
    for pool, samples in by_pool.items():
        taxonomy[pool] = {
            "n": len(samples),
            "pos": sum(1 for s in samples if s.answer == "A"),
            "neg": sum(1 for s in samples if s.answer == "B"),
            "rule": asdict(RULES[pool]),
        }
    print("taxonomy", json.dumps(taxonomy, ensure_ascii=False, indent=2), flush=True)

    sim = {}
    for pool in RULES:
        sim[pool] = similarity_probe([e for e in all_events if e.pool == pool])
    print("similarity_probe", json.dumps(sim, ensure_ascii=False, indent=2), flush=True)

    # 微调冒烟（分池）
    results = []
    for pool, samples in by_pool.items():
        print(f"\n=== finetune smoke {pool} n={len(samples)} ===", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        results.append(train_pool_smoke(pool, samples))

    report = {
        "goal": "multimodal gate smoke: pool-stratified positives + multichannel charts + Decider value-token FT",
        "note": "Images generated for later VLM/news gating; FT here uses left-window numeric features (TimesJev path) because local Decider/Kev are text System-1.",
        "prior_jev_left_only_acc": 0.429,
        "taxonomy": taxonomy,
        "similarity_probe": sim,
        "finetune": results,
        "charts_dir": str(OUT / "charts"),
    }
    (OUT / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved", OUT / "smoke_report.json")


if __name__ == "__main__":
    main()
