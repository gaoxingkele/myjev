#!/usr/bin/env python3
"""上涨/下跌高对比样本 + 与图等长对齐的多通道序列微调冒烟。

目标:
  - 只要「肯定涨 / 肯定跌」；震荡单独剔除（不进二分类训练）
  - 图 N 根 ↔ 序列 N 根（试 100 / 200）
  - 对比: 仅收盘 vs 收盘+量+MACD 交织三通道
  - 看准确率能否冲到 75%+

本地路径: 冻结 Decider-2B + TimesJev value tokens（图同步落盘，供后续 VLM）。
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
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
OUT = ROOT / "runs" / "smoke_updown_multimodal"

# 高对比标签（排除震荡）
HORIZON = 20
UP_GAIN, UP_CLOSE, UP_DD = 0.15, 0.08, -0.10
DOWN_GAIN_MAX, DOWN_CLOSE, DOWN_DD = 0.03, -0.08, -0.12


@dataclass
class Event:
    code: str
    date: str
    label: int  # 1 up / 0 down
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
    hist = dif - dea
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
        "macd_h": hist,
        "rsi": rsi,
        "ma5": pd.Series(c).rolling(5, min_periods=1).mean().to_numpy(),
        "ma20": pd.Series(c).rolling(20, min_periods=1).mean().to_numpy(),
    }


def label_up_down(entry: float, highs, lows, closes) -> str:
    """返回 'up' | 'down' | 'chop'。"""
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
    # extras
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
    """仅左窗 L 根，通道与序列对齐。"""
    n = ev.left_n
    x = np.arange(n)
    c = np.asarray(ev.close)
    o = c  # 近似（训练不依赖 open 视觉）
    # 用 close 画简化 K；量/MACD/RSI 与序列同长
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1.1, 1]})
    ax_k, ax_v, ax_m, ax_r = axes
    for i in range(n):
        color = "#c0392b" if (i == 0 or c[i] >= c[i - 1]) else "#27ae60"
        ax_k.plot([i, i], [c[i] * 0.995, c[i] * 1.005], color=color, lw=0.8)
    ax_k.plot(x, ev.ma5, color="#f39c12", lw=1.0, label="MA5")
    ax_k.plot(x, ev.ma20, color="#3498db", lw=1.0, label="MA20")
    ax_k.set_title(f"{ev.code} {ev.date} L={n} {'UP' if ev.label else 'DOWN'}")
    ax_k.legend(loc="upper left", fontsize=8)
    ax_k.set_ylabel("close+MA")
    ax_v.bar(x, ev.vol, color="#7f8c8d", width=0.8)
    ax_v.set_ylabel("vol")
    ax_m.bar(x, ev.macd_h, color=["#c0392b" if h >= 0 else "#27ae60" for h in ev.macd_h], width=0.8)
    ax_m.set_ylabel("MACD hist")
    ax_r.plot(x, ev.rsi, color="#8e44ad", lw=1.0)
    ax_r.axhline(70, color="gray", ls=":", lw=0.8)
    ax_r.axhline(30, color="gray", ls=":", lw=0.8)
    ax_r.set_ylabel("RSI14")
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
        # 通道归一：vol/macd 用窗口内 z 或相对尺度，避免量纲炸
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


def _fmt(xs: list[float]) -> str:
    return "[" + ", ".join(f"{v:.6g}" for v in xs) + "]"


def to_sample_close_only(ev: Event) -> Sample:
    return Sample(
        id=f"L{ev.left_n}/close/{ev.code}/{ev.date}/{ev.label}",
        task="gate_updown",
        split="train",
        context=f"Stock {ev.code} left window L={ev.left_n} closes only.\nClose: {_fmt(ev.close)}",
        question="Will the next ~20 sessions be a clear UP move (not chop), or a clear DOWN move?",
        options={"A": "Yes — clear UP", "B": "No — clear DOWN"},
        answer="A" if ev.label == 1 else "B",
        series=[ev.close],
        domain="stock",
        source="updown_smoke",
        raw_problem="updown",
    )


def to_sample_multimodal(ev: Event) -> Sample:
    """三通道交织: close, vol_norm, macd_norm → TimesJev X/Y/Z（task=tsqa_classification）。

    文本只放短摘要，避免 L=200 把 context 撑爆；数值统计走 value tokens（对齐图的 L 根）。
    """
    inter = []
    for a, b, c in zip(ev.close, ev.vol, ev.macd_h):
        inter.extend([a, b, c])
    c0, c1 = ev.close[0], ev.close[-1]
    ctx = (
        f"Stock {ev.code} multimodal left window L={ev.left_n} (aligned with chart).\n"
        f"Channels interleaved [close, vol_norm, macd_hist_z] length={len(inter)}.\n"
        f"Close path: {c0:.4g} -> {c1:.4g} ({(c1 / c0 - 1) * 100:.2f}%).\n"
        f"RSI14 last={ev.rsi[-1]:.1f} mid={ev.rsi[len(ev.rsi)//2]:.1f}.\n"
        f"MA5/MA20 last={ev.ma5[-1]:.4g}/{ev.ma20[-1]:.4g}.\n"
        f"VolNorm last={ev.vol[-1]:.3g} MacdZ last={ev.macd_h[-1]:.3g}.\n"
        f"SeriesXYZ: {_fmt(inter)}"
    )
    return Sample(
        id=f"L{ev.left_n}/mm/{ev.code}/{ev.date}/{ev.label}",
        task="tsqa_classification",  # enables X/Y/Z channel stats + cross-corr
        split="train",
        context=ctx,
        question="Will the next ~20 sessions be a clear UP move (not chop), or a clear DOWN move?",
        options={"A": "Yes — clear UP", "B": "No — clear DOWN"},
        answer="A" if ev.label == 1 else "B",
        series=[inter],
        domain="stock",
        source="updown_smoke",
        raw_problem="updown",
    )


def write_selection(key: str, multimodal: bool):
    if multimodal:
        # 取各通道通用统计 + 若干交叉相关
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
    path.write_text(json.dumps({"names": names, "method": "updown smoke", "mm": multimodal}, indent=2), encoding="utf-8")
    return key


def run_ft(tag: str, samples: list[Sample], sel_key: str, n_train_cap: int = 160, epochs: int = 3) -> dict:
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
        "name": f"smoke-updown-{tag}",
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
            "n_samples": min(n_train_cap, len(train)),
            "positive_prefix": "Yes",
            "seed": 7,
            "state": f"selv_noarr:{sel_key}",
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
        "eval": [{"suffix": "test-updown", "data": {"source": "stock", "split": "test"}}],
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
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
    acc = None
    block = metrics.get("train") or metrics
    if isinstance(block, dict):
        for v in block.values():
            if isinstance(v, dict) and "acc" in v:
                acc = v["acc"]
                break
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
        "acc": acc,
        "beats_75": bool(acc is not None and acc >= 0.75),
        "beats_chance": bool(acc is not None and acc > 0.55),
        "metrics": metrics,
        "checkpoint": str(ckpt),
    }


def main():
    assert DECIDER.is_dir(), f"missing {DECIDER}"
    OUT.mkdir(parents=True, exist_ok=True)
    codes = load_codes(80)
    print(f"codes={len(codes)}", flush=True)

    # 收集 L=100 / L=200 事件
    events_by_L: dict[int, list[Event]] = {100: [], 200: []}
    chop_stats = {100: 0, 200: 0}
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
            chop_stats[L] += st["chop_seen"]
            print(f"  L={L} up={st['up']} down={st['down']} chop_seen~{st['chop_seen']}")
            # 每种标签各落几张对齐图
            chart_dir = OUT / "charts" / f"L{L}"
            for ev in evs[:2]:
                render_chart(ev, chart_dir / f"{ev.code}_{ev.date}_{'up' if ev.label else 'dn'}.png")
        time.sleep(0.05)

    summary_data = {
        L: {
            "n": len(evs),
            "up": sum(1 for e in evs if e.label == 1),
            "down": sum(1 for e in evs if e.label == 0),
            "chop_windows_seen": chop_stats[L],
        }
        for L, evs in events_by_L.items()
    }
    print("events", json.dumps(summary_data, indent=2), flush=True)

    results = []
    # 组合: L × {close, multimodal}
    for L, evs in events_by_L.items():
        if len(evs) < 40:
            print(f"skip L={L} too few")
            continue
        # close-only
        sel = write_selection(f"updown-close-L{L}", multimodal=False)
        samples = [to_sample_close_only(e) for e in evs]
        print(f"\n=== FT close-only L={L} n={len(samples)} ===", flush=True)
        results.append(run_ft(f"close_L{L}", samples, sel, n_train_cap=160, epochs=3))

        # multimodal interleaved
        sel_m = write_selection(f"updown-mm-L{L}", multimodal=True)
        samples_m = [to_sample_multimodal(e) for e in evs]
        print(f"\n=== FT multimodal L={L} n={len(samples_m)} ===", flush=True)
        results.append(run_ft(f"mm_L{L}", samples_m, sel_m, n_train_cap=160, epochs=3))

    report = {
        "goal": "clear UP vs clear DOWN (chop excluded); compare close-only vs chart-aligned multimodal series",
        "label_rule": {
            "horizon": HORIZON,
            "up": {"max_gain>=": UP_GAIN, "close>=": UP_CLOSE, "max_dd>=": UP_DD},
            "down": {"max_gain<=": DOWN_GAIN_MAX, "close<=": DOWN_CLOSE, "or_max_dd<=": DOWN_DD},
            "chop": "excluded from binary train/test",
        },
        "alignment": "chart bars == series length (100 or 200); channels close/vol/macd(+rsi/ma in text)",
        "events": summary_data,
        "runs": results,
        "target_acc": 0.75,
        "any_beats_75": any(r.get("beats_75") for r in results if not r.get("skipped")),
        "charts": str(OUT / "charts"),
    }
    (OUT / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== REPORT ===")
    # compact table
    for r in results:
        if r.get("skipped"):
            print(r["tag"], "SKIP", r.get("reason"))
        else:
            print(
                f"{r['tag']}: acc={r['acc']} n_test={r['n_test']} "
                f"beats75={r['beats_75']} vram={r['peak_vram_gb']}GB"
            )
    print("saved", OUT / "smoke_report.json")
    print("any>=75%:", report["any_beats_75"])


if __name__ == "__main__":
    main()
