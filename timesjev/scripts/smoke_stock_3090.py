#!/usr/bin/env python3
"""3090 可行性烟测：冻结 Decider-2B + TimesJev value-token encoder，在股票「起涨后能否≥25%」二分类上训练/评测。

不依赖 gated TSQA。数据来自腾讯日线；特征走 TimesJev pool（含 catch22）。
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from timesjev.schema import Sample
from timesjev.train import sft as sft_mod

DECIDER = Path(r"F:/aicoding/myjev/models/decider-2b")
OUT = ROOT / "runs" / "smoke_stock_3090"
SEL_KEY = "stock-stg-16"
STOCK_AGENT = Path(r"D:/aicoding/stockagent-analysis")


def fetch_closes(code: str, limit: int = 800) -> list[tuple[str, float, float, float, float]] | None:
    """返回 [(date, open, high, low, close), ...]"""
    import json as _json
    import urllib.request

    c = code.zfill(6)
    symbol = f"sh{c}" if c.startswith(("5", "6")) else f"sz{c}"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,{limit},qfq"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = _json.loads(resp.read().decode())
    except Exception as e:
        print(f"  fetch fail {code}: {e}")
        return None
    bars = (data.get("data") or {}).get(symbol, {})
    rows = bars.get("qfqday") or bars.get("day") or []
    out = []
    for b in rows:
        if len(b) < 5:
            continue
        out.append((str(b[0]), float(b[1]), float(b[3]), float(b[4]), float(b[2])))
    return out if len(out) >= 260 else None


def make_sample(code: str, window: list[float], label_yes: bool, sid: str) -> Sample:
    arr = "[" + ", ".join(f"{v:.4f}" for v in window) + "]"
    context = (
        f"Stock {code} daily close prices ending at a candidate uptrend start (left window only).\n"
        f"Close series: {arr}"
    )
    options = {
        "A": "Yes, likely to gain at least 25% within about 40 trading days",
        "B": "No, unlikely to reach a 25% gain in that horizon",
    }
    return Sample(
        id=sid,
        task="stock_uptrend25",
        split="train",
        context=context,
        question="Given only this past close series, will a high-quality 25%+ rise follow?",
        options=options,
        answer="A" if label_yes else "B",
        series=[window],
        domain="stock",
        source="stock_tencent",
        raw_problem=context,
    )


def build_dataset(codes: list[str], left: int = 150, fut: int = 40, max_pos: int = 60, max_neg: int = 60):
    pos, neg = [], []
    for code in codes:
        print(f"fetch {code} ...", flush=True)
        rows = fetch_closes(code)
        if not rows:
            continue
        closes = [r[4] for r in rows]
        opens = [r[1] for r in rows]
        highs = [r[2] for r in rows]
        n = len(closes)
        # positives: future max gain >= 25%
        for t0 in range(left - 1, n - fut - 1, 5):
            entry = opens[t0 + 1]
            if entry <= 0:
                continue
            gain = max(highs[t0 + 1 : t0 + 1 + fut]) / entry - 1
            window = closes[t0 - left + 1 : t0 + 1]
            if gain >= 0.25 and len(pos) < max_pos:
                pos.append(make_sample(code, window, True, f"pos/{code}/{rows[t0][0]}"))
            elif gain < 0.08 and len(neg) < max_neg:
                neg.append(make_sample(code, window, False, f"neg/{code}/{rows[t0][0]}"))
            if len(pos) >= max_pos and len(neg) >= max_neg:
                break
        time.sleep(0.15)
        if len(pos) >= max_pos and len(neg) >= max_neg:
            break
    print(f"dataset pos={len(pos)} neg={len(neg)}")
    return pos, neg


def write_selection():
    # 16 个计算稳定、含部分 catch22 的特征（与 TimesJev pool 名一致）
    names = [
        "std", "range", "IQR", "mean abs step", "std abs step", "max abs step",
        "acf lag1", "acf lag2", "slope", "argmax position", "zero-crossing rate",
        "dominant period", "periodicity", "high_fluctuation", "stats3 median", "c24 SD",
    ]
    path = ROOT / "configs" / "selections" / f"{SEL_KEY}.json"
    path.write_text(json.dumps({
        "names": names,
        "method": "manual smoke selection for stock 3090",
        "dataset": "stock_uptrend25",
    }, indent=2), encoding="utf-8")
    return path


def main():
    assert DECIDER.is_dir() and (DECIDER / "model.safetensors").is_file(), f"missing {DECIDER}"
    write_selection()

    # 股票池：优先用 stockagent 池，取前若干只
    pool_file = STOCK_AGENT / "stock_pool_94.txt"
    codes = []
    if pool_file.is_file():
        for line in pool_file.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                codes.append(line.split("|")[0].zfill(6))
    codes = codes[:20] or ["000155", "000546", "000792", "002009", "600519"]

    pos, neg = build_dataset(codes, max_pos=48, max_neg=48)
    if len(pos) < 16 or len(neg) < 16:
        raise SystemExit(f"样本不足 pos={len(pos)} neg={len(neg)}")

    rng = random.Random(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    # 80/20 split
    def split(xs):
        k = max(4, int(0.2 * len(xs)))
        return xs[k:], xs[:k]

    tr_pos, te_pos = split(pos)
    tr_neg, te_neg = split(neg)
    train = tr_pos + tr_neg
    test = te_pos + te_neg
    rng.shuffle(train)
    rng.shuffle(test)

    data_dir = OUT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.json").write_text(
        json.dumps([s.__dict__ for s in train], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (data_dir / "test.json").write_text(
        json.dumps([s.__dict__ for s in test], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # monkeypatch load_samples for train/eval
    train_ids = {s.id for s in train}
    test_ids = {s.id for s in test}
    all_samples = {s.id: s for s in train + test}

    def load_samples(data_cfg: dict):
        split = data_cfg.get("split", "train")
        ids = train_ids if split == "train" else test_ids
        return [all_samples[i] for i in ids]

    import timesjev.data as data_mod
    import timesjev.train.sft as sft_file
    data_mod.load_samples = load_samples
    sft_file.load_samples = load_samples

    cfg = {
        "name": "smoke-stock-decider2b-3090",
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
            "n_samples": min(64, len(train)),
            "positive_prefix": "Yes",
            "seed": 42,
            "state": f"selv_noarr:{SEL_KEY}",
            "freeze_lm": True,
            "batch_like": "raw",
            "max_ctx_tokens": 4096,
            "monitor": "test-stock",
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
        "eval": [
            {
                "suffix": "test-stock",
                "data": {"source": "stock", "split": "test"},
            }
        ],
    }
    (OUT / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    print("\n=== GPU before train ===")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(torch.cuda.get_device_name(0), "free", torch.cuda.mem_get_info()[0] / 1e9, "GB")

    t0 = time.time()
    ckpt = sft_mod.train_sft(cfg)
    train_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    # evaluate with DeciderTSModel
    from timesjev.eval.runner import run_eval

    eval_cfg = {
        **cfg,
        "name": "smoke-stock-decider2b-3090-test",
        "data": {"source": "stock", "split": "test"},
        "model": {**cfg["model"], "checkpoint": str(ckpt), "backend": "decider_ts"},
    }
    # patch eval load too
    from timesjev.eval import runner as runner_mod
    runner_mod.load_samples = load_samples

    t1 = time.time()
    ev_dir = run_eval(eval_cfg)
    eval_s = time.time() - t1

    metrics = json.loads((Path(ev_dir) / "metrics.json").read_text(encoding="utf-8"))
    report = {
        "direction": "TimesJev-style: frozen Decider-2B + value-token encoder on stock 25% rise labels",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "checkpoint": str(DECIDER),
        "trained_run": str(ckpt),
        "n_train": len(train),
        "n_test": len(test),
        "n_train_used": cfg["train"]["n_samples"],
        "train_seconds": round(train_s, 1),
        "eval_seconds": round(eval_s, 1),
        "peak_vram_gb": round(peak, 2),
        "metrics": metrics,
        "feasible_on_3090": peak < 22.0,
    }
    out_path = OUT / "smoke_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SMOKE REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
