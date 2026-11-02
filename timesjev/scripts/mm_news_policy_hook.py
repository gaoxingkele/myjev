#!/usr/bin/env python3
"""新闻/政策模态接入钩子（里程碑后续）。

当前 live_week_predict_mm.py 已预留 policy 特征向量。
本脚本负责：若存在 Tushare/本地 CCTV 新闻缓存，则生成
  output/policy_heat/daily_policy_heat_<YYYYMMDD>.json
供扫描脚本自动读取。

无 token / 无缓存时安全退出，不阻断主流程。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

OUT = Path(r"D:/aicoding/stockagent-analysis/output/policy_heat")


def main(asof: str = "20261030") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("TUSHARE_TOKEN") or os.environ.get("TUSHARE_TOKEN".lower())
    # try dotenv paths
    for envp in (
        Path(r"D:/aicoding/stockagent-analysis/.env"),
        Path(r"D:/aicoding/stockagent-fourpool/.env"),
    ):
        if envp.is_file() and not token:
            for line in envp.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "TUSHARE" in line.upper() and "=" in line:
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    stub = {
        "trade_date": asof,
        "market_sentiment": "neutral",
        "macro_signal": "policy heat unavailable — stub",
        "themes": [],
        "source": None,
        "status": "unavailable",
    }
    if not token:
        path = OUT / f"daily_policy_heat_{asof}.json"
        path.write_text(json.dumps(stub, ensure_ascii=False, indent=2), encoding="utf-8")
        print("no TUSHARE_TOKEN; wrote stub", path)
        return
    try:
        import tushare as ts

        pro = ts.pro_api(token)
        df = pro.cctv_news(date=asof)
        if df is None or len(df) == 0:
            stub["status"] = "empty_cctv"
            stub["source"] = "tushare.cctv_news"
        else:
            # 不做 LLM 时：用标题关键词粗分情绪，保证模态可训练
            text = " ".join(df.astype(str).fillna("").values.ravel().tolist())[:8000]
            bull_kw = ["增长", "支持", "利好", "回升", "扩大", "稳增长", "降准", "降息"]
            bear_kw = ["风险", "下滑", "制裁", "收紧", "处罚", "暴跌", "战争"]
            b = sum(text.count(k) for k in bull_kw)
            e = sum(text.count(k) for k in bear_kw)
            sent = "bullish" if b > e + 2 else ("bearish" if e > b + 2 else "neutral")
            stub = {
                "trade_date": asof,
                "market_sentiment": sent,
                "macro_signal": f"cctv keyword score bull={b} bear={e}",
                "themes": [{"topic": "cctv_aggregate", "heat_score": min(100, 40 + 5 * abs(b - e)), "benefit_sectors": []}],
                "source": "tushare.cctv_news",
                "status": "ok_keyword",
                "n_rows": int(len(df)),
            }
    except Exception as ex:
        stub["status"] = f"error:{type(ex).__name__}"
        stub["macro_signal"] = str(ex)[:200]
    path = OUT / f"daily_policy_heat_{asof}.json"
    path.write_text(json.dumps(stub, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", path, stub.get("status"), stub.get("market_sentiment"))


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "20261030")
